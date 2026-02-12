"""
ECG Autoregressive-VQ Model Testing Script (Physical-scale Metrics + Mean/SD)

Assumptions:
- Dataset prep normalized both noisy/clean using NOISY-window (mu, sigma).
- Test computes metrics in physical scale if (mu, sigma) are available.
- Supports ONLY autoregressive models via model.generate(x_noisy).
- Optionally uses metadata.json to inject mu/sigma/noise fields if dataset doesn't provide them.

Expected (if using NPY fallback or metadata injection):
- metadata.json containing entries with: filename, mu, sigma, noise_type/noise_combo, snr_db
- files: {filename}_noisy.npy, {filename}_clean.npy (and optional {filename}_label.npy)
"""

import os
import json
import yaml
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from engine.registry import MODELS, DATASETS
from utils.logger import JSONLogger
from utils.flops_utils import estimate_flops
from utils.metrics import denormalize, METRIC_FUNCS, snr_in_db, snr_out_db, delta_snr_db

import data  # noqa: F401
import models  # noqa: F401


def strip_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def _safe_collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        v0 = batch[0][k]
        if isinstance(v0, str):
            out[k] = [b[k] for b in batch]
        elif isinstance(v0, (float, int)):
            out[k] = torch.tensor([b[k] for b in batch])
        elif torch.is_tensor(v0):
            try:
                out[k] = torch.stack([b[k] for b in batch], dim=0)
            except Exception:
                out[k] = [b[k] for b in batch]
        else:
            out[k] = [b[k] for b in batch]
    return out


def load_metadata_map(root: str):
    meta_path = os.path.join(root, "metadata.json")
    meta_map = {}
    if not root or (not os.path.exists(meta_path)):
        return meta_map

    with open(meta_path, "r") as f:
        meta_data = json.load(f)

    for e in meta_data:
        fname = e.get("filename", None)
        if not fname:
            continue
        meta_map[fname] = {
            "mu": float(e.get("mu", 0.0)),
            "sigma": float(e.get("sigma", 1.0)),
            "noise_type": e.get("noise_type", "unknown"),
            "noise_combo": e.get("noise_combo", ""),
            "snr_db": e.get("snr_db", None),
        }
    return meta_map


class MetaAugmentedDataset(Dataset):
    def __init__(self, base_ds: Dataset, root: str, strict_eval: bool = True, phase: str = "test"):
        self.base = base_ds
        self.root = root
        self.phase = phase
        self.strict_eval = strict_eval
        self.meta_map = load_metadata_map(root)
        self.paths = getattr(base_ds, "paths", None)
        if hasattr(base_ds, "signal_len"):
            self.signal_len = getattr(base_ds, "signal_len")

    def __len__(self):
        return len(self.base)

    @staticmethod
    def _stem_from_path(p: str):
        stem = os.path.basename(p)
        stem = stem.replace("_noisy.npy", "").replace("_clean.npy", "").replace(".npy", "")
        return stem

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if not isinstance(item, dict):
            raise TypeError("MetaAugmentedDataset expects base dataset items as dicts.")

        fname = item.get("filename", None)
        if not fname:
            if self.paths is not None and idx < len(self.paths):
                fname = self._stem_from_path(str(self.paths[idx]))
            else:
                for k in ["path", "noisy_path", "filepath", "file_path"]:
                    if k in item and item[k]:
                        fname = self._stem_from_path(str(item[k]))
                        break

        meta = self.meta_map.get(fname, None) if fname else None
        if "filename" not in item:
            item["filename"] = fname if fname else f"idx_{idx}"

        if ("mu" not in item) or ("sigma" not in item):
            if meta is None:
                if self.strict_eval and self.phase in ["val", "test"]:
                    raise RuntimeError(
                        f"Missing metadata for sample '{item['filename']}'. "
                        "metadata.json is required for physical-scale denormalization."
                    )
                item["mu"] = torch.tensor(0.0, dtype=torch.float32)
                item["sigma"] = torch.tensor(1.0, dtype=torch.float32)
            else:
                item["mu"] = torch.tensor(meta["mu"], dtype=torch.float32)
                item["sigma"] = torch.tensor(meta["sigma"], dtype=torch.float32)

        if "noise_type" not in item:
            item["noise_type"] = meta.get("noise_type", "unknown") if meta else "unknown"
        if "noise_combo" not in item:
            item["noise_combo"] = meta.get("noise_combo", "") if meta else ""
        if "snr_db" not in item:
            snr = meta.get("snr_db", None) if meta else None
            item["snr_db"] = -1.0 if snr is None else float(snr)

        return item


class ECGNPYTestDataset(Dataset):
    def __init__(self, root: str, has_label: bool = True):
        self.root = root
        self.has_label = has_label

        meta_path = os.path.join(root, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"metadata.json not found under root: {root}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.samples = []
        for e in meta:
            fname = e.get("filename", None)
            if not fname:
                continue
            noisy_path = os.path.join(root, f"{fname}_noisy.npy")
            clean_path = os.path.join(root, f"{fname}_clean.npy")
            label_path = os.path.join(root, f"{fname}_label.npy")
            if not (os.path.exists(noisy_path) and os.path.exists(clean_path)):
                continue

            self.samples.append({
                "filename": fname,
                "noisy_path": noisy_path,
                "clean_path": clean_path,
                "label_path": label_path,
                "mu": float(e.get("mu", 0.0)),
                "sigma": float(e.get("sigma", 1.0)),
                "noise_type": e.get("noise_type", "unknown"),
                "noise_combo": e.get("noise_combo", ""),
                "snr_db": e.get("snr_db", None),
            })

        if not self.samples:
            raise RuntimeError(f"No valid samples found in {root}.")

        x0 = np.load(self.samples[0]["noisy_path"])
        self.signal_len = int(x0.shape[-1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        noisy = np.load(s["noisy_path"]).astype(np.float32)
        clean = np.load(s["clean_path"]).astype(np.float32)

        if noisy.ndim == 1:
            noisy = noisy[None, :]
        if clean.ndim == 1:
            clean = clean[None, :]

        out = {
            "noisy": torch.from_numpy(noisy),
            "clean": torch.from_numpy(clean),
            "mu": torch.tensor(s["mu"], dtype=torch.float32),
            "sigma": torch.tensor(s["sigma"], dtype=torch.float32),
            "filename": s["filename"],
            "noise_type": s["noise_type"],
            "noise_combo": s["noise_combo"],
            "snr_db": -1.0 if s["snr_db"] is None else float(s["snr_db"]),
        }

        if self.has_label and os.path.exists(s["label_path"]):
            lab = np.load(s["label_path"]).astype(np.int64)
            out["label"] = torch.from_numpy(lab)

        return out


def _sanitize_sigma(sigma: torch.Tensor, eps: float = 1e-6):
    return torch.where(sigma.abs() < eps, torch.ones_like(sigma), sigma)


def _to_BT(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(1) if x.ndim == 3 else x


def evaluate_metrics_vectors(pred, clean, noisy, mu=None, sigma=None, selected=None):
    selected = selected or ["SSD", "MAD", "PRD", "RMSE", "CC", "CosSim", "SNRin", "SNRout", "DeltaSNR"]

    pred_n = _to_BT(pred).double()
    clean_n = _to_BT(clean).double()
    noisy_n = _to_BT(noisy).double() if noisy is not None else None

    pred_p, clean_p, noisy_p = pred_n, clean_n, noisy_n
    if (mu is not None) and (sigma is not None):
        mu = mu.view(-1, 1).to(pred_n).double()
        sigma = _sanitize_sigma(sigma.view(-1, 1).to(pred_n).double())
        pred_p = denormalize(pred_n, mu, sigma)
        clean_p = denormalize(clean_n, mu, sigma)
        if noisy_n is not None:
            noisy_p = denormalize(noisy_n, mu, sigma)

    out = {}
    eps = 1e-12

    if "SSD" in selected:
        diff = pred_n - clean_n
        out["SSD"] = (diff * diff).sum(dim=1).float()

    if "MAD" in selected:
        out["MAD"] = (pred_p - clean_p).abs().amax(dim=1).float()

    if "PRD" in selected:
        num = (pred_p - clean_p).pow(2).sum(dim=1).sqrt()
        den = clean_p.pow(2).sum(dim=1).sqrt().clamp_min(eps)
        out["PRD"] = (100.0 * num / den).float()

    if "SNRout" in selected:
        out["SNRout"] = snr_out_db(pred_p, clean_p).float()

    if ("SNRin" in selected) or ("DeltaSNR" in selected):
        if noisy_p is None:
            raise ValueError("SNRin/DeltaSNR requested but noisy is None.")
        if "SNRin" in selected:
            out["SNRin"] = snr_in_db(noisy_p, clean_p).float()
        if "DeltaSNR" in selected:
            out["DeltaSNR"] = delta_snr_db(noisy_p, clean_p, pred_p).float()

    for name in selected:
        if name in ["SSD", "MAD", "PRD", "SNRin", "SNRout", "DeltaSNR"]:
            continue
        if name in METRIC_FUNCS:
            out[name] = METRIC_FUNCS[name](pred_p, clean_p).float()

    return out


def mean_std_from_vectors(metric_vectors: dict):
    out = {}
    for k, v in metric_vectors.items():
        v = v.detach().float()
        out[k] = {
            "mean": float(v.mean().item()) if v.numel() else 0.0,
            "std": float(v.std(unbiased=False).item()) if v.numel() else 0.0,
        }
    return out


def build_test_dataset(cfg):
    test_cfg = cfg["dataset"]["test"].copy()
    test_cfg["augment"] = {"enable": False}
    root = test_cfg.get("root", "")
    ds_type = cfg["dataset"]["type"]

    try:
        dataset_cls = DATASETS.get(ds_type)
        base_ds = dataset_cls(phase="test", **test_cfg)

        needs_wrap = True
        try:
            s0 = base_ds[0]
            if isinstance(s0, dict) and ("mu" in s0) and ("sigma" in s0):
                needs_wrap = False
        except Exception:
            needs_wrap = True

        if needs_wrap:
            return MetaAugmentedDataset(base_ds, root=root, strict_eval=True, phase="test"), "registry+meta"
        return base_ds, "registry"
    except Exception as e:
        print(f"⚠️ Registry dataset load failed ({e}). Falling back to ECGNPYTestDataset.")
        return ECGNPYTestDataset(root=root, has_label=True), "npy"


def _load_autoregressive_model(cfg, ckpt_path: str, device: str):
    model_cls = MODELS.get(cfg["model"]["type"])
    if model_cls is None:
        raise ValueError(f"Unknown model type: {cfg['model']['type']}")

    model = model_cls(**cfg["model"]["args"]).to(device)
    if not getattr(model, "is_autoregressive", False):
        raise ValueError("This test script supports only autoregressive models (model.is_autoregressive=True).")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(strip_prefix(state), strict=True)
    model.eval()
    return model


@torch.no_grad()
def run_test(cfg, ckpt_path=None, out_json=None, per_level=False, noisy_metrics=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    ckpt_cfg = cfg.get("checkpoint", {})
    base_out_dir = ckpt_cfg.get("out_dir", "outputs")
    os.makedirs(base_out_dir, exist_ok=True)
    logger = JSONLogger(out_dir=base_out_dir)

    test_set, ds_mode = build_test_dataset(cfg)
    batch_size = cfg["train"]["batch_size"]

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
        collate_fn=_safe_collate,
        pin_memory=(device == "cuda"),
    )

    if hasattr(test_set, "signal_len"):
        sig_len = int(test_set.signal_len)
    else:
        b0 = next(iter(test_loader))
        sig_len = int(b0["noisy"].shape[-1])

    if ckpt_path is None:
        ckpt_dir = base_out_dir
        best_ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("best_")])
        ckpt_path = os.path.join(ckpt_dir, best_ckpts[-1] if best_ckpts else "last.pt")

    model = _load_autoregressive_model(cfg, ckpt_path, device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"📐 Model Parameters: {num_params/1e6:.2f} M")

    flops = None
    try:
        flops, params = estimate_flops(model, input_shape=(1, 1, sig_len))
        print(f"⚡ Estimated FLOPs: {flops/1e6:.2f} M | Params: {params/1e6:.2f} M")
    except Exception as e:
        print(f"⚠️ FLOPs estimation skipped (error: {e})")

    selected_metrics = cfg.get("eval", {}).get("metrics", [])
    for m in ["SNRin", "SNRout", "DeltaSNR"]:
        if m not in selected_metrics:
            selected_metrics.append(m)

    allowed_metrics = set(METRIC_FUNCS.keys()) | {"SNRin", "SNRout", "DeltaSNR"}
    selected_metrics = [m for m in selected_metrics if m in allowed_metrics]

    total_inference_time = 0.0
    total_samples = 0

    global_metric_vecs = {m: [] for m in selected_metrics}
    noisy_baseline_vecs = {m: [] for m in selected_metrics}
    per_level_vecs = defaultdict(lambda: defaultdict(list))

    test_iter = tqdm(test_loader, desc=f"🧪 Testing {cfg['model']['type']}", ncols=100)

    did_diag = False

    for batch in test_iter:
        x_noisy = batch["noisy"].to(device)
        x_clean = batch["clean"].to(device)

        mu = batch.get("mu", None)
        sigma = batch.get("sigma", None)
        if (mu is not None) and (sigma is not None):
            mu = mu.to(device).view(-1)
            sigma = sigma.to(device).view(-1)
        else:
            mu, sigma = None, None

        if not did_diag:
            did_diag = True
            if (mu is None) or (sigma is None):
                print("⚠️ mu/sigma not provided by dataset. Metrics will be computed in normalized scale.")
            else:
                nn_ = x_noisy.detach().squeeze(1)
                m_med = nn_.mean(dim=-1).abs().median().item()
                s_med = nn_.std(dim=-1, unbiased=False).median().item()
                print(f"[Diag] noisy_norm median(|mean|)={m_med:.4f}, median(std)={s_med:.4f}")

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        x_hat = model.generate(x_noisy)

        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

        total_inference_time += (t1 - t0)
        total_samples += x_noisy.size(0)

        vecs = evaluate_metrics_vectors(
            pred=x_hat,
            clean=x_clean,
            noisy=x_noisy,
            mu=mu,
            sigma=sigma,
            selected=selected_metrics,
        )
        for k, v in vecs.items():
            global_metric_vecs[k].append(v.detach().cpu())

        if noisy_metrics:
            vecs_noisy = evaluate_metrics_vectors(
                pred=x_noisy,
                clean=x_clean,
                noisy=x_noisy,
                mu=mu,
                sigma=sigma,
                selected=selected_metrics,
            )
            for k, v in vecs_noisy.items():
                noisy_baseline_vecs[k].append(v.detach().cpu())

        if per_level:
            noise_type = batch.get("noise_type", None)
            noise_combo = batch.get("noise_combo", None)
            snr_db = batch.get("snr_db", None)

            if noise_type is None:
                noise_type = ["unknown"] * x_noisy.size(0)
            if noise_combo is None:
                noise_combo = [""] * x_noisy.size(0)
            if snr_db is None:
                snr_db = torch.full((x_noisy.size(0),), -1.0)

            if torch.is_tensor(snr_db):
                snr_db_list = snr_db.detach().cpu().tolist()
            else:
                snr_db_list = [float(s) for s in snr_db]

            for bi in range(x_noisy.size(0)):
                combo = ""
                if isinstance(noise_combo, list):
                    combo = noise_combo[bi]
                if not combo:
                    combo = noise_type[bi] if isinstance(noise_type, list) else "unknown"

                key = (combo, float(snr_db_list[bi]))
                for mk, mv in vecs.items():
                    per_level_vecs[key][mk].append(mv[bi:bi + 1].detach().cpu())

    global_vec_cat = {}
    for k, parts in global_metric_vecs.items():
        global_vec_cat[k] = torch.cat(parts, dim=0) if parts else torch.empty(0)
    metrics_mean_std = mean_std_from_vectors(global_vec_cat)

    noisy_mean_std = {}
    if noisy_metrics:
        noisy_vec_cat = {}
        for k, parts in noisy_baseline_vecs.items():
            noisy_vec_cat[k] = torch.cat(parts, dim=0) if parts else torch.empty(0)
        noisy_mean_std = mean_std_from_vectors(noisy_vec_cat)

    per_level_summary = []
    if per_level:
        for key, mdict in per_level_vecs.items():
            combo, snr = key
            row = {"noise_combo": combo, "snr_db": snr, "count": 0, "metrics": {}}
            for mk, lst in mdict.items():
                v = torch.cat(lst, dim=0) if lst else torch.empty(0)
                row["count"] = int(v.numel()) if row["count"] == 0 else row["count"]
                row["metrics"][mk] = {
                    "mean": float(v.mean().item()) if v.numel() else 0.0,
                    "std": float(v.std(unbiased=False).item()) if v.numel() else 0.0,
                }
            per_level_summary.append(row)
        per_level_summary.sort(key=lambda x: (x["noise_combo"], x["snr_db"]))

    avg_sample_time = total_inference_time / max(total_samples, 1)

    results = {
        "phase": "test",
        "dataset_mode": ds_mode,
        "checkpoint": os.path.basename(ckpt_path),
        "model_params": int(num_params),
        "model_params_million": round(num_params / 1e6, 3),
        "flops": flops,
        "flops_million": round(flops / 1e6, 3) if flops else None,
        "signal_len": int(sig_len),
        "metrics": metrics_mean_std,
        "avg_inference_time_per_sample_sec": round(avg_sample_time, 6),
        "avg_inference_time_per_sample_ms": round(avg_sample_time * 1000, 3),
    }
    if noisy_mean_std:
        results["noisy_metrics"] = noisy_mean_std
    if per_level_summary:
        results["per_level_summary"] = per_level_summary

    print("\n" + "=" * 70)
    print(f"TEST SUMMARY — {os.path.basename(ckpt_path)}")
    print("=" * 70)
    for k in sorted(metrics_mean_std.keys()):
        m = metrics_mean_std[k]["mean"]
        s = metrics_mean_std[k]["std"]
        print(f"{k:<10}: {m:.6f} ± {s:.6f}")

    if noisy_mean_std:
        print("\n--- Baseline (Noisy vs Clean) mean±std ---")
        for k in sorted(noisy_mean_std.keys()):
            m = noisy_mean_std[k]["mean"]
            s = noisy_mean_std[k]["std"]
            print(f"{k:<10}: {m:.6f} ± {s:.6f}")

    if per_level_summary:
        print("\n" + "=" * 70)
        print("Per-Noise/SNR Metrics (mean±std)")
        print("=" * 70)

        all_metric_names = sorted({m for r in per_level_summary for m in r["metrics"].keys()})
        header = (
            f"{'Noise Combo':<22} | {'SNR':>5} | {'Count':>6} | "
            + " | ".join([f"{m:<18}" for m in all_metric_names])
        )
        print(header)
        print("-" * len(header))

        for r in per_level_summary:
            cells = []
            for m in all_metric_names:
                mm = r["metrics"][m]["mean"]
                ss = r["metrics"][m]["std"]
                cells.append(f"{mm:.4f}±{ss:.4f}".ljust(18))
            line = (
                f"{r['noise_combo']:<22} | {r['snr_db']:>5.0f} | {r['count']:>6} | "
                + " | ".join(cells)
            )
            print(line)

    print("\n--- Inference Time ---")
    print(f"Average per sample: {avg_sample_time*1000:.3f} ms")

    if out_json:
        if os.path.isdir(out_json):
            out_json = os.path.join(out_json, "test_results.json")
        out_dir = os.path.dirname(out_json) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved -> {out_json}")

    logger.close()
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate ECG autoregressive VQ model on test set (physical-scale).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (optional).")
    parser.add_argument("--out_json", type=str, default=None, help="Path to save JSON test results (optional).")
    parser.add_argument("--per_level", action="store_true", help="Compute per-noise/SNR metrics.")
    parser.add_argument("--noisy_metrics", action="store_true", help="Also compute baseline (noisy vs clean).")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_test(
        cfg,
        ckpt_path=args.checkpoint,
        out_json=args.out_json,
        per_level=args.per_level,
        noisy_metrics=args.noisy_metrics,
    )


if __name__ == "__main__":
    main()