"""
ECG Denoising Inference Script (Autoregressive + VQ only)

Input:
  - *_noisy.npy (normalized)
  - *_clean.npy (optional, normalized; for plotting residuals)
  - metadata.csv or metadata.json (optional; per-sample mu/sigma for raw-domain reconstruction)

Outputs:
  - *_denoised.npy (normalized)
  - *_denoised_raw.npy (if metadata exists)
  - *_plot.png (if clean exists and plotting enabled)
"""

import os
import json
import csv
import yaml
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  

from engine.registry import MODELS
import data, models, loss  


def strip_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def load_signal_npy(file_path: str):
    if not file_path.endswith(".npy"):
        return None
    try:
        sig = np.load(file_path)
        return np.asarray(sig).flatten().astype(np.float32)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_metadata_map(meta_dir: str):
    meta_map = {}
    csv_path = os.path.join(meta_dir, "metadata.csv")
    json_path = os.path.join(meta_dir, "metadata.json")

    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = str(row.get("filename", "")).strip().lower()
                    if not key:
                        continue
                    meta_map[key] = (float(row["mu"]), float(row["sigma"]))
            print(f"Loaded metadata.csv with {len(meta_map)} entries.")
            return meta_map
        except Exception as e:
            print(f"Failed to parse metadata.csv: {e}")

    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                rows = json.load(f)
            for row in rows:
                key = str(row.get("filename", "")).strip().lower()
                if not key:
                    continue
                meta_map[key] = (float(row["mu"]), float(row["sigma"]))
            print(f"Loaded metadata.json with {len(meta_map)} entries.")
            return meta_map
        except Exception as e:
            print(f"Failed to parse metadata.json: {e}")

    print(" No metadata.csv/json found. Raw reconstruction will be skipped.")
    return meta_map


def denorm(x_norm: np.ndarray, mu: float, sigma: float):
    return (x_norm * sigma + mu).astype(np.float32)


def ensure_pad_1d(x: np.ndarray, multiple: int):
    if multiple <= 1:
        return x, 0
    L = len(x)
    target = int(np.ceil(L / multiple) * multiple)
    pad_len = target - L
    if pad_len > 0:
        x = np.pad(x, (0, pad_len), mode="constant")
    return x, pad_len


def save_array_outputs(out_dir: str, base_name: str, arr: np.ndarray, suffix: str, save_csv: bool):
    npy_path = os.path.join(out_dir, f"{base_name}{suffix}.npy")
    np.save(npy_path, arr.astype(np.float32))
    if save_csv:
        csv_path = os.path.join(out_dir, f"{base_name}{suffix}.csv")
        np.savetxt(csv_path, arr.astype(np.float32), delimiter=",")
    return npy_path


def diff_stats(a: np.ndarray, b: np.ndarray):
    L = min(len(a), len(b))
    if L <= 0:
        return None, None
    d = a[:L] - b[:L]
    return float(np.mean(np.abs(d))), float(np.max(np.abs(d)))


def save_plot_panels(
    out_dir: str,
    base_name: str,
    clean_sig: np.ndarray,
    noisy_sig: np.ndarray,
    pred_sig: np.ndarray,
    domain_label: str,
    show_residual: bool = True,
    colors=None,
):
    if colors is None:
        colors = {
            "clean": "black",
            "noisy": "tab:red",
            "pred": "tab:blue",
            "res_cn": "tab:orange",
            "res_cp": "tab:green",
        }

    L = min(len(clean_sig), len(noisy_sig), len(pred_sig))
    clean_sig = clean_sig[:L]
    noisy_sig = noisy_sig[:L]
    pred_sig = pred_sig[:L]

    panels = [
        ("Clean (Reference)", clean_sig, colors["clean"]),
        ("Noisy (Input)", noisy_sig, colors["noisy"]),
        ("Pred (Model Output)", pred_sig, colors["pred"]),
    ]

    y_min = min(np.min(clean_sig), np.min(noisy_sig), np.min(pred_sig))
    y_max = max(np.max(clean_sig), np.max(noisy_sig), np.max(pred_sig))
    margin = 0.10 * (y_max - y_min) if y_max > y_min else 1.0
    y_lim = (y_min - margin, y_max + margin)

    res_cn = clean_sig - noisy_sig
    res_cp = clean_sig - pred_sig

    nrows = 3 + (1 if show_residual else 0)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 2.8 * nrows), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    fig.suptitle(f"ECG Denoising — {base_name}  [{domain_label} domain]", y=0.995)

    for i, (title, sig, c) in enumerate(panels):
        axes[i].plot(sig, color=c, alpha=0.95)
        axes[i].set_title(title)
        axes[i].set_ylabel("Amplitude")
        axes[i].set_ylim(*y_lim)
        axes[i].grid(True, alpha=0.25)

    if show_residual:
        axr = axes[3]
        axr.plot(res_cn, color=colors["res_cn"], alpha=0.9, label="Clean - Noisy")
        axr.plot(res_cp, color=colors["res_cp"], alpha=0.9, label="Clean - Pred")
        axr.set_title("Residuals")
        axr.set_ylabel("Amplitude")
        axr.grid(True, alpha=0.25)
        axr.legend(loc="upper right", frameon=False)

    axes[-1].set_xlabel("Samples")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{base_name}_plot.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_arvq_model(cfg, ckpt_path: str, device: str):
    model_type = cfg["model"]["type"]
    model_cls = MODELS.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unknown model type in registry: {model_type}")

    model = model_cls(**cfg["model"].get("args", {})).to(device)

    if not getattr(model, "is_autoregressive", False):
        raise ValueError("This script supports only autoregressive models (model.is_autoregressive=True).")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    if "model_state" in checkpoint:
        state = checkpoint["model_state"]
    else:
        state = checkpoint

    model.load_state_dict(strip_prefix(state), strict=True)
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    cfg,
    ckpt_path,
    input_path,
    out_dir="inference_results",
    save_plot=True,
    save_csv=True,
    pad_multiple=16,
    plot_domain="auto",  # auto|norm|raw
    show_residual=True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    model = _load_arvq_model(cfg, ckpt_path, device)
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isdir(input_path):
        noisy_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith("_noisy.npy")
        ]
        noisy_files.sort()
        meta_dir = input_path
    elif os.path.isfile(input_path) and input_path.endswith("_noisy.npy"):
        noisy_files = [input_path]
        meta_dir = os.path.dirname(input_path)
    else:
        raise ValueError("Input must be a *_noisy.npy file or a folder containing them.")

    if not noisy_files:
        raise ValueError("No files ending in '_noisy.npy' found.")

    print(f"Found {len(noisy_files)} noisy input(s). Processing...")

    meta_map = load_metadata_map(meta_dir)

    for noisy_path in noisy_files:
        base_name = os.path.basename(noisy_path).replace("_noisy.npy", "")
        key = base_name.strip().lower()
        clean_path = noisy_path.replace("_noisy.npy", "_clean.npy")

        noisy_norm = load_signal_npy(noisy_path)
        if noisy_norm is None:
            continue

        clean_norm = None
        if os.path.exists(clean_path):
            clean_norm = load_signal_npy(clean_path)
            if clean_norm is not None:
                mad, mx = diff_stats(noisy_norm, clean_norm)
                if mad is not None:
                    print(f"{base_name}: matched clean (mean|diff|={mad:.2e}, max|diff|={mx:.2e})")
        else:
            print(f"{base_name}: no clean reference found")

        noisy_in, pad_len = ensure_pad_1d(noisy_norm, pad_multiple)
        noisy_tensor = torch.tensor(noisy_in, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        pred_norm = model.generate(noisy_tensor)
        pred_norm = pred_norm.detach().cpu().squeeze().numpy().astype(np.float32)

        if pad_len > 0:
            pred_norm = pred_norm[:-pad_len]
            noisy_norm = noisy_norm[:len(pred_norm)]
            if clean_norm is not None:
                clean_norm = clean_norm[:len(pred_norm)]

        save_array_outputs(out_dir, base_name, pred_norm, suffix="_denoised", save_csv=save_csv)

        has_meta = key in meta_map
        noisy_raw = pred_raw = clean_raw = None
        if has_meta:
            mu, sigma = meta_map[key]
            noisy_raw = denorm(noisy_norm, mu, sigma)
            pred_raw = denorm(pred_norm, mu, sigma)
            if clean_norm is not None:
                clean_raw = denorm(clean_norm, mu, sigma)
            save_array_outputs(out_dir, base_name, pred_raw, suffix="_denoised_raw", save_csv=save_csv)

        if save_plot:
            if clean_norm is None:
                print(f"{base_name}: clean missing; skipping plot.")
                continue

            if plot_domain == "raw":
                if not has_meta:
                    domain = "norm"
                    p_clean, p_noisy, p_pred = clean_norm, noisy_norm, pred_norm
                else:
                    domain = "raw"
                    p_clean, p_noisy, p_pred = clean_raw, noisy_raw, pred_raw
            elif plot_domain == "norm":
                domain = "norm"
                p_clean, p_noisy, p_pred = clean_norm, noisy_norm, pred_norm
            else:
                if has_meta:
                    domain = "raw"
                    p_clean, p_noisy, p_pred = clean_raw, noisy_raw, pred_raw
                else:
                    domain = "norm"
                    p_clean, p_noisy, p_pred = clean_norm, noisy_norm, pred_norm

            save_plot_panels(
                out_dir=out_dir,
                base_name=base_name,
                clean_sig=p_clean,
                noisy_sig=p_noisy,
                pred_sig=p_pred,
                domain_label=domain,
                show_residual=show_residual,
            )

    print(f"\n Done. Results saved to: {os.path.abspath(out_dir)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ECG AR+VQ Denoising Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt/.pth).")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Folder containing *_noisy.npy (and optionally *_clean.npy) OR a single *_noisy.npy file.",
    )
    parser.add_argument("--out_dir", type=str, default="inference_results", help="Output directory.")
    parser.add_argument("--no_plot", action="store_true", help="Disable plot saving.")
    parser.add_argument("--no_csv", action="store_true", help="Disable CSV saving.")
    parser.add_argument("--pad_multiple", type=int, default=16, help="Pad input length to a multiple of this value.")
    parser.add_argument("--plot_domain", type=str, default="auto", choices=["auto", "norm", "raw"])
    parser.add_argument("--no_residual", action="store_true", help="Disable residual panel.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_inference(
        cfg=cfg,
        ckpt_path=args.checkpoint,
        input_path=args.input,
        out_dir=args.out_dir,
        save_plot=(not args.no_plot),
        save_csv=(not args.no_csv),
        pad_multiple=args.pad_multiple,
        plot_domain=args.plot_domain,
        show_residual=(not args.no_residual),
    )


if __name__ == "__main__":
    main()
