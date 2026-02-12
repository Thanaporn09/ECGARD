import os, json, numpy as np, torch
from torch.utils.data import Dataset
from engine.registry import DATASETS
from data.augmentation import apply_augmentation

@DATASETS.register
class ECGDenoiseDataset(Dataset):
    def __init__(self, root=None, augment=None, phase="train", fs=250,
                 has_label=True, strict_eval=True):
        self.phase = phase
        self.fs = fs
        self.has_label = has_label
        self.strict_eval = strict_eval
        self.augment_cfg = augment or {"enable": False, "methods": []}

        if (root is None) or (not os.path.isdir(root)):
            raise RuntimeError(f"Invalid root: {root}")

        phase_root = os.path.join(root, phase)
        self.root = phase_root if os.path.isdir(phase_root) else root

        meta_path = os.path.join(self.root, "metadata.json")
        meta_map = {}

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)

            for e in meta:
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
        else:
            # metadata missing
            if (self.phase in ["val", "test"]) and self.strict_eval:
                raise RuntimeError(
                    f"metadata.json not found at {meta_path}. "
                    f"It is required for physical-scale evaluation in phase='{self.phase}'."
                )
            meta_map = {}

        # ------------------------------------------------------------
        # Build sample list from self.root
        # ------------------------------------------------------------
        noisy_paths = sorted([
            os.path.join(self.root, f) for f in os.listdir(self.root)
            if f.endswith("_noisy.npy")
        ])
        if len(noisy_paths) == 0:
            raise RuntimeError(f"No *_noisy.npy files found in {self.root}")

        self.samples = []
        for npth in noisy_paths:
            cpth = npth.replace("_noisy.npy", "_clean.npy")
            if not os.path.exists(cpth):
                raise RuntimeError(f"Missing clean pair for {npth}")

            stem = os.path.basename(npth).replace("_noisy.npy", "")
            lpth = npth.replace("_noisy.npy", "_label.npy")

            # Defaults (acceptable for training if metadata not available)
            m = {
                "mu": 0.0, "sigma": 1.0,
                "noise_type": "unknown", "noise_combo": "", "snr_db": None
            }

            if stem in meta_map:
                m = meta_map[stem]
            else:
                # Strict only for val/test (or only test if you prefer)
                if (self.phase in ["val", "test"]) and self.strict_eval:
                    raise RuntimeError(f"Missing metadata for sample: {stem}")

            self.samples.append({
                "noisy_path": npth,
                "clean_path": cpth,
                "label_path": lpth,
                "filename": stem,
                "mu": m["mu"],
                "sigma": m["sigma"],
                "noise_type": m.get("noise_type", "unknown"),
                "noise_combo": m.get("noise_combo", ""),
                "snr_db": m.get("snr_db", None),
            })

    def __len__(self):
        return len(self.samples)

    def _load_1d_float32(self, path: str) -> np.ndarray:
        x = np.load(path).astype(np.float32)
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 1:
            raise RuntimeError(f"Expected 1D (or [1,T]) array, got {x.shape} from {path}")
        return x

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        noisy = self._load_1d_float32(s["noisy_path"])
        clean = self._load_1d_float32(s["clean_path"])

        # Training augmentation unchanged: noisy only
        if (self.phase == "train") and self.augment_cfg.get("enable", False):
            noisy = apply_augmentation(noisy, self.fs, self.augment_cfg)

        # Base outputs (training remains the same keys)
        out = {
            "noisy": torch.from_numpy(noisy).unsqueeze(0),  # [1, T]
            "clean": torch.from_numpy(clean).unsqueeze(0),  # [1, T]
            "mu": torch.tensor(float(s["mu"]), dtype=torch.float32),
            "sigma": torch.tensor(float(s["sigma"]), dtype=torch.float32),
            "filename": s["filename"],
        }

        # Test-only extras (for your per-level evaluation + physical-scale metrics)
        if self.phase == "test":
            out["noise_type"] = s.get("noise_type", "unknown")
            out["noise_combo"] = s.get("noise_combo", "")
            snr = s.get("snr_db", None)
            out["snr_db"] = -1.0 if snr is None else float(snr)

            if self.has_label and os.path.exists(s["label_path"]):
                lab = np.load(s["label_path"]).astype(np.int64)
                out["label"] = torch.from_numpy(lab)

        return out
