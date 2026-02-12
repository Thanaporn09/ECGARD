# runner_ar_vq.py
from __future__ import annotations

import os
import json
import datetime
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from engine.registry import MODELS, DATASETS, LOSSES
from utils.metrics import evaluate_metrics
from utils.logger import JSONLogger


class Runner:
    """
    Clean runner for **autoregressive + VQ** models only.

    Assumptions about the model:
      - model.is_autoregressive == True
      - training forward: model(x_noisy, x_clean) -> x_hat (teacher forcing allowed)
      - inference: model.generate(x_noisy, ...) -> x_hat
      - optional: model.last_vq_loss (Tensor scalar)
      - optional delineation head:
          model.enable_delineation_head == True and
          model.generate(x_noisy, return_delineation=True) -> (x_hat, del_logits)
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ===================== #
        #        EPOCHS         #
        # ===================== #
        self.epochs = int(cfg["train"].get("epochs", 100))
        self.epoch = 0

        # ===================== #
        #   OUTPUT STRUCTURE    #
        # ===================== #
        ckpt_cfg = cfg.get("checkpoint", {})
        self.out_dir = ckpt_cfg.get("out_dir", "outputs")
        os.makedirs(self.out_dir, exist_ok=True)

        self.ckpt_dir = self.out_dir
        self.logger = JSONLogger(out_dir=self.out_dir)

        # ===================== #
        #   SAVE TRAIN CONFIG   #
        # ===================== #
        config_path = os.path.join(self.out_dir, "config.json")
        try:
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=4)
            print(f"Training config saved at {config_path}")
        except Exception as e:
            print(f"Failed to save config: {e}")

        # ===================== #
        #       DATASETS        #
        # ===================== #
        dataset_cls = DATASETS.get(cfg["dataset"]["type"])
        if dataset_cls is None:
            raise ValueError(f"Unknown dataset type: {cfg['dataset']['type']}")

        bs = int(cfg["train"]["batch_size"])
        num_workers = int(cfg["train"].get("num_workers", 4))
        pin_memory = bool(cfg["train"].get("pin_memory", True))

        train_cfg = cfg["dataset"]["train"]
        self.train_set = dataset_cls(phase="train", **train_cfg)
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        val_cfg = cfg["dataset"]["val"]
        # ensure val has augment disabled
        if "augment" in val_cfg:
            val_cfg["augment"]["enable"] = False
        else:
            val_cfg["augment"] = {"enable": False}
        self.val_set = dataset_cls(phase="val", **val_cfg)
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        test_cfg = cfg["dataset"].get("test", None)
        if test_cfg:
            if "augment" in test_cfg:
                test_cfg["augment"]["enable"] = False
            else:
                test_cfg["augment"] = {"enable": False}
            self.test_set = dataset_cls(phase="test", **test_cfg)
            self.test_loader = DataLoader(
                self.test_set,
                batch_size=bs,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )
        else:
            self.test_set, self.test_loader = None, None

        # ===================== #
        #        MODEL          #
        # ===================== #
        model_cls = MODELS.get(cfg["model"]["type"])
        if model_cls is None:
            raise ValueError(f"Unknown model type: {cfg['model']['type']}")

        self.model = model_cls(**cfg["model"].get("args", {})).to(self.device)

        if not getattr(self.model, "is_autoregressive", False):
            raise ValueError(
                "This runner supports only autoregressive models. "
                "Set model.is_autoregressive=True and implement forward/generate."
            )

        # ===================== #
        #         LOSS          #
        # ===================== #
        loss_cfg = cfg.get("loss", {"type": "L1ReconLoss", "args": {}})
        loss_type = loss_cfg.get("type", "L1ReconLoss")
        loss_args = loss_cfg.get("args", {})

        loss_cls = LOSSES.get(loss_type)
        if loss_cls is None:
            raise ValueError(f"Unknown loss type: {loss_type}")

        self.criterion: nn.Module = loss_cls(**loss_args).to(self.device)

        # ---------------- loss weights ----------------
        train_loss_cfg = cfg.get("train", {})
        self.lambda_recon = float(train_loss_cfg.get("lambda_recon", 1.0))
        self.lambda_vq = float(train_loss_cfg.get("lambda_vq", 1.0))
        self.lambda_delineation = float(train_loss_cfg.get("lambda_delineation", 0.0))

        # delineation loss (optional)
        self.has_delineation = bool(getattr(self.model, "enable_delineation_head", False))
        self.delineation_loss_fn = None
        if self.has_delineation and self.lambda_delineation > 0:
            del_cfg = cfg.get("delineation_loss", {"type": None, "args": {}})
            if del_cfg.get("type") is not None:
                del_loss_cls = LOSSES.get(del_cfg["type"])
                if del_loss_cls is None:
                    raise ValueError(f"Unknown delineation loss type: {del_cfg['type']}")
                self.delineation_loss_fn = del_loss_cls(**del_cfg.get("args", {})).to(self.device)
            else:
                # You can keep lambda_delineation=0 if you don't want this.
                raise ValueError(
                    "Model has delineation head but delineation_loss.type is None while lambda_delineation>0."
                )

        # ===================== #
        #       OPTIMIZER       #
        # ===================== #
        opt_cfg = dict(cfg.get("optimizer", {}))
        opt_type = opt_cfg.pop("type", "AdamW")

        # convert string numbers safely (your original behavior)
        for k, v in list(opt_cfg.items()):
            if isinstance(v, str):
                try:
                    opt_cfg[k] = float(v)
                except ValueError:
                    pass
        if "betas" in opt_cfg and isinstance(opt_cfg["betas"], list):
            opt_cfg["betas"] = tuple(opt_cfg["betas"])

        if opt_type.lower() == "adan":
            try:
                from adan_pytorch import Adan  # type: ignore
            except ImportError as e:
                raise ImportError("adan-pytorch is not installed. Install via: pip install adan-pytorch") from e
            self.opt = Adan(self.model.parameters(), **opt_cfg)
        else:
            opt_class = getattr(optim, opt_type, None)
            if opt_class is None:
                raise ValueError(f"Optimizer {opt_type} not found in torch.optim.")
            self.opt = opt_class(self.model.parameters(), **opt_cfg)

        # ===================== #
        #      SCHEDULER        #
        # ===================== #
        self.scheduler = None
        sched_cfg = cfg.get("scheduler", None)
        if sched_cfg:
            sched_type = sched_cfg.get("type")
            if sched_type == "SequentialLR":
                warm_cfg = sched_cfg.get("warmup", {})
                cos_cfg = sched_cfg.get("cosine", {})

                start_factor = warm_cfg.get("start_factor", 0.1)
                total_iters = warm_cfg.get("total_iters", 5)

                T_max = cos_cfg.get("T_max", max(self.epochs - total_iters, 1))
                eta_min = cos_cfg.get("eta_min", 1e-6)

                warm = optim.lr_scheduler.LinearLR(self.opt, start_factor=start_factor, total_iters=total_iters)
                cosine = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=T_max, eta_min=eta_min)

                self.scheduler = optim.lr_scheduler.SequentialLR(
                    self.opt, [warm, cosine], milestones=[total_iters]
                )
                print(f"📉 Using SequentialLR warmup({total_iters}) → cosine.")
            else:
                _cfg = {k: v for k, v in sched_cfg.items() if k != "type"}
                sched_class = getattr(optim.lr_scheduler, sched_type, None)
                if sched_class is None:
                    raise ValueError(f"Scheduler {sched_type} not found in torch.optim.lr_scheduler.")
                self.scheduler = sched_class(self.opt, **_cfg)
                print(f"📉 Using scheduler: {sched_type}")

        # ===================== #
        #     CHECKPOINT CFG    #
        # ===================== #
        self.ckpt_interval = int(ckpt_cfg.get("interval", 10))
        self.ckpt_monitor = ckpt_cfg.get("monitor", "val_loss")
        self.ckpt_mode = ckpt_cfg.get("mode", "min")

        self.best_metric_value: Optional[float] = None
        self.best_epoch: Optional[int] = None

        self.keep_recent = int(ckpt_cfg.get("keep_recent", 2))
        self.keep_best = bool(ckpt_cfg.get("keep_best", True))
        self.keep_last = bool(ckpt_cfg.get("keep_last", True))
        self.auto_cleanup = bool(ckpt_cfg.get("auto_cleanup", True))

        # ---------------- Resume ----------------
        resume_cfg = ckpt_cfg.get("resume", False)
        if isinstance(resume_cfg, str) and os.path.exists(resume_cfg):
            self._resume_checkpoint(resume_cfg)
        elif resume_cfg is True:
            last_ckpt = os.path.join(self.ckpt_dir, "last.pt")
            if os.path.exists(last_ckpt):
                print(f"🔄 Found last checkpoint: {last_ckpt}, resuming...")
                self._resume_checkpoint(last_ckpt)
            else:
                print("▶️ Starting training from scratch (no last.pt found)")
        else:
            print("▶️ Starting training from scratch (resume disabled)")

    # ============================================================
    # TRAIN
    # ============================================================
    def train(self):
        for epoch in range(self.epoch + 1, self.epochs + 1):
            self.epoch = epoch

            train_loss = self.train_one_epoch()
            val_loss, metrics = self.validate()

            lr = float(self.opt.param_groups[0]["lr"])
            self.logger.log({
                "epoch": epoch,
                "phase": "epoch_summary",
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "lr": round(lr, 10),
                "metrics": metrics or {},
            })

            self._save_checkpoints(metrics, val_loss=val_loss)

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        self.logger.close()

    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss_epoch = 0.0
        num_iters = len(self.train_loader)

        for it, batch in enumerate(self.train_loader, 1):
            x_noisy = batch["noisy"].to(self.device, non_blocking=True)
            x_clean = batch["clean"].to(self.device, non_blocking=True)

            self.opt.zero_grad(set_to_none=True)

            # teacher-forcing style forward (AR)
            x_hat = self.model(x_noisy, x_clean)

            recon_loss = self.lambda_recon * self.criterion(x_hat, x_clean)

            vq_loss = getattr(self.model, "last_vq_loss", None)
            if vq_loss is None:
                vq_loss = torch.zeros((), device=self.device)
            vq_loss_weighted = self.lambda_vq * vq_loss

            # delineation head is typically trained via generate(...) not forward(...) in your codebase,
            # so we keep it OFF for training by default unless you explicitly add it.
            del_loss_weighted = torch.zeros((), device=self.device)

            total = recon_loss + vq_loss_weighted + del_loss_weighted
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.cfg["train"].get("clip_grad", 1.0)))
            self.opt.step()

            total_loss_epoch += float(total.detach().item())

            # logging
            current_lr = float(self.opt.param_groups[0]["lr"])
            if it == 1 or it == num_iters or (it % int(self.cfg["train"].get("log_every", 50)) == 0):
                timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                msg = (
                    f"{timestamp} | epoch: {self.epoch}/{self.epochs} | "
                    f"iter: {it}/{num_iters} | LR: {current_lr:.8e} | "
                    f"total: {float(total):.6f} | recon: {float(recon_loss):.6f} | "
                    f"vq(w): {float(vq_loss_weighted):.6f}"
                )
                print(msg, flush=True)

                self.logger.log({
                    "epoch": self.epoch,
                    "iter": it,
                    "phase": "train",
                    "lr": round(current_lr, 10),
                    "loss_total": round(float(total), 6),
                    "loss_recon": round(float(recon_loss), 6),
                    "loss_vq": round(float(vq_loss_weighted), 6),
                })

        return total_loss_epoch / max(num_iters, 1)

    # ============================================================
    # VALIDATE
    # ============================================================
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0

        all_pred: List[torch.Tensor] = []
        all_clean: List[torch.Tensor] = []
        all_noisy: List[torch.Tensor] = []
        all_mu: List[torch.Tensor] = []
        all_sigma: List[torch.Tensor] = []

        val_iter = tqdm(self.val_loader, desc=f"🧪 Validation Epoch {self.epoch}", ncols=100)

        for batch in val_iter:
            x_noisy = batch["noisy"].to(self.device, non_blocking=True)
            x_clean = batch["clean"].to(self.device, non_blocking=True)

            mu = batch.get("mu", None)
            sigma = batch.get("sigma", None)
            if mu is not None and sigma is not None:
                all_mu.append(mu.detach().cpu().view(-1))
                all_sigma.append(sigma.detach().cpu().view(-1))

            # AR inference (generate)
            if self.has_delineation and self.delineation_loss_fn is not None:
                x_hat, del_logits = self.model.generate(x_noisy, return_delineation=True)

                del_loss = torch.zeros((), device=self.device)
                del_loss_weighted = torch.zeros((), device=self.device)
                heatmap = batch.get("heatmap", None)
                mask_weight = batch.get("mask_weight", None)
                if heatmap is not None:
                    heatmap = heatmap.to(self.device, non_blocking=True)
                    if mask_weight is not None:
                        mask_weight = mask_weight.to(self.device, non_blocking=True)
                    del_loss = self.delineation_loss_fn(del_logits, heatmap, mask=mask_weight)
                    del_loss_weighted = self.lambda_delineation * del_loss
            else:
                x_hat = self.model.generate(x_noisy)
                del_loss = torch.zeros((), device=self.device)
                del_loss_weighted = torch.zeros((), device=self.device)

            recon_loss = self.lambda_recon * self.criterion(x_hat, x_clean)

            vq_loss = getattr(self.model, "last_vq_loss", None)
            if vq_loss is None:
                vq_loss = torch.zeros((), device=self.device)
            vq_loss_weighted = self.lambda_vq * vq_loss

            total = recon_loss + vq_loss_weighted + del_loss_weighted
            total_loss += float(total.detach().item())

            val_iter.set_postfix({
                "val_total": f"{float(total):.6f}",
                "recon": f"{float(recon_loss):.4f}",
                "vq(w)": f"{float(vq_loss_weighted):.4f}",
                **({"del(w)": f"{float(del_loss_weighted):.4f}"} if self.has_delineation and self.lambda_delineation > 0 else {}),
            })

            all_pred.append(x_hat.detach().cpu())
            all_clean.append(x_clean.detach().cpu())
            all_noisy.append(x_noisy.detach().cpu())

        val_loss = total_loss / max(len(self.val_loader), 1)

        # metrics
        selected_metrics = self.cfg.get("eval", {}).get("metrics", [])
        metrics: Dict[str, float] = {}
        if selected_metrics:
            pred = torch.cat(all_pred, dim=0)
            clean = torch.cat(all_clean, dim=0)
            noisy = torch.cat(all_noisy, dim=0)
            mu_cat = torch.cat(all_mu, dim=0) if len(all_mu) > 0 else None
            sigma_cat = torch.cat(all_sigma, dim=0) if len(all_sigma) > 0 else None

            metrics = evaluate_metrics(
                pred, clean, noisy=noisy,
                mu=mu_cat, sigma=sigma_cat,
                selected=selected_metrics
            )

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        lr_value = float(self.opt.param_groups[0]["lr"])
        metric_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()]) if metrics else ""
        print(
            f"{timestamp} | epoch: {self.epoch}/{self.epochs} | phase: val | "
            f"val_loss: {val_loss:.6f} | LR: {lr_value:.8e}"
            + (f" | {metric_str}" if metric_str else ""),
            flush=True
        )

        self.logger.log({
            "epoch": self.epoch,
            "phase": "val",
            "val_loss": round(val_loss, 6),
            "lr": round(lr_value, 10),
            "metrics": {k: round(v, 6) for k, v in metrics.items()},
        })

        return val_loss, metrics

    # ============================================================
    # CHECKPOINTS
    # ============================================================
    def _make_save_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.opt.state_dict() if self.opt else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": {
                self.ckpt_monitor: self.best_metric_value,
                "epoch": self.best_epoch,
            },
            "config": self.cfg,
        }

    def _save_checkpoints(self, metrics: Optional[Dict[str, float]], val_loss: Optional[float] = None):
        metrics_for_monitor = dict(metrics) if metrics else {}
        if val_loss is not None:
            metrics_for_monitor["val_loss"] = float(val_loss)

        def _current_lr() -> Optional[float]:
            return float(self.opt.param_groups[0]["lr"]) if self.opt is not None else None

        # last
        last_path = os.path.join(self.ckpt_dir, "last.pt")
        torch.save(self._make_save_dict(), last_path)
        self.logger.log({"phase": "checkpoint", "file": last_path, "type": "last", "lr": _current_lr()})

        # periodic
        if self.ckpt_interval and self.epoch % int(self.ckpt_interval) == 0:
            ckpt_path = os.path.join(self.ckpt_dir, f"epoch_{self.epoch:03d}.pt")
            torch.save(self._make_save_dict(), ckpt_path)
            print(f"Saved interval checkpoint at epoch {self.epoch}: {ckpt_path}")
            self.logger.log({
                "phase": "checkpoint",
                "file": ckpt_path,
                "type": "periodic",
                "epoch": self.epoch,
                "lr": _current_lr()
            })

        # best
        if metrics_for_monitor and self.ckpt_monitor in metrics_for_monitor:
            current_value = float(metrics_for_monitor[self.ckpt_monitor])
            better = (
                self.best_metric_value is None
                or (self.ckpt_mode == "min" and current_value < float(self.best_metric_value))
                or (self.ckpt_mode == "max" and current_value > float(self.best_metric_value))
            )
            if better:
                prev_best = self.best_metric_value
                self.best_metric_value = current_value
                self.best_epoch = self.epoch

                best_name = f"best_{self.ckpt_monitor}_epoch_{self.epoch:03d}.pt"
                best_path = os.path.join(self.ckpt_dir, best_name)

                # remove older bests for this monitor
                for f in os.listdir(self.ckpt_dir):
                    if f.startswith(f"best_{self.ckpt_monitor}_") and f != os.path.basename(best_path):
                        try:
                            os.remove(os.path.join(self.ckpt_dir, f))
                        except Exception:
                            pass

                torch.save(self._make_save_dict(), best_path)

                print("\n" + "=" * 70)
                print(f"NEW BEST {self.ckpt_monitor.upper()} at epoch {self.epoch}: {current_value:.6f}")
                if prev_best is not None:
                    diff = abs(current_value - float(prev_best))
                    trend = "↓" if self.ckpt_mode == "min" else "↑"
                    print(f"    (previous: {float(prev_best):.6f}, improvement: {diff:.6f} {trend})")
                lr_now = _current_lr()
                if lr_now is not None:
                    print(f"LR at save: {lr_now:.6e}")
                print("=" * 70 + "\n")

                self.logger.log({
                    "phase": "best_checkpoint",
                    "metric": self.ckpt_monitor,
                    "value": round(current_value, 6),
                    "epoch": self.epoch,
                    "file": best_path,
                    "lr": lr_now,
                })

        if self.auto_cleanup:
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        # keep_recent periodic checkpoints
        all_ckpts = sorted(
            [f for f in os.listdir(self.ckpt_dir) if f.startswith("epoch_") and f.endswith(".pt")]
        )
        if self.keep_recent >= 0 and len(all_ckpts) > self.keep_recent:
            to_delete = all_ckpts[:-self.keep_recent] if self.keep_recent > 0 else all_ckpts
            for f in to_delete:
                path = os.path.join(self.ckpt_dir, f)
                try:
                    os.remove(path)
                    self.logger.log({"phase": "cleanup", "file": path, "action": "deleted"})
                except Exception as e:
                    self.logger.log({"phase": "cleanup_error", "file": path, "error": str(e)})

        if not self.keep_last:
            last_path = os.path.join(self.ckpt_dir, "last.pt")
            if os.path.exists(last_path):
                try:
                    os.remove(last_path)
                    self.logger.log({"phase": "cleanup", "file": last_path, "action": "deleted (last)"})
                except Exception:
                    pass

        if not self.keep_best:
            for f in os.listdir(self.ckpt_dir):
                if f.startswith("best_") and f.endswith(".pt"):
                    try:
                        os.remove(os.path.join(self.ckpt_dir, f))
                        self.logger.log({"phase": "cleanup", "file": f, "action": "deleted (best)"})
                    except Exception:
                        pass

    def _resume_checkpoint(self, path: str):
        if not os.path.exists(path):
            print(f"Resume path not found: {path}")
            return

        print(f"🔄 Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state"])

        if "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None and self.opt is not None:
            self.opt.load_state_dict(checkpoint["optimizer_state"])

        if "scheduler_state" in checkpoint and checkpoint["scheduler_state"] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        self.epoch = int(checkpoint.get("epoch", 0))
        best_info = checkpoint.get("best_metric", {})
        self.best_metric_value = best_info.get(self.ckpt_monitor, None)
        self.best_epoch = best_info.get("epoch", None)

        print(f"Resumed at epoch {self.epoch} (best {self.ckpt_monitor}: {self.best_metric_value})")

        lr_now = float(self.opt.param_groups[0]["lr"]) if (self.opt and self.opt.param_groups) else None
        self.logger.log({
            "phase": "resume",
            "epoch": self.epoch,
            "best_metric": self.best_metric_value,
            "lr": lr_now,
        })
