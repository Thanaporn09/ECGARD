import os
import time
import datetime
import torch
from utils.metrics import evaluate_metrics

class Hook:
    priority = 50
    def before_train(self, runner): pass
    def after_train(self, runner): pass
    def before_train_epoch(self, runner): pass
    def after_train_epoch(self, runner): pass
    def after_train_iter(self, runner): pass
    def after_val_epoch(self, runner): pass


class TimerHook(Hook):
    priority = 10
    def before_train(self, runner):
        runner._start_time = time.time()
        runner._epoch_times = []

    def before_train_epoch(self, runner):
        runner._epoch_start = time.time()

    def after_train_epoch(self, runner):
        epoch_time = time.time() - runner._epoch_start
        runner._epoch_times.append(epoch_time)
        avg = sum(runner._epoch_times) / len(runner._epoch_times)
        remaining = (runner.epochs - runner.epoch) * avg / 60
        print(f"[TimerHook] Epoch Time: {epoch_time:.2f}s | ETA: ~{remaining:.1f} min")


class LoggerHook:
    priority = 50

    def after_val_epoch(self, runner):
        time_now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        # Core metrics
        train_loss = getattr(runner, "train_loss", None)
        val_loss = getattr(runner, "val_loss", None)
        metrics = getattr(runner, "val_metrics", {})

        msg = f"{time_now} | Epoch {runner.epoch} | "
        if train_loss is not None:
            msg += f"Train Loss: {train_loss:.6f} | "
        if val_loss is not None:
            msg += f"Val Loss: {val_loss:.6f}"

        # --- Best metric detection ---
        ckpt_monitor = getattr(runner, "ckpt_monitor", "val_loss")
        ckpt_mode = getattr(runner, "ckpt_mode", "min")

        if metrics and ckpt_monitor in metrics:
            current_value = metrics[ckpt_monitor]
            best_value = getattr(runner, "best_metric_value", None)
            improved = False

            if best_value is None:
                improved = True
            elif ckpt_mode == "min" and current_value < best_value:
                improved = True
            elif ckpt_mode == "max" and current_value > best_value:
                improved = True

            if improved:
                runner.best_metric_value = current_value
                runner.best_epoch = runner.epoch
                msg += f" | ✅ New best {ckpt_monitor}: {current_value:.4f} (epoch {runner.epoch})"
                runner.logger.log({
                    "phase": "best_metric",
                    "epoch": runner.epoch,
                    "metric": ckpt_monitor,
                    "value": round(current_value, 6)
                })
            else:
                msg += f" | Best {ckpt_monitor}: {best_value:.4f} (epoch {runner.best_epoch})"

        print(msg)
        runner.logger.log({
            "phase": "val_summary",
            "epoch": runner.epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "metrics": metrics
        })


class CheckpointHook(Hook):
    priority = 30
    def __init__(self, interval=5, out_dir="checkpoints"):
        self.interval = interval
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def after_val_epoch(self, runner):
        if runner.epoch % self.interval == 0:
            path = os.path.join(self.out_dir, f"epoch_{runner.epoch:03d}.pt")
            torch.save(runner.model.state_dict(), path)
            print(f"[CheckpointHook] Saved: {path}")

class MetricHook(Hook):
    priority = 40
    def after_val_epoch(self, runner):
        selected = runner.cfg.get("eval", {}).get("metrics", [])
        pred, clean = runner.last_pred, runner.last_clean
        metrics = evaluate_metrics(pred, clean, selected)
        runner.metrics = metrics
        print("📈 Metrics:", metrics)