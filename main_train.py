import yaml
import data, models, loss   
from engine.runner import Runner
from engine.hooks import TimerHook, LoggerHook, CheckpointHook
import torch


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ECG Denoising Training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    runner = Runner(cfg)
    runner.register_hook(TimerHook())
    runner.register_hook(LoggerHook())
    runner.train()


if __name__ == "__main__":
    main()