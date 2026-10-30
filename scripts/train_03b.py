"""Config-driven synthetic training; real remote-sensing datasets are deferred."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Official SAM2 initialization only")
    parser.add_argument("--resume", type=Path, help="Full Lightning training checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test", action="store_true", help="Test the best validation checkpoint after fit")
    args = parser.parse_args(argv)
    if args.checkpoint and args.resume:
        parser.error("--checkpoint and --resume have different purposes; choose one")
    from projects.sam2_rs_seg.utils.config import load_config
    from projects.sam2_rs_seg.engine.experiment import build_experiment, fit_experiment, write_json

    config = load_config(args.config)
    if args.checkpoint:
        config["model"]["params"].update(pretrained=True, checkpoint_path=str(args.checkpoint.resolve()))
    if args.resume:
        import torch
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        # Restore the original initialization description, not the initialization
        # file itself. Other protocol changes are rejected by on_load_checkpoint.
        previous_model = saved["experiment_config"]["model"]
        if config["model"]["name"] != previous_model["name"]:
            raise ValueError("Full resume cannot change the model name")
        for key in ("pretrained", "checkpoint_path", "device"):
            if key in previous_model.get("params", {}):
                config["model"].setdefault("params", {})[key] = previous_model["params"][key]
    trainer, module, data, checkpoint = build_experiment(config, args.output_dir, resume=args.resume)
    last = fit_experiment(trainer, module, data, args.output_dir, resume=args.resume)
    if not checkpoint.best_model_path:
        raise RuntimeError("No validation checkpoint was saved; check training/validation limits")
    result = {"global_step": trainer.global_step, "epoch": trainer.current_epoch,
              "last_checkpoint": str(last), "best_checkpoint": checkpoint.best_model_path,
              "best_val_mIoU": checkpoint.best_model_score, "metrics": module.epoch_results}
    if args.test:
        result["test"] = trainer.test(module, datamodule=data, ckpt_path="best")
    write_json(args.output_dir / "train_report.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
