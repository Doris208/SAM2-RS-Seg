"""RTX 4090 acceptance: fit, evaluate, full resume and native/Lightning parity."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import math
import os
from pathlib import Path
import sys
import traceback
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Full official SAM2.1 initialization")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke/task03b.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task03b"))
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
                            + uuid.uuid4().hex[:8])
    run_dir.mkdir()
    report = {"status": "FAILED", "stage": "preflight", "run_dir": str(run_dir),
              "scope": "Synthetic data only; metrics are NOT remote-sensing performance"}
    try:
        import torch
        from projects.sam2_rs_seg.engine.experiment import (
            build_experiment, environment_info, fit_experiment, write_json,
        )
        from projects.sam2_rs_seg.models.registry import build_model
        from projects.sam2_rs_seg.utils.config import load_config
        from scripts.task03a_checks import multiscale_gradient_report
        from scripts.task03b_checks import TrainingAudit, check_learning_rates, compare_single_step

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; no CPU fallback or substitute acceptance")
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        config = load_config(args.config)
        config["model"]["params"].update(pretrained=True, checkpoint_path=str(args.checkpoint.resolve()))
        if config["model"]["name"] != "SAM2SemanticSegmentor" or config["model"].get("decode_head") is not None:
            raise ValueError("Official GPU acceptance requires the unchanged 03A model/head")
        if config["trainer"]["accelerator"] != "gpu" or config["trainer"]["precision"] != "32-true":
            raise ValueError("GPU acceptance requires accelerator=gpu, precision=32-true")
        if config["scheduler"].get("interval", "step") != "step":
            raise ValueError("This optimizer-step resume smoke requires scheduler interval=step")
        target_steps = config["trainer"].get("max_steps", -1)
        batches = math.ceil(config["data"]["train_samples"] / config["data"]["batch_size"])
        initial_steps = math.ceil(batches / config["trainer"]["accumulate_grad_batches"])
        if not initial_steps < target_steps <= 16:
            raise ValueError("Smoke requires >1 epoch of updates and at most 16 optimizer steps")
        if (config["scheduler"]["name"] == "PolynomialLR"
                and config["scheduler"]["total_iters"] < target_steps):
            raise ValueError("Smoke optimizer updates must remain within the declared LR budget")
        if config["trainer"]["val_check_interval"] != 1.0:
            raise ValueError("Resume smoke checkpoints at a complete epoch boundary")
        report["environment"] = {**environment_info(), "gpu": torch.cuda.get_device_name(0)}
        write_json(run_dir / "requested_config.json", config)

        def gradient_check(model):
            official = model.encoder.image_encoder
            return multiscale_gradient_report(official.trunk, official.neck, model.decode_head)

        probes = ["encoder.image_encoder.trunk.patch_embed.proj.weight",
                  "encoder.image_encoder.trunk.blocks.9.attn.qkv.weight",
                  "encoder.image_encoder.trunk.blocks.11.attn.qkv.weight",
                  *(f"encoder.image_encoder.neck.convs.{i}.conv.weight" for i in range(4)),
                  "decode_head.layers.0.weight", "decode_head.layers.3.weight"]
        report["stage"] = "initial_training"
        first_config = deepcopy(config)
        first_config["trainer"]["max_steps"] = initial_steps
        first_audit = TrainingAudit(probes=probes, gradient_check=gradient_check)
        trainer, module, data, checkpoint = build_experiment(first_config, run_dir, callbacks=[first_audit])
        if not module.model.encoder.pretrained_loaded:
            raise AssertionError("Official pretrained initialization did not complete")
        report["official_pretrained_loaded"] = True
        import sam2
        report["environment"]["sam2_source"] = str(Path(sam2.__file__).resolve())
        last = fit_experiment(trainer, module, data, run_dir)
        if trainer.global_step != initial_steps or not checkpoint.best_model_path or not Path(checkpoint.best_model_path).is_file():
            raise AssertionError("Initial updates/validation/best checkpoint did not complete")
        updates = first_audit.parameter_updates(module.model)
        if set(updates) != set(probes) or not all(value > 0 for value in updates.values()):
            raise AssertionError("A required deep/neck/head parameter did not update")
        report["initial_training"] = {"global_step": trainer.global_step,
                                      "epoch": trainer.current_epoch, "losses": first_audit.losses,
                                      "metrics": module.epoch_results, "parameter_update_max_abs": updates,
                                      "first_update_gradients": first_audit.first_gradients}
        # This is our own complete training checkpoint, not the official .pt.
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["global_step"] != initial_steps or not saved["optimizer_states"][0]["state"]:
            raise AssertionError("Last checkpoint lacks current optimizer progress")
        if not saved.get("experiment_config"):
            raise AssertionError("Effective experiment configuration absent from checkpoint")
        first_steps = first_audit.steps
        del trainer, module, data, checkpoint, first_audit
        gc.collect()
        torch.cuda.empty_cache()

        report["stage"] = "full_resume"
        resume_audit = TrainingAudit(checkpoint=saved, probes=probes, gradient_check=gradient_check)
        trainer, module, data, checkpoint = build_experiment(config, run_dir, resume=last, callbacks=[resume_audit])
        last = fit_experiment(trainer, module, data, run_dir, resume=last)
        if trainer.global_step != target_steps or resume_audit.restored is None:
            raise AssertionError("Full resume did not restore and advance the target global_step")
        if len(resume_audit.steps) != target_steps - initial_steps:
            raise AssertionError("Unexpected number of resumed optimizer updates")
        all_steps = first_steps + resume_audit.steps
        if [step["global_step_before"] for step in all_steps] != list(range(target_steps)):
            raise AssertionError("Optimizer step sequence has a gap or duplicate")
        final = torch.load(last, map_location="cpu", weights_only=False)
        if final["global_step"] != target_steps or not last.is_file():
            raise AssertionError("Final last.ckpt progress is stale")
        adam_steps = [float(state["step"]) for state in final["optimizer_states"][0]["state"].values()]
        if not adam_steps or any(step != target_steps for step in adam_steps):
            raise AssertionError("AdamW state counters did not continue through resume")
        if final["lr_schedulers"] and final["lr_schedulers"][0]["last_epoch"] != target_steps:
            raise AssertionError("Scheduler update counter differs from optimizer global_step")
        if not checkpoint.best_model_path or not Path(checkpoint.best_model_path).is_file():
            raise AssertionError("Final best validation checkpoint missing")
        report["resume"] = {"restored": resume_audit.restored, "final_global_step": trainer.global_step,
                            "final_epoch": trainer.current_epoch, "losses": resume_audit.losses,
                            "adam_state_steps": sorted(set(adam_steps)),
                            "scheduler_state": final["lr_schedulers"],
                            "last_checkpoint": str(last), "best_checkpoint": checkpoint.best_model_path,
                            "best_val_mIoU": checkpoint.best_model_score}
        report["learning_rate"] = check_learning_rates(all_steps, config["optimizer"], config["scheduler"])
        report["stage"] = "validation_and_test"
        trainer.validate(module, datamodule=data, ckpt_path=None)
        first_matrix = module.epoch_results["val"]["confusion_matrix"].clone()
        trainer.validate(module, datamodule=data, ckpt_path=None)
        if not torch.equal(first_matrix, module.epoch_results["val"]["confusion_matrix"]):
            raise AssertionError("Repeated validation accumulated stale statistics")
        trainer.test(module, datamodule=data, ckpt_path=None)
        report["evaluation"] = {"validation_reset_correct": True, "results": module.epoch_results}
        for split in ("val", "test"):
            result = module.epoch_results[split]
            if result["valid_pixels"].item() <= 0 or not all(torch.isfinite(result[key]).item() for key in ("mIoU", "mF1", "OA")):
                raise AssertionError(f"Invalid {split} dataset metrics")
        parity_batch = next(iter(data.val_dataloader()))
        del trainer, module, data, checkpoint, resume_audit, saved, final
        gc.collect()
        torch.cuda.empty_cache()

        report["stage"] = "native_lightning_single_step"
        parity_model = build_model(config["model"], config["data"])
        report["single_step"] = compare_single_step(
            parity_model, parity_batch, config["optimizer"], num_classes=config["data"]["num_classes"],
            ignore_index=config["data"]["ignore_index"], device="cuda", seed=config["seed"],
            loss_config=config["loss"])
        if not report["single_step"]["ok"]:
            raise AssertionError("Native/Lightning numerical consistency failed; tolerances were not enlarged")
        report.update(status="PASS", stage="complete")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    # Even dependency/preflight failures must emit a non-PASS report.
    import json
    try:
        from projects.sam2_rs_seg.engine.experiment import json_value
        report = json_value(report)
    except ImportError:
        pass
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    for path in (run_dir / "smoke_report.json", output_dir / "smoke_report.json"):
        path.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
