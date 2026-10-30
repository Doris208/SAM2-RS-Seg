"""One float32 forward/CE/backward; no optimization, downloads or training."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument("--checkpoint", type=Path, help="Full official SAM2.1 checkpoint")
    weights.add_argument("--random-init", action="store_true", help="NOT pretrained validation")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--head-channels", type=int, default=128)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=Path, help="Write diagnostics, preferably under outputs/")
    args = parser.parse_args(argv)

    report = {"status": "FAILED", "checkpoint_loaded": False,
              "gradient_policy": "three_scale_all_stages_v2",
              "validation": "random initialization" if args.random_init else "checkpoint",
              "seed": args.seed, "config": args.sam2_config,
              "checkpoint": str(args.checkpoint) if args.checkpoint else None}
    try:
        import torch
        from torch import nn

        from projects.sam2_rs_seg.models import SAM2SemanticSegmentor
        from scripts.task03a_checks import (
            multiscale_gradient_report, parameter_counts, synthetic_batch, validate_masks,
        )

        device = torch.device(args.device)
        report["environment"] = {"device": str(device), "torch": torch.__version__,
                                 "cuda_build": torch.version.cuda,
                                 "cuda_available": torch.cuda.is_available(),
                                 "precision": "float32", "python": sys.version}
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
            report["environment"]["gpu"] = torch.cuda.get_device_name(device)
        torch.manual_seed(args.seed)
        model = SAM2SemanticSegmentor(
            num_classes=args.num_classes, head_channels=args.head_channels,
            pretrained=not args.random_init, checkpoint_path=args.checkpoint,
            sam2_config=args.sam2_config, device=device,
        ).float().train()
        report["checkpoint_loaded"] = model.encoder.pretrained_loaded
        report["checkpoint_loading"] = (
            "official build_sam2 full-model strict load completed before ImageEncoder extraction"
            if model.encoder.pretrained_loaded else "NOT loaded; random initialization")
        official = model.encoder.image_encoder
        import sam2
        report["environment"]["sam2_source"] = str(Path(sam2.__file__).resolve())
        report["official_config"] = {"scalp": official.scalp,
                                     "fpn_top_down_levels": official.neck.fpn_top_down_levels}
        report["parameters"] = {name: parameter_counts(module) for name, module in (
            ("model", model), ("hiera", official.trunk),
            ("fpn_neck", official.neck), ("semantic_head", model.decode_head))}
        shapes = {}

        def capture_trunk(module, inputs, output):
            shapes["hiera"] = [list(feature.shape) for feature in output]

        def capture_neck(module, inputs, output):
            shapes["fpn_neck_before_scalp"] = [list(feature.shape) for feature in output[0]]

        def capture_encoder(module, inputs, output):
            shapes["backbone_fpn"] = [list(feature.shape) for feature in output["backbone_fpn"]]

        def capture_fusion(module, inputs):
            shapes["semantic_fusion"] = list(inputs[0].shape)

        handles = [official.trunk.register_forward_hook(capture_trunk),
                   official.neck.register_forward_hook(capture_neck),
                   official.register_forward_hook(capture_encoder),
                   model.decode_head.layers[0].register_forward_pre_hook(capture_fusion)]
        images, masks = synthetic_batch(1, args.height, args.width, args.num_classes, device)
        validate_masks(masks, args.num_classes)
        try:
            logits = model(images)
        finally:
            for handle in handles:
                handle.remove()
        shapes["images"] = list(images.shape)
        shapes["logits"] = list(logits.shape)
        report["shapes"] = shapes
        expected = [1, args.num_classes, args.height, args.width]
        if list(logits.shape) != expected or not torch.isfinite(logits).all().item():
            raise RuntimeError("Invalid logits shape or nonfinite values")
        # These contracts describe Tiny; alternate configs must be assessed separately.
        if args.sam2_config == "configs/sam2.1/sam2.1_hiera_t.yaml":
            expected_trunk = [[1, c, args.height // s, args.width // s]
                              for c, s in zip((96, 192, 384, 768), (4, 8, 16, 32))]
            expected_fpn = [[1, 256, args.height // s, args.width // s] for s in (4, 8, 16)]
            if shapes["hiera"] != expected_trunk or shapes["backbone_fpn"] != expected_fpn:
                raise RuntimeError("Official Tiny feature contract mismatch")
        expected_fusion = shapes["backbone_fpn"][0].copy()
        expected_fusion[1] *= 3
        if shapes["semantic_fusion"] != expected_fusion:
            raise RuntimeError("Three-scale concatenation shape mismatch")
        report["labels"] = {"ids": masks.unique().tolist(), "ignore_index": 255,
                            "valid_pixels": (masks != 255).sum().item()}
        loss = nn.CrossEntropyLoss(ignore_index=255)(logits, masks)
        if not torch.isfinite(loss).item():
            raise RuntimeError("CrossEntropyLoss is nonfinite")
        report["cross_entropy"] = loss.item()
        loss.backward()
        report["gradients"] = multiscale_gradient_report(
            official.trunk, official.neck, model.decode_head)
        if not report["gradients"]["ok"]:
            raise RuntimeError("Incomplete multiscale gradient coverage: "
                               + ", ".join(report["gradients"]["failures"]))
        report["gradient_note"] = (
            "All three backbone_fpn levels feed the loss. Every Hiera block, all four "
            "neck projections and the semantic head must have finite connected gradients. "
            "All block QKV weights (including blocks.9 and blocks.11), neck projection "
            "weights and semantic fusion/classification weights must have nonzero gradients. "
            "No optimizer step is performed.")
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
