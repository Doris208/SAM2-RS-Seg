"""Read-only Lightning/SAM2 inventory; imports and device operations are opt-in."""

import argparse
import importlib
import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGES = ("torch", "torchvision", "lightning", "pytorch-lightning", "torchmetrics",
            "SAM-2", "numpy", "Pillow", "hydra-core", "iopath", "packaging")
# SAM2 declares the torch/vision floors. Other floors delimit candidate families,
# not a validated or exact version lock.
MINIMUMS = {"torch": "2.5.1", "torchvision": "0.20.1", "lightning": "2.5.0",
            "torchmetrics": "1.7.0", "numpy": "1.24.4", "Pillow": "9.4.0",
            "hydra-core": "1.3.2", "iopath": "0.1.10"}


def inventory():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    tools = {}
    for name in ("nvidia-smi", "nvcc"):
        command = shutil.which(name)
        if not command:
            tools[name] = {"available": False}
            continue
        try:
            result = subprocess.run([command, "--version"], capture_output=True,
                                    text=True, timeout=15, errors="replace")
            tools[name] = {"exit_code": result.returncode,
                           "output": (result.stdout + result.stderr).strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            tools[name] = {"error": str(exc)}
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": packages, "tools": tools,
            "runtime": "not requested",
            "candidate_image": "parallel-cloud PyTorch 25.03; actual contents unverified",
            "status": "inventory only; not an integration validation"}


def candidate_checks(report):
    """Check declarations only; do not infer that a wheel pair or GPU works."""
    issues = {}
    if sys.version_info[:2] < (3, 10):
        issues["python"] = "SAM2 requires Python >=3.10"
    if not report["packages"].get("SAM-2"):
        issues["SAM-2"] = "not installed; Git SHA/config/weight must also be verified"
    try:
        from packaging.version import Version
        for name, minimum in MINIMUMS.items():
            actual = report["packages"].get(name)
            if not actual or Version(actual) < Version(minimum):
                issues[name] = {"minimum_candidate": minimum, "actual": actual}
    except (ImportError, ValueError) as exc:
        issues["version_check"] = str(exc)
    report["candidate_issues"] = issues
    report["candidate_limit"] = (
        "Minimum declarations only. torch/torchvision pairing, image constraints, "
        "nightly builds, Lightning behavior and CUDA remain unverified.")
    return not issues


def runtime_checks(report, require_gpu=False):
    checks = {}
    for name in ("torch", "torchvision", "lightning.pytorch", "torchmetrics", "sam2"):
        try:
            module = importlib.import_module(name)
            checks[name] = {"ok": True, "path": str(getattr(module, "__file__", ""))}
        except Exception as exc:
            checks[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        import torch
        report["torch_cuda"] = torch.version.cuda
        device = "cpu"
        if require_gpu:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU required but unavailable")
            device = "cuda"
            report["gpu"] = {"name": torch.cuda.get_device_name(0),
                             "capability": list(torch.cuda.get_device_capability(0))}
        x = torch.ones(2, 2, device=device, requires_grad=True)
        (x @ x).sum().backward()
        if x.grad is None or not torch.isfinite(x.grad).all():
            raise RuntimeError("Invalid gradient in tiny backward check")
        checks["tensor_backward"] = {"ok": True, "device": device}
        from torchmetrics.classification import (
            MulticlassAccuracy, MulticlassF1Score, MulticlassJaccardIndex,
        )
        prediction = torch.tensor([0, 2, 0, 2], device=device)
        target = torch.tensor([0, 1, 255, 2], device=device)
        arguments = dict(num_classes=3, ignore_index=255)
        iou = MulticlassJaccardIndex(average=None, **arguments).to(device)(prediction, target)
        f1 = MulticlassF1Score(average=None, **arguments).to(device)(prediction, target)
        oa = MulticlassAccuracy(average="micro", **arguments).to(device)(prediction, target)
        if not (torch.allclose(iou, torch.tensor([1., 0., .5], device=device))
                and torch.allclose(f1, torch.tensor([1., 0., 2/3], device=device))
                and torch.allclose(oa, torch.tensor(2/3, device=device))):
            raise RuntimeError("Unexpected TorchMetrics result or ignore handling")
        checks["synthetic_metrics"] = {"ok": True, "device": device, "iou": iou.tolist(),
                                       "f1": f1.tolist(), "oa": oa.item()}
    except Exception as exc:
        checks["tensor_and_metrics"] = {"ok": False, "error": str(exc)}
    try:
        from lightning.pytorch import LightningDataModule, LightningModule, Trainer
        from sam2.modeling.backbones.hieradet import Hiera
        if not all(callable(cls) for cls in (LightningDataModule, LightningModule, Trainer, Hiera)):
            raise RuntimeError("Expected training/model classes are unavailable")
        checks["lightning_and_hiera_import"] = {"ok": True}
    except Exception as exc:
        checks["lightning_and_hiera_import"] = {"ok": False, "error": str(exc)}
    report["runtime"] = checks
    report["status"] = "smoke checks only; no Trainer run, Hiera forward or weight loading"
    return all(item["ok"] for item in checks.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true", help="Opt-in CPU imports/backward/metrics")
    parser.add_argument("--require-gpu", action="store_true", help="Opt-in CUDA smoke operations")
    parser.add_argument("--candidate", action="store_true", help="Check minimum declarations, not a lock")
    parser.add_argument("--json", type=Path, help="Save report (prefer outputs/)")
    args = parser.parse_args(argv)
    report = inventory()
    ok = candidate_checks(report) if args.candidate else True
    if args.runtime or args.require_gpu:
        ok = runtime_checks(report, args.require_gpu) and ok
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
