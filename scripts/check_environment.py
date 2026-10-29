"""Read-only environment inventory; deep imports are opt-in, no installations."""

import argparse
import importlib
import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGES = ("torch", "torchvision", "mmcv", "mmcv-lite", "mmengine",
            "mmsegmentation", "SAM-2", "numpy", "Pillow")
CANDIDATE = {"torch": "2.5.1", "torchvision": "0.20.1", "mmcv": "2.1.0",
             "mmengine": "0.10.7", "mmsegmentation": "1.2.2", "numpy": "1.26.4"}


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
        if command:
            try:
                result = subprocess.run([command, "--version"], capture_output=True,
                                        text=True, timeout=15, errors="replace")
                tools[name] = {"exit_code": result.returncode,
                               "output": (result.stdout + result.stderr).strip()}
            except (OSError, subprocess.TimeoutExpired) as exc:
                tools[name] = {"error": str(exc)}
        else:
            tools[name] = {"available": False}
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": packages,
            "tools": tools, "runtime": "not requested",
            "status": "inventory only; not an integration validation"}


def runtime_checks(report, require_gpu):
    checks = {}
    for name in ("torch", "torchvision", "mmcv", "mmengine", "mmseg", "sam2"):
        try:
            module = importlib.import_module(name)
            checks[name] = {"ok": True, "path": str(getattr(module, "__file__", ""))}
        except Exception as exc:
            checks[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        import torch
        report["torch_cuda"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required but unavailable")
        device = "cuda" if require_gpu else "cpu"
        x = torch.ones(2, 2, device=device, requires_grad=True)
        (x @ x).sum().backward()
        from mmcv.ops import nms
        boxes = torch.tensor([[0., 0., 4., 4.], [0., 0., 4., 4.]], device=device)
        scores = torch.tensor([0.9, 0.8], device=device)
        _, keep = nms(boxes, scores, 0.5)
        if keep.numel() != 1:
            raise RuntimeError("Unexpected NMS result")
        checks["torch_backward_and_mmcv_nms"] = {"ok": True, "device": device}
    except Exception as exc:
        checks["torch_backward_and_mmcv_nms"] = {"ok": False, "error": str(exc)}
    try:
        from mmseg.utils import register_all_modules
        from sam2.modeling.backbones.hieradet import Hiera
        register_all_modules(init_default_scope=True)
        checks["registry_and_hiera_import"] = {"ok": callable(Hiera)}
    except Exception as exc:
        checks["registry_and_hiera_import"] = {"ok": False, "error": str(exc)}
    report["runtime"] = checks
    report["status"] = "smoke checks only; model/data integration still unverified"
    return all(item["ok"] for item in checks.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true", help="Import DL packages and run tiny ops")
    parser.add_argument("--require-gpu", action="store_true", help="Also require and exercise CUDA")
    parser.add_argument("--candidate", action="store_true", help="Check exact candidate package versions")
    parser.add_argument("--json", type=Path, help="Save report (prefer outputs/)")
    args = parser.parse_args()
    report = inventory()
    ok = True
    if args.candidate:
        mismatches = {name: {"expected": wanted, "actual": report["packages"][name]}
                      for name, wanted in CANDIDATE.items()
                      if (report["packages"][name] or "").split("+")[0] != wanted}
        if sys.version_info[:2] != (3, 10):
            mismatches["python"] = {"expected": "3.10", "actual": platform.python_version()}
        if report["packages"]["mmcv-lite"]:
            mismatches["mmcv-lite"] = {"expected": "not installed",
                                      "actual": report["packages"]["mmcv-lite"]}
        report["candidate_mismatches"] = mismatches
        ok = not mismatches
    if args.runtime or args.require_gpu:
        ok = runtime_checks(report, args.require_gpu) and ok
        if args.candidate and report.get("torch_cuda") != "12.1":
            report["candidate_cuda_error"] = "Expected PyTorch CUDA runtime 12.1"
            ok = False
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
