"""Acceptance diagnostics, separate from the production Lightning module."""

from copy import deepcopy
import math

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from projects.sam2_rs_seg.engine.lightning_module import SegmentationLightningModule
from projects.sam2_rs_seg.engine.optim import build_optimizer
from projects.sam2_rs_seg.losses.segmentation import build_loss


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return deepcopy(value)


def assert_state_equal(actual, expected, path="state"):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
            raise AssertionError(f"Restored tensor differs at {path}")
    elif isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"Restored keys differ at {path}")
        for key in expected:
            assert_state_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"Restored sequence differs at {path}")
        for index, item in enumerate(expected):
            assert_state_equal(actual[index], item, f"{path}.{index}")
    elif actual != expected:
        raise AssertionError(f"Restored value differs at {path}: {actual!r} != {expected!r}")


class TrainingAudit(pl.Callback):
    """Observe public hooks: restore BEFORE updates and LR used BY each update."""

    def __init__(self, *, checkpoint=None, probes=(), gradient_check=None):
        self.checkpoint, self.probes = checkpoint, tuple(probes)
        self.gradient_check = gradient_check
        self.steps, self.losses = [], []
        self.restored = None
        self.first_gradients = None

    def on_train_start(self, trainer, module):
        self.initial_parameters = {name: parameter.detach().cpu().clone()
                                   for name, parameter in module.model.named_parameters() if name in self.probes}
        if self.checkpoint is not None:
            checkpoint = self.checkpoint
            if trainer.global_step != checkpoint["global_step"]:
                raise AssertionError("global_step not restored")
            if not checkpoint["epoch"] <= trainer.current_epoch <= checkpoint["epoch"] + 1:
                raise AssertionError("Epoch did not restore at the saved epoch boundary")
            if len(trainer.optimizers) != len(checkpoint["optimizer_states"]):
                raise AssertionError("Optimizer count differs")
            for actual, expected in zip(trainer.optimizers, checkpoint["optimizer_states"]):
                assert_state_equal(actual.state_dict(), expected, "optimizer")
            if len(trainer.lr_scheduler_configs) != len(checkpoint["lr_schedulers"]):
                raise AssertionError("Scheduler count differs")
            for actual, expected in zip(trainer.lr_scheduler_configs, checkpoint["lr_schedulers"]):
                assert_state_equal(actual.scheduler.state_dict(), expected, "scheduler")
            assert_state_equal(module.state_dict(), checkpoint["state_dict"], "model")
            self.restored = {"global_step": trainer.global_step, "epoch": trainer.current_epoch,
                             "saved_epoch": checkpoint["epoch"], "optimizer_exact": True,
                             "scheduler_exact": True, "model_exact": True,
                             "learning_rates": [group["lr"] for group in trainer.optimizers[0].param_groups]}

    def on_before_optimizer_step(self, trainer, module, optimizer):
        self.steps.append({"global_step_before": trainer.global_step,
                           "epoch": trainer.current_epoch,
                           "learning_rates": [group["lr"] for group in optimizer.param_groups]})
        if self.first_gradients is None and self.gradient_check is not None:
            self.first_gradients = self.gradient_check(module.model)
            if not self.first_gradients["ok"]:
                raise AssertionError("Deep SAM2 gradient coverage failed")

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if outputs and "loss" in outputs:
            loss = float(outputs["loss"].detach().cpu())
            if not math.isfinite(loss):
                raise AssertionError("Training loss is nonfinite")
            self.losses.append(loss)

    def parameter_updates(self, model):
        current = dict(model.named_parameters())
        return {name: float((current[name].detach().cpu() - previous).abs().max())
                for name, previous in self.initial_parameters.items()}


def check_learning_rates(steps, optimizer_config, scheduler_config):
    """Validate used LR, not an LR sampled before a scheduler/optimizer event."""
    multipliers = [group["lr_multiplier"] for group in optimizer_config.get("groups", [])]
    # Default group exists unless prefix groups cover every parameter. Infer it
    # from the actual group count; construction already verifies exact coverage.
    count = len(steps[0]["learning_rates"]) if steps else 0
    if count > len(multipliers):
        multipliers.append(1.0)
    if not steps or len(multipliers) != count:
        raise AssertionError("No optimizer updates or unexpected parameter groups")
    for step in steps:
        update = step["global_step_before"] if scheduler_config.get("interval", "step") == "step" else step["epoch"]
        factor = 1.0
        if scheduler_config["name"] == "PolynomialLR":
            factor = max(0., 1. - update / scheduler_config["total_iters"]) ** scheduler_config["power"]
        for actual, multiplier in zip(step["learning_rates"], multipliers):
            expected = optimizer_config["learning_rate"] * multiplier * factor
            if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
                raise AssertionError(f"LR update {update}: {actual} != expected {expected}")
    return {"ok": True, "used_learning_rates": steps}


def tensor_errors(actual, expected, atol, rtol):
    if actual.shape != expected.shape:
        raise AssertionError("Numerical comparison shapes differ")
    delta = (actual.to(torch.float64) - expected.to(torch.float64)).abs()
    relative = delta / expected.to(torch.float64).abs().clamp_min(1e-12)
    return {"max_absolute": delta.max().item(), "max_relative": relative.max().item(),
            "ok": torch.allclose(actual, expected, atol=atol, rtol=rtol)}


def mapping_errors(actual, expected, atol, rtol):
    if set(actual) != set(expected):
        raise AssertionError("Numerical comparison parameter/gradient names differ")
    per_parameter = {}
    for name in expected:
        left, right = actual[name], expected[name]
        if left is None or right is None:
            # Full-model gradients must participate, not merely agree as None.
            per_parameter[name] = {"ok": False, "max_absolute": None, "max_relative": None}
        else:
            per_parameter[name] = tensor_errors(left, right, atol, rtol)
    return {"ok": all(item["ok"] for item in per_parameter.values()),
            "max_absolute": max((item["max_absolute"] or 0.) for item in per_parameter.values()),
            "max_relative": max((item["max_relative"] or 0.) for item in per_parameter.values()),
            "failed_parameters": [name for name, item in per_parameter.items() if not item["ok"]],
            "per_parameter": per_parameter}


class SingleStepCapture(pl.Callback):
    def __init__(self, seed):
        self.seed = seed

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        torch.manual_seed(self.seed)

    def on_before_optimizer_step(self, trainer, module, optimizer):
        if optimizer.state:
            raise AssertionError("Single-step parity requires a fresh AdamW state")
        self.gradients = {name: cpu_tree(parameter.grad) for name, parameter in module.model.named_parameters()
                          if parameter.requires_grad}

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.loss = outputs["loss"].detach().cpu().clone()


def compare_single_step(model, batch, optimizer_config, *, num_classes, ignore_index,
                        device="cuda", seed=42, atol=1e-6, rtol=1e-5, loss_config=None):
    """One requested native reference update versus Lightning automatic update."""
    # Establish identical backend flags BEFORE either path executes a forward.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = model.to(device).float().train()
    model.zero_grad(set_to_none=True)
    batch = {key: value.to(device) for key, value in batch.items() if key in ("image", "mask")}
    initial = cpu_tree(model.state_dict())
    objective_config = deepcopy(loss_config or dict(name="CrossEntropyLoss", ignore_index=ignore_index))
    if objective_config.get("ignore_index") != ignore_index:
        raise ValueError("Numerical comparison loss must use the same ignore_index")
    criterion = build_loss(objective_config, num_classes)
    optimizer = build_optimizer(model, optimizer_config)
    if optimizer.state:
        raise AssertionError("Native optimizer is not fresh")
    # Use math SDPA only for numerical comparison to avoid backend-dependent
    # attention kernel choices. Both paths execute the same official model.
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        torch.manual_seed(seed)
        native_logits = model(batch["image"])
        native_loss = criterion(native_logits, batch["mask"])
        native_loss.backward()
        native_gradients = {name: cpu_tree(parameter.grad) for name, parameter in model.named_parameters()
                            if parameter.requires_grad}
        logits_reference, loss_reference = cpu_tree(native_logits), cpu_tree(native_loss)
        optimizer.step()
        parameters_reference = {name: cpu_tree(parameter) for name, parameter in model.named_parameters()
                                if parameter.requires_grad}
        del native_logits, native_loss, optimizer
        model.load_state_dict(initial, strict=True)
        model.zero_grad(set_to_none=True)
        capture = SingleStepCapture(seed)

        def capture_logits(module, inputs, output):
            capture.logits = cpu_tree(output)

        handle = model.register_forward_hook(capture_logits)
        module = SegmentationLightningModule(model, num_classes, ignore_index,
                                              optimizer=optimizer_config, scheduler={"name": "none"},
                                              loss=objective_config)
        trainer = pl.Trainer(accelerator="gpu" if torch.device(device).type == "cuda" else "cpu",
                             devices=1, precision="32-true", max_steps=1, max_epochs=1,
                             accumulate_grad_batches=1, gradient_clip_val=0., deterministic=True,
                             benchmark=False, logger=False, enable_checkpointing=False,
                             enable_progress_bar=False, enable_model_summary=False,
                             num_sanity_val_steps=0, limit_val_batches=0, callbacks=[capture])
        loader = DataLoader([cpu_tree(batch)], batch_size=None,
                            generator=torch.Generator().manual_seed(seed))
        try:
            trainer.fit(module, train_dataloaders=loader)
        finally:
            handle.remove()
        parameters_actual = {name: cpu_tree(parameter) for name, parameter in model.named_parameters()
                             if parameter.requires_grad}
    result = {"atol": atol, "rtol": rtol, "loss_name": objective_config["name"],
              "attention_backend": "SDPBackend.MATH (both paths)",
              "logits": tensor_errors(capture.logits, logits_reference, atol, rtol),
              "loss": tensor_errors(capture.loss, loss_reference, atol, rtol),
              "gradients": mapping_errors(capture.gradients, native_gradients, atol, rtol),
              "updated_parameters": mapping_errors(parameters_actual, parameters_reference, atol, rtol)}
    result["ok"] = all(result[key]["ok"] for key in ("logits", "loss", "gradients", "updated_parameters"))
    return result
