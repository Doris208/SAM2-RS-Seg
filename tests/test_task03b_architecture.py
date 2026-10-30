"""Five-layer audit regressions; tensor/training checks require cloud CUDA."""

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HAS_YAML = importlib.util.find_spec("yaml") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_YAML:
    from projects.sam2_rs_seg.utils.config import load_config, resume_signature, validate_config
    from projects.sam2_rs_seg.losses.config import validate_loss_config
if HAS_TORCH:
    import torch
    from projects.sam2_rs_seg.losses.segmentation import CrossEntropyLoss, DiceLoss, build_loss
CUDA = HAS_TORCH and torch.cuda.is_available()
HAS_LIGHTNING = (CUDA and HAS_YAML and importlib.util.find_spec("lightning") is not None
                 and importlib.util.find_spec("torchmetrics") is not None)
if HAS_LIGHTNING:
    from torch import nn
    from projects.sam2_rs_seg.engine.experiment import build_experiment, fit_experiment
    from projects.sam2_rs_seg.engine.lightning_module import SegmentationLightningModule
    from projects.sam2_rs_seg.models.registry import MODELS
    from scripts.task03b_checks import TrainingAudit, check_learning_rates, compare_single_step

    class AuditNetwork(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            self.conv = nn.Conv2d(3, num_classes, 1)

        def forward(self, image):
            return self.conv(image)


@unittest.skipUnless(HAS_YAML, "PyYAML unavailable; configuration cases NOT executed")
class ArchitectureConfigTests(unittest.TestCase):
    def config(self):
        return load_config(ROOT / "configs/smoke/task03b.yaml")

    def test_ce_default_and_dice_switch(self):
        config = self.config()
        self.assertEqual(config["loss"], {"name": "CrossEntropyLoss", "ignore_index": 255})
        config["loss"] = {"name": "DiceLoss", "ignore_index": 255, "absent_class": "exclude"}
        validate_config(config)

    def test_invalid_loss_options_and_ignore(self):
        for config in ({"name": "Unknown", "ignore_index": 255},
                       {"name": "DiceLoss", "ignore_index": 0},
                       {"name": "DiceLoss", "ignore_index": 255, "eps": 0},
                       {"name": "DiceLoss", "ignore_index": 255, "smooth": -1},
                       {"name": "DiceLoss", "ignore_index": 255, "absent_class": "implicit"},
                       {"name": "CrossEntropyLoss", "ignore_index": 255, "unknown": True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_loss_config(config, 255)

    def test_epoch_and_step_limits(self):
        for steps, epochs in ((4, -1), (-1, 2), (4, 2)):
            config = self.config()
            config["trainer"].update(max_steps=steps, max_epochs=epochs)
            validate_config(config)
        for steps, epochs in ((-1, -1), (0, 2), (4.5, -1), (True, -1)):
            config = self.config()
            config["trainer"].update(max_steps=steps, max_epochs=epochs)
            with self.subTest(steps=steps, epochs=epochs), self.assertRaises(ValueError):
                validate_config(config)

    def test_scheduler_interval_and_old_checkpoint_signature(self):
        original = self.config()
        original["scheduler"].pop("interval", None)
        explicit_step = deepcopy(original)
        explicit_step["scheduler"]["interval"] = "step"
        self.assertEqual(resume_signature(original), resume_signature(explicit_step))
        no_scheduler = deepcopy(original)
        no_scheduler["scheduler"] = {"name": "none"}
        explicit_none = deepcopy(no_scheduler)
        explicit_none["scheduler"]["interval"] = "epoch"
        self.assertEqual(resume_signature(no_scheduler), resume_signature(explicit_none))
        epoch = deepcopy(original)
        epoch["scheduler"]["interval"] = "epoch"
        validate_config(epoch)
        self.assertNotEqual(resume_signature(original), resume_signature(epoch))
        epoch["scheduler"]["interval"] = "batch"
        with self.assertRaises(ValueError):
            validate_config(epoch)

    def test_pure_components_do_not_import_trainer(self):
        paths = ["projects/sam2_rs_seg/losses/segmentation.py", "projects/sam2_rs_seg/utils/labels.py",
                 "projects/sam2_rs_seg/models/segmentor.py", "projects/sam2_rs_seg/models/registry.py",
                 "projects/sam2_rs_seg/datasets/synthetic_dataset.py",
                 "projects/sam2_rs_seg/evaluation/segmentation_metrics.py"]
        for path in paths:
            tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
            imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
            self.assertFalse(any("lightning" in name or "engine" in name for name in imports), path)
            if "/losses/" in path:
                self.assertFalse(any("torchmetrics" in name for name in imports), path)


@unittest.skipUnless(CUDA, "Loss tensor/gradient cases require cloud CUDA; no local CPU substitute")
class StandaloneLossCUDATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_ce_matches_torch_value_gradient_and_eval_terms(self):
        logits = torch.randn(2, 3, 2, 3, device="cuda", requires_grad=True)
        target = torch.tensor([[[0, 1, 255], [2, 1, 0]], [[1, 0, 2], [255, 0, 1]]], device="cuda")
        criterion = CrossEntropyLoss(3)
        actual = criterion(logits, target)
        expected = torch.nn.functional.cross_entropy(logits, target, ignore_index=255)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        reference_grad = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(grad, reference_grad, rtol=0, atol=0)
        total, count = criterion.evaluation_terms(logits, target)
        torch.testing.assert_close(total, torch.nn.functional.cross_entropy(logits, target,
                                                                           ignore_index=255, reduction="sum"))
        self.assertEqual(count.item(), 10)

    def test_dice_hand_calculated_multiclass_value(self):
        probabilities = torch.tensor([[[[.8, .2]], [[.1, .7]], [[.1, .1]]]], device="cuda")
        logits = probabilities.log().requires_grad_()
        target = torch.tensor([[[0, 1]]], device="cuda")
        loss = DiceLoss(3)(logits, target)
        torch.testing.assert_close(loss, torch.tensor(19/90, device="cuda"))
        included = DiceLoss(3, absent_class="include")(logits, target)
        torch.testing.assert_close(included, torch.tensor(64/135, device="cuda"))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(torch.count_nonzero(logits.grad).item(), 0)

    def test_ignore_pixels_do_not_change_loss_or_receive_gradient(self):
        target = torch.tensor([[[0, 1, 255]]], device="cuda")
        for name in ("CrossEntropyLoss", "DiceLoss"):
            logits = torch.randn(1, 3, 1, 3, device="cuda", requires_grad=True)
            changed = logits.detach().clone()
            changed[:, :, 0, 2] = torch.tensor([-40., 40., 20.], device="cuda")
            changed.requires_grad_()
            criterion = build_loss({"name": name, "ignore_index": 255}, 3)
            loss = criterion(logits, target)
            torch.testing.assert_close(loss, criterion(changed, target), rtol=0, atol=0)
            loss.backward()
            self.assertEqual(torch.count_nonzero(logits.grad[:, :, 0, 2]).item(), 0)
            self.assertGreater(torch.count_nonzero(logits.grad[:, :, 0, 0]).item(), 0)
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_all_ignore_forward_rejects_and_evaluation_returns_zero(self):
        target = torch.full((1, 2, 2), 255, device="cuda", dtype=torch.long)
        for criterion in (CrossEntropyLoss(3), DiceLoss(3)):
            logits = torch.randn(1, 3, 2, 2, device="cuda", requires_grad=True)
            with self.assertRaisesRegex(ValueError, "ignored"):
                criterion(logits, target)
            total, count = criterion.evaluation_terms(logits, target)
            self.assertEqual(count.item(), 0)
            self.assertEqual(total.item(), 0)
            total.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)

    def test_dice_per_image_evaluation_is_partition_invariant(self):
        logits = torch.randn(3, 3, 1, 2, device="cuda", requires_grad=True)
        target = torch.tensor([[[0, 1]], [[2, 2]], [[255, 255]]], device="cuda")
        criterion = DiceLoss(3)
        numerator, count = criterion.evaluation_terms(logits, target)
        pieces = [criterion.evaluation_terms(logits[i:i+1], target[i:i+1]) for i in range(3)]
        torch.testing.assert_close(numerator, sum(item[0] for item in pieces))
        self.assertEqual(count.item(), 2)
        self.assertEqual(sum(item[1] for item in pieces).item(), 2)
        torch.testing.assert_close(criterion(logits, target), numerator / count)

    def test_class_zero_is_valid_and_invalid_targets_rejected(self):
        logits = torch.randn(1, 3, 1, 1, device="cuda", requires_grad=True)
        for criterion in (CrossEntropyLoss(3), DiceLoss(3)):
            target = torch.zeros(1, 1, 1, device="cuda", dtype=torch.long)
            gradient = torch.autograd.grad(criterion(logits, target), logits)[0]
            self.assertGreater(torch.count_nonzero(gradient).item(), 0)
            for value in (-1, 3, 254):
                with self.assertRaises(ValueError):
                    criterion(logits, torch.full_like(target, value))
            with self.assertRaises(ValueError):
                criterion(logits, target.float())
            with self.assertRaises(ValueError):
                criterion(logits[:, :2], target)


@unittest.skipUnless(HAS_LIGHTNING, "CUDA + Lightning integration not executed")
class LossTrainingCUDATests(unittest.TestCase):
    def test_legacy_torch_ce_state_dict_remains_compatible(self):
        legacy = SegmentationLightningModule(AuditNetwork(), 3).cuda()
        legacy.criterion = nn.CrossEntropyLoss(ignore_index=255)
        current = SegmentationLightningModule(AuditNetwork(), 3).cuda()
        result = current.load_state_dict(legacy.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

    def test_loss_switch_with_an_ordinary_non_sam_model(self):
        batch = {"image": torch.randn(1, 3, 8, 8, device="cuda"),
                 "mask": torch.randint(3, (1, 8, 8), device="cuda")}
        for name, expected_type in (("CrossEntropyLoss", CrossEntropyLoss), ("DiceLoss", DiceLoss)):
            config = {"name": name, "ignore_index": 255}
            module = SegmentationLightningModule(AuditNetwork(), 3, loss=config).cuda()
            self.assertIsInstance(module.criterion, expected_type)
            _, loss, _, _ = module.batch_loss(batch, training=True)
            loss.backward()
            self.assertTrue(torch.isfinite(module.model.conv.weight.grad).all())
            self.assertGreater(torch.count_nonzero(module.model.conv.weight.grad).item(), 0)

    def test_epoch_scheduler_dice_training_and_full_resume(self):
        config = load_config(ROOT / "configs/smoke/task03b.yaml")
        config["model"] = {"name": "AuditNetwork", "params": {}}
        config["data"].update(num_classes=3, image_size=[8, 8], train_samples=4, pin_memory=False)
        config["loss"] = {"name": "DiceLoss", "ignore_index": 255}
        config["trainer"].update(max_steps=-1, max_epochs=1, accumulate_grad_batches=2)
        config["scheduler"] = {"name": "PolynomialLR", "total_iters": 2, "power": 1., "interval": "epoch"}
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(MODELS, {"AuditNetwork": (AuditNetwork, 3, 1)}):
            first = TrainingAudit(probes=["conv.weight"])
            trainer, module, data, checkpoint = build_experiment(config, directory, callbacks=[first])
            last = fit_experiment(trainer, module, data, directory)
            self.assertEqual(trainer.global_step, 2)
            self.assertEqual(trainer.lr_scheduler_configs[0].interval, "epoch")
            self.assertTrue(Path(checkpoint.best_model_path).is_file())
            self.assertGreater(first.parameter_updates(module.model)["conv.weight"], 0.)
            saved = torch.load(last, map_location="cpu", weights_only=False)
            self.assertEqual(saved["lr_schedulers"][0]["last_epoch"], 1)
            config["trainer"]["max_epochs"] = 2
            audit = TrainingAudit(checkpoint=saved)
            resumed, module, data, _ = build_experiment(config, directory, resume=last, callbacks=[audit])
            final = fit_experiment(resumed, module, data, directory, resume=last)
            self.assertEqual(resumed.global_step, 4)
            self.assertTrue(audit.restored["optimizer_exact"])
            self.assertTrue(audit.restored["scheduler_exact"])
            self.assertEqual(torch.load(final, map_location="cpu", weights_only=False)["lr_schedulers"][0]["last_epoch"], 2)
            self.assertTrue(check_learning_rates(first.steps + audit.steps, config["optimizer"], config["scheduler"])["ok"])

    def test_dice_native_and_lightning_single_step(self):
        batch = {"image": torch.randn(1, 3, 8, 8), "mask": torch.randint(3, (1, 8, 8))}
        result = compare_single_step(AuditNetwork(), batch,
                                     {"name": "AdamW", "learning_rate": 0.001, "foreach": False},
                                     num_classes=3, ignore_index=255, device="cuda",
                                     loss_config={"name": "DiceLoss", "ignore_index": 255})
        self.assertEqual(result["loss_name"], "DiceLoss")
        self.assertTrue(result["ok"], result)


if __name__ == "__main__":
    unittest.main()
