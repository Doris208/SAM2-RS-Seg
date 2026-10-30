"""03B numerical unit cases and small CUDA integration cases, run in the cloud."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ("torch", "lightning", "torchmetrics", "yaml")
MISSING = [name for name in DEPENDENCIES if importlib.util.find_spec(name) is None]
if not MISSING:
    import lightning.pytorch as pl
    import torch
    from torch import nn
    from projects.sam2_rs_seg.datasets.synthetic_dataset import SyntheticSegmentationDataset
    from projects.sam2_rs_seg.datasets.datamodule import SegmentationDataModule
    from projects.sam2_rs_seg.engine.experiment import build_experiment, fit_experiment
    from projects.sam2_rs_seg.engine.lightning_module import SegmentationLightningModule
    from projects.sam2_rs_seg.engine.optim import build_optimizer, build_scheduler
    from projects.sam2_rs_seg.evaluation.segmentation_metrics import (
        SegmentationMetrics, metrics_from_confusion_matrix,
    )
    from projects.sam2_rs_seg.models.registry import MODELS, HEADS, build_model
    from projects.sam2_rs_seg.utils.config import load_config, validate_config
    from scripts.task03b_checks import TrainingAudit, check_learning_rates, compare_single_step

    class TinySegmentor(nn.Module):
        def __init__(self, num_classes=6):
            super().__init__()
            self.conv = nn.Conv2d(3, num_classes, 1)
            self.dropout = nn.Dropout2d(0.1)

        def forward(self, image):
            return self.conv(self.dropout(image))


@unittest.skipIf(bool(MISSING), "Missing dependencies; NOT executed: " + ", ".join(MISSING))
class Task03BUnitTests(unittest.TestCase):
    def config(self):
        return load_config(ROOT / "configs/smoke/task03b.yaml")

    def test_yaml_and_protocol_mismatches(self):
        config = self.config()
        for section, key, value in (("loss", "ignore_index", 0), ("data", "num_classes", 0),
                                    ("trainer", "devices", 2), ("evaluation", "absent_class", "implicit")):
            bad = deepcopy(config)
            bad[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(bad)

    def test_model_name_channels_sizes_classes_and_kwargs(self):
        config = self.config()
        bad = deepcopy(config["model"])
        bad["name"] = "NotRegistered"
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            build_model(bad, config["data"])
        for changes in ({"input_channels": 7}, {"image_size": [250, 256]}, {"num_classes": 2}):
            data = {**config["data"], **changes}
            with self.assertRaises(ValueError):
                build_model(config["model"], data)
        bad = deepcopy(config["model"])
        bad["params"]["unknown_option"] = True
        with self.assertRaisesRegex(ValueError, "Invalid model parameters"):
            build_model(bad, config["data"])

    def test_head_replacement_without_editing_segmentor(self):
        class FakeSAM(nn.Module):
            def __init__(self, num_classes=6):
                super().__init__()
                self.encoder = nn.Conv2d(3, 256, 1)
                self.encoder.out_channels = 256
                self.decode_head = nn.Identity()

            def forward(self, image):
                feature = self.encoder(image)
                return self.decode_head([feature, feature[:, :, ::2, ::2], feature[:, :, ::4, ::4]],
                                        image.shape[-2:])

        class DummyHead(nn.Module):
            def __init__(self, in_channels, head_channels, num_classes):
                super().__init__()
                self.classifier = nn.Conv2d(in_channels, num_classes, 1)

            def forward(self, features, output_size):
                return torch.nn.functional.interpolate(self.classifier(features[0]), size=output_size,
                                                       mode="bilinear", align_corners=False)

        config = self.config()
        model = {"name": "SAM2SemanticSegmentor", "params": {"num_classes": 6},
                 "decode_head": {"name": "TestDummyHead", "params": {}}}
        with mock.patch.dict(MODELS, {"SAM2SemanticSegmentor": (FakeSAM, 3, 32)}), \
                mock.patch.dict(HEADS, {"TestDummyHead": DummyHead}):
            result = build_model(model, config["data"])
            self.assertIsInstance(result.decode_head, DummyHead)
            self.assertEqual(tuple(result(torch.randn(1, 3, 32, 32)).shape), (1, 6, 32, 32))
            model["decode_head"]["params"]["in_channels"] = 128
            with self.assertRaisesRegex(ValueError, "256 channels"):
                build_model(model, config["data"])

    def test_synthetic_reproducibility_and_split_separation(self):
        args = dict(num_samples=3, image_size=(8, 8), seed=42)
        first = SyntheticSegmentationDataset(**args)
        second = SyntheticSegmentationDataset(**args)
        self.assertTrue(torch.equal(first[1]["image"], second[1]["image"]))
        self.assertTrue(torch.equal(first[1]["mask"], second[1]["mask"]))
        self.assertFalse(torch.equal(first[0]["image"], SyntheticSegmentationDataset(**args, split="val")[0]["image"]))
        self.assertEqual(first[0]["mask"].dtype, torch.long)
        self.assertEqual(first[0]["image"].dtype, torch.float32)
        self.assertEqual(first[0]["mask"].view(-1)[0].item(), 0)

    def test_datamodule_setup_idempotence_and_batch_protocol(self):
        data = SegmentationDataModule(image_size=[8, 8], input_channels=3, num_classes=2,
                                      pin_memory=False, seed=42)
        data.setup("fit")
        original = data.datasets["train"]
        data.setup("fit")
        self.assertIs(original, data.datasets["train"])
        batch = next(iter(data.train_dataloader()))
        self.assertEqual(tuple(batch["image"].shape), (1, 3, 8, 8))
        self.assertEqual(tuple(batch["mask"].shape), (1, 8, 8))
        data.setup("test")
        self.assertIsNot(data.datasets["val"], data.datasets["test"])

    def test_perfect_metrics_and_absent_class_policies(self):
        matrix = torch.diag(torch.tensor([2, 3, 0]))
        result = metrics_from_confusion_matrix(matrix)
        self.assertTrue(torch.isnan(result["IoU"][2]))
        for key in ("mIoU", "mF1", "OA"):
            self.assertEqual(result[key].item(), 1.)
        zero = metrics_from_confusion_matrix(matrix, "zero")
        self.assertAlmostEqual(zero["mIoU"].item(), 2/3)

    def test_partial_errors_class_zero_ignore_and_matrix_direction(self):
        metric = SegmentationMetrics(3)
        target = torch.tensor([[[0, 0, 1, 1, 255]]])
        prediction = torch.tensor([[[0, 1, 1, 1, 2]]])
        metric.update(prediction, target)
        result = metric.compute()
        self.assertTrue(torch.equal(result["confusion_matrix"], torch.tensor([[1, 1, 0], [0, 2, 0], [0, 0, 0]])))
        self.assertAlmostEqual(result["mIoU"].item(), 7/12)
        self.assertAlmostEqual(result["mF1"].item(), 11/15)
        self.assertEqual(result["OA"].item(), 3/4)
        self.assertEqual(result["valid_pixels"].item(), 4)

    def test_predicted_only_class_is_not_excluded(self):
        metric = SegmentationMetrics(3)
        metric.update(torch.tensor([[[1]]]), torch.tensor([[[0]]]))
        result = metric.compute()
        self.assertEqual(result["IoU"][:2].tolist(), [0., 0.])
        self.assertEqual(result["mIoU"].item(), 0.)

    def test_multi_batch_matches_whole_not_batch_metric_average(self):
        target = torch.tensor([[[0, 0, 1, 1]]])
        prediction = torch.tensor([[[0, 1, 1, 1]]])
        whole, split = SegmentationMetrics(3), SegmentationMetrics(3)
        whole.update(prediction, target)
        split.update(prediction[..., :1], target[..., :1])
        split.update(prediction[..., 1:], target[..., 1:])
        self.assertTrue(torch.equal(whole.compute()["confusion_matrix"], split.compute()["confusion_matrix"]))
        self.assertNotAlmostEqual(split.compute()["mIoU"].item(), (1. + 1/3) / 2)

    def test_metric_states_and_reset_are_independent(self):
        module = SegmentationLightningModule(TinySegmentor(), 6)
        module.metrics["train_phase"].update(torch.tensor([[[0]]]), torch.tensor([[[0]]]))
        module.metrics["val_phase"].update(torch.tensor([[[1]]]), torch.tensor([[[0]]]))
        module.metrics["test_phase"].update(torch.tensor([[[2]]]), torch.tensor([[[2]]]))
        module.on_validation_epoch_start()
        self.assertEqual(module.metrics["val_phase"].confusion.confmat.sum().item(), 0)
        self.assertEqual(module.metrics["train_phase"].confusion.confmat.sum().item(), 1)
        self.assertEqual(module.metrics["test_phase"].confusion.confmat.sum().item(), 1)

    def test_invalid_prediction_is_not_silently_filtered(self):
        with self.assertRaises(ValueError):
            SegmentationMetrics(2).update(torch.tensor([[[2]]]), torch.tensor([[[0]]]))

    def test_batch_loss_ignore_zero_and_invalid_targets(self):
        module = SegmentationLightningModule(TinySegmentor(2), 2)
        batch = {"image": torch.randn(1, 3, 8, 8), "mask": torch.zeros(1, 8, 8, dtype=torch.long)}
        batch["mask"][:, 0] = 255
        logits, loss, _, _ = module.batch_loss(batch, training=True)
        expected = nn.CrossEntropyLoss(ignore_index=255)(logits, batch["mask"])
        torch.testing.assert_close(loss, expected)
        for value in (255, -1, 2):
            bad = {"image": batch["image"], "mask": torch.full_like(batch["mask"], value)}
            with self.assertRaises(ValueError):
                module.batch_loss(bad, training=True)
        batch["mask"] = batch["mask"].float()
        with self.assertRaises(ValueError):
            module.batch_loss(batch, training=True)

    def test_wrong_logits_channels_are_rejected(self):
        module = SegmentationLightningModule(TinySegmentor(3), 2)
        with self.assertRaisesRegex(ValueError, "one logits Tensor"):
            module.batch_loss({"image": torch.randn(1, 3, 8, 8),
                               "mask": torch.zeros(1, 8, 8, dtype=torch.long)}, training=True)

    def test_whole_evaluation_with_no_valid_gt_is_rejected(self):
        module = SegmentationLightningModule(TinySegmentor(2), 2)
        module.on_validation_epoch_start()
        module.validation_step({"image": torch.randn(1, 3, 8, 8),
                                "mask": torch.full((1, 8, 8), 255, dtype=torch.long)}, 0)
        self.assertEqual(module._val_loss_sum.item(), 0.)
        with self.assertRaisesRegex(ValueError, "no valid GT"):
            module.on_validation_epoch_end()

    def test_optimizer_coverage_groups_and_duplicate_rejection(self):
        model = TinySegmentor()
        config = dict(name="AdamW", learning_rate=0.001, weight_decay=0.01,
                      groups=[dict(prefix="conv", lr_multiplier=0.1)])
        optimizer = build_optimizer(model, config)
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(set(actual), {id(p) for p in model.parameters()})
        self.assertEqual(len(actual), len(set(actual)))
        config["groups"].append(dict(prefix="conv.weight", lr_multiplier=1.))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            build_optimizer(model, config)

    def test_polynomial_lr_sequence_and_no_scheduler(self):
        model = TinySegmentor()
        optimizer = build_optimizer(model, dict(name="AdamW", learning_rate=0.01))
        scheduler = build_scheduler(optimizer, dict(name="PolynomialLR", total_iters=4, power=1.))
        used = []
        for _ in range(4):
            used.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        torch.testing.assert_close(torch.tensor(used), torch.tensor([0.01, 0.0075, 0.005, 0.0025]))
        self.assertEqual(scheduler.last_epoch, 4)
        self.assertIsNone(build_scheduler(optimizer, {"name": "none"}))


CUDA_AVAILABLE = not MISSING and torch.cuda.is_available()


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA integration not executed; no CPU substitute")
class Task03BCUDAIntegrationTests(unittest.TestCase):
    def test_accumulation_scheduler_full_resume_and_repeated_validation(self):
        config = load_config(ROOT / "configs/smoke/task03b.yaml")
        config["model"] = {"name": "TestTiny", "params": {}}
        config["data"].update(image_size=[8, 8], train_samples=4, pin_memory=False)
        config["trainer"].update(max_steps=2, accumulate_grad_batches=2)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(MODELS, {"TestTiny": (TinySegmentor, 3, 1)}):
            audit = TrainingAudit(probes=["conv.weight"])
            trainer, module, data, best = build_experiment(config, directory, callbacks=[audit])
            last = fit_experiment(trainer, module, data, directory)
            self.assertEqual(trainer.global_step, 2)
            self.assertEqual(len(audit.steps), 2)
            self.assertTrue(Path(best.best_model_path).is_file())
            self.assertGreater(audit.parameter_updates(module.model)["conv.weight"], 0)
            saved = torch.load(last, map_location="cpu", weights_only=False)
            config["trainer"]["max_steps"] = 4
            restore = TrainingAudit(checkpoint=saved)
            resumed, module, data, _ = build_experiment(config, directory, resume=last, callbacks=[restore])
            fit_experiment(resumed, module, data, directory, resume=last)
            self.assertEqual(resumed.global_step, 4)
            self.assertTrue(restore.restored["optimizer_exact"])
            self.assertTrue(restore.restored["scheduler_exact"])
            self.assertTrue(check_learning_rates(audit.steps + restore.steps, config["optimizer"], config["scheduler"])["ok"])
            resumed.validate(module, datamodule=data)
            first = module.epoch_results["val"]["confusion_matrix"].clone()
            resumed.validate(module, datamodule=data)
            self.assertTrue(torch.equal(first, module.epoch_results["val"]["confusion_matrix"]))
            resumed.test(module, datamodule=data)
            self.assertGreater(module.epoch_results["test"]["valid_pixels"].item(), 0)

    def test_native_lightning_single_step_with_random_layer(self):
        torch.manual_seed(42)
        model = TinySegmentor(2)
        batch = {"image": torch.randn(1, 3, 8, 8), "mask": torch.randint(2, (1, 8, 8))}
        result = compare_single_step(model, batch, dict(name="AdamW", learning_rate=0.001,
                                                        weight_decay=0.01, foreach=False),
                                     num_classes=2, ignore_index=255, device="cuda")
        self.assertTrue(result["ok"], result)


if __name__ == "__main__":
    unittest.main()
