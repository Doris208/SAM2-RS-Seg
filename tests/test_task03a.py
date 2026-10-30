"""Task 03A: CPU head checks and optional official random-initialized encoder."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None

if torch is not None:
    from torch import nn
    from torch.nn import functional as F

    from projects.sam2_rs_seg.models import SAM2SemanticSegmentor
    from projects.sam2_rs_seg.models.backbones import SAM2ImageEncoder
    from projects.sam2_rs_seg.models.decode_heads import SemanticSegmentationHead
    from scripts.task03a_checks import (
        gradient_report, multiscale_gradient_report, parameter_counts, synthetic_batch, validate_masks,
    )


@unittest.skipIf(torch is None, "PyTorch unavailable; CPU tensor checks NOT executed")
class SemanticHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)

    def test_six_class_head_shape_loss_and_backward(self):
        head = SemanticSegmentationHead()
        features = [torch.randn(1, 256, s, s, requires_grad=True) for s in (64, 32, 16)]
        images, masks = synthetic_batch(1, 256, 256, 6)
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(masks.dtype, torch.long)
        self.assertEqual(set(masks.unique().tolist()), set(range(6)) | {255})
        logits = head(features, images.shape[-2:])
        self.assertEqual(tuple(logits.shape), (1, 6, 256, 256))
        self.assertTrue(torch.isfinite(logits).all())
        loss = nn.CrossEntropyLoss(ignore_index=255)(logits, masks)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(gradient_report(head)["ok"])
        for feature in features:
            self.assertIsNotNone(feature.grad)
            self.assertTrue(torch.isfinite(feature.grad).all())
            self.assertGreater(torch.count_nonzero(feature.grad).item(), 0)
        self.assertEqual(parameter_counts(head), {"total": 885766, "trainable": 885766})

    def test_two_classes_and_rectangular_output(self):
        head = SemanticSegmentationHead(num_classes=2, head_channels=16)
        logits = head([torch.randn(1, 256, h, w) for h, w in ((8, 12), (4, 6), (2, 3))],
                      (31, 47))
        self.assertEqual(tuple(logits.shape), (1, 2, 31, 47))

    def test_raw_logits_and_exact_interpolation(self):
        head = SemanticSegmentationHead(in_channels=4, head_channels=8, num_classes=2).eval()
        # Force values outside probability ranges to detect accidental activations.
        with torch.no_grad():
            head.layers[-1].weight.zero_()
            head.layers[-1].bias.copy_(torch.tensor([-2., 3.]))
        features = [torch.randn(1, 4, s, s) for s in (8, 4, 2)]
        output = head(features, (19, 23))
        fused = torch.cat([features[0]] + [F.interpolate(x, (8, 8), mode="bilinear",
                                                       align_corners=False) for x in features[1:]], dim=1)
        expected = F.interpolate(head.layers(fused), (19, 23), mode="bilinear", align_corners=False)
        torch.testing.assert_close(output, expected)
        self.assertLess(output.min().item(), 0)
        self.assertGreater(output.max().item(), 1)

    def test_missing_scales_are_rejected(self):
        head = SemanticSegmentationHead()
        for features in (torch.randn(1, 256, 8, 8), [torch.randn(1, 256, 8, 8)]):
            with self.assertRaisesRegex(ValueError, "three backbone_fpn"):
                head(features, (32, 32))

    def test_ignore_pixels_excluded_and_class_zero_supervised(self):
        logits = torch.randn(1, 6, 2, 3, requires_grad=True)
        masks = torch.tensor([[[0, 1, 255], [2, 255, 5]]])
        validate_masks(masks, 6)
        valid = masks != 255
        loss = nn.CrossEntropyLoss(ignore_index=255)(logits, masks)
        expected = F.cross_entropy(logits.permute(0, 2, 3, 1)[valid], masks[valid])
        torch.testing.assert_close(loss, expected)
        changed = logits.detach().clone().permute(0, 2, 3, 1)
        changed[~valid] = torch.tensor([100., -100., 50., 10., -10., 1.])
        torch.testing.assert_close(F.cross_entropy(changed.permute(0, 3, 1, 2), masks,
                                                  ignore_index=255), loss)
        loss.backward()
        self.assertEqual(torch.count_nonzero(logits.grad.permute(0, 2, 3, 1)[~valid]).item(), 0)
        self.assertGreater(torch.count_nonzero(logits.grad[0, :, 0, 0]).item(), 0)

    def test_all_ignore_and_invalid_masks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "All pixels are ignored"):
            validate_masks(torch.full((1, 8, 8), 255, dtype=torch.long), 6)
        for value in (-1, 6, 254):
            with self.assertRaises(ValueError):
                validate_masks(torch.tensor([[[0, value]]]), 6)
        with self.assertRaises(ValueError):
            validate_masks(torch.zeros(1, 8, 8), 6)

    def test_gradient_diagnostics_distinguish_failure_modes(self):
        module = nn.ParameterDict({key: nn.Parameter(torch.ones(2)) for key in
                                   ("frozen", "none", "zero", "nonfinite", "nonzero")})
        module["frozen"].requires_grad_(False)
        module["zero"].grad = torch.zeros(2)
        module["nonfinite"].grad = torch.tensor([float("inf"), float("nan")])
        module["nonzero"].grad = torch.ones(2)
        report = gradient_report(module)
        self.assertEqual(report["counts"], dict.fromkeys(report["counts"], 1))
        self.assertFalse(report["ok"])
        module["nonfinite"].grad.zero_()
        self.assertTrue(gradient_report(module)["ok"])
        self.assertFalse(gradient_report(module, require_all=True)["ok"])
        module["nonzero"].grad.zero_()
        self.assertFalse(gradient_report(module)["ok"])

    def test_checkpoint_argument_errors_never_fall_back(self):
        with self.assertRaisesRegex(ValueError, "requires checkpoint_path"):
            SAM2ImageEncoder(pretrained=True)
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "missing.pt"
            with self.assertRaises(FileNotFoundError):
                SAM2ImageEncoder(pretrained=True, checkpoint_path=missing)
            with self.assertRaisesRegex(ValueError, "requires pretrained=True"):
                SAM2ImageEncoder(pretrained=False, checkpoint_path=missing)

    def test_segmentor_uses_three_scales_and_input_size(self):
        # Wiring check only; this fake encoder does NOT validate official SAM2.
        class FakeEncoder(nn.Module):
            out_channels = 256

            def forward(self, images):
                return {"backbone_fpn": [torch.randn(images.shape[0], 256, h, w)
                                         for h, w in ((8, 12), (4, 6), (2, 3))]}

        with mock.patch("projects.sam2_rs_seg.models.segmentor.SAM2ImageEncoder",
                        return_value=FakeEncoder()):
            model = SAM2SemanticSegmentor(num_classes=2, head_channels=8, pretrained=False)
        output = model(torch.randn(1, 3, 32, 48))
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(tuple(output.shape), (1, 2, 32, 48))


def official_dependency_skip_reason():
    if torch is None:
        return "PyTorch unavailable"
    if sys.version_info < (3, 10):
        return "Official SAM2 requires Python >=3.10; installed interpreter is older"
    for name in ("sam2", "hydra", "iopath", "torchvision"):
        if importlib.util.find_spec(name) is None:
            return f"Official SAM2 dependency unavailable: {name}"
    return ""


SAM2_SKIP_REASON = official_dependency_skip_reason()


@unittest.skipIf(bool(SAM2_SKIP_REASON), SAM2_SKIP_REASON)
class OfficialRandomModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_official_features_full_forward_backward_and_two_classes(self):
        torch.manual_seed(42)
        model = SAM2SemanticSegmentor(pretrained=False).train()
        encoder = model.encoder.image_encoder
        self.assertEqual(encoder.scalp, 1)
        self.assertEqual(encoder.neck.fpn_top_down_levels, [2, 3])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        shapes = {}

        def capture_trunk(module, inputs, output):
            shapes["trunk"] = [tuple(x.shape) for x in output]

        def capture_encoder(module, inputs, output):
            shapes["fpn"] = [tuple(x.shape) for x in output["backbone_fpn"]]
            self.assertEqual(set(output), {"vision_features", "vision_pos_enc", "backbone_fpn"})

        handles = [encoder.trunk.register_forward_hook(capture_trunk),
                   encoder.register_forward_hook(capture_encoder)]
        images, masks = synthetic_batch(1, 256, 256, 6)
        try:
            logits = model(images)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(shapes["trunk"], [(1, 96, 64, 64), (1, 192, 32, 32),
                                           (1, 384, 16, 16), (1, 768, 8, 8)])
        self.assertEqual(shapes["fpn"], [(1, 256, 64, 64), (1, 256, 32, 32), (1, 256, 16, 16)])
        self.assertEqual(tuple(logits.shape), (1, 6, 256, 256))
        self.assertTrue(torch.isfinite(logits).all())
        loss = nn.CrossEntropyLoss(ignore_index=255)(logits, masks)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        report = multiscale_gradient_report(encoder.trunk, encoder.neck, model.decode_head)
        self.assertTrue(report["ok"], report["failures"])
        for parameter in (encoder.trunk.blocks[9].attn.qkv.weight,
                          encoder.trunk.blocks[11].attn.qkv.weight,
                          *(projection.conv.weight for projection in encoder.neck.convs)):
            saved_gradient = parameter.grad
            try:
                for invalid_gradient in (None, torch.zeros_like(parameter)):
                    parameter.grad = invalid_gradient
                    report = multiscale_gradient_report(encoder.trunk, encoder.neck, model.decode_head)
                    self.assertFalse(report["ok"])
            finally:
                parameter.grad = saved_gradient
        del model, logits, loss
        binary = SAM2SemanticSegmentor(num_classes=2, head_channels=16, pretrained=False).eval()
        with torch.no_grad():
            self.assertEqual(tuple(binary(images).shape), (1, 2, 256, 256))

    def test_official_strict_loader_rejects_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "incomplete.pt"
            torch.save({"model": {}}, path)
            with self.assertRaisesRegex(RuntimeError, r"Error\(s\) in loading state_dict"):
                SAM2SemanticSegmentor(pretrained=True, checkpoint_path=path)

    def test_builder_load_failure_is_propagated(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.pt"
            path.write_bytes(b"invalid checkpoint")
            with mock.patch("sam2.build_sam.build_sam2", side_effect=RuntimeError("load failed")):
                with self.assertRaisesRegex(RuntimeError, "load failed"):
                    SAM2ImageEncoder(checkpoint_path=path)


if __name__ == "__main__":
    unittest.main()
