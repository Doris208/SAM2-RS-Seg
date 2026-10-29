"""Tiny synthetic data and deployment-integrity checks; no GPU or public data."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from datasets.label_contract import (
    IGNORE_INDEX, ISPRS_PALETTE, isprs_rgb_to_id,
    loveda2021_raw_to_train_id, validate_ids,
)
from scripts.deployment_manifest import digest, resolve_link, safe_path, verify

try:
    from PIL import Image
except ImportError:
    Image = None


class LabelContractTests(unittest.TestCase):
    def test_isprs_all_classes_and_unknown_color(self):
        self.assertEqual([isprs_rgb_to_id(c) for c in ISPRS_PALETTE], list(range(6)))
        with self.assertRaises(ValueError):
            isprs_rgb_to_id((0, 0, 0))
        with self.assertRaises(ValueError):
            isprs_rgb_to_id((255, 204, 0))  # GeoSeg display yellow is not official GT yellow

    def test_loveda_raw_contract(self):
        self.assertEqual([loveda2021_raw_to_train_id(i) for i in range(8)],
                         [IGNORE_INDEX, 0, 1, 2, 3, 4, 5, 6])
        for bad in (-1, 8, 1.5, True):
            with self.assertRaises(ValueError):
                loveda2021_raw_to_train_id(bad)

    def test_class_zero_is_valid_and_ignore_is_distinct(self):
        validate_ids([0, 1, 2, 3, 4, 5, 255], 6)
        for bad in (-1, 6, 254, 0.5, True):
            with self.assertRaises(ValueError):
                validate_ids([bad], 6)

    @unittest.skipIf(Image is None, "Pillow unavailable; synthetic PNG test skipped")
    def test_single_channel_png_and_nearest_resize(self):
        original = Image.new("L", (3, 2))
        original.putdata([0, 1, 2, 3, 5, 255])
        resized = original.resize((9, 6), resample=Image.Resampling.NEAREST)
        self.assertEqual(set(resized.getdata()), {0, 1, 2, 3, 5, 255})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            resized.save(path)
            with Image.open(path) as loaded:
                self.assertEqual(loaded.mode, "L")
                validate_ids(loaded.getdata(), 6)
                self.assertEqual(list(loaded.getdata()), list(resized.getdata()))


class DeploymentIntegrityTests(unittest.TestCase):
    def test_windows_git_link_placeholder_is_materialized_safely(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "configs" / "model.yaml"
            target.parent.mkdir()
            target.write_text("model: example\n", encoding="utf-8")
            placeholder = root / "model.yaml"
            placeholder.write_text("configs/model.yaml", encoding="utf-8")
            resolved = resolve_link(placeholder, "configs/model.yaml", root)
            self.assertEqual(resolved.read_bytes(), target.read_bytes())
            with self.assertRaises(ValueError):
                resolve_link(placeholder, "../outside.yaml", root)

    def test_hash_verification_detects_tampering_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.py"
            source.write_bytes(b"print('example')\n")
            manifest = {"schema_version": 1, "main": {"commit": "synthetic"},
                        "files": {"sample.py": {"size": source.stat().st_size,
                                               "sha256": digest(source)}}}
            path = root / "source_manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                verify(root, path)
            source.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                verify(root, path)
            source.unlink()
            with self.assertRaisesRegex(RuntimeError, "missing"):
                verify(root, path)

    def test_manifest_cannot_escape_release_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for relative in ("../secret", "/absolute", "nested/../../secret",
                             "nested\\..\\secret"):
                with self.assertRaises(ValueError):
                    safe_path(Path(directory), relative)


if __name__ == "__main__":
    unittest.main()
