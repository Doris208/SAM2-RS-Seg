"""Migration safety tests; no deep-learning installation, GPU, or Git commits."""

import configparser
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import check_environment as environment
from scripts import deployment_manifest as deployment

ROOT = Path(__file__).resolve().parents[1]


class EnvironmentTests(unittest.TestCase):
    def test_inventory_handles_missing_packages_without_importing_frameworks(self):
        with mock.patch.object(environment.metadata, "version",
                               side_effect=environment.metadata.PackageNotFoundError):
            with mock.patch.object(environment.shutil, "which", return_value=None):
                with mock.patch.object(environment.importlib, "import_module",
                                       side_effect=AssertionError("Unexpected runtime import")):
                    result = environment.inventory()
        self.assertTrue(all(version is None for version in result["packages"].values()))
        self.assertEqual(result["runtime"], "not requested")
        self.assertFalse(result["tools"]["nvcc"]["available"])

    def test_cli_runtime_and_gpu_operations_are_explicit(self):
        for flags, expected_call in (([], None), (["--runtime"], False),
                                     (["--require-gpu"], True)):
            report = {"packages": {}, "runtime": "not requested"}
            with mock.patch.object(environment, "inventory", return_value=report):
                with mock.patch.object(environment, "runtime_checks", return_value=True) as check:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(environment.main(flags), 0)
            if expected_call is None:
                check.assert_not_called()
            else:
                check.assert_called_once_with(report, expected_call)

    @unittest.skipIf(importlib.util.find_spec("packaging") is None, "packaging unavailable")
    def test_candidate_is_a_floor_not_an_exact_image_lock(self):
        packages = dict(environment.MINIMUMS)
        packages.update({"torch": "2.7.0a0+vendor", "torchvision": "0.22.0a0",
                         "lightning": "2.5.5", "torchmetrics": "1.7.4", "SAM-2": "1.0"})
        report = {"packages": packages}
        self.assertTrue(environment.candidate_checks(report))
        packages["torch"] = "2.2.0"
        self.assertFalse(environment.candidate_checks(report))
        self.assertIn("torch", report["candidate_issues"])


class MigrationContractTests(unittest.TestCase):
    def test_plan_matches_unchanged_label_and_two_module_contracts(self):
        plan = json.loads((ROOT / "configs/potsdam/hiera_tiny_lightning.plan.json")
                          .read_text(encoding="utf-8"))
        self.assertEqual(plan["status"], "design_only_not_runnable")
        self.assertEqual(plan["model"]["num_classes"], 6)
        self.assertEqual(plan["data"]["train_id_range"], [0, 5])
        for section in ("data", "training", "evaluation"):
            self.assertEqual(plan[section]["ignore_index"], 255)
        self.assertEqual(plan["data"]["label_resize"], "nearest")
        self.assertEqual(plan["evaluation"]["checkpoint_monitor"], "val_mIoU")
        self.assertIsNone(plan["training"]["precision"])
        versions = json.loads((ROOT / "third_party/versions.json").read_text(encoding="utf-8"))
        self.assertEqual(set(versions["repositories"]), {"sam2", "GeoSeg"})
        parser = configparser.ConfigParser()
        parser.read(ROOT / ".gitmodules", encoding="utf-8")
        self.assertEqual({parser[section]["path"] for section in parser.sections()},
                         set(deployment.SUBMODULES))

    def test_two_submodule_export_materializes_links_and_verifies(self):
        # Git state is mocked; all file-copy, link resolution and SHA checks are real.
        # This is NOT acceptance of a clean export from the uncommitted worktree.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "source.py").write_text("value = 1\n", encoding="utf-8")
            for relative in deployment.SUBMODULES:
                folder = root / relative
                folder.mkdir(parents=True)
                (folder / ".git").write_text("fixture pointer\n", encoding="utf-8")
            sam = root / "third_party/sam2"
            (sam / "configs").mkdir()
            (sam / "configs/model.yaml").write_text("model: fixture\n", encoding="utf-8")
            (sam / "model.yaml").write_text("configs/model.yaml\n", encoding="utf-8")
            geo = root / "third_party/GeoSeg"
            (geo / "reference.py").write_text("# fixture\n", encoding="utf-8")
            states = {root: {"commit": "f" * 40, "dirty": False, "status": []},
                      sam: {"commit": "a" * 40, "dirty": False, "status": []},
                      geo: {"commit": "b" * 40, "dirty": False, "status": []}}

            def fake_git(folder, *args):
                if args == ("rev-parse", "--show-toplevel"):
                    return str(root)
                if args[0] == "rev-parse" and args[1].startswith("HEAD:"):
                    return states[root / args[1][5:]]["commit"]
                raise AssertionError((folder, args))

            def tracked(folder, prefix=""):
                if folder == root:
                    return [(prefix + "source.py", "100644", None, root)]
                if folder == sam:
                    return [(prefix + "configs/model.yaml", "100644", None, sam),
                            (prefix + "model.yaml", "120000", "configs/model.yaml", sam)]
                return [(prefix + "reference.py", "100644", None, geo)]

            release = root / "deploy/fixture"
            with mock.patch.object(deployment, "git", side_effect=fake_git):
                with mock.patch.object(deployment, "repository_state",
                                       side_effect=lambda path: dict(states[path])):
                    with mock.patch.object(deployment, "tracked_files", side_effect=tracked):
                        with contextlib.redirect_stdout(io.StringIO()):
                            deployment.create(root, root / "unused.json", release, True)
            manifest_path = release / "source_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["submodules"]), set(deployment.SUBMODULES))
            self.assertEqual(len(manifest["files"]), 4)
            entry = manifest["files"]["third_party/sam2/model.yaml"]
            self.assertTrue(entry["link_materialized"])
            self.assertEqual((release / "third_party/sam2/model.yaml").read_bytes(),
                             (sam / "configs/model.yaml").read_bytes())
            self.assertFalse(list(release.rglob(".git")))
            with contextlib.redirect_stdout(io.StringIO()):
                deployment.verify(release, manifest_path)


if __name__ == "__main__":
    unittest.main()
