"""Create a Git version/hash manifest, export clean sources, or verify an SFTP copy."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

SUBMODULES = ("third_party/sam2", "third_party/mmsegmentation", "third_party/GeoSeg")


def git_executable():
    explicit = os.environ.get("GIT_EXE")
    if explicit:
        return explicit
    windows_git = Path("D:/Git/mingw64/bin/git.exe")
    if os.name == "nt" and windows_git.is_file():
        return str(windows_git)
    command = shutil.which("git")
    if not command:
        raise RuntimeError("Git not found; set GIT_EXE to the real executable")
    return command


def git(root, *args):
    result = subprocess.run([git_executable(), "-C", str(root), *args],
                            capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout.decode("utf-8")


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_path(root, relative):
    pure = PurePosixPath(relative)
    if not pure.parts or pure.is_absolute() or ".." in pure.parts or "\\" in relative:
        raise ValueError(f"Unsafe relative path: {relative}")
    path = root.joinpath(*pure.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes root: {relative}")
    return path


def repository_state(root):
    status = git(root, "status", "--porcelain=v1", "--untracked-files=all",
                 "--ignore-submodules=none")
    return {"commit": git(root, "rev-parse", "HEAD").strip(),
            "dirty": bool(status), "status": status.splitlines()}


def resolve_link(source, target, repository_root):
    """Materialize only a tracked relative link contained within its repository."""
    if Path(target).is_absolute() or "\\" in target:
        raise ValueError(f"Unsupported link target: {target}")
    resolved = (source.parent / target).resolve()
    if not resolved.is_relative_to(repository_root.resolve()) or not resolved.is_file():
        raise ValueError(f"Missing or escaping link target: {target}")
    return resolved


def tracked_files(root, prefix=""):
    for record in git(root, "ls-files", "--stage", "-z").split("\0"):
        if not record:
            continue
        info, name = record.split("\t", 1)
        mode, _, stage = info.split()
        if stage != "0":
            raise RuntimeError("Unmerged Git index")
        if mode == "160000":
            if prefix or name not in SUBMODULES:
                raise RuntimeError(f"Unexpected/nested submodule: {prefix}{name}")
            continue
        target = git(root, "show", f":{name}").strip() if mode == "120000" else None
        yield prefix + name, mode, target, root


def create(root, output, export_dir, require_clean):
    if Path(git(root, "rev-parse", "--show-toplevel").strip()).resolve() != root:
        raise RuntimeError("--root must be the main repository root")
    main = repository_state(root)
    modules = {}
    names = list(tracked_files(root))
    for relative in SUBMODULES:
        module_root = root / relative
        if not (module_root / ".git").is_file():
            raise RuntimeError(f"Submodule not initialized: {relative}")
        state = repository_state(module_root)
        state["recorded_commit"] = git(root, "rev-parse", f"HEAD:{relative}").strip()
        state["matches_recorded_commit"] = state["commit"] == state["recorded_commit"]
        modules[relative] = state
        names.extend(tracked_files(module_root, relative + "/"))
    dirty = main["dirty"] or any(
        item["dirty"] or not item["matches_recorded_commit"] for item in modules.values())
    if dirty and (require_clean or export_dir):
        raise RuntimeError("Uncommitted changes or mismatched submodules; commit/review first")
    manifest = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "main": main, "submodules": modules, "dirty": dirty,
                "hash_basis": ("deployment bytes; tracked links materialized" if export_dir
                               else "working-tree bytes; tracked links unchanged"),
                "files": {}}
    if export_dir:
        if not export_dir.resolve().is_relative_to((root / "deploy").resolve()):
            raise RuntimeError("Export must be inside the ignored root deploy/ directory")
        export_dir.mkdir(parents=True, exist_ok=False)
    for name, mode, target, repository_root in sorted(names, key=lambda item: item[0]):
        source = safe_path(root, name)
        if export_dir and target:
            source = resolve_link(root / name, target, repository_root)
        if not source.is_file():
            raise RuntimeError(f"Tracked file missing: {name}")
        fingerprint = digest(source)
        entry = {"sha256": fingerprint, "size": source.stat().st_size, "git_mode": mode}
        if target:
            entry.update(link_target=target, link_materialized=bool(export_dir))
        manifest["files"][name] = entry
        if export_dir:
            destination = safe_path(export_dir, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if digest(destination) != fingerprint:
                raise RuntimeError(f"File changed during export: {name}")
    if repository_state(root) != main or any(
            repository_state(root / name) != {k: value[k] for k in ("commit", "dirty", "status")}
            for name, value in modules.items()):
        raise RuntimeError("Repository changed while collecting manifest; retry")
    destination = export_dir / "source_manifest.json" if export_dir else output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Manifest: {destination}; files={len(names)}; dirty={dirty}; commit={main['commit']}")


def verify(root, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported manifest schema")
    errors = []
    for name, expected in manifest["files"].items():
        path = safe_path(root, name)
        if not path.is_file():
            errors.append(f"missing: {name}")
        elif path.stat().st_size != expected["size"] or digest(path) != expected["sha256"]:
            errors.append(f"mismatch: {name}")
    if errors:
        raise RuntimeError("\n".join(errors))
    print(f"Verified {len(manifest['files'])} files; source commit={manifest['main']['commit']}")
    print("File hashes verified; this is not a GPU/runtime validation or an authenticity signature.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "verify"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("outputs/version_manifest.json"))
    parser.add_argument("--manifest", type=Path, help="Required for verify")
    parser.add_argument("--export-dir", type=Path, help="New directory under deploy/; clean repo required")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "verify":
            if not args.manifest:
                parser.error("verify requires --manifest")
            verify(args.root.resolve(), args.manifest)
        else:
            create(args.root.resolve(), args.output, args.export_dir, args.require_clean)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
