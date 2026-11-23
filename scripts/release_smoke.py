"""Verify an extracted Evidence Lens RC, install its wheel and grade six views on a clean CPU."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_files(folder, expected):
    members = {}
    for path in folder.rglob("*"):
        require(not path.is_symlink(), "linked release member")
        if not path.is_dir():
            require(path.is_file(), "unexpected release member")
            members[path.relative_to(folder).as_posix()] = path
    release_sha = digest(members["release.json"])
    require(not expected or release_sha == expected, "release.json differs from its supplied checksum")
    release = read(members["release.json"])
    files = release["files"]
    require(release["schema"] == "el-public-model-release-1" and "-rc." in release["version"]
            and set(members) == set(files) | {"release.json", "SHA256SUMS"}, "unexpected RC contents")
    actual = {name: digest(path) for name, path in members.items() if name != "SHA256SUMS"}
    for name, row in files.items():
        require(actual[name] == row["sha256"] and members[name].stat().st_size == row["bytes"],
                "release member changed: " + name)
    checks = "".join(f"{actual[name]}  {name}\n" for name in sorted(actual))
    require(members["SHA256SUMS"].read_bytes() == checks.encode(), "SHA256SUMS differs from extracted files")
    wheel = release["wheel"]
    require(wheel["path"] == "distribution/" + wheel["name"] and wheel["path"] in files and
            actual[wheel["path"]] == wheel["sha256"] and actual["RELEASE_SCOPE.json"] == release["scope_sha256"],
            "wheel or scope differs from its release pin")
    return release, release_sha, actual["demo.json"]


def run(args, cwd):
    subprocess.run([os.fspath(s) for s in args], check=True, cwd=cwd)


def verified_archive(folder, release, path, expected):
    require((path is None) == (expected is None), "supply the original ZIP and its independent digest together")
    if path is None:
        return None
    require(not path.is_symlink() and path.is_file() and digest(path) == expected, "candidate ZIP checksum differs")
    hashes = {name: row["sha256"] for name, row in release["files"].items()}
    hashes.update({name: digest(folder/name) for name in ("release.json", "SHA256SUMS")})
    with zipfile.ZipFile(path) as zipped:
        require(zipped.namelist() == sorted(hashes), "downloaded ZIP file list differs from the extracted release")
        for item in zipped.infolist():
            with zipped.open(item) as member:
                require(item.external_attr >> 16 == 0o100644 and hashlib.file_digest(member, "sha256").hexdigest() == hashes[item.filename],
                        "downloaded ZIP member differs from the extracted release")
    return expected


def smoke(folder, output, *, release_sha=None, wheelhouse=None, offline_uv=False, archive=None, archive_sha=None):
    folder, output = folder.resolve(), output.absolute()
    require(not output.exists() and not output.is_symlink() and not output.resolve().is_relative_to(folder),
            "choose a new output directory outside the extracted release")
    release, pinned, demo_sha = verify_files(folder, release_sha)
    measured_archive = verified_archive(folder, release, archive, archive_sha)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".evidence-lens-smoke-") as scratch:
        work = Path(scratch)
        env = work / "cpu"
        python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        wheel = folder / release["wheel"]["path"]
        started = time.perf_counter()
        if offline_uv:
            require(wheelhouse is None, "select the uv cache or the local pip wheelhouse")
            uv = shutil.which("uv")
            require(uv, "uv is required for --offline-uv")
            run([uv, "venv", "--offline", "--python", sys.executable, env], work)
            run([uv, "pip", "install", "--offline", "--python", python, wheel], work)
        else:
            run([sys.executable, "-I", "-m", "venv", env], work)
            extra = ["--no-index", "--find-links", wheelhouse.resolve()] if wheelhouse else []
            run([python, "-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
                 "--only-binary=:all:", *extra, wheel], work)
        installed = time.perf_counter()
        staged = work / "results"
        staged.mkdir()
        run([python, "-I", "-m", "evidence_lens.l1_demo", "grade", "--bundle", folder,
             "--manifest-sha256", demo_sha, "--output", staged / "grade"], work)
        grade = read(staged / "grade/grade.json")
        graded = time.perf_counter()
        views = grade["views"]
        ids = {f"v{n:02}" for n in range(1, 7)}
        require(grade["passed"] and grade["numerical_l1_parity_exercised"] and set(views) == ids and
                grade["runtime"]["isolated"] and not grade["runtime"]["optional_training_render_packages_present"]
                and grade["runtime"]["packages"]["evidence-lens"] == release["wheel"]["version"]
                and grade["model_id"] == release["model_id"] and grade["absolute_tolerance"] == 1e-6,
                "clean CPU grade or installed identity differs")
        methods = {"constant", "1-alpha", "L0", "L1-full", "L1-geometry"}
        outcomes = [m for view in views.values() for m in view["methods"].values()]
        require(all(view["l0_reason_codes_equal"] and set(view["methods"]) == methods for view in views.values())
                and all(m["passed"] and m["validity_equal"] and m["states_equal"] for m in outcomes),
                "CPU score, finite mask, state or rule-reason comparison failed")
        run([python, "-I", "-m", "evidence_lens.score_maps", "--scores", staged / "grade/scores",
             "--output", staged / "maps"], work)
        report = {"schema": "el-prerelease-cpu-smoke-1", "passed": True, "scope": "authored_plane_cpu_deployment",
                  "version": release["version"], "release_sha256": pinned, "demo_sha256": demo_sha,
                  "verified_zip_sha256": measured_archive,
                  "model_id": release["model_id"], "original_score_run": grade["expected_score_run_id"],
                  "runtime": grade["runtime"], "os": platform.system(), "architecture": platform.machine(),
                  "system_release": platform.release(), "system_version": platform.version(),
                  "macos_product_version": platform.mac_ver()[0] if sys.platform == "darwin" else None,
                  "install_seconds": installed - started, "grade_seconds": graded - installed,
                  "views": sorted(views), "methods": sorted(methods), "finite_masks_equal": True,
                  "state_arrays_equal": True, "l0_reason_arrays_equal": True, "absolute_tolerance": 1e-6,
                  "max_absolute_score_difference": max(m["max_absolute_score_difference"] for m in outcomes
                                                        if m["max_absolute_score_difference"] is not None),
                  "grade_sha256": digest(staged / "grade/grade.json"), "map_index_sha256": digest(staged / "maps/index.json")}
        (staged / "smoke.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        require(not output.exists(), "smoke output destination was filled concurrently")
        staged.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-sha256", help="independently supplied digest of release.json")
    parser.add_argument("--archive", type=Path, help="downloaded original ZIP to bind to this grade")
    parser.add_argument("--archive-sha256", help="independently supplied SHA-256 of that ZIP")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--wheelhouse", type=Path, help="offline pip directory with compatible NumPy/Pillow wheels")
    group.add_argument("--offline-uv", action="store_true", help="use an existing uv package cache, offline")
    args = parser.parse_args()
    try:
        result = smoke(args.bundle, args.output, release_sha=args.release_sha256, wheelhouse=args.wheelhouse,
                       offline_uv=args.offline_uv, archive=args.archive, archive_sha=args.archive_sha256)
    except (KeyError, OSError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
