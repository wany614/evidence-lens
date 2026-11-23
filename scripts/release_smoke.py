"""Verify an extracted Evidence Lens RC, install its wheel and grade six views on a clean CPU."""

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

if sys.version_info < (3, 11):
    raise SystemExit("Evidence Lens CPU smoke requires Python 3.11 or newer; run python3 or py -3 with a supported interpreter.")

VIEWS = tuple(f"v{n:02}" for n in range(1, 7))
METHODS = ("constant", "1-alpha", "L0", "L1-full", "L1-geometry")


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
        require(not path.is_symlink(), "linked release member: " + str(path.relative_to(folder)))
        if not path.is_dir():
            require(path.is_file(), "unexpected release member: " + str(path.relative_to(folder)))
            members[path.relative_to(folder).as_posix()] = path
    require("release.json" in members, "missing release member: release.json")
    release_sha = digest(members["release.json"])
    require(not expected or release_sha == expected, "release.json differs from its supplied checksum")
    release = read(members["release.json"])
    files = release["files"]
    require(release["schema"] == "el-public-model-release-1" and "-rc." in release["version"], "unexpected RC schema or tag")
    expected_members = set(files) | {"release.json", "SHA256SUMS"}
    missing, extra = sorted(expected_members - members.keys()), sorted(members.keys() - expected_members)
    require(not missing and not extra, f"extracted file set differs; missing={missing}, extra={extra}")
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


@contextmanager
def preserve_partial(staged, output, release_sha):
    staged.mkdir()
    try:
        yield staged
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        if any(staged.iterdir()):
            failed = Path(tempfile.mkdtemp(prefix=output.name + "-failed-", dir=output.parent))
            staged.rename(failed / "results")
            (failed / "failure.json").write_text(json.dumps({"passed":False, "release_sha256":release_sha,
                                                            "error":str(exc)}, indent=2)+"\n", encoding="utf-8")
            raise ValueError(f"{exc}; partial smoke evidence saved in {failed}") from exc
        raise


def distributions(grade):
    views = grade.get("views", {})
    require(set(views)==set(VIEWS), "CPU grade view set differs: missing=" + str(sorted(set(VIEWS)-views.keys())) +
            ", extra=" + str(sorted(views.keys()-set(VIEWS))) + "; " + str(grade.get("error","")))
    for view in VIEWS:
        row = views[view]
        require("error" not in row, f"{view}: {row.get('error')}")
        require(row.get("l0_reason_codes_equal") is True, f"{view}/L0: rule reason codes differ")
        methods = row.get("methods", {})
        require(set(methods)==set(METHODS),f"{view}: methods missing={sorted(set(METHODS)-methods.keys())}, extra={sorted(methods.keys()-set(METHODS))}")
        for name in METHODS:
            measured = methods[name]
            n, actual, pixels = (measured.get(k) for k in ("finite_expected","finite_actual","pixels"))
            gap = measured.get("max_absolute_score_difference")
            require(type(n) is int and type(actual) is int and type(pixels) is int and 0 < n == actual <= pixels,
                    f"{view}/{name}: finite population expected={n}, actual={actual}, pixels={pixels}")
            require(type(gap) in (float,int) and math.isfinite(gap) and 0 <= gap <= 1e-6,
                    f"{view}/{name}: maximum finite score difference {gap} exceeds or lacks the 1e-6 comparison")
            for field in ("validity_equal","states_equal","passed"):
                require(measured.get(field) is True, f"{view}/{name}: {field} failed")
    return {v:views[v]["methods"] for v in VIEWS}


def verified_maps(directory, release):
    index = read(directory/"index.json")
    require(index["schema"]=="el-score-maps-1" and index["method"]=="L1-full"
            and index["source"]["model_id"]==release["model_id"] and set(index["views"])==set(VIEWS), "incomplete six-view L1-full map index")
    images = {}
    for view in VIEWS:
        pin = index["views"][view]; metadata = directory/(view+".json")
        require(pin["file"]==metadata.name and pin["map_status"]=="written" and digest(metadata)==pin["sha256"],f"{view}: map metadata differs")
        specs = read(metadata)["maps"]
        for key, suffix in (("continuous", "-risk.png"),("state","-state.png")):
            path=directory/(view+suffix);spec=specs[key]
            require(path.is_file() and not path.is_symlink() and spec["file"]==path.name
                    and path.stat().st_size==spec["bytes"] and spec["bytes"]>8 and digest(path)==spec["sha256"],
                    f"{view}/{key}: missing or changed map PNG {path.name}")
            with path.open("rb") as image:
                require(image.read(8)==b"\x89PNG\r\n\x1a\n",f"{view}/{key}: invalid PNG header")
            images[path.name]=spec["sha256"]
    return images


def verified_archive(folder, release, path, expected):
    require((path is None) == (expected is None), "supply the original ZIP and its independent digest together")
    if path is None:
        return None
    require(not path.is_symlink() and path.is_file() and digest(path) == expected, "candidate ZIP checksum differs")
    hashes = {name: row["sha256"] for name, row in release["files"].items()}
    hashes.update({name: digest(folder/name) for name in ("release.json", "SHA256SUMS")})
    with zipfile.ZipFile(path) as zipped:
        names=zipped.namelist()
        require(names == sorted(hashes), f"ZIP file inventory or order differs; missing={sorted(hashes.keys()-set(names))}, extra={sorted(set(names)-hashes.keys())}")
        for item in zipped.infolist():
            with zipped.open(item) as member:
                require(item.external_attr >> 16 == 0o100644 and hashlib.file_digest(member, "sha256").hexdigest() == hashes[item.filename],
                        "downloaded ZIP member differs from the extracted release")
    return expected


def smoke(folder, output, *, release_sha=None, wheelhouse=None, offline_uv=False, archive=None, archive_sha=None, fixture=False):
    folder, output = folder.resolve(), output.absolute()
    require(not output.exists() and not output.is_symlink() and not output.resolve().is_relative_to(folder),
            "choose a new output directory outside the extracted release")
    release, pinned, demo_sha = verify_files(folder, release_sha)
    require(release["kind"] == ("synthetic_fixture" if fixture else "heldout_study"),
            "official CPU smoke requires a held-out study release; synthetic tests require the explicit --fixture switch")
    if "publication" in release:
        scope=read(folder/"RELEASE_SCOPE.json")
        expected={"publication_status":"public_prerelease","repository":"wany614/evidence-lens","attribution":"wany614",
                  "tag":release["version"],"release_url":"https://github.com/wany614/evidence-lens/releases/tag/"+release["version"]}
        require(scope.get("publication_status")=="public_prerelease" and all(scope.get(k)==expected[k] for k in ("repository","attribution"))
                and release["publication"]==({"publication_status":"synthetic_fixture"} if fixture else expected),
                "public release and its scope identify a different destination or attribution")
    measured_archive = verified_archive(folder, release, archive, archive_sha)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".evidence-lens-smoke-") as scratch, \
            preserve_partial(Path(scratch)/"results", output, pinned) as staged:
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
        worker=subprocess.run([python, "-I", "-m", "evidence_lens.l1_demo", "grade", "--bundle", folder,
                               "--manifest-sha256", demo_sha, "--output", staged/"grade"],
                              cwd=work,capture_output=True,text=True)
        if worker.returncode:
            (staged/"grader.log").write_text(worker.stdout+worker.stderr,encoding="utf-8")
        require((staged/"grade/grade.json").is_file(),f"CPU grade exited {worker.returncode} without grade/grade.json")
        grade = read(staged / "grade/grade.json")
        graded = time.perf_counter()
        compared = distributions(grade)
        require(worker.returncode == 0 and grade["passed"] and grade["numerical_l1_parity_exercised"] and
                grade["runtime"]["isolated"] and not grade["runtime"]["optional_training_render_packages_present"]
                and grade["runtime"]["packages"]["evidence-lens"] == release["wheel"]["version"]
                and grade["model_id"] == release["model_id"] and grade["model_index_sha256"]==release["model_index_sha256"]
                and grade["source_demo_sha256"]==demo_sha and grade["expected_score_run_id"]==release["sources"]["example_scores"]["run_id"]
                and grade["absolute_tolerance"] == 1e-6,
                "clean CPU grade or installed identity differs")
        run([python, "-I", "-m", "evidence_lens.score_maps", "--scores", staged / "grade/scores",
             "--output", staged / "maps"], work)
        images = verified_maps(staged / "maps", release)
        report = {"schema": "el-prerelease-cpu-smoke-1", "passed": True, "kind":release["kind"],
                  "scope": "synthetic_fixture" if fixture else "authored_plane_cpu_deployment",
                  "version": release["version"], "release_sha256": pinned, "demo_sha256": demo_sha,
                  "verified_zip_sha256": measured_archive,
                  "model_id": release["model_id"], "original_score_run": grade["expected_score_run_id"],
                  "runtime": grade["runtime"], "os": platform.system(), "architecture": platform.machine(),
                  "system_release": platform.release(), "system_version": platform.version(),
                  "macos_product_version": platform.mac_ver()[0] if sys.platform == "darwin" else None,
                  "install_seconds": installed - started, "grade_seconds": graded - installed,
                  "views": list(VIEWS), "methods": sorted(METHODS), "comparisons":compared, "map_pngs":images, "finite_masks_equal": True,
                  "state_arrays_equal": True, "l0_reason_arrays_equal": True, "absolute_tolerance": 1e-6,
                  "max_absolute_score_difference": max(m["max_absolute_score_difference"] for row in compared.values() for m in row.values()),
                  "grade_sha256": digest(staged / "grade/grade.json"), "map_index_sha256": digest(staged / "maps/index.json")}
        if "publication" in release:
            report["publication"]=release["publication"]
        (staged / "smoke.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        require(not output.exists(), "smoke output destination was filled concurrently")
        staged.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", action="store_true", help="explicitly grade only a synthetic software fixture")
    parser.add_argument("--release-sha256", help="independently supplied digest of release.json")
    parser.add_argument("--archive", type=Path, help="downloaded original ZIP to bind to this grade")
    parser.add_argument("--archive-sha256", help="independently supplied SHA-256 of that ZIP")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--wheelhouse", type=Path, help="offline pip directory with compatible NumPy/Pillow wheels")
    group.add_argument("--offline-uv", action="store_true", help="use an existing uv package cache, offline")
    args = parser.parse_args()
    try:
        result = smoke(args.bundle, args.output, release_sha=args.release_sha256, wheelhouse=args.wheelhouse,
                       offline_uv=args.offline_uv, archive=args.archive, archive_sha=args.archive_sha256, fixture=args.fixture)
    except (KeyError, OSError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
