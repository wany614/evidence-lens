"""Reproduce the 057 M4 initialization from the published example in a fresh recorder store."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require, verify_bundle


EXAMPLE_SHA = "14204986e0d598adc613dd1a9ef2e5c1584f5be001d3541531d5d9e709707548"
SOURCE_IDS = ("s00", "s01", "s02", "s03")


def run(example, geometry_python, store, project):
    require(sha(example) == EXAMPLE_SHA, "supply the exact v0.3.0-rc.1 public example ZIP")
    rules = project / "protocol/recovery-initialization-v1.json"
    study = project / "protocol/reconstruction-recovery-v1.json"
    with TemporaryDirectory(prefix="el-recovery-public-") as tmp:
        scratch = Path(tmp)
        paths = {}
        with ZipFile(example) as archive:
            names = {"cameras": "source_cameras.json", **{v: "sources/" + v + ".png" for v in SOURCE_IDS}}
            for role, relative in names.items():
                archive_name = "CASE/M4/" + relative
                require(archive.namelist().count(archive_name) == 1, "missing or duplicate example member")
                path = scratch / Path(relative).name
                path.write_bytes(archive.read(archive_name))
                paths[role] = path
        config = {
            "experiment_id": "recovery-public-057-reproduction", "trial_id": "r-public-057-source-initialization",
            "scene_family_id": "scene_057_offset_bell_cage", "condition_id": "M4",
            "generator_source_ids": list(SOURCE_IDS), "source_camera_sha256": sha(paths["cameras"]),
            "source_rgb_sha256": {v: sha(paths[v]) for v in SOURCE_IDS},
            "initialization_protocol_sha256": sha(rules), "recipe_sha256": sha(rules),
            "registration": {"study_sha256": sha(study), "role": "fresh public source reproduction"},
            "source_parent": {"public_example_sha256": EXAMPLE_SHA, "member_root": "CASE/M4/"},
            "recording": {"expected_output_bytes": 1024**2},
        }
        request = scratch / "config.json"
        save(request, config)
        command = [sys.executable, "-m", "evidence_lens.recorder", "--root", str(store), "--source-root", str(project),
                   "--config", str(request), "--kind", "archive_verify"]
        for key, path in paths.items():
            command.extend(["--input", key + "=" + str(path)])
        command += ["--", str(geometry_python), "{source}/scripts/recovery_triangulate.py",
                    "--cameras", "{input:cameras}", "--config", "{config}", "--out", "{out}",
                    "--protocol", "{source}/protocol/recovery-initialization-v1.json"]
        for v in SOURCE_IDS:
            command.extend(["--source", v, "{input:" + v + "}"])
        result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError((result.stdout + result.stderr)[-1600:])
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    root = store / "runs" / receipt["run_id"]
    manifest = verify_bundle(root)["manifest_sha256"]
    meta = load(root / "payload/init.json")
    return {"run_id": receipt["run_id"], "manifest_sha256": manifest, "actual_method": meta["actual_method"],
            "point_count": meta["N"], "init_npz_sha256": sha(root / "payload/init.npz"),
            "parameter_sha256": meta["parameter_sha256"], "source_tracks_sha256": meta["source_tracks_sha256"]}


def export_cell(destination, store, receipt, example, recipe_path):
    """Prepare a fresh synthetic-only cell for replay of the separate CUDA stage."""
    from recovery.contract import cell_manifest, validate_recipe

    require(not destination.exists() and sha(example) == EXAMPLE_SHA, "need a new directory and the released example")
    validate_recipe(load(recipe_path))
    run_root = store / "runs" / receipt["run_id"]
    verify = verify_bundle(run_root)
    require(verify["manifest_sha256"] == receipt["manifest_sha256"] and
            verify["run"]["experiment_id"] == "recovery-public-057-reproduction", "wrong initializer lineage")
    init = load(run_root / "payload/init.json")
    require(init["object_id"] == "scene_057_offset_bell_cage" and init["condition"] == "M4",
            "the public example supplies only the registered M4 positive control")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as td:
        stage = Path(td) / "cell"
        (stage / "sources").mkdir(parents=True)
        inputs = ["source_cameras.json", "render_cameras.json", *("sources/" + s + ".png" for s in SOURCE_IDS)]
        with ZipFile(example) as z:
            for name in inputs:
                n = "CASE/M4/" + name
                require(z.namelist().count(n) == 1, "missing or duplicate public cell member")
                (stage / name).write_bytes(z.read(n))
        for n in ("init.json", "init.npz"):
            shutil.copyfile(run_root / "payload" / n, stage / n)
            inputs.append(n)
        files = {n: {"sha256": sha(stage / n), "bytes": (stage / n).stat().st_size} for n in inputs}
        require(all(files["sources/" + s + ".png"]["sha256"] == init["source_rgb_sha256"][s] for s in SOURCE_IDS)
                and files["source_cameras.json"]["sha256"] == init["source_camera_sha256"]
                and files["init.npz"]["sha256"] == init["init_npz_sha256"], "public reconstruction input differs")
        rid = receipt["run_id"]
        job = {"schema": "el-recovery-cell-1", "purpose": "formal", "object_id": init["object_id"],
               "condition": "M4", "experiment_id": "recovery-public-057-reproduction",
               "trial_id": "r-public-057-reconstruction", "stop_step": 5000,
               "parents": [{"run_id": rid, "manifest_sha256": receipt["manifest_sha256"]}],
               "recipe_sha256": sha(recipe_path), "files": files,
               "input_parents": {n: rid for n in files if n != "render_cameras.json"}}
        save(stage / "job.json", job)
        digest = sha(stage / "job.json")
        cell_manifest(stage, digest)
        stage.rename(destination)
    return {"job_sha256": digest, "recipe_sha256": job["recipe_sha256"], "members": len(files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "example-zip", "geometry-python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--export-worker-cell", type=Path, help="new source-only directory for independent 057 M4 replay")
    parser.add_argument("--recipe", type=Path, help="registered recovery recipe for the public worker cell")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.export_worker_cell and args.export_worker_cell.exists():
        parser.error("choose a new public cell destination")
    result = run(args.example_zip.resolve(), args.geometry_python.expanduser().absolute(), args.store.resolve(), project)
    if args.export_worker_cell:
        result["worker_cell"] = export_cell(args.export_worker_cell.absolute(), args.store.resolve(), result,
                                            args.example_zip, args.recipe or project / "protocol/recovery-gsplat-v1.json")
    print(json.dumps(result))
