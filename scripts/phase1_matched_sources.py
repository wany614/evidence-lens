"""Record native source-view references for the four selected historical R1 families."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evidence_lens._bundle import encoded, load, sha
from evidence_lens._verify import require, verify_bundle


def record(store, output, python, family):
    selection_path = store / "trials" / (family + "-R1") / "trial_selection.json"
    selection = load(selection_path)
    require(selection["selection_status"] == "selected" and selection["condition_id"] == "R1" and
            selection["split"] == "test" and selection["scene_family_id"] == family, "R1 selection mismatch")
    roles = {"blend": ("reference", "payload/scene/" + family + "_v001.blend"),
             "settings": ("reference", "payload/scene/reference_config.json"),
             "cameras": ("preparation", "payload/cameras.json")}
    parents = {role: {k: selection["selected"][role][k] for k in ("run_id", "manifest_sha256")}
               for role in ("reference", "preparation")}
    pins, inputs = {}, {}
    for name, (role, rel) in roles.items():
        path = store / "runs" / parents[role]["run_id"]
        require(sha(path / "manifest.json") == parents[role]["manifest_sha256"], "Historical parent changed")
        artifact = next(f for f in load(path / "artifacts.json")["files"] if f["path"] == rel)
        inputs[name] = path / rel
        pins[name] = artifact["sha256"]
        require(sha(inputs[name]) == pins[name], "Historical input changed")
    require(pins["cameras"] == selection["sources"]["cameras"]["sha256"], "R1 camera selection changed")
    cfg = {"schema": "el-phase1-matched-source-config-1", "scene_family_id": family, "scene_id": selection["scene_id"],
           "split": "test", "condition_id": "R1", "dataset_id": selection["dataset_id"], "dataset_hash": selection["dataset_hash"],
           "camera_absolute_tolerance": 1e-6, "pins": pins, "parents": parents,
           "selection_sha256": sha(selection_path), "recording": {"expected_output_bytes": 8_000_000}}
    cfg_path = output / (family + "-plan.json")
    with cfg_path.open("xb") as stream:
        stream.write(encoded(cfg))
    command = [sys.executable, "-m", "evidence_lens.recorder", "--root", str(store), "--kind", "scene_render",
               "--config", str(cfg_path), "--experiment", "phase1-r1-source", "--trial", family + "-R1-source"]
    for identity in parents.values():
        command.extend(("--parent", identity["run_id"] + "=" + identity["manifest_sha256"]))
    for name, path in inputs.items():
        command.extend(("--input", name + "=" + str(path)))
    command += ["--", str(python.absolute()), "{source}/scripts/phase1_source_reference.py", "--plan", "{config}",
                "--blend", "{input:blend}", "--settings", "{input:settings}", "--cameras", "{input:cameras}", "--out", "{out}"]
    process = subprocess.run(command, capture_output=True, text=True)
    print(process.stdout, end="", flush=True)
    if process.returncode:
        raise RuntimeError(process.stderr or process.stdout)
    rid = json.loads(process.stdout)["run_id"]
    checked = verify_bundle(store / "runs" / rid)
    return {"run_id": rid, "manifest_sha256": checked["manifest_sha256"], "parents": parents,
            "reference_path": "payload/source.npz", "cameras_path": "payload/cameras.json",
            "receipt_path": "payload/validation.json",
            "sha256": {name: checked["artifacts"]["payload/" + name]["sha256"]
                       for name in ("source.npz", "cameras.json", "validation.json")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    families = [r["scene_family_id"] for r in load(Path("protocol/families-v1.json"))["families"] if r["split"] == "test"]
    index = {family: record(args.store.resolve(), args.out, args.reference_python, family) for family in families}
    with (args.out / "matched-source-index.json").open("xb") as stream:
        stream.write(encoded({"schema": "el-phase1-matched-source-index-1", "families": index}))


if __name__ == "__main__":
    main()
