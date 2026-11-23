"""Run the four sealed R0 GPU readbacks from a checked, source-photo-only handoff."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from evidence_lens._bundle import encoded, files, load, sha
from evidence_lens._verify import require
from worker_jobs import dispatch, frozen_json

SETS = {"M2": ["s00", "s03"], "M4": ["s00", "s01", "s02", "s03"]}


def validate(root, batch_hash):
    manifest = root / "batch.json"
    require(sha(manifest) == batch_hash, "Unrecognized batch SHA")
    b = load(manifest)
    require(
        b["schema"] == "el-r0-diagnostic-batch-1"
        and b["batch"] == "r0-dtu-checkpoints-01"
        and [(x["object_id"], x["condition"]) for x in b["jobs"]]
        == [(o, c) for o in ("set1", "set6") for c in SETS],
        "R0 requires the four original main arms",
    )
    tree = files(root / "worker")
    require({k: sha(p) for k, p in tree.items()} == b["worker_files"], "Frozen worker differs")
    for row in b["jobs"]:
        folder = root / "jobs" / (row["object_id"] + "-" + row["condition"])
        path = folder / "job.json"
        require(sha(path) == row["sha256"], "Diagnostic job changed")
        j = load(path)
        cfg = j["config"]
        ids = SETS[row["condition"]]
        require(
            cfg["source_ids"] == ids
            and cfg["object_id"] == row["object_id"]
            and cfg["condition_id"] == row["condition"]
            and cfg["optimization_steps"] == 0
            and j["trial_id"] == "r0-dtu-" + row["object_id"] + "-" + row["condition"]
            and set(j["roles"]) == {"initial", "final", "cameras", *ids}
            and set(files(folder))
            == {
                "job.json",
                "initial.npz",
                "scene.npz",
                "cameras.json",
                *("sources/" + s + ".png" for s in ids),
            },
            "Unexpected R0 input scope",
        )
        names = {
            "initial": "initial.npz",
            "final": "scene.npz",
            "cameras": "cameras.json",
            **{s: "sources/" + s + ".png" for s in ids},
        }
        for role, filename in names.items():
            r = j["roles"][role]
            p = folder / filename
            require(
                r["path"] == filename
                and sha(p) == r["sha256"]
                and p.stat().st_size == r["bytes"]
                and r["parent"] in cfg["originals"].values(),
                "Original diagnostic input differs",
            )
        yield row, j, folder


def launch(store, outgoing, row, j, folder, retry, resume):
    jobsha = row["sha256"]
    cfg = {
        **j["identity"],
        **j["config"],
        "experiment_id": j["experiment_id"],
        "trial_id": j["trial_id"],
        "formal_handoff_sha256": jobsha,
        "recording": {"expected_output_bytes": 5 * 1024**2},
    }
    cfgfile = store / "r0-configs" / j["trial_id"] / "config.json"
    cfgfile.parent.mkdir(parents=True, exist_ok=True)
    frozen_json(cfgfile, cfg)

    def execute():
        args = [
            sys.executable,
            "-m",
            "evidence_lens.recorder",
            "--root",
            str(store),
            "--source-root",
            str(CODE),
            "--config",
            str(cfgfile),
            "--kind",
            "archive_verify",
        ]
        for pin in cfg["originals"].values():
            args += ["--parent", pin["run_id"] + "=" + pin["manifest_sha256"]]
        for role, entry in j["roles"].items():
            args += [
                "--input",
                role + "=" + str(folder / entry["path"]),
                "--input-parent",
                role + "=" + entry["parent"]["run_id"],
            ]
        if retry:
            args += ["--retry-of", retry]
        args += [
            "--",
            "{python}",
            "-B",
            "{source}/scripts/r0_checkpoint_gpu.py",
            "--config",
            "{config}",
            "--initial",
            "{input:initial}",
            "--final",
            "{input:final}",
            "--cameras",
            "{input:cameras}",
            "--output",
            "{out}",
        ]
        for source in cfg["source_ids"]:
            args += ["--source", source, "{input:" + source + "}"]
        p = subprocess.run(
            args,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(CODE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if p.stderr:
            print(p.stderr, file=sys.stderr, flush=True)
        require(bool(p.stdout.strip()), "No recorder receipt")
        return json.loads(p.stdout.strip().splitlines()[-1])

    return dispatch(store, outgoing, j, jobsha, "archive_verify", None, retry, resume, execute)


def main(a):
    require(os.name == "nt", "Windows CUDA host required")
    jobs = list(validate(CODE.parent, a.batch_sha256))
    out = a.returns.resolve()
    out.mkdir(parents=True, exist_ok=True)
    store = a.store.resolve()
    results = []
    for row, j, folder in jobs:
        if a.trial and a.trial != j["trial_id"]:
            continue
        r = launch(store, out, row, j, folder, a.retry, a.resume_preflight)
        results.append(r)
        print(json.dumps(r), flush=True)
    require(results, "Unknown diagnostic trial")
    result = {"schema": "el-r0-windows-returns-1", "batch_sha256": a.batch_sha256, "returns": results}
    name = "r0-dtu-checkpoints-01-summary"
    if a.trial or (out / (name + ".json")).exists():
        name += "-" + hashlib.sha256(encoded(result)).hexdigest()[:16]
    frozen_json(out / (name + ".json"), result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-sha256", required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--returns", type=Path, required=True)
    p.add_argument("--trial")
    p.add_argument("--retry")
    p.add_argument("--resume-preflight")
    a = p.parse_args()
    if (a.retry or a.resume_preflight) and not a.trial:
        p.error("retry/resume requires a specific trial")
    main(a)
