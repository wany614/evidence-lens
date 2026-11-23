"""Record the study boundaries and bind them to the original D1 freeze."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from evidence_lens import _capacity
from evidence_lens._bundle import load, save, sha, utc
from evidence_lens._verify import require, verify_bundle


STUDY = "protocol/reconstruction-recovery-v1.json"
OLD_FREEZE = {
    "run_id": "20261123T195524386855Z-2ea8b1e8f0e94d3a",
    "manifest_sha256": "a90a08799089c1f934c94ae80eead2f5ef316d6571ca308d1360c0c03df1e942",
    "member": "payload/freeze.json",
    "sha256": "49be35c4f44dd35075945cfbeffc1d8a8b39f2624b54603b605659c05f71955b",
}


def verify_registration(source, config, previous, output):
    study = load(source / STUDY)
    require(set(config["registered_files"]) == {STUDY, *study["inherits"].values()},
            "registration does not cover all study rules")
    for name, digest in config["registered_files"].items():
        require(sha(source / name) == digest, f"registration source changed: {name}")
    old = load(previous)
    require(sha(previous) == OLD_FREEZE["sha256"] and old["run_id"] == OLD_FREEZE["run_id"],
            "wrong D1 freeze")
    require(old["protocol_sha256"] == sha(source / study["inherits"]["dtu"]),
            "historical DTU protocol differs")
    for name, digest in old["code_sha256"].items():
        require(sha(source / name) == digest, f"frozen scientific source differs: {name}")
    original = old["protocol"]
    for field in ("M2", "M4", "sources", "targets", "light"):
        require(original["camera"][field] == study["dtu_input"][field], f"changed camera registration: {field}")
    require(study["release_commit"] == original["release"]["commit"] and
            study["R2"]["limits"]["risk_training_runs"] == study["R2"]["limits"]["risk_recalibrations"] == 0,
            "wrong risk release or training permission")
    save(output / "registration.json", {
        "schema": "el-recovery-registration-1", "run_id": os.environ["EL_RUN_ID"], "utc": utc(),
        "protocol_sha256": sha(source / STUDY), "registered_files": config["registered_files"],
        "previous_freeze": OLD_FREEZE, "historical_source_sha256": old["code_sha256"],
        "release": original["release"], "models": original["models"], "scopes": original["scopes"],
        "sources_and_targets": study["dtu_input"], "roles": study["roles"],
        "future_objects": study["future_objects"], "selection": study["R2"]["selection"],
        "resource_baseline": config["resource_baseline"],
    })


def register(store, project):
    record = verify_bundle(store / "runs" / OLD_FREEZE["run_id"])
    require(record["manifest_sha256"] == OLD_FREEZE["manifest_sha256"] and
            record["run"]["execution_status"] == "succeeded", "unverified historical freeze")
    work = store / "work" / "r-control"
    work.mkdir(exist_ok=True, parents=True)
    requested = work / "registration-request.json"
    study = load(project / STUDY)
    hashes = {name: sha(project / name) for name in (STUDY, *study["inherits"].values())}
    selected = work / "registration.json"
    if selected.exists():
        pin = load(selected)
        evidence = verify_bundle(store / "runs" / pin["run_id"])
        result = store / "runs" / pin["run_id"] / pin["member"]
        require(evidence["manifest_sha256"] == pin["manifest_sha256"] and sha(result) == pin["sha256"]
                and load(result)["registered_files"] == hashes, "conflicting registration")
        return pin
    if requested.exists():
        require(load(requested)["registered_files"] == hashes, "prior registration request differs")
    else:
        save(requested, {
            "trial_id": "r-study-registration-v1", "experiment_id": "reconstruction-recovery-v1",
            "registered_files": hashes, "parent": OLD_FREEZE,
            "resource_baseline": {"utc": utc(), "active_logical_bytes": _capacity.bytes_on_tree(store),
                                  "volume_free_bytes": shutil.disk_usage(store).free},
            "recording": {"expected_output_bytes": 262144},
        })
    old_path = store / "runs" / OLD_FREEZE["run_id"] / OLD_FREEZE["member"]
    cmd = [sys.executable, "-m", "evidence_lens.recorder", "--source-root", str(project),
           "--root", str(store), "--kind", "archive_verify", "--config", str(requested),
           "--parent", OLD_FREEZE["run_id"] + "=" + OLD_FREEZE["manifest_sha256"],
           "--input", "previous=" + str(old_path), "--input-parent", "previous=" + OLD_FREEZE["run_id"],
           "--", "{python}", "{source}/scripts/recovery_register.py", "worker",
           "--source", "{source}", "--config", "{config}", "--previous", "{input:previous}", "--out", "{out}"]
    outcome = subprocess.run(cmd, cwd=project, capture_output=True, text=True)
    if outcome.returncode:
        raise RuntimeError((outcome.stdout + outcome.stderr)[-2000:])
    run = json.loads(outcome.stdout.strip().splitlines()[-1])
    root = store / "runs" / run["run_id"]
    proof = verify_bundle(root)
    pin = {"run_id": run["run_id"], "manifest_sha256": proof["manifest_sha256"],
           "member": "payload/registration.json", "sha256": sha(root / "payload/registration.json")}
    save(selected, pin)
    return pin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    entry = actions.add_parser("register")
    entry.add_argument("--store", type=Path, required=True)
    entry.add_argument("--project", type=Path, default=Path.cwd())
    worker = actions.add_parser("worker")
    for arg in ("source", "config", "previous", "out"):
        worker.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    if args.action == "register":
        print(json.dumps(register(args.store.resolve(), args.project.resolve())))
    else:
        verify_registration(args.source, load(args.config), args.previous, args.out)
