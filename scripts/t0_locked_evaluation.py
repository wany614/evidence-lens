"""Recover a rejected pre-opening check by running the immutable software-lock snapshot."""

import argparse
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from evidence_lens._bundle import encoded, load, sha
from evidence_lens._verify import require, verify_bundle
from make_handoff import publish
from t0_records import resolve


def recorded_stage(path):
    spec = importlib.util.spec_from_file_location("t0_record_from_sealed_source", path)
    module = importlib.util.module_from_spec(spec)
    # Never emit interpreter caches in a sealed source directory.
    before = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = before
    return module.record


def run(store, history, failed_id, references):
    store, history = store.resolve(), history.resolve()
    state = store / "work/controller"
    lock_pin = load(state / "opening-software-cross-store/selected.json")
    verified = verify_bundle(store / "runs" / lock_pin["run_id"])
    require(verified["manifest_sha256"] == lock_pin["manifest_sha256"], "software archive identity differs")
    lock = load(resolve((store, history), lock_pin))
    sealed = store / "runs" / lock_pin["run_id"] / "source"
    for filename, expected in lock["files"].items():
        require(sha(sealed / filename) == expected, "selected software snapshot differs: " + filename)
    if references:
        # This controller derives its entire project source tree from its own location in the recorded snapshot.
        return subprocess.run(
            [
                sys.executable,
                "-B",
                str(sealed / "scripts/t0_study.py"),
                "--store",
                str(store),
                "--history",
                str(history),
                "evaluate",
                "--references",
                str(references.resolve()),
            ],
            env={**os.environ, "PYTHONPATH": str(sealed / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
            check=True,
        ).returncode
    old = state / "collective-reference-opening"
    with (old / "sealed-source-controller.lock").open("a+b") as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX)
        if (old / "selected.json").exists():
            existing = load(old / "selected.json")
            resolve((store, history), existing)
            return existing
        failed = store / "runs" / failed_id
        check = verify_bundle(failed)
        require(
            check["run"]["execution_status"] == "failed"
            and check["run"]["trial_id"] == "t0-collective-reference-opening"
            and not any((failed / "payload").iterdir()),
            "initial attempt did not stop before granting access",
        )
        failed_cfg = load(failed / "config.resolved.yaml")
        sources = failed_cfg["sources"]
        matrix = state / "closed-matrix.json"
        require(
            sources["software_lock"] == lock_pin and failed_cfg["extra_sha256"]["matrix"] == sha(matrix),
            "different matrix or opening software",
        )
        guard = [
            "{python}",
            "{source}/scripts/t0_opening_software.py",
            "open",
            "--config",
            "{config}",
            "--store",
            str(store),
            "--history",
            str(history),
            "--request",
            "{input:matrix}",
            "--out",
            "{out}",
        ]
        original_record = recorded_stage(sealed / "scripts/t0_records.py")
        pin = original_record(
            store,
            history,
            "collective-reference-opening-locked-source",
            "archive_verify",
            sources,
            {
                "role": "independent reference release from exact sealed software tree",
                "previous_attempt": {"run_id": failed_id, "manifest_sha256": check["manifest_sha256"]},
                "failure_reason": "a live controller received an engineering change after the lock; the rejected attempt granted no target access",
            },
            guard,
            "payload/opening.json",
            400000,
            {"matrix": matrix},
        )
        publish(old / "selected.json", encoded(pin))
        return pin


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("store", "history"):
        p.add_argument("--" + n, type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--failed-opening-run")
    group.add_argument("--references", type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.store, a.history, a.failed_opening_run, a.references)))
