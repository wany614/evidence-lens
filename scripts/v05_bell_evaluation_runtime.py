"""Run the frozen bell evaluator with the original release profile supplied as declared data."""

import argparse
import json
import os
from pathlib import Path

from dual_records import bind, resolve, selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import verify_bundle
from evidence_lens_bundle import common
from evidence_lens_bundle.common import require
from v05_bell_common import TARGETS
from v05_bell_evaluate import PROCEDURE, PROGRAM, RESULT, opening_gate, run

PROFILE_HASH = "99587dd70b67a84debb81fa1dc84e93729c8980bc4531edecfb56112a15eb191"
RECORDED = "a-bell-landscape-comparison-packaged"


def existing_attempts(store):
    history = []
    for entry in sorted((store / "work/controller" / RESULT).glob("attempt-*.jsonl")):
        row = json.loads(entry.read_text().splitlines()[-1])
        path = Path(row["path"])
        require(path.parent in (store / "runs", store / "quarantine"), "unrecognized former run location")
        previous = verify_bundle(path)
        require(
            previous["manifest_sha256"] == row["manifest_sha256"]
            and previous["run"]["execution_status"] == "failed",
            "wrong original terminal",
        )
        reason = row["failure_reason"]
        if "missing or unsafe member: profile.json" in (path / "stderr.log").read_text():
            reason = "required release profile.json absent from Python-only source snapshot"
        history.append(
            {
                "run_id": row["run_id"],
                "manifest_sha256": row["manifest_sha256"],
                "disposition": row["producer_disposition"],
                "failure_reason": reason,
            }
        )
    return history


def control(store, retry):
    roots = (store, store / "history")
    pins = {
        k: selected(store, name)
        for k, name in (
            ("registration", "a-bell-register-six"),
            ("program", PROGRAM),
            ("opening", "a-bell-reference-opening"),
        )
    }
    opening = load(resolve(roots, pins["opening"]))
    opening_gate(opening, pins["registration"])
    pins.update({k: opening[k] for k in ("bundles_archive", "scores_archive")})
    pins["release_profile"] = bind(store, pins["bundles_archive"]["run_id"], "inputs/profile/profile.json")
    require(pins["release_profile"]["sha256"] == PROFILE_HASH, "different original release profile")
    pins["references"] = selected(store, "a-bell-references")
    refs = load(resolve(roots, pins["references"]))
    require(
        refs["opening"] == pins["opening"] and refs["role"] == "references", "unexpected reference parent"
    )
    for view in TARGETS:
        item = refs["products"]["landscape"][view]["display_fp32"]
        require(item["path"] == f"landscape/{view}-display-fp32.npz", "unregistered reference path")
        pin = bind(store, pins["references"]["run_id"], "payload/" + item["path"])
        require(pin["sha256"] == item["sha256"], "different independent reference")
        pins["reference_" + view] = pin
    details = {"operation": PROCEDURE, "prior_runtime_attempts": existing_attempts(store)}
    command = [
        "{python}",
        "-B",
        "{source}/scripts/v05_bell_evaluation_runtime.py",
        "worker",
        "--config",
        "{config}",
        "--out",
        "{out}",
    ]
    for key in pins:
        command += ["--input", key, "{input:" + key + "}"]
    return stage(
        store,
        RECORDED,
        "evaluation",
        pins,
        details,
        command,
        "payload/bell-comparison.json",
        3_000_000,
        retry=retry,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("evaluate", "worker"))
    for key in ("store", "config", "out"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--input", action="append", nargs=2, default=[])
    parser.add_argument("--retry")
    args = parser.parse_args()
    if args.action == "evaluate":
        print(json.dumps(control(args.store.resolve(), args.retry)))
    else:
        cfg = load(args.config)
        inputs = {k: Path(v) for k, v in args.input}
        source = inputs["release_profile"]
        require(
            source.name == "profile.json"
            and sha(source) == cfg["sources"]["release_profile"]["sha256"] == PROFILE_HASH,
            "unverified release calibration and model metadata",
        )
        common.ASSETS = source.parent
        save(
            args.out / "runtime-recovery.json",
            {
                "run_id": os.environ["EL_RUN_ID"],
                "prior_runtime_attempts": cfg["prior_runtime_attempts"],
                "release_profile": cfg["sources"]["release_profile"],
                "frozen_program": cfg["sources"]["program"],
                "opening": cfg["sources"]["opening"],
            },
        )
        run(inputs, cfg, args.out)
