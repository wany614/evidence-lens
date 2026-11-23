"""Record explicit recovery study inputs and a versioned worker using the existing experiment recorder."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from dtu_eval import read_pinned
from evidence_lens.formal_inputs import require, compact_pin
from evidence_lens.scoring_io import file_hash
from recovery_eval_io import authority, permit_frozen, original
from vision_sources import find_run, signed
from vision_record import publish

STAGES = {
    "baseline": ("evaluation", "recovery_eval_baseline", "baseline.json", 2_000_000),
    "quality": ("evaluation", "recovery_eval_quality", "quality.json", 120_000_000),
    "select": ("report", "recovery_eval_select", "selection.json", 2_000_000),
    "adapt": ("derivation", "recovery_score_case", "case-provenance.json", 22_000_000),
    "score": ("model_score", "dtu_score", "scores.json", 15_000_000),
    "risk": ("evaluation", "recovery_eval_risk", "risk.json", 27_000_000),
    "audit": ("archive_verify", "recovery_eval_audit", "audit.json", 1_000_000),
    "report": ("report", "recovery_eval_report", "report.json", 150_000_000),
}

CASE_ROLES = {
    "object_id",
    "condition",
    "original_case",
    "original_provenance",
    "references",
    "reference_views",
}


def check_input_roles(stage, req):
    """Resolve only authorized artifacts after inspecting the request's role names."""
    if stage in ("adapt", "score", "risk", "audit"):
        require(
            req.get("object_id") in ("set1", "set6") and req.get("condition") in ("M2", "M4"),
            "frozen risk evaluations are limited to the four registered DTU cells",
        )
    if stage == "quality":
        require(
            set(req) <= CASE_ROLES | {"schema", "protocol", "recipe", "recipe_id", "outputs", "freeze"},
            "checkpoint quality request includes an unregistered input role",
        )
    elif stage == "baseline":
        require(
            set(req) <= {"schema", "protocol", "cells", "recipe", "freeze"}
            and all(set(c) <= CASE_ROLES for c in req["cells"]),
            "original RGB comparator contains an unregistered role",
        )
    elif stage == "select":
        require(
            set(req) == {"schema", "protocol", "quality"}
            and all(p["member"] == "payload/quality.json" for p in req["quality"]),
            "development selection accepts just named reconstruction quality records",
        )
    elif stage == "adapt":
        require(
            set(req)
            == {
                "schema",
                "object_id",
                "condition",
                "protocol",
                "recipe",
                "freeze",
                "original_case",
                "outputs",
                "depth_probe",
            },
            "source-only adapter includes an unregistered input role",
        )
    elif stage == "score":
        require(
            req.get("schema") == "el-recovery-score-request-1"
            and set(req)
            == {
                "schema",
                "object_id",
                "condition",
                "protocol",
                "recipe",
                "freeze",
                "case",
                "provenance",
                "release",
            },
            "source-only model scoring permits exactly declared deployment roles",
        )


def pins(obj):
    if isinstance(obj, dict):
        if {"run_id", "manifest_sha256", "member", "sha256"} <= obj.keys():
            yield obj
        else:
            for x in obj.values():
                yield from pins(x)
    elif isinstance(obj, list):
        for x in obj:
            yield from pins(x)


def needs_freeze(stage, req):
    return (
        stage in ("adapt", "score", "risk", "audit", "report")
        or req.get("object_id") == "set6"
        or any(cell.get("object_id") == "set6" for cell in req.get("cells", ()))
    )


def run(a):
    project = a.project.resolve()
    store = a.store.resolve()
    roots = (store, a.shelf.resolve() if a.shelf else store.parent / "cold-shelf")
    req = signed(a.request, a.request_sha256)
    check_input_roles(a.action, req)
    authority(roots, req)
    if needs_freeze(a.action, req):
        permit_frozen(roots, req)
    kind, module, product, estimate = STAGES[a.action]
    ph = a.request_sha256
    name = "recovery-" + a.action + "-" + ph[:14]
    out = a.state.resolve() / name
    out.mkdir(parents=True, exist_ok=True)
    roles = {"request": (a.request.resolve(), None)}
    parents = {}
    for k, pin in enumerate(pins(req)):
        r, p = read_pinned(roots, pin)
        parents[r.run["run_id"]] = compact_pin(pin)
        roles["original" + format(k, "03")] = (
            p,
            r.run["run_id"] if pin["member"].startswith("payload/") else None,
        )
    if a.action == "score":
        _, cp = read_pinned(roots, req["case"])
        _, _, pr = original(roots, req["provenance"])
        _, rp = read_pinned(roots, req["release"])
        require(
            pr["schema"] == "el-recovery-deployment-case-1"
            and pr["case_sha256"] == req["case"]["sha256"]
            and pr["freeze"] == req["freeze"]
            and pr["object_id"] == req["object_id"]
            and pr["condition"] == req["condition"],
            "deployment provenance changed",
        )
        from dtu_score import RELEASE_SHA

        require(
            req["release"]["sha256"] == RELEASE_SHA and a.cpu_python,
            "need original CPU package and interpreter",
        )
        worker = [
            str(a.cpu_python.absolute()),
            "-I",
            "-B",
            "{source}/scripts/dtu_score.py",
            "--cpu-dir",
            str(rp.parent),
            "--release-sha256",
            RELEASE_SHA,
            "--case",
            str(cp.parent),
            "--case-sha256",
            req["case"]["sha256"],
            "--threads",
            "4",
        ]
    else:
        worker = [
            "{python}",
            "{source}/scripts/" + module + ".py",
            "--store",
            str(store),
            "--shelf",
            str(roots[1]),
            "--request",
            "{input:request}",
            "--request-sha256",
            ph,
        ]
    cfg = {
        "experiment_id": "reconstruction-recovery-v1",
        "trial_id": name,
        "action": a.action,
        "request_sha256": ph,
        "upstream": parents,
        "object_id": req.get("object_id"),
        "condition_id": req.get("condition"),
        "recording": {"expected_output_bytes": estimate},
    }
    with (out / "lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        publish(out / "config.json", cfg)

        def accept(item):
            pin = compact_pin(item)
            r = find_run(roots, pin)
            require(
                r.run["execution_status"] == "succeeded" and r.run["trial_id"] == name,
                "unexpected selected run",
            )
            p = r.open("payload/" + product)
            v = {**pin, "member": "payload/" + product, "sha256": file_hash(p)}
            publish(out / "selected.json", v)
            return v

        if (out / "selected.json").exists():
            return accept(json.loads((out / "selected.json").read_bytes()))
        history = sorted(out.glob("attempt-*.jsonl"))
        if history:
            prev = json.loads(history[-1].read_text().strip().splitlines()[-1])
            if prev["execution_status"] == "succeeded":
                return accept(prev)
            require(
                a.retry_of == prev["run_id"] and len(history) == 1,
                "failed attempt requires its explicit one-time retry or an independently documented revision",
            )
        else:
            require(not a.retry_of, "retry cannot precede the original attempt")
        cmd = [
            sys.executable,
            "-m",
            "evidence_lens.recorder",
            "--root",
            str(store),
            "--source-root",
            str(project),
            "--storage-policy",
            str(project / "configs/storage.json"),
            "--kind",
            kind,
            "--config",
            str(out / "config.json"),
        ]
        for r, p in sorted(parents.items()):
            cmd += ["--parent", r + "=" + p["manifest_sha256"]]
        for k, (p, r) in roles.items():
            cmd += ["--input", k + "=" + str(p)]
            if r:
                cmd += ["--input-parent", k + "=" + r]
        if a.retry_of:
            cmd += ["--retry-of", a.retry_of]
        log = out / ("attempt-%02d.jsonl" % (len(history) + 1))
        with log.open("xb") as f:
            proc = subprocess.run(
                [*cmd, "--", *worker, "--output", "{out}"],
                cwd=project,
                stdout=f,
                stderr=f,
                env={**os.environ, "PYTHONPATH": str(project / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
            )
        last = log.read_text().strip().splitlines()[-1]
        require(proc.returncode == 0, "recorded stage failed: " + last[-750:])
        return accept(json.loads(last))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=STAGES)
    for n in ("project", "store", "state", "request"):
        p.add_argument("--" + n, required=True, type=Path)
    p.add_argument("--shelf", type=Path)
    p.add_argument("--cpu-python", type=Path)
    p.add_argument("--retry-of")
    p.add_argument("--request-sha256", required=True)
    print(json.dumps(run(p.parse_args()), sort_keys=True))
