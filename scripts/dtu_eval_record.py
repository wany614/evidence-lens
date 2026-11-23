"""Record one D1 evaluation, independent replay or numeric report from a SHA-pinned explicit request."""

import argparse
import json
from pathlib import Path

from evidence_lens.formal_inputs import require
from evidence_lens.scoring_io import file_hash
from dtu_score_record import Recorder
from dtu_eval import read_pinned
from vision_sources import signed

STAGES = {
    "evaluate": ("el-dtu-eval-request-1", "evaluation", "evaluation", "dtu_eval", 35_000_000),
    "replay": (
        "el-dtu-evaluation-replay-request-1",
        "archive_verify",
        "replay",
        "dtu_eval_replay",
        1_000_000,
    ),
    "diagnostic": (
        "el-dtu-source-budget-request-1",
        "evaluation",
        "diagnostic",
        "dtu_eval_diagnostic",
        1_000_000,
    ),
    "report": ("el-dtu-report-request-1", "report", "report", "dtu_report", 110_000_000),
}


def stage(a):
    store = a.store.resolve()
    rec = Recorder(a.project, store, a.shelf or store.parent / "cold-shelf", a.state)
    request = a.request.resolve()
    req = signed(request, a.request_sha256)
    schema, kind, receipt, script, estimate = STAGES[a.action]
    require(req["schema"] == schema, "request schema does not match selected stage")
    roles = {"request": (request, None)}
    parents = []

    def add(key, spec, success=True):
        r, p = read_pinned(rec.roots, spec, success=success)
        # A failed run is evidence. Recorder scientific parents must have succeeded.
        passed = r.run["execution_status"] == "succeeded"
        roles[key] = (p, r.pin if passed and spec["member"].startswith("payload/") else None)
        if passed:
            parents.append(r.pin)

    if a.action == "report":
        name = "joint-report"
        params = {"scientific_cells": len(req["cells"])}
        for i, cell in enumerate(req["cells"]):
            for key in ("evaluation", "replay"):
                add(f"{key}{i:02}", cell[key])
        for i, spec in enumerate(req.get("declared_stages", [])):
            add(f"resource{i:02}", spec["original"], success=spec.get("status", "succeeded") == "succeeded")
        for i, cell in enumerate(req.get("diagnostics", [])):
            add(f"budget{i:02}", cell["diagnostic"])
        for i, spec in enumerate(req.get("authorities", [])):
            add(f"authority{i:02}", spec)
    else:
        require(
            req["role"] in ("debug", "external")
            and req["object_id"] == {"debug": "set1", "external": "set6"}[req["role"]],
            "unregistered object role",
        )
        add("authority", req["authority"])
        if req["role"] == "external":
            auth = json.loads(roles["authority"][0].read_bytes())
            require(
                auth["schema"] == "el-dtu-reference-gate-1" and auth["object_ids"] == ["set6"],
                "external target references still sealed",
            )
        name = "-".join((a.action, req["role"], req["object_id"], req["condition"], req["arm"]))
        params = {
            "d1_role": req["role"],
            "object_id": req["object_id"],
            "condition_id": req["condition"],
            "arm": req["arm"],
        }
        for k in (
            "references",
            "case",
            "case_provenance",
            "readback",
            "scores",
            "release",
            "failure",
            "evaluation",
        ):
            if req.get(k):
                add(k, req[k], success=k != "failure")
    return rec.stage(
        name,
        kind,
        parents,
        roles,
        [
            "{python}",
            "{source}/scripts/" + script + ".py",
            "--store",
            str(rec.store),
            "--shelf",
            str(rec.shelf),
            "--request",
            "{input:request}",
            "--request-sha256",
            file_hash(request),
        ],
        receipt + ".json",
        estimate,
        params,
        a.retry_of,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=STAGES)
    for k in ("project", "store", "state", "request"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--shelf", type=Path)
    p.add_argument("--request-sha256", required=True)
    p.add_argument("--retry-of")
    print(json.dumps(stage(p.parse_args()), sort_keys=True))
