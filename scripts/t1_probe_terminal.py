"""Seal the frozen probe's original 20,000-point guard failure and its five unavailable decisions."""

import argparse
import hashlib
import itertools
import json
import os
import zipfile
from pathlib import Path

from make_handoff import publish
from t0_reconstruct import short
from t1_access import CONDITIONS, OBJECTS, POOL, S0, STRATEGIES
from t1_receive import import_zip
from t1_study import DEFAULT, authority, bind, locate, selected, stage

from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens._snapshot import commit_shared
from evidence_lens._verify import require, verify_bundle

REASON = "The frozen probe requires exactly 20000 final Gaussians. The eligible, successfully fitted M2 endpoint had fewer. No query/visibility support or registered geometric fallback was computed."


def planned_random(strategy, obj):
    if not strategy.startswith("random-"):
        return None
    seed = int(strategy.split("-")[-1])
    pairs = list(itertools.combinations(POOL, 2))
    return list(
        min(pairs, key=lambda p: (hashlib.sha256(f"random|{seed}|{obj}|{p[0]}|{p[1]}".encode()).digest(), p))
    )


def confirm(roots, obj, returns):
    store = roots[0]
    entry, _, f = authority(roots)
    m2 = selected(store, "t1-readback-" + obj + "-M2-fixed")
    original = load(locate(roots, m2))
    require(
        original["verified"] and original["status"] == "succeeded" and original["completed_steps"] == 5000,
        "Only a separately successful M2 may enter this gate diagnosis",
    )
    g = original["origins"]["generation"]
    checkpoint = bind(roots, g, "payload/checkpoints/step-5000.pt")
    cam = f["sources"]["query_cameras"]
    folder = returns / "probes" / obj
    reply = load(folder / "completed.json")
    rid = reply["run_id"]
    require(
        reply["schema"] == "el-t1-probe-return-1"
        and reply["status"] == "failed"
        and reply["checkpoint_producer"] == g["run_id"]
        and reply["checkpoint_sha256"] == checkpoint["sha256"]
        and reply["cameras"] == cam
        and reply["zip"] == rid + ".zip"
        and sha(folder / reply["zip"]) == reply["zip_sha256"],
        "Original failed probe does not match the certified M2",
    )
    kept = store / "transport/Returns/t1-v1/probes" / obj
    for n in ("completed.json", reply["zip"]):
        publish(kept / n, (folder / n).read_bytes())
    with zipfile.ZipFile(kept / reply["zip"]) as z:
        cfg = json.loads(z.read("config.resolved.yaml"))
        req = json.loads(z.read("inputs.json"))
        run = json.loads(z.read("run.json"))
        error = z.read("stderr.log").decode()
        code = z.read("source/scripts/b2_probe_worker.py")
        require(
            cfg["object_id"] == obj
            and cfg["pool"] == POOL
            and cfg["entry"] == entry
            and cfg["input_roles"] == ["checkpoint", "cameras"]
            and cfg["checkpoint_sha256"] == checkpoint["sha256"]
            and cfg["cameras_sha256"] == cam["sha256"]
            and cfg["trial_id"] == "t1-probe-" + obj
            and cfg["recipe_sha256"] == original["recipe_sha256"]
            and run["execution_status"] == "failed"
            and run["run_kind"] == "render"
            and run["run_id"] == rid
            and "payload/render.json" not in z.namelist()
            and not any(n.startswith("payload/") and n.endswith(".npz") for n in z.namelist())
            and 'require(len(ck["params"]["means"]) == 20000, "checkpoint point count differs")' in error
            and error.rstrip().endswith("BundleError: checkpoint point count differs")
            and hashlib.sha256(code).hexdigest() == f["selection_code"]["scripts/b2_probe_worker.py"]
            and 0 < original["checkpoints"][-1]["N"] < 20000,
            "Unclassified probe failure; infrastructure must be diagnosed separately",
        )
        supplied = {"checkpoint": checkpoint, "cameras": cam}
        for i in req["inputs"]:
            known = supplied[i["role"]]
            require(
                i["parent_run_id"] == known["run_id"]
                and len(i["files"]) == 1
                and i["files"][0]["sha256"] == known["sha256"],
                "Changed input to failed scientific process",
            )
            if i["storage"] == "shared":
                commit_shared(locate(roots, known), store, known["sha256"])
    if not (store / "runs" / rid).is_dir():
        import_zip(roots, kept / reply["zip"], reply["zip_sha256"], rid, reply["manifest_sha256"], entry, req)
    v = verify_bundle(store / "runs" / rid)
    require(
        v["manifest_sha256"] == reply["manifest_sha256"] and v["run"]["execution_status"] == "failed",
        "Failed run not preserved in original form",
    )
    failure = {**short(reply), "member": "run.json", "sha256": sha(store / "runs" / rid / "run.json")}
    pins = {"entry": entry, "M2": m2, "geometry": cam, "kit": selected(store, "t1-probe-kit")}
    config = {
        "object_id": obj,
        "actual_final_gaussians": original["checkpoints"][-1]["N"],
        "failed_probe": failure,
        "checkpoint_sha256": checkpoint["sha256"],
        "generation": g,
        "failure_rule_sha256": f["selection_code"]["scripts/b2_probe_worker.py"],
    }
    cmd = [
        "{python}",
        "-B",
        "{source}/scripts/t1_probe_terminal.py",
        "worker",
        "--config",
        "{config}",
        "--out",
        "{out}",
        "--summary",
        "{input:reply}",
        "--error",
        "{input:diagnostic}",
        "--manifest",
        "{input:manifest}",
    ]
    # An unsuccessful numerical run cannot be a successful scientific parent of a recorder job.
    # Preserve its original pin and pass authenticated copies of its failure evidence as files.
    diagnosis = store / "work/failed-probe-receipts" / obj
    for filename in ("stderr.log", "manifest.json"):
        publish(diagnosis / filename, (store / "runs" / rid / filename).read_bytes())
    previous = store / "work/controller" / ("t1-decisions-" + obj) / "attempt-01.jsonl"
    if previous.is_file():
        old = json.loads(previous.read_text().splitlines()[-1])
        require(
            old["failure_reason"] == "BundleError: input diagnostic references an undeclared parent",
            "Unexpected earlier verification attempt",
        )
        config["previous_evidence_ingest_failure"] = {
            k: old[k] for k in ("run_id", "manifest_sha256", "failure_reason")
        }
    result = stage(
        roots,
        "t1-probe-terminal-" + obj,
        "archive_verify",
        pins,
        config,
        cmd,
        "choices.json",
        65000,
        {
            "reply": kept / "completed.json",
            "diagnostic": diagnosis / "stderr.log",
            "manifest": diagnosis / "manifest.json",
        },
    )
    publish(store / "work/controller" / ("t1-decisions-" + obj) / "selected.json", encoded(result))
    return result


def worker(cfg, summary, error, manifest, out):
    failure = cfg["failed_probe"]
    reply = load(summary)
    obj = cfg["object_id"]
    require(
        sha(manifest) == failure["manifest_sha256"] == reply["manifest_sha256"]
        and reply["run_id"] == failure["run_id"]
        and reply["status"] == "failed"
        and "checkpoint point count differs" in error.read_text()
        and obj in OBJECTS,
        "Unverified scientific terminal",
    )
    manifest_files = {r["path"]: r["sha256"] for r in load(manifest)["files"]}
    require(sha(error) == manifest_files["stderr.log"], "Changed original error")
    common = {
        "run_id": os.environ["EL_RUN_ID"],
        "object_id": obj,
        "status": "failed",
        "failed_probe": failure,
        "reason": REASON,
        "actual_final_gaussians": cfg["actual_final_gaussians"],
        "required_gaussians": 20000,
        "m2_checkpoint_sha256": cfg["checkpoint_sha256"],
        "generation": cfg["generation"],
        "pool": POOL,
        "legal_photo_ids_before_selection": S0,
        "completed_pair_decisions": 0,
        "candidate_photo_access_authorized": False,
        "actual_candidate_observations": 0,
        "candidate_range_requests": 0,
        "evaluation_reference_reads": 0,
    }
    hashes = {}
    for name in STRATEGIES:
        save(
            out / (name + ".json"),
            dict(
                schema="el-t1-strategy-unavailable-1",
                **common,
                strategy_id=name,
                selected_official_ids=None,
                registered_random_pair_only=planned_random(name, obj),
                fallback_reason=None,
                random_pair_note="The registered seed fixates a pair without images; the frozen joint selector did not execute."
                if name.startswith("random-")
                else None,
            ),
        )
        hashes[name] = sha(out / (name + ".json"))
    save(
        out / "choices.json",
        dict(
            schema="el-t1-probe-terminal-1",
            **common,
            results={n: None for n in STRATEGIES},
            selections=hashes,
            unavailable_conditions=CONDITIONS[1:],
            numeric_guard_code_sha256=cfg["failure_rule_sha256"],
            frozen_scientific_attempts=1,
            replacements=0,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("confirm", "worker"))
    p.add_argument("--store", type=Path, default=DEFAULT)
    p.add_argument("--object", choices=OBJECTS)
    for k in ("returns", "out", "config", "summary", "error", "manifest"):
        p.add_argument("--" + k, type=Path)
    a = p.parse_args()
    roots = a.store.resolve(), a.store.resolve() / "history"
    if a.action == "confirm":
        print(json.dumps(confirm(roots, a.object, a.returns)))
    else:
        worker(load(a.config), a.summary, a.error, a.manifest, a.out)
