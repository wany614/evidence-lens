"""Accept named batch-two Windows returns in frozen order and audit original five checkpoints and all views."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import zipfile

from evidence_lens import _capacity
from evidence_lens._bundle import asset_path, load, sha
from evidence_lens._index import writer_gate
from evidence_lens._verify import require, verify_bundle, verify_shared
from b2_reconstruct import PROJECT
from b2_reconstruct_history import original
from make_handoff import publish
from recovery_receive import restore_shared
from t0_records import bind, record, resolve
from t0_reconstruct import retain_original_return, short
from t0_reconstruct_intake import receive_verified_zip


@lru_cache(maxsize=4)
def historical_roots(store, prior, old, rid, manifest, member_sha):
    proof = verify_bundle(store / "runs" / rid)
    p = store / "runs" / rid / "payload/history.json"
    doc = load(p)
    config = load(store / "runs" / rid / "config.resolved.yaml")
    require(
        proof["manifest_sha256"] == manifest
        and proof["run"]["run_kind"] == "archive_verify"
        and proof["run"]["experiment_id"] == "b2-cross-store-reconstruction-readback"
        and proof["run"]["execution_status"] == "succeeded"
        and sha(p) == member_sha
        and doc["run_id"] == rid
        and doc["schema"] == "el-b2-reconstruction-history-1"
        and doc["historical_parents"] == config["historical_parents"]
        and doc["dispatch_registry_sha256"] == config["dispatch_registry_sha256"]
        and doc["verified_runs"] == len(doc["checked_manifests"]),
        "Invalid deep history proof",
    )
    anchors = {}
    for k, d in doc["historical_parents"].items():
        owner, path = original(prior, old, k, d)
        v = verify_bundle(path)
        require(
            v["manifest_sha256"] == doc["checked_manifests"][k] == d
            and v["run"]["run_id"] == k
            and v["run"]["execution_status"] == "succeeded",
            "Changed external parent",
        )
        anchors[k] = v
    return doc["dispatch_registry_sha256"], anchors


def accept(store, old, job, row, ledger, returns, proof, anchors, interpreter):
    trial = job["trial_id"]
    ticket = job["ticket"]
    rw = store / "runs" / ticket["run_id"]
    t = load(resolve((store, old), ticket))
    ready = load(rw / "payload/READY.json")
    require(
        job["trial_id"] == row["trial_id"] == ready["trial_id"] == t["trial_id"]
        and job["job_sha256"] == row["job_sha256"] == ready["job_sha256"] == t["job_sha256"]
        and job["package"] == t["package"]
        and t["package"]["sha256"] == row["package_sha256"],
        "Wrong returned dispatch",
    )
    outname = (
        "b2-c6-readback-" + ready["object_id"] + "-" + ready["condition"]
        if ready["cohort"]
        else "b2-d-readback-" + trial
    )
    selection = store / "work/controller" / outname / "selected.json"
    if selection.is_file():
        sel = load(selection)
        receipt = load(resolve((store, old), sel))
        saved = store / "transport/Returns/b2-v1" / trial / "completed.json"
        require(
            receipt["return_summary_sha256"] == row["completed_sha256"] == sha(saved)
            and receipt["job_sha256"] == row["job_sha256"]
            and load(saved)["runs"] == row["runs"]
            and receipt["status"] == row["status"],
            "Changed return after accepted readback",
        )
        return sel
    summary, staged, local = retain_original_return(
        t, trial, returns / trial, row["completed_sha256"], store / "transport/Returns/b2-v1"
    )
    require(
        row["runs"] == summary["runs"]
        and row["status"] == summary["status"]
        and len(summary["files"]) == row["files"]
        and sum(v["bytes"] for v in summary["files"].values()) + row["completed_bytes"] == row["return_bytes"]
        and (local / "completed.json").stat().st_size == row["completed_bytes"],
        "Independent Windows ledger differs",
    )
    parents = {
        "dispatch": ticket,
        "bridge": ready["freeze"],
        "init": ready["initialization"],
        "source": ready["source"],
    }
    origins = {k: short(v) for k, v in parents.items()}
    policy = PROJECT / "configs/storage.json"
    for doc, arch in staged:
        with zipfile.ZipFile(arch) as z:
            inputs = json.loads(z.read("inputs.json"))
            require(
                hashlib.sha256(z.read("manifest.json")).hexdigest() == doc["manifest_sha256"],
                "Changed original manifest",
            )
        need = sum(e["bytes"] for h, e in doc["shared_inputs"].items() if not asset_path(store, h).exists())
        with writer_gate(store):
            room = _capacity.check(store, need, _capacity.policy(policy, PROJECT))
            require(not room["reason"], str(room["reason"]))
            restore_shared(local, store, doc, inputs)
        tr = receive_verified_zip(arch, store, doc["sha256"], policy, anchors, proof)
        require(
            tr["status"] in ("accepted", "identical")
            and tr["run_id"] == doc["run"]
            and tr["manifest_sha256"] == doc["manifest_sha256"],
            "Original rejected: " + str(tr.get("reason")),
        )
        v = verify_bundle(store / "runs" / doc["run"])
        verify_shared(v, store)
        require(
            v["run"]["execution_status"] == doc["status"]
            and v["run"]["trial_id"] == trial
            and v["run"]["run_kind"] == doc["run_kind"],
            "Original stage status or identity differs",
        )
        origins[doc["run_kind"]] = {"run_id": doc["run"], "manifest_sha256": doc["manifest_sha256"]}
        if doc["status"] == "succeeded":
            parents[doc["run_kind"]] = bind(
                store,
                doc["run"],
                "payload/" + ("reconstruction.json" if doc["run_kind"] == "generation" else "outputs.json"),
            )
    parents["history"] = proof
    cfg = {
        "experiment_id": "b2-original-reconstruction-readback",
        "trial_id": trial + "-readback",
        "scene_family_id": ready["object_id"],
        "condition_id": ready["condition"],
        "originals": origins,
        "job_sha256": row["job_sha256"],
        "return_sha256": row["completed_sha256"],
        "ledger_sha256": sha(ledger),
    }
    cmd = [
        str(interpreter.absolute()),
        "-B",
        "{source}/scripts/b2_worker_readback.py",
        "--store",
        str(store),
        "--config",
        "{config}",
        "--out",
        "{out}",
    ]
    return record(
        store,
        old,
        outname,
        "archive_verify",
        parents,
        cfg,
        cmd,
        "payload/reconstruction-readback.json",
        14 * 1024**2,
        {"windowsledger": ledger, "summary": local / "completed.json"},
    )


def run(store, prior, old, registry, expected_batch, proof, returns, interpreter, workers):
    reg = load(registry)
    digest = sha(registry)
    require(reg["schema"] == "el-b2-windows-batch-1" and digest == expected_batch, "Wrong batch original")
    require(proof["member"] == "payload/history.json", "Wrong history proof member")
    registries, anchors = historical_roots(
        store, prior, old, proof["run_id"], proof["manifest_sha256"], proof["sha256"]
    )
    require(digest in registries, "The selected batch has no independent historical verification")
    jobs = reg["jobs"]
    tasks = []
    for n, job in enumerate(jobs, 1):
        ledger = returns / (registry.stem + f"-completed-{n:02d}.json")
        if not ledger.exists():
            break
        doc = load(ledger)
        require(
            doc["schema"] == "el-b2-windows-cumulative-1"
            and doc["batch_sha256"] == digest
            and doc["registered_jobs"] == len(jobs)
            and doc["completed"] == n == len(doc["jobs"])
            and [r["trial_id"] for r in doc["jobs"]] == [j["trial_id"] for j in jobs[:n]]
            and doc["jobs"][:-1] == [t[1] for t in tasks],
            "Divergent Windows cumulative ledger or order",
        )
        local = store / "transport/Returns/b2-v1" / ledger.name
        publish(local, ledger.read_bytes())
        tasks.append((job, doc["jobs"][-1], local))

    def one(task):
        job, row, ledger = task
        pin = accept(store, old, job, row, ledger, returns, proof, anchors, interpreter)
        print(job["trial_id"], json.dumps(pin, sort_keys=True), flush=True)
        return pin

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, tasks))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("store", "batch-one", "history", "registry", "proof", "returns"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--registry-sha", required=True)
    p.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    p.add_argument("--python", type=Path, default=PROJECT / ".venv-vision-cpu/bin/python")
    a = p.parse_args()
    run(
        a.store.resolve(),
        a.batch_one.resolve(),
        a.history.resolve(),
        a.registry,
        a.registry_sha,
        load(a.proof),
        a.returns,
        a.python,
        a.workers,
    )
