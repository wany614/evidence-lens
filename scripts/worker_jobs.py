"""Durable formal Windows stage dispatch and transfer receipts."""

import hashlib
import zipfile
from pathlib import Path

from evidence_lens._bundle import RUN_ID, load, save, sha, utc
from evidence_lens import _capacity
from evidence_lens._index import writer_gate
from evidence_lens._verify import require, verify_bundle
from evidence_lens.archive import pack
from evidence_lens.recorder import worker_started


def frozen_json(path, record):
    if path.exists():
        require(load(path) == record, f"conflicting frozen formal job file: {path.name}")
    else:
        save(path, record)


def matching_runs(store, job, job_sha, kind, retry):
    found, pending = [], []
    for tree in ("runs", "work"):
        for path in sorted((store / tree).glob("*/run.json")):
            if not RUN_ID.fullmatch(path.parent.name):
                continue
            meta = load(path)
            if (meta["trial_id"], meta["experiment_id"], meta["run_kind"]) != (job["trial_id"], job["experiment_id"], kind):
                continue
            require(load(path.parent / "config.resolved.yaml").get("formal_handoff_sha256") == job_sha,
                    "trial already has a different formal handoff")
            if (meta.get("requested_retry_of_run_id") or meta["retry_of_run_id"]) == retry:
                (pending if tree == "work" else found).append(path.parent)
    return found, pending


def check_generation(store, rid, job, digest):
    require(isinstance(rid, str) and RUN_ID.fullmatch(rid), "render requires an explicit generation run ID")
    path = store / "runs" / rid
    record = verify_bundle(path)
    meta = record["run"]
    require(meta["run_id"] == rid and meta["run_kind"] == "generation" and meta["execution_status"] == "succeeded" and
            meta["trial_id"] == job["trial_id"] and meta["experiment_id"] == job["experiment_id"] and
            record["inputs"]["parents"] == [job["reference"]] and
            load(path / "config.resolved.yaml").get("formal_handoff_sha256") == digest,
            "generation is not a successful producer for this exact formal handoff")
    generation = load(path / "payload/generation.json")
    require(generation["producer_run_id"] == rid and generation["camera_sha256"] == job["files"]["cameras.json"] and
            generation["input_sha256"] == job["files"]["input.png"] and
            generation["ply_sha256"] == record["artifacts"]["payload/scene.ply"]["sha256"], "invalid generation provenance")
    return {"run_id": rid, "manifest_sha256": record["manifest_sha256"]}


def outgoing_receipt(store, outgoing, path, job_sha):
    meta = load(path / "run.json")
    rid = meta["run_id"]
    package = outgoing / (rid + ".zip")
    receipt_path = outgoing / (rid + ".receipt.json")
    manifest = sha(path / "manifest.json")
    if receipt_path.exists():
        receipt = load(receipt_path)
        require(receipt["handoff_sha256"] == job_sha and receipt["run"] == rid and receipt["manifest_sha256"] == manifest and
                receipt["package"] == package.name and receipt["bytes"] == package.stat().st_size and
                receipt["sha256"] == sha(package), "outgoing run, transfer ZIP or durable receipt changed")
        return receipt
    if not package.exists():
        settings = load(path / "config.resolved.yaml")["recording"]["storage_limits"]
        # Include return containers on this project volume while they are staged.
        needed = _capacity.bytes_on_tree(path) + 65536
        space = _capacity.check(store, needed, settings)
        external = 0 if outgoing.resolve().is_relative_to(store.resolve()) else _capacity.bytes_on_tree(outgoing)
        require(not space["reason"] and space["active_bytes"] + external + needed <= settings["local_active_batch_target_bytes"],
                "sealed computation retained; outgoing packaging deferred by configured disk limits")
        packed = pack(path, package, store)
        digest = packed["container_sha256"]
    else:
        # Pack may have published cleanly immediately before a controller crash.
        # Compare all bytes to the sealed run before issuing its first receipt.
        checked = verify_bundle(path)
        members = {r["path"]: r["sha256"] for r in checked["manifest"]["files"]}
        members.update({"manifest.json": manifest, "completion.json": sha(path / "completion.json")})
        with zipfile.ZipFile(package) as z:
            require(sorted(z.namelist()) == sorted(members), "interrupted outgoing pack has different members")
            for name, expected in members.items():
                with z.open(name) as source:
                    require(hashlib.file_digest(source, "sha256").hexdigest() == expected, f"changed packed member {name}")
        digest = sha(package)
    receipt = {"schema": "el-worker-return-1", "handoff_sha256": job_sha, "run": rid, "manifest_sha256": manifest,
               "package": package.name, "bytes": package.stat().st_size, "sha256": digest, "status": meta["execution_status"],
               "trial_id": meta["trial_id"], "run_kind": meta["run_kind"], "exit_code": meta["exit_code"]}
    frozen_json(receipt_path, receipt)
    return receipt


def dispatch(store, outgoing, job, digest, kind, generation, retry, resume_preflight, launch):
    """Reserve before spawn; recover a sealed package if the controller died after launch."""
    trial_dir = store / "formal" / job["identity"]["dataset_hash"] / job["trial_id"]
    trial_dir.mkdir(parents=True, exist_ok=True)
    # This is a per-trial lock, independent of the archive/recorder store writer.
    with writer_gate(trial_dir):
        frozen_json(trial_dir / "handoff.json", {"sha256": digest, "reference": job["reference"], "readback": job["readback"]})
        stage = trial_dir / kind
        stage.mkdir(exist_ok=True)
        frozen_json(stage / "parents.json", {"reference": job["reference"], "generation": generation})
        found, pending = matching_runs(store, job, digest, kind, retry)
        require(not pending, "formal stage has a recorder package still in work; wait for sealing before resuming")
        if not retry:
            # A successful linked scientific retry completes the original queue item.
            for original in found:
                retries, active_retries = matching_runs(store, job, digest, kind, original.name)
                require(not active_retries, "linked scientific retry is still in progress")
                successful = [r for r in retries if load(r / "run.json")["execution_status"] == "succeeded"]
                require(len(successful) <= 1, "multiple successful linked retries found")
                if successful:
                    return outgoing_receipt(store, outgoing, successful[0], digest)
        if retry:
            require(RUN_ID.fullmatch(retry), "invalid scientific retry ID")
            original = store / "runs" / retry
            attempted = verify_bundle(original)["run"]
            require(attempted["run_kind"] == kind and attempted["trial_id"] == job["trial_id"] and
                    attempted["execution_status"] != "succeeded" and not attempted["retry_of_run_id"] and worker_started(original, attempted) and
                    load(original / "config.resolved.yaml").get("formal_handoff_sha256") == digest,
                    "retry must name this formal stage's failed scientific attempt")
        key = retry or "initial"
        # A preflight may be repaired without consuming the scientific retry. Its
        # explicit run ID records why a new submission was requested.
        if resume_preflight:
            require(RUN_ID.fullmatch(resume_preflight) and resume_preflight in {p.name for p in found},
                    "--resume-preflight must name an existing matching preflight failure")
            failed = store / "runs" / resume_preflight
            require(not worker_started(failed, load(failed / "run.json")), "a scientific failure requires --retry-of")
        scientific = [p for p in found if worker_started(p, load(p / "run.json"))]
        require(len(scientific) <= 1, "multiple recorded scientific attempts exist for this dispatch")
        if scientific:
            return outgoing_receipt(store, outgoing, scientific[0], digest)
        if found and not resume_preflight:
            return outgoing_receipt(store, outgoing, found[-1], digest)
        if resume_preflight:
            require(found[-1].name == resume_preflight, "name the most recent preflight failure to try again")
            key += "-after-" + resume_preflight
        marker = stage / (key + ".json")
        require(not marker.exists(), "dispatch reserved but no sealed recorder run found; inspect the recorded worker before relaunch")
        save(marker, {"requested_at_utc": utc(), "retry_of": retry, "after_preflight": resume_preflight})
        try:
            result = launch()
        except OSError:
            # Popen itself failed; no recorder was created by this request.
            marker.unlink()
            raise
        require(isinstance(result, dict) and RUN_ID.fullmatch(result["run_id"]), "recorder did not return a sealed package")
        path = store / "runs" / result["run_id"]
        require(path.resolve() == Path(result["path"]).resolve() and
                load(path / "config.resolved.yaml").get("formal_handoff_sha256") == digest,
                "recorder returned an unrelated package")
        return outgoing_receipt(store, outgoing, path, digest)
