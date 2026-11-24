"""Receive an original Windows ZIP whose frozen lineage spans the old and new stores."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import shutil
import tempfile
import threading
import time
import zipfile

from evidence_lens import _capacity
from evidence_lens._bundle import freeze_tree, load, sha
from evidence_lens._dedup import compact_inputs
from evidence_lens._index import writer_gate
from evidence_lens._verify import parent_directory, require, verify_bundle, verify_dependencies
from evidence_lens.archive import make_receipt, project_receipt, remove_stage, unpack


_lock = threading.Lock()


@lru_cache(maxsize=2)
def historical_roots(store, history, rid, manifest_sha, member_sha):
    bundle = store / "runs" / rid
    verified = verify_bundle(bundle)
    run = verified["run"]
    require(run["run_kind"] == "archive_verify" and run["execution_status"] == "succeeded"
            and run["experiment_id"] == "t0-cross-store-archive-verification"
            and verified["manifest_sha256"] == manifest_sha, "Unverified cross-store history receipt")
    path = bundle / "payload/cross-store-audit.json"
    require(sha(path) == member_sha, "Changed cross-store proof")
    doc = load(path)
    config = load(bundle / "config.resolved.yaml")
    require(doc["schema"] == "el-t0-cross-store-audit-1" and doc["run_id"] == rid
            and doc["historical_parents"] == config["historical_parents"]
            and doc["verified_runs"] == len(doc["checked_manifests"])
            and doc["dispatch_registry_sha256"] == config["dispatch_registry_sha256"]
            == sha(store / "transport/Handoffs/t0-v1/frozen-12.json")
            and doc["freeze"] == config["freeze"], "Historical proof differs from authorized dispatch registry")
    anchors = {}
    for rid, digest in doc["historical_parents"].items():
        require(doc["checked_manifests"].get(rid) == digest, "Unverified historical boundary")
        root = parent_directory(history, {"run_id":rid,"manifest_sha256":digest}, history.parent / "cold-shelf")
        old = verify_bundle(root)
        require(old["manifest_sha256"] == digest and old["run"]["run_id"] == rid
                and old["run"]["execution_status"] == "succeeded", "Historical source does not match audit")
        anchors[rid] = old
    return doc["freeze"], anchors


def receive_cross_store(source, root, digest, policy, history, evidence, freeze):
    """Reuse the stock byte, dependency, duplicate, locking and receipt checks at the explicit store boundary."""
    with _lock:
        binding, anchors = historical_roots(root, history, evidence["run_id"],
                                             evidence["manifest_sha256"], evidence["sha256"])
    require(freeze == binding and evidence["member"] == "payload/cross-store-audit.json",
            "Return and audited history reference different freezes")
    started = time.monotonic()
    staging = root / "staging"
    staging.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=staging, prefix="t0-receive-"))
    copy = stage / "bundle"
    rid = manifest_sha = None
    source_size = source.stat().st_size
    metadata = {"source_kind":"zip", "container_sha256":digest, "source_name":source.name,
                "bytes":source_size, "historical_verification":evidence}
    with writer_gate(root):
        try:
            with zipfile.ZipFile(source) as archive:
                estimate = source_size + sum(i.file_size for i in archive.infolist())
            space = _capacity.check(root, estimate, _capacity.policy(policy, Path.cwd()))
            if not space["reason"]:
                incoming = stage / "incoming.zip"
                shutil.copyfile(source, incoming)
                require(sha(incoming) == digest, "Original Windows container SHA changed")
                copy.mkdir()
                unpack(incoming, copy)
                verified = verify_bundle(copy)
                rid, manifest_sha = verified["run"]["run_id"], verified["manifest_sha256"]
                require(rid not in anchors, "Incoming identity collides with a historical root")
                # The six old roots and their complete old-store DAG were verified from original
                # bundles by the named history recorder. All new ancestors are read in the new root.
                upstreams = dict(anchors)
                verify_dependencies(verified, root, upstreams)
                local = {k:v for k,v in upstreams.items() if k not in anchors}
                out = root / "runs" / rid
                reuse = {}
                if out.exists():
                    require(verify_bundle(out)["manifest_sha256"] == manifest_sha,
                            "Conflict with the existing immutable run")
                    status = "identical"
                else:
                    out.parent.mkdir(exist_ok=True)
                    freeze_tree(copy)
                    reuse = compact_inputs(root, copy, verified, local)
                    copy.rename(out)
                    status = "accepted"
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, zipfile.BadZipFile) as exc:
            quarantined = root / "quarantine" / stage.name
            quarantined.parent.mkdir(exist_ok=True)
            stage.rename(quarantined)
            path, receipt = make_receipt(root, "transfer", run_id=rid, manifest_sha256=manifest_sha,
                status="quarantined", reason=str(exc), quarantine_path=quarantined.relative_to(root).as_posix(),
                elapsed_seconds=time.monotonic()-started, **metadata)
            project_receipt(root, path)
            return receipt
        if space["reason"]:
            remove_stage(stage)
            path, receipt = make_receipt(root, "transfer", run_id=None, manifest_sha256=None,
                status="deferred_capacity", reason=space["reason"], storage_preflight=space,
                elapsed_seconds=time.monotonic()-started, **metadata)
            project_receipt(root, path)
            return receipt
        path, receipt = make_receipt(root, "transfer", run_id=rid, manifest_sha256=manifest_sha,
            status=status, reason=None, local_input_storage=reuse, elapsed_seconds=time.monotonic()-started, **metadata)
        project_receipt(root, path, out, verified, local)
        remove_stage(stage)
        return receipt
