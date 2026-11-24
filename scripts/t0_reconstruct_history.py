"""Validate external immutable parents and their original ancestry for a separately stored T0 study."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from evidence_lens._bundle import encoded, load, sha
from evidence_lens._verify import parent_directory, require, verify_bundle, verify_dependencies, verify_shared


def missing_parents(store, tickets):
    pending = list(tickets)
    known, external = {}, {}
    while pending:
        pin = pending.pop()
        rid, digest = pin["run_id"], pin["manifest_sha256"]
        if rid in known:
            require(known[rid] == digest, "One ID has two manifests")
            continue
        known[rid] = digest
        root = store / "runs" / rid
        if not root.is_dir():
            external[rid] = digest
            continue
        verified = verify_bundle(root)
        require(verified["manifest_sha256"] == digest, "Changed frozen dispatch lineage")
        pending.extend(verified["inputs"]["parents"])
        if verified["inputs"]["retry_of"]:
            pending.append(verified["inputs"]["retry_of"])
    return external


def historical_dag(history, shelf, external):
    pending = [{"run_id": k, "manifest_sha256": v} for k, v in external.items()]
    paths, digests, edges = {}, {}, {}
    while pending:
        ref = pending.pop()
        rid, digest = ref["run_id"], ref["manifest_sha256"]
        if rid in digests:
            require(digests[rid] == digest, "Conflicting historical manifest")
            continue
        root = parent_directory(history, ref, shelf)
        require(root.is_dir() and sha(root / "manifest.json") == digest, "Missing historical manifest " + rid)
        manifest = load(root / "manifest.json")
        require(manifest["run_id"] == rid, "Wrong historical run ID")
        paths[rid], digests[rid] = root, digest
        refs = [*manifest["parents"], *([manifest["retry_of"]] if manifest["retry_of"] else [])]
        edges[rid] = [p["run_id"] for p in refs]
        pending.extend(refs)
    # Independently establish acyclicity before using the verified in-memory cache.
    closed, active = set(), set()
    def visit(rid):
        require(rid not in active, "Cyclic historical ancestry")
        if rid not in closed:
            active.add(rid)
            for p in edges[rid]:
                visit(p)
            active.remove(rid)
            closed.add(rid)
    for rid in external:
        visit(rid)
    return paths, digests


def audit(store, history, config, registry, out):
    reg = load(registry)
    require(sha(registry) == config["dispatch_registry_sha256"]
            and reg["freeze"] == config["freeze"], "Wrong frozen registry")
    external = missing_parents(store, [j["ticket"] for j in reg["jobs"]])
    require(external == config["historical_parents"] and external, "Different historical boundary")
    paths, digests = historical_dag(history, history.parent / "cold-shelf", external)
    def check(item):
        rid, path = item
        verified = verify_bundle(path)
        require(verified["manifest_sha256"] == digests[rid] and verified["run"]["run_id"] == rid,
                "Changed historical original " + rid)
        return rid, verified
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = dict(pool.map(check, paths.items()))
    seen = set()
    for result in verified.values():
        verify_dependencies(result, history, verified, shelf=history.parent / "cold-shelf", verified_assets=seen)
    for rid in external:
        require(verified[rid]["run"]["execution_status"] == "succeeded", "Failed external parent")
    report = {
        "schema":"el-t0-cross-store-audit-1", "run_id":os.environ["EL_RUN_ID"],
        "dispatch_registry_sha256":sha(registry), "freeze":reg["freeze"],
        "historical_parents":external,"checked_manifests":dict(sorted(digests.items())),
        "verified_runs":len(paths), "verified_edges":sum(len(v["inputs"]["parents"]) + bool(v["inputs"]["retry_of"])
                                                          for v in verified.values()),
        "historical_bundle_member_bytes":sum(v["bytes"] for v in verified.values()),
        "verified_shared_assets":len(seen), "new_target_reference_files_opened":0,
    }
    (out / "cross-store-audit.json").write_bytes(encoded(report))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("store", "history", "registry", "config", "out"):
        parser.add_argument("--"+key,required=True,type=Path)
    a=parser.parse_args()
    audit(a.store.resolve(),a.history.resolve(),load(a.config),a.registry,a.out)
