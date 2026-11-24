"""Record complete byte, dependency and CAS checks across the two historical research stores."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from evidence_lens._bundle import encoded, load, sha
from evidence_lens._verify import parent_directory, require, verify_bundle, verify_dependencies
from t0_reconstruct_history import missing_parents
from t0_records import record


def original(batch_one, historical, rid, digest):
    if (batch_one / "runs" / rid).is_dir():
        return batch_one, batch_one / "runs" / rid
    return historical, parent_directory(
        historical, dict(run_id=rid, manifest_sha256=digest), historical.parent / "cold-shelf"
    )


def historical_boundary(store, registries):
    tickets = [job["ticket"] for registry in registries for job in load(registry)["jobs"]]
    return missing_parents(store, tickets)


def audit(store, prior, old, config, registries, out):
    external = historical_boundary(store, registries)
    require(
        external
        and external == config["historical_parents"]
        and [sha(p) for p in registries] == config["dispatch_registry_sha256"],
        "Changed dispatch lineage",
    )
    data = {
        "schema": "el-b2-reconstruction-history-1",
        "run_id": os.environ["EL_RUN_ID"],
        "dispatch_registry_sha256": config["dispatch_registry_sha256"],
        **verify_external(prior, old, external),
    }
    (out / "history.json").write_bytes(encoded(data))


def verify_external(prior, old, external):
    """Verify every ancestor and shared asset at its actual batch-one or historical location."""
    digs, paths, owners, edges = {}, {}, {}, {}

    def collect(rid, digest, active):
        require(rid not in active, "Cyclic historical ancestry")
        if rid in digs:
            require(digs[rid] == digest, "Historical manifest conflict")
            return
        owner, path = original(prior, old, rid, digest)
        require(path.is_dir() and sha(path / "manifest.json") == digest, "Absent historical original " + rid)
        m = load(path / "manifest.json")
        require(m["run_id"] == rid, "Different historical run")
        digs[rid] = digest
        paths[rid] = path
        owners[rid] = owner
        parents = m["parents"] + ([m["retry_of"]] if m["retry_of"] else [])
        edges[rid] = len(parents)
        for p in parents:
            collect(p["run_id"], p["manifest_sha256"], active | {rid})

    for rid, digest in external.items():
        collect(rid, digest, set())

    def validate(rid):
        b = verify_bundle(paths[rid])
        require(b["manifest_sha256"] == digs[rid] and b["run"]["run_id"] == rid, "Historical bytes changed")
        return rid, b

    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = dict(pool.map(validate, paths))
    assets = {prior: set(), old: set()}
    for rid, b in verified.items():
        owner = owners[rid]
        verify_dependencies(b, owner, verified, verified_assets=assets[owner])
    require(
        all(verified[r]["run"]["execution_status"] == "succeeded" for r in external),
        "Failed external scientific parent",
    )
    return {
        "historical_parents": external,
        "checked_manifests": digs,
        "verified_runs": len(digs),
        "verified_edges": sum(edges.values()),
        "verified_shared_assets": sum(map(len, assets.values())),
        "verified_historical_member_bytes": sum(b["bytes"] for b in verified.values()),
    }


def register(store, prior, old, name, registries):
    parents = {}
    raws = {}
    for i, f in enumerate(registries):
        raws["registry" + str(i)] = f
        reg = load(f)
        require(reg["schema"] == "el-b2-windows-batch-1" and reg["jobs"], "Invalid dispatch registry")
        for j in reg["jobs"]:
            parents["ticket" + str(len(parents))] = j["ticket"]
    cfg = {
        "experiment_id": "b2-cross-store-reconstruction-readback",
        "trial_id": "b2-reconstruction-history-" + name,
        "historical_parents": historical_boundary(store, registries),
        "dispatch_registry_sha256": [sha(x) for x in registries],
    }
    cmd = [
        "{python}",
        "{source}/scripts/b2_reconstruct_history.py",
        "audit",
        "--store",
        str(store),
        "--batch-one",
        str(prior),
        "--history",
        str(old),
        "--config",
        "{config}",
        "--out",
        "{out}",
    ]
    for role in raws:
        cmd += ["--registry", "{input:" + role + "}"]
    return record(
        store,
        old,
        "b2-reconstruction-history-" + name,
        "archive_verify",
        parents,
        cfg,
        cmd,
        "payload/history.json",
        2**20,
        raws,
    )


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=["register", "audit"])
    for n in ("store", "batch-one", "history"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--registry", type=Path, action="append", required=True)
    p.add_argument("--name")
    p.add_argument("--config", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    s = a.store.resolve()
    o = a.history.resolve()
    b = a.batch_one.resolve()
    if a.operation == "register":
        print(register(s, b, o, a.name, a.registry))
    else:
        audit(s, b, o, load(a.config), a.registry, a.out)
