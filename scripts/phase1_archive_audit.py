"""Read and verify the historical study catalog against the active store, cold shelf and restore."""

import argparse
import graphlib
from collections import Counter
from pathlib import Path

from evidence_lens._bundle import asset_path, load, save, sha
from evidence_lens._verify import require, verify_bundle, verify_dependencies


def audit(catalog, store, shelf, expected, out):
    require(sha(catalog) == expected, "Archive catalog identity differs")
    index = load(catalog)
    roots = {"active": store / "runs", "cold": shelf / "runs",
             "restore/family-tabletop-v1": store / "restore/family-tabletop-v1/runs"}
    checked, shared = {}, {}
    counts = Counter()
    manifest_bytes = members = 0
    for rid, pin in index["original_runs"].items():
        root = roots[pin["location"]]
        bundle = root / rid
        result = verify_bundle(bundle)
        require(result["run"]["run_id"] == rid and result["manifest_sha256"] == pin["manifest_sha256"]
                and result["run"]["execution_status"] == pin["execution_status"], "Changed original run")
        require(bool(pin["archive_receipts"]), "Missing historical archive receipt")
        for obj in result["shared"]:
            sig = obj["sha256"]
            if sig not in shared:
                source = asset_path(store, sig)
                require(source.is_file() and source.stat().st_size == obj["bytes"] and sha(source) == sig,
                        "Shared asset missing or changed")
                shared[sig] = obj["bytes"]
            require(obj["bytes"] == shared[sig], "Shared asset length disagreement")
        manifest = load(bundle / "manifest.json")
        manifest_bytes += sum(row["bytes"] for row in manifest["files"])
        members += len(manifest["files"])
        checked[rid] = result
        counts[pin["location"]] += 1
    graph = {rid: {x["run_id"] for x in result["inputs"]["parents"]} for rid, result in checked.items()}
    for rid, result in checked.items():
        retry = result["inputs"].get("retry_of")
        if retry:
            graph[rid].add(retry["run_id"])
        require(graph[rid] <= checked.keys(), "Missing parent or retry run")
    for rid in graphlib.TopologicalSorter(graph).static_order():
        # Shared objects have already been byte-verified above for every distinct digest.
        verify_dependencies({**checked[rid], "shared": []}, store, checked)
    report = {"schema": "el-phase1-local-archive-verification-1", "catalog_sha256": expected,
              "original_runs": len(checked), "locations": dict(counts),
              "execution_status": dict(Counter(r["run"]["execution_status"] for r in checked.values())),
              "verified_manifest_members": members, "logical_member_bytes_including_repetition": manifest_bytes,
              "distinct_shared_objects": len(shared), "distinct_shared_bytes": sum(shared.values()),
              "provider_scope": "Verified local recovered bytes and earlier recorded provider receipts.",
              "verified_runs": {rid: r["manifest_sha256"] for rid, r in checked.items()}}
    out.mkdir(parents=True, exist_ok=True)
    save(out / "archive-audit.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "store", "shelf", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected", required=True)
    audit(**vars(parser.parse_args()))
