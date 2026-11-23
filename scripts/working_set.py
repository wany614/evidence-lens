"""Move cloud-verified sealed runs intact between the active store and a local cold shelf."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from evidence_lens._bundle import RUN_ID, encoded, load, sha
from evidence_lens._index import writer_gate
from evidence_lens._verify import require, verify_archive_receipt as verify_receipt, verify_bundle
from make_handoff import publish


def locations(store, shelf):
    store, shelf = store.resolve(), shelf.resolve()
    require(not store.is_relative_to(shelf) and not shelf.is_relative_to(store),
            "choose a separate shelf outside the active store")
    shelf.mkdir(parents=True, exist_ok=True)
    require(store.stat().st_dev == shelf.stat().st_dev,
            "atomic parking requires folders on the same filesystem")
    return store, shelf


def relocate(store, shelf, rid, digest, receipt_id, *, to_shelf):
    verify_receipt(store, rid, digest, receipt_id)
    active = store / "runs" / rid
    cold = shelf / "runs" / rid
    old, new = (active, cold) if to_shelf else (cold, active)
    with writer_gate(store):
        require(old.is_dir() != new.is_dir(), "run must exist at exactly one of the two locations")
        existing = old if old.is_dir() else new
        verified = verify_bundle(existing)
        require(verified["manifest_sha256"] == digest and verified["run"]["run_id"] == rid,
                "run content differs from its archived manifest")
        marker = {"run_id": rid, "manifest_sha256": digest, "archive_receipt_id": receipt_id}
        publish(shelf / "catalog" / (rid + ".json"), encoded(marker))
        if existing == old:
            new.parent.mkdir(parents=True, exist_ok=True)
            os.rename(old, new)
        require(sha(new / "manifest.json") == digest, "relocated manifest differs")
    return marker


def report_view(store, shelf, output):
    """Expose only the existing immutable records required by report_study roster/assemble."""
    require(not output.exists(), "choose a new report evidence directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with writer_gate(store), tempfile.TemporaryDirectory(dir=output.parent, prefix=".report-view-") as temporary:
        tree = Path(temporary) / "view"
        tree.mkdir()
        seen = set()
        for parent in (store / "runs", shelf / "runs"):
            if not parent.is_dir():
                continue
            for run in sorted(parent.iterdir()):
                rid = run.name
                require(RUN_ID.fullmatch(rid) and rid not in seen and run.is_dir() and not run.is_symlink(),
                        "ambiguous, linked or invalid evidence run")
                seen.add(rid)
                seal = load(run / "completion.json")
                require(seal["run_id"] == rid and sha(run / "manifest.json") == seal["manifest_sha256"],
                        "evidence run ID or manifest differs from its original seal")
                if parent == shelf / "runs":
                    parked = load(shelf / "catalog" / (rid + ".json"))
                    verify_receipt(store, rid, parked["manifest_sha256"], parked["archive_receipt_id"])
                    require(sha(run / "manifest.json") == parked["manifest_sha256"], "cold manifest differs")
                head = ("manifest.json", "completion.json", "run.json", "config.resolved.yaml", "environment.json",
                        "inputs.json", "artifacts.json", "metrics.csv", "telemetry.csv", "telemetry_schema.json")
                names = {name for name in head if (run / name).is_file()}
                payload = run / "payload"
                if payload.is_dir():
                    names.update(p.relative_to(run).as_posix() for p in payload.rglob("*.json"))
                for name in ("payload/board/board_v01.png", "payload/curves/family.npz"):
                    if (run / name).is_file():
                        names.add(name)
                for name in names:
                    source = run / name
                    require(source.resolve() == source and not (source.stat().st_mode & 0o222),
                            "report evidence must be regular, sealed and read-only")
                    target = tree / "runs" / rid / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source, target)
        target = tree / "receipts/transfer"
        target.mkdir(parents=True)
        for receipt in (store / "receipts/transfer").glob("*.json"):
            os.link(receipt, target / receipt.name)
        (tree / "scope.txt").write_text("Selected metadata, curves and boards for reporting. "
                                         "Use the original sealed bundles or their archives to restore a full run.\n")
        tree.rename(output)
    return {"evidence_runs": len(seen)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True, type=Path)
    p.add_argument("--shelf", required=True, type=Path)
    modes = p.add_subparsers(dest="action", required=True)
    for mode in ("park", "restore"):
        sub = modes.add_parser(mode)
        sub.add_argument("--run", required=True)
        sub.add_argument("--manifest", required=True)
        sub.add_argument("--archive-receipt", required=True)
    modes.add_parser("report-view").add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    store, shelf = locations(a.store, a.shelf)
    if a.action == "report-view":
        result = report_view(store, shelf, a.out.resolve())
    else:
        result = relocate(store, shelf, a.run, a.manifest, a.archive_receipt, to_shelf=a.action == "park")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
