"""Measure a sealed general-stage census and the local store over a declared scan interval."""

import argparse
import csv
import os
import shutil
import stat
import sys
from collections import Counter
from pathlib import Path

from evidence_lens._bundle import load, save, utc


RUN_FIELDS = (
    "run_id",
    "status",
    "kind",
    "started_utc",
    "ended_utc",
    "wall_seconds",
    "sampled_peak_tree_rss_bytes",
    "files",
    "logical_bytes",
    "allocated_stat_bytes",
    "manifest_sha256",
)


def scan(path, local_inodes=None, global_inodes=None, exclude=None):
    """Use metadata only and skip symlinks, including links into previous research stores."""
    t = Counter()
    pending = [path]
    while pending:
        p = pending.pop()
        if p == exclude:
            continue
        try:
            s = p.lstat()
            if stat.S_ISDIR(s.st_mode):
                pending.extend(p.iterdir())
            elif stat.S_ISLNK(s.st_mode):
                t["symlinks_skipped"] += 1
            elif stat.S_ISREG(s.st_mode):
                t["files"] += 1
                t["logical_bytes"] += s.st_size
                blocks = s.st_blocks * 512
                t["allocated_stat_bytes"] += blocks
                identity = (s.st_dev, s.st_ino)
                if local_inodes is not None and identity not in local_inodes:
                    local_inodes.add(identity)
                    t["hardlink_distinct_files"] += 1
                    t["hardlink_distinct_logical_bytes"] += s.st_size
                    t["hardlink_distinct_allocated_stat_bytes"] += blocks
                if global_inodes is not None and identity not in global_inodes:
                    global_inodes.add(identity)
                    t["first_seen_logical_bytes"] += s.st_size
                    t["first_seen_allocated_stat_bytes"] += blocks
        except FileNotFoundError:
            t["entries_vanished_during_scan"] += 1
    return t


def csv_table(path, fields, rows):
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def measure(store, out):
    start = utc()
    rid = os.environ["EL_RUN_ID"]
    own = store / "work" / rid
    parts = sorted(store.iterdir())
    candidates = [
        folder
        for section in ("runs", "quarantine")
        for folder in sorted((store / section).glob("*"))
        if not folder.is_symlink()
        and (folder / "run.json").is_file()
        and (folder / "completion.json").is_file()
    ]
    rows = []
    for folder in candidates:
        run, c = load(folder / "run.json"), load(folder / "completion.json")
        if c["sealed_at_utc"] > start:
            continue
        if (
            run["run_id"] != folder.name
            or c["run_id"] != folder.name
            or run["execution_status"] != c["execution_status"]
        ):
            raise ValueError("census encountered an inconsistent native completion")
        size = scan(folder)
        rows.append(
            {
                "run_id": run["run_id"],
                "status": c["execution_status"],
                "kind": run["run_kind"],
                "started_utc": run["started_at_utc"],
                "ended_utc": run["ended_at_utc"],
                "wall_seconds": run["elapsed_seconds_monotonic"],
                "sampled_peak_tree_rss_bytes": run.get("sampled_peak_tree_rss_bytes"),
                **{k: size[k] for k in ("files", "logical_bytes", "allocated_stat_bytes")},
                "manifest_sha256": c["manifest_sha256"],
            }
        )
    rows.sort(key=lambda x: x["run_id"])
    if len({r["run_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate native run identity")
    csv_table(out / "runs.csv", RUN_FIELDS, rows)
    components = []
    seen = set()
    all_totals = Counter()
    fields = (
        "files",
        "logical_bytes",
        "allocated_stat_bytes",
        "hardlink_distinct_files",
        "hardlink_distinct_logical_bytes",
        "hardlink_distinct_allocated_stat_bytes",
        "symlinks_skipped",
        "entries_vanished_during_scan",
    )
    # Include only standard store component names in the public projection.
    standard = {"runs", "quarantine", "receipts", "staging", "transport", "work", "history", "index", "assets"}
    extra = Counter()
    for p in parts:
        counts = scan(p, set(), seen, exclude=own)
        all_totals.update(counts)
        if p.name in standard:
            components.append({"component": p.name, **{k: counts[k] for k in fields}})
        else:
            extra.update(counts)
    if extra:
        components.append({"component": "other_top_level", **{k: extra[k] for k in fields}})
    csv_table(out / "components.csv", ("component", *fields), components)
    capacity = shutil.disk_usage(store)
    from evidence_lens._bundle import sha

    save(
        out / "resources.json",
        {
            "schema": "el-general-size-ledger-2",
            "run_id": rid,
            "start_utc": start,
            "finished_utc": utc(),
            "sealed_runs": len(rows),
            "execution_states": dict(Counter(x["status"] for x in rows)),
            **{k: sum(x[k] for x in rows) for k in ("files", "logical_bytes", "allocated_stat_bytes")},
            "local_store": {
                "files": all_totals["files"],
                "logical_bytes": all_totals["logical_bytes"],
                "allocated_stat_bytes": all_totals["allocated_stat_bytes"],
                "distinct_inodes": len(seen),
                "hardlink_distinct_logical_bytes": all_totals["first_seen_logical_bytes"],
                "hardlink_distinct_allocated_stat_bytes": all_totals["first_seen_allocated_stat_bytes"],
                "symlinks_skipped": all_totals["symlinks_skipped"],
                "entries_vanished_during_scan": all_totals["entries_vanished_during_scan"],
            },
            "free_volume_bytes": capacity.free,
            "volume_bytes": capacity.total,
            "tables_sha256": {p.name: sha(p) for p in (out / "runs.csv", out / "components.csv")},
            "scope": "Native run and completion records present at scan start and sealed by that time. The separate local-store inventory includes assets, transport, work, indexes, receipts and history-link metadata. The census's own active work directory is excluded.",
            "size_basis": "Path totals sum regular files and POSIX stat blocks. Distinct-inode totals count hard links once. APFS clones can share extents across different inodes; filesystem blocks do not measure net physical storage. Symbolic links are counted separately and never followed.",
            "time_basis": "The local store and free volume were sampled over the given interval. Run wall intervals can overlap or contain child runs.",
            "partial_artifacts": "Native terminal failures retain their own rows. Pre-launch and transport failures have separate receipts. Store-level sizes include live work visible during the scan.",
        },
    )


def record(store, phase):
    from general_records import stage

    pin = load(store / "work/controller/t1-resources/selected.json")
    return stage(
        store,
        store / "history",
        "g6-storage-" + phase,
        "derivation",
        {"t1-resources": pin},
        {"phase": phase, "scope": "local general store, its sealed native records and full component sizes"},
        [sys.executable, "scripts/general_resource_ledger.py", "--store", str(store), "--out", "{out}"],
        "payload/resources.json",
        350000,
    )


def export(store, phase, destination):
    from evidence_lens._bundle import encoded, sha
    from general_records import resolve

    pin = load(store / "work/controller" / ("g6-storage-" + phase) / "selected.json")
    root = resolve((store, store / "history"), pin).parent
    data = load(root / "resources.json")
    target = destination / phase
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in {"resources.json": pin["sha256"], **data["tables_sha256"]}.items():
        if name not in {"resources.json", "runs.csv", "components.csv"} or sha(root / name) != expected:
            raise ValueError("unexpected or changed ledger member")
        raw = (root / name).read_bytes()
        dest = target / name
        if dest.exists() and dest.read_bytes() != raw:
            raise ValueError("a different ledger already occupies this phase")
        if not dest.exists():
            with dest.open("xb") as f:
                f.write(raw)
    ref = target / "origin.json"
    if ref.exists() and ref.read_bytes() != encoded(pin):
        raise ValueError("ledger origin changed")
    if not ref.exists():
        save(ref, pin)
    return pin


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True, type=Path)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--out", type=Path)
    choice.add_argument("--record", choices=("candidate", "final"))
    choice.add_argument("--export", choices=("candidate", "final"))
    p.add_argument("--public-root", type=Path, default=Path("docs/general-closeout/resources"))
    a = p.parse_args()
    if a.record:
        import json

        print(json.dumps(record(a.store.resolve(), a.record), sort_keys=True))
    elif a.export:
        export(a.store.resolve(), a.export, a.public_root)
    else:
        measure(a.store.resolve(), a.out)
