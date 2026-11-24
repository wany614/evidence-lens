"""Produce portable tables from completed recorder payloads, retaining incomplete configurations and raw rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


from evidence_lens_bundle.replay import csv_rows, percentiles


def read(path):
    return json.loads(path.read_bytes())


def table(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def csv_write(root, name, rows):
    if rows:
        csv_rows(root / name, rows)


def summarize(platform, g3, g4):
    configs, cold, candidates, loads, cache_rows, raw, replay_rows = [], [], [], [], [], [], []
    if g3:
        receipt = read(g3 / "g3.json")
        selection = read(g3 / "selection.json")
        cold = [{"platform": platform, **r} for r in table(g3 / "cold-starts.csv")]
        candidates = [
            {"platform": platform, "method": m, "selected": v["threads"] == s["threads"], **v}
            for m, s in selection.items()
            for v in s["candidates"]
        ]
        actual = {(r["scene"], r["method"], r["path"]): r for r in receipt["configurations"]}
        for condition in ("bell-M4", "beads-M2", "bell-M2", "beads-M4"):
            for method in ("MS-L1", "MS-ft"):
                for route in ("file", "resident-cache", "resident-no-cache"):
                    cfg = f"{condition}-{method}-{route}"
                    hot = g3 / cfg / "hot"
                    r = actual.get((condition, method, route), {})
                    rows = table(hot / "frames.csv") if (hot / "frames.csv").is_file() else []
                    raw.extend({"platform": platform, **x} for x in rows)
                    c = {k: v for k, v in r.get("resource", {}).items()}
                    result = {
                        "platform": platform,
                        "condition": condition,
                        "method": method,
                        "path": route,
                        "N": len(rows),
                        "requested_N": receipt["N_per_hot_configuration"],
                        "cold_N": sum(
                            x["scene"] == condition and x["method"] == method and x["path"] == route
                            for x in cold
                        ),
                        "distinct_views": len({x["target"] for x in rows}),
                        "status": r.get("status", "blocked_or_failed"),
                        "exact_reference": r.get("exact_reference"),
                        **c,
                        **{"cache_" + k: v for k, v in (r.get("cache") or {}).items()},
                    }
                    for field in (
                        "service_seconds",
                        "feature_seconds",
                        "model_seconds",
                        "postprocess_seconds",
                        "serialization_seconds",
                        "html_seconds",
                        "target_decode_seconds",
                        "identity_and_cache_seconds",
                    ):
                        stats = percentiles([float(x[field]) for x in rows if x.get(field)])
                        result.update(
                            {
                                field.removesuffix("_seconds") + "_" + q + "_ms": 1000 * x
                                if x is not None
                                else None
                                for q, x in stats.items()
                                if q != "N"
                            }
                        )
                    for k in ("canvas", "scoreable", "unknown", "low", "high"):
                        result[k] = sum(int(x[k]) for x in rows)
                    configs.append(result)
    if g4:
        g = read(g4 / "g4.json")
        for spec in g["loads"] + [{"id": "version-switch", "method": "MS-L1"}]:
            directory = g4 / spec["id"]
            r = read(directory / "replay.json")
            prefix = {"platform": platform, "run": spec["id"]}
            replay_rows.extend({**prefix, **x} for x in table(directory / "frames.csv"))
            row = {
                **prefix,
                "method": r["method"],
                "input_hz": r["input_hz"],
                "actual_seconds": r["observed_load_and_drain_seconds"],
                "scheduled": r["scheduled"],
                "accepted": r["accepted"],
                "unique_offered": r["distinct_offered_views"],
                "unique_serviced": r["distinct_serviced_views"],
                "repeated_offers": r["repeated_offers"],
                **r["status_counts"],
                "completion_rate": r["completion_rate"],
                "drop_rate": r["drop_rate"],
                "outputs_per_wall_second": r["delivered_outputs_per_wall_second"],
                **r["pixel_denominators"],
                **read(directory / "resources.json")["resource"],
            }
            for field, stats in r["latency_seconds"].items():
                row[field.removesuffix("_seconds") + "_N"] = stats["N"]
                row.update(
                    {
                        field.removesuffix("_seconds") + "_" + q + "_ms": x * 1000 if x is not None else None
                        for q, x in stats.items()
                        if q != "N"
                    }
                )
            loads.append(row)
            for e in r["cache_events"]:
                cache_rows.append(
                    {
                        **prefix,
                        "frame_id": e["frame_id"],
                        "sequence": e["sequence_id"],
                        "map_revision": e["map_revision"],
                        "source_revision": e["source_revision"],
                        "prepare_ms": 1000 * e["prepare_seconds"],
                        "miss_delta": e["cache_after"]["misses"] - e["cache_before"]["misses"],
                        "hit_delta": e["cache_after"]["hits"] - e["cache_before"]["hits"],
                    }
                )
    return {
        "g3-configurations.csv": configs,
        "g3-cold-starts.csv": cold,
        "g3-thread-candidates.csv": candidates,
        "g3-frames.csv": raw,
        "g4-loads.csv": loads,
        "g4-version-events.csv": cache_rows,
        "g4-frames.csv": replay_rows,
    }


def main(a):
    a.out.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    for platform in ("mac", "windows"):
        g3, g4 = getattr(a, platform + "_g3"), getattr(a, platform + "_g4")
        if g3 or g4:
            for n, rows in summarize(platform, g3, g4).items():
                all_rows.setdefault(n, []).extend(rows)
    for n, rows in all_rows.items():
        csv_write(a.out, n, rows)
    index = {
        n: {"rows": len(rows), "sha256": hashlib.sha256((a.out / n).read_bytes()).hexdigest()}
        for n, rows in all_rows.items()
        if rows
    }
    (a.out / "tables.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("mac-g3", "mac-g4", "windows-g3", "windows-g4", "out"):
        p.add_argument("--" + n, type=Path, required=n == "out")
    main(p.parse_args())
