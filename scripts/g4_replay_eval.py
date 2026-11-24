"""Run exhaustive original-sequence output parity, then timed finite-queue G4 replay."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import time
from pathlib import Path

import numpy as np

from evidence_lens_bundle.common import json_bytes, require, write
from evidence_lens_bundle.replay import ARRAY_KEYS, Sequence, csv_rows, replay, words
from evidence_lens_bundle.resident import load_scorer
from evidence_lens_bundle.scoring import counts, reasons, score
from g3_perf_resources import Monitor, environment

MAIN = ("beads-M2", "beads-M4", "bell-M2", "bell-M4")
SWITCH = ("beads-M4", "beads-map2-M4", "beads-map2-source2-M4")


def evidence(root, sequence_id, index):
    spec = index[sequence_id]
    seq = Sequence(root / sequence_id, sequence_id=sequence_id, expected_sha256=spec["bundle_sha256"])
    require(
        seq.map_revision == spec["map_revision"]
        and seq.source_revision == spec["source_revision"]
        and seq.bundle.producer["renderer"]["receipt_sha256"] == spec["renderer_revision"],
        "prepared revisions differ from render ticket",
    )
    n = len(seq.bundle.targets)
    require(
        n == spec["distinct_target_poses"] and n == (120 if sequence_id in MAIN else 24),
        "duplicate or missing trajectory poses",
    )
    return seq


def expected_index(path):
    with path.open(encoding="utf-8", newline="") as f:
        return {
            (row["sequence_id"], row["view"]): {k: row[k + "_sha256"] for k in ARRAY_KEYS}
            for row in csv.DictReader(f)
        }


def verify(root, index, cpu, threads, method, names, out):
    start = time.perf_counter()
    rows, comparisons = [], []
    out.mkdir(parents=True)
    for name in names:
        seq = evidence(root, name, index)
        dest = out / "reference" / name
        dest.parent.mkdir(exist_ok=True)
        base = score(seq.bundle.root, dest, method, cpu_dir=cpu, threads=threads)
        scorer = load_scorer(method, models_dir=cpu, threads=threads)
        ctx = scorer.prepare_sources(seq.sources, version_binding=seq.binding)
        for view in base["views"]:
            v = view["id"]
            a = scorer.score_frame(
                ctx, seq.frame(v), seq.bundle.cameras[v], frame_meta=seq.meta(v, "offline-" + v)
            )
            with np.load(dest / view["arrays"]["path"], allow_pickle=False) as z:
                digest = words(z)
                different = {
                    k: int(
                        np.count_nonzero(
                            np.ascontiguousarray(z[k]).view(np.uint32 if k == "risk" else np.uint8)
                            != np.ascontiguousarray(a[k]).view(np.uint32 if k == "risk" else np.uint8)
                        )
                    )
                    for k in ARRAY_KEYS
                }
            entry = {
                "sequence_id": name,
                "view": v,
                "bundle_sha256": seq.bundle.sha256,
                "map_revision": seq.map_revision,
                "source_revision": seq.source_revision,
                "prediction_sha256": seq.bundle.producer["frames"][v]["file"]["sha256"],
                "method": method,
                "model_id": scorer.model_id,
                "exact": not any(different.values()),
                **{k + "_sha256": val for k, val in digest.items()},
                **{k + "_different_words": val for k, val in different.items()},
                **counts(a),
                **{"unknown_" + k: val for k, val in reasons(a).items()},
                "arrays": (dest / view["arrays"]["path"]).relative_to(out).as_posix(),
            }
            rows.append(entry)
        comparisons.append(
            {
                "sequence_id": name,
                "views": len(seq.bundle.targets),
                "distinct_poses": index[name]["distinct_target_poses"],
                "reference_model": base["model"],
                "cache": scorer.cache_info(),
                "pixels": base["counts"],
            }
        )
    csv_rows(out / "offline-index.csv", rows)
    passed = all(r["exact"] for r in rows)
    summary = {
        "schema": "el-static-replay-parity-1",
        "method": method,
        "frames": len(rows),
        "passed": passed,
        "different_words": {k: sum(r[k + "_different_words"] for r in rows) for k in ARRAY_KEYS},
        "sequences": comparisons,
        "seconds": time.perf_counter() - start,
    }
    write(out / "parity.json", summary)
    require(passed, "stopped before timed runs: offline and array outputs differ")
    return summary


def run(args):
    require(os.environ.get("EL_RUN_ID"), "run formal G4 after starting the recorder")
    require(args.platform in ("mac", "windows"), "unknown engineering platform")
    args.out.mkdir(parents=True, exist_ok=True)
    contract = json_bytes((args.trajectories / "trajectories.json").read_bytes())
    index = {s["sequence_id"]: s for s in contract["sequences"]}
    require(set(index) == set(MAIN) | set(SWITCH), "wrong original trajectory roster")
    settings = json_bytes(args.selection.read_bytes())
    models = ("MS-L1", "MS-ft") if args.platform == "mac" else ("MS-L1",)
    parity, results = {}, []
    for method in models:
        threads = settings[method]["threads"]
        names = (*MAIN, *SWITCH[1:]) if method == "MS-L1" else MAIN
        with Monitor() as m:
            result = verify(
                args.trajectories, index, args.models, threads, method, names, args.out / ("parity-" + method)
            )
        result["resource"] = m.summary()
        parity[method] = result
    # All offline comparisons precede the G4 timed load. Re-read each completed result against those digests.
    for method in models:
        expected = expected_index(args.out / ("parity-" + method) / "offline-index.csv")
        threads = settings[method]["threads"]
        for hz in (10, 30):
            name = f"bell-M4-{method}-{hz}Hz"
            seq = evidence(args.trajectories, "bell-M4", index)
            scorer = load_scorer(method, models_dir=args.models, threads=threads)
            # Warm source preparation and first target before measuring steady-state input. Every later target is recomputed.
            ctx = scorer.prepare_sources(seq.sources, version_binding=seq.binding)
            v = seq.bundle.targets[0]
            scorer.score_frame(ctx, seq.frame(v), seq.bundle.cameras[v], frame_meta=seq.meta(v, "warmup"))
            with Monitor() as m:
                run = replay(
                    [seq], [(0, seq)], scorer, method, args.out / name, hz=hz, duration=60, expected=expected
                )
            write(
                args.out / name / "resources.json",
                {"resource": m.summary(), "environment": environment(method)},
            )
            csv_rows(args.out / name / "resources.csv", m.samples)
            results.append(
                {
                    "id": name,
                    "method": method,
                    "inputs": run["accepted"],
                    "status": run["status_counts"],
                    "unique_offered": run["distinct_offered_views"],
                    "repeated_offers": run["repeated_offers"],
                    "full_wall_seconds": run["observed_load_and_drain_seconds"],
                    "resource": m.summary(),
                    "latency": run["latency_seconds"],
                    "delivered_outputs_per_wall_second": run["delivered_outputs_per_wall_second"],
                }
            )
    switch = [evidence(args.trajectories, s, index) for s in SWITCH]
    a, b, c = switch
    require(
        a.map_revision != b.map_revision == c.map_revision
        and a.source_revision == b.source_revision != c.source_revision,
        "switch requires genuinely different checkpoints then different source camera/images",
    )
    require(
        any(
            a.bundle.producer["frames"][v]["file"]["sha256"] != b.bundle.producer["frames"][v]["file"]["sha256"]
            for v in b.bundle.targets
        ),
        "new map reused old prediction bytes",
    )
    require(
        any(
            a.bundle.producer["frames"][v]["file"]["sha256"] != b.bundle.producer["frames"][v]["file"]["sha256"]
            for v in a.bundle.order
        ),
        "new map reused old source self-renders",
    )
    require(
        set(x["sha256"] for x in b.bundle.meta["sources"].values())
        != set(x["sha256"] for x in c.bundle.meta["sources"].values()),
        "new sources reused identical evidence",
    )

    def same_render(left, right):
        return all(
            np.array_equal(left[k].view(np.uint32), right[k].view(np.uint32)) for k in ("rgb", "z", "alpha")
        )

    changed_targets = sum(not same_render(a.frame(v), b.frame(v)) for v in b.bundle.targets)
    changed_self = sum(not same_render(a.sources[v][1], b.sources[v][1]) for v in a.bundle.order)
    unchanged_map_targets = sum(same_render(b.frame(v), c.frame(v)) for v in b.bundle.targets)
    require(
        changed_targets and changed_self, "registered second Gaussian map must change actual render array words"
    )
    require(
        unchanged_map_targets == len(b.bundle.targets),
        "source-only switch changed target rendering on the fixed map",
    )
    method = "MS-L1"
    expected = expected_index(args.out / ("parity-" + method) / "offline-index.csv")
    scorer = load_scorer(method, models_dir=args.models, threads=settings[method]["threads"])
    with Monitor() as m:
        transition = replay(
            switch,
            [(0, a), (20, b), (40, c)],
            scorer,
            method,
            args.out / "version-switch",
            hz=10,
            duration=6,
            expected=expected,
        )
    events = transition["cache_events"]
    require(
        [x["sequence_id"] for x in events] == list(SWITCH)
        and all(x["cache_after"]["misses"] > x["cache_before"]["misses"] for x in events),
        "registered map/source changes did not build independent source contexts",
    )
    write(
        args.out / "version-switch/resources.json",
        {"resource": m.summary(), "environment": environment(method)},
    )
    csv_rows(args.out / "version-switch/resources.csv", m.samples)
    write(
        args.out / "g4.json",
        {
            "schema": "el-static-replay-tests-1",
            "run_id": os.environ["EL_RUN_ID"],
            "platform": platform.platform(),
            "status": "done",
            "parity": parity,
            "loads": results,
            "version_switch": events,
            "switched_target_arrays_different": changed_targets,
            "switched_source_self_arrays_different": changed_self,
            "same_map_target_arrays_exact_after_source_switch": unchanged_map_targets,
            "version_switch_status": transition["status_counts"],
            "switch_wall_seconds": transition["observed_load_and_drain_seconds"],
            "switch_resources": m.summary(),
            "renderer_run": contract["run_id"],
            "labels_accessed": False,
            "new_risk_training_updates": 0,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("trajectories", "models", "selection", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--platform", choices=("mac", "windows"), required=True)
    run(p.parse_args())
