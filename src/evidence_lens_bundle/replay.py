"""Bounded, serial scoring of a prepared static camera sequence on CPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .bundle import Bundle
from .common import encoded, json_bytes, member, profile, require, sha, write
from .replay_sidecar import CAPABILITIES, register_views
from .scoring import counts, reasons

ARRAY_KEYS = ("risk", "valid", "state", "unknown_reason")


def words(a):
    """Index owned arrays by their actual contiguous words, including FP32 NaN payloads."""
    return {k: hashlib.sha256(np.ascontiguousarray(a[k]).view(np.uint8)).hexdigest() for k in ARRAY_KEYS}


def percentiles(values):
    if not values:
        return {"N": 0, "p50": None, "p95": None, "p99": None, "max": None}
    quantiles = np.percentile(values, [50, 95, 99, 100])
    return {"N": len(values), **dict(zip(("p50", "p95", "p99", "max"), map(float, quantiles)))}


def csv_rows(file, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with file.open("x", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


class Sequence:
    """Read a validated, source-authorized bundle. All frame decoding precedes replay ingress."""

    def __init__(self, directory, *, sequence_id=None, expected_sha256=None, preload=True):
        from .resident import binding_from_bundle, frame_meta_from_bundle

        start = time.perf_counter()
        self.bundle = b = Bundle(directory, expected_sha256)
        self.sequence_id = sequence_id or Path(directory).name
        self.sources = {s: b.source(s) for s in b.order}
        self.map_revision = b.producer["checkpoint_sha256"]
        # Self-renders depend on the map. Source-generation changes independently on RGB, camera or order.
        camera_doc = json_bytes(member(b.root, b.meta["camera"], expected="cameras.json"))
        self.source_revision = sha(
            encoded([[s, b.meta["sources"][s]["sha256"], camera_doc["views"][s]] for s in b.order])
        )
        self.binding = binding_from_bundle(
            b, map_revision=self.map_revision, source_revision=self.source_revision
        )
        self.version_key = sha(encoded(self.binding))
        self.frames = {v: b.frame(v) for v in b.targets} if preload else {}
        self.frame_meta = {
            v: frame_meta_from_bundle(b, v, sequence_id=self.sequence_id, frame_id=v, binding=self.binding)
            for v in self.frames
        }
        self.load_seconds = time.perf_counter() - start

    def frame(self, view):
        return self.frames[view] if view in self.frames else self.bundle.frame(view)

    def meta(self, view, key):
        from .resident import frame_meta_from_bundle

        if view in self.frame_meta:
            return {**self.frame_meta[view], "frame_id": key}
        return frame_meta_from_bundle(
            self.bundle, view, sequence_id=self.sequence_id, frame_id=key, binding=self.binding
        )


class LatestQueue:
    """One pending frame, at most one in service; each offered frame obtains one terminal disposition."""

    def __init__(self):
        self.condition = threading.Condition()
        self.pending = None
        self.current_version = None
        self.closed = False
        self.rows = []

    def offer(self, item):
        with self.condition:
            require(not self.closed, "queue already closed")
            seq, view, row = item
            row["enqueue_time"] = time.monotonic()
            row["accepted"] = True
            row["queue_on_arrival"] = int(self.pending is not None)
            if self.pending is not None:
                old = self.pending[2]
                old.update(
                    status="dropped",
                    reason="replaced_by_latest_pending",
                    disposition_time=row["enqueue_time"],
                    replaced_by=row["frame_id"],
                    pending_seconds_before_drop=row["enqueue_time"] - old["enqueue_time"],
                )
            self.rows.append(row)
            self.pending = item
            self.current_version = (seq.sequence_id, seq.version_key)
            self.condition.notify()

    def take(self):
        with self.condition:
            self.condition.wait_for(lambda: self.closed or self.pending is not None)
            if self.pending is None:
                return None
            item, self.pending = self.pending, None
            item[2]["start_time"] = time.monotonic()
            return item

    def finish(self, item, completed, max_age):
        seq, _, row = item
        with self.condition:
            version = (seq.sequence_id, seq.version_key)
            age = completed - row["input_ready_time"]
            why = None
            if version != self.current_version:
                why = "superseded_map_or_source"
            elif age > max_age:
                why = "result_age_limit"
            row.update(
                status="expired" if why else "finished",
                reason=why or "scored_current_snapshot",
                finish_time=completed,
                disposition_time=completed,
                age_seconds=age,
                service_seconds=completed - row["start_time"],
                wait_seconds=row["start_time"] - row["enqueue_time"],
            )
            return why is None

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def output_html(out, summary, last):
    from .report import image_tag, maps, png
    from .scoring import rgb_preview

    last_text, pictures = "No current completed frame during this load.", ""
    if last is not None:
        seq, view, row, a = last
        last_text = f"{row['frame_id']}, {seq.sequence_id} / {view}. Local result from its recorded revisions."
        preview = (
            [(rgb_preview(seq.frame(view)["rgb"]), "Predicted RGB")] if seq.frame(view) is not None else []
        )
        preview += [
            (png(pic), name)
            for pic, name in zip(maps(a), ("Continuous risk", "Threshold state", "Unknown reason"))
        ]
        pictures = "".join(
            "<figure>" + image_tag(raw, name) + "<figcaption>" + name + "</figcaption></figure>"
            for raw, name in preview
        )
    timing = []
    for k, v in summary["latency_seconds"].items():
        cells = [html.escape(k), str(v["N"])] + [
            "—" if v[x] is None else f"{1000 * v[x]:.2f}" for x in ("p50", "p95", "p99", "max")
        ]
        timing.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    status = " · ".join(f"{k} {v}" for k, v in summary["status_counts"].items())
    (out / "replay.html").write_text(
        f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Static camera replay</title><style>body{{font:17px system-ui;max-width:70rem;margin:2rem auto;padding:0 1rem}}
figure{{display:inline-block;margin:.4rem}}img{{width:220px;max-width:100%}}td,th{{text-align:left;padding:.3rem .8rem}}table{{border-collapse:collapse}}</style>
<h1>Static camera replay</h1><p>{html.escape(summary["method"])} · {summary["input_hz"]:g} Hz input · {status}</p>
<h2>Last delivered result</h2><p>{html.escape(last_text)}</p>{pictures}
<h2>Measured latency</h2><table><tr><th>Boundary</th><th>N</th><th>p50 ms</th><th>p95 ms</th><th>p99 ms</th><th>Max ms</th></tr>
{"".join(timing)}</table><p>Input was pre-rendered and decoded. The CPU scored each serviced frame. Upstream Gaussian rendering uses CUDA.
Sample percentiles include serviced frames even when an otherwise valid result expired. See the frame ledger for every scheduled arrival.</p></html>""",
        encoding="utf-8",
    )


def replay(sequences, changes, scorer, method, out, *, hz, duration, max_age=0.25, expected=None, sidecar=None):
    """Changes is an ordered list of (first offered frame index, predecoded Sequence object)."""
    require(
        math.isfinite(hz)
        and 0 < hz <= 1000
        and math.isfinite(duration)
        and duration > 0
        and 0 < max_age < float("inf"),
        "positive bounded replay settings required",
    )
    n = math.ceil(hz * duration - 1e-10)
    require(
        n > 0
        and changes
        and changes[0][0] == 0
        and all(0 <= t < n and seq in sequences for t, seq in changes)
        and all(a[0] < b[0] for a, b in zip(changes, changes[1:])),
        "invalid version-change schedule",
    )
    require(
        all(set(seq.frames) == set(seq.bundle.targets) for seq in sequences),
        "all target inputs must be prepared before scheduled replay",
    )
    require(not out.exists() or not any(out.iterdir()), "expected empty replay output")
    out.mkdir(parents=True, exist_ok=True)
    default_meta_id, metadata_records, metadata_views = register_views(sidecar, sequences)
    model = profile()["models"][method]
    queue = LatestQueue()
    cache_events, producer_errors = [], []
    anchor_utc = datetime.now(timezone.utc).isoformat()
    begin = time.monotonic()
    elapsed_cpu = time.process_time()
    start_at = begin + 0.02

    def produce():
        try:
            cursor = 0
            for i in range(n):
                planned = start_at + i / hz
                wait = planned - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                if cursor + 1 < len(changes) and i == changes[cursor + 1][0]:
                    cursor += 1
                start, seq = changes[cursor]
                view = seq.bundle.targets[(i - start) % len(seq.bundle.targets)]
                frame_id = f"ingress-{i:06d}"
                meta_id = metadata_views.get((seq.sequence_id, view), default_meta_id)
                metadata = metadata_records[meta_id]
                ready = time.monotonic()
                row = {
                    "sequence_id": seq.sequence_id,
                    "frame_id": frame_id,
                    "target_view": view,
                    "input_ready_time": ready,
                    "planned_arrival_time": planned,
                    "enqueue_time": None,
                    "start_time": None,
                    "finish_time": None,
                    "disposition_time": None,
                    "map_revision": seq.map_revision,
                    "source_revision": seq.source_revision,
                    "renderer_revision": seq.binding["renderer_revision"],
                    "renderer_receipt_sha256": seq.binding["renderer_receipt_sha256"],
                    "model_id": model["model_id"],
                    "bundle_sha256": seq.bundle.sha256,
                    "prediction_sha256": seq.bundle.producer["frames"][view].get("file", {}).get("sha256"),
                    "schedule_lateness_seconds": ready - planned,
                    "acquisition_timestamp": metadata["acquisition_timestamp"],
                    "acquisition_clock_domain": metadata["clock_domain"],
                    "sidecar_sha256": meta_id,
                    "age_seconds": None,
                    "service_seconds": None,
                    "wait_seconds": None,
                }
                queue.offer((seq, view, row))
            remaining = start_at + duration - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        except Exception as exc:
            producer_errors.append(str(exc))
        finally:
            queue.close()

    thread = threading.Thread(target=produce, name="scheduled-static-ingress")
    thread.start()
    previous, context, last = None, None, None
    try:
        while (item := queue.take()) is not None:
            seq, view, row = item
            try:
                if previous is not seq:
                    old = scorer.cache_info()
                    t = time.monotonic()
                    context = scorer.prepare_sources(seq.sources, version_binding=seq.binding)
                    delta = time.monotonic() - t
                    cache_events.append(
                        {
                            "frame_id": row["frame_id"],
                            "sequence_id": seq.sequence_id,
                            "map_revision": seq.map_revision,
                            "source_revision": seq.source_revision,
                            "prepare_seconds": delta,
                            "cache_before": old,
                            "cache_after": scorer.cache_info(),
                        }
                    )
                    previous = seq
                metadata = metadata_records[row["sidecar_sha256"]]
                frame_meta = seq.meta(view, row["frame_id"])
                frame_meta.update(
                    clock_domain=metadata["clock_domain"],
                    capture_time=metadata["acquisition_timestamp"],
                    pose_covariance=metadata["pose_covariance"],
                    dynamic_mask=metadata["dynamic_mask"],
                )
                a = scorer.score_frame(
                    context, seq.frame(view), seq.bundle.cameras[view], frame_meta=frame_meta
                )
                completed = time.monotonic()
                identity = a["identity"]
                require(
                    identity["frame_meta"]["frame_id"] == row["frame_id"]
                    and identity["frame_meta"]["sequence_id"] == seq.sequence_id
                    and identity["map_revision"] == row["map_revision"]
                    and identity["source_revision"] == row["source_revision"],
                    "result identity crossed frames or revisions",
                )
                present = queue.finish(item, completed, max_age)
                digest = words(a)
                if expected is not None:
                    require(expected[(seq.sequence_id, view)] == digest, "offline and replay words differ")
                row["frame_input_sha256"] = identity["frame_input_sha256"]
                row["output_sha256"] = identity["output_sha256"]
                row["source_context_sha256"] = identity["source_context_sha256"]
                row.update(
                    {
                        **a["timing"],
                        **counts(a),
                        **{"unknown_" + k: v for k, v in reasons(a).items()},
                        **{k + "_sha256": v for k, v in digest.items() if k in ARRAY_KEYS},
                        "accepted": True,
                        "exact_offline_match": expected is not None,
                    }
                )
                # Caller-facing service includes switching costs. Numerical kernel timings stay separate.
                row["score_call_seconds"] = a["timing"]["service_seconds"]
                row["service_seconds"] = completed - row["start_time"]
                if present:
                    last = (seq, view, row, a)
            except Exception as exc:
                now = time.monotonic()
                row.update(
                    status="failed",
                    reason=type(exc).__name__ + ": " + str(exc),
                    finish_time=now,
                    disposition_time=now,
                    age_seconds=now - row["input_ready_time"],
                    service_seconds=now - row["start_time"],
                    wait_seconds=row["start_time"] - row["enqueue_time"],
                )
    finally:
        thread.join()
    end = time.monotonic()
    statuses = {
        s: sum(row["status"] == s for row in queue.rows) for s in ("finished", "dropped", "expired", "failed")
    }
    serviced = [r for r in queue.rows if r["start_time"] is not None]
    accepted = len(queue.rows)
    completed = statuses["finished"]
    span = end - start_at
    timing_fields = (
        "service_seconds",
        "wait_seconds",
        "age_seconds",
        "schedule_lateness_seconds",
        "score_call_seconds",
        "feature_seconds",
        "model_seconds",
        "postprocess_seconds",
    )
    summary = {
        "schema": "el-static-replay-run-1",
        "run_id": os.getenv("EL_RUN_ID"),
        "method": method,
        "model": model,
        "clock": {
            "domain": "process monotonic seconds, time.monotonic()",
            "monotonic_anchor": begin,
            "utc_anchor": anchor_utc,
            "input_ready": "predecoded evidence released to this call by scheduler",
            "acquisition": "optional unconverted input clock",
        },
        "queue": {
            "pending_capacity": 1,
            "workers": 1,
            "policy": "replace latest pending; in-flight snapshot is immutable",
            "max_result_age_seconds": max_age,
            "max_pending_observed": int(bool(accepted)),
        },
        "input_hz": hz,
        "requested_seconds": duration,
        "input_schedule_start": start_at,
        "actual_finish": end,
        "observed_load_and_drain_seconds": span,
        "process_cpu_seconds": time.process_time() - elapsed_cpu,
        "scheduled": n,
        "accepted": accepted,
        "status_counts": statuses,
        "service_starts": len(serviced),
        "completion_rate": completed / n,
        "drop_rate": statuses["dropped"] / n,
        "delivered_outputs_per_wall_second": completed / span,
        "scoring_completions_per_wall_second": (completed + statuses["expired"]) / span,
        "latency_seconds": {
            k: percentiles([r[k] for r in queue.rows if r.get(k) is not None]) for k in timing_fields
        },
        "distinct_offered_views": len({(r["sequence_id"], r["target_view"]) for r in queue.rows}),
        "distinct_serviced_views": len({(r["sequence_id"], r["target_view"]) for r in serviced}),
        "repeated_offers": accepted - len({(r["sequence_id"], r["target_view"]) for r in queue.rows}),
        "repeated_serviced_views": len(serviced)
        - len({(r["sequence_id"], r["target_view"]) for r in serviced}),
        "pixel_denominators": {
            k: sum(r.get(k, 0) for r in serviced) for k in ("canvas", "scoreable", "low", "high", "unknown")
        },
        "parity_checked": expected is not None,
        "producer_errors": producer_errors,
        "cache_events": cache_events,
        "cache": scorer.cache_info(),
        "sidecar": metadata_records[default_meta_id],
        "sidecar_records": metadata_records,
        "capabilities": CAPABILITIES,
        "preload_seconds_by_sequence": {s.sequence_id: s.load_seconds for s in sequences},
        "last_delivered_frame": None if last is None else last[2]["frame_id"],
    }
    csv_rows(out / "frames.csv", queue.rows)
    write(out / "replay.json", summary)
    if last is not None:
        np.savez_compressed(out / "last-result.npz", **{k: last[3][k] for k in ARRAY_KEYS})
    output_html(out, summary, last)
    require(
        accepted == n and not statuses["failed"] and not producer_errors,
        "replay failed; see the original frame ledger",
    )
    return summary


def main(argv=None):
    from .resident import load_scorer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--switch", action="append", default=[], metavar="FRAME_INDEX=BUNDLE")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", choices=tuple(profile()["models"]), default="MS-L1")
    p.add_argument("--models", type=Path)
    p.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads for neural models")
    p.add_argument("--hz", type=float, default=10)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--max-age", type=float, default=0.25, help="maximum input-ready result age in seconds")
    p.add_argument("--sidecar", type=Path)
    a = p.parse_args(argv)
    changes = [(0, Sequence(a.bundle))]
    for spec in a.switch:
        index, sep, path = spec.partition("=")
        require(sep == "=", "supply FRAME_INDEX=BUNDLE")
        changes.append((int(index), Sequence(Path(path))))
    scorer = load_scorer(a.model, models_dir=a.models, threads=a.threads)
    replay(
        [s for _, s in changes],
        changes,
        scorer,
        a.model,
        a.out,
        hz=a.hz,
        duration=a.seconds,
        max_age=a.max_age,
        sidecar=json_bytes(a.sidecar.read_bytes()) if a.sidecar else None,
    )


if __name__ == "__main__":
    main()
