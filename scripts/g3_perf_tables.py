"""Produce portable tables from completed recorder payloads, retaining incomplete configurations and raw rows."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
from html.parser import HTMLParser
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from evidence_lens_bundle.common import require
from evidence_lens_bundle.replay import csv_rows, percentiles


def read(path):
    return json.loads(path.read_bytes())


def table(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def csv_write(root, name, rows):
    if rows:
        csv_rows(root / name, rows)


TICKET = "fad00db69c7127466f64014a5073e84262d289815fe6858bed87d5b910e84b6c"
FREEZE = "a31f0caef9bee1f17bb98c8cdb5feef544c916ad32806b6f46360f8051412c64"
CHANNELS = ("risk_sha256", "valid_sha256", "state_sha256", "unknown_reason_sha256")
WEIGHTS = {
    "MS-L1": "d0a42f0458390710aba87d43da48770309ca78fdf93cfba6a47229e186b9f02a",
    "MS-ft": "99dfef7c82b808939895ffbbd0327513011920e1f47a5d0556d0eb6c6c6d0fd3",
}
PIXELS = ("canvas", "scoreable", "low", "high", "unknown")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_receipt(path):
    data = read(path)
    require(
        data["cpu_ticket_sha256"] == TICKET and data["engineering_config_sha256"] == FREEZE,
        "received data use a different formal CPU implementation",
    )
    return data


def registered_loads(platform):
    models = ("MS-L1", "MS-ft") if platform == "mac" else ("MS-L1",)
    rows = [
        {"id": f"bell-M4-{method}-{hz}Hz", "method": method, "input_hz": hz, "requested_seconds": 60}
        for method in models
        for hz in (10, 30)
    ]
    return rows + [{"id": "version-switch", "method": "MS-L1", "input_hz": 10, "requested_seconds": 6}]


def heat_identity(frame, condition, method, route, worker_threads):
    require(
        (frame["scene"], frame["method"], frame["path"]) == (condition, method, route),
        "raw G3 observation belongs to another workload",
    )
    actual = int(frame["torch_threads"]) if frame["torch_threads"] else None
    require(
        actual == (worker_threads if method == "MS-ft" else None), "G3 observation used different threads"
    )
    return {"condition": condition, "worker_threads_argument": worker_threads, **frame}


def check_pixels(row):
    canvas, valid, low, high, unknown = (int(row[k]) for k in PIXELS)
    require(
        canvas == 256 * 256 and low + high == valid and valid + unknown == canvas,
        "canvas counts differ",
    )
    if row.get("unknown_reasons"):
        missing = json.loads(row["unknown_reasons"]).values()
    else:
        missing = (
            int(v) for k, v in row.items() if k.startswith("unknown_") and not k.endswith("_sha256") and v
        )
    require(sum(missing) == unknown, "unknown reasons do not partition unknown pixels")


def verify_load(rows, receipt, offline):
    require(
        len(rows) == receipt["scheduled"] == receipt["accepted"],
        "missing ingress ledger rows",
    )
    observed = {k: 0 for k in receipt["status_counts"]}
    by_id = {r["frame_id"]: r for r in rows}
    unique = {(r["sequence_id"], r["target_view"]) for r in rows}
    origin, end = receipt["input_schedule_start"], receipt["actual_finish"]
    require(
        math.isclose(end - origin, receipt["observed_load_and_drain_seconds"], abs_tol=1e-7)
        and round(receipt["input_hz"] * receipt["requested_seconds"]) == receipt["scheduled"],
        "independent monotonic load window or scheduled count differs",
    )
    require(len(unique) == receipt["distinct_offered_views"], "unique-offer count differs")
    require(len(rows) - len(unique) == receipt["repeated_offers"], "repeated-offer count differs")
    for index, r in enumerate(rows):
        require(
            r["frame_id"] == f"ingress-{index:06d}" and r["accepted"] == "True",
            "invalid or duplicate ingress ID",
        )
        ready, enqueue, terminal = (
            float(r[k]) for k in ("input_ready_time", "enqueue_time", "disposition_time")
        )
        planned = float(r["planned_arrival_time"])
        require(
            ready <= enqueue <= terminal <= end
            and int(r["queue_on_arrival"]) in (0, 1)
            and math.isclose(planned, origin + index / receipt["input_hz"], rel_tol=0, abs_tol=1e-7)
            and math.isclose(ready - planned, float(r["schedule_lateness_seconds"]), abs_tol=1e-7),
            "invalid queue times/depth",
        )
        status = r["status"]
        observed[status] += 1
        if status == "dropped":
            require(
                not r["start_time"] and r["replaced_by"],
                "dropped row lacks valid replacement",
            )
            newer = by_id[r["replaced_by"]]
            require(
                newer["frame_id"] > r["frame_id"] and float(newer["enqueue_time"]) == terminal,
                "discarded item and its replacement do not share the recorded queue event",
            )
        elif status in ("finished", "expired"):
            start, finish = float(r["start_time"]), float(r["finish_time"])
            require(
                enqueue <= start <= finish
                and math.isclose(start - enqueue, float(r["wait_seconds"]), abs_tol=1e-7)
                and math.isclose(finish - start, float(r["service_seconds"]), abs_tol=1e-7)
                and math.isclose(finish - ready, float(r["age_seconds"]), abs_tol=1e-7),
                "result age differs from observed clocks",
            )
            require(
                r["exact_offline_match"] == "True"
                and tuple(r[k] for k in CHANNELS) == offline[r["sequence_id"], r["target_view"]],
                "served output differs from independent offline ledger",
            )
            check_pixels(r)
    require(
        observed == receipt["status_counts"],
        "terminal counts differ from worker summary",
    )
    require(
        math.isclose(observed["finished"] / len(rows), receipt["completion_rate"])
        and math.isclose(observed["dropped"] / len(rows), receipt["drop_rate"])
        and math.isclose(
            observed["finished"] / receipt["observed_load_and_drain_seconds"],
            receipt["delivered_outputs_per_wall_second"],
        ),
        "delivered, discarded or wall-clock-throughput numerator differs",
    )
    require(
        all(sum(int(r[k] or 0) for r in rows) == receipt["pixel_denominators"][k] for k in PIXELS),
        "aggregated replay pixels differ from the raw ledger",
    )
    for name, original in receipt["latency_seconds"].items():
        recalculated = percentiles([float(r[name]) for r in rows if r.get(name)])
        require(
            recalculated["N"] == original["N"]
            and all(
                (recalculated[q] == original[q])
                if recalculated[q] is None or original[q] is None
                else math.isclose(recalculated[q], original[q], rel_tol=1e-12, abs_tol=1e-12)
                for q in ("p50", "p95", "p99", "max")
            ),
            "latency sample percentiles differ",
        )
    elapsed = float(rows[-1]["enqueue_time"]) - float(rows[0]["enqueue_time"])
    return (len(rows) - 1) / elapsed if elapsed else None


def verify_offline(root, receipt, platform, method):
    rows = table(root / ("parity-" + method) / "offline-index.csv")
    summary = []
    expected = {}
    require(
        receipt["passed"] and len(rows) == receipt["frames"],
        "offline reference row count differs",
    )
    for r in rows:
        check_pixels(r)
        require(r["method"] == method and r["sequence_id"], "offline frame lacks its method or sequence")
        key = (r["sequence_id"], r["view"])
        require(
            key not in expected and r["exact"] == "True",
            "duplicate view or different reference words",
        )
        require(
            all(int(r[k.replace("_sha256", "_different_words")]) == 0 for k in CHANNELS),
            "unequal arrays",
        )
        expected[key] = tuple(r[k] for k in CHANNELS)
    for seq in receipt["sequences"]:
        selected = [r for r in rows if r["sequence_id"] == seq["sequence_id"]]
        model = seq["reference_model"]
        require(
            model["weights_sha256"] == WEIGHTS[method],
            "wrong weights in offline comparison",
        )
        require(
            len(selected) == seq["views"] == seq["distinct_poses"],
            "missing distinct offline poses",
        )
        for k in PIXELS:
            require(
                sum(int(r[k]) for r in selected) == seq["pixels"][k],
                "offline pixel denominator differs",
            )
        summary.append(
            {
                "platform": platform,
                "method": method,
                "sequence": seq["sequence_id"],
                "frames": len(selected),
                "distinct_poses": seq["distinct_poses"],
                "exact_frames": len(selected),
                **model,
                **seq["pixels"],
                **{k + "_different_words": 0 for k in receipt["different_words"]},
            }
        )
    return expected, summary, [{"platform": platform, "sequence": r["sequence_id"], **r} for r in rows]


def summarize(platform, g3, g4):
    configs, cold, candidates, loads, cache_rows, raw, replay_rows, backends = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    candidate_raw, offline_raw, parity, resource_samples, switches = [], [], [], [], []
    if g3 and (g3 / "g3.json").is_file():
        receipt = checked_receipt(g3 / "g3.json")
        checked_words = {}
        selection = read(g3 / "selection.json")
        require(
            selection == receipt["selection"],
            "separate selection and scored configuration disagree",
        )
        cold = [{"platform": platform, **r} for r in table(g3 / "cold-starts.csv")]
        for method, settings in selection.items():
            reference_dir = g3 / (method + "-selection-reference")
            reference = {}
            if (reference_dir / "frames.csv").is_file():
                for r in table(reference_dir / "frames.csv"):
                    check_pixels(r)
                    reference[r["target"]] = tuple(r[k] for k in CHANNELS)
                    candidate_raw.append(
                        {
                            "platform": platform,
                            "scope": "selection-file-reference",
                            **heat_identity(r, "bell-M4", method, "file", 4),
                        }
                    )
            for candidate in settings["candidates"]:
                threads = candidate["threads"]
                path = g3 / f"{method}-candidate-{threads}" / "summary.json"
                detail = read(path) if path.is_file() else {}
                if (path.parent / "frames.csv").is_file():
                    observations = table(path.parent / "frames.csv")
                    same = []
                    for r in observations:
                        check_pixels(r)
                        same.append(tuple(r[k] for k in CHANNELS) == reference[r["target"]])
                        candidate_raw.append(
                            {
                                "platform": platform,
                                "scope": "thread-candidate",
                                **heat_identity(r, "bell-M4", method, "resident-cache", threads),
                            }
                        )
                    require(
                        len(observations) == candidate["N"] and all(same) == candidate["exact"],
                        "thread candidate sample or reference comparison differs",
                    )
                env = detail.get("environment", {})
                candidates.append(
                    {
                        "platform": platform,
                        "method": method,
                        "selected": threads == settings["threads"],
                        **candidate,
                        **detail.get("resource", {}),
                    }
                )
                backends.append(
                    {
                        "platform": platform,
                        "method": method,
                        "configured_neural_threads": env.get("torch_intraop_threads"),
                        **{
                            k: env.get(k)
                            for k in (
                                "os",
                                "machine",
                                "processor",
                                "python",
                                "numpy",
                                "torch",
                                "physical_cpus",
                                "logical_cpus",
                                "available_logical_cpus",
                                "torch_interop_threads",
                            )
                        },
                        "numpy_blas": (env.get("numpy_blas") or {}).get("name"),
                        "numpy_lapack": (env.get("numpy_lapack") or {}).get("name"),
                        "numpy_blas_version": (env.get("numpy_blas") or {}).get("version"),
                        "parallel_backend": env.get("torch_parallel_backend"),
                        "thread_environment": json.dumps(env.get("thread_env")),
                        "loaded_pools": json.dumps(
                            [
                                {
                                    k: t.get(k)
                                    for k in (
                                        "user_api",
                                        "internal_api",
                                        "num_threads",
                                        "version",
                                        "threading_layer",
                                        "architecture",
                                    )
                                }
                                for t in env["loaded_thread_pools"]
                            ]
                        )
                        if env.get("loaded_thread_pools") is not None
                        else None,
                    }
                )
        actual = {(r["scene"], r["method"], r["path"]): r for r in receipt["configurations"]}
        for condition in ("bell-M4", "beads-M2", "bell-M2", "beads-M4"):
            for method in ("MS-L1", "MS-ft"):
                for route in ("file", "resident-cache", "resident-no-cache"):
                    cfg = f"{condition}-{method}-{route}"
                    hot = g3 / cfg / "hot"
                    r = actual.get((condition, method, route), {})
                    rows = table(hot / "frames.csv") if (hot / "frames.csv").is_file() else []
                    detail = read(hot / "summary.json") if (hot / "summary.json").is_file() else {}
                    if r:
                        require(
                            len(rows) == r["N"],
                            "hot ledger sample count differs from native selection",
                        )
                    for item in rows:
                        check_pixels(item)
                        key = (condition, method, item["target"])
                        words = tuple(item[k] for k in CHANNELS)
                        if route == "file" and key not in checked_words:
                            checked_words[key] = words
                        require(
                            checked_words.get(key) == words,
                            "hot output differs from original file bytes",
                        )
                    raw.extend(
                        {
                            "platform": platform,
                            **heat_identity(x, condition, method, route, selection[method]["threads"]),
                        }
                        for x in rows
                    )
                    c = {k: v for k, v in r.get("resource", {}).items()}
                    result = {
                        "platform": platform,
                        "condition": condition,
                        "method": method,
                        "path": route,
                        "measurement_role": {
                            "file": "required file baseline",
                            "resident-cache": "required final resident",
                            "resident-no-cache": "cache ablation",
                        }[route],
                        "N": len(rows),
                        "requested_N": receipt["N_per_hot_configuration"],
                        "cold_N": sum(
                            x["scene"] == condition and x["method"] == method and x["path"] == route
                            for x in cold
                        ),
                        "distinct_views": len({x["target"] for x in rows}),
                        "status": r.get("status", "blocked_or_failed"),
                        "recorded_failure_reasons": json.dumps(
                            [
                                f
                                for f in receipt["failures"]
                                if f["scope"] == cfg
                                or f["scope"].startswith(cfg + "/")
                                or f["scope"] == method
                            ]
                        ),
                        "exact_reference": r.get("exact_reference"),
                        "exact_intra_process_repeat": r.get("exact_intra_process_repeat"),
                        "configured_torch_threads": r.get("torch_threads"),
                        **detail.get("setup", {}),
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
                        "model_and_bundle_load_seconds",
                        "source_decode_seconds",
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
                    measured_wall = sum(float(x["wall_seconds"]) for x in rows)
                    measured_cpu = sum(float(x["cpu_seconds"]) for x in rows)
                    result["numerical_service_cpu_seconds"] = measured_cpu
                    result["numerical_service_wall_seconds"] = measured_wall
                    result["numerical_service_busy_logical_cpus"] = (
                        measured_cpu / measured_wall if measured_wall else None
                    )
                    for k in ("canvas", "scoreable", "unknown", "low", "high"):
                        result[k] = sum(int(x[k]) for x in rows)
                    configs.append(result)
        for file in sorted(g3.rglob("resources.csv")):
            phase = file.parent.relative_to(g3).as_posix()
            resource_samples.extend(
                {"platform": platform, "stage": "g3", "condition": phase, **r} for r in table(file)
            )
    if g4 and (g4 / "g4.json").is_file():
        g = checked_receipt(g4 / "g4.json")
        offline_by_method = {}
        for method, report in g.get("parity", {}).items():
            if not report["passed"] or not (g4 / ("parity-" + method) / "offline-index.csv").is_file():
                continue
            expected, summaries, observations = verify_offline(g4, report, platform, method)
            offline_by_method[method] = expected
            parity.extend(summaries)
            offline_raw.extend(observations)
        if g.get("version_switch_status") is not None:
            switches.append(
                {
                    "platform": platform,
                    "run_id": g["run_id"],
                    "renderer_run": g["renderer_run"],
                    **{
                        k: g[k]
                        for k in (
                            "switched_target_arrays_different",
                            "switched_source_self_arrays_different",
                            "same_map_target_arrays_exact_after_source_switch",
                            "switch_wall_seconds",
                        )
                    },
                    **g["version_switch_status"],
                }
            )
        declared = {item["id"]: item for item in g.get("loads", [])}
        if g.get("version_switch_status") is not None:
            declared["version-switch"] = {"id": "version-switch", "method": "MS-L1"}
        for spec in registered_loads(platform):
            directory = g4 / spec["id"]
            if (
                spec["id"] not in declared
                or spec["method"] not in offline_by_method
                or not all((directory / n).is_file() for n in ("replay.json", "frames.csv", "resources.json"))
            ):
                continue
            r = read(directory / "replay.json")
            require(
                r["model"]["weights_sha256"] == WEIGHTS[spec["method"]]
                and r["method"] == spec["method"]
                and r["input_hz"] == spec["input_hz"]
                and r["requested_seconds"] == spec["requested_seconds"],
                "timed load used a different registered model, rate or duration",
            )
            prefix = {"platform": platform, "run": spec["id"], "method": r["method"]}
            observations = table(directory / "frames.csv")
            actual_hz = verify_load(observations, r, offline_by_method[r["method"]])
            replay_rows.extend({**prefix, **x} for x in observations)
            row = {
                **prefix,
                "status": "failed" if r["producer_errors"] or r["status_counts"]["failed"] else "done",
                "failure_reasons": json.dumps(r["producer_errors"]),
                "model_id": r["model"]["model_id"],
                "weights_sha256": r["model"]["weights_sha256"],
                "input_hz": r["input_hz"],
                "actual_enqueue_hz_first_to_last": actual_hz,
                "actual_seconds": r["observed_load_and_drain_seconds"],
                "load_start_monotonic": r["input_schedule_start"],
                "load_finish_monotonic": r["actual_finish"],
                "scheduled": r["scheduled"],
                "accepted": r["accepted"],
                "unique_offered": r["distinct_offered_views"],
                "unique_serviced": r["distinct_serviced_views"],
                "repeated_offers": r["repeated_offers"],
                "repeated_serviced_views": r["repeated_serviced_views"],
                "requested_seconds": r["requested_seconds"],
                "preload_seconds_by_sequence": json.dumps(r["preload_seconds_by_sequence"], sort_keys=True),
                "input_clock": r["clock"]["domain"],
                "utc_anchor": r["clock"]["utc_anchor"],
                "max_pending": r["queue"]["max_pending_observed"],
                **{"cache_" + k: v for k, v in r["cache"].items()},
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
        for file in sorted(g4.rglob("resources.csv")):
            phase = file.parent.relative_to(g4).as_posix()
            resource_samples.extend(
                {"platform": platform, "stage": "g4", "condition": phase, **r} for r in table(file)
            )
    represented = {r["run"] for r in loads}
    for spec in registered_loads(platform):
        if spec["id"] in represented:
            continue
        prefix = {"platform": platform, "run": spec["id"], "method": spec["method"]}
        source = g4 / spec["id"] / "frames.csv" if g4 else None
        observations = table(source) if source and source.is_file() else []
        replay_rows.extend({**prefix, **r} for r in observations)
        status = (
            "failed" if any(r.get("status") == "failed" for r in observations) else "blocked_missing_evidence"
        )
        loads.append(
            {
                **prefix,
                "status": status,
                "input_hz": spec["input_hz"],
                "requested_seconds": spec["requested_seconds"],
                "raw_rows_available": len(observations),
                "failure_reasons": "Registered run lacks a complete native load, parity or resource receipt.",
            }
        )
    return {
        "g3-configurations.csv": configs,
        "g3-platform-backends.csv": backends,
        "g3-cold-starts.csv": cold,
        "g3-thread-candidates.csv": candidates,
        "g3-thread-candidate-frames.csv": candidate_raw,
        "g3-frames.csv": raw,
        "g4-offline-parity.csv": parity,
        "g4-offline-frames.csv": offline_raw,
        "g4-loads.csv": loads,
        "g4-version-events.csv": cache_rows,
        "g4-version-checks.csv": switches,
        "g4-frames.csv": replay_rows,
        "process-resource-samples.csv": resource_samples,
    }


class ReplayImages(HTMLParser):
    def __init__(self):
        super().__init__()
        self.panels = []

    def handle_starttag(self, tag, attrs):
        from PIL import Image

        values = dict(attrs)
        if tag == "img":
            src = values["src"]
            prefix = "data:image/png;base64,"
            require(src.startswith(prefix), "sample report image is external")
            data = base64.b64decode(src[len(prefix) :], validate=True)
            with Image.open(io.BytesIO(data)) as pic:
                require(
                    pic.size == (256, 256) and pic.format == "PNG",
                    "invalid sample report image",
                )
                pic.load()
            self.panels.append(values["alt"])
        elif "src" in values or "href" in values:
            raise ValueError("sample report includes an external reference")


def replay_example(root, out):
    import numpy as np
    from evidence_lens_bundle.replay import words

    directory = root / "bell-M4-MS-L1-30Hz"
    receipt = read(directory / "replay.json")
    rows = table(directory / "frames.csv")
    frame = next(r for r in rows if r["frame_id"] == receipt["last_delivered_frame"])
    require(
        frame["status"] == "finished",
        "sample report corresponds to an undisplayed result",
    )
    with np.load(directory / "last-result.npz", allow_pickle=False) as arrays:
        require(
            all(frame[k + "_sha256"] == v for k, v in words(arrays).items()),
            "sample report arrays differ",
        )
    raw = (directory / "replay.html").read_bytes()
    html = raw.decode("utf-8")
    parser = ReplayImages()
    parser.feed(html)
    parser.close()
    require(
        parser.panels == ["Predicted RGB", "Continuous risk", "Threshold state", "Unknown reason"],
        "missing panel",
    )
    require(
        f"{frame['frame_id']}, {frame['sequence_id']} / {frame['target_view']}" in html,
        "sample identity differs",
    )
    (out / "replay-example.html").write_bytes(raw)
    (out / "replay-example-arrays.npz").write_bytes((directory / "last-result.npz").read_bytes())
    return {
        "native_run": receipt["run_id"],
        "condition": directory.name,
        "frame": frame["frame_id"],
        "target_view": frame["target_view"],
        "map_revision": frame["map_revision"],
        "source_revision": frame["source_revision"],
        "output_sha256": frame["output_sha256"],
        "images": parser.panels,
        "html_sha256": sha(out / "replay-example.html"),
        "arrays_sha256": sha(out / "replay-example-arrays.npz"),
    }


def verify_native(root, phase):
    """Verify the native closure and every raw member this projection may open."""
    folder = root.parent
    manifest_path, completion_path = folder / "manifest.json", folder / "completion.json"
    require(
        root.name == "payload" and completion_path.is_file() and manifest_path.is_file(),
        "expected a native recorder payload with both manifest and completion",
    )
    completion, manifest = read(completion_path), read(manifest_path)
    require(
        completion["manifest_sha256"] == sha(manifest_path)
        and completion["run_id"] == manifest["run_id"] == folder.name,
        "native manifest digest or run identity differs from its completion",
    )
    rows = {r["path"]: r for r in manifest["files"]}
    require(len(rows) == len(manifest["files"]), "duplicate native manifest member")
    sample = {"payload/bell-M4-MS-L1-30Hz/replay.html", "payload/bell-M4-MS-L1-30Hz/last-result.npz"}

    def consumed(name):
        return name.startswith("payload/") and (name.endswith((".csv", ".json")) or name in sample)

    on_disk = {p.relative_to(folder).as_posix() for p in root.rglob("*") if p.is_file()}
    members = {n for n in rows if consumed(n)}
    require(members == {n for n in on_disk if consumed(n)}, "raw payload inventory differs from its manifest")
    for name in sorted(members):
        p, item = folder / name, rows[name]
        require(
            p.stat().st_size == item["bytes"] and sha(p) == item["sha256"],
            "raw payload digest differs from its native index: " + name,
        )
    key = "payload/" + phase + ".json"
    native = {
        "run_id": manifest["run_id"],
        "manifest_sha256": completion["manifest_sha256"],
        "native_execution": completion["execution_status"],
        "completion_sha256": sha(completion_path),
        "receipt_sha256": rows.get(key, {}).get("sha256"),
        "verified_raw_members": len(members),
    }
    require(
        key in rows or completion["execution_status"] != "succeeded", "succeeded native run lacks its receipt"
    )
    return native


def provenance(root, phase):
    evidence = verify_native(root, phase)
    path = root / (phase + ".json")
    if not path.is_file():
        return {**evidence, "status": "failed", "reason": "sealed native run lacks its stage receipt"}
    receipt = checked_receipt(path)
    require(receipt["run_id"] == evidence["run_id"], "stage receipt belongs to another native run")
    evidence["status"] = receipt["status"] if evidence["native_execution"] == "succeeded" else "failed"
    if phase == "g3":
        evidence.update(
            {
                "budget_seconds": receipt["budget_seconds"],
                "measured_seconds": receipt["wall_seconds"],
                "selected_threads": {m: s["threads"] for m, s in receipt["selection"].items()},
                "requested_hot_frames_per_configuration": receipt["N_per_hot_configuration"],
                "configurations_reported": len(receipt["configurations"]),
                "failures": receipt["failures"],
            }
        )
    else:
        evidence.update(
            {
                "renderer_run": receipt.get("renderer_run"),
                "labels_accessed": receipt.get("labels_accessed"),
                "new_risk_training_updates": receipt.get("new_risk_training_updates"),
                "parity": {
                    m: {
                        k: v
                        for k, v in r.items()
                        if k
                        in (
                            "frames",
                            "passed",
                            "different_words",
                            "seconds",
                            "resource",
                        )
                    }
                    for m, r in receipt.get("parity", {}).items()
                },
            }
        )
    return evidence


def archive(out, names):
    path = out / "evidence-lens-cpu-replay-measurements.zip"
    with ZipFile(path, "x", compression=ZIP_DEFLATED) as z:
        for name in sorted(names):
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            z.writestr(info, (out / name).read_bytes(), compresslevel=6)
    with ZipFile(path) as z:
        require(
            z.testzip() is None and set(z.namelist()) == set(names),
            "raw archive failed readback",
        )
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}


def main(a):
    a.out.mkdir(parents=True, exist_ok=True)
    all_rows, sources = {}, {}
    for platform in ("mac", "windows"):
        g3, g4 = getattr(a, platform + "_g3"), getattr(a, platform + "_g4")
        if g3 or g4:
            sources[platform] = {
                p: provenance(root, p)
                if root
                else {"status": "blocked_missing_evidence", "reason": "No native payload supplied"}
                for p, root in (("g3", g3), ("g4", g4))
            }
            portable = summarize(platform, g3, g4)
            replay_source = sources[platform]["g4"]
            replay_source["registered_workloads"] = {
                r["run"]: {"status": r["status"], "failure_reasons": r["failure_reasons"]}
                for r in portable["g4-loads.csv"]
            }
            if replay_source["status"] == "done" and any(
                r["status"] != "done" for r in portable["g4-loads.csv"]
            ):
                replay_source["reported_status"] = replay_source["status"]
                replay_source["status"] = "incomplete"
            for n, rows in portable.items():
                all_rows.setdefault(n, []).extend(rows)
    for n, rows in all_rows.items():
        csv_write(a.out, n, rows)
    index = {
        n: {
            "rows": len(rows),
            "bytes": (a.out / n).stat().st_size,
            "sha256": sha(a.out / n),
        }
        for n, rows in all_rows.items()
        if rows
    }
    names = list(index)
    result = {
        "schema": "el-static-engineering-measurements-1",
        "attribution": {
            "creator": "wany614 / Evidence Lens",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "originals": "Authored helical beads and offset bell cage, with registered views and source photographs.",
        },
        "timing_boundaries": {
            "g3": "One real prepared target per score. File service includes score-file persistence. Resident service returns arrays. HTML is timed separately.",
            "g4": "Scheduled predecoded CPU inputs. Service is dequeue to scored result. Age starts at observed input readiness. A separate host monotonic stopwatch extends from the schedule origin through producer completion and queue drain. Both stamps are in the native replay receipt; delivered throughput counts finished frames over their difference.",
            "resource_samples": "0.1-second process-tree sampler. One occupied logical CPU equals 100 percent. Frame clocks use host-local monotonic time.",
            "quantiles": "Sample p50/p95/p99/max. Repeated trajectory views are engineering calls.",
        },
        "cpu_ticket_sha256": TICKET,
        "engineering_config_sha256": FREEZE,
        "sources": sources,
        "tables": index,
    }
    if a.mac_g4 and any(
        r["run"] == "bell-M4-MS-L1-30Hz" and r["platform"] == "mac" and r["status"] == "done"
        for r in all_rows["g4-loads.csv"]
    ):
        result["html_example"] = replay_example(a.mac_g4, a.out)
        names.extend(["replay-example.html", "replay-example-arrays.npz"])
    (a.out / "tables.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    packed = archive(a.out, names + ["tables.json"])
    (a.out / "public-archive.json").write_text(json.dumps(packed, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("mac-g3", "mac-g4", "windows-g3", "windows-g4", "out"):
        p.add_argument("--" + n, type=Path, required=n == "out")
    main(p.parse_args())
