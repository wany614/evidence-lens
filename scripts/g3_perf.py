"""Bounded G3 measurement of original file scoring and warm serial arrays on original static evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from evidence_lens_bundle.bundle import Bundle
from evidence_lens_bundle.common import encoded, json_bytes, member, profile, require, row, write
from evidence_lens_bundle.replay import ARRAY_KEYS, Sequence, csv_rows, percentiles, words
from evidence_lens_bundle.report import render as html_report
from evidence_lens_bundle.resident import load_scorer
from evidence_lens_bundle.scoring import counts, reasons, score
from g3_perf_resources import Monitor, available_cpus, environment

CASES = ("bell-M4", "beads-M2", "bell-M2", "beads-M4")
MODELS = ("MS-L1", "MS-ft")
PATHS = ("file", "resident-cache", "resident-no-cache")
COLUMNS = (
    "phase",
    "scene",
    "method",
    "path",
    "torch_threads",
    "index",
    "target",
    "bundle_sha256",
    "model_id",
    "wall_seconds",
    "cpu_seconds",
    "service_seconds",
    "feature_seconds",
    "model_seconds",
    "postprocess_seconds",
    "source_decode_seconds",
    "model_and_bundle_load_seconds",
    "target_decode_seconds",
    "identity_and_cache_seconds",
    "serialization_seconds",
    "html_seconds",
    "cache_peak_bytes",
    "canvas",
    "scoreable",
    "low",
    "high",
    "unknown",
    "unknown_reasons",
    "risk_sha256",
    "valid_sha256",
    "state_sha256",
    "unknown_reason_sha256",
    "exact_repeat",
)


def clone_one(bundle, view, dest):
    """A single-target file workload retaining the exact physical arrays, authorized PNGs and cameras."""
    dest.mkdir(parents=True)
    b = bundle
    source_camera = json_bytes(member(b.root, b.meta["camera"], expected="cameras.json"))
    source_camera["views"] = {s: source_camera["views"][s] for s in (*b.order, view)}
    raw = encoded(source_camera)
    (dest / "cameras.json").write_bytes(raw)
    cm = row("cameras.json", raw)
    records = [*b.meta["sources"].values()]
    frames = {s: b.producer["frames"][s] for s in (*b.order, view)}
    records += [f["file"] for f in frames.values() if f["status"] == "ready"]
    for entry in records:
        data = member(b.root, entry)
        path = dest / entry["path"]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
    prod = {**b.producer, "target_order": [view], "frames": frames, "camera_sha256": cm["sha256"]}
    raw = encoded(prod)
    (dest / "producer.json").write_bytes(raw)
    write(
        dest / "bundle.json",
        {**b.meta, "target_order": [view], "camera": cm, "producer": row("producer.json", raw)},
    )
    return Bundle(dest)


def slim_inputs(trajectories, work):
    indices = json_bytes((trajectories / "workloads.json").read_bytes())
    by_id = {r["sequence_id"]: r for r in indices["sequences"]}
    prep = {}
    for name in CASES:
        b = Bundle(trajectories / name, by_id[name]["bundle_sha256"])
        views = list(b.targets)
        step = np.linspace(0, len(views) - 1, min(4, len(views)), dtype=int)
        chosen = [views[i] for i in step]
        prep[name] = {view: clone_one(b, view, work / name / view).sha256 for view in chosen}
        write(work / name / "view-order.json", {"views": chosen})
    return prep


class Workload:
    def __init__(self, root, args, base):
        self.method, self.path, self.threads, self.models = args.method, args.path, args.threads, args.models
        self.root, self.work, self.name = root, base, args.scene
        self.views = json_bytes((root / args.scene / "view-order.json").read_bytes())["views"]
        self.scorer = None
        self.sequences = {}
        self.contexts = {}
        self.setup = {}
        self.file_step = 0

    def prepare(self, targets):
        if self.path == "file":
            return
        start = time.perf_counter()
        self.scorer = load_scorer(
            self.method,
            models_dir=self.models,
            threads=self.threads,
            cache_bytes=(64 << 20) if self.path == "resident-cache" else 0,
        )
        loaded = time.perf_counter()
        for v in targets:
            seq = self.sequences[v] = Sequence(self.root / self.name / v)
            # Each single-target projection keeps identical source binding, so one source context owns all four views.
            if not self.contexts:
                self.contexts["sources"] = self.scorer.prepare_sources(seq.sources, version_binding=seq.binding)
        self.setup = {
            "load_model_seconds": loaded - start,
            "sequence_predecode_seconds": sum(s.load_seconds for s in self.sequences.values()),
            "first_source_prepare_seconds": self.contexts["sources"].prepare_seconds,
            "total_setup_seconds": time.perf_counter() - start,
        }

    def step(self, view, include_html=True):
        if self.path == "file":
            dest = self.work / f"file-{self.file_step}"
            self.file_step += 1
            begin, cpu = time.perf_counter(), time.process_time()
            folder = self.root / self.name / view
            receipt = score(folder, dest, self.method, cpu_dir=self.models, threads=self.threads)
            finish, cpu_finish, ready_mono = time.perf_counter(), time.process_time(), time.monotonic()
            first = receipt["views"][0]
            with np.load(dest / first["arrays"]["path"], allow_pickle=False) as z:
                digest = words(z)
            html = None
            if include_html:
                t = time.perf_counter()
                html_report(dest, dest / "report.html")
                html = time.perf_counter() - t
            result = {
                "service_seconds": finish - begin,
                **first["timing"],
                "model_and_bundle_load_seconds": receipt["cost"]["bundle_and_model_load_seconds"],
                "source_decode_seconds": receipt["cost"]["source_decode_seconds"],
                "html_seconds": html,
                "wall_seconds": finish - begin,
                "cpu_seconds": cpu_finish - cpu,
                "cache_peak_bytes": 0,
                **first["counts"],
                "unknown_reasons": first["unknown_reasons"],
                "bundle_sha256": receipt["bundle_sha256"],
            }
            shutil.rmtree(dest)
        else:
            seq = self.sequences[view]
            meta = seq.meta(view, view)
            begin, cpu = time.perf_counter(), time.process_time()
            a = self.scorer.score_frame(
                self.contexts["sources"], seq.frame(view), seq.bundle.cameras[view], frame_meta=meta
            )
            finish, cpu_finish, ready_mono = time.perf_counter(), time.process_time(), time.monotonic()
            digest = words(a)
            result = {
                **a["timing"],
                "wall_seconds": finish - begin,
                "cpu_seconds": cpu_finish - cpu,
                "cache_peak_bytes": self.scorer.cache_info()["peak_bytes"],
                **counts(a),
                "unknown_reasons": reasons(a),
                "bundle_sha256": seq.bundle.sha256,
            }
        result["ready_monotonic"] = ready_mono
        result.update({k + "_sha256": v for k, v in digest.items()})
        result["unknown_reasons"] = json.dumps(result["unknown_reasons"], separators=(",", ":"), sort_keys=True)
        return result


def measure(a):
    begin = time.perf_counter()
    expected = json_bytes(a.expected.read_bytes()) if a.expected is not None else {}
    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Workload(a.inputs, a, Path(tmp))
        if a.action == "cold":
            with Monitor() as monitor:
                work.prepare((work.views[0],))
                data = work.step(work.views[0], include_html=False)
            write(
                a.out / "cold.json",
                {
                    "elapsed_inside_seconds": time.perf_counter() - begin,
                    "first_result_monotonic": data["ready_monotonic"],
                    "frame": data,
                    "resource": monitor.summary(),
                    "setup": work.setup,
                    "pid": os.getpid(),
                },
            )
            return
        with Monitor() as monitor:
            work.prepare(work.views)
            rows = []
            warm = work.step(work.views[0])
            with (a.out / "frames.csv").open("x", newline="", encoding="utf-8", buffering=1) as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
                writer.writeheader()
                for i in range(a.n):
                    if time.monotonic() >= a.deadline:
                        break
                    v = work.views[i % len(work.views)]
                    data = work.step(v)
                    digest = {k: data[k + "_sha256"] for k in ARRAY_KEYS}
                    matches = expected.get(v) == digest if v in expected else None
                    row = {
                        "phase": a.action,
                        "scene": a.scene,
                        "method": a.method,
                        "path": a.path,
                        "torch_threads": a.threads if a.method != "MS-L1" else None,
                        "index": i,
                        "target": v,
                        "model_id": profile()["models"][a.method]["model_id"],
                        **data,
                        "exact_repeat": matches,
                    }
                    writer.writerow(row)
                    rows.append(row)
            env = environment(a.method)
    numeric = [x for x in COLUMNS if x.endswith("_seconds")]
    partial = len(rows) != a.n
    hashes = {}
    consistent = True
    for r in rows:
        d = {k: r[k + "_sha256"] for k in ARRAY_KEYS}
        if r["target"] in hashes:
            consistent = consistent and hashes[r["target"]] == d
        hashes[r["target"]] = d
    summary = {
        "scene": a.scene,
        "method": a.method,
        "path": a.path,
        "torch_threads": a.threads if a.method != "MS-L1" else None,
        "phase": a.action,
        "N": len(rows),
        "requested_N": a.n,
        "status": "blocked_budget" if partial else "done",
        "exact_reference": (all(x["exact_repeat"] for x in rows) if expected and rows else None),
        "exact_intra_process_repeat": consistent,
        "expected": hashes,
        "warmup": warm,
        "setup": work.setup,
        "cache": work.scorer.cache_info() if work.scorer else None,
        "environment": env,
        "resource": monitor.summary(),
        "latency_seconds": {k: percentiles([r[k] for r in rows if r.get(k) is not None]) for k in numeric},
        "pixels": {k: sum(r[k] for r in rows) for k in ("canvas", "scoreable", "low", "high", "unknown")},
    }
    write(a.out / "summary.json", summary)
    csv_rows(a.out / "resources.csv", monitor.samples)


def child(base, method, path, threads, scene, out, action, n, expected, deadline):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        action,
        "--inputs",
        str(base.inputs),
        "--out",
        str(out),
        "--method",
        method,
        "--path",
        path,
        "--threads",
        str(threads),
        "--scene",
        scene,
        "--n",
        str(n),
        "--deadline",
        str(deadline),
    ]
    if base.models:
        cmd += ["--models", str(base.models)]
    if expected is not None:
        cmd += ["--expected", str(expected)]
    before = time.monotonic()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None, None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=remaining)
    except subprocess.TimeoutExpired:
        out.mkdir(parents=True, exist_ok=True)
        write(
            out / "partial.json",
            {"status": "blocked_budget", "native_frames_csv": (out / "frames.csv").exists()},
        )
        return None, "deadline"
    wall = time.monotonic() - before
    if proc.returncode != 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "worker-error.txt").write_text(proc.stderr[-9000:], encoding="utf-8")
        return None, "worker_failed"
    file = out / ("cold.json" if action == "cold" else "summary.json")
    result = json_bytes(file.read_bytes())
    result["parent_process_to_exit_seconds"] = wall
    if action == "cold":
        result["process_start_to_first_result_seconds"] = result["first_result_monotonic"] - before
    return result, None


def run(a):
    require(os.environ.get("EL_RUN_ID"), "formal G3 requires a recorder")
    require(0 < a.budget <= 1800, "configuration measurement has a 30 minute platform cap")
    a.out.mkdir(parents=True, exist_ok=True)
    resources, failures, cold_rows, configs = [], [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        a.inputs = Path(tmp) / "one-frame"
        prep = slim_inputs(a.trajectories, a.inputs)
        write(a.out / "single-frame-loads.json", prep)
        # The clock starts after prepared workloads exist. Candidate scans, cold starts and all hot paths share it.
        start = time.monotonic()
        deadline = start + a.budget
        hardware = max(1, min(available_cpus(), 8))
        thread_options = list(dict.fromkeys((1, 2, 4, hardware)))
        selected = {}
        for method in MODELS:
            # A 4-thread released reference and each candidate consume the same four development M4 targets.
            baseline = a.out / (method + "-selection-reference")
            ref, err = child(a, method, "file", 4, "bell-M4", baseline, "candidate", 4, None, deadline)
            if err or ref is None:
                failures.append({"scope": method, "reason": err or "budget_before_selection"})
                continue
            ref_file = baseline / "expected.json"
            write(ref_file, ref["expected"])
            choices = []
            for threads in thread_options if method == "MS-ft" else [4]:
                target = a.out / f"{method}-candidate-{threads}"
                x, err = child(
                    a, method, "resident-cache", threads, "bell-M4", target, "candidate", 8, ref_file, deadline
                )
                if x:
                    choices.append(
                        {
                            "threads": threads,
                            "N": x["N"],
                            "exact": x["exact_reference"],
                            "p50_service_seconds": x["latency_seconds"]["service_seconds"]["p50"],
                        }
                    )
                elif err:
                    failures.append({"scope": str(target.name), "reason": err})
            viable = [x for x in choices if x["N"] == 8 and x["exact"]]
            winner = min(viable, key=lambda x: x["p50_service_seconds"])["threads"] if viable else 4
            selected[method] = {
                "threads": winner,
                "candidates": choices,
                "valid_candidates": len(viable),
                "selection": "lowest sample median among candidates with exact 4-thread file-reference arrays",
                "numpy_extra_thread_candidate": None,
                "ms_l1_threads": "--threads does not control NumPy",
            }
        write(a.out / "selection.json", selected)
        # Prioritize complete file/selected path coverage; the uncached comparison follows all primary configurations.
        for path in PATHS:
            for scene in CASES:
                for method in MODELS:
                    if method not in selected:
                        continue
                    threads = selected[method]["threads"]
                    tag = f"{scene}-{method}-{path}"
                    target = a.out / tag
                    expected = (
                        a.out / (scene + "-" + method + "-file") / "expected.json" if path != "file" else None
                    )
                    if expected is not None and not expected.is_file():
                        failures.append({"scope": tag, "reason": "missing measured reference"})
                        continue
                    if time.monotonic() >= deadline:
                        failures.append({"scope": tag, "reason": "budget exhausted before configuration"})
                        continue
                    for i in range(3):
                        c, err = child(
                            a,
                            method,
                            path,
                            threads,
                            scene,
                            target / f"cold-{i + 1}",
                            "cold",
                            1,
                            expected,
                            deadline,
                        )
                        if c:
                            cold_rows.append(
                                {
                                    "scene": scene,
                                    "method": method,
                                    "path": path,
                                    "process": i + 1,
                                    "pid": c["pid"],
                                    "parent_to_exit_seconds": c["parent_process_to_exit_seconds"],
                                    "start_to_first_result_seconds": c["process_start_to_first_result_seconds"],
                                    "process_cpu_seconds": c["resource"]["process_cpu_seconds"],
                                    "peak_sampled_tree_rss_bytes": c["resource"]["peak_sampled_tree_rss_bytes"],
                                    "inside_to_first_seconds": c["elapsed_inside_seconds"],
                                    "first_service_seconds": c["frame"]["service_seconds"],
                                }
                            )
                        else:
                            failures.append({"scope": tag + "/cold", "reason": err or "budget"})
                            break
                    x, err = child(
                        a, method, path, threads, scene, target / "hot", "hot", a.n, expected, deadline
                    )
                    if x is None:
                        failures.append({"scope": tag, "reason": err or "budget"})
                        continue
                    if path == "file":
                        write(target / "expected.json", x["expected"])
                    configs.append(
                        {
                            k: x[k]
                            for k in (
                                "scene",
                                "method",
                                "path",
                                "torch_threads",
                                "N",
                                "requested_N",
                                "status",
                                "exact_reference",
                                "exact_intra_process_repeat",
                                "pixels",
                                "latency_seconds",
                                "resource",
                                "cache",
                            )
                        }
                    )
                    if x["N"] != a.n or x["exact_reference"] is False or not x["exact_intra_process_repeat"]:
                        failures.append(
                            {"scope": tag, "reason": "budget shortfall or numerical membership changed"}
                        )
                    resources.append({"configuration": tag, **x["resource"]})
        budget_used = time.monotonic() - start
    csv_rows(a.out / "cold-starts.csv", cold_rows)
    csv_rows(a.out / "configuration-costs.csv", resources)
    write(
        a.out / "g3.json",
        {
            "schema": "el-original-cpu-performance-1",
            "run_id": os.environ["EL_RUN_ID"],
            "release": profile()["release_index_sha256"],
            "N_per_hot_configuration": a.n,
            "budget_seconds": a.budget,
            "wall_seconds": budget_used,
            "selection": selected,
            "configurations": configs,
            "failures": failures,
            "status": "blocked"
            if failures or len(configs) != len(CASES) * len(MODELS) * len(PATHS)
            else "done",
            "unit": "one 256 x 256 prepared frame, one to four registered camera views repeated; target inference always runs",
            "file_boundary": "verified one-target bundle to persisted numerical scores. HTML and readback are separately timed.",
            "cold_boundary": "parent monotonic immediately before fresh interpreter spawn to child monotonic at first persisted/returned result; exit and inner startup recorded separately",
            "upstream": "gsplat 1.5.3 CUDA for new views; separate upstream-frames.csv and transfer receipts",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "cold", "hot", "candidate"))
    p.add_argument("--trajectories", type=Path)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--models", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--method", choices=MODELS)
    p.add_argument("--path", choices=PATHS)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--scene", choices=CASES)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--budget", type=float, default=1800)
    p.add_argument("--expected", type=Path)
    p.add_argument("--deadline", type=float, default=float("inf"))
    args = p.parse_args()
    if args.action == "run":
        run(args)
    else:
        measure(args)
