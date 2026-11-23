"""Prediction-only batch scores and measured native Mac process costs."""

import platform
import subprocess
import sys
import time

import numpy as np
import psutil
import torch

from evidence_lens_vision.contract import file_record, output_dir, put, read, require
from .data import SCOPES, VIEWS, Dataset, signed
from .deploy import load_model, score_grid, score_native
from .train import CATALOG


def batch(a):
    require(
        (a.split == "test") == bool(a.opening_sha256),
        "held-out inference requires the common pre-existing opening",
    )
    method_doc = None
    if a.split == "test":
        from evidence_lens.stage3_gate import require_opening, get_frozen_method

        require(a.store is not None, "the scientific test entry requires the authenticated study store")
        # This precedes opening the dataset: a split manifest itself contains outcome counts.
        marker = require_opening(a.store, CATALOG, opening_sha256=a.opening_sha256, shelf=a.shelf)
        method_doc = get_frozen_method(a.store, marker, shelf=a.shelf)
        require(
            method_doc["models"][a.method]["model_index_sha256"] == a.model_sha256,
            "the official test entry requires the chosen primary model",
        )
    cat = signed(a.catalog, CATALOG)
    out = output_dir(a.output)
    torch.set_num_threads(a.threads)
    stuff = load_model(a.model, a.model_sha256, "cpu")
    _, pre, idx = stuff
    require(idx["method"] == a.method and idx["phase"] == "formal", "incorrect formal model")
    if method_doc:
        wanted = method_doc["models"][a.method]
        require(
            wanted["weights_sha256"] == idx["files"]["weights.safetensors"]["sha256"]
            and wanted["threshold"] == idx["threshold"]
            and wanted["model_id"] == idx["model_id"],
            "method freeze and model disagree",
        )
    ds = Dataset(a.dataset, a.dataset_sha256, a.split, cat, CATALOG, pre)
    if method_doc:
        require(
            ds.document["opening_sha256"] == a.opening_sha256
            and ds.document["method_sha256"] == marker["method_sha256"],
            "held-out transport was sealed under another method or opening",
        )
    return save_scores(stuff, a.model_sha256, ds, out, "cpu", a.opening_sha256)


def save_scores(stuff, model_sha, ds, out, device, opening, canonical_all=()):
    idx = stuff[2]
    rows = []
    originals = {
        (r["family"], r["condition"], r["view"]): r["score"] for r in canonical_all if r["score"] is not None
    }
    for row in ds.rows:
        f, c, v = (row[k] for k in ("family", "condition", "view"))
        rgb = ds.rgb(row) if row["status"] == "ready" else None
        cache = {}
        modes = {}
        for scope in SCOPES:
            mode = row["modes"][scope]
            entry = {
                "source_ids": mode["source_ids"],
                "alias_of": mode["alias_of"],
                "file": None,
                "deploy_sha256": mode["deploy"]["sha256"] if mode["deploy"] else None,
            }
            if rgb is not None:
                if mode["alias_of"]:
                    score = cache[mode["alias_of"]]
                elif (scope == "all" or (c, scope) == ("M2", "two")) and (f, c, v) in originals:
                    score = originals[f, c, v]
                else:
                    x, ok = ds.deployment(row, scope)
                    arrays = score_grid(stuff, rgb, x, ok, device)
                    p = out / f / c / scope / (v + ".npz")
                    p.parent.mkdir(exist_ok=True, parents=True)
                    np.savez_compressed(p, **arrays)
                    score = file_record(p, out)
                entry["file"] = score
                cache[scope] = score
            modes[scope] = entry
        rows.append(
            {
                "family": f,
                "condition": c,
                "view": v,
                "status": row["status"],
                "failure": row["failure"],
                "modes": modes,
                "rgb_sha256": row["rgb"]["sha256"] if row["rgb"] else None,
            }
        )
    result = {
        "schema": "el-stage3-risk-scores-1",
        "split": ds.split,
        "dataset_sha256": ds.sha,
        "method": idx["method"],
        "model_id": idx["model_id"],
        "model_index_sha256": model_sha,
        "weights_sha256": idx["files"]["weights.safetensors"]["sha256"],
        "threshold": idx["threshold"],
        "feature_schema_sha256": idx["feature_schema_sha256"],
        "catalog_sha256": idx["catalog_sha256"],
        "opening_sha256": opening,
        "views": rows,
    }
    put(out / "scores.json", result)
    return result


def warm(a):
    out = output_dir(a.output)
    torch.set_num_threads(a.threads)
    start = time.perf_counter()
    model = load_model(a.model, a.model_sha256)
    load_seconds = time.perf_counter() - start
    rows = []
    for scope in a.scopes:
        for view in VIEWS:
            for repeat in range(a.repetitions):
                where = out / scope / view / f"repeat-{repeat + 1}"
                rec = score_native(
                    a.model,
                    a.model_sha256,
                    a.case,
                    a.case_sha256,
                    view,
                    scope,
                    where,
                    a.threads,
                    model_info=model,
                )
                rows.append(
                    {
                        "scope": scope,
                        "view": view,
                        "repeat": repeat + 1,
                        "seconds": rec["timings"]["complete_seconds"],
                        "receipt": file_record(where / "native-score.json", out),
                        "score": file_record(where / "scores.npz", out),
                    }
                )
    result = {
        "schema": "el-stage3-native-warm-cost-1",
        "model_sha256": a.model_sha256,
        "case_sha256": a.case_sha256,
        "model_load_seconds": load_seconds,
        "views": rows,
    }
    put(out / "warm.json", result)
    return result


def measured_subprocess(command, log):
    start = time.perf_counter()
    peak = 0
    samples = 0
    with log.open("xb") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=stream, shell=False)
        handle = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                allproc = [handle, *handle.children(recursive=True)]
            except psutil.Error:
                allproc = []
            rss = 0
            for child in allproc:
                try:
                    rss += child.memory_info().rss
                except psutil.Error:
                    pass
            peak = max(peak, rss)
            samples += 1
            time.sleep(0.01)
    require(proc.returncode == 0, "native cost process failed; inspect " + str(log))
    return {
        "wall_started_before_python_seconds": time.perf_counter() - start,
        "sampled_peak_process_tree_rss_bytes": peak,
        "rss_samples": samples,
        "sampling_seconds": 0.01,
    }


def costs(a, script):
    out = output_dir(a.output)
    require(
        len(set(a.scopes)) == len(a.scopes)
        and all(s in SCOPES for s in a.scopes)
        and {"one", "all"} <= set(a.scopes)
        and 1 <= a.repetitions <= 3,
        "cost must measure native one/all cases; repetitions 1–3",
    )
    tools = [sys.executable, "-I", "-B", str(script)]
    common = [
        "--model",
        str(a.model.resolve()),
        "--model-sha256",
        a.model_sha256,
        "--case",
        str(a.case.resolve()),
        "--case-sha256",
        a.case_sha256,
        "--threads",
        str(a.threads),
    ]
    cold = {}
    for s in a.scopes:
        wd = out / ("cold-" + s)
        measured = measured_subprocess(
            [*tools, "score-native", *common, "--view", "v01", "--scope", s, "--output", str(wd)],
            out / ("cold-" + s + ".log"),
        )
        cold[s] = {
            **measured,
            "receipt": file_record(wd / "native-score.json", out),
            "score": file_record(wd / "scores.npz", out),
        }
    w = out / "warm"
    wm = measured_subprocess(
        [
            *tools,
            "_warm",
            *common,
            "--output",
            str(w),
            "--repetitions",
            str(a.repetitions),
            "--scopes",
            *a.scopes,
        ],
        out / "warm.log",
    )
    log = read(w / "warm.json")
    scopes = {}
    for s in a.scopes:
        values = [v["seconds"] for v in log["views"] if v["scope"] == s]
        scopes[s] = {
            "timed_native_views": len(values),
            "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.percentile(values, 95)),
        }
        with (
            np.load(out / cold[s]["score"]["path"], allow_pickle=False) as ca,
            np.load(
                w
                / next(
                    v["score"]["path"]
                    for v in log["views"]
                    if v["view"] == "v01" and v["scope"] == s and v["repeat"] == 1
                ),
                allow_pickle=False,
            ) as wa,
        ):
            require(
                set(ca.files) == set(wa.files) == {"risk", "valid", "state"}
                and all(np.array_equal(ca[k], wa[k], equal_nan=True) for k in ca.files),
                "cold/warm native score or threshold state differs",
            )
    result = {
        "schema": "el-stage3-native-cost-1",
        "model_sha256": a.model_sha256,
        "case_sha256": a.case_sha256,
        "platform": platform.platform(),
        "grid": [256, 256],
        "model_threads": a.threads,
        "cold_v01_by_scope": cold,
        "warm_process": wm,
        "warm_model_load_seconds": log["model_load_seconds"],
        "warm_views": scopes,
        "warm_receipt": file_record(w / "warm.json", out),
        "cold_warm_v01_equal": True,
    }
    put(out / "costs.json", result)
    return result
