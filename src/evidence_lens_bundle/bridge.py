"""Compare released CPU scores with resident/cache output without opening target references."""

import argparse
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np

from evidence_lens.scoring_io import new_output
from .bindings import binding_from_bundle, frame_meta_from_bundle
from .bridge_history import check_joint_closure, historical_case, pinned
from .bundle import Bundle
from .cli import main as cli
from .common import (
    RELEASE_INDEX_SHA256,
    digest,
    encoded,
    exact_keys,
    json_bytes,
    member,
    no_replace_file,
    profile,
    require,
    sha,
    text,
    write,
)
from .numerics import CHANNELS, METHODS, checked_arrays, compare_arrays, matched_sets
from .resident import load_scorer

MODULES = (
    "evidence_lens_bundle.resident",
    "evidence_lens_bundle.bindings",
    "evidence_lens_bundle.bundle",
    "evidence_lens_bundle.scoring",
    "evidence_lens_bundle.cli",
    "evidence_lens_bundle.common",
    "evidence_lens_bundle.report",
    "evidence_lens_bundle.legacy",
    "evidence_lens.features",
    "evidence_lens.l1",
    "evidence_lens.multisource",
    "evidence_lens.multisource_io",
    "evidence_lens.multisource_model",
    "evidence_lens.scoring_io",
    "evidence_lens_multirisk.deploy",
    "evidence_lens_multirisk.model",
    "evidence_lens_vision.model",
    "evidence_lens_bundle.bridge",
    "evidence_lens_bundle.bridge_history",
    "evidence_lens_bundle.numerics",
    "evidence_lens_bundle.frozen_reasons",
)


def configuration(threads, cache_bytes):
    require(
        type(threads) is int and 1 <= threads <= 256 and type(cache_bytes) is int and cache_bytes > 0,
        "positive selected thread count and cache budget required",
    )
    return {
        "schema": "el-engineering-bridge-config-1",
        "threads": threads,
        "cache_bytes": cache_bytes,
        "release_index_sha256": RELEASE_INDEX_SHA256,
        "default_method": "MS-L1",
        "default_engine": "file",
        "source_sha256": {m: sha(Path(importlib.util.find_spec(m).origin).read_bytes()) for m in MODULES},
    }


def environment():
    versions = {}
    for name in ("numpy", "pillow", "torch", "safetensors", "evidence-lens"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "python": sys.version,
        "versions": versions,
    }


def original_file_scores(bundle, method, models_dir, folder, threads):
    folder.mkdir(parents=True)
    score_dir = folder / "scores"
    require(
        cli(
            [
                "score",
                "--bundle",
                str(bundle.root),
                "--bundle-sha256",
                bundle.sha256,
                "--model",
                method,
                "--threads",
                str(threads),
                "--models",
                str(models_dir),
                "--out",
                str(score_dir),
            ]
        )
        == 0,
        "file reference command failed",
    )
    raw = (score_dir / "scores.json").read_bytes()
    receipt = json_bytes(raw)
    html = folder / "report.html"
    require(
        cli(["report", "--scores", str(score_dir), "--scores-sha256", sha(raw), "--out", str(html)]) == 0,
        "file HTML command failed",
    )
    arrays = {}
    for i, view in enumerate(receipt["views"]):
        data = member(score_dir, view["arrays"], expected=f"views/{i:03d}/scores.npz")
        with np.load(io.BytesIO(data), allow_pickle=False) as a:
            arrays[view["id"]] = checked_arrays(dict(a), receipt["model"]["threshold"])
    return arrays, {
        "scores_sha256": sha(raw),
        "html_sha256": sha(html.read_bytes()),
        "local_html": html.name,
        "input_bundle_sha256": bundle.sha256,
        "frozen_model": receipt["model"],
        "counts": receipt["counts"],
    }


def run_cell(cell, stage, models_dir, cfg, dest, tmp):
    obj = text(cell["object_id"], "object ID")
    layout = text(cell["layout"], "layout")
    require(layout in ("M2-fixed", "M4-fixed"), "numerical bridge only consumes fixed M2/M4 layouts")
    dev = stage == "original_development"
    old_reader = None
    if dev:
        exact_keys(cell, ("object_id", "layout", "bundle"), "original control")
        pinned(cell["bundle"], "bundle.json")
        bundle = Bundle(cell["bundle"]["directory"], cell["bundle"]["sha256"])
        history = {"provenance": "Current same-host file reference command and its original threshold."}
    else:
        exact_keys(
            cell,
            ("object_id", "layout", "case", "readback", "geometry_bundle", "historical"),
            "frozen deployment score cell",
        )
        bundle, old_reader, history = historical_case(cell, tmp, models_dir, t1=stage == "sealed_T1")
    require(layout == bundle.condition + "-fixed", "wrong layout for this evidence")
    source = {s: bundle.source(s) for s in bundle.order}
    binding = binding_from_bundle(bundle)
    pending = {v: {"old": {}, "off": {}, "on": {}, "inputs": {}} for v in bundle.targets}
    refs, caches = {}, {}
    for method in METHODS:
        if dev:
            arrays, refs[method] = original_file_scores(
                bundle, method, models_dir, dest / method, cfg["threads"]
            )
            for v, a in arrays.items():
                pending[v]["old"][method] = a
        for label, cap in (("off", 0), ("on", cfg["cache_bytes"])):
            engine = load_scorer(method, models_dir=models_dir, threads=cfg["threads"], cache_bytes=cap)
            ctx = engine.prepare_sources(source, version_binding=binding)
            for view in bundle.targets:
                meta = frame_meta_from_bundle(bundle, view, binding=binding, sequence_id=obj + "/" + layout)
                out = engine.score_frame(ctx, bundle.frame(view), bundle.cameras[view], frame_meta=meta)
                pending[view][label][method] = {k: out[k] for k in CHANNELS}
                pending[view]["inputs"][method] = out["identity"]["frame_input_sha256"]
            caches[method + "_" + label] = {
                "source_context_sha256": ctx.key,
                "cache": engine.cache_info(),
                "prepare_seconds": ctx.prepare_seconds,
            }
    results = {}
    for view, row in pending.items():
        if old_reader:
            old = old_reader(view)
            if old is None:
                require(bundle.frame(view) is None, "missing archived scoring on a ready prediction")
                results[view] = {
                    "status": "historical_render_failed",
                    "reason": bundle.producer["frames"][view]["reason"],
                }
                continue
            row["old"] = old
        comparisons = {}
        for label, (a, b) in {
            "old_to_off": ("old", "off"),
            "old_to_on": ("old", "on"),
            "off_to_on": ("off", "on"),
        }.items():
            comparisons[label] = {
                "methods": {
                    m: compare_arrays(row[a][m], row[b][m], profile()["models"][m]["threshold"])
                    for m in METHODS
                },
                "matched_acceptance": matched_sets(
                    row[a], row[b], obj, bundle.condition, view, profile()["models"]["MS-L1"]["threshold"]
                ),
            }
        results[view] = {"status": "compared", "inputs": row["inputs"], "comparisons": comparisons}
    return {
        "object_id": obj,
        "layout": layout,
        "bundle_sha256": bundle.sha256,
        "reference": history,
        "files": refs,
        "resident": caches,
        "views": results,
    }


def run(manifest, expected, models_dir, output):
    raw = manifest.read_bytes()
    require(sha(raw) == expected, "input roster SHA-256 differs")
    m = json_bytes(raw)
    exact_keys(m, ("schema", "stage", "configuration", "joint_closure", "cells"), "label-blind manifest")
    require(
        m["schema"] == "el-numerical-bridge-input-1"
        and m["stage"] in ("original_development", "legacy_control", "sealed_T1"),
        "unsupported bridge stage",
    )
    _, cfg = pinned(m["configuration"], "engineering-config.json")
    require(
        configuration(cfg["threads"], cfg["cache_bytes"]) == cfg,
        "frozen engineering sources or settings differ",
    )
    require(isinstance(m["cells"], list) and m["cells"], "empty fixed evidence roster")
    closure = None
    if m["stage"] == "sealed_T1":
        closure = check_joint_closure(m["joint_closure"], m["cells"])
    else:
        require(m["joint_closure"] is None, "development control cannot claim T1 joint closure")
    cells = []
    with new_output(output) as dest:
        for n, c in enumerate(m["cells"]):
            if "terminal_failure" in c:
                require(
                    m["stage"] == "sealed_T1", "only frozen T1 can supply an unscored terminal disposition"
                )
                exact_keys(c, ("object_id", "layout", "terminal_failure"), "closed cell failure")
                exact_keys(c["terminal_failure"], ("reason", "receipt_sha256"), "frozen failure")
                text(c["terminal_failure"]["reason"], "terminal failure")
                digest(c["terminal_failure"]["receipt_sha256"], "terminal disposition")
                cells.append(
                    {
                        **c,
                        "status": "not_comparable",
                        "views": {
                            f"v{i:02d}": {
                                "status": "failed_upstream",
                                "methods": list(METHODS),
                                "reason": c["terminal_failure"]["reason"],
                            }
                            for i in range(1, 7)
                        },
                    }
                )
                continue
            with tempfile.TemporaryDirectory() as temporary:
                cells.append(
                    run_cell(c, m["stage"], models_dir, cfg, dest / f"cell-{n:02d}", Path(temporary))
                )
        views = [v for cell in cells for v in cell["views"].values() if v["status"] == "compared"]
        count = len(views) * len(METHODS)
        scheduled_views = sum(len(cell["views"]) for cell in cells)
        summary = {
            "registered_cells": len(cells),
            "failed_cells": sum("terminal_failure" in c for c in cells),
            "scheduled_method_views": scheduled_views * len(METHODS),
            "compared_method_views": count,
            "unavailable_method_views": (scheduled_views - len(views)) * len(METHODS),
        }
        for name in ("old_to_off", "old_to_on", "off_to_on"):
            summary[name + "_bit_exact"] = sum(
                v["comparisons"][name]["methods"][method]["bit_exact"] for v in views for method in METHODS
            )
        result = {
            "schema": "el-numerical-bridge-1",
            "run_id": os.getenv("EL_RUN_ID"),
            "stage": m["stage"],
            "manifest_sha256": expected,
            "configuration_sha256": m["configuration"]["sha256"],
            "joint_closure": closure,
            "model_release_sha256": RELEASE_INDEX_SHA256,
            "hardware_and_runtime": environment(),
            "engineering_config": cfg,
            "summary": summary,
            "target_reference_inputs": 0,
            "true_labels_or_evaluator_masks_read": False,
            "cells": cells,
        }
        write(dest / "numerical-bridge.json", result)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("freeze-config", help="Pin this engineering build and its settings before comparison.")
    a.add_argument("--threads", type=int, default=4)
    a.add_argument("--cache-bytes", type=int, default=64 << 20)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("run", help="Compare deployment-only arrays on the signed input roster.")
    for key in ("manifest", "models", "out"):
        b.add_argument("--" + key, type=Path, required=True)
    b.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze-config":
        data = encoded(configuration(args.threads, args.cache_bytes))
        no_replace_file(args.out, data)
        print(json.dumps({"sha256": sha(data)}))
        return 0
    summary = run(args.manifest, args.manifest_sha256, args.models, args.out)
    print(json.dumps(summary))
    return 0 if summary["off_to_on_bit_exact"] == summary["compared_method_views"] else 2


if __name__ == "__main__":
    sys.exit(main())
