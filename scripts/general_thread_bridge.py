"""Registered, label-blind comparison of the original four-thread scores and the selected Mac settings."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PLAN_SHA = "577eb49bcf71687163075c8ded4dde1662f53a877d899839e5ded24c5af250fb"
MIXED = "selected_MS-ft8_with_archived_others"
WHEEL_JSON = {"profile.json", "MS-L1/model.json", "MS-L1/preprocessing.json", "MS-L1/coefficients.json"}


def snapshot(argv):
    """Restore the wheel JSON that is delivered as a separate recorder input."""
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", type=Path)
    parser.add_argument("python", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    actual = {p.relative_to(args.assets).as_posix() for p in args.assets.rglob("*") if p.is_file()}
    if actual != WHEEL_JSON:
        raise ValueError("unexpected wheel JSON inputs")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        shutil.copytree(Path(__file__).resolve().parents[1] / "src", src)
        assets = src / "evidence_lens_bundle/assets"
        assets.parent.chmod(0o755)
        shutil.copytree(args.assets, assets, dirs_exist_ok=True)
        env = {**os.environ, "PYTHONPATH": str(src), "PYTHONDONTWRITEBYTECODE": "1"}
        return subprocess.run([args.python, "-B", __file__, *args.command], env=env).returncode


def program_identities():
    scripts = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((scripts / name).read_bytes()).hexdigest()
        for name in ("general_thread_bridge.py", "general_t1_bridge_roster.py")
    }


def study_plan(path):
    from evidence_lens_bundle.common import json_bytes, require, sha

    raw = path.read_bytes()
    require(sha(raw) == PLAN_SHA, "study plan differs from the prior registered bytes")
    return json_bytes(raw)


def register(args):
    from evidence_lens_bundle.bridge import configuration
    from evidence_lens_bundle.common import (
        digest,
        json_bytes,
        no_replace_file,
        encoded,
        read_file,
        require,
        sha,
    )
    from general_t1_bridge_roster import recorder_member

    plan = study_plan(args.plan)
    g3 = plan["development_selection"]
    selection = {
        "run_id": g3["run_id"],
        "member": g3["member"],
        "sha256": g3["sha256"],
        "manifest_sha256": digest(args.selection_manifest_sha256, "independent G3 manifest"),
    }
    _, chosen = recorder_member(args.store, selection, "selection.json")
    require(
        chosen == g3["development_choices"]
        and chosen["MS-ft"]["threads"] == 8
        and chosen["MS-L1"]["threads"] == 4,
        "independent development selection differs",
    )
    config_raw = args.g3_config.read_bytes()
    require(sha(config_raw) == g3["controller_config_sha256"], "G3 controller declaration differs")
    declared = json_bytes(config_raw)
    g3_folder = args.store / "runs" / selection["run_id"]
    g3_manifest = json_bytes(read_file(g3_folder, "manifest.json"))
    resolved_raw = read_file(g3_folder, "config.resolved.yaml")
    resolved = json_bytes(resolved_raw)
    require(
        any(
            f["path"] == "config.resolved.yaml" and f["sha256"] == sha(resolved_raw)
            for f in g3_manifest["files"]
        ),
        "G3 config is outside the authenticated selection run",
    )
    require(
        all(resolved.get(k) == value for k, value in declared.items() if k != "recording")
        and all(resolved["recording"].get(k) == value for k, value in declared["recording"].items()),
        "G3 selection run used a different declared configuration",
    )
    _, four = recorder_member(args.store, plan["primary_configuration"], "engineering-config.json")
    require(
        four == configuration(plan["original_threads"], plan["cache_bytes"]),
        "primary engineering program or configuration differs",
    )
    eight = configuration(plan["engineering_threads"], plan["cache_bytes"])
    require(eight == {**four, "threads": 8}, "selected settings modified other engineering inputs")
    result = {
        "schema": "el-t1-selected-thread-registration-1",
        "run_id": os.getenv("EL_RUN_ID"),
        "plan_sha256": PLAN_SHA,
        "plan": plan,
        "worker_sources_sha256": program_identities(),
        "selection": selection,
        "sealed_g3_resolved_config_sha256": sha(resolved_raw),
        "engineering_config": eight,
        "T1_inputs_opened_at_registration": 0,
    }
    no_replace_file(args.out, encoded(result))


def file_outputs(bundle, method, models_dir, folder, threads):
    import numpy as np
    from evidence_lens_bundle.cli import main as cli
    from evidence_lens_bundle.common import json_bytes, member, require, sha
    from evidence_lens_bundle.numerics import checked_arrays

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
                "--models",
                str(models_dir),
                "--threads",
                str(threads),
                "--out",
                str(folder),
            ]
        )
        == 0,
        "real file scoring command failed",
    )
    raw = (folder / "scores.json").read_bytes()
    receipt = json_bytes(raw)
    actual_threads = receipt["cost"]["configured_neural_threads"]
    require(
        actual_threads == (threads if method in ("MS-ft", "MS-frozen") else None),
        "file scorer used a different neural thread setting",
    )
    require(
        receipt["method"] == method
        and receipt["bundle_sha256"] == bundle.sha256
        and tuple(v["id"] for v in receipt["views"]) == bundle.targets,
        "wrong file outputs",
    )
    arrays = {}
    for pos, view in enumerate(receipt["views"]):
        raw_array = member(folder, view["arrays"], expected=f"views/{pos:03d}/scores.npz")
        with np.load(io.BytesIO(raw_array), allow_pickle=False) as loaded:
            arrays[view["id"]] = checked_arrays(dict(loaded), receipt["model"]["threshold"])
    return arrays, {
        "scores_sha256": sha(raw),
        "configured_neural_threads": actual_threads,
        "model_index_sha256": receipt["model"]["model_index_sha256"],
    }


def resident_outputs(bundle, method, sources, binding, metadata, models_dir, cfg, capacity):
    from evidence_lens_bundle.common import require
    from evidence_lens_bundle.numerics import CHANNELS, checked_arrays
    from evidence_lens_bundle.resident import load_scorer

    engine = load_scorer(method, models_dir=models_dir, threads=cfg["threads"], cache_bytes=capacity)
    torch_threads = None
    if method in ("MS-frozen", "MS-ft"):
        import torch

        torch_threads = torch.get_num_threads()
        require(torch_threads == cfg["threads"], "resident scorer did not honor the selected thread setting")
    context = engine.prepare_sources(sources, version_binding=binding)
    outputs, frame_inputs = {}, {}
    for view in bundle.targets:
        result = engine.score_frame(
            context, bundle.frame(view), bundle.cameras[view], frame_meta=metadata[view]
        )
        outputs[view] = checked_arrays({k: result[k] for k in CHANNELS}, engine.threshold)
        frame_inputs[view] = result["identity"]["frame_input_sha256"]
    return outputs, {
        "source_context_sha256": context.key,
        "frame_inputs_sha256": frame_inputs,
        "observed_neural_threads": torch_threads,
        "cache": engine.cache_info(),
    }


def compare_view(groups, old_view, obj, condition, view, plan):
    from evidence_lens_bundle.bindings import array_sha
    from evidence_lens_bundle.common import encoded, profile, require, sha
    from evidence_lens_bundle.numerics import compare_arrays, matched_sets

    methods = plan["methods"]
    groups[MIXED] = {**groups["archived4"], "MS-ft": groups["resident_on8"]["MS-ft"]}
    identities = {
        path: {m: {k: array_sha(array) for k, array in g[m].items()} for m in methods}
        for path, g in groups.items()
    }
    primary = old_view["comparisons"]["old_to_on"]["methods"]
    require(
        all(identities["archived4"][m] == primary[m]["old_sha256"] for m in methods),
        "archived bits or recovered reasons differ from the sealed primary bridge",
    )
    signatures = {path: sha(encoded(ids)) for path, ids in identities.items()}
    thresholds = {m: s["threshold"] for m, s in profile()["models"].items()}
    pairs, records, previously_compared = {}, {}, {}
    for left, right in [*plan["pairings"], ["archived4", MIXED]]:
        name = left + "_to_" + right
        key = (signatures[left], signatures[right])
        if key not in previously_compared:
            previously_compared[key] = name
            records[name] = {
                "methods": {
                    m: compare_arrays(groups[left][m], groups[right][m], thresholds[m]) for m in methods
                },
                "matched_acceptance": matched_sets(
                    groups[left], groups[right], obj, condition, view, thresholds["MS-L1"]
                ),
            }
        pairs[name] = previously_compared[key]
    return {
        "status": "compared",
        "path_array_group_sha256": signatures,
        "pair_to_comparison": pairs,
        "comparisons": records,
    }


def compare_cell(cell, old_cell, cfg, plan, models_dir, folder):
    from evidence_lens_bundle.bindings import binding_from_bundle, frame_meta_from_bundle
    from evidence_lens_bundle.bridge_history import historical_case
    from evidence_lens_bundle.common import encoded, require, sha

    obj, layout = cell["object_id"], cell["layout"]
    if "terminal_failure" in cell:
        require(
            old_cell["terminal_failure"] == cell["terminal_failure"], "original failure disposition changed"
        )
        return {**cell, "status": "not_comparable", "views": old_cell["views"]}
    bundle, old_reader, history = historical_case(cell, folder, models_dir, t1=True)
    primary_metadata = {
        **bundle.meta,
        "creation": {**bundle.meta["creation"], "run_id": plan["primary_bridge"]["run_id"]},
    }
    require(
        history["archived_threads"] == plan["original_threads"]
        and old_cell["bundle_sha256"] == sha(encoded(primary_metadata))
        and set(old_cell["views"]) == set(bundle.targets),
        "historical settings, deployment bytes or target roster differ from the primary bridge",
    )
    binding = binding_from_bundle(bundle)
    sources = {name: bundle.source(name) for name in bundle.order}
    metadata = {
        v: frame_meta_from_bundle(bundle, v, binding=binding, sequence_id=obj + "/" + layout)
        for v in bundle.targets
    }
    by_path = {"file8": {}, "resident_off8": {}, "resident_on8": {}}
    engines = {p: {} for p in by_path}
    for method in plan["methods"]:
        by_path["file8"][method], engines["file8"][method] = file_outputs(
            bundle, method, models_dir, folder / ("file-" + method), cfg["threads"]
        )
        for name, cap in (("resident_off8", 0), ("resident_on8", cfg["cache_bytes"])):
            by_path[name][method], engines[name][method] = resident_outputs(
                bundle, method, sources, binding, metadata, models_dir, cfg, cap
            )
    results = {}
    for v in bundle.targets:
        require(old_cell["views"][v]["status"] == "compared", "unexpected archived view disposition")
        groups = {p: {m: a[v] for m, a in group.items()} for p, group in by_path.items()}
        groups["archived4"] = old_reader(v)
        require(groups["archived4"] is not None, "registered archived score absent")
        results[v] = compare_view(groups, old_cell["views"][v], obj, bundle.condition, v, plan)
    return {
        "object_id": obj,
        "layout": layout,
        "bundle_sha256": bundle.sha256,
        "status": "compared",
        "primary_bundle_sha256_with_original_creation_run_id": old_cell["bundle_sha256"],
        "historical": history,
        "engineering_readback": engines,
        "views": results,
    }


def summarize(plan, cells):
    from evidence_lens_bundle.common import require
    from evidence_lens_bundle.numerics import CHANNELS

    views = [v for cell in cells for v in cell["views"].values()]
    available = [v for v in views if v["status"] == "compared"]
    methods = plan["methods"]
    denominators = {
        "registered_cells": len(cells),
        "failed_cells": sum("terminal_failure" in c for c in cells),
        "registered_method_views": len(views) * len(methods),
        "available_method_views": len(available) * len(methods),
        "missing_method_views": (len(views) - len(available)) * len(methods),
    }
    require(
        all(
            denominators[k] == plan[k]
            for k in ("registered_method_views", "available_method_views", "missing_method_views")
        ),
        "registered denominator changed",
    )
    pairs = {}
    for left, right in [*plan["pairings"], ["archived4", MIXED]]:
        name = left + "_to_" + right
        metrics = {}
        comparisons = [v["comparisons"][v["pair_to_comparison"][name]] for v in available]
        for method in methods:
            per_view = [d["methods"][method] for d in comparisons]
            anchors = [row for d in comparisons for row in d["matched_acceptance"]["methods"][method]]
            paired = [a for a in anchors if a["old_attainable"] and a["new_attainable"]]
            signed_min = [d["min_signed_new_minus_old"] for d in per_view if d["common_finite"]]
            signed_max = [d["max_signed_new_minus_old"] for d in per_view if d["common_finite"]]
            metrics[method] = {
                "scheduled_method_views": len(views),
                "compared_method_views": len(per_view),
                "unavailable_method_views": len(views) - len(per_view),
                "bit_exact": sum(d["bit_exact"] for d in per_view),
                "changed_pixels": {k: sum(d["changed_pixels"][k] for d in per_view) for k in CHANNELS},
                "min_signed_delta": min(signed_min, default=None),
                "max_signed_delta": max(signed_max, default=None),
                "max_absolute_delta": max(
                    (d["max_abs_difference_common_finite"] for d in per_view if d["common_finite"]),
                    default=None,
                ),
                "threshold_crossings_common_finite": sum(
                    d["threshold_crossings_common_finite"] for d in per_view
                ),
                "attainability_changes": sum(a["old_attainable"] != a["new_attainable"] for a in anchors),
                "attainable_anchors": len(paired),
                "attainable_anchors_changing_members": sum(
                    bool(a["added_pixels"] or a["removed_pixels"]) for a in paired
                ),
                "at_20pct_added_pixels": sum(a["added_pixels"] for a in paired if a["anchor"] == "20%_canvas"),
                "at_20pct_removed_pixels": sum(
                    a["removed_pixels"] for a in paired if a["anchor"] == "20%_canvas"
                ),
            }
        pairs[name] = {
            "methods": metrics,
            "bit_exact_method_views": sum(m["bit_exact"] for m in metrics.values()),
            "changed_common_domain_pixels": sum(
                d["matched_acceptance"]["changed_common_mask"] for d in comparisons
            ),
        }
    return {**denominators, "comparisons": pairs}


def run(args):
    from evidence_lens.scoring_io import new_output
    from evidence_lens_bundle.bridge import configuration, environment
    from evidence_lens_bundle.bridge_history import check_joint_closure
    from evidence_lens_bundle.common import json_bytes, require, sha, write
    from general_t1_bridge_roster import recorder_member

    raw = args.registration.read_bytes()
    require(sha(raw) == args.registration_sha256, "registration SHA-256 differs")
    reg = json_bytes(raw)
    registration_pin = {
        "run_id": reg["run_id"],
        "member": "payload/registration.json",
        "sha256": args.registration_sha256,
        "manifest_sha256": args.registration_manifest_sha256,
    }
    _, sealed_registration = recorder_member(args.store, registration_pin, "registration.json")
    require(sealed_registration == reg, "registration was not sealed before the supplemental run")
    plan = study_plan(args.plan)
    require(
        reg["schema"] == "el-t1-selected-thread-registration-1"
        and reg["plan_sha256"] == PLAN_SHA
        and reg["plan"] == plan
        and reg["worker_sources_sha256"] == program_identities(),
        "registered program or study plan changed",
    )
    _, selection = recorder_member(args.store, reg["selection"], "selection.json")
    require(
        selection == plan["development_selection"]["development_choices"],
        "performance selection changed since registration",
    )
    cfg = configuration(plan["engineering_threads"], plan["cache_bytes"])
    require(cfg == reg["engineering_config"], "engineering scorer changed after registration")
    _, four = recorder_member(args.store, plan["primary_configuration"], "engineering-config.json")
    require(four == {**cfg, "threads": plan["original_threads"]}, "four-thread baseline configuration differs")
    _, roster = recorder_member(args.store, plan["primary_roster"], "roster.json")
    _, primary = recorder_member(args.store, plan["primary_bridge"], "numerical-bridge.json")
    closure = check_joint_closure(roster["joint_closure"], roster["cells"])
    require(
        closure["run_id"] == plan["closure"]["run_id"]
        and closure["matrix_sha256"] == plan["closure"]["sha256"]
        and closure["manifest_sha256"] == plan["closure"]["manifest_sha256"]
        and roster["stage"] == primary["stage"] == "sealed_T1"
        and primary["joint_closure"] == closure
        and primary["manifest_sha256"] == plan["primary_roster"]["sha256"]
        and primary["engineering_config"] == four,
        "primary bridge does not belong to the registered matrix",
    )
    olds = {(c["object_id"], c["layout"]): c for c in primary["cells"]}
    cells = []
    for cell in roster["cells"]:
        with tempfile.TemporaryDirectory() as tmp:
            cells.append(
                compare_cell(cell, olds[cell["object_id"], cell["layout"]], cfg, plan, args.models, Path(tmp))
            )
    summary = summarize(plan, cells)
    result = {
        "schema": "el-t1-selected-thread-bridge-1",
        "run_id": os.getenv("EL_RUN_ID"),
        "registration": registration_pin,
        "preregistration": reg,
        "joint_closure": closure,
        "hardware_and_runtime": environment(),
        "target_reference_inputs": 0,
        "scientific_labels_or_mask_read": False,
        "summary": summary,
        "cells": cells,
    }
    with new_output(args.out) as dest:
        write(dest / "selected-thread-bridge.json", result)
    print(json.dumps({k: v for k, v in summary.items() if k != "comparisons"}))


def main():
    if sys.argv[1:2] == ["snapshot"]:
        return snapshot(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register")
    reg.add_argument("--selection-manifest-sha256", required=True)
    reg.add_argument("--g3-config", type=Path, required=True)
    compare = sub.add_parser("run")
    compare.add_argument("--registration", type=Path, required=True)
    compare.add_argument("--registration-sha256", required=True)
    compare.add_argument("--registration-manifest-sha256", required=True)
    compare.add_argument("--models", type=Path, required=True)
    for command in (reg, compare):
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--store", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (register if args.command == "register" else run)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
