"""Project the verified, separately opened Stage C experiment into a portable, hash-bound scientific report."""

import argparse
import csv
import re
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path

from evidence_lens._bundle import load
from evidence_lens.formal_inputs import compact_pin, require
from evidence_lens.multisource import CONDITIONS, METHODS, SCOPES, TARGETS
from evidence_lens.scoring_io import new_output, write_json
from evidence_lens.stage3_gate import get_frozen_method, require_opening
from stage3_risk_public_tables import table
from stage3_risk_sources import CATALOG, STUDY, asset
from vision_sources import find_run, json_member, member, signed

ORIGINAL = frozenset(
    (
        "family-metrics.csv",
        "region-metrics.csv",
        "risk-coverage.csv",
        "paired-reconstruction.csv",
        "source-ablation.csv",
        "matched-all-scope-ap.csv",
        "resources.csv",
        "study-summary.json",
    )
)
FROZEN = ("model_id", "model_index_sha256", "weights_sha256", "threshold")


def proof(run, name, digest=None):
    _, spec = member(run, name, digest)
    return {**compact_pin(run.pin), "member": name, "bytes": spec["bytes"], "sha256": spec["sha256"]}


def copy_original_tables(run, evaldoc, dst):
    inf = evaldoc["public_numerical"]
    require(inf["path"] == "public-numerical/manifest.json", "unexpected numerical source")
    idx, _ = json_member(run, "payload/" + inf["path"], inf)
    files = idx["files"]
    require(
        idx["source_run_id"] == run.pin["run_id"]
        and idx["schema"] == "el-stage3-public-numerical-manifest-1"
        and len(files) == len(ORIGINAL)
        and {r["path"] for r in files} == ORIGINAL,
        "missing, duplicate or unregistered confirmation table",
    )
    for spec in files:
        path, _ = member(run, "payload/public-numerical/" + spec["path"], spec)
        shutil.copyfile(path, dst / spec["path"])
    shutil.copyfile(run.open("payload/" + inf["path"]), dst / "confirmation-manifest.json")
    summary = load(dst / "study-summary.json")
    require(
        summary["scientific_run_id"] == run.pin["run_id"]
        and summary["split"] == "test"
        and all(
            summary[k] == evaldoc[k]
            for k in ("models", "families", "dataset_sha256", "method_sha256", "opening_sha256")
        ),
        "held-out public rows refer to different inputs",
    )
    return summary


def audit_cells(datasets, costs, opening):
    require(
        costs["schema"] == "el-stage3-joint-acquisition-costs-1"
        and costs["opening_sha256"] == opening
        and costs["catalog_sha256"] == CATALOG,
        "independent complete acquisition ledger required",
    )
    formal = {(c["family"], c["condition"]): (split, c) for split, d in datasets.items() for c in d["cells"]}
    require(
        len(formal) == sum(len(d["cells"]) for d in datasets.values()) == 32,
        "expected 32 unique production acquisitions",
    )
    witnessed = {}
    for r in costs["trials"]:
        if r["phase"] != "production":
            continue
        key = r["family"], r["condition"]
        require(key in formal and key not in witnessed, "missing or repeated formal cost key")
        split, c = formal[key]
        require(
            r["split"] == split
            and r["selection_sha256"] == c["selection_sha256"]
            and r["selection"] == c["selected"]
            and (r["selection_status"] == "selected") == (c["failure"] is None),
            "selection and independent acquisition ledger disagree",
        )
        q = c.get("cost") or {}
        if not c["failure"]:
            pairs = {
                "actual_updates": "gaussian_optimization_steps",
                "source_pixels": "sampled_source_pixels",
                "optimization_seconds": "optimization_gpu_synchronized_seconds",
                "generation_stage_seconds": "reconstruction_worker_seconds",
                "render_stage_seconds": "render_worker_seconds",
                "generation_wall_seconds": "reconstruction_recorder_wall_seconds",
                "render_wall_seconds": "render_recorder_wall_seconds",
                "gen_rss_peak_bytes": "reconstruction_peak_tree_rss_bytes",
                "render_rss_peak_bytes": "render_peak_tree_rss_bytes",
            }
            require(
                all(r[a] == q[b] for a, b in pairs.items()),
                "independent original resource measurements disagree",
            )
        witnessed[key] = r
    engineering = Counter(r["phase"] for r in costs["trials"] if r["phase"] != "production")
    require(
        set(witnessed) == set(formal) and engineering == {"engineering-pilot": 4, "engineering-repeat": 1},
        "incomplete production or engineering resource cohort",
    )


def paired_quality(summary, pairs):
    measures = 0
    for f, expected in summary["paired_reconstruction_by_family"].items():
        require(f in pairs, "independent held-out or development family is missing")
        for c in CONDITIONS:
            raw = pairs[f]["paired"][c]
            expected_cond = expected[c]
            for source, target in (
                ("Y", "paired_Y"),
                ("ZY", "paired_ZY"),
                ("rgb_abs_sum", "paired_rgb_abs_sum"),
                ("z_abs_sum", "paired_z_abs_sum"),
                ("rgb_mae", "rgb_mae"),
                ("depth_mae_m", "depth_mae_m"),
                ("depth_event_rate", "depth_error_fraction"),
                ("rgb_event_rate", "rgb_error_fraction"),
            ):
                a, b = raw[source], expected_cond[target]
                require(
                    (a is None and b is None)
                    or (a is not None and b is not None and math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-13)),
                    "independent paired numerator or metric differs for " + f + "/" + c + "/" + source,
                )
                measures += 1
            require(
                raw["E"] == expected["paired_E"] and raw["Z"] == expected["paired_Z"],
                "paired RGB or depth denominator differs",
            )
            measures += 2
    return measures


def native_rows(parity, originals):
    lookup = {m: {(r["condition"], r["view"]): r for r in j["views"]} for m, j in originals.items()}
    out = []
    for r in parity["results"]:
        m, c, v, s = (r[k] for k in ("method", "condition", "view", "scope"))
        native = lookup[m][c, v]["modes"][s]
        q = r["comparison"] or {}
        require(
            r["cpu"] == native["file"] and r["alias_of"] == native["alias_of"], "unexpected CPU measurement"
        )
        row = {k: r[k] for k in ("method", "condition", "view", "scope", "alias_of", "status", "threshold")}
        row.update(launched_process_wall_seconds=native.get("external_process_wall_seconds"))
        for k in (
            "pixels",
            "producer_valid",
            "cpu_valid",
            "common_finite",
            "support_disagreement",
            "state_disagreement",
            "bitwise_mismatches_on_common",
            "threshold_crossings",
            "producer_low_to_cpu_high",
            "producer_high_to_cpu_low",
            "max_abs",
            "mean_abs",
            "max_float32_ulp",
            "bounded_numeric_pass",
            "state_exact",
            "bitwise_and_support_exact",
        ):
            row[k] = q.get(k)
        for eps in ("1e-06", "1e-05", "1e-04"):
            for side in ("producer", "cpu"):
                row[side + "_near_" + eps] = (
                    q.get("near_threshold_on_common_finite", {}).get(eps, {}).get(side)
                )
        out.append(row)
    require(
        len(out) == len(METHODS) * len(CONDITIONS) * len(TARGETS) * len(SCOPES)
        and sum(r["alias_of"] is None for r in out) == 120,
        "incomplete registered CPU validation table",
    )
    return out


def readable(summary, learn, parity, acquisition, qa, case, per_family, strata, timings, neural_cpu):
    def v(x, percent=False):
        return "undefined" if x is None else (f"{100 * x:.2f}%" if percent else f"{x:.4f}")

    addr = {(r["condition"], r["scope"], r["method"]): r for r in summary["family_equal_overall"]}
    method = learn["decision"]["default"]
    s = [
        "# Four risk methods on two multiview Gaussian reconstructions",
        "",
        f"The frozen deployment choice is **{method}**. The four held-out procedural families were opened after the method and operating thresholds were recorded. All targets are native 256 × 256 images. M2 trains one scene from the two endpoint cameras. M4 starts the same scene and optimizer from the same parameters with all four source cameras.",
        "",
        f"With all generator sources available to the scorer, {method} reached common-domain AP {v(addr['M2', 'all', method]['common_ap']['mean'])} on M2 and {v(addr['M4', 'all', method]['common_ap']['mean'])} on M4. Its low-risk region covered {v(addr['M2', 'all', method]['low_coverage']['mean'], True)} and {v(addr['M4', 'all', method]['low_coverage']['mean'], True)} of valid reference pixels. Error incidence within those low-risk regions was {v(addr['M2', 'all', method]['low_error']['mean'], True)} and {v(addr['M4', 'all', method]['low_error']['mean'], True)}, respectively.",
        "",
        "## Held-out operating points",
        "",
        "Average precision (AP) compares the four methods on pixels where the reference and prediction are RGB-evaluable and all four scores are finite at that scorer scope. Within a family, views with both label classes receive equal weight. The final mean gives equal weight to families with defined AP. A pixel is a confirmed error when its three-channel mean absolute display-sRGB difference exceeds 0.05. Low risk requires a finite score strictly below the frozen threshold and an evaluable RGB comparison.",
        "",
        "| Reconstruction | Scorer inputs | Method | Common AP | Families | Common positive | Common / E | Low / V | Error / low | Missed / Y | Unknown / E |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in CONDITIONS:
        for scope in SCOPES:
            if (c, scope) == ("M2", "two"):
                continue
            for m in METHODS:
                g = addr[c, scope, m]
                ap = g["common_ap"]
                scores = [
                    v(g[k]["mean"], True)
                    for k in (
                        "common_prevalence",
                        "common_coverage_E",
                        "low_coverage",
                        "low_error",
                        "missed_error_fraction",
                        "unknown_E",
                    )
                ]
                s.append(
                    "| "
                    + " | ".join(
                        (
                            c,
                            ("two = all" if (c, scope) == ("M2", "all") else scope),
                            m,
                            v(ap["mean"]),
                            f"{ap['defined_families']}/{ap['planned_families']}",
                            *scores,
                        )
                    )
                    + " |"
                )
    s.extend(
        [
            "",
            "### Operating points by held-out family",
            "",
            f"The following rows show the frozen default {method} with all available sources. Errors falling in low or unknown are included in missed / Y. Families with no two-class view retain their original counts.",
            "",
            "| Family | Reconstruction | V | E | Four-method S | Confirmed Y | AP | Defined views | Low / V | Error / low |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    def csvfloat(r, k):
        return float(r[k]) if r[k] else None

    for row in per_family:
        if row["method"] == method and row["scope"] == "all":
            s.append(
                "| "
                + " | ".join(
                    [
                        row["family"],
                        row["condition"],
                        *[f"{int(row[k]):,}" for k in ("V", "E", "S", "Y")],
                        v(csvfloat(row, "common_ap")),
                        row["common_defined_views"] + "/" + row["common_scheduled_views"],
                        v(csvfloat(row, "low_coverage"), True),
                        v(csvfloat(row, "low_error"), True),
                    ]
                )
                + " |"
            )
    s.extend(
        [
            "",
            "### Visibility and support in the default model",
            "",
            "Error incidence is measured conditional on evaluable pixels in the named region. Edge, support and conflict rows can overlap the geometric visibility classes. The final row contains pixels where all four methods returned unknown.",
            "",
            "| Reconstruction | Region | Confirmed errors / E | Four-method S / E | Common AP | Families with AP |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for r in strata:
        if r["method"] != method or r["scope"] != "all":
            continue
        for n in (
            "observed",
            "disoccluded",
            "outside",
            "edge",
            "interior",
            "support_0",
            "support_1",
            "conflict",
            "all_four_unknown",
        ):
            z = r["regions"][n]
            a = z["common_ap"]
            s.append(
                f"| {r['condition']} | {n.replace('_', ' ')} | {v(z['rgb_error_fraction']['mean'], True)} | {v(z['common_coverage_E']['mean'], True)} | {v(a['mean'])} | {a['defined_families']}/{a['planned_families']} |"
            )
    counts = summary["dispositions"]
    s += [
        "",
        f"Returned {counts['returned_target_views']} of {counts['planned_target_views']} target renders across {counts['planned_cells']} reconstructions. V counts valid reference RGB. E counts pixels with both RGBs valid. Y counts confirmed errors in E. Misses include errors assessed low and errors with an unknown score. Missing RGB and unknown risk have separate denominators. Empty strata and one-class rankings retain their counts and have undefined AP. [Family counts](family-metrics.csv) · [Stratum counts](region-metrics.csv) · [Risk–coverage curves](risk-coverage.csv)",
        "",
        "## Changes on the same target",
        "",
        "Switching scorer inputs preserves the optimized scene and its target RGB. Score-one reads s00, score-two reads s00 and s03, and score-all reads every generator source. The new models trained with the complete source set for each reconstruction. The one-source setting tests how those fitted models extrapolate. M2 score-two and score-all share the same score bytes. The B transfer always reads s00. The matched-scope table compares scores on the intersection of all four methods and all three scorer scopes on the same pixels. [Scorer-source differences](source-ablation.csv) · [Matched-pixel AP and prevalence counts](matched-all-scope-ap.csv)",
        "",
        "The following M4 minus M2 differences use the same reference pixels within each family. RGB error incidence and RGB MAE use the paired RGB domain. Depth MAE uses the separate paired positive optical-axis expected-depth domain with valid opacity of at least 0.5 and valid reference geometry. Target camera distance, opacity, expected-depth and background coverage are available per view. [Per-target reconstruction measures](paired-reconstruction.csv)",
        "",
        "| Family | Paired RGB pixels | Paired depth pixels | Δ RGB MAE | Δ confirmed error fraction | Δ depth MAE (m) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for f, z in summary["paired_reconstruction_by_family"].items():
        d = z["M4_minus_M2"]
        s.append(
            f"| {f} | {z['paired_E']:,} | {z['paired_Z']:,} | {v(d['rgb_mae'])} | {v(d['rgb_error_fraction'], True)} | {v(d['depth_mae_m'])} |"
        )
    decisions = learn["decision"]["comparisons"]
    diffs = {
        k: ", ".join(
            "undefined" if z is None else f"{z:+.4f}" for z in decisions[k]["paired_seed_differences"]
        )
        for k in ("MS-frozen", "MS-L1")
    }
    s.extend(
        [
            "",
            f"Across the two development seeds, fine-tuning changed common-domain AP by {diffs['MS-frozen']} versus the matched frozen encoder and by {diffs['MS-L1']} versus MS-L1. The registered promotion rule required a positive gain for both seeds and a mean gain of at least 0.01 against each comparator, together with the coverage and low-error limits. The development decision selected {method}.",
        ]
    )
    recovered = [r for r in learn["neural_candidates"] if r["optimizer_steps_in_this_record"] == 0]
    if recovered:
        rr = recovered[0]
        s.extend(
            [
                "",
                f"The original Windows {rr['method']} fit with seed {rr['seed']} completed {rr['optimizer_steps']} optimizer steps, then failed its deployment-schema export. The Mac recovery performed zero additional updates and inherited the exact selected weights and all 48 Windows score-all arrays. M2 score-two uses the same two sources and bytes. The remaining smaller-source maps were scored on the Mac CPU. The three other pairs of method and seed produced all development scope scores on Windows. The held-out four-method comparison used original Mac CPU inference across every source subset. The first held-out frozen-encoder scoring process lacked a private recorder dependency and produced zero maps. Two subsequent recorder preflights rejected an unrelated source-tree change and a registry copy in a different parent directory. Both ended before inference. The accepted retry used the entire original scorer snapshot and its exact frozen request. [Export evidence](engineering-and-failures.json)",
            ]
        )
    budget = acquisition["production"]
    eng = acquisition["engineering"]
    s += [
        "",
        "## Training, repeatability and cost",
        "",
        f"The scientific fit used {qa['totals']['train']['families']} training families and {qa['totals']['train']['samples']:,} sampled pixels. The four development families supplied thresholds for the new methods and the paired-seed deployment decision. The B logistic coefficients and threshold were retained. Development used seeds 1729 and 2718 with the registered primary weights from 1729. [Development measurements](development-summary.json) · [Training and selection](training-and-selection.json)",
        "",
        f"The native CPU and producer comparison measured {parity['measured_unique_comparisons']}/{parity['planned_unique_comparisons']} distinct target, method and source-subset combinations on {case['family']}. Support disagreed on {parity['support_disagreement']:,} pixels. {parity['threshold_crossings']:,} finite values crossed the frozen cutoffs. Absolute risk tolerance was {parity['absolute_tolerance']:g}. The bounded numerical check {'passed' if parity['bounded_all'] else 'did not pass'} for all scheduled measurements. Strict state equality {'held' if parity['strict_state_all'] else 'did not hold'} across all scheduled measurements. Bitwise equality of all finite risks and support {'held' if parity['bitwise_all'] else 'did not hold'}. [Original-versus-native comparisons and process timings](mac-native-and-qa.csv) · [Cold and warm CPU runs](neural-cpu-cost.json)",
        "",
        f"The acquisition ledger contains {budget['trials']} scientific reconstructions and {eng['trials']} engineering reconstructions, including the extra engineering repeat. Separate recorder intervals can overlap. The recorded sums represent stage workloads. CUDA allocation and reservation counters describe the framework allocator. Device capacity and any available driver samples have separate fields. [Scientific and engineering resource ledger](acquisition-costs.json) · [Selected recorder stages](stage-receipts.csv) · [Recovery and missing observations](engineering-and-failures.json)",
        "",
        f"Four held-out families support a paired descriptive comparison. Counts and per-family outcomes show the range that the macro averages combine. The accompanying public example uses the preregistered development family {case['family']} and its six target cameras under each reconstruction. [Independent raw-pixel and scikit-learn AP check](heldout-pixel-recheck.json) · [Scientific provenance](provenance.json) · [Held-out table checksums](confirmation-manifest.json)",
    ]
    s.extend(
        [
            "",
            "### Measured reconstruction and native scoring cost",
            "",
            "Per-condition medians below use every scientific reconstruction with the identified measurement. Optimizer time includes the synchronized GPU update loop. Reconstruction and rendering wall times use their separate recorder clocks.",
            "",
            "| Condition | Runs with timing | GPU optimization median (s) | Reconstruction wall median (s) | Render wall median (s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for c in CONDITIONS:
        costs = budget["by_condition"].get(c, {})
        vals = [
            costs.get(k, {})
            for k in ("optimization_seconds", "generation_wall_seconds", "render_wall_seconds")
        ]
        s.append(f"| {c} | {vals[0].get('n', 0)} | " + " | ".join(v(z.get("median")) for z in vals) + " |")
    s.extend(
        [
            "",
            "Each cold native score below launched a fresh Mac process, read the authorized source frames and calibration, performed inference and wrote the completed map. The extra M2 alias incurred no second call.",
            "",
            "| Method | Measured / planned calls | End-to-end median (s) | End-to-end p95 (s) |",
            "|---|---:|---:|---:|",
        ]
    )
    for m in METHODS:
        q = timings[m]
        s.append(
            f"| {m} | {q['measured']}/{q['scheduled']} | {v(q['p50_seconds'])} | {v(q['p95_seconds'])} |"
        )
    s.extend(
        [
            "",
            "The independent warm CPU runs kept the model in memory while reading native source evidence for each target.",
            "",
            "| Method | Condition | Scorer inputs | Warm measured targets | Warm median (s) | Warm p95 (s) | Peak process-tree RSS (MiB) |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for q in neural_cpu:
        for scope, zz in q["warm_views"].items():
            s.append(
                f"| {q['method']} | {q['condition']} | {scope} | {zz['timed_native_views']} | {v(zz['median_seconds'])} | {v(zz['p95_seconds'])} | {q['warm_process']['sampled_peak_process_tree_rss_bytes'] / (1024**2):.1f} |"
            )
    return "\n".join(s) + "\n"


def run_report(roots, request, opening, out, rid):
    mk = require_opening(roots[0], CATALOG, opening_sha256=opening, shelf=roots[1])
    method = get_frozen_method(roots[0], mk, roots[1])
    require(
        request["schema"] == "el-stage3-joint-report-request-1"
        and request["opening_sha256"] == opening
        and request["catalog_sha256"] == CATALOG,
        "joint study request differs from opening",
    )
    sources = {}
    roles = {}
    runs = {}

    def take(key, spec):
        pin = compact_pin(spec)
        r = find_run(roots, pin)
        require(r.run["execution_status"] == "succeeded", "unsuccessful scientific source: " + key)
        data, _ = json_member(r, spec["member"], spec["sha256"])
        sources[key] = proof(r, spec["member"], spec["sha256"])
        roles.setdefault(pin["run_id"], []).append(key)
        runs[pin["run_id"]] = r
        return r, data

    def secondary(key):
        return take(key, request["sources"][key])

    for k in ("tensor_comparison", "cpu_validation", "production_budget"):
        take(k, method["components"][k])
    held, he = secondary("confirmation")
    dv, de = secondary("development")
    _, sel = secondary("selection")
    _, parity = secondary("parity")
    _, pixel = secondary("pixel-replay")
    require(
        pixel["schema"] == "el-stage3-raw-confirmation-replay-1"
        and pixel["passed"]
        and pixel["method_sha256"] == mk["method_sha256"]
        and pixel["opening_sha256"] == opening
        and compact_pin(pixel["original_evaluation"]) == compact_pin(held.pin)
        and pixel["original_evaluation"]["sha256"] == sources["confirmation"]["sha256"],
        "independent heldout validation uses another original or method",
    )
    _, ex = secondary("example")
    _, fit = secondary("fit")
    br, baseline = secondary("baseline_development")
    _, qa = secondary("input_qa")
    _, cost = secondary("acquisition_costs")
    audit_run, quality = secondary("acquisition_quality")
    require(
        quality["schema"] == "el-stage3-acquisition-ledger-1"
        and quality["opening_sha256"] == opening
        and not quality["still_sealed"]
        and quality["catalog_sha256"] == CATALOG
        and quality["source_costs_sha256"] == sources["acquisition_costs"]["sha256"]
        and len(quality["with_actual_records"]) == 32,
        "independent acquisition quality is incomplete",
    )
    raw_pairs, _ = json_member(audit_run, "payload/family-pairs.json", quality["tables"]["family-pairs.json"])
    ds = {s: secondary(s + "_dataset")[1] for s in ("train", "development", "test")}
    _, label_replay = secondary("acquisition_dataset_qa")
    truth_hashes = {}
    for row in ds["test"]["views"]:
        truth_hashes.setdefault((row["family"], row["view"]), {})[row["condition"]] = row["truth"]["sha256"]
    compared_truth = {
        (r["family"], r["view"]): r["truth_sha256"] for r in label_replay["views"] if r["passed"]
    }
    require(
        label_replay["schema"] == "el-stage3-reconstruction-parity-1"
        and label_replay["passed"]
        and label_replay["split"] == "test"
        and label_replay["opening_sha256"] == opening
        and compact_pin(label_replay["source_ledger"]) == compact_pin(sources["acquisition_quality"])
        and compact_pin(label_replay["dataset"]) == compact_pin(sources["test_dataset"])
        and sorted(label_replay["families"]) == sorted(mk["test_families"])
        and compared_truth == truth_hashes
        and len(label_replay["views"]) == len(compared_truth)
        and label_replay["distinct_source_labels"] == 48
        and label_replay["validated_view_pairs"] == 24
        and label_replay["cell_and_common_count_sum_checks"] == 1248,
        "independent held-out original reconstruction and label comparison is incomplete",
    )
    require(
        method["study_sha256"] == STUDY
        and he["schema"] == de["schema"] == "el-stage3-risk-evaluation-1"
        and he["split"] == "test"
        and de["split"] == "development"
        and he["families"] == mk["test_families"]
        and he["method_sha256"] == mk["method_sha256"]
        and he["opening_sha256"] == opening
        and he["dataset_sha256"] == sources["test_dataset"]["sha256"]
        and de["dataset_sha256"]
        == sources["development_dataset"]["sha256"]
        == sel["development_dataset_sha256"]
        and sel["decision"]["default"] == method["default_method"]
        and qa["passed"]
        and qa["catalog_sha256"] == CATALOG
        and all(
            qa["datasets"][s]["sha256"] == sources[s + "_dataset"]["sha256"] for s in ("train", "development")
        )
        and all(
            all(
                he["models"][m][k] == method["models"][m][k] == de["models"][m][k] == sel["models"][m][k]
                for k in FROZEN
            )
            for m in METHODS
        ),
        "study method, data, development decision or heldout evidence differs",
    )
    audit_cells(ds, cost, opening)
    original = {m: secondary("native_" + m)[1] for m in METHODS}
    nr = native_rows(parity, original)
    pred = method["components"]
    cpu_run = find_run(roots, pred["cpu_validation"])
    cpudoc = load(cpu_run.open(pred["cpu_validation"]["member"]))
    cpu_cost = []
    for item in cpudoc["cases"]:
        orig = item["original_cost"]
        receipt, _ = json_member(cpu_run, "payload/" + orig["path"], orig)
        cpu_cost.append(
            {
                "method": item["method"],
                "condition": item["condition"],
                "case_sha256": item["case_sha256"],
                "model_index_sha256": item["model_sha256"],
                "original_cost_sha256": orig["sha256"],
                **{
                    k: receipt[k]
                    for k in (
                        "grid",
                        "cold_warm_v01_equal",
                        "cold_v01_by_scope",
                        "warm_process",
                        "warm_views",
                    )
                },
            }
        )
    from evidence_lens.stage3_decision import validate_fit_origin

    candidates = []
    neural_recoveries = []
    actual_optimizer_steps = {}
    for k, spec in sorted(request["sources"].items()):
        if not k.startswith("candidate_"):
            continue
        r, t = take(k, spec)
        validate_fit_origin(r, roots[0], roots[1])
        if r.run["run_kind"] == "derivation":
            opt_pin = compact_pin(t["optimization_parent"])
            failed = find_run(roots, opt_pin)
            require(
                t["optimizer_steps_in_this_record"] == 0
                and t["optimizer_steps"] > 0
                and failed.run["execution_status"] == "failed",
                "incorrect recovery work accounting",
            )
            roles.setdefault(opt_pin["run_id"], []).append("original_neural_optimization_and_export_failure")
            runs[opt_pin["run_id"]] = failed
            actual_optimizer_steps[opt_pin["run_id"]] = t["optimizer_steps"]
            actual_optimizer_steps[r.pin["run_id"]] = 0
            neural_recoveries.append(
                {
                    "scope": "neural export",
                    "status": "failed",
                    "source": opt_pin,
                    "recovered_by": compact_pin(r.pin),
                    "original_optimizer_steps": t["optimizer_steps"],
                    "additional_optimizer_steps": 0,
                    "original_exit_code": failed.run["exit_code"],
                    "reason": "The Windows optimization completed and its deployment export failed the schema-order check. The Mac derivation reused the exact selected weights and Windows score-all maps.",
                    "development_scoring": {
                        "windows": "M2/all (byte-identical to M2/two), M4/all",
                        "mac_cpu": "M2/one; M4/one and M4/two",
                    },
                }
            )
        else:
            opt_pin = compact_pin(r.pin)
            actual_optimizer_steps[r.pin["run_id"]] = t["optimizer_steps"]
        pr, pps = json_member(r, "payload/independent-proof/verified-updates.json")
        require(
            t["phase"] == "formal"
            and pr["run_id"] == r.pin["run_id"]
            and pr["weights_sha256"] == t["weights_sha256"],
            "training or tensor projection differs",
        )
        row = {
            k: t[k]
            for k in (
                "method",
                "seed",
                "optimizer_steps",
                "epochs",
                "best_epoch",
                "stop_reason",
                "threshold",
                "best_development_family_equal_ap",
                "model_index_sha256",
                "weights_sha256",
                "wall_seconds",
                "peak_allocated_cuda",
                "peak_reserved_cuda",
            )
        }
        row.update(
            {
                k: pr[k]
                for k in (
                    "verified_tensors",
                    "frozen_tensors",
                    "backbone_changed_scalars",
                    "changed_scalars_by_group",
                    "nonzero_task_gradient_minibatches",
                )
            }
        )
        row.update(
            **compact_pin(r.pin), scientific_member_sha256=spec["sha256"], tensor_proof_sha256=pps["sha256"]
        )
        row.update(
            optimization_parent=opt_pin,
            optimizer_steps_in_this_record=actual_optimizer_steps[r.pin["run_id"]],
            development_score_producer=(
                "Windows score-all reused; smaller new scopes from a Mac CPU derivation"
                if r.run["run_kind"] == "derivation"
                else "Windows"
            ),
        )
        candidates.append(row)
    require(
        {(r["method"], r["seed"]) for r in candidates} == {(m, s) for m in METHODS[2:] for s in (1729, 2718)}
        and len(candidates) == 4,
        "missing paired neural fit",
    )
    for k, sp in request["sources"].items():
        if k not in sources:
            take(k, sp)
    freeze = find_run(roots, mk["method"])
    sources["method_freeze"] = proof(freeze, "payload/method-freeze.json", mk["method_sha256"])
    roles.setdefault(freeze.pin["run_id"], []).append("method_freeze")
    runs[freeze.pin["run_id"]] = freeze
    from stage3_risk_engineering import observations

    engineering = observations(roots, request["engineering"])
    failures = [*neural_recoveries, *engineering]
    correction = load(br.open("config.resolved.yaml")).get("engineering_correction")
    if correction:
        fr = find_run(roots, correction["previous_failed"])
        require(
            fr.run["execution_status"] == "failed" and not any(n.startswith("payload/") for n in fr.files),
            "packaging correction source differs",
        )
        failures.append(
            {
                "scope": "threshold packaging",
                "status": "failed",
                "source": compact_pin(fr.pin),
                "recovered_by": compact_pin(br.pin),
                "exit_code": fr.run["exit_code"],
                "previous_worker_sha256": correction["previous_worker_sha256"],
                "corrected_worker_sha256": correction["corrected_worker_sha256"],
                "reason": "Output index inherited read-only permissions from its fitted input; fresh output index created from the original coefficient and preprocessing bytes.",
            }
        )
        roles.setdefault(fr.pin["run_id"], []).append("failed_threshold_packaging")
        runs[fr.pin["run_id"]] = fr
    acquisition_attempts = [
        {
            k: x[k]
            for k in (
                "trial_id",
                "phase",
                "stage",
                "run_id",
                "manifest_sha256",
                "status",
                "retry_of",
                "transfer_zip_bytes",
                "wall_seconds",
            )
        }
        for x in cost["attempts"]
        if x["status"] != "succeeded"
    ]
    failures.extend({"scope": "acquisition attempt", **a} for a in acquisition_attempts)
    stages = []
    for key, r in runs.items():
        rr = r.run
        stages.append(
            {
                "roles": ",".join(sorted(roles[key])),
                **compact_pin(r.pin),
                **{
                    k: rr[k]
                    for k in (
                        "run_kind",
                        "experiment_id",
                        "execution_status",
                        "elapsed_seconds_monotonic",
                        "sampled_peak_tree_rss_bytes",
                        "code_commit",
                        "code_snapshot_sha256",
                    )
                },
                "actual_optimizer_steps_in_this_record": actual_optimizer_steps.get(key),
                "logical_payload_bytes": sum(
                    v["bytes"] for n, v in r.files.items() if n.startswith("payload/")
                ),
            }
        )
    cands = sorted(candidates, key=lambda x: (x["seed"], x["method"]))
    recovered_primary = {
        q["method"]
        for q in cands
        if q["seed"] == method["primary_seed"] and q["optimizer_steps_in_this_record"] == 0
    }
    for q in nr:
        m = q["method"]
        c = q["condition"]
        scope = q["scope"]
        if m in METHODS[:2]:
            producer = "Mac CPU baseline"
        elif m in recovered_primary and scope != "all" and (c, scope) != ("M2", "two"):
            producer = "Mac CPU terminal-export recovery"
        else:
            producer = "Windows CUDA"
        q["original_score_platform"] = producer
        q["repeat_score_platform"] = "Mac CPU native source-to-score"
    lp = (
        "status",
        "reason",
        "C",
        "solver",
        "candidate_C",
        "candidates",
        "fold_attempts",
        "training_dataset_sha256",
        "max_abs_numpy_sklearn",
        "parity_rows",
        "preprocessing_sha256",
    )
    learn = {
        "schema": "el-stage3-public-training-and-selection-1",
        "lightweight": {k: fit[k] for k in lp},
        "thresholds": method["thresholds"],
        "models": {m: {k: v[k] for k in FROZEN} for m, v in method["models"].items()},
        "decision": sel["decision"],
        "paired_development": sel["summaries"],
        "neural_candidates": cands,
        "baseline_development": {
            "threshold": baseline["threshold"],
            "development_common_ap": baseline["development_common_ap"],
        },
    }
    with new_output(out) as dest:
        pub = dest / "public-numerical"
        pub.mkdir()
        summary = copy_original_tables(held, he, pub)
        devspec = de["public_numerical"]
        di, _ = json_member(dv, "payload/" + devspec["path"], devspec)
        spec = next(x for x in di["files"] if x["path"] == "study-summary.json")
        dpath, _ = member(dv, "payload/public-numerical/study-summary.json", spec)
        shutil.copyfile(dpath, pub / "development-summary.json")
        cross = {
            "heldout": paired_quality(summary, raw_pairs),
            "development": paired_quality(load(dpath), raw_pairs),
        }
        write_json(
            pub / "paired-original-quality.json",
            {
                "schema": "el-stage3-public-original-reconstruction-readback-1",
                "compared_metrics": cross,
                "source": sources["acquisition_quality"],
                "source_member_sha256": quality["tables"]["family-pairs.json"],
                "family_pairs": raw_pairs,
                "independent_heldout_dataset_comparison": {
                    "source": sources["acquisition_dataset_qa"],
                    **{
                        k: label_replay[k]
                        for k in (
                            "passed",
                            "validated_view_pairs",
                            "distinct_source_labels",
                            "cell_and_common_count_sum_checks",
                            "quantities",
                        )
                    },
                },
            },
        )
        write_json(pub / "training-and-selection.json", learn)
        public_grade = {k: v for k, v in pixel.items() if k not in {"per_view_records"}}
        public_grade["original_per_view_row_count"] = len(pixel["per_view_records"])
        write_json(pub / "heldout-pixel-recheck.json", public_grade)
        table(pub / "mac-native-and-qa.csv", nr)
        write_json(
            pub / "neural-cpu-cost.json", {"schema": "el-stage3-public-neural-cpu-costs-1", "cases": cpu_cost}
        )
        table(pub / "stage-receipts.csv", stages)
        # Public projections omit device-local paths; the separate stage audit supplies full numerical observations.
        public_cost = {k: cost[k] for k in ("schema", "scope", "opening_sha256", "source_notes", "summary")}
        public_cost["trials"] = [
            {k: v for k, v in r.items() if k not in {"recorded_failure"}} for r in cost["trials"]
        ]
        public_cost["attempts"] = [
            {k: v for k, v in r.items() if k not in {"failure_reason"}} for r in cost["attempts"]
        ]
        public_cost["failure_details"] = (
            "See the reported stage, attempt and immutable source manifest for each failure."
        )
        write_json(pub / "acquisition-costs.json", public_cost)
        write_json(
            pub / "engineering-and-failures.json",
            {
                "schema": "el-stage3-engineering-and-failures-1",
                "observed": failures,
                "acquisition_failed_attempts": len(acquisition_attempts),
                "lightweight_threshold_packaging_failures": int(correction is not None),
                "neural_export_failures": len(neural_recoveries),
                "engineering_readback_and_public_export_failures": sum(
                    r["status"] == "failed" and not r["scope"].startswith("heldout_cpu_") for r in engineering
                ),
                "heldout_scoring_environment_failures": sum(
                    r["scope"] == "heldout_cpu_environment" for r in engineering
                ),
                "heldout_scoring_retry_preflight_rejections": sum(
                    r["scope"] in {"heldout_cpu_retry_preflight", "heldout_cpu_catalog_preflight"}
                    for r in engineering
                ),
                "local_engineering_incidents_without_a_recorder_run": sum(
                    r["status"] == "local_engineering_incident" for r in engineering
                ),
                "scientific_dispositions": he["dispositions"],
                "coverage": "Acquisition attempt failures, recorded learning packaging correction and formal scientific target return counts are separate.",
            },
        )
        provenance = {
            "schema": "el-stage3-public-joint-provenance-1",
            "joint_scientific_run_id": rid,
            "catalog_sha256": CATALOG,
            "study_sha256": STUDY,
            "method_sha256": mk["method_sha256"],
            "opening_sha256": opening,
            "sources": sources,
            "input_qa": {"totals": qa["totals"], "ancestry": qa["ancestry"]},
            "public_example": {
                "family": ex["family"],
                "targets": ex["targets"],
                "conditions": ex["conditions"],
            },
        }
        write_json(pub / "provenance.json", provenance)
        with (pub / "family-metrics.csv").open(encoding="utf-8", newline="") as handle:
            per_family = list(csv.DictReader(handle))
        macros_spec = he["files"]["macros"]
        from stage3_risk_replay import array_member

        strata = array_member(held, macros_spec)
        (pub / "study.md").write_text(
            readable(
                summary,
                learn,
                parity,
                cost["summary"],
                qa,
                ex,
                per_family,
                strata,
                {m: q["timing"] for m, q in original.items()},
                cpu_cost,
            ),
            encoding="utf-8",
        )
        for public_file in pub.iterdir():
            require(
                not re.search(
                    r"(?:/Users/|/Volumes/|/private/var/|/var/folders/|/mnt/|[A-Za-z]:[\\/]|file://|/home/|https?://|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})",
                    public_file.read_text(encoding="utf-8"),
                ),
                "private machine location or external address in " + public_file.name,
            )
        manifest = {
            "schema": "el-stage3-public-joint-manifest-1",
            "source_run_id": rid,
            "heldout": sources["confirmation"],
            "catalog_sha256": CATALOG,
            "study_sha256": STUDY,
            "method_sha256": mk["method_sha256"],
            "opening_sha256": opening,
            "files": [asset(pub, p) for p in sorted(pub.iterdir())],
        }
        write_json(pub / "manifest.json", manifest)
        doc = {
            "schema": "el-stage3-joint-study-report-1",
            "run_id": rid,
            "catalog_sha256": CATALOG,
            "study_sha256": STUDY,
            "method_sha256": mk["method_sha256"],
            "opening_sha256": opening,
            "heldout": sources["confirmation"],
            "models": learn["models"],
            "default_method": method["default_method"],
            "dispositions": he["dispositions"],
            "sources": sources,
            "public_numerical": asset(dest, pub / "manifest.json"),
        }
        write_json(dest / "report.json", doc)
    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "shelf", "request", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--opening-sha256", required=True)
    a = parser.parse_args()
    res = run_report(
        (a.store, a.shelf),
        signed(a.request, a.request_sha256),
        a.opening_sha256,
        a.output,
        os.environ["EL_RUN_ID"],
    )
    print(json.dumps({"heldout": res["heldout"], "public_numerical": res["public_numerical"]}))
