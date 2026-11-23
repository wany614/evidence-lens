"""Freeze B0 development operating points and compare native risk maps on five- and seven-method domains."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from evidence_lens._bundle import HASH, load
from evidence_lens.formal_inputs import require, signature
from evidence_lens.l1 import KEYS
from evidence_lens.l1_freeze import threshold_for
from evidence_lens.scoring_io import new_output, rgb_valid, write_json
from vision_data import TRUTH, file_for, open_dataset, packed
from vision_sources import METHODS, find_run, signed, test_authority


FIVE = METHODS[:5]
TWO = METHODS[5:]
SCORE_SETS = ("original-five", *TWO)
RATES = {
    "low_coverage": ("L", "V"),
    "low_risk_error": ("LY", "L"),
    "detection": ("HY", "Y"),
    "warning_precision": ("HY", "H"),
    "warning_coverage": ("H", "V"),
    "unknown_fraction": ("U", "E"),
    "unknown_error": ("UY", "U"),
    "scoreable_coverage": ("A", "V"),
    "rgb_incidence": ("Y", "E"),
    "confirmed_error_of_V": ("Y", "V"),
    "invalid_fraction": ("invalid", "V"),
    "rgb_mae": ("rgb_error_sum", "E"),
    "sensitivity_025": ("Y025", "E"),
    "sensitivity_100": ("Y100", "E"),
    "depth_eligible_fraction": ("Z", "surface"),
    "depth_mae_m": ("z_error_sum", "Z"),
    "depth_high_fraction": ("ZY", "Z"),
    "five_common_coverage": ("S5", "V"),
    "seven_common_coverage": ("S7", "V"),
    "five_common_prevalence": ("S5Y", "S5"),
    "seven_common_prevalence": ("S7Y", "S7"),
}


def test_access(opening, catalog_pin):
    require(opening is not None and len(opening) == 2, "canonical test-opening store and cold shelf required")
    roots = [Path(r).resolve() for r in opening]
    digest = test_authority(roots, catalog_pin)
    marker = load(roots[0] / "protocol/stage2" / catalog_pin[1] / "test-opening.json")
    run = find_run(
        roots, {"run_id": marker["freeze_run_id"], "manifest_sha256": marker["freeze_manifest_sha256"]}
    )
    return digest, signed(run.open(marker["method_member"]), marker["method_sha256"])


def guard_test(ds, opening):
    """Recheck dataset linkage after test_access was called on the pinned catalog, before transport was opened."""
    require(opening and len(opening) == 2, "a verified opening is required")
    roots = [Path(x).resolve() for x in opening]
    require(
        len(ds["material_indices"]) == 1 and ds["material_indices"][0]["cohort"] == "v11_new",
        "wrong test cohort",
    )
    sha = ds["material_indices"][0]["catalog_sha256"]
    from vision_sources import check_gate

    digest = check_gate(roots, sha, {r["family"] for r in ds["views"]})
    require(digest == ds["test_unlock_sha256"], "dataset belongs to a different opening")
    marker = load(roots[0] / "protocol/stage2" / sha / "test-opening.json")
    run = find_run(
        roots, {"run_id": marker["freeze_run_id"], "manifest_sha256": marker["freeze_manifest_sha256"]}
    )
    return signed(run.open(marker["method_member"]), marker["method_sha256"])


def model_matches(method, name, sha):
    return any(
        Path(f["path"]).name == "model.json" and f["sha256"] == sha for f in method["models"][name]["files"]
    )


def identity(row):
    return row["family"], row["condition"], row["view"]


def score_sources(ds, ds_sha, pins, required=SCORE_SETS):
    results = {}
    keys = {identity(row) for row in ds["views"]}
    for path, sha in pins:
        path = Path(path)
        doc = signed(path, sha)
        method = doc["method"]
        require(
            doc["schema"] == "el-vision-scores-1"
            and method in required
            and method not in results
            and doc["dataset_sha256"] == ds_sha
            and HASH.fullmatch(doc["model_sha256"]),
            "wrong score identity",
        )
        views = {identity(row): row for row in doc["views"]}
        require(
            len(views) == len(doc["views"]) and set(views) == keys,
            "all planned native views required in score index",
        )
        results[method] = (path.parent, views, {"sha256": sha, "model_sha256": doc["model_sha256"]})
    require(set(results) == set(required), "original five and both preselected neural methods required")
    return results


def deploy_contract(root, row):
    if row["status"] == "render_failed":
        return None
    rgb = np.load(file_for(root, row["rgb"]), mmap_mode="r", allow_pickle=False)
    dep = packed(file_for(root, row["full"]["deploy"]), ("scoreable", "l0", "alpha_score"))
    valid_rgb = rgb_valid(rgb)
    require(
        dep["scoreable"].dtype == bool
        and dep["scoreable"].shape == valid_rgb.shape
        and not (dep["scoreable"] & ~valid_rgb).any(),
        "illegal deployment mask",
    )
    return {**dep, "rgb_valid": valid_rgb}


def read_scores(row, sources, thresholds=None, deploy=None):
    result, statuses = {}, {}
    shape = row["height"], row["width"]
    for k, (root, entries, _) in sources.items():
        rec = entries[identity(row)]
        rgb = row["rgb"]
        dep = row["full"]["deploy"]
        require(
            rec["rgb_sha256"] == (rgb["sha256"] if rgb else None)
            and rec["deploy_sha256"] == (dep["sha256"] if dep else None),
            "predictions come from another source grid",
        )
        methods = FIVE if k == "original-five" else (k,)
        if rec["status"] != "scored":
            require(
                rec["status"] in ("render_failed", "unsupported")
                and rec["score"] is None
                and rec["reason"]
                and (rec["status"] == "render_failed") == (row["status"] == "render_failed"),
                "missing predictions require an explicitly declared reason",
            )
            for m in methods:
                result[m] = np.full(shape, np.nan, np.float32)
        else:
            require(row["status"] == "completed" and rec["score"], "available scores on a failed target")
            path = file_for(root, rec["score"])
            with np.load(path, allow_pickle=False) as a:
                for m in methods:
                    risk = a[KEYS[m] if k == "original-five" else "risk"]
                    require(
                        risk.shape == shape
                        and risk.dtype == np.float32
                        and not np.isinf(risk).any()
                        and ((risk[np.isfinite(risk)] >= 0) & (risk[np.isfinite(risk)] <= 1)).all(),
                        "invalid native risk",
                    )
                    if m == "L0":
                        require(
                            np.isin(risk[np.isfinite(risk)], [0, 1]).all(),
                            "L0 must remain the fixed binary rule",
                        )
                    if k != "original-five":
                        valid = a["valid"]
                        require(
                            valid.dtype == bool and np.array_equal(valid, np.isfinite(risk)),
                            "neural validity disagrees with scores",
                        )
                        if thresholds is not None:
                            state = a["state"]
                            compare = np.where(
                                valid, (risk.astype(np.float64) >= thresholds[m]).astype(np.uint8), 255
                            )
                            require(
                                state.dtype == np.uint8 and np.array_equal(compare, state),
                                "neural states differ from frozen threshold",
                            )
                    result[m] = risk
        for m in methods:
            statuses[m] = {"status": rec["status"], "reason": rec["reason"]}
    if deploy is not None:
        for method, key in (("L0", "l0"), ("1-alpha", "alpha_score")):
            if statuses[method]["status"] == "scored":
                require(
                    np.array_equal(result[method], deploy[key], equal_nan=True),
                    "raw scored baseline differs from recorded native features",
                )
        if statuses["constant"]["status"] == "scored":
            require(
                np.array_equal(np.isfinite(result["constant"]), deploy["rgb_valid"]),
                "constant availability differs from predicted RGB",
            )
        for m in (k for k in (*FIVE[-2:], *TWO) if k in result):
            require(
                not (np.isfinite(result[m]) & ~deploy["scoreable"]).any(),
                "learned score on unsupported raw input",
            )
    return result, statuses


def ratio(a, b, label):
    return {
        "value": a / b if b else None,
        "numerator": a,
        "denominator": b,
        "missing_reason": None if b else "empty_" + label,
    }


def groups(values, y):
    """Whole equal binary32 score groups, ascending; weights are native pixel counts."""
    if not len(values):
        return np.empty(0, np.float32), np.empty(0, np.int64), np.empty(0, np.int64)
    i = np.argsort(values, kind="stable")
    a = values[i]
    start = np.r_[0, np.flatnonzero(a[1:] != a[:-1]) + 1]
    return a[start], np.diff(np.r_[start, len(a)]), np.add.reduceat(y[i].astype(np.int64), start)


def ranking(values, y):
    _, sizes, errors = groups(values, y)
    n, p = len(y), int(y.sum())
    if p == 0 or p == n:
        return {
            "ap": None,
            "positive": p,
            "n": n,
            "missing_reason": "empty_mask" if n == 0 else "all_negative" if p == 0 else "all_positive",
        }
    pos = np.cumsum(errors[::-1])
    pop = np.cumsum(sizes[::-1])
    ap = float((errors[::-1] * (pos / pop)).sum(dtype=np.float64) / p)
    return {"ap": ap, "positive": p, "n": n, "missing_reason": None}


def system_counts(t, scores, tau, mask):
    v = t["reference_valid"] & mask
    e = t["rgb_evaluable"] & mask
    y = (t["rgb_high_error"] == 1) & e
    c5 = e.copy()
    c7 = e.copy()
    for m in METHODS:
        finite = np.isfinite(scores[m])
        c7 &= finite
        if m in FIVE:
            c5 &= finite
    z = t["depth_evaluable"] & mask
    err = t["rgb_error"]
    n = lambda x: int(np.count_nonzero(x))
    base = {
        "V": n(v),
        "E": n(e),
        "invalid": n(v) - n(e),
        "Y": n(y),
        "S5": n(c5),
        "S5Y": n(y & c5),
        "S7": n(c7),
        "S7Y": n(y & c7),
        "Y025": n(e & (err > np.float32(0.025))),
        "Y100": n(e & (err > np.float32(0.1))),
        "rgb_error_sum": float(err[e].sum(dtype=np.float64)),
        "surface": n(t["reference_surface"] & mask),
        "Z": n(z),
        "ZY": n(z & (t["depth_high_error"] == 1)),
        "z_error_sum": float(t["depth_error"][z].sum(dtype=np.float64)),
    }
    result = {}
    for m in METHODS:
        score = scores[m]
        ok = e & np.isfinite(score)
        lo = ok & (score.astype(np.float64) < tau[m])
        hi = ok & ~lo
        un = e & ~ok
        counts = {
            **base,
            "A": n(ok),
            "H": n(hi),
            "L": n(lo),
            "U": n(un),
            "HY": n(hi & y),
            "LY": n(lo & y),
            "UY": n(un & y),
        }
        counts["missed_errors"] = counts["LY"] + counts["UY"]
        require(
            counts["L"] + counts["H"] + counts["U"] == base["E"]
            and counts["missed_errors"] + counts["HY"] == base["Y"],
            "incorrect risk-state partition",
        )
        common7 = ranking(score[c7], y[c7])
        own = common7 if np.array_equal(ok, c7) else ranking(score[ok], y[ok])
        common5 = (common7 if np.array_equal(c5, c7) else ranking(score[c5], y[c5])) if m in FIVE else None
        result[m] = {
            "counts": counts,
            "rates": {k: ratio(counts[a], counts[b], b) for k, (a, b) in RATES.items()},
            "ranking": {"own": own, "five_common": common5, "seven_common": common7},
        }
    return result


def freeze_dev(root, ds_sha, source_pins, catalogs, output, run_id, baseline_only=False, baseline_pin=None):
    from vision_baselines import full_cohort

    ds = open_dataset(root, ds_sha, "development")
    full_cohort(ds, catalogs, "development")
    methods = FIVE if baseline_only else METHODS
    sources = score_sources(ds, ds_sha, source_pins, ("original-five",) if baseline_only else SCORE_SETS)
    origin = {m: r[2] for m, r in sources.items()}
    collected = {m: defaultdict(list) for m in methods}
    denom = defaultdict(int)
    for row in ds["views"]:
        truth = packed(file_for(root, row["full"]["truth"]), ("reference_valid", "rgb_evaluable"))
        v, e = truth["reference_valid"], truth["rgb_evaluable"]
        f = row["family"]
        require(
            int(v.sum()) == row["V"] and int(e.sum()) == row["E"] and not (e & ~v).any(),
            "bad E or V denominator",
        )
        denom[f] += int(v.sum())
        scores, _ = read_scores(row, sources, deploy=deploy_contract(root, row))
        for m in methods:
            collected[m][f].append(scores[m][e & np.isfinite(scores[m])])
    require(any(denom.values()), "no development V available for an operating-point freeze")
    choices = {}
    for m in methods:
        vals = {f: np.concatenate(s) for f, s in collected.pop(m).items()}
        if m == "L0":
            cov = {f: int((a < 0.5).sum()) / denom[f] for f, a in vals.items() if denom[f]}
            choices[m] = {"tau": 0.5, "coverage": float(np.mean(list(cov.values()))), "selection": "fixed rule"}
        else:
            choices[m] = threshold_for(vals, denom)
    doc = {
        "schema": "el-vision-baseline-thresholds-1" if baseline_only else "el-vision-thresholds-1",
        "run_id": run_id,
        "development_dataset_sha256": ds_sha,
        "method_sources": origin,
        "thresholds": {m: d["tau"] for m, d in choices.items()},
        "choices": choices,
        "family_V": dict(denom),
        "primary_seed": 1729,
        "coverage_rule": "own finite scores in E, low < tau, over V; whole ties",
    }
    if baseline_pin:
        original = signed(*baseline_pin)
        require(
            original["schema"] == "el-vision-baseline-thresholds-1"
            and original["development_dataset_sha256"] == ds_sha
            and original["method_sources"]["original-five"] == origin["original-five"]
            and all(original["thresholds"][m] == doc["thresholds"][m] for m in FIVE),
            "adding the neural methods changed an original operating point",
        )
        doc["independent_original_thresholds_sha256"] = baseline_pin[1]
    with new_output(output) as out:
        write_json(out / "thresholds.json", doc)
    return doc


def merge_groups(view_groups):
    vals, sizes, errors = (np.concatenate([g[i] for g in view_groups]) for i in range(3))
    if not len(vals):
        return vals, sizes, errors
    order = np.argsort(vals, kind="stable")
    vals = vals[order]
    starts = np.r_[0, np.flatnonzero(vals[1:] != vals[:-1]) + 1]
    return vals[starts], np.add.reduceat(sizes[order], starts), np.add.reduceat(errors[order], starts)


def compact_curve(view_groups, denominator, tau):
    vals, sizes, errors = merge_groups(view_groups)
    n = np.r_[0, np.cumsum(sizes)]
    pos = np.r_[0, np.cumsum(errors)]
    points = []
    for percentile in range(0, 101, 5):
        requested = denominator * percentile / 100
        hi = min(int(np.searchsorted(n, requested)), len(n) - 1)
        j = min((hi, max(0, hi - 1)), key=lambda i: (abs(n[i] - requested), n[i]))
        points.append(
            {
                "requested_coverage": percentile / 100,
                "accepted": int(n[j]),
                "positive": int(pos[j]),
                "V": denominator,
                "coverage": int(n[j]) / denominator if denominator else None,
                "risk": int(pos[j]) / int(n[j]) if n[j] else None,
            }
        )
    j = int(np.searchsorted(vals.astype(np.float64), tau, side="left"))
    return {
        "score_group_count": len(vals),
        "frozen_tau": tau,
        "frozen_L": int(n[j]),
        "frozen_LY": int(pos[j]),
        "points": points,
    }


def aggregate_method(rows):
    counts = {k: sum(r["counts"][k] for r in rows) for k in rows[0]["counts"]}
    ranks = {}
    for domain in ("own", "five_common", "seven_common"):
        entries = [r["ranking"][domain] for r in rows if r["ranking"][domain] is not None]
        if not entries:
            ranks[domain] = None
            continue
        ap = [e["ap"] for e in entries if e["ap"] is not None]
        ranks[domain] = {
            "ap": float(np.mean(ap)) if ap else None,
            "defined_views": len(ap),
            "planned_views": len(rows),
            "missing_reason": None if ap else "no_two_class_view",
        }
    return {
        "counts": counts,
        "rates": {k: ratio(counts[a], counts[b], b) for k, (a, b) in RATES.items()},
        "ranking": ranks,
    }


def macro_families(rows):
    out = {}
    for key in RATES:
        vals = [r["rates"][key]["value"] for r in rows if r["rates"][key]["value"] is not None]
        out[key] = {
            "value": float(np.mean(vals)) if vals else None,
            "defined_families": len(vals),
            "total_families": len(rows),
        }
    for key in ("own", "five_common", "seven_common"):
        vals = [
            r["ranking"][key]["ap"] for r in rows if r["ranking"][key] and r["ranking"][key]["ap"] is not None
        ]
        out[key + "_ap"] = {
            "value": float(np.mean(vals)) if vals else None,
            "defined_families": len(vals),
            "total_families": len(rows),
        }
    return out


def paired_report(families):
    rows = []

    def metric(method, key):
        return method["ranking"]["seven_common"]["ap"] if key == "ap" else method["rates"][key]["value"]

    def compare(fam, cond, left, right, comparison, a, b):
        for name in (
            "ap",
            "low_coverage",
            "low_risk_error",
            "detection",
            "unknown_fraction",
            "confirmed_error_of_V",
            "seven_common_prevalence",
        ):
            x, y = metric(a, name), metric(b, name)
            rows.append(
                {
                    "family": fam,
                    "condition": cond,
                    "left": left,
                    "right": right,
                    "comparison": comparison,
                    "metric": name,
                    "left_value": x,
                    "right_value": y,
                    "difference": x - y if x is not None and y is not None else None,
                }
            )

    by = {(r["family"], r["condition"]): r["methods"] for r in families}
    for (fam, cond), methods in by.items():
        for left, right in (
            ("L2-vision-ft", "L2-frozen"),
            ("L2-vision-ft", "L1-full"),
            ("L2-frozen", "L1-full"),
        ):
            compare(fam, cond, left, right, "same_condition", methods[left], methods[right])
        if cond != "B0":
            for m in METHODS:
                compare(fam, cond, cond, "B0", m, methods[m], by[fam, "B0"][m])
    groups = defaultdict(list)
    for r in rows:
        groups[(r["condition"], r["left"], r["right"], r["comparison"], r["metric"])].append(r)
    macro = []
    for (cond, left, right, comparison, metric_name), members in sorted(groups.items()):
        valid = [r for r in members if r["difference"] is not None]
        macro.append(
            {
                "condition": cond,
                "left": left,
                "right": right,
                "comparison": comparison,
                "metric": metric_name,
                "mean_paired_difference": float(np.mean([r["difference"] for r in valid])) if valid else None,
                "defined_families": [r["family"] for r in valid],
                "scheduled_families": len(members),
            }
        )
    return {"family_differences": rows, "family_equal_differences": macro}


def evaluate(root, ds_sha, split, source_pins, threshold_pin, output, opening, catalogs):
    from vision_baselines import full_cohort

    marker = method = None
    if split == "test":
        require(len(catalogs) == 1, "new confirmation catalog only")
        marker, method = test_access(opening, catalogs[0])
    ds = open_dataset(root, ds_sha, split, marker)
    full_cohort(ds, catalogs, split)
    tdoc = signed(*threshold_pin)
    tau = tdoc["thresholds"]
    require(
        tdoc["schema"] == "el-vision-thresholds-1"
        and set(tau) == set(METHODS)
        and tdoc["primary_seed"] == 1729
        and tau["L0"] == 0.5
        and all(np.isfinite(v) and 0 <= v <= 1.000001 for v in tau.values()),
        "invalid frozen thresholds",
    )
    sources = score_sources(ds, ds_sha, source_pins)
    for tag, (_, _, spec) in sources.items():
        require(
            spec["model_sha256"] == tdoc["method_sources"][tag]["model_sha256"],
            "evaluated model differs from development freeze",
        )
    if split == "test":
        official = {m: (v.get("tau") if isinstance(v, dict) else v) for m, v in method["thresholds"].items()}
        require(tau == official, "numerical thresholds differ from the sealed opening")
        for m in METHODS[3:]:
            model_hash = sources["original-five" if m in FIVE else m][2]["model_sha256"]
            require(model_matches(method, m, model_hash), "deployed model identity differs from opening: " + m)
    else:
        require(
            tdoc["development_dataset_sha256"] == ds_sha
            and all(s[2]["sha256"] == tdoc["method_sources"][k]["sha256"] for k, s in sources.items()),
            "development score index differs from selection",
        )
    by_cell = defaultdict(list)
    for row in ds["views"]:
        by_cell[(row["family"], row["condition"])].append(row)
    summaries = []
    status = []
    with new_output(output) as out:
        for (family, condition), rows in sorted(by_cell.items()):
            view_rows = []
            curve_groups = defaultdict(list)
            folder = out / family / condition
            folder.mkdir(parents=True)
            for row in rows:
                t = packed(file_for(root, row["full"]["truth"]), TRUTH)
                v, e, err = t["reference_valid"], t["rgb_evaluable"], t["rgb_error"]
                require(
                    int(v.sum()) == row["V"]
                    and int(e.sum()) == row["E"]
                    and not (e & ~v).any()
                    and np.array_equal(np.isfinite(err), e)
                    and np.array_equal(t["rgb_high_error"][e], err[e] > np.float32(0.05))
                    and np.all(t["rgb_high_error"][~e] == 255),
                    "truth differs from pixel contract",
                )
                scores, disp = read_scores(row, sources, tau, deploy_contract(root, row))
                y = t["rgb_high_error"] == 1
                for m, s in scores.items():
                    ok = e & np.isfinite(s)
                    curve_groups[m].append(groups(s[ok], y[ok]))
                # All formal cells retain native system statistics. Visibility and boundary slices are descriptive.
                regions = {
                    "all": np.ones(v.shape, bool),
                    "observed": t["visibility"] == 1,
                    "disoccluded": t["visibility"] == 2,
                    "outside_input_fov": t["visibility"] == 3,
                    "background_or_invalid": t["visibility"] == 0,
                    "edge": t["edge"],
                    "interior": ~t["edge"],
                }
                measures = {r: system_counts(t, scores, tau, mask) for r, mask in regions.items()}
                write_json(
                    folder / (row["view"] + ".json"),
                    {"region": measures, "status": disp, "truth_sha256": row["full"]["truth"]["sha256"]},
                )
                view_rows.append(measures)
                status.append(
                    {
                        "family": family,
                        "condition": condition,
                        "view": row["view"],
                        "render": row["status"],
                        "reason": row["reason"],
                        "scientific_disposition": row.get("scientific_disposition", "selected"),
                        "failure_kind": row.get("failure_kind"),
                        "methods": disp,
                        "V": row["V"],
                        "E": row["E"],
                    }
                )
            family_metrics = {
                r: {m: aggregate_method([vr[r][m] for vr in view_rows]) for m in METHODS} for r in view_rows[0]
            }
            curves = {m: compact_curve(g, sum(r["V"] for r in rows), tau[m]) for m, g in curve_groups.items()}
            write_json(folder / "curves.json", curves)
            write_json(folder / "family.json", family_metrics)
            summaries.append(
                {
                    "family": family,
                    "condition": condition,
                    "group": rows[0].get("group"),
                    "methods": family_metrics["all"],
                }
            )
        macro = []
        for condition in sorted({r["condition"] for r in summaries}):
            rows = [r for r in summaries if r["condition"] == condition]
            for group in sorted({None, *[r["group"] for r in rows if r["group"]]}, key=lambda v: v or ""):
                chosen = rows if group is None else [r for r in rows if r["group"] == group]
                macro.append(
                    {
                        "condition": condition,
                        "cohort": group or "all",
                        "methods": {m: macro_families([r["methods"][m] for r in chosen]) for m in METHODS},
                    }
                )
        cases = []
        for row in ds["views"]:
            if row["view"] != "v01":
                continue
            cases.append(
                {
                    **{k: row[k] for k in ("family", "condition", "view", "status", "rgb", "full")},
                    "score_sources": {
                        k: entries[identity(row)]["score"] for k, (_, entries, _) in sources.items()
                    },
                }
            )
        output_files = ("families.json", "macro.json", "paired.json", "status.json", "case_index.json")
        for f, data in zip(output_files, (summaries, macro, paired_report(summaries), status, cases)):
            write_json(out / f, data)
        result = {
            "schema": "el-vision-comparison-1",
            "split": split,
            "dataset_sha256": ds_sha,
            "thresholds_sha256": threshold_pin[1],
            "inputs": {m: s[2] for m, s in sources.items()},
            "methods": list(METHODS),
            "thresholds": tau,
            "family_count": len({r["family"] for r in ds["views"]}),
            "planned_target_views": len(ds["views"]),
            "test_opening_sha256": ds["test_unlock_sha256"],
            "files": {p.relative_to(out).as_posix(): signature(p) for p in sorted(out.rglob("*.json"))},
        }
        write_json(out / "evaluation.json", result)
    return result


def main():
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze-dev")
    p.add_argument("--run-id", default=os.environ.get("EL_RUN_ID"))
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--baseline-freeze", nargs=2, metavar=("THRESHOLDS", "SHA256"))
    p = sub.add_parser("evaluate")
    p.add_argument("--split", required=True, choices=("development", "test"))
    p.add_argument("--thresholds", nargs=2, required=True)
    p.add_argument("--opening", nargs=2, metavar=("STORE", "COLD_SHELF"))
    for cmd in sub.choices.values():
        cmd.add_argument("--dataset", type=Path, required=True)
        cmd.add_argument("--sha256", required=True)
        cmd.add_argument("--scores", nargs=2, action="append", required=True, metavar=("SCORES_JSON", "SHA256"))
        cmd.add_argument("--catalog", nargs=2, action="append", required=True, metavar=("CATALOG", "SHA256"))
        cmd.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if a.command == "freeze-dev":
        require(a.run_id, "recorded development freeze ID required")
        freeze_dev(
            a.dataset, a.sha256, a.scores, a.catalog, a.output, a.run_id, a.baseline_only, a.baseline_freeze
        )
    else:
        evaluate(a.dataset, a.sha256, a.split, a.scores, a.thresholds, a.output, a.opening, a.catalog)


if __name__ == "__main__":
    main()
