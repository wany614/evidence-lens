"""Evaluate the frozen four-method C matrix on original target grids and source-count ablations."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np

from evidence_lens.formal_inputs import require, verified_file
from evidence_lens.l1 import risk_state
from evidence_lens.multisource import CONDITIONS, METHODS, SCOPES, TARGETS, schema_sha256, source_ids
from evidence_lens.scoring_io import new_output, write_json
from stage3_risk_data import dataset, read_asset, union_visibility
from stage3_risk_learning import balanced, mean_defined
from stage3_risk_sources import asset, CATALOG, STUDY
from vision_evaluate import groups, ranking, compact_curve
from vision_sources import signed


RATES = {
    "rgb_error_fraction": ("Y", "E"),
    "confirmed_error_fraction_V": ("Y", "V"),
    "invalid_fraction": ("invalid", "V"),
    "rgb_mae": ("rgb_abs_sum", "E"),
    "depth_mae_m": ("depth_abs_sum", "Z"),
    "depth_signed_mean_m": ("depth_signed_sum", "Z"),
    "depth_relative_mae": ("depth_rel_sum", "Z"),
    "depth_high_fraction": ("ZY", "Z"),
    "depth_eligible_fraction": ("Z", "surface"),
    "common_coverage_E": ("S", "E"),
    "common_coverage_V": ("S", "V"),
    "common_prevalence": ("SY", "S"),
    "available_V": ("A", "V"),
    "low_coverage": ("L", "V"),
    "low_error": ("LY", "L"),
    "high_precision": ("HY", "H"),
    "error_detection": ("HY", "Y"),
    "missed_error_fraction": ("missed", "Y"),
    "unknown_E": ("U", "E"),
    "unknown_error": ("UY", "U"),
}
REGIONS = (
    "all",
    "observed",
    "disoccluded",
    "outside",
    "unresolved_or_background",
    "edge",
    "interior",
    "conflict",
    "no_conflict",
    "depth_conflict",
    "colour_conflict",
    "unknown_support",
    "support_0",
    "support_1",
    "support_2",
    "support_3",
    "support_4",
    "all_four_unknown",
)


def rates(counts):
    return {k: (counts[a] / counts[b] if counts[b] else None) for k, (a, b) in RATES.items()}


def score_indices(roster, d, digest):
    out = {}
    require(set(roster) == set(METHODS), "four score records required")
    for m in METHODS:
        rp = roster[m]
        root = Path(rp["directory"])
        j = signed(root / "scores.json", rp["sha256"])
        require(
            j["schema"] == "el-stage3-risk-scores-1"
            and j["method"] == m
            and j["dataset_sha256"] == digest
            and j["catalog_sha256"] == d["catalog_sha256"]
            and j["feature_schema_sha256"] == schema_sha256()
            and np.isfinite(j["threshold"])
            and 0 <= j["threshold"] <= 1.000001,
            "wrong original score parent",
        )
        for key in ("model_id", "model_index_sha256", "weights_sha256", "threshold"):
            require(j[key] == rp[key], f"substituted model, parameters, or threshold: {m}/{key}")
        require(len(j["views"]) == len(d["views"]), "incomplete scored targets")
        for v, row in zip(j["views"], d["views"], strict=True):
            require(
                all(v[k] == row[k] for k in ("family", "condition", "view", "status"))
                and set(v["modes"]) == set(SCOPES),
                "score ledger identifies another target",
            )
            for s in SCOPES:
                a = v["modes"][s]
                b = row["modes"][s]
                require(
                    a["source_ids"] == b["source_ids"]
                    and a["alias_of"] == b["alias_of"]
                    and (a["file"] is None) == (v["status"] == "render_failed"),
                    "score row or ablation identity differs",
                )
                if a["alias_of"]:
                    require(
                        a["file"] == v["modes"][a["alias_of"]]["file"], "M2 two/all original bytes differ"
                    )
            if m == "B-L1-transfer" and v["status"] == "ready":
                require(
                    len({v["modes"][s]["file"]["sha256"] for s in SCOPES}) == 1,
                    "s00 transfer depends on scorer subset",
                )
        out[m] = (root, j)
    return out


def view_scores(droot, row, indices, i, scope, t, deploy):
    shape = t["reference_valid"].shape
    result = {}
    for m, (root, j) in indices.items():
        if row["status"] != "ready":
            result[m] = np.full(shape, np.nan, np.float32)
            continue
        spec = j["views"][i]["modes"][scope]["file"]
        a = read_asset(root, spec)
        require(
            set(a) == {"risk", "valid", "state"}
            and a["risk"].dtype == np.float32
            and a["valid"].dtype == bool
            and a["state"].dtype == np.uint8
            and all(v.shape == shape for v in a.values())
            and not np.isinf(a["risk"]).any(),
            "wrong native score array",
        )
        risk = a["risk"]
        ok = a["valid"]
        require(
            np.array_equal(np.isfinite(risk), ok)
            and np.all((risk[ok] >= 0) & (risk[ok] <= 1))
            and np.array_equal(risk_state(risk, j["threshold"]), a["state"]),
            "score/state/threshold differs",
        )
        if m == "B-L1-transfer":
            require(
                np.array_equal(risk, read_asset(droot, row["transfer"]), equal_nan=True),
                "s00 frozen transfer scores changed",
            )
        else:
            require(
                np.array_equal(ok, deploy["scoreable"]),
                "returned method changed its declared native support: " + m,
            )
        result[m] = risk
    return result


def regions(t, scopes, deploy, scores):
    shape = t["reference_valid"].shape
    vis = union_visibility(t, scopes)
    un = np.logical_and.reduce([~np.isfinite(sc) for sc in scores.values()])
    if deploy is None:
        # No prediction means no observable support count. A measured count of zero is different.
        measured = np.zeros(shape, bool)
        support = np.zeros(shape, np.uint8)
        conflict = depth = colour = support
    else:
        measured = np.ones(shape, bool)
        support = deploy["support_count"]
        conflict = deploy["conflict_count"]
        depth = deploy["depth_conflict_count"]
        colour = deploy["colour_conflict_count"]
    return {
        "all": np.ones(shape, bool),
        "observed": vis == 1,
        "disoccluded": vis == 2,
        "outside": vis == 3,
        "unresolved_or_background": vis == 0,
        "edge": t["edge"],
        "interior": ~t["edge"],
        "conflict": measured & (conflict > 0),
        "no_conflict": measured & (conflict == 0),
        "depth_conflict": measured & (depth > 0),
        "colour_conflict": measured & (colour > 0),
        "unknown_support": ~measured,
        **{"support_" + str(n): measured & (support == n) for n in range(5)},
        "all_four_unknown": un,
    }


def measurement(t, scores, tau, mask):
    v = t["reference_valid"] & mask
    e = t["rgb_evaluable"] & mask
    y = (t["rgb_high_error"] == 1) & e
    z = t["depth_evaluable"] & mask
    require(not np.any(e & ~v) and np.all((t["rgb_high_error"][e] <= 1)), "invalid RGB denominators")
    common = e & np.logical_and.reduce([np.isfinite(x) for x in scores.values()])

    def n(a):
        return int(np.count_nonzero(a))

    base = {
        "V": n(v),
        "E": n(e),
        "invalid": n(v) - n(e),
        "Y": n(y),
        "S": n(common),
        "SY": n(common & y),
        "surface": n(t["reference_surface"] & mask),
        "Z": n(z),
        "ZY": n(z & (t["depth_high_error"] == 1)),
        "rgb_abs_sum": float(t["rgb_error"][e].sum(dtype=np.float64)),
        "depth_abs_sum": float(t["depth_error"][z].sum(dtype=np.float64)),
        "depth_signed_sum": float(t["signed_depth_error"][z].sum(dtype=np.float64)),
        "depth_rel_sum": float(t["relative_depth_error"][z].sum(dtype=np.float64)),
    }
    result = {}
    for m, x in scores.items():
        ok = e & np.isfinite(x)
        lo = ok & (x.astype(np.float64) < tau[m])
        hi = ok & ~lo
        un = e & ~ok
        c = {
            **base,
            "A": n(ok),
            "L": n(lo),
            "H": n(hi),
            "U": n(un),
            "LY": n(lo & y),
            "HY": n(hi & y),
            "UY": n(un & y),
        }
        c["missed"] = c["LY"] + c["UY"]
        require(
            c["L"] + c["H"] + c["U"] == c["E"] and c["missed"] + c["HY"] == c["Y"],
            "state count partition fails",
        )
        cr = ranking(x[common], y[common])
        own = cr if np.array_equal(ok, common) else ranking(x[ok], y[ok])
        result[m] = {"counts": c, "rates": rates(c), "ranking": {"common": cr, "own": own}}
    return result


def pooled(rows):
    c = {k: sum(r["counts"][k] for r in rows) for k in rows[0]["counts"]}
    r = {}
    for domain in ("own", "common"):
        vals = [x["ranking"][domain]["ap"] for x in rows]
        r[domain] = {
            "ap": mean_defined(vals),
            "defined_views": sum(a is not None for a in vals),
            "planned_views": len(vals),
        }
    return {"counts": c, "rates": rates(c), "ranking": r}


def family_macro(rows):
    vals = {k: [r["rates"][k] for r in rows] for k in RATES}
    vals.update({d + "_ap": [r["ranking"][d]["ap"] for r in rows] for d in ("common", "own")})
    return {
        k: {
            "mean": mean_defined(a),
            "defined_families": sum(x is not None for x in a),
            "planned_families": len(a),
        }
        for k, a in vals.items()
    }


def paired_reconstruction(root, d):
    table = {(r["family"], r["condition"], r["view"]): r for r in d["views"]}
    output = []
    for f in d["families"]:
        for v in TARGETS:
            rows = [table[f, c, v] for c in CONDITIONS]
            ts = [read_asset(root, r["truth"]) for r in rows]
            require(
                rows[0]["reference_member_sha256"] == rows[1]["reference_member_sha256"]
                and np.array_equal(ts[0]["reference_valid"], ts[1]["reference_valid"]),
                "paired cases lack common reference",
            )
            e = ts[0]["rgb_evaluable"] & ts[1]["rgb_evaluable"]
            z = ts[0]["depth_evaluable"] & ts[1]["depth_evaluable"]
            row = {
                "family": f,
                "view": v,
                "V": int(ts[0]["reference_valid"].sum()),
                "paired_E": int(e.sum()),
                "paired_Z": int(z.sum()),
            }
            for c, t in zip(CONDITIONS, ts, strict=True):
                row[c] = {
                    "E": int(t["rgb_evaluable"].sum()),
                    "Z": int(t["depth_evaluable"].sum()),
                    "paired_Y": int((t["rgb_high_error"][e] == 1).sum()),
                    "paired_rgb_abs_sum": float(t["rgb_error"][e].sum(dtype=np.float64)),
                    "paired_ZY": int((t["depth_high_error"][z] == 1).sum()),
                    "paired_z_abs_sum": float(t["depth_error"][z].sum(dtype=np.float64)),
                }
            output.append(row)
    groups = {}
    for f in d["families"]:
        vs = [a for a in output if a["family"] == f]
        result = {k: sum(r[k] for r in vs) for k in ("V", "paired_E", "paired_Z")}
        for c in CONDITIONS:
            vals = {k: sum(r[c][k] for r in vs) for k in vs[0][c]}
            for k, num, div in (
                ("rgb_mae", "paired_rgb_abs_sum", "paired_E"),
                ("rgb_error_fraction", "paired_Y", "paired_E"),
                ("depth_mae_m", "paired_z_abs_sum", "paired_Z"),
                ("depth_error_fraction", "paired_ZY", "paired_Z"),
            ):
                vals[k] = vals[num] / result[div] if result[div] else None
            result[c] = vals
        result["M4_minus_M2"] = {
            k: result["M4"][k] - result["M2"][k] if None not in (result["M4"][k], result["M2"][k]) else None
            for k in ("rgb_mae", "rgb_error_fraction", "depth_mae_m", "depth_error_fraction", "E", "Z")
        }
        groups[f] = result
    return {
        "direction": "M4 minus M2 on original shared E and separately shared depth domain",
        "views": output,
        "families": groups,
    }


def study(root, ds_sha, split, roster, out, *, roots=None, opening=None, boards=True):
    d = dataset(root, ds_sha, split, roots=roots, opening=opening)
    require(
        split in ("development", "test") and d["cohort"] == "formal" and len(d["families"]) == 4,
        "wrong formal cohort",
    )
    if split == "test":
        from evidence_lens.stage3_gate import get_frozen_method, require_opening

        mk = require_opening(roots[0], CATALOG, opening_sha256=opening, shelf=roots[1])
        m = get_frozen_method(roots[0], mk, shelf=roots[1])
        for n in METHODS:
            require(
                all(
                    m["models"][n][k] == roster[n][k]
                    for k in ("model_id", "model_index_sha256", "weights_sha256", "threshold")
                ),
                "test score roster differs from official method " + n,
            )
    idx = score_indices(roster, d, ds_sha)
    tau = {m: j["threshold"] for m, (_, j) in idx.items()}
    views = []
    family = []
    curvebins = defaultdict(list)
    caseindex = []
    ablation_views = []
    with new_output(out) as dst:
        for i, r in enumerate(d["views"]):
            f, c, v = r["family"], r["condition"], r["view"]
            t = read_asset(root, r["truth"])
            seen = {}
            picture = {}
            for scope in SCOPES:
                alias = r["modes"][scope]["alias_of"]
                if alias:
                    views.append({**seen[alias], "scope": scope, "alias_of": alias})
                    picture[scope] = picture[alias]
                    continue
                dep = read_asset(root, r["modes"][scope]["deploy"]) if r["status"] == "ready" else None
                scores = view_scores(root, r, idx, i, scope, t, dep)
                if scope == "all" or (scope == "two" and c == "M2"):
                    if r["status"] == "ready":
                        sel = read_asset(root, r["selection_support"])
                        require(
                            np.array_equal(
                                sel,
                                t["rgb_evaluable"]
                                & np.logical_and.reduce([np.isfinite(s) for s in scores.values()]),
                            ),
                            "registered selection support disagrees with four original score records",
                        )
                masks = regions(t, source_ids(c, scope), dep, scores)
                result = {name: measurement(t, scores, tau, mask) for name, mask in masks.items()}
                row = {
                    "family": f,
                    "condition": c,
                    "view": v,
                    "scope": scope,
                    "alias_of": None,
                    "status": r["status"],
                    "reconstruction": r.get("case"),
                    "failure": r["failure"],
                    "source_ids": list(source_ids(c, scope)),
                    "reference_member_sha256": r["reference_member_sha256"],
                    "prediction_member_sha256": r["prediction_member_sha256"],
                    "selection_sha256": r["selection_sha256"],
                    "source_camera_distances": r.get("source_camera_distances"),
                    "alpha_depth": {
                        key: r.get(key)
                        for key in (
                            "surface",
                            "depth_comparable",
                            "predicted_ed_valid",
                            "alpha_valid",
                            "alpha_at_least_half",
                            "alpha_mean",
                        )
                    },
                    "regions": result,
                }
                views.append(row)
                seen[scope] = row
                picture[scope] = scores
                for method, x in scores.items():
                    ok = t["rgb_evaluable"] & np.isfinite(x)
                    curvebins[f, c, scope, method].append(groups(x[ok], t["rgb_high_error"][ok]))
            shared = t["rgb_evaluable"] & np.logical_and.reduce(
                [np.isfinite(a) for g in picture.values() for a in g.values()]
            )
            ablation_views.append(
                {
                    "family": f,
                    "condition": c,
                    "view": v,
                    "E": int(t["rgb_evaluable"].sum()),
                    "common_all_scopes": int(shared.sum()),
                    "common_positives": int((t["rgb_high_error"][shared] == 1).sum()),
                    "AP": {
                        scope: {
                            m: ranking(picture[scope][m][shared], t["rgb_high_error"][shared])["ap"]
                            for m in METHODS
                        }
                        for scope in SCOPES
                    },
                }
            )
            if boards:
                from stage3_risk_boards import save_case

                caseindex.append(save_case(root, dst, r, t, picture, tau))
        for f in d["families"]:
            for c in CONDITIONS:
                for scope in SCOPES:
                    alias = "two" if (c, scope) == ("M2", "all") else None
                    selected = [
                        r for r in views if (r["family"], r["condition"], r["scope"]) == (f, c, scope)
                    ]
                    for method in METHODS:
                        regions_out = {
                            k: pooled([v["regions"][k][method] for v in selected]) for k in REGIONS
                        }
                        overall = regions_out["all"]
                        curve = compact_curve(
                            curvebins[f, c, alias or scope, method], overall["counts"]["V"], tau[method]
                        )
                        require(
                            (curve["frozen_L"], curve["frozen_LY"])
                            == (overall["counts"]["L"], overall["counts"]["LY"]),
                            "risk-coverage cutoff disagrees with system operating counts",
                        )
                        family.append(
                            {
                                "family": f,
                                "condition": c,
                                "scope": scope,
                                "alias_of": alias,
                                "method": method,
                                "threshold": tau[method],
                                "regions": regions_out,
                                "risk_coverage": curve,
                            }
                        )
        macros = []
        paired = []
        for c in CONDITIONS:
            for scope in SCOPES:
                for method in METHODS:
                    g = [r for r in family if (r["condition"], r["scope"], r["method"]) == (c, scope, method)]
                    macros.append(
                        {
                            "condition": c,
                            "scope": scope,
                            "alias_of": g[0]["alias_of"],
                            "method": method,
                            "regions": {k: family_macro([r["regions"][k] for r in g]) for k in REGIONS},
                        }
                    )
        table = {(r["family"], r["condition"], r["scope"], r["method"]): r["regions"]["all"] for r in family}
        for f in d["families"]:
            for m in METHODS:
                for c in CONDITIONS:
                    for left, right in (("one", "two"), ("one", "all"), ("two", "all")):
                        a, b = table[f, c, left, m], table[f, c, right, m]

                        # AP is based on each scope's declared four-method intersection; scope-specific prevalence/coverage accompany it.
                        def rank_data(x):
                            return {
                                "common_ap": x["ranking"]["common"]["ap"],
                                "defined_views": x["ranking"]["common"]["defined_views"],
                                **{
                                    k: x["rates"][k]
                                    for k in (
                                        "common_coverage_E",
                                        "common_prevalence",
                                        "low_coverage",
                                        "low_error",
                                    )
                                },
                            }

                        ra, rb = rank_data(a), rank_data(b)
                        paired.append(
                            {
                                "family": f,
                                "condition": c,
                                "method": m,
                                "from": left,
                                "to": right,
                                "alias_same_result": (c, left, right) == ("M2", "two", "all"),
                                "left": ra,
                                "right": rb,
                                "difference": {
                                    k: rb[k] - ra[k] if None not in (ra[k], rb[k]) else None for k in ra
                                },
                            }
                        )
        recon = paired_reconstruction(root, d)
        metrics = [
            ("views", views),
            ("families", family),
            ("macros", macros),
            ("score-ablation", paired),
            ("same-pixels-across-scopes", ablation_views),
            ("reconstruction", recon),
            (
                "resources",
                [
                    {
                        "family": c["family"],
                        "condition": c["condition"],
                        "failure": c.get("failure"),
                        "selected": c["selected"],
                        "attempts": c.get("attempts"),
                        "cost": c.get("cost"),
                    }
                    for c in d["cells"]
                ],
            ),
            ("case-index", caseindex),
        ]
        for name, obj in metrics:
            write_json(dst / (name + ".json"), obj)
        doc = {
            "schema": "el-stage3-risk-evaluation-1",
            "run_id": os.environ.get("EL_RUN_ID"),
            "catalog_sha256": CATALOG,
            "study_sha256": STUDY,
            "dataset_sha256": ds_sha,
            "split": split,
            "opening_sha256": opening,
            "method_sha256": d.get("method_sha256"),
            "families": d["families"],
            "methods": list(METHODS),
            "models": {
                m: {
                    k: x[k]
                    for k in ("sha256", "model_id", "model_index_sha256", "weights_sha256", "threshold")
                }
                for m, x in roster.items()
            },
            "dispositions": {
                "planned_cells": len(d["cells"]),
                "planned_target_views": len(d["views"]),
                "returned_target_views": sum(r["status"] == "ready" for r in d["views"]),
                "scheduled_scorer_modes": len(views),
                "nonredundant_scorer_modes": sum(r["alias_of"] is None for r in views),
            },
            "files": {n: asset(dst, dst / (n + ".json")) for n, _ in metrics},
        }
        from stage3_risk_public_tables import public_tables

        doc["public_numerical"] = public_tables(dst, doc, d, family, macros, recon, paired, ablation_views)
        write_json(dst / "evaluation.json", doc)
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("dataset", "roster", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--dataset-sha256", required=True)
    p.add_argument("--roster-sha256", required=True)
    p.add_argument("--split", choices=("development", "test"), required=True)
    p.add_argument("--store", type=Path)
    p.add_argument("--shelf", type=Path)
    p.add_argument("--opening-sha256")
    a = p.parse_args()
    result = study(
        a.dataset,
        a.dataset_sha256,
        a.split,
        signed(a.roster, a.roster_sha256),
        a.output,
        roots=(a.store, a.shelf),
        opening=a.opening_sha256,
    )
    print(json.dumps(result["dispositions"]))
