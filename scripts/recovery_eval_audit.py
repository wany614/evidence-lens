"""Recompute frozen counts, tie-group AP, matched-coverage errors and saved labels from raw original pixels."""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from dtu_eval import mask, photo
from dtu_eval_replay import independent_labels, ap, equal
from dtu_score import METHODS, VIEWS
from evidence_lens.formal_inputs import require, verified_file
from evidence_lens.scoring_io import new_output, write_json
from recovery_eval_io import authority, permit_frozen, original
from vision_sources import signed


def count_replay(v, e, y, aux, risk, finite_all, tau):
    own = e & np.isfinite(risk)
    low = own & (risk.astype(np.float64) < tau)
    high = own & ~low
    unknown = e & ~own
    common = e & finite_all
    pos = y == 1
    require(not np.any(e & ~v) and np.all(np.isin(y[e], [0, 1])), "invalid independent labels")
    masks = {
        "V": v,
        "E": e,
        "Y": pos & e,
        "V_minus_E": v & ~e,
        "Y3": e & (aux > np.float32(0.05)),
        "C": common,
        "CY": common & pos,
        "S": own,
        "SY": own & pos,
        "L": low,
        "LY": low & pos,
        "H": high,
        "HY": high & pos,
        "U": unknown,
        "UY": unknown & pos,
        "missed_Y": (low | unknown) & pos,
    }
    return {name: int(np.count_nonzero(a)) for name, a in masks.items()}, {
        "common": ap(risk, y, common),
        "own": ap(risk, y, own),
    }


def coverage_replay(rows, versions, obj, cond, view):
    """Use order statistics plus Boolean masks; no shared sorting or coverage implementation with the evaluator."""
    v = versions["new"]["V"]
    N = int(v.sum())
    seed = int.from_bytes(
        hashlib.sha256(f"el-recovery-coverage-v1|{obj}|{cond}|{view}".encode()).digest()[:8], "big"
    )
    perm = np.random.Generator(np.random.PCG64(seed)).permutation(v.size)
    rank = np.argsort(perm).reshape(v.shape)
    domains = {side: rec["common"] for side, rec in versions.items()}
    paired = domains["old"] & domains["new"]
    seen = set()
    for row in rows:
        dname, name, m, a = (row[k] for k in ("domain", "version", "method", "anchor"))
        key = (dname, name, m, a)
        require(key not in seen, "duplicated matched-coverage record")
        seen.add(key)
        d = paired if dname == "paired_old_new" else domains[name]
        require(dname in ("paired_old_new", "within_" + name), "unknown matched coverage domain")
        rec = versions[name]
        values = rec["scores"][m][d]
        y = rec["y"][d] == 1
        if a.endswith("%_V"):
            percent = int(a.split("%")[0])
            div, rest = divmod(N * percent, 100)
            k = div + (rest > 50)
        else:
            which = a.split("_frozen_")[0]
            require(which in versions and (dname == "paired_old_new" or which == name), "wrong frozen anchor")
            bas = versions[which]
            k = int(np.count_nonzero(d & (bas["scores"]["MS-L1"].astype(np.float64) < bas["tau"]["MS-L1"])))
        require(
            row["V"] == N
            and row["eligible"] == len(values)
            and row["eligible_positive"] == int(y.sum())
            and row["requested_k"] == k,
            "incorrect matched-coverage denominator or comparator",
        )
        if k > len(values):
            require(
                row["reason"] == "outside_support" and row["accepted"] is None, "out-of-range point estimated"
            )
            continue
        if not k:
            require(
                row["reason"] == "empty_low" and row["accepted"] == 0 and row["risk"] is None,
                "wrong empty risk",
            )
            continue
        cut = np.partition(values, k - 1)[k - 1]
        below, ties = values < cut, values == cut
        take = k - int(below.sum())
        pos0, pos1 = int(y[below].sum()), int(y[ties].sum())
        tie_count = int(ties.sum())
        ranks = rank[d][ties]
        threshold = np.partition(ranks, take - 1)[take - 1]
        selected = pos0 + int(y[ties][ranks <= threshold].sum())
        expected = {
            "accepted": k,
            "accepted_positive": selected,
            "cutoff_score": float(cut),
            "cutoff_group_size": tie_count,
            "accepted_in_cutoff_group": take,
            "min_possible_accepted_positive": pos0 + max(0, take - (tie_count - pos1)),
            "max_possible_accepted_positive": pos0 + min(take, pos1),
            "reason": None,
        }
        require(
            all(row[key] == value for key, value in expected.items()),
            "independent matched cutoff or counts differ",
        )
    require(len(seen) == 4 * 9 * 2 + 8 * 10, "missing registered matched-coverage points")
    return len(seen)


def work(roots, req, out, rid):
    require(req["schema"] == "el-recovery-independent-audit-request-1", "wrong independent audit request")
    authority(roots, req)
    freeze = permit_frozen(roots, req)
    _, ep, doc = original(roots, req["evaluation"])
    inputs = doc["input_pins"]
    require(
        doc["schema"] == "el-recovery-frozen-risk-evaluation-1"
        and doc["run_id"] == req["evaluation"]["run_id"]
        and inputs["freeze"] == req["freeze"]
        and inputs["protocol"] == req["protocol"]
        and doc["step"] == freeze["stop_step"]
        and (req["object_id"], req["condition"]) == (doc["object_id"], doc["condition"]),
        "different scientific cell or final freeze",
    )
    _, rp, refs = original(roots, inputs["references"])
    _, _, historic = original(roots, req["historical_evaluation"])
    old_by_view = {r["view"]: r for r in historic["rows"]}
    require(
        historic["object_id"] == req["object_id"]
        and historic["condition"] == req["condition"]
        and historic["arm"] == "main-256"
        and historic["thresholds"] == doc["thresholds"],
        "wrong historical measurement or released operating points",
    )
    cases, scores = {}, {}
    for version, key in (("old", "original_case"), ("new", "new_case")):
        _, p, d = original(roots, inputs[key])
        cases[version] = (p.parent, d)
        _, p, d = original(roots, inputs[version + "_scores"])
        scores[version] = (p.parent, d)
        require(
            d["case_sha256"] == inputs[key]["sha256"] and d["models"] == doc["models"], "score origin differs"
        )
    gaps = []
    grouped = defaultdict(list)
    records = []
    checks = 0
    covered = 0
    measured = {r["view"]: r for r in doc["rows"]}
    require(set(measured) == set(VIEWS), "wrong six-view count")
    for view in VIEWS:
        rawref = refs["views"][view]
        ref = photo(rp.parent, rawref["rgb"])
        roi = mask(rp.parent, rawref["valid_mask"])
        versions = {}
        for vn in ("old", "new"):
            root, cm = cases[vn]
            spec = cm["renders"].get(view)
            pred = None
            if spec is not None:
                with np.load(verified_file(root, spec["path"], spec), allow_pickle=False) as z:
                    pred = z["rgb"]
            v, e, y, err, aux = independent_labels(ref, pred, roi)
            meta = measured[view]["versions"][vn]
            pix = meta["pixels"]
            with np.load(verified_file(ep.parent, pix["path"], pix), allow_pickle=False) as z:
                for name, a in (("V", v), ("E", e), ("Y", y), ("rgb_mae", err), ("min3_mae", aux)):
                    require(
                        z[name].dtype == a.dtype and np.array_equal(a, z[name], equal_nan=True),
                        "different saved pixel " + name,
                    )
                    checks += a.size
            risk = {}
            base, book = scores[vn]
            for m in METHODS:
                if pred is None:
                    risk[m] = np.full(v.shape, np.nan, np.float32)
                    continue
                q = book["views"][view]["methods"][m]
                blob = q["file"]
                require(
                    q["target_prediction_sha256"] == spec["sha256"],
                    "deployment score used another reconstruction",
                )
                with np.load(verified_file(base, blob["path"], blob), allow_pickle=False) as z:
                    risk[m] = z["risk"]
                    finite = np.isfinite(risk[m])
                    require(
                        np.array_equal(finite, z["valid"])
                        and np.all(z["state"][~finite] == 255)
                        and np.array_equal(
                            z["state"][finite], risk[m][finite].astype(np.float64) >= doc["thresholds"][m]
                        ),
                        "raw risk/state/threshold mismatch",
                    )
            all_finite = np.logical_and.reduce([np.isfinite(risk[m]) for m in METHODS])
            versions[vn] = {
                "V": v,
                "common": e & all_finite,
                "y": y,
                "scores": risk,
                "tau": doc["thresholds"],
            }
            for m in METHODS:
                n, aps = count_replay(v, e, y, aux, risk[m], all_finite, doc["thresholds"][m])
                estimate = meta["metrics"][m]
                require(
                    all(estimate["counts"][k] == val for k, val in n.items()), "raw confusion counts differ"
                )
                require(
                    estimate["counts"]["rgb_mae_sum"] == float(err[e].sum(dtype=np.float64))
                    and estimate["counts"]["aux3_mae_sum"] == float(aux[e].sum(dtype=np.float64)),
                    "original floating-point RGB or neighbourhood error differs",
                )
                for dom, value in aps.items():
                    require(equal(estimate[dom]["ap"], value), "sklearn tied AP differs")
                    if value is not None:
                        gaps.append(abs(estimate[dom]["ap"] - value))
                if vn == "old":
                    require(
                        estimate == old_by_view[view]["domains"]["full_crop"][m],
                        "old result differs from sealed D1",
                    )
                grouped[vn, m].append((n, aps))
                records.append({"view": view, "version": vn, "method": m, "counts": n, "AP": aps})
        covered += coverage_replay(
            [r for r in doc["matched_coverage"] if r["view"] == view],
            versions,
            req["object_id"],
            req["condition"],
            view,
        )
    for (version, m), data in grouped.items():
        est = doc["aggregate"][version][m]
        for k in data[0][0]:
            require(est["counts"][k] == sum(row[k] for row, _ in data), "incorrect condition total")
        for dom in ("own", "common"):
            use = [(r, a[dom]) for r, a in data if a[dom] is not None]
            val = sum(a for _, a in use) / len(use) if use else None
            n, d = ("CY", "C") if dom == "common" else ("SY", "S")
            prev = sum(r[n] / r[d] for r, _ in use) / len(use) if use else None
            require(
                equal(val, est["AP"][dom]["ap"])
                and equal(prev, est["matched_view_prevalence"][dom]["value"])
                and len(use) == est["AP"][dom]["defined_views"],
                "AP or prevalence condition aggregation mismatch",
            )
    result = {
        "schema": "el-recovery-independent-audit-1",
        "passed": True,
        "run_id": rid,
        "object_id": req["object_id"],
        "condition": req["condition"],
        "evaluation": req["evaluation"],
        "freeze": req["freeze"],
        "historical_evaluation": req["historical_evaluation"],
        "max_abs_sklearn_AP_difference": max(gaps, default=None),
        "independent_method_views": len(records),
        "saved_label_and_error_values_compared": checks,
        "matched_points_compared": covered,
        "records": records,
    }
    with new_output(out) as dest:
        write_json(dest / "audit.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("store", "shelf", "request", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--request-sha256", required=True)
    a = p.parse_args()
    r = work((a.store, a.shelf), signed(a.request, a.request_sha256), a.output, os.environ["EL_RUN_ID"])
    print(json.dumps({k: r[k] for k in ("passed", "independent_method_views", "matched_points_compared")}))
