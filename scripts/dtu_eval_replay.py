"""Read back original DTU inputs and check every saved evaluation pixel and ranking independently."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from dtu_eval import read_pinned, mask, photo
from dtu_score import METHODS, VIEWS
from evidence_lens.formal_inputs import require, verified_file
from evidence_lens.scoring_io import new_output, write_json
from vision_sources import signed


def independent_labels(ref, pred, allowed):
    v = allowed & np.isfinite(ref).all(-1) & np.logical_and(ref >= 0, ref <= 1).all(-1)
    e = (
        np.zeros_like(v)
        if pred is None
        else v & np.isfinite(pred).all(-1) & np.logical_and(pred >= 0, pred <= 1).all(-1)
    )
    y = np.full(v.shape, 255, np.uint8)
    err = np.full(v.shape, np.nan, np.float32)
    aux = err.copy()
    if pred is None:
        return v, e, y, err, aux
    d = np.abs(ref[e] - pred[e])
    err[e] = (d[:, 0] + d[:, 1] + d[:, 2]) / np.float32(3)
    y[e] = err[e] > np.float32(0.05)
    padded = np.pad(np.where(v[:, :, None], ref, np.nan), ((1, 1), (1, 1), (0, 0)), constant_values=np.nan)
    h, w = v.shape
    candidates = []
    for i in range(3):
        for j in range(3):
            diff = np.abs(pred - padded[i : i + h, j : j + w])
            candidates.append((diff[:, :, 0] + diff[:, :, 1] + diff[:, :, 2]) / np.float32(3))
    # Skip invalid neighbours individually. Every evaluable centre is itself an eligible neighbour.
    aux[e] = np.fmin.reduce(np.stack(candidates), axis=0)[e]
    return v, e, y, err, aux


def ap(risk, y, valid):
    a = y[valid] == 1
    return float(average_precision_score(a, risk[valid])) if a.any() and (~a).any() else None


def equal(a, b):
    return a is None and b is None or a is not None and b is not None and abs(a - b) <= 1e-12


def check(roots, request, out, rid):
    require(request["schema"] == "el-dtu-evaluation-replay-request-1", "wrong regrade request")
    _, auth = read_pinned(roots, request["authority"])
    if request["role"] == "external":
        g = json.loads(auth.read_bytes())
        require(
            g["schema"] == "el-dtu-reference-gate-1" and request["object_id"] in g["object_ids"],
            "external photo replay remains sealed",
        )
    _, src = read_pinned(roots, request["evaluation"])
    d = json.loads(src.read_bytes())
    inputs = d["input_pins"]
    require(
        d["schema"] == "el-dtu-condition-evaluation-1"
        and d["role"] == request["role"]
        and d["object_id"] == request["object_id"]
        and request["authority"] == inputs["authority"],
        "unexpected scientific output or authority",
    )
    refs = case = ss = None
    if d["status"] != "reference_unavailable":
        _, rp = read_pinned(roots, inputs["references"])
        refs = json.loads(rp.read_bytes())
    if "case" in inputs:
        _, cp = read_pinned(roots, inputs["case"])
        case = json.loads(cp.read_bytes())
    if "scores" in inputs and d["status"] == "ready":
        _, sp = read_pinned(roots, inputs["scores"])
        ss = json.loads(sp.read_bytes())
    obs = []
    by = {}
    gaps = []
    for r in d["rows"]:
        v = r["view"]
        if refs is None:
            require(all(a is None for a in r["domains"].values()), "missing references acquired counts")
            continue
        rr = refs["views"][v]
        valid = mask(rp.parent, rr["valid_mask"])
        ref = photo(rp.parent, rr["rgb"])
        if case is not None and v in case["renders"]:
            p = verified_file(cp.parent, case["renders"][v]["path"], case["renders"][v])
            with np.load(p, allow_pickle=False) as z:
                pred = z["rgb"]
        else:
            pred = None
        risks = {}
        states = {}
        for m in METHODS:
            if ss and ss["views"][v]["status"] == "ready":
                fs = ss["views"][v]["methods"][m]["file"]
                pp = verified_file(sp.parent, fs["path"], fs)
                with np.load(pp, allow_pickle=False) as z:
                    risks[m] = z["risk"]
                    states[m] = z["state"]
            else:
                risks[m] = np.full(valid.shape, np.nan, np.float32)
                states[m] = np.full(valid.shape, 255, np.uint8)
        pp = verified_file(src.parent, r["labels"]["path"], r["labels"])
        with np.load(pp, allow_pickle=False) as saved:
            for dom in ("full_crop", "object"):
                if dom == "object" and rr.get("object_mask") is None:
                    require(r["domains"][dom] is None, "invented object domain")
                    continue
                zone = valid if dom == "full_crop" else valid & mask(rp.parent, rr["object_mask"])
                vv, e, y, err, aux = independent_labels(ref, pred, zone)
                for k, a in zip(("V", "E", "Y", "rgb_mae", "min3_mae"), (vv, e, y, err, aux)):
                    other = saved[dom + "_" + k]
                    require(
                        other.dtype == a.dtype and np.array_equal(other, a, equal_nan=True),
                        "different original pixel " + k,
                    )
                common = e & np.logical_and.reduce([np.isfinite(x) for x in risks.values()])
                yy = e & (y == 1)
                for m in METHODS:
                    s = e & np.isfinite(risks[m])
                    lo = s & (risks[m].astype(np.float64) < d["thresholds"][m])
                    hi = s & ~lo
                    u = e & ~s
                    require(
                        np.array_equal(states[m][s], hi[s]) and np.all(states[m][u] == 255),
                        "incorrect frozen state",
                    )
                    maps = {
                        "V": vv,
                        "E": e,
                        "V_minus_E": vv & ~e,
                        "Y": yy,
                        "Y3": e & (aux > np.float32(0.05)),
                        "C": common,
                        "CY": common & yy,
                        "S": s,
                        "SY": s & yy,
                        "H": hi,
                        "HY": hi & yy,
                        "L": lo,
                        "LY": lo & yy,
                        "U": u,
                        "UY": u & yy,
                        "missed_Y": (u | lo) & yy,
                    }
                    counts = {k: int(np.count_nonzero(val)) for k, val in maps.items()}
                    orig = r["domains"][dom][m]
                    require(
                        all(counts[k] == orig["counts"][k] for k in counts), "original target counts differ"
                    )
                    aps = {k: ap(risks[m], y, ms) for k, ms in [("own", s), ("common", common)]}
                    require(
                        all(equal(val, orig[k]["ap"]) for k, val in aps.items()), "independent AP differs"
                    )
                    gaps.extend(abs(val - orig[k]["ap"]) for k, val in aps.items() if val is not None)
                    by.setdefault((dom, m), []).append((counts, aps))
                    obs.append({"view": v, "domain": dom, "method": m, "counts": counts, "AP": aps})
    for (dom, m), rows in by.items():
        g = d["aggregate"][dom]["methods"][m]
        total = {k: sum(cs[k] for cs, _ in rows) for k in rows[0][0]}
        require(all(total[k] == g["counts"][k] for k in total), "object total differs")
        for k in ("own", "common"):
            vals = [a[k] for _, a in rows if a[k] is not None]
            value = sum(vals) / len(vals) if vals else None
            require(
                equal(value, g["AP"][k]["ap"]) and g["AP"][k]["defined_views"] == len(vals),
                "object AP differs",
            )
        require(
            g["risk_coverage"]["frozen_L"] == total["L"] and g["risk_coverage"]["frozen_LY"] == total["LY"],
            "frozen curve differs",
        )
    res = {
        "schema": "el-dtu-original-pixel-replay-1",
        "run_id": rid,
        "passed": True,
        "evaluation": request["evaluation"],
        "records": obs,
        "object_id": d["object_id"],
        "condition": d["condition"],
        "planned_views": len(VIEWS),
        "measured_method_domains": len(obs),
        "max_abs_ap_gap": max(gaps) if gaps else None,
    }
    with new_output(out) as dest:
        write_json(dest / "replay.json", res)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("store", "shelf", "request", "output"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--request-sha256", required=True)
    a = p.parse_args()
    z = check((a.store, a.shelf), signed(a.request, a.request_sha256), a.output, os.environ.get("EL_RUN_ID"))
    print(
        json.dumps(
            {
                k: z[k]
                for k in ("passed", "object_id", "condition", "measured_method_domains", "max_abs_ap_gap")
            }
        )
    )
