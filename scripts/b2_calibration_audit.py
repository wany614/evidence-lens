"""Independently re-count C6 families and verify every committed full and leave-one-out candidate."""

import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from b2_intake_record import ROOT, locate, record_step
from t0_dtu_archive import check, digest, write_json


def load(path):
    return json.loads(path.read_bytes())


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def member(base, spec):
    rel = Path(spec["path"])
    check(not rel.is_absolute() and ".." not in rel.parts, "invalid member")
    raw = base / rel
    check(
        raw.stat().st_size == spec["bytes"]
        and digest(raw.read_bytes()) == spec["sha256"],
        "member content differs",
    )
    return raw


def frames(roots, cells, obj, condition, truthpin):
    receipt = locate(roots, truthpin)
    ref = load(receipt)
    cell = cells[obj, condition]
    case_path = locate(roots, cell["case"]) if cell.get("case") else None
    case = load(case_path) if case_path else None
    scores_path = locate(roots, cell["scores"]) if cell.get("scores") else None
    scores = load(scores_path) if scores_path else None
    vv = ee = yy = uu = rendered = 0
    scored = []
    events = []
    deploy = []
    for v, row in sorted(ref["views"].items()):
        verified = member(receipt.parent, row["rgb"])
        with Image.open(verified) as im:
            check(im.mode == "RGB" and im.size == (256, 256), "native RGB8 required")
            truth = np.asarray(im, dtype=np.float32) / np.float32(255)
        valid = np.isfinite(truth).all(axis=2)
        pred = None
        if case and v in case["renders"]:
            rendered += 1
            with np.load(
                member(case_path.parent, case["renders"][v]), allow_pickle=False
            ) as archive:
                pred = archive["rgb"]
        e = (
            np.zeros(valid.shape, bool)
            if pred is None
            else valid & np.all(np.isfinite(pred) & (pred >= 0) & (pred <= 1), axis=2)
        )
        y = np.zeros_like(e)
        if e.any():
            y[e] = np.abs(pred[e] - truth[e]).mean(
                axis=1, dtype=np.float32
            ) > np.float32(0.05)
        risk = np.full(valid.shape, np.nan, np.float32)
        if scores and scores["views"][v]["status"] == "ready":
            row = scores["views"][v]["methods"]["MS-L1"]["file"]
            with np.load(
                member(scores_path.parent, row), allow_pickle=False
            ) as archive:
                risk = archive["risk"]
                check(
                    np.array_equal(np.isfinite(risk), archive["valid"]),
                    "wrong score-support mask",
                )
        finite = np.isfinite(risk)
        s = finite & e
        deploy.append(risk[finite])
        scored.append(risk[s])
        events.append(y[s])
        vv += int(valid.sum())
        ee += int(e.sum())
        yy += int(y.sum())
        uu += int((e & ~s).sum())
    return {
        "V": vv,
        "E": ee,
        "Y": yy,
        "U_E": uu,
        "rendered": rendered,
        "deploy": np.sort(np.concatenate(deploy)).astype(np.float64),
        "scored": np.concatenate(scored).astype(np.float64),
        "events": np.concatenate(events),
    }


def direct_histograms(obj, thresholds):
    # Each pixel enters the first strictly larger threshold. Prefix the entry histograms once.
    if thresholds[0] is None:
        return [None] * len(thresholds), [None] * len(thresholds)
    boundary = np.asarray(thresholds, np.float64)
    bucket = np.searchsorted(boundary, obj["scored"], side="right")
    allpix = np.bincount(bucket, minlength=len(boundary) + 1).cumsum()
    errors = np.bincount(bucket[obj["events"]], minlength=len(boundary) + 1).cumsum()
    return [int(x) for x in allpix[:-1]], [int(x) for x in errors[:-1]]


def verify_weighted_quantiles(observations, thresholds):
    groups = [g["deploy"] for g in observations if len(g["deploy"])]
    if not groups:
        check(thresholds == [None] * 201, "all-empty candidate rows differ")
        return
    check(
        len(thresholds) == 201 and thresholds[0] == min(g[0] for g in groups),
        "minimum strict candidate differs",
    )
    selected = np.nextafter(np.asarray(thresholds[1:], np.float64), -np.inf)
    left = [np.searchsorted(g, selected, side="left") for g in groups]
    right = [np.searchsorted(g, selected, side="right") for g in groups]
    for j in range(200):
        target = Fraction((j + 1) * len(groups), 200)
        before = sum(
            (Fraction(int(x[j]), len(g)) for x, g in zip(left, groups)), Fraction()
        )
        through = sum(
            (Fraction(int(x[j]), len(g)) for x, g in zip(right, groups)), Fraction()
        )
        check(
            before < target <= through,
            "score candidate is not the whole-tie inverse of the object-equal CDF",
        )


def optional_int(raw):
    return int(raw) if raw not in (None, "") else None


def audit(cfg, out):
    roots = [Path(v) for v in cfg["roots"]]
    pins = cfg["pinned"]
    fitpath = locate(roots, pins["fit"])
    res = load(fitpath)
    root = fitpath.parent
    opening = load(locate(roots, pins["opening"]))
    grid = load(locate(roots, pins["candidates"]))["candidate_thresholds"]
    check(
        opening["score_candidates"] == pins["candidates"] == res["score_candidates"]
        and res["opening"] == pins["opening"],
        "calibration output and opening/candidate commitment differ",
    )
    cells = {(c["object_id"], c["condition"]): c for c in opening["terminal_cells"]}
    objects = opening["object_ids"]
    samples = {
        (o, c): frames(roots, cells, o, c, res["references"][o])
        for o in objects
        for c in ("M2", "M4")
    }
    detail = csv_rows(root / "candidate-object-counts.csv")
    curves = csv_rows(root / "candidate-curves.csv")
    loo = csv_rows(root / "leave-one-out.csv")
    keyed = {
        (r["condition"], r["fold"], r["object_id"], int(r["anchor"])): r for r in detail
    }
    aggregate = {(r["condition"], r["fold"], int(r["anchor"])): r for r in curves}
    leaves = {(r["condition"], r["held_out"]): r for r in loo}
    check(
        len(keyed) == len(detail) == 14472
        and len(aggregate) == len(curves) == 2814
        and len(leaves) == 12,
        "incomplete aggregate or family curve",
    )
    checked = 0
    summary = {}
    for condition in ("M2", "M4"):
        profile = load(root / "profiles" / (condition + ".json"))
        for fold, taus in grid[condition].items():
            omitted = fold.removeprefix("held-out-") if fold != "full" else None
            train = [o for o in objects if o != omitted]
            verify_weighted_quantiles([samples[o, condition] for o in train], taus)
            bins = {o: direct_histograms(samples[o, condition], taus) for o in train}
            best = None
            eligible = []
            for anchor, tau in enumerate(taus):
                cover = []
                risk = []
                for obj in train:
                    d = samples[obj, condition]
                    l, ly = [a[anchor] for a in bins[obj]]
                    saved = keyed[condition, fold, obj, anchor]
                    counts = {
                        "V": d["V"],
                        "E": d["E"],
                        "Y": d["Y"],
                        "S": len(d["scored"]),
                        "U_E": d["U_E"],
                        "V_minus_E": d["V"] - d["E"],
                        "L": l,
                        "LY": ly,
                    }
                    check(
                        all(optional_int(saved[k]) == x for k, x in counts.items()),
                        "independent raw-count discrepancy",
                    )
                    if d["V"] and l is not None:
                        cover.append(Fraction(l, d["V"]))
                    if l:
                        risk.append(Fraction(ly, l))
                    checked += 1
                c = (
                    sum(cover, Fraction()) / len(train)
                    if len(cover) == len(train)
                    else None
                )
                r = (
                    sum(risk, Fraction()) / len(train)
                    if len(risk) == len(train)
                    else None
                )
                good = (
                    c is not None
                    and r is not None
                    and c >= Fraction(1, 5)
                    and r <= Fraction(1, 20)
                )
                original = aggregate[condition, fold, anchor]
                check(
                    (original["feasible"] == "True") == good
                    and (
                        float(original["mean_coverage"])
                        if original["mean_coverage"]
                        else None
                    )
                    == (float(c) if c is not None else None)
                    and (
                        float(original["mean_low_error"])
                        if original["mean_low_error"]
                        else None
                    )
                    == (float(r) if r is not None else None),
                    "independent aggregate differs",
                )
                if good:
                    eligible.append((-c, r, tau, anchor))
            if eligible:
                best = min(eligible)
            picked = None if best is None else best[-1]
            if omitted is None:
                check(
                    picked == profile["selected_anchor"]
                    and (None if best is None else best[2]) == profile["tau"],
                    "profile selection differs",
                )
                summary[condition] = {
                    "status": profile["status"],
                    "feasible_candidates": len(eligible),
                    "selected_anchor": picked,
                }
            else:
                stated = leaves[condition, omitted]
                check(
                    optional_int(stated["selected_anchor"]) == picked,
                    "held-out training selection differs",
                )
                if best:
                    l, ly = direct_histograms(samples[omitted, condition], [best[2]])
                    check(
                        (int(stated["L"]), int(stated["LY"])) == (l[0], ly[0]),
                        "held-out application differs",
                    )
                else:
                    check(
                        stated["L"] == stated["LY"] == ""
                        and stated["held_out_status"] == "no_fitted_operating_point",
                        "infeasible fold mislabelled as an observed operating point",
                    )
    write_json(
        out / "calibration-audit.json",
        {
            "schema": "el-b2-independent-calibration-audit-1",
            "passed": True,
            "fit": pins["fit"],
            "opening": pins["opening"],
            "score_candidates": pins["candidates"],
            "native_prediction_views": sum(d["rendered"] for d in samples.values()),
            "checked_full_and_fold_cutoffs": len(aggregate),
            "checked_object_anchor_counts": checked,
            "checked_leave_one_out_fits": len(leaves),
            "raw_scores_used": "MS-L1",
            "cdf_check": "exact per-object rational mass strictly below and through observed tie",
            "count_check": "first strictly larger candidate per pixel; cumulative integer bucket counts",
            "outcomes": summary,
            "T1_photographs_read": 0,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("audit", "worker"))
    for key in ("store", "batch-one", "history", "fit", "out", "config"):
        p.add_argument("--" + key, type=Path)
    a = p.parse_args()
    if a.action == "worker":
        audit(load(a.config), a.out)
    else:
        roots = (
            a.store.resolve(),
            a.batch_one.resolve(),
            (a.history or ROOT / ".research").resolve(),
        )
        fit = load(a.fit)
        raw = load(locate(roots, fit))
        priors = {
            "fit": fit,
            "opening": raw["opening"],
            "candidates": raw["score_candidates"],
            **{"ref" + o: pin for o, pin in raw["references"].items()},
        }
        print(
            json.dumps(
                record_step(
                    roots,
                    "b2-independent-calibration-audit",
                    "b2_calibration_audit.py",
                    "payload/calibration-audit.json",
                    {"roots": list(map(str, roots))},
                    priors,
                    worker_args=("worker",),
                    extra_code=[
                        "scripts/b2_intake_record.py",
                        "src/evidence_lens/_bundle.py",
                        "src/evidence_lens/_verify.py",
                    ],
                    estimate=1024**2,
                )
            )
        )
