"""Independently recalculate all T1 reconstruction counts and paired endpoints from original FP32 arrays."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
from t1_access import CONDITIONS, OBJECTS, TARGETS
from t1_study import DEFAULT, PROJECT, locate, selected, stage

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require


def records(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def same(value, saved):
    return (value is None and saved == "") or (
        value is not None and saved != "" and math.isclose(value, float(saved), rel_tol=2e-14, abs_tol=1e-12)
    )


def audit(roots, cfg, out):
    pins = cfg["sources"]
    term = load(locate(roots, pins["terminals"]))
    eval_file = locate(roots, pins["report"])
    require(
        term["evaluation_code_sha256"]["t1_nbv_independent.py"]
        == sha(PROJECT / "scripts/t1_nbv_independent.py")
        and load(eval_file)["terminals"] == pins["terminals"]
        and load(eval_file)["opening"] == pins["opening"],
        "Wrong pre-opening auditor or endpoint report",
    )
    viewrows = records(eval_file.parent / "reconstruction-views.csv")
    byview = {(r["object_id"], r["condition"], r["view"]): r for r in viewrows}
    condrows = records(eval_file.parent / "reconstruction-conditions.csv")
    bycond = {(r["object_id"], r["condition"]): r for r in condrows}
    paired = {r["object_id"]: r for r in records(eval_file.parent / "paired.csv")}
    require(
        len(byview) == len(viewrows) == 252 and len(bycond) == len(condrows) == 42 and len(paired) == 6,
        "Independent denominator mismatch",
    )
    references, aggregate, support = {}, {}, {}
    n_views = bad_count = comparable_count = 0
    for obj in OBJECTS:
        p = locate(roots, pins["ref-" + obj])
        book = load(p)
        require(book["opening"] == pins["opening"], "Reference from another opening")
        for view in TARGETS:
            e = book["views"][view]
            raw = p.parent / e["rgb"]["path"]
            require(
                sha(raw) == e["rgb"]["sha256"] and e["valid_mask"]["kind"] == "all_true",
                "Invalid reference input",
            )
            with Image.open(raw) as image:
                require(image.mode == "RGB" and image.size == (256, 256), "Unexpected reference image")
                references[obj, view] = np.asarray(image, dtype=np.float32) / np.float32(255)
    for c in term["cells"]:
        obj, cond = c["object_id"], c["condition"]
        cp = c["case"]
        case = locate(roots, cp) if cp else None
        book = load(case) if case else None
        sums = []
        E_all = Y_all = 0
        masks = []
        for v in TARGETS:
            rr = byview[obj, cond, v]
            target = references[obj, v]
            valid = np.isfinite(target).all(axis=-1) & (target.min(axis=-1) >= 0) & (target.max(axis=-1) <= 1)
            use = np.zeros_like(valid)
            if book and v in book["renders"]:
                desc = book["renders"][v]
                native = case.parent / desc["path"]
                require(sha(native) == desc["sha256"], "Unverified original target prediction")
                with np.load(native, allow_pickle=False) as data:
                    rgb = data["rgb"]
                require(rgb.dtype == np.float32 and rgb.shape == target.shape, "Non-FP32 reconstruction")
                use = valid & np.isfinite(rgb).all(-1) & (rgb.min(axis=-1) >= 0) & (rgb.max(axis=-1) <= 1)
                distance = np.abs(rgb[use] - target[use])
                # Match the registered FP32 sum and division; summation across pixels uses FP64.
                err = ((distance[:, 0] + distance[:, 1]) + distance[:, 2]) / np.float32(3)
            else:
                err = np.empty(0, np.float32)
            n, bad = err.size, int(np.count_nonzero(err > np.float32(0.05)))
            total = float(np.sum(err, dtype=np.float64))
            mask = hashlib.sha256(use.tobytes(order="C")).hexdigest()
            require(
                rr["E_mask_sha256"] == mask
                and int(rr["V"]) == int(valid.sum())
                and int(rr["E"]) == n
                and int(rr["Y"]) == bad
                and same(total, rr["rgb_mae_sum"])
                and same(total / n if n else None, rr["MAE"]),
                f"Independent original-array disagreement for {obj}/{cond}/{v}",
            )
            E_all += n
            Y_all += bad
            sums.append(total)
            masks.append(mask)
            n_views += 1
        value = sum(sums) / E_all if E_all else None
        cr = bycond[obj, cond]
        require(
            int(cr["E"]) == E_all
            and int(cr["Y"]) == Y_all
            and same(value, cr["MAE"])
            and same(Y_all / E_all if E_all else None, cr["Y_over_E"]),
            "Independent object endpoint mismatch",
        )
        aggregate[obj, cond], support[obj, cond] = value, masks
        comparable_count += E_all
        bad_count += Y_all
    deltas = []
    for obj in OBJECTS:
        row = paired[obj]
        u, r = aggregate[obj, "M4-uniform"], aggregate[obj, "M4-risk"]
        delta = (
            r - u
            if r is not None and u is not None and support[obj, "M4-risk"] == support[obj, "M4-uniform"]
            else None
        )
        require(same(delta, row["risk_minus_uniform_MAE"]), "Independent paired contrast mismatch")
        vals = [aggregate[obj, x] for x in CONDITIONS[-3:]]
        complete = all(v is not None for v in vals) and all(
            support[obj, x] == support[obj, CONDITIONS[-3]] for x in CONDITIONS[-3:]
        )
        for calc, key in ((mean, "random_mean_MAE"), (min, "random_min_MAE"), (max, "random_max_MAE")):
            require(same(calc(vals) if complete else None, row[key]), "Independent random summary mismatch")
        if delta is not None:
            deltas.append(delta)
    data = load(eval_file)
    require(
        data["paired_defined"] == len(deltas)
        and (
            data["risk_minus_uniform_object_equal"] == mean(deltas)
            if deltas
            else data["risk_minus_uniform_object_equal"] is None
        ),
        "Independent equal-object estimate mismatch",
    )
    save(
        out / "independent-nbv.json",
        {
            "schema": "el-t1-independent-nbv-1",
            "run_id": os.environ["EL_RUN_ID"],
            "report": pins["report"],
            "opening": pins["opening"],
            "checked_views": n_views,
            "checked_conditions": len(bycond),
            "checked_paired_objects": len(paired),
            "scheduled_pixels_including_logical_aliases": 252 * 256 * 256,
            "comparable_pixels_including_logical_aliases": comparable_count,
            "erroneous_pixels_including_logical_aliases": bad_count,
            "mismatches": 0,
            "risk_minus_uniform_equal_object": mean(deltas) if deltas else None,
        },
    )


def run(roots):
    s = roots[0]
    pins = {
        "terminals": selected(s, "t1-terminals"),
        "opening": selected(s, "t1-opening"),
        "report": selected(s, "t1-nbv-evaluation"),
    }
    pins.update({"ref-" + o: selected(s, "t1-reference-" + o) for o in OBJECTS})
    cmd = [
        "{python}",
        "-B",
        "{source}/scripts/t1_nbv_independent.py",
        "worker",
        "--config",
        "{config}",
        "--out",
        "{out}",
        "--store",
        str(s),
    ]
    return stage(roots, "t1-nbv-independent", "archive_verify", pins, {}, cmd, "independent-nbv.json", 90000)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("audit", "worker"))
    p.add_argument("--store", type=Path, default=DEFAULT)
    p.add_argument("--out", type=Path)
    p.add_argument("--config", type=Path)
    a = p.parse_args()
    roots = a.store.resolve(), a.store.resolve() / "history"
    if a.operation == "audit":
        print(json.dumps(run(roots)))
    else:
        audit(roots, load(a.config), a.out)
