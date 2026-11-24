"""Apply the frozen full-crop NBV endpoint to every registered T1 trajectory after the joint opening."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np
from dtu_eval import photo
from recovery_eval_quality import measurement
from t0_evaluate import table
from t1_access import CONDITIONS, OBJECTS, TARGETS, H
from t1_study import DEFAULT, PROJECT, locate, selected, stage

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from evidence_lens.multisource_io import Case

RANDOM = CONDITIONS[-3:]


def endpoint(roots, cfg, out):
    pins = cfg["sources"]
    t = load(locate(roots, pins["terminals"]))
    op = load(locate(roots, pins["opening"]))
    require(
        t["schema"] == "el-t1-reconstruction-terminals-1"
        and op["schema"] == "el-t1-joint-opening-1"
        and op["condition_count"] == 42
        and op["fixed_score_count"] == 288
        and op["predictions_and_scores_sealed"] is True
        and t["entry"] == op["entry"],
        "Joint terminal authority required",
    )
    require(
        t["evaluation_code_sha256"]["t1_nbv_evaluate.py"] == sha(PROJECT / "scripts/t1_nbv_evaluate.py"),
        "Reconstruction endpoint evaluator differed after opening",
    )
    f = load(locate(roots, pins["freeze"]))
    for n in (
        "scripts/recovery_eval_quality.py",
        "scripts/dtu_eval.py",
        "scripts/b2_evaluate_development.py",
    ):
        require(
            sha(PROJECT / n) == f["scientific_code_sha256"][n], "Frozen numerical endpoint source changed"
        )
    views, conditions, paired = [], [], []
    by_key = {(r["object_id"], r["condition"]): r for r in t["cells"]}
    for o in OBJECTS:
        path = locate(roots, pins["ref-" + o])
        ref_doc = load(path)
        require(
            ref_doc["object_id"] == o
            and ref_doc["opening"] == pins["opening"]
            and [ref_doc["views"][v]["official_view_id"] for v in TARGETS] == H,
            "Wrong target provenance",
        )
        ref = {v: photo(path.parent, ref_doc["views"][v]["rgb"]) for v in TARGETS}
        by = {}
        for branch in CONDITIONS:
            cell = by_key[o, branch]
            cp = cell["case"]
            case = Case(locate(roots, cp).parent, cp["sha256"]) if cp else None
            rr = []
            for v in TARGETS:
                mask = ref_doc["views"][v]["valid_mask"]
                require(
                    mask == {"kind": "all_true", "width": 256, "height": 256}
                    and ref_doc["views"][v]["object_mask"] is None,
                    "Changed photo domain",
                )
                pred = case.frame(v) if case else None
                values, (_, e, _, _) = measurement(
                    ref[v], pred["rgb"] if pred else None, np.ones((256, 256), dtype=bool)
                )
                row = dict(
                    object_id=o,
                    condition=branch,
                    view=v,
                    official_target=ref_doc["views"][v]["official_view_id"],
                    **values,
                    E_mask_sha256=hashlib.sha256(e.tobytes(order="C")).hexdigest(),
                    status=cell["status"],
                    case_sha256=cp["sha256"] if cp else None,
                    alias_of=cell["alias_of"],
                    reference_sha256=ref_doc["views"][v]["rgb"]["sha256"],
                )
                views.append(row)
                rr.append(row)
            totals = {k: sum(r[k] for r in rr) for k in ("V", "E", "Y", "rgb_mae_sum")}
            E = totals["E"]
            row = dict(
                object_id=o,
                condition=branch,
                **totals,
                MAE=totals["rgb_mae_sum"] / E if E else None,
                Y_over_E=totals["Y"] / E if E else None,
                comparable_views=sum(r["E"] > 0 for r in rr),
                scheduled_views=6,
                V_minus_E=totals["V"] - E,
                status=cell["status"],
                alias_of=cell["alias_of"],
            )
            conditions.append(row)
            by[branch] = (row, [r["E_mask_sha256"] for r in rr])
        risk, uniform = by["M4-risk"], by["M4-uniform"]
        same = risk[1] == uniform[1]
        comparable = same and risk[0]["MAE"] is not None and uniform[0]["MAE"] is not None
        random = [by[n][0]["MAE"] for n in RANDOM]
        ok = all(v is not None for v in random) and all(by[k][1] == by[RANDOM[0]][1] for k in RANDOM)
        paired.append(
            dict(
                object_id=o,
                M2_MAE=by["M2-fixed"][0]["MAE"],
                fixed_M4_MAE=by["M4-fixed"][0]["MAE"],
                risk_MAE=risk[0]["MAE"],
                uniform_MAE=uniform[0]["MAE"],
                risk_minus_uniform_MAE=risk[0]["MAE"] - uniform[0]["MAE"] if comparable else None,
                risk_Y=risk[0]["Y"],
                uniform_Y=uniform[0]["Y"],
                same_risk_uniform_E_masks=same,
                all_branch_E_masks_identical=all(v[1] == risk[1] for v in by.values()),
                **{n + "_MAE": by[n][0]["MAE"] for n in RANDOM},
                random_scheduled=3,
                random_defined=sum(v is not None for v in random),
                random_mean_MAE=mean(random) if ok else None,
                random_min_MAE=min(random) if ok else None,
                random_max_MAE=max(random) if ok else None,
                risk_minus_random_mean_MAE=risk[0]["MAE"] - mean(random)
                if ok and risk[0]["MAE"] is not None and risk[1] == by[RANDOM[0]][1]
                else None,
            )
        )
    table(out, "reconstruction-views.csv", views)
    table(out, "reconstruction-conditions.csv", conditions)
    table(out, "paired.csv", paired)
    comparisons = [r["risk_minus_uniform_MAE"] for r in paired if r["risk_minus_uniform_MAE"] is not None]
    object_equal = []
    for c in CONDITIONS:
        available = [r for r in conditions if r["condition"] == c and r["E"]]
        object_equal.append(
            {
                "condition": c,
                "objects_scheduled": 6,
                "objects_defined": len(available),
                "MAE": mean(r["MAE"] for r in available) if available else None,
                "Y_over_E": mean(r["Y_over_E"] for r in available) if available else None,
            }
        )
    table(out, "reconstruction-object-equal.csv", object_equal)
    save(
        out / "nbv.json",
        {
            "schema": "el-t1-nbv-endpoints-1",
            "run_id": os.environ["EL_RUN_ID"],
            "opening": pins["opening"],
            "terminals": pins["terminals"],
            "objects": OBJECTS,
            "conditions": CONDITIONS,
            "scheduled_views": len(views),
            "comparable_views": sum(r["E"] > 0 for r in views),
            "risk_minus_uniform_object_equal": mean(comparisons) if comparisons else None,
            "paired_defined": len(comparisons),
            "risk_better": sum(v < 0 for v in comparisons),
            "exact_ties": sum(v == 0 for v in comparisons),
            "uniform_better": sum(v > 0 for v in comparisons),
            "object_mask": None,
            "risk_training": 0,
            "threshold_refits": 0,
        },
    )


def run(roots):
    store = roots[0]
    terminals = selected(store, "t1-terminals")
    t = load(locate(roots, terminals))
    pins = {"terminals": terminals, "freeze": t["freeze"], "opening": selected(store, "t1-opening")}
    pins.update({"ref-" + o: selected(store, "t1-reference-" + o) for o in OBJECTS})
    cmd = [
        "{python}",
        "-B",
        "{source}/scripts/t1_nbv_evaluate.py",
        "worker",
        "--config",
        "{config}",
        "--out",
        "{out}",
        "--store",
        str(store),
    ]
    return stage(roots, "t1-nbv-evaluation", "evaluation", pins, {}, cmd, "nbv.json", 700000)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("evaluate", "worker"))
    p.add_argument("--store", type=Path, default=DEFAULT)
    for k in ("out", "config"):
        p.add_argument("--" + k, type=Path)
    a = p.parse_args()
    roots = a.store.resolve(), a.store.resolve() / "history"
    if a.operation == "evaluate":
        print(json.dumps(run(roots)))
    else:
        endpoint(roots, load(a.config), a.out)
