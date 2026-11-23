"""Independently read original RGBs, evidence masks and all seven native scores for a frozen cohort."""

import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score

from evidence_lens._bundle import load, sha
from evidence_lens.formal_inputs import PinnedRun, compact_pin, require
from evidence_lens.scoring_io import write_json

KEYS = {
    "constant": "constant",
    "1-alpha": "alpha_score",
    "L0": "l0",
    "L1-full": "l1_full",
    "L1-geometry": "l1_geometry",
}
METHODS = (*KEYS, "L2-frozen", "L2-vision-ft")
GROUPS = ("own", "five_common", "seven_common")
SOURCE = (
    "source_input_sha256",
    "source_camera_sha256",
    "source_prediction_sha256",
    "target_prediction_sha256",
    "render_pin",
)


def check(root, shelf, request):
    def run(pin):
        return PinnedRun(
            next(r for r in (root, shelf) if (r / "runs" / pin["run_id"]).is_dir()), compact_pin(pin)
        )

    def checked(original, rel, spec):
        if isinstance(spec, str):
            spec = {"sha256": spec}
        path = original.open(rel)
        entry = original.files[rel]
        require(
            all(entry[k] == spec[k] for k in ("bytes", "sha256") if k in spec), "consumer and recorder differ"
        )
        return path

    def document(original, rel, spec):
        return load(checked(original, rel, spec))

    def good(a):
        return np.logical_and(np.logical_and(a >= 0, a <= 1), np.isfinite(a)).all(axis=2)

    def identity(r):
        return (r["family"], r["condition"], r["view"])

    def number(a):
        return int(np.count_nonzero(a))

    stage = {}
    receipts = {}
    for role, pin in request["stages"].items():
        record = run(pin)
        require(record.run["execution_status"] == "succeeded", "unsuccessful QA source")
        stage[role] = record
        receipts[role] = document(record, "payload/" + pin["receipt"], pin["receipt_sha256"])
    ds, materials, evaluation = (receipts[k] for k in ("dataset", "materials", "evaluation"))
    method = document(run(request["method"]), "payload/method-freeze.json", request["method_sha256"])
    marker = root / "protocol/stage2" / request["catalog_sha256"] / "test-opening.json"
    require(
        sha(marker) == request["opening_sha256"]
        and load(marker)["method_sha256"] == request["method_sha256"],
        "another opening",
    )
    require(
        evaluation["dataset_sha256"] == request["stages"]["dataset"]["receipt_sha256"]
        and evaluation["test_opening_sha256"] == request["opening_sha256"]
        and compact_pin(request["method"])
        in [compact_pin(a) for a in load(stage["evaluation"].open("inputs.json"))["parents"]],
        "evaluation belongs to another dataset, opening or direct frozen parent",
    )
    if request["cohort"] == "new_confirmation":
        require(
            evaluation["schema"] == "el-vision-comparison-1"
            and evaluation["split"] == "test"
            and evaluation["thresholds_sha256"] == method["thresholds_sha256"],
            "formal evaluation refers to different frozen numerical thresholds",
        )
    else:
        require(
            evaluation["schema"] == "el-vision-historical-regression-1"
            and evaluation["method_sha256"] == request["method_sha256"],
            "historical evaluation method changed",
        )
    if request["cohort"] == "new_confirmation":
        require(ds["test_unlock_sha256"] == request["opening_sha256"], "new data came from another opening")
    tau = {m: (z["tau"] if isinstance(z, dict) else z) for m, z in method["thresholds"].items()}
    require(
        tau == evaluation["thresholds"] and tuple(evaluation["methods"]) == METHODS,
        "threshold or method drift",
    )
    cells = {(r["family"], r["condition"]): r for r in materials["cells"]}
    scored = {}
    for label, names in (
        ("score-original-five", tuple(KEYS)),
        ("score-L2-frozen", (METHODS[-2],)),
        ("score-L2-vision-ft", (METHODS[-1],)),
    ):
        doc = receipts[label]
        require(
            doc["dataset_sha256"] == evaluation["dataset_sha256"]
            and doc["method_freeze_sha256"] == request["method_sha256"],
            "scores used another dataset or method",
        )
        require(
            doc["model_sha256"] == evaluation["inputs"][doc["method"]]["model_sha256"]
            and request["stages"][label]["receipt_sha256"] == evaluation["inputs"][doc["method"]]["sha256"],
            "wrong score receipt",
        )
        scored[label] = ({identity(r): r for r in doc["views"]}, names)
    audit, ap_cases, worst = [], Counter(), 0.0
    score_states, populations = Counter(), Counter()
    for data in ds["views"]:
        f, c, v = identity(data)
        cell = cells[f, c]
        view = cell["views"][v]
        ref = run(cell["parents"]["reference"])
        spec = view["reference"]
        with np.load(checked(ref, spec["path"], spec), allow_pickle=False) as x:
            truth = x["rgb"]
        V = good(truth)
        E = np.zeros_like(V)
        Y = np.zeros_like(V)
        if view.get("prediction"):
            pred = run(cell["parents"]["prediction"])
            spec = view["prediction"]
            with np.load(checked(pred, spec["path"], spec), allow_pickle=False) as x:
                rgb = x["rgb"]
            E = V & good(rgb)
            # Frozen label is a strict display-RGB channel mean in binary32.
            difference = np.abs(truth[E] - rgb[E])
            Y[E] = (difference[:, 0] + difference[:, 1] + difference[:, 2]) / np.float32(3) > np.float32(0.05)
        spec = data["full"]["truth"]
        with np.load(checked(stage["dataset"], "payload/" + spec["path"], spec), allow_pickle=False) as x:
            for observed, name in ((V, "reference_valid"), (E, "rgb_evaluable"), (Y, "rgb_high_error")):
                expected = x[name]
                if name == "rgb_high_error":
                    expected = (expected == 1) & E
                require(
                    np.array_equal(observed, expected),
                    "original reference/prediction RGB disagrees with recorded labels",
                )
        risk = {}
        for label, (rows, names) in scored.items():
            r = rows[f, c, v]
            require(all(r.get(k) == data.get(k) for k in SOURCE), "wrong condition-associated native score")
            spec = r["score"]
            if spec is None:
                require(
                    r["status"] in ("render_failed", "unsupported") and r["reason"],
                    "missing score without reason",
                )
                risk.update({m: np.full(V.shape, np.nan, np.float32) for m in names})
            else:
                with np.load(checked(stage[label], "payload/" + spec["path"], spec), allow_pickle=False) as x:
                    for m in names:
                        risk[m] = x[KEYS.get(m, "risk")]
        masks = {m: E & np.isfinite(raw) for m, raw in risk.items()}
        c5 = np.logical_and.reduce([masks[m] for m in KEYS])
        c7 = np.logical_and.reduce([masks[m] for m in METHODS])
        vals = {
            "V": number(V),
            "E": number(E),
            "Y": number(Y),
            "S5": number(c5),
            "S5Y": number(c5 & Y),
            "S7": number(c7),
            "S7Y": number(c7 & Y),
        }
        populations.update(vals)
        name = f"{f}/{c}/{v}.json"
        measured = document(stage["evaluation"], "payload/" + name, evaluation["files"][name])["region"][
            "all"
        ]
        row = {"family": f, "condition": c, "view": v, **vals, "ranked_domain_checks": 0}
        for m in METHODS:
            p = risk[m]
            own = masks[m]
            lo = own & (p.astype(np.float64) < tau[m])
            hi = own & ~lo
            u = E & ~own
            own_counts = {
                "A": number(own),
                "L": number(lo),
                "H": number(hi),
                "U": number(u),
                "LY": number(Y & lo),
                "HY": number(Y & hi),
                "UY": number(Y & u),
            }
            exp = measured[m]["counts"]
            require(
                all(exp[k] == n for k, n in (vals | own_counts).items()),
                "native operating numerator mismatch",
            )
            for k, n in own_counts.items():
                score_states[m + "." + k] += n
            cache = {}
            for domain, mask in zip(GROUPS, (own, c5, c7)):
                potential = measured[m]["ranking"][domain]
                if domain == "five_common" and m not in KEYS:
                    require(potential is None, "neural method in baseline-only domain")
                    continue
                ys = Y[mask]
                n = ys.size
                pos = number(ys)
                require(potential["n"] == n and potential["positive"] == pos, "AP population differs")
                mask_bytes = np.packbits(mask).tobytes()
                if mask_bytes not in cache:
                    cache[mask_bytes] = float(average_precision_score(ys, p[mask])) if 0 < pos < n else None
                fresh = cache[mask_bytes]
                old = potential["ap"]
                if fresh is None:
                    require(old is None, "undefined AP assigned a value")
                    ap_cases["undefined"] += 1
                else:
                    diff = abs(fresh - old)
                    require(diff <= 2e-14, "sklearn grouped AP disagrees")
                    worst = max(worst, diff)
                    ap_cases["defined"] += 1
                row["ranked_domain_checks"] += 1
        audit.append(row)
    expected = 8 if request["cohort"] == "new_confirmation" else 4
    require(len(cells) == 3 * expected and len(audit) == 18 * expected, "wrong cohort roster")
    return {
        "schema": "el-vision-independent-native-confirmation-1",
        "run_id": os.environ["EL_RUN_ID"],
        "cohort": request["cohort"],
        "method_sha256": request["method_sha256"],
        "opening_sha256": request["opening_sha256"],
        "stage_pins": request["stages"],
        "sklearn_version": sklearn.__version__,
        "label_expression": "native FP32 mean absolute RGB difference > .05",
        "families": expected,
        "family_conditions": len(cells),
        "views": len(audit),
        "ap_checks": dict(ap_cases),
        "largest_abs_difference_from_recorded_ap": worst,
        "original_rgb_denominators": dict(populations),
        "native_method_counts": dict(score_states),
        "views_checked": audit,
    }


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    for k in ("root", "shelf", "request", "out"):
        a.add_argument("--" + k, type=Path, required=True)
    arg = a.parse_args()
    write_json(arg.out / "native-readback.json", check(arg.root, arg.shelf, load(arg.request)))
