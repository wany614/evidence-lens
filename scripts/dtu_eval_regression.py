"""Independently compare the authorized D1 score grids to the published 057 CPU outputs."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from dtu_score import METHODS, SCOPES, VIEWS, RELEASE_SHA
from dtu_eval_metrics import labels
from evidence_lens.formal_inputs import require, verified_file
from evidence_lens.scoring_io import file_hash, new_output, write_json


def original(root, spec):
    return verified_file(root, spec["path"], spec)


def check(assets, scores, destination, rid):
    reg = json.loads((assets / "regression.json").read_bytes())
    require(
        reg["schema"] == "el-dtu-057-regression-roster-1" and reg["release_sha256"] == RELEASE_SHA,
        "wrong regression material",
    )
    ds = {c: json.loads(p.read_bytes()) for c, p in scores.items()}
    rows = []
    rgb = []
    for c in ("M2", "M4"):
        pins = reg["cases"][c]
        root = scores[c].parent
        sc = ds[c]
        require(
            sc["case_sha256"] == pins["case_sha256"] and sc["release_sha256"] == RELEASE_SHA,
            "wrong frozen scores",
        )
        case = assets / "cases" / c
        case_file = case / "case.json"
        cm = json.loads(case_file.read_bytes())
        require(file_hash(case_file) == sc["case_sha256"], "case identity changed")
        for v in VIEWS:
            ref = pins["views"][v]
            with np.load(original(assets, ref["original_fp32"]), allow_pickle=False) as z:
                fp = z["rgb"]
            with Image.open(original(assets, ref["rgb_png"])) as im:
                png = np.asarray(im, dtype=np.float32) / np.float32(255)
            pp = original(case, cm["renders"][v])
            with np.load(pp, allow_pickle=False) as z:
                pred = z["rgb"]
            mask = np.ones(fp.shape[:2], bool)
            vv, e, y, mae, _ = labels(fp, pred, mask)
            _, e8, y8, mae8, _ = labels(png, pred, mask)
            with np.load(original(assets, ref["measurements"]), allow_pickle=False) as ex:
                require(
                    all(
                        np.array_equal(ex[k], mat, equal_nan=True)
                        for k, mat in zip(("V", "E", "Y", "rgb_mae"), (vv, e, y, mae))
                    ),
                    "original independent FP32 RGB labels differ",
                )
            require(np.array_equal(e, e8), "changing only reference encoding changed valid support")
            rgb.append(
                {
                    "condition": c,
                    "view": v,
                    "V": int(vv.sum()),
                    "E": int(e.sum()),
                    "original_fp32_Y": int((y == 1).sum()),
                    "png_Y": int((y8 == 1).sum()),
                    "changed_labels": int(np.count_nonzero(y != y8)),
                    "max_reference_quantization_abs": float(np.max(np.abs(fp - png))),
                    "rgb8_mae_sum": float(mae8[e8].sum(dtype=np.float64)),
                    "fp32_mae_sum": float(mae[e].sum(dtype=np.float64)),
                }
            )
            for m in METHODS:
                spec = sc["views"][v]["methods"][m]
                g = ref["scores"][m]
                require(
                    spec["scope"] == g["scope"] == SCOPES[m]
                    and spec["target_prediction_sha256"] == file_hash(pp),
                    "wrong published method scope or target",
                )
                with (
                    np.load(original(root, spec["file"]), allow_pickle=False) as aa,
                    np.load(original(assets, g["file"]), allow_pickle=False) as gg,
                ):
                    a, b = aa["risk"], gg["risk"]
                    valid = aa["valid"]
                    other = gg["valid"]
                    require(
                        np.array_equal(valid, other) and np.array_equal(aa["state"], gg["state"]),
                        "CPU support or operating state differs from committed Mac example",
                    )
                    gap = np.abs(a[valid].astype(np.float64) - b[valid])
                    maximum = float(gap.max()) if len(gap) else 0.0
                    require(maximum <= 1e-4, "published bounded risk tolerance exceeded")
                    rows.append(
                        {
                            "condition": c,
                            "view": v,
                            "method": m,
                            "scope": SCOPES[m],
                            "valid": int(valid.sum()),
                            "unknown": int((~valid).sum()),
                            "max_abs": maximum,
                            "threshold_crossings": int(np.count_nonzero(aa["state"] != gg["state"])),
                            "score_bitwise_equal": a.tobytes() == b.tobytes(),
                            "measured_sha256": spec["file"]["sha256"],
                            "published_sha256": g["file"]["sha256"],
                        }
                    )
    doc = {
        "schema": "el-dtu-057-regression-1",
        "run_id": rid,
        "release_sha256": RELEASE_SHA,
        "passed": True,
        "scores": {c: file_hash(x) for c, x in scores.items()},
        "authorized_calls": len(rows),
        "finite_score_values": sum(x["valid"] for x in rows),
        "max_risk_abs": max(x["max_abs"] for x in rows),
        "threshold_crossings": sum(x["threshold_crossings"] for x in rows),
        "support_and_state_equal": True,
        "all_risks_bitwise_equal": all(x["score_bitwise_equal"] for x in rows),
        "fp32_label_views_exact": len(rgb),
        "rgb8_reference_changed_labels": sum(x["changed_labels"] for x in rgb),
        "views": rows,
        "reference_encoding": rgb,
    }
    with new_output(destination) as out:
        write_json(out / "regression.json", doc)
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("assets", "scores-M2", "scores-M4", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    j = check(
        a.assets, {c: getattr(a, "scores_" + c) for c in ("M2", "M4")}, a.output, os.environ.get("EL_RUN_ID")
    )
    print(
        json.dumps(
            {
                k: j[k]
                for k in (
                    "passed",
                    "authorized_calls",
                    "max_risk_abs",
                    "threshold_crossings",
                    "rgb8_reference_changed_labels",
                )
            }
        )
    )
