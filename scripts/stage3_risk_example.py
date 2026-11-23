"""Create publishable original-world numerical expectations for the preregistered first C development family."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from evidence_lens.formal_inputs import require, verified_file
from evidence_lens.multisource import CONDITIONS, METHODS, SCOPES, SOURCES, TARGETS
from evidence_lens.scoring_io import Camera, new_output, write_json
from stage3_risk_boards import save_case
from stage3_risk_data import dataset, read_asset, reference_arrays, union_visibility
from stage3_risk_evaluate import score_indices, view_scores
from stage3_risk_sources import CATALOG, STUDY, asset, link_asset
from vision_sources import find_run, member, signed
from evidence_lens.multisource_io import Case


def export(root, ds_sha, roster, roots, output, rid):
    d = dataset(root, ds_sha, "development")
    f = d["families"][0]
    require(f.startswith("scene_057_"), "wrong registered public original-world example")
    sets = score_indices(roster, d, ds_sha)
    tau = {m: j["threshold"] for m, (_, j) in sets.items()}
    cell = {c["condition"]: c for c in d["cells"] if c["family"] == f}
    require(
        cell["M2"]["selected"]["reference"] == cell["M4"]["selected"]["reference"],
        "different public-example worlds",
    )
    ref = find_run(roots, cell["M2"]["selected"]["reference"])
    idx = {m: [] for m in METHODS}
    expected = []
    with new_output(output) as out:
        for i, row in enumerate(d["views"]):
            if row["family"] != f:
                continue
            c, v = row["condition"], row["view"]
            t = read_asset(root, row["truth"])
            require(
                row["status"] == "ready" and row["case"],
                "registered example lacks a target, preserve failed disposition in scientific report",
            )
            case = Case(verified_file(root, row["case"]["path"], row["case"]).parent, row["case"]["sha256"])
            path, rs = member(ref, "payload/reference/" + v + ".npz", row["reference_member_sha256"])
            original = reference_arrays(path, case.cameras[v])
            prediction = case.frame(v)
            V = np.isfinite(original["rgb"]).all(-1)
            E = V & np.isfinite(prediction["rgb"]).all(-1)
            err = np.full(V.shape, np.nan, np.float32)
            err[E] = np.mean(np.abs(prediction["rgb"][E] - original["rgb"][E]), axis=-1, dtype=np.float32)
            y = np.full(V.shape, 255, np.uint8)
            y[E] = err[E] > np.float32(0.05)
            for name, value in (
                ("reference_valid", V),
                ("rgb_evaluable", E),
                ("rgb_high_error", y),
                ("rgb_error", err),
            ):
                require(
                    np.array_equal(value, t[name], equal_nan=True),
                    "independently recomputed RGB labels differ: " + name,
                )
            h = out / "expected-errors" / c / (v + ".npz")
            h.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                h,
                V=V,
                E=E,
                Y=y,
                rgb_mae=err,
                edge=t["edge"],
                depth_evaluable=t["depth_evaluable"],
                depth_abs_error_m=t["depth_error"],
                depth_high_error=t["depth_high_error"],
                **{"visibility_" + s: original["visibility_" + s] for s in SOURCES},
            )
            pics = {}
            entries = {}
            for scope in SCOPES:
                alias = row["modes"][scope]["alias_of"]
                if alias:
                    pics[scope] = pics[alias]
                else:
                    x = read_asset(root, row["modes"][scope]["deploy"])
                    pics[scope] = view_scores(root, row, sets, i, scope, t, x)
                for m, (score_root, j) in sets.items():
                    a = j["views"][i]["modes"][scope]
                    key = (m, scope)
                    if m == "B-L1-transfer" and scope != "one":
                        written = entries[m, "one"]
                    elif a["alias_of"]:
                        written = entries[m, a["alias_of"]]
                    else:
                        file = verified_file(score_root, a["file"]["path"], a["file"])
                        dest = out / "expected-scores" / m / c / v / (scope + ".npz")
                        link_asset(file, dest)
                        written = asset(out, dest)
                    entries[key] = written
            for m in METHODS:
                idx[m].append(
                    {
                        "family": f,
                        "condition": c,
                        "view": v,
                        "case_sha256": case.sha256,
                        "status": row["status"],
                        "modes": {
                            s: {
                                **{k: row["modes"][s][k] for k in ("source_ids", "alias_of")},
                                "file": entries[m, s],
                            }
                            for s in SCOPES
                        },
                    }
                )
            board = save_case(root, out, row, t, pics, tau)
            expected.append(
                {
                    "condition": c,
                    "view": v,
                    "case_sha256": case.sha256,
                    "source_selection_sha256": row["selection_sha256"],
                    "reference_sha256": rs["sha256"],
                    "prediction_sha256": row["prediction_member_sha256"],
                    "V": int(V.sum()),
                    "E": int(E.sum()),
                    "Y": int((y == 1).sum()),
                    "error": asset(out, h),
                    "board": board["file"],
                }
            )
        files = {}
        for m in METHODS:
            doc = {
                "schema": "el-stage3-public-expected-scores-1",
                "family": f,
                "method": m,
                "catalog_sha256": CATALOG,
                "source_scores_sha256": roster[m]["sha256"],
                **{
                    k: roster[m][k] for k in ("model_id", "model_index_sha256", "weights_sha256", "threshold")
                },
                "views": idx[m],
            }
            p = out / (m + "-scores.json")
            write_json(p, doc)
            files[m] = asset(out, p)
        result = {
            "schema": "el-stage3-public-expected-example-1",
            "run_id": rid,
            "catalog_sha256": CATALOG,
            "study_sha256": STUDY,
            "development_dataset_sha256": ds_sha,
            "family": f,
            "conditions": list(CONDITIONS),
            "targets": list(TARGETS),
            "primary_view": "v01",
            "cases": expected,
            "scores": files,
            "reference": cell["M2"]["selected"]["reference"],
            "case_status": "all scheduled native original-world outputs included",
            "native_prediction": "same reconstruction across scorer source subsets",
        }
        write_json(out / "expected-example.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("dataset", "roster", "store", "shelf", "output"):
        p.add_argument("--" + k, required=True, type=Path)
    for k in ("dataset", "roster"):
        p.add_argument("--" + k + "-sha256", required=True)
    a = p.parse_args()
    q = export(
        a.dataset,
        a.dataset_sha256,
        signed(a.roster, a.roster_sha256),
        (a.store, a.shelf),
        a.output,
        os.environ["EL_RUN_ID"],
    )
    print(json.dumps({"family": q["family"], "native_target_records": len(q["cases"])}))
