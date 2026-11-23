"""Build a portable, signed Stage C dataset from sealed native runs. Keep reference channels outside deployment."""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from evidence_lens.formal_inputs import require, verified_file
from evidence_lens.l1_data import sample_indices, seed_for
from evidence_lens.labels import label_view
from evidence_lens.multisource import (
    COLUMNS,
    CONDITIONS,
    SCOPES,
    SOURCES,
    TARGETS,
    extract,
    schema_sha256,
    source_ids,
)
from evidence_lens.multisource_model import load_transfer, transfer_risk, validate_rows
from evidence_lens.scoring_io import check_rgb, depth_valid, file_hash, new_output, write_json
from stage3_risk_sources import (
    CATALOG,
    STUDY,
    LEARNING,
    TRANSFER,
    asset,
    authorities,
    case_sources,
    pin_selections,
)
from vision_sources import find_run, signed


def union_visibility(ref, ids):
    c = np.stack([ref["visibility_" + s] for s in ids])
    return np.select([(c == 1).any(0), (c == 2).any(0), (c == 3).any(0)], [1, 2, 3], default=0).astype(
        np.uint8
    )


def reference_arrays(path, cam):
    with np.load(path, allow_pickle=False) as a:
        names = (
            "rgb",
            "z",
            "position",
            "object_id",
            "reference_surface",
            "edge",
            *("visibility_" + s for s in SOURCES),
        )
        r = {k: a[k] for k in names}
    check_rgb(r["rgb"], cam, "reference")
    require(
        r["z"].dtype == np.float32
        and r["z"].shape == cam.shape
        and r["position"].dtype == np.float32
        and r["position"].shape == (*cam.shape, 3)
        and r["object_id"].shape == cam.shape
        and r["object_id"].dtype.kind in "iu",
        "invalid native reference arrays",
    )
    require(
        r["reference_surface"].dtype == bool
        and np.array_equal(r["reference_surface"], depth_valid(r["z"]))
        and np.array_equal(r["reference_surface"], np.isfinite(r["position"]).all(-1)),
        "inconsistent reference surface",
    )
    for s in SOURCES:
        x = r["visibility_" + s]
        require(
            x.dtype == np.uint8
            and x.shape == cam.shape
            and np.all(x <= 3)
            and not ((x != 0) & ~r["reference_surface"]).any(),
            "invalid real source visibility",
        )
    require(
        r["edge"].shape == cam.shape and np.all((r["edge"] == 0) | (r["edge"] == 1)), "invalid reference edge"
    )
    r["edge"] = r["edge"].astype(bool)
    return r


def save(root, path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npy":
        np.save(path, arrays, allow_pickle=False)
    else:
        np.savez_compressed(path, **arrays)
    return asset(root, path)


def pack_view(out, cell, view, case, original, cam, transfer, split, group, fold):
    f, c = cell["family"], cell["condition"]
    folder = out / "views" / f / c / view
    r = reference_arrays(original[0], cam)
    r["visibility"] = union_visibility(r, CONDITIONS[c])
    pred = case.frame(view) if case is not None else None
    labels = label_view(r, pred, case is not None and case.pred["depth_verified"])
    labels.update({"visibility_" + s: r["visibility_" + s] for s in SOURCES})
    row = {
        "family": f,
        "condition": c,
        "view": view,
        "height": cam.height,
        "width": cam.width,
        "group": group,
        "cv_fold": fold,
        "case": cell.get("case"),
        "reference_member_sha256": original[1]["sha256"],
        "selection_sha256": cell["selection_sha256"],
        "status": "ready" if pred is not None else "render_failed",
        "failure": None,
        "modes": {},
        "truth": save(out, folder / "truth.npz", labels),
        "V": int(labels["reference_valid"].sum()),
        "E": int(labels["rgb_evaluable"].sum()),
        "Y": int((labels["rgb_high_error"] == 1).sum()),
        "rgb": None,
        "transfer": None,
        "selection_support": None,
        "reference_rgb": None,
        "prediction_member_sha256": None,
    }
    # The independent evaluation and six-view boards can use these lossless native reference pixels.
    if split != "train":
        row["reference_rgb"] = save(out, folder / "reference-rgb.npy", r["rgb"])
    if pred is None:
        row["failure"] = cell["failure"] if case is None else {"stage": "render", "reason": case.failed[view]}
        for mode in SCOPES:
            row["modes"][mode] = {
                "source_ids": list(source_ids(c, mode)),
                "alias_of": "two" if (c, mode) == ("M2", "all") else None,
                "samples": None,
                "deploy": None,
                "eligible": 0,
                "scoreable_pixels": 0,
            }
        return row
    row["prediction_member_sha256"] = case.meta["renders"][view]["sha256"]
    row["rgb"] = save(out, folder / "prediction-rgb.npy", pred["rgb"])
    sources = case.sources("all")
    b = transfer_risk(pred, cam, sources["s00"], transfer, case.pred["depth_verified"])
    row["transfer"] = save(out, folder / "B-L1-transfer.npy", b)
    # Scopes share this actual target and source self-renders. Each independent reduction only sees its subset.
    for mode in SCOPES:
        if (c, mode) == ("M2", "all"):
            row["modes"][mode] = {**row["modes"]["two"], "alias_of": "two"}
            continue
        ids = source_ids(c, mode)
        x = extract(pred, cam, {s: sources[s] for s in ids}, depth_verified=case.pred["depth_verified"])
        validate_rows(x["X"])
        eligible = labels["rgb_evaluable"] & x["scoreable"]
        use = ids == CONDITIONS[c]
        spec = {
            "source_ids": list(ids),
            "alias_of": None,
            "samples": None,
            "deploy": None,
            "eligible": int(eligible.sum()),
            "scoreable_pixels": int(x["scoreable"].sum()),
        }
        if split != "train":
            spec["deploy"] = save(out, folder / (mode + "-deployment.npz"), x)
        if split == "train" and use:
            chosen = sample_indices(eligible, f, c + "/" + view, CATALOG).astype(np.int32)
            prob = len(chosen) / spec["eligible"] if spec["eligible"] else 0.0
            spec["samples"] = save(
                out,
                folder / "training.npz",
                {
                    "index": chosen,
                    "X": x["X"].reshape(-1, len(COLUMNS))[chosen],
                    "Y": labels["rgb_high_error"].ravel()[chosen],
                    "inclusion_probability": np.array(prob, np.float64),
                },
            )
            spec.update(sample_count=len(chosen), sampling_seed=str(seed_for(f, c + "/" + view, CATALOG)))
        if use and split != "train":
            row["selection_support"] = save(out, folder / "selection-support.npy", eligible & np.isfinite(b))
        row["modes"][mode] = spec
    return row


def build(roots, catalog, study, learning, request, request_hash, out, run_id):
    req = signed(request, request_hash)
    require(
        req["schema"] == "el-stage3-risk-source-index-1"
        and req["catalog_sha256"] == CATALOG
        and req["study_sha256"] == STUDY
        and req["learning_sha256"] == LEARNING,
        "invalid Stage C data request",
    )
    split, cohort = req["split"], req["cohort"]
    roster, protocol, rows, marker = authorities(
        catalog, study, learning, split, cohort, roots, req.get("opening_sha256")
    )
    require(req["selections"] == pin_selections(roots, rows, cohort), "selection roster changed")
    tr = find_run(roots, TRANSFER)
    require(tr.run["execution_status"] == "succeeded", "Stage B transfer parent unavailable")
    idx = tr.open("payload/model/model.json")
    for n in json.loads(idx.read_bytes())["files"]:
        tr.open("payload/model/" + n)
    b = protocol["stage_b"]
    transfer = load_transfer(
        idx.parent, b["transfer_model_index_sha256"], b["transfer_weights_sha256"], b["transfer_threshold"]
    )
    ordered = sorted(r["scene_family_id"] for r in roster["families"] if r["split"] == "train")
    cells = []
    views = []
    by = {r["scene_family_id"]: r for r in rows}
    with new_output(out) as dst:
        shutil.copyfile(catalog, dst / "catalog.json")
        for selection in req["selections"]:
            f, c = selection["family"], selection["condition"]
            cd = dst / "cases" / f / c
            case, truth, cams, cell = case_sources(roots, selection, by[f], cd)
            require(
                all(cam.shape == (256, 256) for cam in cams.values()), "unexpected formal native resolution"
            )
            cell["case"] = asset(dst, cd / "case.json") if case else None
            cells.append(cell)
            fold = ordered.index(f) % 4 if split == "train" else None
            for v in TARGETS:
                views.append(pack_view(dst, cell, v, case, truth[v], cams[v], transfer, split, split, fold))
        for f in by:
            pair = [q for q in cells if q["family"] == f]
            require(
                len(pair) == 2
                and pair[0]["selected"]["reference"] == pair[1]["selected"]["reference"]
                and pair[0]["cameras_sha256"] == pair[1]["cameras_sha256"]
                and all(
                    pair[0]["source_rgb_sha256"][s] == pair[1]["source_rgb_sha256"][s]
                    for s in CONDITIONS["M2"]
                ),
                "M2/M4 differ in original world, anchors or calibrated camera",
            )
            if all(q["failure"] is None for q in pair):
                require(
                    pair[0]["selected"]["render"] != pair[1]["selected"]["render"]
                    and pair[0]["selected"]["reconstruction"] != pair[1]["selected"]["reconstruction"]
                    and pair[0]["initial_parameter_sha256"] == pair[1]["initial_parameter_sha256"],
                    "two reconstruction conditions reuse a single run",
                )
        doc = {
            "schema": "el-stage3-risk-dataset-1",
            "run_id": run_id,
            "split": split,
            "cohort": cohort,
            "catalog_sha256": CATALOG,
            "study_sha256": STUDY,
            "learning_sha256": LEARNING,
            "feature_schema_sha256": schema_sha256(),
            "columns": list(COLUMNS),
            "families": list(by),
            "source_request_sha256": request_hash,
            "opening_sha256": req.get("opening_sha256"),
            "method_sha256": (marker or {}).get("method_sha256"),
            "transfer": {
                "run": TRANSFER,
                "model_index_sha256": b["transfer_model_index_sha256"],
                "weights_sha256": b["transfer_weights_sha256"],
                "threshold": b["transfer_threshold"],
            },
            "cells": cells,
            "views": views,
        }
        write_json(dst / "dataset.json", doc)
    return doc


def dataset(root, expected, split, *, roots=None, opening=None):
    root = Path(root)
    if split == "test":
        require(roots and opening, "test authority required before reading a test dataset")
        from evidence_lens.stage3_gate import require_opening

        marker = require_opening(roots[0], CATALOG, opening_sha256=opening, shelf=roots[1])
    d = signed(root / "dataset.json", expected)
    require(
        d["schema"] == "el-stage3-risk-dataset-1"
        and d["catalog_sha256"] == CATALOG
        and d["study_sha256"] == STUDY
        and d["learning_sha256"] == LEARNING
        and d["split"] == split
        and d["feature_schema_sha256"] == schema_sha256()
        and d["columns"] == list(COLUMNS),
        "risk dataset authority/schema mismatch",
    )
    cat = signed(root / "catalog.json", CATALOG)
    wanted = [
        r["scene_family_id"]
        for r in cat["families"]
        if r["split"] == split and (d["cohort"] != "pilot" or r["pilot"])
    ]
    require(
        d["cohort"] in ("pilot", "formal")
        and (d["cohort"] != "pilot" or split == "train")
        and d["families"] == wanted,
        "dataset names differ from the exact registered split",
    )
    if split == "test":
        require(
            d["method_sha256"] == marker["method_sha256"] and d["opening_sha256"] == opening,
            "wrong held-out method/opening",
        )
    require(
        len(set(d["families"])) == len(d["families"])
        and [(r["family"], r["condition"], r["view"]) for r in d["views"]]
        == [(f, c, v) for f in d["families"] for c in CONDITIONS for v in TARGETS],
        "duplicate or incomplete target roster",
    )
    for r in d["views"]:
        base = f"views/{r['family']}/{r['condition']}/{r['view']}/"
        require(
            r["status"] in ("ready", "render_failed")
            and r["truth"]["path"] == base + "truth.npz"
            and all(
                r.get(k) is None or r[k]["path"] == base + n
                for k, n in (
                    ("rgb", "prediction-rgb.npy"),
                    ("transfer", "B-L1-transfer.npy"),
                    ("reference_rgb", "reference-rgb.npy"),
                    ("selection_support", "selection-support.npy"),
                )
            ),
            "target row uses another instrument, family or view",
        )
        for mode, v in r["modes"].items():
            foldermode = v["alias_of"] or mode
            require(
                (v["deploy"] is None or v["deploy"]["path"] == base + foldermode + "-deployment.npz")
                and (v["samples"] is None or v["samples"]["path"] == base + "training.npz"),
                "source subset references a different original target",
            )
        require(
            set(r["modes"]) == set(SCOPES)
            and all(r["modes"][s]["source_ids"] == list(source_ids(r["condition"], s)) for s in SCOPES),
            "source-subset bytes disagree",
        )
        if r["condition"] == "M2":
            require(
                r["modes"]["all"] == {**r["modes"]["two"], "alias_of": "two"}, "M2 two/all fail byte identity"
            )
    return d


def read_asset(root, entry):
    p = verified_file(root, entry["path"], entry)
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=False)
    with np.load(p, allow_pickle=False) as z:
        return {s: z[s] for s in z.files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("store", "shelf", "catalog", "study", "learning", "request", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--request-sha256", required=True)
    a = p.parse_args()
    result = build(
        (a.store, a.shelf),
        a.catalog,
        a.study,
        a.learning,
        a.request,
        a.request_sha256,
        a.output,
        os.environ["EL_RUN_ID"],
    )
    print(
        json.dumps({"split": result["split"], "cells": len(result["cells"]), "views": len(result["views"])})
    )


if __name__ == "__main__":
    main()
