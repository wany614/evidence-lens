"""Prepare the shared B pixel identifiers, deployable inputs and independently checked native labels."""

import argparse
import json
import errno
import os
import shutil
from pathlib import Path

import numpy as np

from evidence_lens.formal_inputs import require, signature, verified_file
from evidence_lens.l1 import ALL_COLUMNS, NUMERIC, feature_rows, schema_digest
from evidence_lens.l1_data import _validate_columns, sample_indices, seed_for
from evidence_lens.labels import label_view
from evidence_lens.risk_model import score_features
from evidence_lens.scoring_io import TARGETS, new_output, write_json
from vision_sources import find_run, signed


TRUTH = (
    "reference_valid",
    "rgb_evaluable",
    "rgb_high_error",
    "rgb_error",
    "reference_surface",
    "depth_evaluable",
    "depth_high_error",
    "depth_error",
    "signed_depth_error",
    "relative_depth_error",
    "signed_relative_depth_error",
    "visibility",
    "edge",
    "low_support",
    "low_alpha",
    "alpha_valid",
)


def packed(path, names):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in names}


def assetspec(path, root):
    return {"path": path.relative_to(root).as_posix(), **signature(path)}


def sample_identity(cell, view):
    return view if cell["condition"] == "B0" else cell["condition"] + "/" + view


def resolve(run, entry):
    require(
        entry["path"] in run.files
        and all(run.files[entry["path"]][k] == entry[k] for k in ("bytes", "sha256")),
        "material index differs from sealed member",
    )
    return run.open(entry["path"])


def raw_features(arrays, shape):
    _validate_columns(arrays, shape)
    require(
        arrays["scoreable"].dtype == np.bool_ and all(not np.isinf(arrays[k]).any() for k in NUMERIC),
        "illegal scoreability or infinite raw feature",
    )
    checks = score_features(arrays, True)
    for key in ("l0", "alpha_score"):
        require(np.array_equal(arrays[key], checks[key], equal_nan=True), "deployable score differs: " + key)


def verify_truth(path, ref, pred):
    truth = packed(path, TRUTH)
    computed = label_view(ref, pred, True)
    for k, a in truth.items():
        b = computed[k]
        require(
            a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True),
            f"independent source labels disagree: {k}",
        )
    return truth


def old_parity(run, history, view, idx, x, y, mask, split):
    recorded = history["views"][view]
    old = packed(
        resolve(run, {"path": f"payload/{view}.npz", **{k: recorded[k] for k in ("bytes", "sha256")}}),
        ("index", "X", "Y", "scoreable"),
    )
    if split == "train":
        require(np.array_equal(idx, old["index"]), "historical B0 sampling coordinates differ")
        choose = slice(None)
    else:
        require(np.array_equal(np.flatnonzero(mask), old["index"]), "historical development E differs")
        choose = np.searchsorted(old["index"], idx)
    require(
        np.array_equal(old["X"][choose], x, equal_nan=True)
        and np.array_equal(old["Y"][choose], y)
        and old["scoreable"][choose].all(),
        "historical feature or supervision parity failed",
    )


def emit_cell(cell, catalog_hash, roots, root):
    family, cond, split = (cell[k] for k in ("family", "condition", "split"))
    terminal = cell.get("scientific_disposition") == "terminal_upstream_failure"
    roles = ("reference",) if terminal else ("prediction", "rules", "labels", "reference")
    runs = {k: find_run(roots, cell["parents"][k]) for k in roles}
    old, history = None, None
    if "historical_extract" in cell:
        old = find_run(roots, cell["historical_extract"])
        history = signed(old.open("payload/family.json"), cell["historical_extract"]["extract_manifest_sha256"])
    result = []
    for view in TARGETS:
        entry = cell["views"][view]
        shape = entry["height"], entry["width"]
        require(shape == (512, 512), "formal target grid changed")
        directory = root / family / cond / view
        directory.mkdir(parents=True)
        truth_path = None if terminal else resolve(runs["labels"], entry["labels"])
        ref = packed(
            resolve(runs["reference"], entry["reference"]), ("rgb", "z", "position", "edge", "visibility")
        )
        pred = None
        if entry["status"] == "completed":
            pred = packed(resolve(runs["prediction"], entry["prediction"]), ("rgb", "alpha", "z"))
            require(
                pred["rgb"].shape == (*shape, 3) and pred["rgb"].dtype == np.float32,
                "invalid native predicted RGB",
            )
        truth = verify_truth(truth_path, ref, pred) if truth_path else label_view(ref, None, False)
        v, e, high = (truth[k] for k in ("reference_valid", "rgb_evaluable", "rgb_high_error"))
        require(
            v.shape == shape and int(v.sum()) == entry["V"] and int(e.sum()) == entry["E"],
            "sealed denominators differ from pixels",
        )
        full = {"deploy": None, "truth": None} if split != "train" else None
        rgb = None
        if pred is not None:
            fpath = resolve(runs["rules"], entry["features"])
            a = packed(fpath, (*ALL_COLUMNS, "scoreable", "alpha_score", "frame_relation", "l0"))
            raw_features(a, shape)
            eligible = a["scoreable"] & e
            index = sample_indices(eligible, family, sample_identity(cell, view), catalog_hash).astype(np.int32)
            x = feature_rows(a, index)
            path = directory / "rgb.npy"
            np.save(path, pred["rgb"], allow_pickle=False)
            rgb = assetspec(path, root)
            if full is not None:
                dense = np.stack([a[k] for k in ALL_COLUMNS], axis=-1).astype(np.float32, copy=False)
                path = directory / "deploy.npz"
                np.savez_compressed(
                    path, X=dense, scoreable=a["scoreable"], l0=a["l0"], alpha_score=a["alpha_score"]
                )
                full["deploy"] = assetspec(path, root)
        else:
            require(
                entry["status"] == "render_failed" and entry["reason"] and not e.any(),
                "missing render disposition",
            )
            eligible = np.zeros(shape, bool)
            index = np.empty(0, np.int32)
            x = np.empty((0, len(ALL_COLUMNS)), np.float32)
        label = high.ravel()[index]
        require(
            x.shape == (len(index), 26) and label.dtype == np.uint8 and np.isin(label, [0, 1]).all(),
            "invalid training rows",
        )
        if old is not None:
            old_parity(old, history, view, index, x, label, e, split)
        path = directory / "samples.npz"
        np.savez_compressed(path, index=index, X=x, Y=label)
        if full is not None:
            path_t = directory / "truth.npz"
            if truth_path:
                shutil.copyfile(truth_path, path_t)
            else:
                np.savez_compressed(path_t, **truth)
            full["truth"] = assetspec(path_t, root)
            if truth_path:
                require(
                    full["truth"]["sha256"] == entry["labels"]["sha256"], "original evaluator labels changed"
                )
        population = int(eligible.sum())
        result.append(
            {
                "family": family,
                "scene": cell["scene"],
                "condition": cond,
                "view": view,
                "trial_id": cell["trial_id"],
                "cv_fold": cell["cv_fold"],
                "group": cell.get("group"),
                "height": shape[0],
                "width": shape[1],
                "status": entry["status"],
                "reason": entry["reason"],
                "rgb": rgb,
                "samples": assetspec(path, root),
                "full": full,
                "V": int(v.sum()),
                "E": int(e.sum()),
                "scientific_disposition": cell.get("scientific_disposition", "selected"),
                "failure_kind": cell.get("failure_kind"),
                "truth_origin": "terminal_reference" if terminal else "sealed_evaluation",
                "positive_E": int((high == 1).sum()),
                "eligible": population,
                "sample_count": len(index),
                "sample_probability": len(index) / population if population else None,
                "sample_seed": str(seed_for(family, sample_identity(cell, view), catalog_hash)),
                "source_input_sha256": cell["input_sha256"],
                "selection_sha256": cell["selection_sha256"],
                "historical_learning_parity": old is not None,
            }
        )
    return result


def prepare(index_pins, roots, split, output, pilot_families=None, gate=None, test_catalog=None):
    """Read only the explicitly supplied material indices belonging to the requested split."""
    if split == "test":
        from vision_sources import test_authority

        require(test_authority(roots, test_catalog) == gate, "opening fingerprint differs")
    indices, cells = [], []
    for path, sha in index_pins:
        doc = signed(path, sha)
        require(doc["schema"] == "el-vision-material-index-1" and doc["split"] == split, "split mismatch")
        if split == "test":
            require(
                doc["cohort"] == "v11_new"
                and doc["catalog_sha256"] == test_catalog[1]
                and gate == doc["test_unlock_sha256"],
                "wrong or still sealed confirmation material",
            )
        indices.append({"sha256": sha, "catalog_sha256": doc["catalog_sha256"], "cohort": doc["cohort"]})
        for cell in doc["cells"]:
            require(
                cell["split"] == split and (split == "test" or cell["condition"] == "B0"),
                "split/condition leakage",
            )
            cells.append((cell, doc["catalog_sha256"]))
    ids = [(r["family"], r["condition"]) for r, _ in cells]
    require(len(ids) == len(set(ids)) and cells, "overlapping or empty material indices")
    if pilot_families:
        pilot_families = set(pilot_families)
        require(
            split == "train" and pilot_families <= {f for f, _ in ids},
            "pilot is confined to named training families",
        )
        cells = [(r, c) for r, c in cells if r["family"] in pilot_families]
    with new_output(output) as out:
        views = []
        for cell, sha in sorted(cells, key=lambda item: (item[0]["family"], item[0]["condition"])):
            views.extend(emit_cell(cell, sha, roots, out))
        result = {
            "schema": "el-vision-dataset-1",
            "split": split,
            "scope": "engineering_pilot" if pilot_families else "specified_materials",
            "columns": ALL_COLUMNS,
            "feature_schema_digest": schema_digest(),
            "label_version": "el-labels-1",
            "supervision": "E & deployable scoreable; native flattened index=row*width+column",
            "material_indices": indices,
            "test_unlock_sha256": gate,
            "sample_cap": 4096,
            "views": views,
        }
        write_json(out / "dataset.json", result)
    return result


def open_dataset(path, expected, split, test_marker_sha256=None):
    require(
        split != "test" or isinstance(test_marker_sha256, str),
        "test-opening check must precede reading its dataset",
    )
    path = Path(path)
    d = signed(path / "dataset.json", expected)
    require(
        d["schema"] == "el-vision-dataset-1"
        and d["split"] == split
        and d["columns"] == ALL_COLUMNS
        and d["feature_schema_digest"] == schema_digest()
        and d["label_version"] == "el-labels-1",
        "incompatible dataset split or feature semantics",
    )
    require(
        split != "test" or d["test_unlock_sha256"] == test_marker_sha256,
        "test dataset belongs to another opening",
    )
    keys = [(r["family"], r["condition"], r["view"]) for r in d["views"]]
    require(
        len(keys) == len(set(keys))
        and keys
        and all(
            {v for f, c, v in keys if (f, c) == (ff, cc)} == set(TARGETS)
            for ff, cc in {(f, c) for f, c, _ in keys}
        ),
        "missing planned targets or repeated dataset row",
    )
    for r in d["views"]:
        require(
            r["status"] in ("completed", "render_failed")
            and (r["rgb"] is not None) == (r["status"] == "completed")
            and (r["full"] is None) == (split == "train")
            and (r["condition"] == "B0" or split == "test"),
            "invalid status, full-domain partition or condition",
        )
    return d


def file_for(root, spec):
    return verified_file(root, spec["path"], spec)


def sample_data(root, row):
    a = packed(file_for(root, row["samples"]), ("index", "X", "Y"))
    i, x, y = a["index"], a["X"], a["Y"]
    require(
        i.ndim == 1
        and i.dtype == np.int32
        and (not i.size or (i[0] >= 0 and i[-1] < row["height"] * row["width"] and (np.diff(i) > 0).all()))
        and len(i) == row["sample_count"] == min(4096, row["eligible"])
        and x.dtype == np.float32
        and x.shape == (len(i), 26)
        and y.dtype == np.uint8
        and y.shape == i.shape
        and np.isin(y, [0, 1]).all(),
        "invalid supervised sample format",
    )
    flags = x[:, len(NUMERIC) :]
    require(
        not np.isinf(x[:, : len(NUMERIC)]).any()
        and np.isin(flags, [0, 1]).all()
        and (flags[:, :4].sum(axis=1) == 1).all()
        and np.array_equal(flags[:, 4:], ~np.isfinite(x[:, : len(NUMERIC)])),
        "illegal correspondence/missing flags",
    )
    return i, x, y


def training_table(root, d):
    names, xs, ys, fs, vs = sorted({r["family"] for r in d["views"]}), [], [], [], []
    for row in d["views"]:
        _, x, y = sample_data(root, row)
        xs.append(x)
        ys.append(y)
        fs.append(np.full(len(y), names.index(row["family"]), np.int32))
        vs.append(np.full(len(y), TARGETS.index(row["view"]), np.int32))
    return (names, *(np.concatenate(v) for v in (xs, ys, fs, vs)))


def combine(datasets, split, output):
    """Join disjoint pinned learning transports with identical original asset bytes."""
    require(split in ("train", "development"), "new confirmation data is never joined to old learning material")
    with new_output(output) as out:
        rows, pins, parents = [], {}, []
        keys = set()
        for root, sha in datasets:
            root = Path(root)
            ds = open_dataset(root, sha, split)
            parents.append(sha)
            for spec in ds["material_indices"]:
                pins[spec["sha256"]] = spec
            for row in ds["views"]:
                key = (row["family"], row["condition"], row["view"])
                require(key not in keys, "duplicate family/condition/view across input transports")
                keys.add(key)
                for item in [row["samples"], row["rgb"], *((row["full"] or {}).values())]:
                    if item is None:
                        continue
                    source = file_for(root, item)
                    dst = out / item["path"]
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        require(assetspec(dst, out) == item, "conflicting shared image or array")
                        continue
                    try:
                        os.link(source, dst)
                    except OSError as exc:
                        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES):
                            raise
                        shutil.copyfile(source, dst)
                    require(assetspec(dst, out) == item, "transport member changed")
                rows.append(row)
        d = {
            **ds,
            "scope": "specified_materials",
            "material_indices": list(pins.values()),
            "source_datasets": parents,
            "views": sorted(rows, key=lambda r: (r["family"], r["condition"], r["view"])),
        }
        write_json(out / "dataset.json", d)
    return d


def preprocess_training(root, expected, output, catalogs, pilot=False):
    from evidence_lens.l1_fit import preprocess, weights

    from vision_baselines import full_cohort

    d = open_dataset(root, expected, "train")
    if pilot:
        require(
            d["scope"] == "engineering_pilot" and len(catalogs) == 1,
            "pilot requires explicitly designated training material",
        )
        catalog = signed(*catalogs[0])
        names = {r["family"] for r in d["views"]}
        require(
            catalog["schema"] == "el-vision-family-roster-1"
            and 1 <= len(names) <= 4
            and names <= set(catalog["pilot_families"])
            and {m["catalog_sha256"] for m in d["material_indices"]} == {catalogs[0][1]},
            "pilot preprocessing is confined to predeclared new training families",
        )
    else:
        full_cohort(d, catalogs, "train")
    names, x, _, f, v = training_table(root, d)
    require(len(x) > 0, "no supervised training pixels")
    p = preprocess(x, weights(f, v), "L1-full")
    doc = {
        **p,
        "schema": "el-vision-preprocessing-1",
        "training_dataset_sha256": expected,
        "family_ids": names,
        "number_of_rows": len(x),
        "scope": d["scope"],
        "rules": "Median of observed training values. Impute then weighted-center and scale in float64. Preserve flags.",
    }
    with new_output(output) as out:
        write_json(out / "preprocessing.json", doc)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="command", required=True)
    p = cmds.add_parser("prepare")
    p.add_argument("--index", nargs=2, action="append", required=True, metavar=("MATERIALS_JSON", "SHA256"))
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--shelf", type=Path, required=True)
    p.add_argument("--split", choices=("train", "development", "test"), required=True)
    p.add_argument("--pilot-family", action="append")
    p.add_argument("--test-gate-sha256")
    p.add_argument("--test-catalog", nargs=2)
    p = cmds.add_parser("combine")
    p.add_argument("--dataset", nargs=2, action="append", required=True, metavar=("ROOT", "DATASET_SHA256"))
    p.add_argument("--split", choices=("train", "development"), required=True)
    p = cmds.add_parser("preprocess")
    p.add_argument("--catalog", nargs=2, action="append", required=True, metavar=("CATALOG", "SHA256"))
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    for sub in cmds.choices.values():
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(
            args.index,
            [args.store.resolve(), args.shelf.resolve()],
            args.split,
            args.output,
            args.pilot_family,
            args.test_gate_sha256,
            args.test_catalog,
        )
        name = "dataset.json"
    elif args.command == "combine":
        combine(args.dataset, args.split, args.output)
        name = "dataset.json"
    else:
        preprocess_training(args.dataset, args.sha256, args.output, args.catalog, args.pilot)
        name = "preprocessing.json"
    print(json.dumps({"file": name, **signature(args.output / name)}))


if __name__ == "__main__":
    main()
