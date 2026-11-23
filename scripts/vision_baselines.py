"""Refit the two grouped LR baselines or emit full-native five-method scores from a pinned fit."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evidence_lens.formal_inputs import hash_json, require, signature
from evidence_lens.l1 import FEATURE_SCHEMA, KEYS, MODELS, load_package, predict_rows, schema_digest, transform
from evidence_lens.l1_fit import (
    CANDIDATES,
    EXPORT_TOLERANCE,
    _candidate_scores,
    _train,
    folds_for,
    preprocess,
    weights,
)
from evidence_lens.scoring_io import TARGETS, file_hash, new_output, rgb_valid, write_json
from vision_data import assetspec, file_for, open_dataset, packed, sample_data, training_table
from vision_sources import signed


def full_cohort(d, sources, split):
    expected, folds, hashes = set(), {}, []
    expected_cells, groups = set(), {}
    for path, sha in sources:
        c = signed(path, sha)
        rows = [r for r in c["families"] if r["split"] == split]
        names = [r["scene_family_id"] for r in rows]
        require(
            len(set(names)) == len(names) and not expected.intersection(names),
            "duplicate family across source catalogs",
        )
        expected.update(names)
        for row in rows:
            expected_cells.update((row["scene_family_id"], condition) for condition in row["trials"])
            groups[row["scene_family_id"]] = row.get("test_group")
        folds.update({r["scene_family_id"]: r.get("cv_fold") for r in rows})
        hashes.append(sha)
    require(
        len(expected) == {"train": 28, "development": 8, "test": 8}[split]
        and {r["family"] for r in d["views"]} == expected
        and d["scope"] == "specified_materials"
        and {(r["family"], r["condition"]) for r in d["views"]} == expected_cells
        and {v["catalog_sha256"] for v in d["material_indices"]} == set(hashes),
        "formal run requires the complete preregistered cohort",
    )
    for r in d["views"]:
        require(
            r["cv_fold"] == folds[r["family"]]
            and r.get("group") == groups[r["family"]]
            and (r["height"], r["width"]) == (512, 512),
            "family assignment, test group or native target size differs",
        )
    return hash_json(sorted(hashes))


def write_training_receipt(root, d, names, x, y, f, v, w, out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from evidence_lens.l1 import ALL_COLUMNS, NUMERIC

    coords = []
    probabilities = []
    for row in d["views"]:
        index, _, _ = sample_data(root, row)
        coords.append(index)
        probabilities.append(
            np.full(
                len(index),
                row.get("sample_probability", len(index) / row["eligible"] if row["eligible"] else 0),
                np.float64,
            )
        )
    index = np.concatenate(coords)
    h, w_native = 512, 512
    require(((index >= 0) & (index < h * w_native)).all(), "sampled pixel falls outside native target")
    table = {
        "family": pa.DictionaryArray.from_arrays(pa.array(f), pa.array(names)),
        "view": pa.DictionaryArray.from_arrays(pa.array(v), pa.array(TARGETS)),
        "index": index,
        "row": index // w_native,
        "column": index % w_native,
        "Y": y,
        "statistical_weight": w,
        "inclusion_probability": np.concatenate(probabilities),
    }
    for i, key in enumerate(ALL_COLUMNS):
        table[key] = x[:, i] if key in NUMERIC else x[:, i].astype(bool)
    dest = out / "training-samples.parquet"
    pq.write_table(pa.table(table), dest, compression="zstd")
    return signature(dest)


def fit(root, expected, catalogs, output, run_id):
    import sklearn

    require(sklearn.__version__ == "1.9.1", "use the registered sklearn solver version 1.9.1")
    d = open_dataset(root, expected, "train")
    roster_sha = full_cohort(d, catalogs, "train")
    names, x, y, f, v = training_table(root, d)
    entries = {r["family"]: r for r in d["views"]}
    folds = {name: {"pin": {"cv_fold": entries[name]["cv_fold"]}} for name in names}
    require(
        all(type(r["pin"]["cv_fold"]) is int and r["pin"]["cv_fold"] in range(4) for r in folds.values()),
        "registered four-fold assignment is incomplete",
    )
    w = weights(f, v)
    ds = SimpleNamespace(families=folds)
    assignment, attempts = folds_for(ds, f, y)
    with new_output(output) as out:
        table = write_training_receipt(root, d, names, x, y, f, v, w, out)
        cv = {
            "schema": "el-vision-lr-cv-1",
            "train_dataset_sha256": expected,
            "fold_attempts": attempts,
            "candidates": list(CANDIDATES),
            "models": {},
            "sklearn": sklearn.__version__,
        }
        if assignment is not None:
            for method in MODELS:
                scores, choice = _candidate_scores(ds, x, y, f, v, assignment, method)
                cv["models"][method] = {"candidates": scores, "selected_C": choice}
        write_json(out / "cv_results.json", cv)
        result = {
            "schema": "el-vision-lr-fit-1",
            "run_id": run_id,
            "split_manifest_sha256": roster_sha,
            "training_dataset_sha256": expected,
            "cv_sha256": file_hash(out / "cv_results.json"),
            "families": names,
            "rows": len(y),
            "positive_rows": int(y.sum()),
            "training_table": table,
        }
        if assignment is None or any(cv["models"][m]["selected_C"] is None for m in MODELS):
            result.update(status="insufficient_data", reason="no_two_class_training_folds_or_validation_views")
            write_json(out / "fit.json", result)
            return result
        model = out / "model"
        model.mkdir()
        write_json(model / "feature_schema.json", FEATURE_SCHEMA)
        parity = {}
        for method in MODELS:
            p = preprocess(x, w, method)
            z = transform(x, p)
            c = cv["models"][method]["selected_C"]
            clf, stats = _train(z, y, w, c)
            p.update(
                C=c,
                classes=clf.classes_.astype(int).tolist(),
                coefficients=clf.coef_[0].tolist(),
                intercept=float(clf.intercept_[0]),
                fit_statistics=stats,
            )
            write_json(model / (method + ".json"), p)
            parity[method] = float(
                np.max(np.abs(clf.predict_proba(z)[:, 1] - predict_rows(x, p).astype(np.float64)))
            )
        require(max(parity.values()) <= EXPORT_TOLERANCE, "serialized logistic scores differ from sklearn")
        counts = {}
        for name in names:
            rows = [r for r in d["views"] if r["family"] == name]
            n = sum(r["E"] for r in rows)
            pos = sum(r["positive_E"] for r in rows)
            counts[name] = {"E": n, "Y": pos, "prevalence": pos / n if n else None}
        prior = np.mean([r["prevalence"] for r in counts.values() if r["E"]])
        index = {
            "schema": "el-l1-package-1",
            "status": "fit",
            "fit_run_id": run_id,
            "model_id": run_id,
            "feature_schema_digest": schema_digest(),
            "split_manifest_sha256": roster_sha,
            "training_manifest_sha256": expected,
            "constant_risk": float(prior),
            "training_sources": {n: entries[n]["source_input_sha256"] for n in names},
            "files": {
                p.name: file_hash(p)
                for p in (model / "feature_schema.json", *(model / (m + ".json") for m in MODELS))
            },
        }
        write_json(model / "model.json", index)
        load_package(model, require_frozen=False)
        result.update(
            status="fitted",
            constant_family_counts=counts,
            max_abs_parity=parity,
            model_sha256=file_hash(model / "model.json"),
        )
        write_json(out / "fit.json", result)
    return result


def infer_five(rgb, deploy, model):
    shape = rgb.shape[:2]
    x, ok = deploy["X"], deploy["scoreable"]
    require(
        rgb.dtype == x.dtype == np.float32
        and rgb.shape == (*shape, 3)
        and x.shape == (*shape, 26)
        and ok.shape == shape
        and ok.dtype == bool,
        "native deployment array mismatch",
    )
    result = {
        "constant": np.where(rgb_valid(rgb), np.float32(model["index"]["constant_risk"]), np.float32(np.nan)),
        "alpha_score": deploy["alpha_score"],
        "l0": deploy["l0"],
    }
    ids = np.flatnonzero(ok)
    for method in MODELS:
        scores = np.full(shape, np.nan, np.float32)
        for start in range(0, len(ids), 32768):
            rows = ids[start : start + 32768]
            scores.ravel()[rows] = predict_rows(x.reshape(-1, 26)[rows], model["models"][method])
        result[KEYS[method]] = scores
    return result


def score(root, expected, model_dir, model_sha, output, split, opening=None, test_catalog=None):
    from vision_evaluate import model_matches, test_access

    marker = None
    if split == "test":
        marker, method = test_access(opening, test_catalog)
        require(
            all(model_matches(method, m, model_sha) for m in MODELS),
            "test requested before selected LR package freeze",
        )
    ds = open_dataset(root, expected, split, marker)
    model = load_package(model_dir, require_frozen=False, expected_index_sha256=model_sha)
    train_sources = model["index"]["training_sources"]
    require(
        not set(train_sources).intersection(r["family"] for r in ds["views"])
        and not set(v for v in train_sources.values() if v is not None).intersection(
            r["source_input_sha256"] for r in ds["views"]
        ),
        "train and evaluation identities overlap",
    )
    views = []
    with new_output(output) as out:
        for row in ds["views"]:
            item = {k: row[k] for k in ("family", "condition", "view", "reason")}
            item.update(rgb_sha256=None, deploy_sha256=None, score=None, status=row["status"])
            if row["status"] == "completed":
                rgb = np.load(file_for(root, row["rgb"]), mmap_mode="r", allow_pickle=False)
                dep = packed(file_for(root, row["full"]["deploy"]), ("X", "scoreable", "l0", "alpha_score"))
                scores = infer_five(rgb, dep, model)
                dest = out / row["family"] / row["condition"] / (row["view"] + ".npz")
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, **scores)
                item.update(
                    rgb_sha256=row["rgb"]["sha256"],
                    deploy_sha256=row["full"]["deploy"]["sha256"],
                    score=assetspec(dest, out),
                    status="scored",
                )
            views.append(item)
        index = {
            "schema": "el-vision-scores-1",
            "method": "original-five",
            "model_sha256": model_sha,
            "dataset_sha256": expected,
            "views": views,
        }
        write_json(out / "scores.json", index)
    return index


def main():
    import json, os

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    cmd = sub.add_parser("fit")
    cmd.add_argument("--catalog", nargs=2, action="append", required=True, metavar=("CATALOG", "SHA256"))
    cmd.add_argument("--run-id", default=os.environ.get("EL_RUN_ID"))
    cmd = sub.add_parser("score")
    cmd.add_argument("--split", choices=("development", "test"), required=True)
    cmd.add_argument("--model", type=Path, required=True)
    cmd.add_argument("--model-sha256", required=True)
    cmd.add_argument("--opening", nargs=2, metavar=("STORE", "COLD_SHELF"))
    cmd.add_argument("--test-catalog", nargs=2)
    for cmd in sub.choices.values():
        cmd.add_argument("--dataset", type=Path, required=True)
        cmd.add_argument("--sha256", required=True)
        cmd.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.action == "fit":
        require(a.run_id, "recorded LR fit ID required")
        fit(a.dataset, a.sha256, a.catalog, a.output, a.run_id)
        name = "fit.json"
    else:
        score(a.dataset, a.sha256, a.model, a.model_sha256, a.output, a.split, a.opening, a.test_catalog)
        name = "scores.json"
    print(json.dumps({"file": name, **signature(a.output / name)}))


if __name__ == "__main__":
    main()
