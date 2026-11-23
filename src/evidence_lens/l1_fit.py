"""Grouped, family-weighted L2 logistic regression on the frozen B0 training samples."""

import time
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .evaluation import _ap, _groups
from .l1 import ALL_COLUMNS, FEATURE_SCHEMA, MODELS, columns, load_package, predict_rows, schema_digest, transform
from .l1_data import Dataset, SAMPLE_CAP, SEED, check
from .scoring_io import TARGETS, file_hash, new_output, write_json


CANDIDATES = (0.1, 1.0, 10.0)
AP_TIE = 1e-6
EXPORT_TOLERANCE = 1e-6


def weights(families, views):
    n = len(families)
    result = np.empty(n, np.float64)
    distinct = np.unique(families)
    for family in distinct:
        idx = np.flatnonzero(families == family)
        _, codes, counts = np.unique(views[idx], return_inverse=True, return_counts=True)
        result[idx] = n / (len(distinct) * len(counts) * counts[codes])
    return result


def preprocess(x, w, method):
    nums, cols = columns(method)
    a = x[:, [ALL_COLUMNS.index(k) for k in nums]].astype(np.float64)
    missing = ~np.isfinite(a)
    dead = missing.all(axis=0)
    medians = np.zeros(len(nums), np.float64)
    for j in np.flatnonzero(~dead):
        medians[j] = np.median(a[~missing[:, j], j])
    a = np.where(missing, medians, a)
    means = np.average(a, axis=0, weights=w)
    variance = np.average((a - means) ** 2, axis=0, weights=w)
    constant = np.all(a == a[0], axis=0)
    scales = np.where(constant | (variance == 0), 1.0, np.sqrt(variance))
    return {"schema": "el-l1-coefficients-1", "method": method, "feature_schema_digest": schema_digest(),
            "numeric_columns": nums, "columns": cols, "medians": medians.tolist(), "all_missing": dead.tolist(),
            "means": means.tolist(), "scales": scales.tolist(), "constant_numeric": constant.tolist()}


def _train(z, y, w, c):
    clf = LogisticRegression(C=c, l1_ratio=0, solver="lbfgs", max_iter=1000, tol=1e-6,
                             fit_intercept=True, class_weight=None)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as events, threadpool_limits(limits=1):
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(z, y, sample_weight=w)
    conv = [str(ev.message) for ev in events if issubclass(ev.category, ConvergenceWarning)]
    return clf, {"seconds": time.perf_counter() - start, "iterations": int(clf.n_iter_[0]),
                 "converged": not conv, "convergence_warnings": conv}


def folds_for(dataset, f, y):
    names = list(dataset.families)
    attempts = []
    for k in (4, 3, 2):
        assignment = np.array([dataset.families[name]["pin"]["cv_fold"] for name in names]) if k == 4 else np.arange(len(names)) % k
        folds, possible = [], True
        for j in range(k):
            train, val = np.flatnonzero(assignment != j), np.flatnonzero(assignment == j)
            tm, vm = np.isin(f, train), np.isin(f, val)
            positive, negative = int(y[tm].sum()), int(tm.sum() - y[tm].sum())
            possible &= bool(len(val) and positive and negative)
            folds.append({"fold": j, "train_families": [names[i] for i in train], "validation_families": [names[i] for i in val],
                          "train_rows": int(tm.sum()), "train_positive": positive, "train_negative": negative,
                          "validation_rows": int(vm.sum()), "validation_positive": int(y[vm].sum())})
        attempts.append({"k": k, "usable_training_folds": possible, "folds": folds})
        if possible:
            return assignment, attempts
    return None, attempts


def _view_aps(prediction, y, f, v, val_names, names):
    by_family = {}
    for fid in val_names:
        views = {}
        for vid, name in enumerate(TARGETS):
            mask = (f == fid) & (v == vid)
            _, counts, positives = _groups(prediction[mask], y[mask])
            n, pos = int(counts.sum()), int(positives.sum())
            ap = _ap(counts, positives)
            reason = None if ap is not None else ("no_rows" if not n else "no_positives" if not pos else "no_negatives")
            views[name] = {"ap": ap, "status": "evaluated" if reason is None else "ranking_not_evaluable",
                           "reason": reason, "rows": n, "positive": pos, "negative": n - pos}
        valid = [r["ap"] for r in views.values() if r["ap"] is not None]
        by_family[names[fid]] = {"views": views, "valid_views": len(valid), "family_ap": float(np.mean(valid)) if valid else None}
    return by_family


def choose_c(means):
    defined = {c: v for c, v in means.items() if v is not None}
    if not defined:
        return None
    best = max(defined.values())
    return min(c for c, v in defined.items() if best - v <= AP_TIE)


def _candidate_scores(dataset, x, y, f, v, assignment, method):
    names = list(dataset.families)
    grouped = {c: {"folds": [], "family_aps": {}} for c in CANDIDATES}
    for fold in range(int(assignment.max()) + 1):
        val_f = np.flatnonzero(assignment == fold)
        valid = np.isin(f, val_f)
        train = ~valid
        w = weights(f[train], v[train])
        params = preprocess(x[train], w, method)
        train_z, val_z = transform(x[train], params), transform(x[valid], params)
        for c in CANDIDATES:
            clf, stats = _train(train_z, y[train], w, c)
            predictions = clf.predict_proba(val_z)[:, 1].astype(np.float32) if val_z.shape[0] else np.empty(0, np.float32)
            scores = _view_aps(predictions, y[valid], f[valid], v[valid], val_f, names)
            grouped[c]["folds"].append({"fold": fold, **stats, "train_rows": int(train.sum()),
                                        "weight_sum": float(w.sum()), "fold_preprocessing": params, "validation": scores,
                                        "coefficients": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
                                        "classes": clf.classes_.astype(int).tolist()})
            grouped[c]["family_aps"].update({n: row["family_ap"] for n, row in scores.items()})
    means = {}
    for c, candidate in grouped.items():
        values = [v for v in candidate["family_aps"].values() if v is not None]
        means[c] = float(np.mean(values)) if values else None
        candidate.update(family_macro_ap=means[c], valid_families=len(values))
    return grouped, choose_c(means)


def _table(dataset, output):
    names = list(dataset.families)
    features, labels, fids, vids, xs, ys, probs = [], [], [], [], [], [], []
    for fid, family in enumerate(names):
        for vid, view in enumerate(TARGETS):
            doc = dataset.read(family, view)
            idx = doc["index"]
            info = dataset.families[family]["views"][view]
            n = len(idx)
            features.append(doc["X"])
            labels.append(doc["Y"])
            fids.append(np.full(n, fid, np.uint8))
            vids.append(np.full(n, vid, np.uint8))
            row, col = np.divmod(idx, info["shape"][1])
            xs.append(col.astype(np.int32))
            ys.append(row.astype(np.int32))
            probs.append(np.full(n, info["sample_probability"] or 0, np.float64))
    x, y, f, v = map(np.concatenate, (features, labels, fids, vids))
    w = weights(f, v)
    columns_out = {"row": np.concatenate(ys), "column": np.concatenate(xs), "Y": y,
                   "sample_probability": np.concatenate(probs), "weight": w}
    for k in ("scene_family_id", "trial_id", "score_run_id", "evaluation_run_id"):
        dictionary = names if k == "scene_family_id" else [dataset.families[n]["pin"][k] for n in names]
        columns_out[k] = pa.DictionaryArray.from_arrays(pa.array(f, type=pa.int32()), pa.array(dictionary))
    columns_out["camera_id"] = pa.DictionaryArray.from_arrays(pa.array(v, type=pa.int32()), pa.array(TARGETS))
    for j, name in enumerate(ALL_COLUMNS):
        columns_out[name] = x[:, j] if name in FEATURE_SCHEMA["numeric_features"] else x[:, j].astype(bool)
    pq.write_table(pa.table(columns_out), output, compression="zstd")
    return x, y, f, v, w


def fit(root, output, run_id):
    check(isinstance(run_id, str) and run_id.strip(), "a fit run ID is required")
    check(sklearn.__version__ == "1.9.1", f"expected scikit-learn 1.9.1, got {sklearn.__version__}")
    ds = Dataset(root, "train")
    with new_output(output) as out:
        table = out / "training_features.parquet"
        x, y, f, v, w = _table(ds, table)
        assignment, attempts = folds_for(ds, f, y)
        manifest = {"schema": "el-training-dataset-1", "fit_run_id": run_id, "split": "train", "condition_id": "B0",
                    "split_manifest_sha256": ds.index["split_manifest_sha256"], "family_sources": ds.families,
                    "input_hashes": ds.audit.files, "feature_schema_digest": schema_digest(), "seed": SEED,
                    "sample_cap_per_view": SAMPLE_CAP, "row_count": len(y), "positive_count": int(y.sum()),
                    "selection": "uniform without replacement on scoreable & E, source self-render excluded",
                    "preprocessing_version": "el-l1-fold-median-weighted-z-1",
                    "coordinates_and_features_file": table.name, "coordinates_and_features_sha256": file_hash(table),
                    "fold_attempts": attempts}
        write_json(out / "training_dataset_manifest.json", manifest)
        cv = {"schema": "el-l1-cv-1", "fit_run_id": run_id, "models": {}, "C_grid": list(CANDIDATES),
              "ap_tie_tolerance": AP_TIE, "fold_attempts": attempts, "sklearn_version": sklearn.__version__,
              "solver": "lbfgs", "l1_ratio": 0, "class_weight": None, "max_iter": 1000, "tol": 1e-6,
              "scipy": scipy.__version__, "numpy": np.__version__, "pyarrow": pa.__version__}
        reason = None
        if assignment is None:
            reason = "two_class_training_folds_unavailable"
        else:
            for method in MODELS:
                candidates, c = _candidate_scores(ds, x, y, f, v, assignment, method)
                cv["models"][method] = {"candidates": candidates, "selected_C": c}
                if c is None:
                    reason = "no_two_class_validation_views"
        write_json(out / "cv_results.json", cv)
        if reason:
            result = {"schema": "el-model-fit-1", "fit_run_id": run_id, "status": "insufficient_data", "reason": reason,
                      "training_manifest_sha256": file_hash(out / "training_dataset_manifest.json"),
                      "cv_sha256": file_hash(out / "cv_results.json"), "rows": len(y), "positives": int(y.sum())}
            write_json(out / "fit.json", result)
            return result
        folder = out / "model"
        folder.mkdir()
        write_json(folder / "feature_schema.json", FEATURE_SCHEMA)
        fitted = {}
        for method in MODELS:
            params = preprocess(x, w, method)
            z = transform(x, params)
            c = cv["models"][method]["selected_C"]
            clf, stats = _train(z, y, w, c)
            params.update(C=c, coefficients=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]),
                          classes=clf.classes_.astype(int).tolist(), fit_statistics=stats)
            write_json(folder / (method + ".json"), params)
            fitted[method] = clf.predict_proba(z)[:, 1]
        family_error = {}
        for name, rec in ds.families.items():
            n = sum(info["E"] for info in rec["views"].values())
            pos = sum(info["positive_E"] for info in rec["views"].values())
            family_error[name] = {"positive": pos, "evaluable": n, "error_rate": pos/n if n else None}
        prior = float(np.mean([row["error_rate"] for row in family_error.values() if row["error_rate"] is not None]))
        index = {"schema": "el-l1-package-1", "status": "fit", "fit_run_id": run_id, "model_id": run_id,
                 "feature_schema_digest": schema_digest(), "split_manifest_sha256": ds.index["split_manifest_sha256"],
                 "training_manifest_sha256": file_hash(out / "training_dataset_manifest.json"), "constant_risk": prior,
                 "training_sources": {name: r["source_input_sha256"] for name, r in ds.families.items()},
                 "files": {p.name: file_hash(p) for p in [folder / "feature_schema.json", *(folder/(m + ".json") for m in MODELS)]}}
        write_json(folder / "model.json", index)
        loaded = load_package(folder, require_frozen=False)
        errors = {}
        for method in MODELS:
            exported = predict_rows(x, loaded["models"][method])
            errors[method] = float(np.max(np.abs(exported.astype(np.float64) - fitted[method])))
        check(max(errors.values()) <= EXPORT_TOLERANCE, "persisted NumPy scores differ from sklearn beyond 1e-6")
        result = {"schema": "el-model-fit-1", "status": "fitted", "fit_run_id": run_id, "families": len(ds.families),
                  "rows": len(y), "positives": int(y.sum()), "constant_prior_family_counts": family_error,
                  "training_manifest_sha256": index["training_manifest_sha256"], "cv_sha256": file_hash(out/"cv_results.json"),
                  "model_index_sha256": file_hash(folder/"model.json"), "parity_rows": len(y),
                  "export_absolute_tolerance": EXPORT_TOLERANCE, "max_sklearn_numpy_error": errors}
        write_json(out / "fit.json", result)
        card = [
            f"# Evidence Lens L1 fit {run_id}", "",
            f"The two logistic regressions use {len(y):,} B0 pixel samples from {len(ds.families)} synthetic scene families. "
            f"{int(y.sum()):,} samples have mean absolute display-sRGB error greater than 0.05. Each family contributes equal total weight.",
            "", "| Model | Selected C | Held-back training-family macro AP | Largest absolute export difference |",
            "| --- | ---: | ---: | ---: |",
        ]
        for method in MODELS:
            c = cv["models"][method]["selected_C"]
            heldback = cv["models"][method]["candidates"][c]["family_macro_ap"]
            card.append(f"| {method} | {c:g} | {heldback:.6f} | {errors[method]:.3g} |")
        card.extend(["", "## Population and use", "",
                     "Training families are " + ", ".join(ds.families) + ". The training manifest records each source, camera view, "
                     "pixel coordinate, selection probability and grouped fold. The constant baseline uses the complete evaluable "
                     f"family error rates and has a frozen training score of {prior:.6f}.",
                     "", "The full model uses geometry, opacity and image consistency. The geometry model excludes the three "
                     "RGB measurements and their associated missing flags. Both score finite target RGB with valid opacity and "
                     "positive validated camera-axis depth. Failed target rendering or invalid target inputs yields unknown. "
                     "Source-side missing measurements use the fitted medians and explicit indicators.",
                     "", "The numerical scores rank synthetic novel-view RGB error. Their calibration and transfer to photographs "
                     "are untested. Development coverage, test-family results, resource measurements and failure examples are "
                     "recorded by the corresponding downstream runs.", ""])
        (out / "model_card.md").write_text("\n".join(card), encoding="utf8")
    return result
