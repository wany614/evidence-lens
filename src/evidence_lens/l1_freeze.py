"""Choose operating thresholds on complete development B0 images, preserving whole ties."""

import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np

from .l1 import KEYS, MODELS, load_package, predict_rows
from .l1_data import Dataset, check
from .scoring_io import TARGETS, file_hash, new_output, write_json


METHODS = (*MODELS, "constant", "1-alpha")


def threshold_for(lists, denominators):
    """Find the attainable macro-family coverage nearest 4/5 with exact comparisons at the boundary."""
    valid = {f: n for f, n in denominators.items() if n > 0}
    check(valid, "development has no reference-valid denominator")
    ordered = {f: np.sort(lists[f]) for f in valid}
    distinct = np.unique(np.concatenate(list(ordered.values())))
    def coverage(group_count):
        if group_count == 0:
            return Fraction(0)
        limit = distinct[group_count - 1]
        return sum((Fraction(int(np.searchsorted(ordered[f], limit, side="right")), n)
                    for f, n in valid.items()), Fraction(0)) / len(valid)
    def tau(j):
        if j == 0:
            return 0.0
        v = float(distinct[j - 1])
        if j == len(distinct):
            return float(np.nextafter(v, np.inf))
        return (v + float(distinct[j])) / 2
    target = Fraction(4, 5)
    left, right = 0, len(distinct)
    while left < right:
        m = (left + right) // 2
        if coverage(m) < target:
            left = m + 1
        else:
            right = m
    feasible = sorted({left, max(0, left - 1)})
    selected = min(feasible, key=lambda j: (abs(coverage(j) - target), coverage(j)))
    return {"tau": tau(selected), "coverage": float(coverage(selected)), "accepted_distinct_groups": selected,
            "candidate_count": len(distinct) + 1, "eligible_families": sorted(valid),
            "closest_candidates": [{"groups": j, "coverage": float(coverage(j)), "tau": tau(j)} for j in feasible]}


def operating_counts(scores, y, v, tau):
    finite = np.isfinite(scores)
    low = finite & (scores.astype(np.float64) < tau)
    n = int(low.sum())
    errors = int(y[low].sum())
    return {"V": v, "E": len(y), "L": n, "H": int(finite.sum()) - n, "U": int((~finite).sum()),
            "Y": int(y.sum()), "Y_in_L": errors, "Y_in_H": int(y[finite & ~low].sum()), "Y_in_U": int(y[~finite].sum()),
            "overall_coverage": n/v if v else None, "coverage_null_reason": None if v else "no_valid_reference",
            "low_risk_error_null_reason": None if n else "no_low_risk_pixels",
            "low_risk_error_rate": errors/n if n else None}


def freeze(root, fit_dir, output, run_id):
    check(isinstance(run_id, str) and run_id.strip(), "a threshold run ID is required")
    fit_dir = Path(fit_dir)
    fit = load_package(fit_dir, require_frozen=False)
    check(fit["index"]["status"] == "fit", "threshold selection needs the fitted package")
    ds = Dataset(root, "dev")
    base = fit["index"]
    check(ds.index["split_manifest_sha256"] == base["split_manifest_sha256"], "fit and development split pins differ")
    train_sources = base["training_sources"]
    check(not set(ds.families).intersection(train_sources)
          and not {r["source_input_sha256"] for r in ds.families.values()}.intersection(train_sources.values()),
          "training and development family/source overlap")
    with new_output(output) as out:
        all_views, paths = {}, {}
        for f, rec in ds.families.items():
            all_views[f], paths[f] = {}, {}
            folder = out / "development_scores" / f
            folder.mkdir(parents=True)
            for view in TARGETS:
                doc = ds.read(f, view)
                ok, y = doc["scoreable"], doc["Y"]
                scores = {"constant": np.full(y.shape, base["constant_risk"], np.float32), "1-alpha": doc["alpha_score"]}
                raw = doc["X"][ok]
                for method in MODELS:
                    arr = np.full(y.shape, np.nan, np.float32)
                    arr[ok] = predict_rows(raw, fit["models"][method])
                    scores[method] = arr
                file = folder / (view + ".npz")
                np.savez_compressed(file, index=doc["index"], Y=y, **{KEYS[k]: a for k, a in scores.items()})
                paths[f][view] = {"path": file.relative_to(out).as_posix(), "sha256": file_hash(file)}
                all_views[f][view] = (rec["views"][view]["V"], y, scores)
        denominators = {f: sum(v[0] for v in views.values()) for f, views in all_views.items()}
        if not any(denominators.values()):
            result = {"schema": "el-development-freeze-1", "status": "insufficient_data", "reason": "no_reference_valid_pixels",
                      "run_id": run_id, "input_hashes": ds.audit.files, "development_scores": paths}
            write_json(out / "development.json", result)
            return result
        choices = {}
        for method in METHODS:
            values = {f: np.concatenate([s[method][np.isfinite(s[method])] for _, _, s in views.values()])
                      for f, views in all_views.items()}
            choice = threshold_for(values, denominators)
            tau = choice["tau"]
            choice["views"] = {f: {v: operating_counts(s[method], y, n, tau) for v, (n, y, s) in views.items()}
                               for f, views in all_views.items()}
            counts = {}
            for f, views in choice["views"].items():
                sums = {k: sum(row[k] for row in views.values()) for k in ("V", "E", "L", "H", "U", "Y", "Y_in_L", "Y_in_H", "Y_in_U")}
                sums["overall_coverage"] = sums["L"] / sums["V"] if sums["V"] else None
                sums["low_risk_error_rate"] = sums["Y_in_L"] / sums["L"] if sums["L"] else None
                sums["coverage_null_reason"] = None if sums["V"] else "no_valid_reference"
                sums["low_risk_error_null_reason"] = None if sums["L"] else "no_low_risk_pixels"
                counts[f] = sums
            choice["families"] = counts
            rates = [r["low_risk_error_rate"] for r in counts.values() if r["low_risk_error_rate"] is not None]
            choice["family_macro_low_risk_error_rate"] = float(np.mean(rates)) if rates else None
            choices[method] = choice
        folder = out / "model"
        folder.mkdir()
        for name in base["files"]:
            shutil.copyfile(fit_dir / name, folder / name)
        thresholds = {"schema": "el-l1-thresholds-1", "threshold_run_id": run_id, "fit_run_id": base["fit_run_id"],
                      "split_manifest_sha256": base["split_manifest_sha256"], "thresholds": {m: r["tau"] for m, r in choices.items()},
                      "low_risk_rule": "finite score < tau", "target": 0.8, "selection": "whole ties, nearest family-macro coverage, lower coverage on ties"}
        write_json(folder / "thresholds.json", thresholds)
        index = dict(base, status="frozen", model_id=run_id, threshold_run_id=run_id,
                     development_sources={name: rec["source_input_sha256"] for name, rec in ds.families.items()},
                     files={**base["files"], "thresholds.json": file_hash(folder / "thresholds.json")})
        write_json(folder / "model.json", index)
        load_package(folder)
        result = {"schema": "el-development-freeze-1", "status": "frozen", "run_id": run_id, "fit_run_id": base["fit_run_id"],
                  "split_manifest_sha256": ds.index["split_manifest_sha256"],
                  "fit_package_files": fit["file_hashes"], "input_hashes": ds.audit.files, "family_sources": ds.families,
                  "development_scores": paths, "methods": choices, "model_index_sha256": file_hash(folder / "model.json")}
        write_json(out / "development.json", result)
    return result
