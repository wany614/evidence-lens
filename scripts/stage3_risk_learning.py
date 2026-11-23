"""Fit only the multi-source light baseline and freeze its one all-source development operating point."""

from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import argparse
import json
import os
import shutil

import numpy as np

from evidence_lens.formal_inputs import require
from evidence_lens.multisource import COLUMNS, CONDITIONS, NUMERIC, SCOPES, TARGETS, schema_sha256, source_ids
from evidence_lens.multisource_model import light_risk, load_light, transform, validate_rows
from evidence_lens.l1 import risk_state
from evidence_lens.l1_fit import CANDIDATES, choose_c
from evidence_lens.scoring_io import file_hash, new_output, write_json
from stage3_risk_data import dataset, read_asset
from stage3_risk_sources import CATALOG, STUDY, LEARNING, asset
from vision_evaluate import ranking


def mean_defined(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def balanced(values):
    """(family, condition, view) -> scalar or None; return family-equal mean and all group values."""
    cells = defaultdict(list)
    for (family, condition, view), value in values.items():
        cells[family, condition].append(value)
    names = sorted({x[0] for x in cells})
    by = {f: {c: mean_defined(cells[f, c]) for c in CONDITIONS} for f in names}
    return mean_defined([mean_defined(c.values()) for c in by.values()]), by


def threshold_for(cells, denominators):
    """Nearest exactly attainable macro L/V. Family, then positive-V conditions, have equal weight."""
    ordered = {k: np.sort(np.asarray(cells.get(k, []), np.float32)) for k, n in denominators.items() if n > 0}
    require(bool(ordered), "development has no positive reference-valid denominator")
    for k, x in ordered.items():
        require(
            np.isfinite(x).all() and np.all((x >= 0) & (x <= 1)) and len(x) <= denominators[k],
            "invalid threshold inputs",
        )
    allvals = np.unique(np.concatenate(list(ordered.values())))
    families = sorted({f for f, c in ordered})

    def coverage(j):
        x = allvals[j - 1] if j else None
        by = defaultdict(list)
        for (f, c), s in ordered.items():
            n = int(np.searchsorted(s, x, side="right")) if j else 0
            by[f].append(Fraction(n, int(denominators[f, c])))
        return sum((sum(by[f], Fraction(0)) / len(by[f]) for f in families), Fraction(0)) / len(families)

    def tau(j):
        if not j:
            return 0.0
        last = float(allvals[j - 1])
        return float(np.nextafter(last, np.inf)) if j == len(allvals) else (last + float(allvals[j])) / 2.0

    l, r = 0, len(allvals)
    while l < r:
        j = (l + r) // 2
        if coverage(j) < Fraction(4, 5):
            l = j + 1
        else:
            r = j
    choices = sorted({l, max(l - 1, 0)})
    best = min(choices, key=lambda j: (abs(coverage(j) - Fraction(4, 5)), coverage(j)))
    return {
        "tau": tau(best),
        "coverage": float(coverage(best)),
        "accepted_distinct_groups": best,
        "candidate_count": len(allvals) + 1,
        "eligible_families": families,
        "denominators": [
            {"family": f, "condition": c, "V": int(n), "finite_evaluable": len(ordered.get((f, c), ()))}
            for (f, c), n in denominators.items()
        ],
        "closest_candidates": [{"groups": j, "coverage": float(coverage(j)), "tau": tau(j)} for j in choices],
    }


def weights(groups):
    n = len(groups)
    w = np.empty(n, np.float64)
    families = np.unique(groups[:, 0])
    for f in families:
        fmask = groups[:, 0] == f
        conditions = np.unique(groups[fmask, 1])
        for c in conditions:
            idx = np.flatnonzero(fmask & (groups[:, 1] == c))
            _, codes, counts = np.unique(groups[idx, 2], return_inverse=True, return_counts=True)
            w[idx] = n / (len(families) * len(conditions) * len(counts) * counts[codes])
    return w


def preprocessor(x, w):
    validate_rows(x)
    a = x[:, : len(NUMERIC)].astype(np.float64)
    ok = np.isfinite(a)
    dead = ~ok.any(0)
    med = np.zeros(len(NUMERIC))
    for j in np.flatnonzero(~dead):
        med[j] = np.median(a[ok[:, j], j])
    a = np.where(ok, a, med)
    mean = np.average(a, axis=0, weights=w)
    var = np.average((a - mean) ** 2, axis=0, weights=w)
    constant = np.all(a == a[0], axis=0)
    scale = np.where((var == 0) | constant, 1.0, np.sqrt(var))
    return {
        "schema": "el-stage3-risk-preprocessing-1",
        "feature_schema_sha256": schema_sha256(),
        "columns": list(COLUMNS),
        "numeric_count": len(NUMERIC),
        "medians": med.tolist(),
        "means": mean.tolist(),
        "scales": scale.tolist(),
        "all_missing": dead.tolist(),
        "constant_numeric": constant.tolist(),
    }


def sample_table(root, d):
    xs = []
    ys = []
    gs = []
    inventory = []
    for row in d["views"]:
        f, c, v = row["family"], row["condition"], row["view"]
        mode = row["modes"]["all"]
        s = mode["samples"]
        info = {
            "family": f,
            "condition": c,
            "view": v,
            "status": row["status"],
            "file": s,
            "eligible": mode["eligible"],
        }
        if s:
            a = read_asset(root, s)
            x, y, index = (a[k] for k in ("X", "Y", "index"))
            validate_rows(x)
            require(
                index.dtype == np.int32
                and index.ndim == 1
                and np.all(np.diff(index) > 0)
                and np.all((index >= 0) & (index < row["width"] * row["height"]))
                and len(index) == len(x) == len(y)
                and len(index) == min(4096, mode["eligible"])
                and y.dtype == np.uint8
                and np.all(y <= 1)
                and a["inclusion_probability"].shape == ()
                and a["inclusion_probability"].dtype == np.float64
                and float(a["inclusion_probability"])
                == (len(y) / mode["eligible"] if mode["eligible"] else 0.0),
                "bad sampled row identities or labels",
            )
            truth = read_asset(root, row["truth"])
            require(
                np.all(truth["rgb_evaluable"].ravel()[index])
                and np.array_equal(y, truth["rgb_high_error"].ravel()[index]),
                "sample label changed",
            )
            if len(y):
                xs.append(x)
                ys.append(y)
                gs.append(
                    np.tile(
                        [d["families"].index(f), list(CONDITIONS).index(c), TARGETS.index(v)], (len(y), 1)
                    )
                )
            info.update(rows=len(y), positive=int(y.sum()))
        else:
            require(
                row["status"] == "render_failed" and mode["eligible"] == 0,
                "missing successful training sample",
            )
            info.update(rows=0, positive=0)
        inventory.append(info)
    x = np.concatenate(xs) if xs else np.empty((0, len(COLUMNS)), np.float32)
    y = np.concatenate(ys) if ys else np.empty(0, np.uint8)
    g = np.concatenate(gs) if gs else np.empty((0, 3), int)
    return x, y, g, inventory


def train(x, y, g, c):
    from evidence_lens.l1_fit import _train

    w = weights(g)
    p = preprocessor(x, w)
    clf, stats = _train(transform(x, p, np.float64), y, w, c)
    return p, clf, stats


def fit(root, ds_sha, out, run_id):
    d = dataset(root, ds_sha, "train")
    require(
        d["cohort"] == "formal" and len(d["families"]) == 8,
        "formal LR requires eight registered training families",
    )
    x, y, g, inv = sample_table(root, d)
    names = d["families"]
    ordinals = np.array([sorted(names).index(n) for n in names])
    attempts = []
    selected = None
    for k in (4, 3, 2):
        folds = ordinals % k
        trial = []
        for j in range(k):
            valid = np.isin(g[:, 0], np.flatnonzero(folds == j))
            classes = np.bincount(y[~valid], minlength=2)
            trial.append(
                {
                    "fold": j,
                    "train": [n for i, n in enumerate(names) if folds[i] != j],
                    "validation": [n for i, n in enumerate(names) if folds[i] == j],
                    "training_class_counts": classes.tolist(),
                    "validation_rows": int(valid.sum()),
                }
            )
        attempts.append({"k": k, "folds": trial})
        if all(r["validation"] and min(r["training_class_counts"]) > 0 for r in trial):
            selected = folds
            break
    result = {
        "schema": "el-stage3-light-fit-1",
        "run_id": run_id,
        "catalog_sha256": CATALOG,
        "study_sha256": STUDY,
        "learning_sha256": LEARNING,
        "feature_schema_sha256": schema_sha256(),
        "training_dataset_sha256": ds_sha,
        "training_view_samples": inv,
        "candidate_C": list(CANDIDATES),
        "fold_attempts": attempts,
        "candidates": {},
        "status": "unfit",
        "reason": "no_supported_grouped_folds",
    }
    with new_output(out) as dst:
        means = {}
        if selected is not None:
            for c in CANDIDATES:
                values = {}
                fold_stats = []
                for j in sorted(set(selected)):
                    val = np.isin(g[:, 0], np.flatnonzero(selected == j))
                    p, clf, stats = train(x[~val], y[~val], g[~val], c)
                    fold_stats.append({"fold": int(j), "preprocessing": p, "solver": stats})
                    sub = g[val]
                    yy = y[val]
                    pred = (
                        clf.predict_proba(transform(x[val], p, np.float64))[:, 1].astype(np.float32)
                        if val.any()
                        else np.empty(0, np.float32)
                    )
                    for f in np.flatnonzero(selected == j):
                        for ci, cond in enumerate(CONDITIONS):
                            for vi, v in enumerate(TARGETS):
                                m = (sub[:, 0] == f) & (sub[:, 1] == ci) & (sub[:, 2] == vi)
                                values[names[f], cond, v] = ranking(pred[m], yy[m])["ap"]
                macro, by = balanced(values)
                means[c] = macro
                result["candidates"][str(c)] = {
                    "mean_ap": macro,
                    "family_condition_ap": by,
                    "folds": fold_stats,
                    "views": [
                        {"family": f, "condition": cond, "view": v, "ap": a}
                        for (f, cond, v), a in values.items()
                    ],
                }
            best = choose_c(means)
            if best is None:
                result["reason"] = "no_two_class_validation_view"
            else:
                prep, clf, stats = train(x, y, g, best)
                package = dst / "package"
                package.mkdir()
                write_json(package / "preprocessing.json", prep)
                coef = {
                    "schema": "el-stage3-lr-coefficients-1",
                    "method": "MS-L1",
                    "feature_schema_sha256": schema_sha256(),
                    "columns": list(COLUMNS),
                    "classes": [0, 1],
                    "C": best,
                    "preprocessing_sha256": file_hash(package / "preprocessing.json"),
                    "coefficients": clf.coef_[0].tolist(),
                    "intercept": float(clf.intercept_[0]),
                }
                write_json(package / "coefficients.json", coef)
                files = {k: asset(package, package / k) for k in ("coefficients.json", "preprocessing.json")}
                model = {
                    "schema": "el-stage3-light-package-1",
                    "status": "fit",
                    "method": "MS-L1",
                    "model_id": run_id,
                    "fit_run_id": run_id,
                    "feature_schema_sha256": schema_sha256(),
                    "study_sha256": STUDY,
                    "training_dataset_sha256": ds_sha,
                    "weights_sha256": files["coefficients.json"]["sha256"],
                    "files": files,
                }
                write_json(package / "model.json", model)
                parity = 0.0
                for start in range(0, len(x), 32768):
                    q = x[start : start + 32768]
                    expected = clf.predict_proba(transform(q, prep, np.float64))[:, 1]
                    actual = light_risk({"X": q, "scoreable": np.ones(len(q), bool)}, coef, prep)
                    parity = max(parity, float(np.max(np.abs(actual - expected))))
                require(parity <= 1e-6, "native NumPy export differs from sklearn")
                result.update(
                    status="fit",
                    reason=None,
                    C=best,
                    solver=stats,
                    max_abs_numpy_sklearn=parity,
                    parity_rows=len(x),
                    model_index_sha256=file_hash(package / "model.json"),
                    preprocessing_sha256=files["preprocessing.json"]["sha256"],
                )
        write_json(dst / "fit.json", result)
    return result


def modes_risks(root, row, coefficients, prep):
    out = {}
    if row["status"] != "ready":
        return out
    for mode in SCOPES:
        entry = row["modes"][mode]
        if entry["alias_of"]:
            out[mode] = out[entry["alias_of"]]
        else:
            x = read_asset(root, entry["deploy"])
            score = light_risk(x, coefficients, prep)
            require(np.array_equal(np.isfinite(score), x["scoreable"]), "LR changed deployable support")
            out[mode] = score
    return out


def baseline_scores(root, d, ds_sha, coef, prep, tau, method, index, weights_sha, model_id, out):
    out.mkdir(parents=True)
    roster = []
    for row in d["views"]:
        f, c, v = row["family"], row["condition"], row["view"]
        scores = modes_risks(root, row, coef, prep) if method == "MS-L1" else {}
        fixed = (
            read_asset(root, row["transfer"])
            if method == "B-L1-transfer" and row["status"] == "ready"
            else None
        )
        modes = {}
        for s in SCOPES:
            alias = row["modes"][s]["alias_of"]
            if alias:
                file = modes[alias]["file"]
            elif row["status"] != "ready":
                file = None
            elif method == "B-L1-transfer" and s != "one":
                file = modes["one"]["file"]
            else:
                risk = fixed if fixed is not None else scores[s]
                p = out / f / c / v / (s + ".npz")
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(p, risk=risk, valid=np.isfinite(risk), state=risk_state(risk, tau))
                file = asset(out, p)
            modes[s] = {"source_ids": list(source_ids(c, s)), "alias_of": alias, "file": file}
        roster.append({"family": f, "condition": c, "view": v, "status": row["status"], "modes": modes})
    write_json(
        out / "scores.json",
        {
            "schema": "el-stage3-risk-scores-1",
            "method": method,
            "model_id": model_id,
            "model_index_sha256": index,
            "weights_sha256": weights_sha,
            "threshold": tau,
            "dataset_sha256": ds_sha,
            "feature_schema_sha256": schema_sha256(),
            "catalog_sha256": CATALOG,
            "views": roster,
        },
    )


def freeze(root, ds_sha, fit_dir, fit_sha, out, run_id):
    d = dataset(root, ds_sha, "development")
    require(
        d["cohort"] == "formal" and len(d["families"]) == 4,
        "formal light thresholds require four registered development families",
    )
    model, coef, prep = load_light(fit_dir, fit_sha, require_frozen=False)
    require(model["status"] == "fit", "development cutoff must start from a pure trained model")
    vals = defaultdict(list)
    den = defaultdict(int)
    views = []
    for r in d["views"]:
        f, c, v = r["family"], r["condition"], r["view"]
        den[f, c] += r["V"]
        t = read_asset(root, r["truth"])
        e = t["rgb_evaluable"]
        if r["status"] == "ready":
            dep = read_asset(root, r["modes"]["all"]["deploy"])
            sc = light_risk(dep, coef, prep)
            b = read_asset(root, r["transfer"])
            sel = read_asset(root, r["selection_support"])
            require(
                np.array_equal(sel, e & np.isfinite(sc) & np.isfinite(b)),
                "registered common selection domain changed",
            )
            vals[f, c].append(sc[e & np.isfinite(sc)])
            ap = ranking(sc[sel], t["rgb_high_error"][sel])
        else:
            ap = ranking(np.empty(0, np.float32), np.empty(0, np.uint8))
        views.append({"family": f, "condition": c, "view": v, **ap})
    cutoff = threshold_for({k: np.concatenate(v) for k, v in vals.items()}, den)
    with new_output(out) as dst:
        package = dst / "package"
        shutil.copytree(fit_dir, package)
        m = {
            **model,
            "status": "frozen",
            "model_id": run_id,
            "development_dataset_sha256": ds_sha,
            "fit_model_index_sha256": fit_sha,
            "threshold": cutoff["tau"],
        }
        write_json(package / "model.json", m)
        sha = file_hash(package / "model.json")
        for method in ("MS-L1", "B-L1-transfer"):
            transfer = d["transfer"]
            args = (
                (cutoff["tau"], method, sha, m["weights_sha256"], run_id)
                if method == "MS-L1"
                else (
                    transfer["threshold"],
                    method,
                    transfer["model_index_sha256"],
                    transfer["weights_sha256"],
                    transfer["run"]["run_id"],
                )
            )
            baseline_scores(root, d, ds_sha, coef, prep, *args, dst / method)
        macro, by = balanced({(r["family"], r["condition"], r["view"]): r["ap"] for r in views})
        result = {
            "schema": "el-stage3-light-development-1",
            "run_id": run_id,
            "development_dataset_sha256": ds_sha,
            "fit_index_sha256": fit_sha,
            "model_index_sha256": sha,
            "weights_sha256": m["weights_sha256"],
            "threshold": cutoff,
            "development_common_ap": macro,
            "by_family_condition": by,
            "view_ap": views,
            "transfer": d["transfer"],
            "scores": {k: asset(dst, dst / k / "scores.json") for k in ("MS-L1", "B-L1-transfer")},
        }
        write_json(dst / "development.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("fit", "freeze", "score-test"))
    for k in ("dataset", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--dataset-sha256", required=True)
    p.add_argument("--model", type=Path)
    p.add_argument("--model-sha256")
    p.add_argument("--store", type=Path)
    p.add_argument("--shelf", type=Path)
    p.add_argument("--opening-sha256")
    a = p.parse_args()
    rid = os.environ["EL_RUN_ID"]
    if a.action == "fit":
        result = fit(a.dataset, a.dataset_sha256, a.output, rid)
    elif a.action == "freeze":
        result = freeze(a.dataset, a.dataset_sha256, a.model, a.model_sha256, a.output, rid)
    else:
        d = dataset(a.dataset, a.dataset_sha256, "test", roots=(a.store, a.shelf), opening=a.opening_sha256)
        m, coef, prep = load_light(a.model, a.model_sha256)
        result = {
            "schema": "el-stage3-test-baselines-1",
            "run_id": rid,
            "dataset_sha256": a.dataset_sha256,
            "scores": {},
        }
        from evidence_lens.stage3_gate import require_opening

        mk = require_opening(a.store, CATALOG, opening_sha256=a.opening_sha256, shelf=a.shelf)
        from vision_sources import find_run, json_member

        method = json_member(
            find_run((a.store, a.shelf), mk["method"]), "payload/method-freeze.json", mk["method_sha256"]
        )[0]
        with new_output(a.output) as out:
            for name in ("MS-L1", "B-L1-transfer"):
                entry = method["models"][name]
                expected = (
                    (m["model_id"], a.model_sha256, m["weights_sha256"], m["threshold"])
                    if name == "MS-L1"
                    else (
                        d["transfer"]["run"]["run_id"],
                        d["transfer"]["model_index_sha256"],
                        d["transfer"]["weights_sha256"],
                        d["transfer"]["threshold"],
                    )
                )
                require(
                    tuple(entry[k] for k in ("model_id", "model_index_sha256", "weights_sha256", "threshold"))
                    == expected,
                    "baseline differs from officially opened method",
                )
                baseline_scores(
                    a.dataset,
                    d,
                    a.dataset_sha256,
                    coef,
                    prep,
                    entry["threshold"],
                    name,
                    entry["model_index_sha256"],
                    entry["weights_sha256"],
                    entry["model_id"],
                    out / name,
                )
                result["scores"][name] = asset(out, out / name / "scores.json")
            write_json(out / "baselines.json", result)
    print(json.dumps({"schema": result["schema"]}))


if __name__ == "__main__":
    main()
