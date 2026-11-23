"""Independent held-out B0 RGB-pixel readback using archived predictions and scikit-learn AP."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.metrics import average_precision_score

from evidence_lens._bundle import load, sha
from evidence_lens.formal_inputs import PinnedRun, require
from evidence_lens.scoring_io import write_json

COUNT = ("V", "E", "Y", "S", "SY", "A", "L", "H", "U", "HY", "LY", "UY", "missed_errors")
RATES = {
    "low_coverage": ("L", "V"),
    "low_risk_error": ("LY", "L"),
    "detection": ("HY", "Y"),
    "unknown_fraction": ("U", "E"),
    "common_coverage": ("S", "V"),
}


def mask_sha(mask):
    payload = (
        np.asarray(mask.shape, dtype="<u8").tobytes()
        + np.packbits(mask.reshape(-1), bitorder="little").tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def counts_and_rates(item):
    for metric, (a, b) in RATES.items():
        item[metric] = item[a] / item[b] if item[b] else None
    return item


def per_view(reference, prediction, scores, thresholds):
    require(
        reference.shape == prediction.shape
        and reference.ndim == 3
        and reference.shape[-1] == 3
        and reference.dtype == prediction.dtype == np.float32,
        "RGB grid/dtype mismatch",
    )
    v = np.isfinite(reference).all(-1)
    e = v & np.isfinite(prediction).all(-1)
    # Preserve the registered FP32 per-channel subtraction and mean at the decision threshold.
    y = e & (np.abs(prediction - reference).mean(axis=-1, dtype=np.float32) > np.float32(0.05))
    common = e.copy()
    for val in scores.values():
        require(
            val.shape == e.shape and val.dtype == np.float32 and not np.isinf(val).any(), "score grid/dtype"
        )
        common &= np.isfinite(val)
    labels = y[common]
    prefix = {
        key: int(np.count_nonzero(a))
        for key, a in {"V": v, "E": e, "Y": y, "S": common, "SY": labels}.items()
    }
    hashes = {k + "_mask_sha256": mask_sha(m) for k, m in (("V", v), ("E", e), ("S", common))}
    eligible = bool(labels.size and labels.any() and not labels.all())
    results = []
    for method, raw in scores.items():
        valid = e & np.isfinite(raw)
        high = valid & (raw.astype(np.float64) >= thresholds[method])
        low = valid & ~high
        unknown = e & ~valid
        record = {
            **prefix,
            **{
                k: int(np.count_nonzero(m))
                for k, m in {
                    "A": valid,
                    "L": low,
                    "H": high,
                    "U": unknown,
                    "HY": high & y,
                    "LY": low & y,
                    "UY": unknown & y,
                    "missed_errors": (~high) & y,
                }.items()
            },
        }
        require(
            record["H"] + record["L"] + record["U"] == record["E"]
            and record["HY"] + record["LY"] + record["UY"] == record["Y"]
            and record["LY"] + record["UY"] == record["missed_errors"],
            "non-partitioning archived score states",
        )
        score = raw[common]
        ap = float(average_precision_score(labels, score)) if eligible else None
        results.append(
            {
                "method": method,
                "tau": thresholds[method],
                "common_ap": ap,
                "eligible_cameras": int(eligible),
                "distinct_common_scores": int(np.unique(score).size),
                **hashes,
                **counts_and_rates(record),
            }
        )
    return results


def close(a, b, tolerance=1e-12):
    return (
        a is None and b is None or a is not None and b is not None and abs(float(a) - float(b)) <= tolerance
    )


def group_stats(records):
    sums = {k: sum(r[k] for r in records) for k in COUNT}
    aps = [r["common_ap"] for r in records if r["common_ap"] is not None]
    return {**counts_and_rates(sums), "common_ap": mean(aps) if aps else None, "eligible_cameras": len(aps)}


def readcsv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def savecsv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def build(args):
    cfg = load(args.config)
    plan = load(args.plan)
    require(
        sha(args.plan) == cfg["plan_sha256"] and plan["schema"] == "el-phase1-rgb-regression-plan-1",
        "wrong frozen cohort",
    )
    args.out.mkdir(exist_ok=True)
    require(not any(args.out.iterdir()), "require fresh output")
    methods = plan["methods"]
    sources = {}
    parents = {}

    def pin(spec):
        key = spec["run_id"]
        if key not in parents:
            roots = [base for base in (args.store, args.shelf) if (base / "runs" / key).is_dir()]
            require(roots, "missing selected run")
            parents[key] = PinnedRun(roots[0].resolve(), spec)
        require(
            parents[key].pin["manifest_sha256"] == spec["manifest_sha256"]
            and parents[key].run["execution_status"] == "succeeded",
            "selected run mismatch",
        )
        return parents[key]

    def file(run, name):
        path = run.open(name)
        key = run.pin["run_id"] + "/" + name
        sources[key] = {k: run.files[name][k] for k in ("sha256", "bytes")}
        return path

    threshold = pin(plan["threshold_run"])
    tfile = file(threshold, "payload/model/thresholds.json")
    model = load(file(threshold, "payload/model/model.json"))
    require(
        plan["thresholds_sha256"] == sha(tfile) == model["files"]["thresholds.json"]
        and model["model_id"] == plan["model_id"],
        "threshold/model identity",
    )
    taus = load(tfile)["thresholds"] | {"L0": 0.5}
    prior = np.float32(model["constant_risk"])
    require(set(taus) == set(methods), "five fixed methods required")
    rows = []
    byfam = defaultdict(list)
    for case in plan["cases"]:
        ref, pred, rules, l1 = [
            pin(case[k]) for k in ("reference", "render", "baseline_scores", "learned_scores")
        ]
        refdoc, renderdoc, ruledoc, learned = [
            load(file(r, "payload/" + n))
            for r, n in (
                (ref, "reference.json"),
                (pred, "prediction.json"),
                (rules, "score.json"),
                (l1, "l1_score.json"),
            )
        ]
        require(
            learned["model_id"] == model["model_id"]
            and learned["thresholds"] == {k: v for k, v in taus.items() if k != "L0"}
            and ruledoc["producer_run_id"] == learned["producer_run_id"] == pred.pin["run_id"],
            "archived scores model/prediction mismatch",
        )
        for view in plan["views"]:
            path = "payload/" + view + ".npz"
            rr = "payload/reference/" + view + ".npz"
            for run, rel, expected in (
                (ref, rr, refdoc["views"][view]["reference_sha256"]),
                (pred, path, renderdoc["views"][view]["sha256"]),
                (rules, path, ruledoc["views"][view]["score_sha256"]),
                (l1, path, learned["views"][view]["sha256"]),
            ):
                require(run.files[rel]["sha256"] == expected, "archived camera payload link mismatch")
            with (
                np.load(file(ref, rr), allow_pickle=False) as a,
                np.load(file(pred, path), allow_pickle=False) as b,
                np.load(file(rules, path), allow_pickle=False) as rule,
                np.load(file(l1, path), allow_pickle=False) as learned_scores,
            ):
                reference, prediction, alpha = a["rgb"], b["rgb"], b["alpha"]
                scores = {
                    "constant": np.where(np.isfinite(prediction).all(-1), prior, np.float32(np.nan)),
                    "1-alpha": np.where(
                        np.isfinite(alpha) & (alpha >= 0) & (alpha <= 1), 1 - alpha, np.float32(np.nan)
                    ),
                    "L0": rule["l0"],
                    "L1-full": learned_scores["l1_full"],
                    "L1-geometry": learned_scores["l1_geometry"],
                }
                require(
                    np.array_equal(scores["1-alpha"], rule["alpha_score"], equal_nan=True),
                    "opacity baseline differs from raw alpha",
                )
                for method, key in methods.items():
                    raw = scores[method]
                    ok = np.isfinite(raw)
                    state = np.full(raw.shape, 255, np.uint8)
                    state[ok] = (raw[ok].astype(np.float64) >= taus[method]).astype(np.uint8)
                    require(
                        np.array_equal(raw, learned_scores[key], equal_nan=True)
                        and np.array_equal(ok, learned_scores[key + "_valid"])
                        and np.array_equal(state, learned_scores[key + "_state"]),
                        "original per-pixel score/state disagreement",
                    )
                for result in per_view(reference, prediction, scores, taus):
                    item = {"family": case["family"], "view": view, **result}
                    rows.append(item)
                    byfam[case["family"], item["method"]].append(item)
    families = [{"family": f, "method": m, **group_stats(group)} for (f, m), group in byfam.items()]
    macro = []
    for m in methods:
        group = [r for r in families if r["method"] == m]
        summary = {
            "method": m,
            **{k: sum(r[k] for r in group) for k in COUNT},
            "eligible_cameras": sum(r["eligible_cameras"] for r in group),
        }
        for k in ("common_ap", *RATES):
            vals = [r[k] for r in group if r[k] is not None]
            summary[k] = mean(vals) if vals else None
            summary[k + "_families"] = len(vals)
        macro.append(summary)

    # Read the previous evaluation only after every raw pixel and all new reductions are complete.
    report = pin(plan["report"])
    seal = load(file(report, "payload/public-numerical/selection.json"))
    require(
        seal["method_freeze_sha256"] == plan["method_freeze_sha256"]
        and seal["catalog_sha256"] == plan["catalog_sha256"],
        "wrong original publication",
    )
    matched = {
        (r["scene_family_id"], r["condition_id"]): (r["run_id"], r["manifest_sha256"])
        for r in seal["comparison_trials"]
    }
    group_public = file(report, "payload/public-numerical/family-equal-summary.csv")
    require(
        seal["files"]["family-equal-summary.csv"]["sha256"] == sha(group_public),
        "original public table hash mismatch",
    )
    lookup = {
        (r["method"], r["metric"]): r
        for r in readcsv(group_public)
        if r["condition"] == "B0" and r["region"] == "all"
    }
    worst = 0.0
    for c in plan["cases"]:
        fr = c["family"]
        comparison = pin(c["comparison"])
        require(
            matched[fr, "B0"] == (comparison.pin["run_id"], comparison.pin["manifest_sha256"]),
            "comparison mismatch",
        )
        old = load(file(comparison, "payload/formal_evaluation.json"))
        require(
            old["thresholds"] == taus
            and old["model_id"] == model["model_id"]
            and old["scene_family_id"] == fr,
            "wrong original thresholds",
        )
        for row in (r for r in rows if r["family"] == fr):
            region = old["views"][row["view"]]["regions"]["all"]
            meth = region["methods"][row["method"]]
            require(
                all(row[k] == (region["counts"] | meth["counts"])[k] for k in COUNT), "camera count mismatch"
            )
            require(
                all(row[k] == region[k] for k in ("S_mask_sha256", "V_mask_sha256", "E_mask_sha256")),
                "camera mask membership mismatch",
            )
            orig = meth["common_ranking"]
            require(
                row["distinct_common_scores"] == orig["unique_score_count"]
                and close(row["common_ap"], orig["ap"]),
                "camera AP or tie group mismatch",
            )
            if row["common_ap"] is not None:
                worst = max(worst, abs(row["common_ap"] - orig["ap"]))
        for row in (r for r in families if r["family"] == fr):
            g = old["family"]["all"]
            m = g["methods"][row["method"]]
            require(
                all(row[k] == (g["counts"] | m["counts"])[k] for k in COUNT)
                and close(row["common_ap"], m["common_ranking"]["ap"])
                and row["eligible_cameras"] == m["common_ranking"]["eligible_view_count"],
                "family AP/count mismatch",
            )
            for k in RATES:
                metric = (m["system"] | g["outcomes"])[k]
                require(close(row[k], metric["value"]), "frozen family operating metric")
    for r in macro:
        for k in ("common_ap", *RATES):
            archived = lookup[r["method"], k]
            expected = float(archived["mean"]) if archived["mean"] else None
            require(
                close(r[k], expected) and r[k + "_families"] == int(archived["eligible_family_count"]),
                "frozen macro AP/rate disagreement",
            )
    savecsv(args.out / "camera-methods.csv", rows)
    savecsv(args.out / "family-methods.csv", families)
    savecsv(args.out / "headline.csv", macro)
    summary = {
        "schema": "el-phase1-rgb-regression-1",
        "run_id": args.run_id,
        "plan_sha256": sha(args.plan),
        "source_report": plan["report"],
        "threshold_run": plan["threshold_run"],
        "tau": taus,
        "methods": len(methods),
        "families": len(plan["cases"]),
        "targets": len(rows) // len(methods),
        "camera_method_rows": len(rows),
        "family_method_rows": len(families),
        "compared_camera_AP_max_abs": worst,
        "macro": macro,
        "sklearn": version("scikit-learn"),
        "numpy": version("numpy"),
        "status": "matched",
    }
    write_json(args.out / "summary.json", summary)
    write_json(
        args.out / "inputs.json",
        {
            "parents": {rid: r.pin["manifest_sha256"] for rid, r in parents.items()},
            "members": sources,
            "ledger": "Original report and comparisons consumed after raw reductions.",
        },
    )
    lines = [
        "# Original B0 RGB headline regression",
        "",
        "All four original held-out families and six target cameras per family. AP uses scikit-learn on the original scores of the common finite mask with whole ties. Within each family, AP averages eligible target views. Coverage and accepted error pool their own numerators and denominators over its six views. The final percentages and AP give eligible families equal weight.",
        "",
        "| Method | Common AP | Coverage L/V | Accepted errors LY/L | H | L | U | HY | LY | UY | Missed LY + UY |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in macro:
        ratios = [
            "—" if r[k] is None else (f"{r[k]:.6f}" if k == "common_ap" else f"{r[k]:.4%}")
            for k in ("common_ap", "low_coverage", "low_risk_error")
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    r["method"],
                    *ratios,
                    *[f"{r[k]:,}" for k in ("H", "L", "U", "HY", "LY", "UY", "missed_errors")],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The counts above are sums over all 24 camera grids. [Per-camera counts and masks](camera-methods.csv) · [Per-family results](family-methods.csv) · [Exact headline values](headline.csv) · [Original asset digests](inputs.json)",
        ]
    )
    (args.out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "run_id": args.run_id,
        "targets": summary["targets"],
        "camera_method_rows": len(rows),
        "max_ap_abs": worst,
        "status": "matched",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("config", "plan", "store", "shelf", "out"):
        parser.add_argument("--" + n, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    print(json.dumps(build(parser.parse_args()), allow_nan=False))
