"""Write an object-level T0 report from the independent evaluator's numerical tables."""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean

from evidence_lens._bundle import load
from dtu_score import METHODS


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(r, key, percent=False):
    if r.get(key) in ("", None):
        return "N/A"
    return f"{100 * float(r[key]):.2f}%" if percent else f"{float(r[key]):.4f}"


def product(src, roster, out):
    ev = load(src / "evaluation.json")
    objects = ev["objects"]
    conditions = read(src / "method-conditions.csv")
    equal = read(src / "object-equal.csv")
    matched = read(src / "matched-conditions.csv")
    per_source = read(src / "source-fit.csv")
    meta = {r["object_id"]: r for r in roster["objects"]}
    lines = [
        "# Frozen evaluation on six new DTU subjects"
        if len(objects) == 6
        else "# Frozen evaluation on new DTU subjects",
        "",
        f"The collective opening covered {ev['scored_entries']} scored entries out of {ev['scheduled_entries']} scheduled. Each physical subject contributes two conditions and six target views per condition. Ground truth is the full centre-cropped photograph at the registered camera. A pixel is an error when its mean absolute display-RGB difference exceeds 0.05.",
        "",
        "## Reconstruction and the released MS-L1 operating point",
        "",
        "| Object | Sources | Source MAE | Target MAE | RGB errors / E | Low / V | Errors / low | Unknown / E |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for obj in objects:
        for cond in ("M2", "M4"):
            (r,) = [
                x
                for x in conditions
                if x["object_id"] == obj and x["condition"] == cond and x["method"] == "MS-L1"
            ]
            ss = [
                x for x in per_source if x["object_id"] == obj and x["condition"] == cond and x["source_MAE"]
            ]
            sm = mean([float(x["source_MAE"]) for x in ss]) if ss else None
            lines.append(
                f"| {obj} ({meta[obj]['primary_subject']}) | {cond} | {num({'v': sm}, 'v')} | {num(r, 'MAE')} | {r['Y']}/{r['E']} | {r['L']}/{r['V']} | {r['LY']}/{r['L']} | {r['U']}/{r['E']} |"
            )
    lines += [
        "",
        "## Four frozen methods on identical current candidates",
        "",
        "Each subject receives equal weight. AP ranks RGB errors, with prevalence averaged over identical defined views. At 20% coverage each method accepts the same per-view quota within `within_new`. The last two columns use each released threshold on its own complete application domain.",
        "",
        "| Sources | Method | Common AP | Same-domain prevalence | Error at 20% | Low / V | Errors / low |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for cond in ("M2", "M4"):
        for m in METHODS:
            lookup = {r["metric"]: r for r in equal if (r["condition"], r["method"]) == (cond, m)}
            vals = [
                num(lookup[k], "object_equal_mean", i > 0)
                for i, k in enumerate(
                    ("common_AP", "common_prevalence", "within_new_risk_20%_V", "coverage", "low_error")
                )
            ]
            lines.append("| " + " | ".join([cond, m, *vals]) + " |")
    lines += [
        "",
        "## Counts at matched 20% coverage",
        "",
        "Each cell gives errors among accepted pixels at the same per-view quota. The candidate population, its label counts, attainable views, cutoff tie sizes and tie bounds for every coverage are in the per-view table.",
        "",
        "| Subject | Sources | " + " | ".join(METHODS) + " |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for obj in objects:
        for cond in ("M2", "M4"):
            selected = {
                x["method"]: x
                for x in matched
                if (x["object_id"], x["condition"], x["anchor"]) == (obj, cond, "20%_V")
            }
            lines.append(
                "| "
                + " | ".join(
                    [
                        obj,
                        cond,
                        *[
                            f"{selected[m]['accepted_positive']}/{selected[m]['accepted']}"
                            if int(selected[m]["accepted"])
                            else "N/A"
                            for m in METHODS
                        ],
                    ]
                )
                + " |"
            )
    lines += [
        "",
        "These six physical subjects share the calibrated rig, lighting, crop and camera layout. They provide no measurement on a new rig or an arbitrary reconstruction asset. No independently verified object mask was available. Original thresholds, weights and score support were fixed before access to target photographs. Both the change from authored scenes to real photographs and the change in reconstruction recipe affect transfer.",
        "",
        "[Per-view counts and mask hashes](method-views.csv) · [Per-object results](method-conditions.csv) · [All matched quotas](matched-conditions.csv) · [Per-view cutoff and tie details](matched-views.csv) · [Object-equal results](object-equal.csv) · [Unknown reasons](unknown-reasons.csv) · [Source fits](source-fit.csv) · [Scheduled and successful entries](system.csv) · [Evaluation and reference-access receipt](evaluation.json)",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tables", type=Path, required=True)
    p.add_argument("--roster", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    product(a.tables, load(a.roster), a.out)
