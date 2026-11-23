"""Recompute original-development operating points and the preregistered paired-seed default decision."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np

from evidence_lens.formal_inputs import require
from evidence_lens.multisource import CONDITIONS, METHODS, TARGETS
from evidence_lens.scoring_io import new_output, write_json
from stage3_risk_data import dataset, read_asset
from stage3_risk_evaluate import measurement, pooled, score_indices, view_scores
from stage3_risk_learning import mean_defined, threshold_for
from stage3_risk_sources import CATALOG, STUDY, asset
from vision_sources import signed

SEEDS = (1729, 2718)


def promoted(rows, rule):
    require(
        rule["initial_default"] == "MS-L1"
        and rule["candidate"] == "MS-ft"
        and rule["paired_seeds"] == list(SEEDS)
        and rule["scope"] == "all",
        "unsupported promotion protocol",
    )
    comparisons = {}
    checks = {}
    for opponent in rule["competitors"]:
        gaps = []
        for seed in SEEDS:
            ft = rows[str(seed)]["MS-ft"]
            base = rows[str(seed)][opponent]
            valid = ft["complete"] and base["complete"] and ft["ap"] is not None and base["ap"] is not None
            gaps.append(ft["ap"] - base["ap"] if valid else None)
        defined = all(x is not None for x in gaps)
        comparisons[opponent] = {
            "paired_seed_differences": gaps,
            "mean_difference": mean_defined(gaps) if defined else None,
        }
        checks[opponent + "_AP"] = bool(
            defined and min(gaps) > 0 and np.mean(gaps) >= rule["minimum_mean_ap_gain_against_each"]
        )
    for seed in SEEDS:
        ft = rows[str(seed)]["MS-ft"]
        base = rows[str(seed)]["MS-L1"]
        valid = (
            ft["complete"]
            and base["complete"]
            and all(r[k] is not None for r in (ft, base) for k in ("low_coverage", "low_error"))
        )
        checks[str(seed) + "_coverage"] = bool(
            valid and ft["low_coverage"] >= base["low_coverage"] - rule["maximum_low_coverage_loss_vs_light"]
        )
        checks[str(seed) + "_low_error"] = bool(
            valid and ft["low_error"] <= base["low_error"] + rule["maximum_low_error_increase_vs_light"]
        )
    selected = "MS-ft" if all(checks.values()) else "MS-L1"
    return {
        "default": selected,
        "release_seed": rule["primary_weight_seed"],
        "checks": checks,
        "comparisons": comparisons,
        "decision": "all preregistered comparisons passed"
        if selected == "MS-ft"
        else "at least one preregistered comparison was absent or below its limit",
    }


def development_grade(root, ds_sha, pair_rosters, studyfile, out, run_id):
    protocol = signed(studyfile, STUDY)
    d = dataset(root, ds_sha, "development")
    require(
        len(d["families"]) == 4 and set(pair_rosters) == set(map(str, SEEDS)),
        "wrong development or seed cohort",
    )
    summaries = {}
    details = {}
    hashes = {}
    for seed in SEEDS:
        roster = pair_rosters[str(seed)]
        idx = score_indices(roster, d, ds_sha)
        tau = {m: j["threshold"] for m, (_, j) in idx.items()}
        published = {}
        for m in METHODS[2:]:
            spec = roster[m]["reported_development"]
            reports = signed(Path(spec["path"]), spec["sha256"])
            require(len(reports["views"]) == len(d["views"]), "incomplete producer dev ledger")
            published[m] = reports
        value = defaultdict(list)
        den = defaultdict(int)
        rankings = defaultdict(list)
        for i, r in enumerate(d["views"]):
            f, c, v = r["family"], r["condition"], r["view"]
            t = read_asset(root, r["truth"])
            dep = read_asset(root, r["modes"]["all"]["deploy"]) if r["status"] == "ready" else None
            scores = view_scores(root, r, idx, i, "all", t, dep)
            den[f, c] += int(t["reference_valid"].sum())
            e = t["rgb_evaluable"]
            cm = e & np.logical_and.reduce([np.isfinite(x) for x in scores.values()])
            if r["status"] == "ready":
                require(
                    np.array_equal(cm, read_asset(root, r["selection_support"])),
                    "independent four-method dev domain differs",
                )
                h = r["selection_support"]["sha256"]
                require(hashes.setdefault((f, c, v), h) == h, "two seeds use different approved domains")
            s = measurement(t, scores, tau, np.ones(e.shape, bool))
            for m, score in scores.items():
                value[m, f, c].append(score[e & np.isfinite(score)])
                rankings[m, f, c].append(s[m])
                if m in published:
                    raw = published[m]["views"][i]
                    native = s[m]["ranking"]["common"]
                    v_ap, producer_ap = native["ap"], raw["ap"]
                    require(
                        (raw["family"], raw["condition"], raw["view"]) == (f, c, v)
                        and raw["n"] == native["n"]
                        and (
                            (v_ap is None and producer_ap is None)
                            or (
                                v_ap is not None
                                and producer_ap is not None
                                and abs(v_ap - producer_ap) <= 1e-12
                            )
                        ),
                        "independent AP or selection sample size differs from the actual training producer",
                    )
        by = {}
        summaries[str(seed)] = {}
        for m in METHODS:
            lists = {(f, c): np.concatenate(value[m, f, c]) for f, c in den}
            if m == "B-L1-transfer":
                cutoff = {
                    "tau": protocol["stage_b"]["transfer_threshold"],
                    "source": "unchanged Stage B cutoff",
                }
            else:
                cutoff = threshold_for(lists, den)
            require(
                cutoff["tau"] == tau[m], "frozen producer threshold differs from native score groups: " + m
            )
            cells = {f: {c: pooled(rankings[m, f, c]) for c in CONDITIONS} for f in d["families"]}

            def macro(key):
                def metric(q):
                    return q["ranking"]["common"]["ap"] if key == "ap" else q["rates"][key]

                return mean_defined(
                    [mean_defined([metric(q) for q in cells[f].values()]) for f in d["families"]]
                )

            all_defined = all(
                q["ranking"]["common"]["defined_views"]
                and q["rates"]["low_coverage"] is not None
                and q["rates"]["low_error"] is not None
                for f in cells.values()
                for q in f.values()
            )
            a = {
                "ap": macro("ap"),
                "low_coverage": macro("low_coverage"),
                "low_error": macro("low_error"),
                "complete": bool(all_defined),
                "defined_two_class_views": sum(
                    q["ranking"]["common"]["defined_views"] for f in cells.values() for q in f.values()
                ),
                "planned_target_views": len(d["views"]),
                "defined_coverage_cells": sum(
                    q["rates"]["low_coverage"] is not None for f in cells.values() for q in f.values()
                ),
            }
            if m in published:
                require(
                    a["ap"] is not None
                    and abs(a["ap"] - published[m]["family_equal_ap"]) <= 1e-12
                    and abs(cutoff["coverage"] - published[m]["threshold"]["coverage"]) <= 1e-12,
                    "producer and independent development macro AP or coverage differ",
                )
            by[m] = {
                "family_condition": cells,
                "threshold": cutoff,
                "source": {
                    k: roster[m][k]
                    for k in ("model_id", "model_index_sha256", "weights_sha256", "threshold", "sha256")
                },
            }
            summaries[str(seed)][m] = a
        details[str(seed)] = by
    for m in ("B-L1-transfer", "MS-L1"):
        require(
            details["1729"][m] == details["2718"][m] and summaries["1729"][m] == summaries["2718"][m],
            "common baseline changed between paired seeds",
        )
    decision = promoted(summaries, protocol["promotion"])
    thresholds = {m: details["1729"][m]["threshold"] for m in METHODS}
    with new_output(out) as dest:
        write_json(
            dest / "thresholds.json",
            {
                "schema": "el-stage3-threshold-recalculation-1",
                "catalog_sha256": CATALOG,
                "dataset_sha256": ds_sha,
                "methods": thresholds,
                "all_seed_models": {
                    s: {m: x["source"] for m, x in rows.items()} for s, rows in details.items()
                },
            },
        )
        write_json(
            dest / "development-recomputation.json",
            {
                "schema": "el-stage3-development-recomputation-1",
                "views": len(d["views"]),
                "summaries": summaries,
                "family_condition": details,
                "common_mask_sha256": [
                    {"family": f, "condition": c, "view": v, "sha256": sha}
                    for (f, c, v), sha in hashes.items()
                ],
            },
        )
        result = {
            "schema": "el-stage3-neural-selection-1",
            "run_id": run_id,
            "catalog_sha256": CATALOG,
            "study_sha256": STUDY,
            "development_dataset_sha256": ds_sha,
            "cohort": "development",
            "methods": list(METHODS),
            "models": {m: details["1729"][m]["source"] for m in METHODS},
            "decision": decision,
            "summaries": summaries,
            "files": {
                n: asset(dest, dest / n) for n in ("thresholds.json", "development-recomputation.json")
            },
        }
        write_json(dest / "selection.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("dataset", "roster", "study", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("dataset", "roster"):
        p.add_argument("--" + k + "-sha256", required=True)
    a = p.parse_args()
    sel = development_grade(
        a.dataset,
        a.dataset_sha256,
        signed(a.roster, a.roster_sha256),
        a.study,
        a.output,
        os.environ["EL_RUN_ID"],
    )
    print(json.dumps(sel["decision"]))
