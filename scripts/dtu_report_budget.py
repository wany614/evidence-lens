"""Export the original set1 paired-budget receipt as tables, alongside the frozen four-cell report."""

import argparse
import json
import os
from pathlib import Path

from dtu_eval import read_pinned
from dtu_report import csv_file
from dtu_score import VIEWS
from evidence_lens.formal_inputs import require
from evidence_lens.scoring_io import file_hash, new_output, write_json
from vision_sources import signed


def export(roots, request, out, rid):
    require(
        request["schema"] == "el-dtu-budget-appendix-request-1"
        and set(request["main_evaluations"]) == {"M2", "M4"},
        "missing original set1 endpoint pins",
    )
    _, raw = read_pinned(roots, request["diagnostic"])
    source = json.loads(raw.read_bytes())
    require(
        source["schema"] == "el-dtu-paired-set1-budget-diagnostic-1"
        and source["object_id"] == "set1"
        and source["models_scored"] == 0
        and set(source["conditions"]) == {"M2", "M4"},
        "wrong original reconstruction diagnostic",
    )
    targets = []
    sources = []
    endpoints = []
    for cond in ("M2", "M4"):
        _, path = read_pinned(roots, request["main_evaluations"][cond])
        main = json.loads(path.read_bytes())
        m = source["conditions"][cond]["endpoints"]
        require(set(m) == {"256", "1024"}, "incomplete registered budget pair")
        require(
            main["schema"] == "el-dtu-condition-evaluation-1"
            and main["role"] == "debug"
            and main["object_id"] == "set1"
            and main["condition"] == cond
            and main["arm"] == "main-256"
            and main["input_pins"]["references"]["sha256"] == source["reference_sha256"]
            and main["identity"]["crop_sha256"] == source["crop_sha256"],
            "wrong frozen main evaluation or reference",
        )
        require(
            m["256"]["same_initial_parameter_sha256"]
            == m["1024"]["same_initial_parameter_sha256"]
            == main["reconstruction_readback"]["initial_parameter_sha256"],
            "different original initialization",
        )
        for step in ("256", "1024"):
            z = m[step]
            require(
                z["updates"] == int(step)
                and z["gaussians"] == 4096
                and set(z["target_views"]) == set(VIEWS)
                and z["GPU_probe"]["passed"] is True,
                "invalid registered endpoint",
            )
            totals = {k: 0 for k in ("V", "E", "Y")}
            mae_sum = 0.0
            for i, v in enumerate(VIEWS):
                r = z["target_views"][v]
                old = main["rows"][i]
                require(
                    old["view"] == v and old["reference_sha256"] == r["target_rgb_sha256"],
                    "different photographic identity",
                )
                if step == "256":
                    native = old["domains"]["full_crop"]["MS-L1"]["counts"]
                    require(
                        old["prediction_sha256"] == r["prediction_sha256"]
                        and all(native[k] == r[k] for k in totals)
                        and native["rgb_mae_sum"] == r["rgb_absolute_error_sum"],
                        "independent main view differs",
                    )
                for k in totals:
                    totals[k] += r[k]
                mae_sum += r["rgb_absolute_error_sum"]
                targets.append(
                    {
                        "object_id": "set1",
                        "condition": cond,
                        "steps": int(step),
                        "view": v,
                        "official_view": old["official_view_id"],
                        **r,
                        "Y_over_E": r["Y"] / r["E"] if r["E"] else None,
                    }
                )
            p = z["target_pooled"]
            require(
                all(p[k] == n for k, n in totals.items())
                and (p["MAE"] is None if not p["E"] else abs(mae_sum / p["E"] - p["MAE"]) <= 1e-14),
                "wrong totals in original diagnostic",
            )
            endpoints.append(
                {
                    "object_id": "set1",
                    "condition": cond,
                    "steps": int(step),
                    "gaussians": z["gaussians"],
                    "sampled_source_pixels": z["source_pixels"],
                    "source_mean_MAE": z["source_mean_MAE"],
                    "update_seconds": z["optimization_s"],
                    **p,
                    "generation_run_id": z["generation"]["run_id"],
                    "generation_manifest_sha256": z["generation"]["manifest_sha256"],
                    "render_run_id": z["render"]["run_id"],
                    "render_manifest_sha256": z["render"]["manifest_sha256"],
                }
            )
            sources.extend(
                {
                    "object_id": "set1",
                    "condition": cond,
                    "steps": int(step),
                    "source": s,
                    "sampled_pixels": r["pixels"],
                    "source_updates": r["samples"],
                    "initial_MAE": r["initial"],
                    "final_MAE": r["mae"],
                }
                for s, r in sorted(z["sources"].items())
            )
    with new_output(out) as dest:
        csv_file(dest / "target-budget.csv", targets)
        csv_file(dest / "source-budget.csv", sources)
        csv_file(dest / "endpoints.csv", endpoints)
        lines = [
            "# Development-only reconstruction budget",
            "",
            "Each endpoint uses DTU set1, the same source photographs, calibration, starting parameters and 4,096 Gaussians. The main and diagnostic endpoints ran separately. The 1024-step arm measured source and target RGB error. It added zero risk scores.",
            "",
            "| Condition | Updates | Mean source MAE | Target MAE | Target errors / E |",
            "|---|---:|---:|---:|---:|",
        ]
        for a in endpoints:
            lines.append(
                f"| {a['condition']} | {a['steps']} | {a['source_mean_MAE']:.4f} | {a['MAE']:.4f} | {a['Y']:,} / {a['E']:,} ({100 * a['Y_over_E']:.2f}%) |"
            )
        lines += [
            "",
            "The [four endpoint totals](endpoints.csv), [individual targets](target-budget.csv) and [source views](source-budget.csv) preserve the original counts and timing. Each 256-step target also agrees with the frozen independent RGB evaluation on reference and prediction identity, V, E, Y and the sum of absolute mean RGB error.",
        ]
        (dest / "budget.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        files = {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in sorted(dest.iterdir())}
        result = {
            "schema": "el-dtu-development-budget-appendix-1",
            "run_id": rid,
            "diagnostic": request["diagnostic"],
            "main_evaluations": request["main_evaluations"],
            "models_scored": 0,
            "frozen_main_views_checked": 12,
            "files": files,
        }
        write_json(dest / "appendix.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("store", "shelf", "request", "output"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--request-sha256", required=True)
    a = p.parse_args()
    print(
        json.dumps(
            export(
                (a.store, a.shelf), signed(a.request, a.request_sha256), a.output, os.environ.get("EL_RUN_ID")
            ),
            sort_keys=True,
        )
    )
