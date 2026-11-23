"""Score the two authorized original development cases with a single frozen package. No label/dataset input."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from evidence_lens.multisource import CONDITIONS, METHODS, SCOPES, TARGETS, source_ids
from evidence_lens.multisource_io import Case, require
from evidence_lens.l1 import risk_state
from evidence_lens.scoring_io import file_hash, new_output, write_json
from stage3_risk_sources import asset, CATALOG


def native(cases, package, model_sha, weight_sha, model_id, method, tau, python, output, family):
    require(
        family.startswith("scene_057_") and method in METHODS, "unknown fixed development example or method"
    )
    index = json.loads((package / "model.json").read_bytes())
    require(
        file_hash(package / "model.json") == model_sha and index["model_id"] == model_id,
        "native frozen model identity mismatch",
    )
    paths = {
        "B-L1-transfer": "L1-full.json",
        "MS-L1": "coefficients.json",
        "MS-frozen": "weights.safetensors",
        "MS-ft": "weights.safetensors",
    }
    require(file_hash(package / paths[method]) == weight_sha, "native frozen parameter identity mismatch")
    if method != "B-L1-transfer":
        require(index["threshold"] == tau, "original model threshold mismatch")
    rows = []
    measured = []
    with new_output(output) as out:
        for condition in CONDITIONS:
            path, sha = cases[condition]
            case = Case(path, sha)
            require(case.condition == condition, "swapped native development reconstruction")
            for view in TARGETS:
                failed = view in case.failed
                row = {
                    "condition": condition,
                    "view": view,
                    "case_sha256": case.sha256,
                    "status": "render_failed" if failed else "ready",
                    "modes": {},
                }
                for scope in SCOPES:
                    alias = "two" if (condition, scope) == ("M2", "all") else None
                    inf = {"source_ids": list(source_ids(condition, scope)), "alias_of": alias, "file": None}
                    row["modes"][scope] = inf
                    if failed:
                        continue
                    if alias:
                        inf.update(file=row["modes"][alias]["file"], alias_measurement=alias)
                        continue
                    stage = out / condition / view / scope
                    stage.parent.mkdir(parents=True, exist_ok=True)
                    if method in ("MS-frozen", "MS-ft"):
                        command = [
                            str(python),
                            "-I",
                            "-B",
                            str(package / "score-native.py"),
                            "--threads",
                            "4",
                            "--model-sha256",
                            model_sha,
                            "--case",
                            str(path),
                            "--case-sha256",
                            sha,
                            "--view",
                            view,
                            "--scope",
                            scope,
                            "--output",
                            str(stage),
                        ]
                        receipt = "native-score.json"
                    else:
                        command = [
                            str(python),
                            "-B",
                            "-m",
                            "evidence_lens.multisource_model",
                            "--method",
                            method,
                            "--case",
                            str(path),
                            "--case-sha256",
                            sha,
                            "--package",
                            str(package),
                            "--model-sha256",
                            model_sha,
                            "--weights-sha256",
                            weight_sha,
                            "--tau",
                            str(tau),
                            "--view",
                            view,
                            "--scope",
                            scope,
                            "--output",
                            str(stage),
                        ]
                        receipt = "score.json"
                    t = time.perf_counter()
                    done = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env={
                            **os.environ,
                            "PYTHONDONTWRITEBYTECODE": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                            "OMP_NUM_THREADS": "4",
                        },
                    )
                    wall = time.perf_counter() - t
                    require(
                        done.returncode == 0,
                        "native CLI failed for "
                        + f"{method}/{condition}/{view}/{scope}: "
                        + done.stderr[-1800:],
                    )
                    rec = json.loads((stage / receipt).read_bytes())
                    spec = asset(out, stage / "scores.npz")
                    require(
                        rec["case_sha256"] == sha
                        and rec["method"] == method
                        and rec["view"] == view
                        and rec["scope"] == scope
                        and rec["model_sha256"] == model_sha,
                        "native output identifies a different method or source case",
                    )
                    with np.load(stage / "scores.npz", allow_pickle=False) as z:
                        require(
                            set(z) == {"risk", "state", "valid"}
                            and np.array_equal(risk_state(z["risk"], tau), z["state"])
                            and np.array_equal(np.isfinite(z["risk"]), z["valid"]),
                            "native output states mismatch",
                        )
                    inf.update(
                        file=spec, receipt=asset(out, stage / receipt), external_process_wall_seconds=wall
                    )
                    measured.append(wall)
                rows.append(row)
        summary = {
            "schema": "el-stage3-native-example-scores-1",
            "family": family,
            "catalog_sha256": CATALOG,
            "model_id": model_id,
            "model_index_sha256": model_sha,
            "weights_sha256": weight_sha,
            "method": method,
            "threshold": tau,
            "views": rows,
            "timing": {
                "kind": "separate process per nonredundant target and subset; includes interpreter startup, model and deployable inputs, complete output and process exit",
                "measured": len(measured),
                "scheduled": 30,
                "total_seconds": sum(measured),
                "p50_seconds": float(np.median(measured)) if measured else None,
                "p95_seconds": float(np.percentile(measured, 95)) if measured else None,
            },
        }
        write_json(out / "native-scores.json", summary)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("case-m2", "case-m4", "package", "python", "output"):
        ap.add_argument("--" + key, type=Path, required=True)
    for key in (
        "case-m2-sha256",
        "case-m4-sha256",
        "model-sha256",
        "weights-sha256",
        "model-id",
        "method",
        "family",
    ):
        ap.add_argument("--" + key, required=True)
    ap.add_argument("--threshold", type=float, required=True)
    a = ap.parse_args()
    s = native(
        {"M2": (a.case_m2, a.case_m2_sha256), "M4": (a.case_m4, a.case_m4_sha256)},
        a.package,
        a.model_sha256,
        a.weights_sha256,
        a.model_id,
        a.method,
        a.threshold,
        a.python,
        a.output,
        a.family,
    )
    print(json.dumps(s["timing"]))
