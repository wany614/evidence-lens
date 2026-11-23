"""Compare original producer FP32 risk maps to independently run native CPU maps at the same frozen threshold."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from evidence_lens.formal_inputs import require
from evidence_lens.l1 import risk_state
from evidence_lens.multisource import METHODS, SCOPES, TARGETS, CONDITIONS
from evidence_lens.scoring_io import new_output, write_json
from stage3_risk_data import dataset, read_asset
from stage3_risk_evaluate import score_indices
from stage3_risk_sources import CATALOG, STUDY
from vision_sources import signed

PROTOCOL_SHA = "52eba842a518fb91ff87352cb4d812fa37623675af38b41e12950911657c1cd0"
TOL = 1e-4


def compare(a, b, tau):
    for value in (a, b):
        score = value["risk"]
        require(
            score.dtype == np.float32
            and value["valid"].dtype == np.bool_
            and value["state"].dtype == np.uint8
            and np.array_equal(np.isfinite(score), value["valid"])
            and not np.isinf(score).any()
            and np.all((score[value["valid"]] >= 0) & (score[value["valid"]] <= 1))
            and np.array_equal(value["state"], risk_state(score, tau)),
            "malformed scored risk/valid/state",
        )
    require(a["risk"].shape == b["risk"].shape, "cross-platform score shape mismatch")
    av, bv = a["valid"], b["valid"]
    both = av & bv
    sa, sb = a["state"], b["state"]
    x, y = a["risk"][both], b["risk"][both]
    dx = x.astype(np.float64)
    dy = y.astype(np.float64)
    delta = np.abs(dx - dy)
    bits1 = x.view(np.uint32).astype(np.int64)
    bits2 = y.view(np.uint32).astype(np.int64)
    crossed = (dx < tau) != (dy < tau)
    support = int((av != bv).sum())
    mismatch = int((sa != sb).sum())
    finite = len(x)
    bit_mismatch = int((bits1 != bits2).sum())
    maxabs = float(delta.max()) if finite else None
    near = {}
    for eps in (1e-6, 1e-5, 1e-4):
        near[format(eps, ".0e")] = {
            "producer": int((np.abs(dx - tau) <= eps).sum()),
            "cpu": int((np.abs(dy - tau) <= eps).sum()),
        }
    return {
        "pixels": int(av.size),
        "producer_valid": int(av.sum()),
        "cpu_valid": int(bv.sum()),
        "common_finite": finite,
        "support_disagreement": support,
        "support_exact": not support,
        "state_disagreement": mismatch,
        "state_exact": not mismatch,
        "bitwise_mismatches_on_common": bit_mismatch,
        "bitwise_equal_on_common": finite - bit_mismatch,
        "bitwise_and_support_exact": bool(finite and not support and not bit_mismatch),
        "producer_low_to_cpu_high": int(((sa == 0) & (sb == 1)).sum()),
        "producer_high_to_cpu_low": int(((sa == 1) & (sb == 0)).sum()),
        "threshold_crossings": int(crossed.sum()),
        "producer_finite_to_unknown": int((av & ~bv).sum()),
        "producer_unknown_to_finite": int((~av & bv).sum()),
        "near_threshold_on_common_finite": near,
        "farthest_crossing_distance_producer": float(np.abs(dx[crossed] - tau).max())
        if crossed.any()
        else None,
        "farthest_crossing_distance_cpu": float(np.abs(dy[crossed] - tau).max()) if crossed.any() else None,
        "max_abs": maxabs,
        "mean_abs": float(delta.mean()) if finite else None,
        "abs_percentiles_50_95_99": np.percentile(delta, [50, 95, 99]).tolist() if finite else None,
        "max_float32_ulp": int(np.abs(bits1 - bits2).max()) if finite else None,
        "bounded_numeric_pass": bool(finite and not support and maxabs <= TOL),
    }


def grade(root, ds_sha, original, cpu, protocol, output, run_id):
    study = signed(protocol, PROTOCOL_SHA)
    require(study["catalog_sha256"] == CATALOG and study["study_sha256"] == STUDY, "parity authority differs")
    d = dataset(root, ds_sha, "development")
    f = d["families"][0]
    require(f.startswith("scene_057_"), "wrong preregistered first development family")
    producers = score_indices(original, d, ds_sha)
    require(set(cpu) == set(METHODS), "four native CPU methods required")
    results = []
    for m in METHODS:
        path = Path(cpu[m]["directory"])
        j = signed(path / "native-scores.json", cpu[m]["sha256"])
        require(
            j["schema"] == "el-stage3-native-example-scores-1"
            and j["method"] == m
            and j["family"] == f
            and all(
                j[k] == original[m][k]
                for k in ("model_id", "model_index_sha256", "weights_sha256", "threshold")
            )
            and j["catalog_sha256"] == CATALOG,
            "Mac native model/cutoff differs from original",
        )
        ip, orig = producers[m]
        expected = {(c, v) for c in CONDITIONS for v in TARGETS}
        by = {(r["condition"], r["view"]): r for r in j["views"]}
        require(
            len(j["views"]) == len(by) == 12 and set(by) == expected, "incomplete native case target records"
        )
        for i, r in enumerate(d["views"]):
            if r["family"] != f:
                continue
            c, v = r["condition"], r["view"]
            n = by[c, v]
            require(
                n["case_sha256"] == (r["case"]["sha256"] if r["case"] else None)
                and n["status"] == r["status"],
                "native source case changed",
            )
            for scope in SCOPES:
                a = orig["views"][i]["modes"][scope]
                b = n["modes"][scope]
                require(
                    a["source_ids"] == b["source_ids"] and a["alias_of"] == b["alias_of"],
                    "CPU source subset changed",
                )
                if a["alias_of"]:
                    require(n["modes"][a["alias_of"]]["file"] == b["file"], "CPU alias output changed")
                record = {
                    "family": f,
                    "condition": c,
                    "view": v,
                    "method": m,
                    "scope": scope,
                    "alias_of": a["alias_of"],
                    "status": r["status"],
                    "producer": a["file"],
                    "cpu": b["file"],
                    "threshold": orig["threshold"],
                }
                if a["file"]:
                    record["comparison"] = compare(
                        read_asset(ip, a["file"]), read_asset(path, b["file"]), orig["threshold"]
                    )
                else:
                    require(b["file"] is None, "observed native view for a failed producer")
                    record["comparison"] = None
                results.append(record)
    canonical = [r["comparison"] for r in results if r["alias_of"] is None and r["comparison"] is not None]
    out = {
        "schema": "el-stage3-native-risk-parity-1",
        "run_id": run_id,
        "catalog_sha256": CATALOG,
        "study_sha256": STUDY,
        "family": f,
        "dataset_sha256": ds_sha,
        "protocol_sha256": PROTOCOL_SHA,
        "relative_tolerance": 0,
        "absolute_tolerance": TOL,
        "planned_unique_comparisons": 120,
        "measured_unique_comparisons": len(canonical),
        "all_scheduled_present": len(canonical) == 120,
        "bounded_on_returned": bool(canonical and all(r["bounded_numeric_pass"] for r in canonical)),
        "bounded_all": bool(len(canonical) == 120 and all(r["bounded_numeric_pass"] for r in canonical)),
        "strict_state_all": bool(len(canonical) == 120 and all(r["state_exact"] for r in canonical)),
        "bitwise_all": bool(len(canonical) == 120 and all(r["bitwise_and_support_exact"] for r in canonical)),
        "threshold_crossings": sum(r["threshold_crossings"] for r in canonical),
        "support_disagreement": sum(r["support_disagreement"] for r in canonical),
        "results": results,
    }
    with new_output(output) as dst:
        write_json(dst / "parity.json", out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("dataset", "producer", "cpu", "protocol", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("dataset", "producer", "cpu"):
        p.add_argument("--" + k + "-sha256", required=True)
    a = p.parse_args()
    q = grade(
        a.dataset,
        a.dataset_sha256,
        signed(a.producer, a.producer_sha256),
        signed(a.cpu, a.cpu_sha256),
        a.protocol,
        a.output,
        os.environ["EL_RUN_ID"],
    )
    print(
        json.dumps(
            {
                k: q[k]
                for k in (
                    "bounded_all",
                    "strict_state_all",
                    "bitwise_all",
                    "threshold_crossings",
                    "support_disagreement",
                )
            }
        )
    )
