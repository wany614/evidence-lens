"""Audit trajectory-specific observations, visibility support and the cost of choosing two cameras."""

import argparse
import json
import os
from pathlib import Path

from b2_records import selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from t0_evaluate import table
from t0_records import resolve


def audit(roots, cfg, matrix, out):
    pins = cfg["sources"]
    book = load(matrix)
    require(
        book["observations"] == 10 and len(book["sources"]) == 10,
        "incomplete per-trajectory access list",
    )
    rules = load(resolve(roots, pins["protocol"]))["protocol"]
    summary = []
    util = []
    probes = []
    observations = []
    for obj in ("set1", "set6"):
        cp = pins["choices" + obj]
        path = resolve(roots, cp)
        choice = load(path)
        rr = load(resolve(roots, pins["probe" + obj]))
        gpu = rr["gpu_parent"]
        total_unknown = sum(x["unknown_full"] for x in choice["per_query"].values())
        total_canvas = sum(x["canvas"] for x in choice["per_query"].values())
        require(
            rr["checkpoint_sha256"] == choice["m2_checkpoint_sha256"]
            and len(choice["pair_utilities"]) == 28
            and choice["pool"] == rules["candidate_ids"],
            "unexpected selector pool",
        )
        summary.append(
            {
                "object_id": obj,
                "render_run_id": gpu["run_id"],
                "render_manifest_sha256": gpu["manifest_sha256"],
                "native_model_sha256": rr["checkpoint_sha256"],
                "render_worker_s": rr["worker_wall_seconds"],
                "render_recorder_readback": pins["probe" + obj]["run_id"],
                "selector_run_id": cp["run_id"],
                "selector_manifest_sha256": cp["manifest_sha256"],
                "query_unknown": total_unknown,
                "query_canvas": total_canvas,
                "eligible_lattice_points": choice["point_count_before_dedup"],
                "unique_3d_points": choice["assessed_query_points"],
                "feature_score_s": choice["feature_and_score_s"],
                "full_selection_s": choice["selection_stage_s"],
                "torch_peak_allocated_bytes": rr["resources"][
                    "torch_allocated_peak_bytes"
                ],
                "torch_peak_reserved_bytes": rr["resources"][
                    "torch_reserved_peak_bytes"
                ],
                "fallback": choice["fallback_reason"],
                "source_RGB_Z_alpha_replay_max_abs": max(
                    v
                    for entry in choice["source_replay_max_abs"].values()
                    for v in entry.values()
                ),
            }
        )
        util.extend(
            {
                "object_id": obj,
                "first_id": row["pair"][0],
                "second_id": row["pair"][1],
                "uniform_covered_points": row["uniform"],
                "sum_of_continuous_risk_on_union": row["risk"],
                "optical_axis_angle_rad": row["axis_angle_rad"],
            }
            for row in choice["pair_utilities"]
        )
        probes.extend(
            {"object_id": obj, "query": k, **v} for k, v in choice["per_query"].items()
        )
        for row in (r for r in book["sources"] if r["object_id"] == obj):
            b = row["branch"]
            decision = load(resolve(roots, row["selection"]))
            source_path = resolve(roots, row["source"])
            source = load(source_path)
            require(
                row["selection"]["run_id"] == cp["run_id"]
                and row["selection"]["sha256"] == choice["selections"][b]
                and decision["selected_official_ids"]
                == choice["results"][b]
                == row["pair"]
                and source["selection"] == row["selection"],
                "source-access receipt differs from sealed choice",
            )
            before = load(roots[0] / "runs" / cp["run_id"] / "completion.json")[
                "sealed_at_utc"
            ]
            after = load(roots[0] / "runs" / row["source"]["run_id"] / "run.json")[
                "started_at_utc"
            ]
            require(
                after > before, "candidate access began before both choices were sealed"
            )
            observations.append(
                {
                    "object_id": obj,
                    "branch": b,
                    "new_camera_1": row["pair"][0],
                    "new_camera_2": row["pair"][1],
                    "prior_photo_ids": "24,22",
                    "new_photos": 2,
                    "decision_run_id": cp["run_id"],
                    "decision_member_sha256": row["selection"]["sha256"],
                    "sealed_utc": before,
                    "photo_job_start_utc": after,
                    "photo_access_run_id": row["source"]["run_id"],
                    "source_receipt_sha256": row["source"]["sha256"],
                    "fit_trial": row["trial_id"],
                    "reuse_of": row.get("reuse_of"),
                }
            )
    table(out, "probe-costs.csv", summary)
    table(out, "pair-utilities.csv", util)
    table(out, "query-support.csv", probes)
    table(out, "observations.csv", observations)
    save(
        out / "access-audit.json",
        {
            "schema": "el-b2-observation-audit-1",
            "run_id": os.environ["EL_RUN_ID"],
            "objects": 2,
            "observations": len(observations),
            "new_photos_per_trajectory": 2,
            "unique_fit_requests": book["unique_jobs"],
            "observation_index_sha256": sha(matrix),
            "protocol": pins["protocol"],
            "files": {
                n: sha(out / n)
                for n in (
                    "probe-costs.csv",
                    "pair-utilities.csv",
                    "query-support.csv",
                    "observations.csv",
                )
            },
            "cost_scope": "Four virtual renders, eight candidate visibility renders and two source-identity control renders are performed together for each shared D start. Random pair draws require only candidate IDs. Both coverage strategies reuse the same query scores and visibility calculation.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("record", "worker"))
    for name in ("store", "history", "config", "matrix", "out"):
        p.add_argument("--" + name, type=Path)
    a = p.parse_args()
    roots = (a.store.resolve(), a.history.resolve())
    if a.action == "worker":
        audit(roots, load(a.config), a.matrix, a.out)
    else:
        s, h = roots
        matrix = s / "work/controller/d-authorized-branches.json"
        items = load(matrix)
        pins = {"protocol": selected(s, "nbv-method")}
        for o in ("set1", "set6"):
            pins["choices" + o] = selected(s, "nbv-choices-" + o)
            pins["probe" + o] = selected(s, "nbv-probe-receive-" + o)
        for i, r in enumerate(items["sources"]):
            pins["source" + str(i)] = r["source"]
        print(
            json.dumps(
                stage(
                    s,
                    h,
                    "nbv-observation-audit",
                    "report",
                    pins,
                    {
                        "role": "compare choices, observations and compute under equal photo budgets"
                    },
                    [
                        "{python}",
                        "{source}/scripts/b2_nbv_observations.py",
                        "worker",
                        "--matrix",
                        "{input:matrix}",
                        "--out",
                        "{out}",
                        "--config",
                        "{config}",
                        "--store",
                        str(s),
                        "--history",
                        str(h),
                    ],
                    "payload/access-audit.json",
                    90000,
                    {"matrix": matrix},
                ),
                sort_keys=True,
            )
        )
