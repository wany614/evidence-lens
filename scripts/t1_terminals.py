"""Audit every registered T1 trajectory and freeze all endpoints before joint reference access."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from t0_evaluate import table
from t1_access import (
    CONDITIONS,
    FIXED,
    OBJECTS,
    S0,
    SLOTS,
    STRATEGIES,
    TARGETS,
    decisions,
)
from t1_study import (
    DEFAULT,
    PROJECT,
    authority,
    bind,
    locate,
    science_input,
    selected,
    stage,
)

from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens._verify import require
from evidence_lens.multisource_io import Case


def transcript(roots, pin):
    f = locate(roots, pin)
    base = next(r / "runs" / pin["run_id"] for r in roots if (r / "runs" / pin["run_id"]).is_dir())
    return load(f), load(base / "run.json"), load(base / "completion.json")


def seal(roots):
    store = roots[0]
    entry, freeze, _ = authority(roots)
    pins = {"entry": entry, "freeze": freeze}
    plan = {}
    for obj in OBJECTS:
        decisions_pin = selected(store, "t1-decisions-" + obj)
        pins["choices-" + obj] = decisions_pin
        for strategy in STRATEGIES:
            pins["decision-" + obj + "-" + strategy] = bind(
                roots, decisions_pin, "payload/" + strategy + ".json"
            )
        outcome = load(locate(roots, decisions_pin))
        blocked = outcome["schema"] == "el-t1-probe-terminal-1"
        if not blocked:
            pins["scope-" + obj] = bind(roots, decisions_pin, "payload/selector-input-audit.json")
        plan[obj] = {}
        for name in CONDITIONS:
            if blocked and name != "M2-fixed":
                plan[obj][name] = {
                    "object_id": obj,
                    "condition": name,
                    "status": "failed",
                    "source": None,
                    "initialization": None,
                    "ticket": None,
                    "readback": None,
                    "case": None,
                    "alias_of": None,
                    "fit_key_sha256": None,
                    "initialization_method": None,
                    "initial_points": None,
                    "qualified": None,
                    "disposition": decisions_pin,
                    "reason": outcome["reason"],
                    "official_camera_order": [],
                    "planned_source_order": FIXED if name == "M4-fixed" else None,
                    "generation": None,
                    "render": None,
                    "candidate_photo_access_authorized": False,
                }
                continue
            rec = load(store / "work/branches" / obj / (name + ".json"))
            canonical = rec["owner"]
            bundle = selected(store, "t1-readback-" + obj + "-" + canonical)
            data = load(locate(roots, bundle))
            case = bind(roots, bundle, "payload/case/case.json") if data["status"] == "succeeded" else None
            cell = {
                "object_id": obj,
                "condition": name,
                "status": data["status"],
                "readback": bundle,
                "case": case,
                "source": rec["source"],
                "initialization": rec["initialization"],
                "ticket": rec["ticket"],
                "alias_of": rec["alias_of"],
                "fit_key_sha256": hashlib.sha256(encoded(rec["source_key"])).hexdigest(),
                "initialization_json_sha256": rec["init_json_sha256"],
                "initialization_array_sha256": rec["init_npz_sha256"],
                "scientific_job_sha256": rec["scientific_job_sha256"],
            }
            for k in ("source", "initialization", "readback", "ticket", "case"):
                if cell[k]:
                    pins[f"{obj}-{name}-{k}"] = cell[k]
            plan[obj][name] = cell
    cmd = [
        "{python}",
        "-B",
        "{source}/scripts/t1_terminals.py",
        "worker",
        "--config",
        "{config}",
        "--store",
        str(store),
        "--out",
        "{out}",
    ]
    return stage(
        roots,
        "t1-terminals",
        "report",
        pins,
        {
            "plan": plan,
            "evaluated_code": {
                f: sha(PROJECT / "scripts" / f) for f in ("t1_nbv_evaluate.py", "t1_nbv_independent.py")
            },
        },
        cmd,
        "terminals.json",
        700000,
    )


def validate_probe_failure(roots, obj, joint, choice, dec, choices, query, choice_seal):
    failed = joint["failed_probe"]
    base = roots[0] / "runs" / failed["run_id"]
    run = load(base / "run.json")
    require(
        joint["object_id"] == obj
        and joint["completed_pair_decisions"] == 0
        and joint["candidate_photo_access_authorized"] is False
        and joint["actual_candidate_observations"] == 0
        and joint["results"] == {k: None for k in STRATEGIES}
        and run["execution_status"] == "failed"
        and sha(base / "manifest.json") == failed["manifest_sha256"]
        and sha(base / "run.json") == failed["sha256"],
        "Unproven frozen scientific gate failure",
    )
    query.append(
        {
            "object_id": obj,
            "probe_run": failed["run_id"],
            "status": "failed",
            "actual_final_gaussians": joint["actual_final_gaussians"],
            "probe_recorder_seconds": run["elapsed_seconds_monotonic"],
            "reason": joint["reason"],
        }
    )
    for k in STRATEGIES:
        row = load(dec["choice-" + k])
        require(
            sha(dec["choice-" + k]) == joint["selections"][k]
            and row["object_id"] == obj
            and row["strategy_id"] == k
            and row["status"] == "failed"
            and row["selected_official_ids"] is None
            and row["actual_candidate_observations"] == 0
            and row["failed_probe"] == failed,
            "Wrong trajectory terminal receipt",
        )
        choices.append(
            {
                "object_id": obj,
                "strategy": k,
                "first": None,
                "second": None,
                "fallback": None,
                "decision_sha256": joint["selections"][k],
                "time_sealed": choice_seal["sealed_at_utc"],
                "status": "failed",
                "reason": joint["reason"],
                "preregistered_random_pair_only": row["registered_random_pair_only"],
                "observation_run": None,
            }
        )


def worker(roots, cfg, out):
    pins = cfg["sources"]
    frozen = load(locate(roots, pins["freeze"]))
    require(
        set(cfg["plan"]) == set(OBJECTS) and all(set(v) == set(CONDITIONS) for v in cfg["plan"].values()),
        "Exactly 42 registered cells required",
    )
    rows, observations, choices, query, util = [], [], [], [], []
    unique, access = {}, []
    for obj in OBJECTS:
        choice = pins["choices-" + obj]
        joint, _, choice_seal = transcript(roots, choice)
        dec = {"choice-" + k: locate(roots, pins["decision-" + obj + "-" + k]) for k in STRATEGIES}
        blocked = joint["schema"] == "el-t1-probe-terminal-1"
        if blocked:
            validate_probe_failure(roots, obj, joint, choice, dec, choices, query, choice_seal)
        else:
            decisions(joint, dec, obj)
            audit = load(locate(roots, pins["scope-" + obj]))
            require(
                audit["permitted_photo_ids"] == S0
                and audit["candidate_RGB_reads"]
                == audit["evaluation_poses"]
                == audit["evaluation_reference_reads"]
                == 0
                and set(joint["results"]) == set(STRATEGIES),
                "Selector scope",
            )
            probe_pin = audit["input_hashes"]["predictions"]
            probe, prun, _ = transcript(roots, probe_pin)
            require(
                probe["checkpoint_sha256"] == joint["m2_checkpoint_sha256"]
                and len(joint["pair_utilities"]) == 28,
                "Wrong M2 checkpoint or incomplete utility grid",
            )
            query.append(
                {
                    "object_id": obj,
                    "probe_run": probe_pin["run_id"],
                    "probe_gpu_seconds": probe["worker_wall_seconds"],
                    "probe_recorder_seconds": prun["elapsed_seconds_monotonic"],
                    "unique_support": joint["assessed_query_points"],
                    "raw_support": joint["point_count_before_dedup"],
                    "feature_score_seconds": joint["feature_and_score_s"],
                    "selector_seconds": joint["selection_stage_s"],
                    "fallback": joint["fallback_reason"],
                    "max_source_rerender_difference": max(
                        x for m in joint["source_replay_max_abs"].values() for x in m.values()
                    ),
                }
            )
            for r in joint["pair_utilities"]:
                util.append(
                    {
                        "object_id": obj,
                        "first": r["pair"][0],
                        "second": r["pair"][1],
                        "uniform": r["uniform"],
                        "risk": r["risk"],
                        "axis_angle_rad": r["axis_angle_rad"],
                    }
                )
        for name in CONDITIONS:
            cell = cfg["plan"][obj][name]
            if blocked and name != "M2-fixed":
                require(
                    cell["disposition"] == choice
                    and cell["status"] == "failed"
                    and cell["reason"]
                    and all(
                        cell[k] is None
                        for k in ("source", "initialization", "ticket", "readback", "case", "alias_of")
                    )
                    and not (roots[0] / "work/controller" / ("t1-source-" + obj + "-" + name)).exists(),
                    "A failed collective selection cannot acquire candidate photos or run M4",
                )
                access.append(
                    {
                        "object_id": obj,
                        "condition": name,
                        "run_id": None,
                        "authorized": "",
                        "observed": "",
                        "source_HTTP_bytes": 0,
                        "elapsed_seconds": None,
                        "disposition": choice["run_id"],
                    }
                )
                rows.append(cell)
                continue
            source, start, sealed = transcript(roots, cell["source"])
            req = source["official_by_slot"]
            ids = SLOTS[name[:2]]
            cameras = [req[k] for k in ids]
            if name == "M2-fixed":
                expected = S0
            else:
                require(
                    choice_seal["sealed_at_utc"] < start["started_at_utc"] and source["decisions"] == choice,
                    "Candidate accessed before five signed choices",
                )
                expected = FIXED if name == "M4-fixed" else [24, *joint["results"][name[3:]], 22]
            require(
                cameras == expected
                and source["logical_condition"] == name
                and source["object_id"] == obj
                and source["entry"] == pins["entry"]
                and source["reference_photo_reads"] == source["other_branch_candidate_observations"] == 0,
                "Source observation mismatch",
            )
            p = locate(roots, bind(roots, cell["source"], "payload/access.json"))
            acquisition = load(p)
            require(
                acquisition["authorized_order"] == acquisition["accessed_order"] == expected
                and acquisition["observed_reference_count"] == 0,
                "Access trace differs from authorization",
            )
            access.append(
                {
                    "object_id": obj,
                    "condition": name,
                    "run_id": cell["source"]["run_id"],
                    "authorized": " ".join(map(str, expected)),
                    "observed": " ".join(map(str, acquisition["accessed_order"])),
                    "source_HTTP_bytes": acquisition["HTTP_response_bytes"],
                    "elapsed_seconds": acquisition["elapsed_seconds"],
                    "started_utc": start["started_at_utc"],
                    "sealed_utc": sealed["sealed_at_utc"],
                }
            )
            for slot, cam in zip(ids, cameras, strict=True):
                ref = source["sources"][slot]
                orig = bind(roots, cell["source"], "payload/" + ref["original_path"])
                png = bind(roots, cell["source"], "payload/" + ref["rgb"]["path"])
                require(
                    orig["sha256"] == ref["original_sha256"] and png["sha256"] == ref["rgb"]["sha256"],
                    "Original or crop changed",
                )
                observations.append(
                    {
                        "object_id": obj,
                        "condition": name,
                        "slot": slot,
                        "camera": cam,
                        "observation_run": cell["source"]["run_id"],
                        "retrieval": ref["acquisition"]["kind"],
                        "original_sha256": orig["sha256"],
                        "origin_crc32": ref["original_crc32"],
                        "cropped_sha256": png["sha256"],
                        "decision_run": choice["run_id"] if name[:2] == "M4" else None,
                    }
                )
            if name[3:] in STRATEGIES:
                d = load(dec["choice-" + name[3:]])
                require(
                    source["own_decision"] == pins["decision-" + obj + "-" + name[3:]],
                    "Wrong trajectory decision",
                )
                choices.append(
                    {
                        "object_id": obj,
                        "strategy": name[3:],
                        "first": d["selected_official_ids"][0],
                        "second": d["selected_official_ids"][1],
                        "fallback": d["fallback_reason"],
                        "decision_sha256": sha(dec["choice-" + name[3:]]),
                        "time_sealed": choice_seal["sealed_at_utc"],
                        "observation_run": cell["source"]["run_id"],
                    }
                )
            roles, key = science_input(roots, cell["source"], frozen)
            identity = hashlib.sha256(encoded(key)).hexdigest()
            require(
                identity == cell["fit_key_sha256"], "Distinct ordered scientific inputs cannot be aliased"
            )
            init = load(locate(roots, cell["initialization"]))
            ticket = load(locate(roots, cell["ticket"]))
            actualjob = load(locate(roots, bind(roots, cell["ticket"], "payload/job.json")))
            require(
                init["recipe_sha256"] == actualjob["recipe_sha256"] == roles["recipe"]["sha256"]
                and actualjob["stop_step"] == 5000
                and actualjob["object_id"] == obj
                and actualjob["condition"] == name[:2]
                and actualjob["identity"]["split"] == "test"
                and actualjob["files"]["source_cameras.json"]["sha256"] == key["input_bytes"]["cameras"]
                and actualjob["files"]["render_cameras.json"]["sha256"] == key["input_bytes"]["rendercameras"]
                and all(
                    actualjob["files"]["sources/" + n + ".png"]["sha256"] == key["input_bytes"][n]
                    for n in ids
                )
                and actualjob["files"]["init.json"]["sha256"]
                == cell["initialization_json_sha256"]
                == cell["initialization"]["sha256"]
                and actualjob["files"]["init.npz"]["sha256"]
                == cell["initialization_array_sha256"]
                == bind(roots, cell["initialization"], "payload/init.npz")["sha256"]
                == init["init_npz_sha256"]
                and cell["scientific_job_sha256"]
                == ticket["job_sha256"]
                == bind(roots, cell["ticket"], "payload/job.json")["sha256"],
                "Initialization or complete fit/renderer configuration differs",
            )
            record = load(locate(roots, cell["readback"]))
            alias = cell["alias_of"]
            if alias:
                own = cfg["plan"][obj][alias]
                require(
                    own["alias_of"] is None
                    and own["condition"] in CONDITIONS[: CONDITIONS.index(name)]
                    and all(
                        own[k] == cell[k]
                        for k in (
                            "ticket",
                            "initialization",
                            "readback",
                            "case",
                            "status",
                            "fit_key_sha256",
                            "initialization_json_sha256",
                            "initialization_array_sha256",
                            "scientific_job_sha256",
                        )
                    ),
                    "Alias lacks a byte-identical independent original",
                )
            else:
                require(identity not in unique, "Duplicate fit on the same scientific inputs")
                unique[identity] = cell
            require(
                record["object_id"] == obj
                and record["condition"] == name[:2]
                and record["status"] == cell["status"]
                and record["target_reference_files_opened"] == 0
                and record["risk_scorings"] == 0,
                "Readback changed",
            )
            if cell["case"]:
                case = Case(locate(roots, cell["case"]).parent, cell["case"]["sha256"])
                require(
                    record["verified"]
                    and record["completed_steps"] == 5000
                    and case.sha256 == record["case_sha256"]
                    and case.meta["target_ids"] == TARGETS
                    and set(case.meta["renders"]) == set(ids + TARGETS),
                    "Missing expected endpoint predictions",
                )
            else:
                require(
                    record["status"] == "failed" and record["failure_reasons"],
                    "Failure lacks a sealed reason",
                )
            cell.update(
                official_camera_order=cameras,
                planned_source_order=cameras,
                initialization_method=init["actual_method"],
                initial_points=init["N"],
                qualified=init["eligibility"]["qualified"],
                generation=record["origins"]["generation"],
                render=record["origins"].get("render"),
                reason=record.get("failure_reasons"),
            )
            rows.append(cell)
    require(
        len(rows) == 42
        and len(choices) == 30
        and len(util) == 28 * sum(r.get("status") != "failed" for r in query),
        "Incomplete scientific terminal matrix",
    )
    table(
        out,
        "conditions.csv",
        [
            {
                "object_id": c["object_id"],
                "condition": c["condition"],
                "status": c["status"],
                "alias_of": c["alias_of"],
                "planned_source_order": " ".join(map(str, c.get("planned_source_order") or [])),
                "ordered_photos": " ".join(map(str, c["official_camera_order"])),
                "reason": c.get("reason"),
                "source": c["source"]["run_id"] if c["source"] else None,
                "initialization": c["initialization"]["run_id"] if c["initialization"] else None,
                "initialization_method": c["initialization_method"],
                "initial_points": c["initial_points"],
                "qualified": c["qualified"],
                "fit_identity": c["fit_key_sha256"],
                "generation": c["generation"]["run_id"] if c["generation"] else None,
                "render": c["render"]["run_id"] if c["render"] else None,
                "readback": c["readback"]["run_id"] if c["readback"] else None,
                "case_sha256": c["case"]["sha256"] if c["case"] else None,
            }
            for c in rows
        ],
    )
    for name, data in (
        ("decisions", choices),
        ("observations", observations),
        ("source-access", access),
        ("query-cost", query),
        ("pair-utilities", util),
    ):
        table(out, name + ".csv", data)
    save(
        out / "terminals.json",
        {
            "schema": "el-t1-reconstruction-terminals-1",
            "run_id": os.environ["EL_RUN_ID"],
            "entry": pins["entry"],
            "freeze": pins["freeze"],
            "T1_objects": OBJECTS,
            "conditions": CONDITIONS,
            "cells": rows,
            "condition_count": len(rows),
            "reconstruction_fit_count": len(unique),
            "aliases": sum(c["alias_of"] is not None for c in rows),
            "undispatched_conditions": sum(c["ticket"] is None for c in rows),
            "failed_conditions": sum(c["status"] != "succeeded" for c in rows),
            "sealed_strategy_terminals": len(choices),
            "sealed_decisions": sum(c.get("status") != "failed" for c in choices),
            "unavailable_decisions": sum(c.get("status") == "failed" for c in choices),
            "preopening_reference_photo_reads": 0,
            "additional_selected_risk_scores": 0,
            "new_risk_training_count": 0,
            "evaluation_code_sha256": cfg["evaluated_code"],
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("seal", "worker"))
    p.add_argument("--store", type=Path, default=DEFAULT)
    for k in ("out", "config"):
        p.add_argument("--" + k, type=Path)
    a = p.parse_args()
    roots = a.store.resolve(), a.store.resolve() / "history"
    if a.operation == "seal":
        print(json.dumps(seal(roots)))
    else:
        worker(roots, load(a.config), a.out)
