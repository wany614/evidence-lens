"""Seal the prospective T1 endpoints and bind tested methods to the holdout without opening it."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from b2_records import selected, stage
from dtu_score import METHODS, RELEASE_SHA
from evidence_lens._bundle import load, save, sha, utc
from evidence_lens._verify import require
from t0_records import bind, resolve

PROJECT = Path(__file__).resolve().parents[1]
RELEASE = "20261123T193505072501Z-6046afc1fce4480b"
PRESPEC_ROLES = {"cohort", "selector", "threshold_method", "reconstruction", "risk_release"}


def scopes(s, cohort):
    """Inventory actual B acquisition receipts without opening any photographic member."""
    rows = []
    schemas = {"el-b2-dtu-source-1": "sources", "el-b2-development-source-1": "sources",
               "el-b2-dtu-references-1": "references"}
    for root in (s / "runs", s / "quarantine"):
        for payload in root.glob("*/payload"):
            records = sorted(set(payload.rglob("*provenance.json")) | set(payload.rglob("references.json")))
            require(not list(payload.rglob("http.jsonl")) or records,
                    "unaccounted photographic transfer log in " + payload.parent.name)
            for f in records:
                d = load(f)
                require(d.get("schema") in schemas and isinstance(d.get("object_id"), str)
                        and d["object_id"] and (d.get("sources") or d.get("views")),
                        "unaccounted provenance record in " + payload.parent.name)
                rows.append({"run_id": payload.parent.name, "object": d["object_id"],
                             "stage": d["schema"], "kind": schemas[d["schema"]], "sha256": sha(f)})
    ids = {r["object"] for r in rows}
    require(
        ids <= set(cohort["splits"]["C6"]) | {"set1", "set6"}
        and not ids.intersection(cohort["splits"]["T1"])
        and {r["object"] for r in rows if r["kind"] == "references"}
        == set(cohort["splits"]["C6"]),
        "batch B access receipts include unexpected families, or incompletely account for C6 references",
    )
    return sorted(rows, key=lambda r: r["run_id"])


def worker(c, inputs, store, history, out, mode):
    roots = (store, history)
    pins = c["sources"]
    prespec = load(inputs)
    cohort = load(resolve(roots, pins["cohort"]))
    require(
        len(set(cohort["splits"]["T1"])) == 6
        and not set(cohort["splits"]["T1"]).intersection(cohort["splits"]["C6"]),
        "T1 and C6 must be disjoint physical cohorts",
    )
    s0 = set(prespec["camera_contract"]["S0"])
    u = set(prespec["camera_contract"]["U_pool"])
    h = set(prespec["camera_contract"]["H_eval"])
    require(
        not (s0 & u or s0 & h or u & h)
        and len(u) == 8
        and len(h) == len(cohort["target_ids"]) == 6
        and prespec["max_main_reconstructions"]
        == 6 * prespec["cell_budget_per_object"]
        == 42,
        "holdout roles or budgets differ",
    )
    common = {
        "run_id": os.environ["EL_RUN_ID"],
        "sealed_decision_utc": utc(),
        "cohort": pins["cohort"],
        "T1_objects": cohort["splits"]["T1"],
        "prospective_protocol_sha256": sha(inputs),
        "endpoints": prespec,
        "preparation_scope": "metadata only; no T1 photograph authorized by this record",
    }
    if mode == "spec-worker":
        save(
            out / "prospective.json",
            {"schema": "el-b2-t1-prospective-1", **common, "members": pins},
        )
        return
    prior = load(resolve(roots, pins["prospective"]))
    require(
        prior["schema"] == "el-b2-t1-prospective-1" and prior["prospective_protocol_sha256"] == sha(inputs)
        and set(prior["members"]) == PRESPEC_ROLES
        and all(prior["members"][k] == pins[k] for k in PRESPEC_ROLES),
        "T1 endpoints changed after prospective seal",
    )
    release = load(resolve(roots, pins["risk_release"]))
    cal = load(resolve(roots, pins["calibration"]))
    method = load(resolve(roots, pins["threshold_method"]))
    nbv = load(resolve(roots, pins["selector"]))
    d_eval = load(resolve(roots, pins["d_evaluation"]))
    access = load(resolve(roots, pins["d_observations"]))
    require(
        pins["risk_release"]["sha256"] == RELEASE_SHA
        and set(release["models"]) == set(METHODS)
        and release["models"] == method["models"]
        and nbv["protocol"]["random_selector_seeds"] == [137101, 137102, 137103]
        and nbv["protocol"]["primary_optimization_seed"] == 43171
        and nbv["protocol"]["physical_endpoint_ids"] == prespec["camera_contract"]["S0"]
        and nbv["protocol"]["candidate_ids"] == prespec["camera_contract"]["U_pool"],
        "risk release or selected-view algorithm changed",
    )
    require(
        cal["schema"] == "el-b2-calibration-results-1"
        and cal["cohort"] == pins["cohort"]
        and cal["opening"] == pins["c6_opening"]
        and cal["calibration_method"] == pins["threshold_method"]
        and cal["fitted_conditions"] == 2
        and cal["leave_one_object_out_fits"] == 12
        and cal["candidate_rows"] == 2814
        and cal["T1_access"] == cal["new_risk_models"] == 0,
        "incomplete C6 results",
    )
    require(
        d_eval["schema"] == "el-b2-d-evaluation-1"
        and d_eval["objects"] == ["set1", "set6"]
        and d_eval["T1_photographs_read"] == 0
        and access["schema"] == "el-b2-observation-audit-1"
        and access["observations"] == 10,
        "development process incomplete",
    )
    profiles = {k: load(resolve(roots, pins["profile" + k])) for k in ("M2", "M4")}
    for k, doc in profiles.items():
        fp = pins["profile" + k]
        require(
            fp["run_id"] == pins["calibration"]["run_id"]
            and fp["sha256"] == cal["profiles"][k]["sha256"]
            and doc["condition"] == k and doc["method"] == "MS-L1"
            and doc["lineage"]["opening"] == pins["c6_opening"]
            and doc["evaluated_on_holdout"] is False
            and doc["status"] == cal["outcomes"][k]["status"]
            and doc["tau"] == cal["outcomes"][k]["tau"]
            and doc["status"]
            in ("empirical_calibration_only", "no_feasible_operating_point")
            and (
                (doc["status"] == "no_feasible_operating_point")
                == (doc.get("tau") is None)
            ),
            "threshold/status mismatch",
        )
        grade = load(resolve(roots, pins["cli_validation" if k == "M2" else "cli_validation_M4"]))
        proof = grade["calibration"]
        require(grade["schema"] == "el-b2-asset-score-readback-1" and grade["passed"]
                and grade["target_reference_photographs_read"] == 0
                and proof["condition"] == k and proof["status"] == doc["status"]
                and proof["profile_sha256"] == fp["sha256"] and proof["mismatches"] == []
                and proof["applied"] == (doc["tau"] is not None),
                "clean fixed-layout CPU application differs from its C6 outcome")
    parity = load(resolve(roots, pins["asset_parity"]))
    registered = load(resolve(roots, pins["asset_protocol"]))
    tol = registered["comparisons"]["absolute_tolerance"]
    require(registered["schema"] == "el-asset-registration-1"
            and parity["schema"] == "el-b2-gaussian-parity-1"
            and parity["passed"] and parity["protocol_sha256"] == registered["protocol_sha256"]
            and len(parity["rows"]) == 30
            and len({(r["asset"], r["colmap"], r["view_id"]) for r in parity["rows"]}) == 30
            and all(r["passed"] and r["camera_max_abs"] <= tol["camera_K_and_pose"]
                    and set(r["channels"]) == {"alpha", "z", "rgb"}
                    and all(ch["support_difference"] == 0 and ch["passed"] and ch["max_abs"] <= ch["tolerance"]
                            == tol["expected_camera_Z_m" if name == "z" else name]
                            for name, ch in r["channels"].items()) for r in parity["rows"]),
            "standard asset evidence differs from the preregistered CUDA comparisons")
    intake = load(resolve(roots, pins["c6_resources"]))
    require(intake["schema"] == "el-b2-public-cohort-1" and intake["cohort"] == pins["cohort"]
            and intake["included_reference_photographs"] and intake["unique_C6_photographs"] == 60
            and intake["T1_photograph_reads"] == 0, "incomplete final acquisition accounting")
    geometry = load(resolve(roots, pins["query_cameras"]))["views"]
    rig = load(resolve(roots, pins["heldout_camera_geometry"]))["views"]
    require(pins["heldout_camera_geometry"]["sha256"] == cohort["camera"]["conditions"]["M2"]["render_cameras"]["sha256"]
            and tuple(cohort["target_view_ids"][v] for v in cohort["target_ids"]) == tuple(prespec["camera_contract"]["H_eval"]),
            "evaluation camera geometry differs from the registered physical rig")
    query = [cam for view, cam in geometry.items() if view.startswith("q")]
    require(len(query) == 4 and all(not np.allclose(q["c2w"], rig[v]["c2w"], rtol=0, atol=1e-9)
                                    for q in query for v in cohort["target_ids"]),
            "virtual query duplicates an evaluation pose")
    rows = scopes(store, cohort)
    save(
        out / "acquisition-audit.json",
        {
            "schema": "el-b2-access-inventory-1",
            "scope": "All batch-store admitted and quarantined photographic acquisition receipts and HTTP logs. Directories of the independently exposed D originals are outside this scan.",
            "rows": rows,
            "T1_photo_receipts": sum(
                r["object"] in cohort["splits"]["T1"] for r in rows
            ),
        },
    )
    save(
        out / "t1-freeze.json",
        {
            "schema": "el-b2-t1-freeze-1",
            **common,
            "sources": pins,
            "scientific_code_sha256": c["scientific_code_sha256"],
            "released_models": release["models"],
            "layout_profiles": {
                k: {
                    "pin": pins["profile" + k],
                    "status": v["status"],
                    "tau": v.get("tau"),
                }
                for k, v in profiles.items()
            },
            "selection": nbv["protocol"],
            "selection_code": nbv["scientific_files"],
            "risk_training_in_this_study": 0,
            "T1_photographic_opening_authorized": False,
            "calibration_heldout_status": {k: ("no fitted operating point" if d["tau"] is None
                                               else "awaiting independent T1 evaluation") for k, d in profiles.items()},
            "query_evaluation_pose_matches": 0,
            "acquisition_audit_sha256": sha(out / "acquisition-audit.json"),
        },
    )


def run(a):
    s, h = a.store.resolve(), a.history.resolve()
    if a.action in ("spec-worker", "freeze-worker"):
        return worker(load(a.config), a.protocol, s, h, a.out, a.action)
    spec = PROJECT / "protocol/b2-t1-confirmation-v1.json"
    pins = {
        "cohort": selected(s, "b2-dtu-cohort"),
        "selector": selected(s, "nbv-method"),
        "threshold_method": selected(s, "b2-calibration-method"),
        "reconstruction": selected(s, "b2-reconstruction-seed-43171"),
        "risk_release": bind(h, RELEASE, "payload/cpu/release.json"),
    }
    if a.action == "prespec":
        tag = "prospective-t1-endpoints"
        output = "prospective.json"
        task = "spec-worker"
        extra = {}
    else:
        tag = "t1-freeze"
        output = "t1-freeze.json"
        task = "freeze-worker"
        evidence = load(a.evidence)
        require(pins.keys().isdisjoint(evidence), "external evidence cannot override a prospective method")
        pins.update(evidence)
        required = {
            "prospective",
            "c6_opening",
            "calibration",
            "profileM2",
            "profileM4",
            "d_evaluation",
            "d_observations",
            "asset_parity",
            "asset_protocol",
            "cli_validation",
            "cli_validation_M4",
            "c6_resources",
            "query_cameras",
            "heldout_camera_geometry",
        }
        require(
            required <= pins.keys(),
            "missing final batch evidence roles: "
            + str(sorted(required - pins.keys())),
        )
        names = [
            "scripts/b2_t1_freeze.py",
            "scripts/b2_evaluate_development.py",
            "scripts/b2_c6_study.py",
            "scripts/dtu_score.py",
            "scripts/t0_evaluate.py",
            "scripts/dtu_eval_metrics.py",
            "scripts/dtu_eval.py",
            "scripts/recovery_eval_quality.py",
            "scripts/b2_calibration.py",
            "scripts/b2_nbv_select.py",
            "scripts/b2_probe_geometry.py",
            "src/evidence_lens/multisource.py",
            "src/evidence_lens/multisource_io.py",
            "src/evidence_lens/multisource_model.py",
            "scripts/recovery_eval_coverage.py",
            "scripts/b2_reconstruct.py",
            "scripts/b2_worker_readback.py",
            "scripts/b2_worker_development.py",
            "src/evidence_lens_bundle/calibration.py",
        ]
        extra = {"scientific_code_sha256": {n: sha(PROJECT / n) for n in names}}
    return stage(
        s,
        h,
        tag,
        "archive_verify",
        pins,
        {"role": "prospective confirmation protocol", **extra},
        [
            "{python}",
            "{source}/scripts/b2_t1_freeze.py",
            task,
            "--protocol",
            "{input:protocol}",
            "--store",
            str(s),
            "--history",
            str(h),
            "--config",
            "{config}",
            "--out",
            "{out}",
        ],
        "payload/" + output,
        300000,
        {"protocol": spec},
        a.retry_of,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action", choices=("prespec", "seal", "spec-worker", "freeze-worker")
    )
    for n in ("store", "history", "config", "out", "protocol", "evidence"):
        p.add_argument("--" + n, type=Path)
    p.add_argument("--retry-of")
    print(json.dumps(run(p.parse_args()), sort_keys=True))
