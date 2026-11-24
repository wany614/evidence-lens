"""Create a fixed-layout deployment-only bundle and scope proof from pinned acquisition and reconstruction."""

import argparse
import json
import os
from pathlib import Path

from evidence_lens.multisource_io import Case
from evidence_lens_bundle.bundle import Bundle
from evidence_lens_bundle.calibration import declarations
from evidence_lens_bundle.common import SEMANTICS, profile, require, write
from evidence_lens_bundle.legacy import adapt_authenticated_case
from evidence_lens_bundle.proof import geometry, run_pin
from t0_records import resolve


def create(request, roots, out):
    pins = json.loads(request.read_bytes())["roles"]
    paths = {k: resolve(roots, p) for k, p in pins.items()}
    j = {k: json.loads(path.read_bytes()) for k, path in paths.items()}
    r = j["readback"]
    s = j["source"]
    co = j["cohort"]
    br = j["freeze"]
    lock = j["method"]
    g = j["generation"]
    p = j["render"]
    d = j["depth_probe"]
    case = Case(paths["case"].parent, pins["case"]["sha256"])
    cond = case.condition
    obj = r["object_id"]
    gen = tuple(case.generator)
    require(
        r["schema"] == "el-b2-reconstruction-readback-1"
        and r["verified"]
        and r["status"] == "succeeded"
        and r["scientific_sample"]
        and not r["forced_fallback_fixture"]
        and r["selection"] is None
        and r["layout"] == cond + "-fixed"
        and r["target_reference_files_opened"] == 0,
        "unsupported or unverified reconstruction",
    )
    require(
        obj == s["object_id"]
        and obj in co["splits"]["C6"]
        and obj not in co["splits"]["T1"]
        and r["condition"] == s["condition"] == cond
        and r["source"] == pins["source"]
        and r["freeze"] == pins["freeze"]
        and r["cohort"] == s["cohort"] == lock["cohort"] == pins["cohort"],
        "unregistered object, layout or parent",
    )
    require(
        r["case_sha256"] == case.sha256
        and pins["case"]["run_id"] == pins["readback"]["run_id"] == r["run_id"],
        "case does not share successful checked return",
    )
    require(
        r["origins"]["generation"] == run_pin(pins["generation"])
        and r["origins"]["render"] == run_pin(pins["render"])
        and g["run_id"]
        == case.meta["reconstruction_run_id"]
        == p["reconstruction_run_id"]
        == pins["generation"]["run_id"]
        and r["source_renderer_id"] == pins["render"]["run_id"],
        "wrong generation or render",
    )
    scope = dict(lock["scope"])
    require(
        lock["schema"] == "el-b2-calibration-method-1"
        and lock["method"] == "MS-L1"
        and lock["target_rgb_reads"] == 0
        and all(
            lock["models"]["MS-L1"][k] == profile()["models"]["MS-L1"][k]
            for k in ("model_id", "model_index_sha256", "weights_sha256", "threshold")
        )
        and all(
            lock["scope"][k] == profile()["models"]["MS-L1"][k]
            for k in ("feature_schema_sha256", "preprocessing_sha256")
        ),
        "wrong frozen deployment method",
    )
    require(
        r["depth_probe"] == pins["depth_probe"]
        and br["origins"]["depth-probe.json"] == pins["depth_probe"]
        and case.pred["depth_verified"]
        and case.pred["depth_probe_sha256"] == pins["depth_probe"]["sha256"]
        and d["passed"],
        "unbound ED check",
    )
    require(
        r["recipe_sha256"]
        == br["recipe_sha256"]
        == pins["recipe"]["sha256"]
        == g["identity"]["recipe_sha256"]
        == p["recipe_sha256"]
        == scope["recipe_sha256"]
        and r["completed_steps"] == j["recipe"]["stop_step"] == scope["stop_step"],
        "wrong reconstruction recipe or endpoint",
    )
    require(
        r["cap"] == scope["reconstructor"]["gaussian_cap"]
        and r["optimizer_seed"]
        == br["optimizer_seed"]
        == j["recipe"]["seed"]
        == scope["reconstructor"]["optimizer_seed"]
        and br["worker_tree_sha256"] == scope["reconstructor"]["worker_tree_sha256"]
        and r["worker"]["gsplat"] == scope["renderer"]["version"],
        "wrong producer, budget or seed",
    )
    require(
        s["lighting_id"] == scope["lighting_id"]
        and list(gen) == s["source_ids"]
        and s["source_view_ids"] == co["source_view_ids"]
        and s["target_view_ids"] == co["target_view_ids"]
        and r["official_by_slot"] == {x: s["source_view_ids"][x] for x in gen},
        "different fixed-layout physical source IDs",
    )
    for n in ("source_cameras", "render_cameras"):
        key = n[:-1] + "_sha256"
        require(
            case.meta[n]["sha256"]
            == s[key]
            == r[key]
            == case.pred[key]
            == co["camera"]["conditions"][cond][n]["sha256"],
            "camera identity mismatch: " + n,
        )
    require(
        all(s["sources"][sid]["rgb"] == case.meta["source_images"][sid] for sid in gen)
        and case.pred["source_rgb_sha256"] == g["identity"]["source_rgb_sha256"] == p["source_rgb_sha256"],
        "wrong actual sources",
    )
    stage = p["checkpoints"][str(r["completed_steps"])]
    require(
        stage["checkpoint_sha256"]
        == case.meta["checkpoint_sha256"]
        == r["checkpoints"][-1]["checkpoint_sha256"]
        and set(case.meta["renders"]) == set(stage["views"])
        and all(case.meta["renders"][x]["sha256"] == stage["views"][x]["sha256"] for x in stage["views"]),
        "wrong selected native render or checkpoint",
    )
    rig = {k: pins[k] for k in ("camera_plan", "crop", "metric")}
    require(
        co["camera"]["plan_sha256"] == s["camera_plan_sha256"] == rig["camera_plan"]["sha256"]
        and co["camera"]["crop_sha256"] == s["crop_sha256"] == rig["crop"]["sha256"],
        "unmatched original camera layout",
    )
    physical = {**s, "rig_registration": run_pin(rig["camera_plan"]), "source_images": s["sources"]}
    geom = geometry(case, physical, j, rig)
    for field, val in (("rgb_encoding", "rgb"), ("depth_semantics", "z"), ("alpha_semantics", "alpha")):
        require(case.pred[field] == SEMANTICS[val] == scope[field], "calibration channel semantics differ")
    scope.update(
        source_ids=list(gen),
        source_camera_sha256=case.pred["source_camera_sha256"],
        render_camera_sha256=case.pred["render_camera_sha256"],
    )
    destination = out / "bundle"
    adapt_authenticated_case(
        case,
        destination,
        input_domain="DTU fixed-layout source photographs, lighting 3 and the registered metric cameras",
        generator={"run_id": pins["generation"]["run_id"], "receipt_sha256": pins["generation"]["sha256"]},
        renderer={"run_id": pins["render"]["run_id"], "receipt_sha256": pins["render"]["sha256"]},
        geometry=geom,
        proof_sha256=pins["readback"]["sha256"],
    )
    result = Bundle(destination)
    write(
        out / "scope.json",
        {
            "schema": "el-b2-deployment-scope-1",
            "run_id": os.getenv("EL_RUN_ID"),
            "verified": True,
            "condition": cond,
            "object_id": obj,
            "target_reference_photographs_opened": 0,
            "bundle": declarations(result),
            "scope": scope,
            "origins": pins,
        },
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--request", type=Path, required=True)
    ap.add_argument("--root", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    create(a.request, a.root, a.out)
