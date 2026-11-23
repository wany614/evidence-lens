"""Adapt only the selected recovery checkpoint's native renders and allowed images to the published CPU CASE contract."""

import argparse
import json
import os
import shutil
from pathlib import Path
from dtu_score import VIEWS
from evidence_lens.formal_inputs import require, verified_file
from evidence_lens.multisource_io import Case, decode_prediction
from evidence_lens.scoring_io import new_output, write_json, file_hash
from recovery_eval_io import Outputs, authority, recipe, permit_frozen, original
from vision_sources import signed


def artifact(base, p):
    return {"path": p.relative_to(base).as_posix(), "bytes": p.stat().st_size, "sha256": file_hash(p)}


def copy(base, p, rel):
    q = base / rel
    q.parent.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(p, q)
    require(file_hash(p) == file_hash(q), "case projection changed original bytes")
    return artifact(base, q)


def check_original_case(record, case, req):
    pin = req["original_case"]
    meta = json.loads(record.open("payload/case-provenance.json").read_bytes())
    require(
        pin["member"] == "payload/case/case.json"
        and meta["schema"] == "el-dtu-case-provenance-1"
        and meta["run_id"] == pin["run_id"] == record.pin["run_id"]
        and meta["case_sha256"] == case.sha256 == pin["sha256"]
        and meta["object_id"] == req["object_id"]
        and meta["condition"] == case.condition == req["condition"]
        and meta["arm"] == "main-256"
        and meta["optimization_steps"] == 256
        and {s: source["rgb"] for s, source in meta["source_images"].items()} == case.meta["source_images"],
        "original D1 identity, source photographs, condition or archived arm differs",
    )
    return {
        **record.pin,
        "member": "payload/case-provenance.json",
        "sha256": record.files["payload/case-provenance.json"]["sha256"],
    }


def work(roots, req, out, rid):
    require(
        req["schema"] == "el-recovery-case-request-1"
        and set(req)
        == {
            "schema",
            "object_id",
            "condition",
            "protocol",
            "recipe",
            "freeze",
            "original_case",
            "outputs",
            "depth_probe",
        },
        "case adaptation has an unexpected input role",
    )
    auth = authority(roots, req)
    require(
        req["object_id"] in ("set1", "set6") and req["condition"] in ("M2", "M4"),
        "the synthetic control is registered only for reconstruction quality",
    )
    recipe(roots, req)
    f = permit_frozen(roots, req)
    source_run, p, _ = original(roots, req["original_case"])
    old = Case(p.parent, req["original_case"]["sha256"])
    origin = check_original_case(source_run, old, req)
    data = Outputs(roots, req, old)
    require(
        data.data.get("original_case") == req["original_case"]
        and data.data.get("original_case_provenance") == origin
        and data.data["object_id"] == req["object_id"]
        and data.data["condition"] == req["condition"] == old.condition
        and data.data["recipe_sha256"] == req["recipe"]["sha256"] == f["recipe_sha256"]
        and data.data["protocol_sha256"] == auth["protocol_sha256"] == f["protocol_sha256"],
        "normalized endpoint, original deployment case and frozen recipe must identify the same cell",
    )
    step = f["stop_step"]
    row = data.data["checkpoints"].get(str(step))
    require(row is not None and row["step"] == step, "frozen checkpoint is absent")
    require(
        data.data["status"] == "completed" and data.data["completed_steps"] >= step,
        "interim resource cutoffs cannot substitute for the frozen stopped fit",
    )
    if req["object_id"] == "set1":
        require(
            f["selection"]["winner"]["checkpoint_sha256"][old.condition] == row["checkpoint_sha256"],
            "deployment selected a different development checkpoint",
        )
    if req["object_id"] == "set6":
        require(
            old.condition in f["authorized_new_conditions"]["set6"] and data.data["completed_steps"] == step,
            "regression condition or its registered exact step differs",
        )
    _, _, probe = original(roots, req["depth_probe"])
    require(probe.get("passed") is True, "independent original renderer projection/ED probe required")
    require(
        data.data.get("depth_verified") is True
        and data.data.get("depth_probe") == req["depth_probe"]
        and data.data.get("depth_semantics") == "expected_camera_z_m"
        and data.data.get("rgb_encoding") == "display_srgb_0_1"
        and data.data.get("alpha_semantics") == "accumulated_opacity",
        "render output semantic evidence absent",
    )
    with new_output(out) as dest:
        base = dest / "case"
        base.mkdir()
        oldmeta = old.meta
        cams = {
            k: copy(base, verified_file(old.root, oldmeta[k]["path"], oldmeta[k]), oldmeta[k]["path"])
            for k in ("source_cameras", "render_cameras")
        }
        sources = {
            s: copy(base, verified_file(old.root, e["path"], e), "sources/" + s + ".png")
            for s, e in oldmeta["source_images"].items()
        }
        renders = {}
        for v in (*old.generator, *VIEWS):
            actual = data.asset(step, v)
            require(
                actual is not None and actual.suffix.lower() == ".npz",
                "frozen risk scoring requires real complete source and target RGB/ED/alpha renders",
            )
            decode_prediction(actual.read_bytes(), old.cameras[v])
            renders[v] = copy(base, actual, "predictions/" + v + ".npz")
        pred = {
            "schema": "el-stage3-prediction-1",
            "wrapper_schema": "el-recovery-render-adapter-1",
            "reconstruction_run_id": data.data["reconstruction_run_id"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "generator_source_ids": list(old.generator),
            "target_ids": list(VIEWS),
            "source_camera_sha256": cams["source_cameras"]["sha256"],
            "render_camera_sha256": cams["render_cameras"]["sha256"],
            "source_rgb_sha256": {s: e["sha256"] for s, e in sources.items()},
            "rgb_encoding": "display_srgb_0_1",
            "depth_semantics": "expected_camera_z_m",
            "alpha_semantics": "accumulated_opacity",
            "depth_verified": True,
            "depth_probe_sha256": req["depth_probe"]["sha256"],
            "views": {v: {k: r[k] for k in ("sha256", "bytes")} for v, r in renders.items()},
            "failed_views": {},
        }
        pp = base / "prediction.json"
        write_json(pp, pred)
        cm = {
            "schema": "el-stage3-deploy-case-1",
            "condition": old.condition,
            "generator_source_ids": list(old.generator),
            "target_ids": list(VIEWS),
            "source_images": sources,
            **cams,
            "renders": renders,
            "prediction": artifact(base, pp),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "reconstruction_run_id": pred["reconstruction_run_id"],
        }
        cp = base / "case.json"
        write_json(cp, cm)
        ref = Case(base, file_hash(cp))
        require(
            all(ref.frame(v) is not None for v in (*old.generator, *VIEWS)),
            "unreadable projected native case",
        )
        d = {
            "schema": "el-recovery-deployment-case-1",
            "run_id": rid,
            "case_sha256": file_hash(cp),
            "case_member": "payload/case/case.json",
            "object_id": req["object_id"],
            "condition": old.condition,
            "recipe_sha256": data.data["recipe_sha256"],
            "protocol_sha256": auth["protocol_sha256"],
            "freeze": req["freeze"],
            "step": step,
            "checkpoint_sha256": row["checkpoint_sha256"],
            "parameter_source": data.data["reconstruction_run_id"],
            "source_images": pred["source_rgb_sha256"],
            "original_provenance": origin,
            "render_sha256": {v: r["sha256"] for v, r in renders.items()},
            "originals": {
                k: req[k] for k in ("protocol", "recipe", "original_case", "outputs", "depth_probe")
            },
            "reference_reads": 0,
        }
        write_json(dest / "case-provenance.json", d)
    return d


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("store", "shelf", "request", "output"):
        p.add_argument("--" + n, required=True, type=Path)
    p.add_argument("--request-sha256", required=True)
    a = p.parse_args()
    print(
        json.dumps(
            work(
                (a.store, a.shelf), signed(a.request, a.request_sha256), a.output, os.environ.get("EL_RUN_ID")
            )["case_sha256"]
        )
    )
