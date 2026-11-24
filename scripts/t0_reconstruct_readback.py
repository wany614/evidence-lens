"""Verify the original Windows run pair and build the deployment CASE using evidence from this cell."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

from evidence_lens._bundle import encoded, load, sha
from evidence_lens._verify import require, verify_bundle
from evidence_lens.multisource_io import Case
from recovery_outputs import (
    actual_inputs,
    calibration_readback,
    checkpoint_audit,
    png_readback,
    trace_summary,
    verify_code,
)
from recovery_receive import authenticate_native


CHECKPOINTS = (0, 500, 1500, 3000, 5000)
TARGETS = tuple(f"v{i:02d}" for i in range(1, 7))
SOURCES = {"M2": ("s00", "s03"), "M4": ("s00", "s01", "s02", "s03")}


def same_pin(pin):
    return {k: pin[k] for k in ("run_id", "manifest_sha256")}


def spec(path):
    return {"bytes": path.stat().st_size, "sha256": sha(path)}


class Original:
    def __init__(self, store, pin):
        self.pin = same_pin(pin)
        self.root = store / "runs" / pin["run_id"]
        result = verify_bundle(self.root)
        require(result["manifest_sha256"] == pin["manifest_sha256"], "Original manifest differs")
        self.run = result["run"]
        self.files = {x["path"]: x for x in result["manifest"]["files"]}
        self.parents = result["inputs"]["parents"]

    def file(self, name):
        f = self.root / name
        require(
            name in self.files and spec(f) == {k: self.files[name][k] for k in ("bytes", "sha256")},
            "Changed sealed member " + name,
        )
        return f

    def doc(self, name):
        return load(self.file(name))


def copy(base, original, name):
    dest = base / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, dest)
    require(sha(dest) == sha(original), "Projected numerical bytes differ")
    return {"path": name, **spec(dest)}


def finish(roots, cfg, out):
    bundle = {k: Original(roots, p) for k, p in cfg["originals"].items()}
    dispatch = bundle["dispatch"]
    job = dispatch.doc("payload/job.json")
    ready = dispatch.doc("payload/READY.json")
    require(
        sha(dispatch.file("payload/job.json")) == cfg["job_sha256"] == ready["job_sha256"], "Wrong dispatch"
    )
    obj, cond = ready["object_id"], ready["condition"]
    ids = SOURCES[cond]
    forced = ready["forced_fallback_fixture"]
    gp = bundle["generation"]
    rp = bundle.get("render")
    init = bundle["init"].doc("payload/init.json")
    require(
        gp.run["run_kind"] == "generation"
        and gp.run["trial_id"] == ready["trial_id"]
        and gp.run["experiment_id"] == job["experiment_id"]
        and gp.doc("config.resolved.yaml")["cell_sha256"] == cfg["job_sha256"]
        and sorted(gp.parents, key=lambda r: r["run_id"])
        == sorted(job["parents"], key=lambda r: r["run_id"]),
        "Wrong optimizer identity or parents",
    )
    if rp:
        rcfg = rp.doc("config.resolved.yaml")
        require(
            rp.run["run_kind"] == "render"
            and rp.run["trial_id"] == ready["trial_id"]
            and rp.run["experiment_id"] == job["experiment_id"]
            and rcfg["generation"] == gp.pin
            and rcfg["cell_sha256"] == cfg["job_sha256"]
            and sorted(rp.parents, key=lambda r: r["run_id"])
            == sorted([*job["parents"], gp.pin], key=lambda r: r["run_id"]),
            "Wrong selected renderer or parents",
        )
    states = {k: bundle[k].run["execution_status"] for k in ("generation", "render") if k in bundle}
    info = {
        "schema": "el-t0-reconstruction-readback-1",
        "run_id": os.environ["EL_RUN_ID"],
        "object_id": obj,
        "condition": cond,
        "trial_id": ready["trial_id"],
        "forced_fallback_fixture": forced,
        "scientific_sample": not forced,
        "source": ready["source"],
        "freeze": ready["freeze"],
        "recipe_sha256": job["recipe_sha256"],
        "initialization": ready["initialization"],
        "actual_method": init["actual_method"],
        "initial_points": init["N"],
        "natural_eligibility": init["eligibility"],
        "target_reference_files_opened": 0,
        "risk_scorings": 0,
        "origins": cfg["originals"],
        "states": states,
        "job_sha256": cfg["job_sha256"],
        "return_summary_sha256": cfg["return_sha256"],
    }
    if set(states.values()) != {"succeeded"} or rp is None:
        require(
            "succeeded" != states.get("generation") or rp is not None,
            "Missing renderer after successful generation",
        )
        info.update(
            verified=False,
            status="failed",
            failure_reasons={
                k: bundle[k].run["failure_reason"] for k, v in states.items() if v != "succeeded"
            },
        )
        (out / "reconstruction-readback.json").write_bytes(encoded(info))
        return
    bridge = bundle["bridge"]
    source = bundle["source"]
    recipe = bridge.doc("payload/recipe.json")
    g = gp.doc("payload/reconstruction.json")
    issued = gp.doc("payload/issued.json")
    r = rp.doc("payload/outputs.json")
    gcfg = gp.doc("config.resolved.yaml")
    ident = g["identity"]
    worker = ident["worker"]
    sc = source.file("payload/source_cameras.json")
    rc = source.file("payload/render_cameras.json")
    source_images = {s: source.file("payload/sources/" + s + ".png") for s in ids}
    ssha = {s: sha(f) for s, f in source_images.items()}
    require(
        ident == issued["identity"] == gcfg["science"] == rcfg["science"]
        and all(r[k] == val for k, val in ident.items())
        and issued["run_id"] == g["run_id"] == gp.pin["run_id"] == r["reconstruction_run_id"]
        and ident["object_id"] == obj
        and ident["condition"] == cond
        and ident["recipe_sha256"] == job["recipe_sha256"] == sha(bridge.file("payload/recipe.json"))
        and ident["generator_source_ids"] == list(ids)
        and ident["source_rgb_sha256"] == ssha
        and ident["source_camera_sha256"] == sha(sc)
        and ident["init_sha256"] == sha(bundle["init"].file("payload/init.npz"))
        and ident["init_json_sha256"] == sha(bundle["init"].file("payload/init.json"))
        and r["render_camera_sha256"] == sha(rc)
        and r["rgb_encoding"] == "display_srgb_float32"
        and r["depth_semantics"] == "opacity-normalized expected camera Z in metres",
        "Different frozen science or calibrated pixels",
    )
    require(
        g["first_completed_step"] == 0
        and g["last_completed_step"]
        == g["updates_in_process"]
        == g["schedule_horizon"]
        == g["job_stop_step"]
        == 5000
        and g.get("resume_origin") is None
        and set(issued["checkpoints"]) == set(map(str, CHECKPOINTS))
        and issued["checkpoints"] == g["checkpoints"]
        and set(r["checkpoints"]) == {"5000"}
        and g["actual_method"] == init["actual_method"],
        "Incomplete registered 5000-step schedule",
    )
    tree = verify_code(bridge.file("payload/worker.zip"), gp.root, worker)
    require(
        verify_code(bridge.file("payload/worker.zip"), rp.root, worker)
        == tree
        == ready["worker_tree_sha256"],
        "Generator and renderer sources differ",
    )
    probe = bridge.doc("payload/depth-probe.json")
    qual = bridge.doc("payload/resource-probe.json")
    previous = bridge.doc("payload/bridge.json")
    historical = bridge.doc("payload/historical/" + previous["prior_fits"][0]["run_id"] + ".json")[
        "identity"
    ]["worker"]
    require(
        probe["passed"]
        and qual["passed"]
        and qual["geometry"]["actual"] == probe
        and qual["worker_tree_sha256"] == tree
        and worker == historical
        and qual["actual_worker_snapshots"]["projection"] == worker["source_sha256"]
        and probe["resources"]["gsplat"] == worker["gsplat"] == "1.5.3",
        "Measured renderer ED proof is for another executable",
    )
    pg = {k: v["sha256"] for k, v in actual_inputs(gp.root).items()}
    pr = {k: v["sha256"] for k, v in actual_inputs(rp.root).items()}
    require(
        pg
        == gcfg["role_hashes"]
        == {
            "recipe": ident["recipe_sha256"],
            "init": ident["init_sha256"],
            "initmeta": ident["init_json_sha256"],
            "cameras": sha(sc),
            **ssha,
        }
        and pr
        == rcfg["role_hashes"]
        == {"cameras": sha(rc), "c5000": issued["checkpoints"]["5000"]["checkpoint_sha256"]},
        "Unexpected optimizer or renderer inputs",
    )
    calibration = calibration_readback(load(sc), load(rc), ids)
    require(
        g["endpoint_frame_sha256"] == r["endpoint_frame_sha256"] == calibration[1], "Endpoint frame differs"
    )
    trace = trace_summary(gp.file("payload/steps.jsonl"), g, ids, recipe)
    checkpoints = []
    for step in CHECKPOINTS:
        point = issued["checkpoints"][str(step)]
        ck = gp.file(f"payload/checkpoints/step-{step:04d}.pt")
        require(
            point["path"] == ck.relative_to(gp.root / "payload").as_posix()
            and point["bytes"] == ck.stat().st_size
            and point["checkpoint_sha256"] == sha(ck)
            and point["producer_run_id"] == gp.pin["run_id"]
            and 0 < point["gaussians"] <= 20000,
            "Checkpoint not an original issued member",
        )
        counts = {s: (step + len(ids) - 1 - i) // len(ids) for i, s in enumerate(ids)}
        d = checkpoint_audit(
            ck,
            genid=gp.pin["run_id"],
            identity=ident,
            original=init,
            recipe=recipe,
            step=step,
            n=point["gaussians"],
            counts=counts,
            calibration=calibration,
        )
        checkpoints.append(
            {
                "step": step,
                "N": point["gaussians"],
                "checkpoint_sha256": sha(ck),
                "parameter_sha256": d,
                "source_counts": counts,
            }
        )
    selected = r["checkpoints"]["5000"]
    orig = rcfg["artifact_origins"]["c5000"]
    require(
        orig == {**gp.pin, "member": "payload/checkpoints/step-5000.pt", "execution_status": "succeeded"}
        and selected["step"] == 5000
        and selected["producer_run_id"] == gp.pin["run_id"]
        and selected["checkpoint_sha256"] == checkpoints[-1]["checkpoint_sha256"]
        and selected["gaussians"] == checkpoints[-1]["N"]
        and set(selected["views"]) == set(ids + TARGETS),
        "Endpoint renderer checkpoint or view roster changed",
    )
    base = out / "case"
    base.mkdir()
    cams = {
        k: copy(base, path, k + ".json") for k, path in [("source_cameras", sc), ("render_cameras", rc)]
    }
    photos = {s: copy(base, path, "sources/" + s + ".png") for s, path in source_images.items()}
    frames = {}
    diagnostics = {}
    for sid in ids + TARGETS:
        v = selected["views"][sid]
        name = "step-5000/" + sid
        require(
            v["path"] == name + ".npz" and v["png_path"] == name + ".png",
            "Noncanonical original native frame",
        )
        native = rp.file("payload/" + v["path"])
        png = rp.file("payload/" + v["png_path"])
        require(
            spec(native) == {k: v[k] for k in ("bytes", "sha256")}
            and spec(png) == {"bytes": v["png_bytes"], "sha256": v["png_sha256"]},
            "Original native member differs",
        )
        pixels = authenticate_native(native)
        png_readback(png, native)
        require(
            pixels["pixels"] == pixels["valid_opacity"] == pixels["finite_display_rgb"] == 65536
            and pixels["nonzero_opacity_invalid_depth"] == pixels["zero_opacity_non_nan_depth"] == 0,
            "Invalid native output semantics",
        )
        diagnostics[sid] = pixels
        frames[sid] = copy(base, native, "predictions/" + sid + ".npz")
    pred = {
        "schema": "el-stage3-prediction-1",
        "reconstruction_run_id": gp.pin["run_id"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "generator_source_ids": list(ids),
        "target_ids": list(TARGETS),
        "source_camera_sha256": sha(sc),
        "render_camera_sha256": sha(rc),
        "source_rgb_sha256": ssha,
        "rgb_encoding": "display_srgb_0_1",
        "depth_semantics": "expected_camera_z_m",
        "alpha_semantics": "accumulated_opacity",
        "depth_verified": True,
        "depth_probe_sha256": previous["origins"]["depth-probe.json"]["sha256"],
        "views": {k: {a: v[a] for a in ("bytes", "sha256")} for k, v in frames.items()},
        "failed_views": {},
    }
    (base / "prediction.json").write_bytes(encoded(pred))
    case = {
        "schema": "el-stage3-deploy-case-1",
        "condition": cond,
        "generator_source_ids": list(ids),
        "target_ids": list(TARGETS),
        "source_images": photos,
        **cams,
        "renders": frames,
        "prediction": {"path": "prediction.json", **spec(base / "prediction.json")},
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "reconstruction_run_id": gp.pin["run_id"],
    }
    (base / "case.json").write_bytes(encoded(case))
    ck = Case(base, sha(base / "case.json"))
    require(
        all(ck.frame(v) is not None for v in ids + TARGETS), "Deployed CASE cannot reopen a predicted view"
    )
    info.update(
        verified=True,
        status="succeeded",
        completed_steps=5000,
        cap=20000,
        trace=trace,
        checkpoints=checkpoints,
        frames=diagnostics,
        source_renderer_id=rp.pin["run_id"],
        worker=worker,
        depth_probe=previous["origins"]["depth-probe.json"],
        source_camera_sha256=sha(sc),
        render_camera_sha256=sha(rc),
        scene_scale_m=init["scene_scale_m"],
        case_sha256=ck.sha256,
    )
    (out / "reconstruction-readback.json").write_bytes(encoded(info))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("store", "config", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    finish(a.store, load(a.config), a.out)
