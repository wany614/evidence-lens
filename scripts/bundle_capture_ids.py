"""Exercise native non-contiguous IDs on a previously exposed, private DTU deployment Case."""

import argparse
import json
import os
from pathlib import Path
import tempfile

from bundle_build import environment, run
from bundle_original import check, put, sha
from bundle_score_record import released_models


def spec(rel, data):
    return {"path": rel, "bytes": len(data), "sha256": sha(data)}


def job(a):
    # This worker reads only sealed deployment evidence and camera/transform documents.
    case = json.loads((a.case / "case.json").read_bytes())
    prov = json.loads(a.provenance.read_bytes())
    plan = json.loads(a.plan.read_bytes())
    metric = json.loads(a.metric.read_bytes())
    crop = json.loads(a.crop.read_bytes())
    pred = json.loads((a.case / case["prediction"]["path"]).read_bytes())
    check(
        prov["schema"] == "el-dtu-case-provenance-1"
        and prov["object_id"] == "set1"
        and prov["condition"] == case["condition"] == "M4"
        and sha((a.case / "case.json").read_bytes()) == prov["case_sha256"],
        "only the explicitly exposed D1 set1 M4 is authorized",
    )
    check(
        prov["camera_plan_sha256"] == sha(a.plan.read_bytes())
        and plan["schema"] == "el-dtu-camera-plan-1"
        and plan["metric_frame_sha256"] == sha(a.metric.read_bytes())
        and plan["crop_sha256"] == sha(a.crop.read_bytes())
        and plan["camera_sha256"] == prov["camera_sha256"] == case["render_cameras"]["sha256"],
        "wrong official calibration",
    )
    check(
        "Rectified" in plan["evidence"]["rectification"]
        and plan["converter"] == crop["id"] == "dtu-crop256-v1"
        and metric["metres_per_native_unit"] == 0.001
        and metric["additional_scale"] == 1,
        "unknown metric/rectification provenance",
    )
    for k in ("source_view_ids", "target_view_ids"):
        check(prov[k] == plan[k], "source/target mapping changed")
    names = {s: str(n) for s, n in {**prov["source_view_ids"], **prov["target_view_ids"]}.items()}
    order = [names[k] for k in case["generator_source_ids"]]
    targets = [names[k] for k in case["target_ids"]]
    check(
        order == ["24", "23", "17", "22"] and targets == ["16", "18", "25", "37", "35", "36"],
        "unexpected object/view selection",
    )
    check(
        pred["checkpoint_sha256"] == case["checkpoint_sha256"]
        and pred["reconstruction_run_id"] == prov["generation"]["run_id"]
        and pred["run_id"] == prov["render"]["run_id"],
        "render and reconstruction differ",
    )
    b = a.out / "bundle"
    b.mkdir()

    def original(s):
        raw = (a.case / s["path"]).read_bytes()
        check(len(raw) == s["bytes"] and sha(raw) == s["sha256"], s["path"])
        return raw

    def add(path, raw):
        p = b / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        return spec(path, raw)

    original(case["prediction"])
    camera = json.loads(original(case["render_cameras"]))
    src_cam = json.loads(original(case["source_cameras"]))
    check(
        all(camera["views"][n] == src_cam["views"][n] for n in case["generator_source_ids"]),
        "source poses differ",
    )
    camera["views"] = {
        names[n]: {**m, "projection": "pinhole", "distortion": "none"} for n, m in camera["views"].items()
    }
    put(b / "cameras.json", camera)
    cal = spec("cameras.json", (b / "cameras.json").read_bytes())
    photos = {}
    for s in case["generator_source_ids"]:
        check(
            prov["source_images"][s]["official_view_id"] == int(names[s])
            and case["source_images"][s]["sha256"]
            == prov["source_images"][s]["rgb"]["sha256"]
            == pred["source_rgb_sha256"][s],
            "input photo identity",
        )
        photos[names[s]] = add("sources/" + names[s] + ".png", original(case["source_images"][s]))
    frames = {}
    for s in [*case["generator_source_ids"], *case["target_ids"]]:
        r = case["renders"][s]
        check(
            pred["views"][s]["sha256"] == r["sha256"] and pred["views"][s]["bytes"] == r["bytes"],
            "render file identity",
        )
        frames[names[s]] = {"status": "ready", "file": add("renders/" + names[s] + ".npz", original(r))}
    producer = {
        "schema": "el-render-evidence-1",
        "generator": {
            "run_id": pred["reconstruction_run_id"],
            "receipt_sha256": pred["reconstruction_sha256"],
        },
        "renderer": {"run_id": pred["run_id"], "receipt_sha256": case["prediction"]["sha256"]},
        "checkpoint_sha256": pred["checkpoint_sha256"],
        "camera_sha256": cal["sha256"],
        "source_rgb_sha256": {s: r["sha256"] for s, r in photos.items()},
        "source_order": order,
        "target_order": targets,
        "semantics": {
            "rgb": pred["rgb_encoding"],
            "z": pred["depth_semantics"],
            "alpha": pred["alpha_semantics"],
        },
        "depth_verified": pred["depth_verified"],
        "depth_probe_sha256": pred["depth_probe_sha256"],
        "frames": frames,
    }
    put(b / "producer.json", producer)
    doc = {
        "schema": "el-evidence-bundle-1",
        "creation": {
            "run_id": os.environ["EL_RUN_ID"],
            "implementation": "DTU recorded camera-ID import",
            "code_sha256": sha(Path(__file__).read_bytes()),
            "config_sha256": sha(a.provenance.read_bytes()),
            "original_case_sha256": prov["case_sha256"],
        },
        "input_domain": "Private exposed DTU set1 M4, official camera IDs",
        "condition": "M4",
        "source_order": order,
        "target_order": targets,
        "primary_source_id": names["s00"],
        "endpoint_source_ids": [str(n) for n in metric["endpoint_ids"]],
        "camera": cal,
        "geometry": {
            "prepared_camera_and_z_unit": "m",
            "meters_per_original_unit": metric["metres_per_native_unit"],
            "metric_evidence": plan["metric_frame_sha256"],
            "crop": f"Original {crop['width']} x {crop['height']}; ({crop['left']}, {crop['top']}) side {crop['side']}; manifest SHA-256 {plan['crop_sha256']}",
            "resize": crop["resampling"],
            "undistortion": plan["evidence"]["rectification"],
            "scoring_resampling": "none",
        },
        "producer": spec("producer.json", (b / "producer.json").read_bytes()),
        "sources": photos,
        "risk_profile": {
            "id": "v0.3.0-rc.1/original-thresholds",
            "sha256": "99587dd70b67a84debb81fa1dc84e93729c8980bc4531edecfb56112a15eb191",
            "release_index_sha256": "150b04e702f5386b5808ac17545cd073e3d4d837ea789e6ae167168ac086084d",
        },
    }
    put(b / "bundle.json", doc)
    with tempfile.TemporaryDirectory(dir=a.out.parent) as tmp:
        folder = Path(tmp)
        cli, inventory = environment(folder, a.wheel)
        released_models(a.cpu, folder / "models")
        verified = json.loads(run([cli, "validate", "--bundle", b]))
        methods = {}
        for method in ("MS-L1", "B-L1-transfer"):
            out = a.out / "scores" / method
            run(
                [
                    cli,
                    "score",
                    "--bundle",
                    b,
                    "--bundle-sha256",
                    verified["bundle_sha256"],
                    "--model",
                    method,
                    "--models",
                    folder / "models",
                    "--out",
                    out,
                ]
            )
            score = json.loads((out / "scores.json").read_bytes())
            methods[method] = {
                "score_sha256": sha((out / "scores.json").read_bytes()),
                "scorer_source_ids": score["scorer_source_ids"],
                "counts": score["counts"],
            }
        check(
            methods["MS-L1"]["scorer_source_ids"] == order
            and methods["B-L1-transfer"]["scorer_source_ids"] == ["24"],
            "unapproved source selection",
        )
        put(
            a.out / "capture-ids.json",
            {
                "schema": "el-bundle-external-camera-ids-1",
                "run_id": os.getenv("EL_RUN_ID"),
                "case_sha256": prov["case_sha256"],
                "camera_plan_sha256": sha(a.plan.read_bytes()),
                "physical_object": "set1",
                "scope": "exposed engineering example",
                "target_reference_inputs": 0,
                "bundle": verified,
                "prepared_meter_scale": metric["metres_per_native_unit"],
                "models": methods,
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("case", "provenance", "plan", "crop", "metric", "wheel", "cpu", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    job(p.parse_args())
