"""Render matched-grid R1 source evidence from a sealed, independently generated Blender scene."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import bpy
import numpy as np

REFERENCE = Path(__file__).resolve().parents[1] / "src/evidence_lens/blender_reference"
sys.path.insert(0, str(REFERENCE))
from passes import display_srgb, geometry_passes, png_codes, png_pixels, read_passes, save_png, sha256, write_json
from scene import camera_record, evaluated_bvh, render_settings, validate_camera
from validation import render


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def produce(plan, blend, settings, cameras, out):
    cfg = read(plan)
    for role, path in (("blend", blend), ("settings", settings), ("cameras", cameras)):
        if sha256(path) != cfg["pins"][role]:
            raise ValueError(f"Unexpected sealed {role}")
    spec = read(settings)
    reference = spec["reference"]
    expected = read(cameras)["views"]["source"]
    if (spec["scene_family_id"] != cfg["scene_family_id"] or
            list(bpy.app.version) != reference["blender_version"]):
        raise ValueError("Scene identity or Blender build differs from the reference")
    bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False, use_scripts=False)
    scene = bpy.context.scene
    camera = scene.camera
    if (scene.name != reference["scene_name"] or camera.name != "source" or
            (expected["width"], expected["height"]) != (256, 256) or
            scene.render.resolution_percentage != 100):
        raise ValueError("Unexpected R1 source scene or render size")
    scene.render.resolution_x, scene.render.resolution_y = expected["width"], expected["height"]
    scene.cycles.seed = reference["seed"]
    measured = camera_record(scene, camera, expected["purpose"])
    for field in ("width", "height", "projection", "sensor_fit", "distortion"):
        if measured[field] != expected[field]:
            raise ValueError(f"Camera mismatch in {field}")
    residuals = {}
    for field in ("K", "K_edge", "w2c", "c2w", "clip_m", "lens_shift", "pixel_aspect"):
        residuals[field] = float(np.max(np.abs(np.array(measured[field]) - expected[field])))
    if max(residuals.values()) > cfg["camera_absolute_tolerance"]:
        raise ValueError(f"Camera mismatch in frozen transforms: {residuals}")
    projected = validate_camera(scene, camera, expected, [o["center"] for o in reference["objects"]])
    tolerances = reference["validation"]
    if projected["max_error_px"] > tolerances["projection_tolerance_px"]:
        raise ValueError("Independent Blender landmark projection exceeds frozen tolerance")
    out.mkdir(parents=True, exist_ok=True)
    native = out / "source_blender.png"
    render(scene, out / "source.exr", native)
    linear, position, object_id, channels = read_passes(out / "source.exr")
    rgb = display_srgb(linear)
    geometry, info = geometry_passes(position, object_id, expected, expected, evaluated_bvh(scene), tolerances)
    delta = int(np.max(np.abs(png_pixels(native).astype(int) - png_codes(rgb))))
    if (not np.isfinite(rgb).all() or delta > tolerances["display_tolerance_code_values"] or
            not info["surface_pixels"] or info["max_reprojection_px"] > tolerances["render_pixel_tolerance_px"]):
        raise ValueError("Matched source geometry or colour calibration failed")
    np.savez_compressed(out / "source.npz", rgb=rgb, **geometry)
    save_png(out / "source.png", rgb)
    shutil.copyfile(cameras, out / "cameras.json")
    write_json(out / "validation.json", {
        "schema": "el-phase1-matched-source-1", "family": spec["scene_family_id"], "passed": True,
        "parents": cfg["parents"], "input_sha256": cfg["pins"], "seed": reference["seed"],
        "ground_truth_grid_wh": [expected["width"], expected["height"]],
        "camera_max_abs_by_field": residuals, "landmarks": projected, "geometry": info,
        "native_png_max_code_delta": delta, "native_channels": channels, "runtime": render_settings(scene),
        "note": "Native 256-ray reference for R1 source geometry. The actual neural input is the archived 2x2 RGB reduction.",
        "outputs": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("plan", "blend", "settings", "cameras", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    produce(**vars(parser.parse_args()))
