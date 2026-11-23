"""Generate a fixed Blender truth package. Execute inside the recorder and reference environment."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
import time

import bpy
import numpy as np
import OpenEXR
from PIL import __version__ as pillow_version

# Also runnable when this file is archived and used with Blender's --python flag.
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from passes import REASONS, display_srgb, geometry_passes, png_codes, png_pixels, read_passes, save_png, sha256, write_json
from scene import build_world, camera_bundle, camera_record, evaluated_bvh, ocio_config_path, render_settings, validate_camera
from validation import calibration_fixture, render, seed_comparison, source_resize


class Events:
    def __init__(self, path):
        self.path = path

    def emit(self, stage, status, **details):
        row = {"utc": datetime.now(timezone.utc).isoformat(), "stage": stage, "status": status, **details}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    @contextmanager
    def record(self, stage, **details):
        start = time.perf_counter()
        self.emit(stage, "started", **details)
        try:
            yield
        except Exception as exc:
            self.emit(stage, "failed", seconds=time.perf_counter()-start, error=str(exc), **details)
            raise
        self.emit(stage, "completed", seconds=time.perf_counter()-start, **details)


def validate_config(config):
    cfg = config["reference"]
    if cfg["schema"] != "el-reference-config-1" or tuple(cfg["blender_version"]) != bpy.app.version:
        raise ValueError(f"Unsupported config schema or Blender version; actual {bpy.app.version}")
    if cfg["width"] != 512 or cfg["height"] != 512 or cfg["preview_width"] * 2 != cfg["width"]:
        raise ValueError("This frozen version uses 512x512 and 256x256 half-resolution inputs")
    if set(cfg["camera"]["positions"]) != {"source", *(f"v{i:02d}" for i in range(1, 7))}:
        raise ValueError("Source and six prescribed camera IDs are required")
    if cfg["seed"] == cfg["repeat_seed"] or set(cfg["camera"]["development"]) != {"v01", "v02"}:
        raise ValueError("Different seeds and two development cameras are required")
    ids = [obj["id"] for obj in cfg["objects"]]
    if len(set(ids)) != len(ids) or any(i <= 0 or i > 65535 for i in ids):
        raise ValueError("Scene objects require distinct positive uint16 IDs")
    if cfg["color"] != {"display": "sRGB", "view": "Standard", "look": "None", "exposure": 0, "gamma": 1}:
        raise ValueError("This release implements Standard with sRGB display, no Look, exposure 0 and gamma 1")
    if "formal" in config:
        family = config["scene_family_id"]
        if (not re.fullmatch(r"scene_\d{3}_[a-z0-9_]+", family) or config["scene_id"] != family+"/v001" or
                cfg.get("scene_name") != family+"_v001" or cfg["validation"].get("occlusion_profile") != "opaque_mesh_pairs"):
            raise ValueError("Formal family identity and frozen scene settings disagree")
    return cfg


def produce(config_path, out):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = validate_config(config)
    tol = cfg["validation"]
    out.mkdir(parents=True, exist_ok=True)
    # Refuse to append to an existing experiment's outputs. Recorder allocates a fresh directory.
    if any(out.iterdir()):
        raise ValueError(f"Reference payload must be empty: {out}")
    reference, validation, frozen = out/"reference", out/"validation", out/"scene"
    native, repeats = reference/"linear", validation/"second_seed"
    for folder in [native, repeats, frozen]:
        folder.mkdir(parents=True)
    logs = Events(out/"reference_events.jsonl")
    for name in ["produce.py", "passes.py", "scene.py", "validation.py"]:
        shutil.copyfile(CODE/name, frozen/name)
    fields = ["scene_id", "scene_family_id", "reference"]
    if "formal" in config:
        fields += ["formal", "scene_version", "dataset_id", "dataset_hash", "split", "trial_id", "condition_id"]
    write_json(frozen/"reference_config.json", {k: config[k] for k in fields})
    with logs.record("render_calibration_fixture"):
        fixture = calibration_fixture(cfg, validation)
        write_json(validation/"calibration.json", fixture)
    with logs.record("build_freeze_world"):
        scene, cameras = build_world(cfg)
        actual_settings = render_settings(scene)
        ocio = ocio_config_path()
        actual_settings["ocio_config_sha256"] = sha256(ocio)
        actual_settings["ocio_config_filename"] = ocio.name
        actual_settings["numpy_version"] = np.__version__
        actual_settings["openexr_python_version"] = OpenEXR.__version__
        actual_settings["pillow_version"] = pillow_version
        write_json(frozen/"runtime.json", actual_settings)
        views = {v: camera_record(scene, camera, "input" if v == "source" else "research_target" if "formal" in config
                                  else "development" if v in cfg["camera"]["development"] else "nearby_held_out")
                 for v, camera in cameras.items()}
        write_json(out/"cameras.json", camera_bundle(views))
        if "formal" in config and sha256(out/"cameras.json") != config["camera_set_hash"]:
            raise ValueError("Rendered camera definitions differ from the pre-acquisition camera package")
        points = [spec["center"] for spec in cfg["objects"]]
        projections = {v: validate_camera(scene, camera, views[v], points) for v, camera in cameras.items()}
        for name, v in views.items():
            projections[name]["translation_from_source_m"] = float(np.linalg.norm(
                np.asarray(v["c2w"])[:3, 3] - np.asarray(views["source"]["c2w"])[:3, 3]))
        bvh = evaluated_bvh(scene)
        blend_path = frozen/(scene.name+".blend")
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), compress=True, copy=True, relative_remap=False)
        frozen_hash = sha256(blend_path)
        # Rendering changes the active camera, seed and output path only in this in-memory scene copy.
        work = scene.copy()
        work.name = "Reference render working copy"
    summary = {"schema": "el-reference-validation-1", "fixture": fixture["tests"], "cameras": projections,
               "thresholds": tol, "view_geometry": {}, "noise": {}, "tests": {}}
    primary = {}
    versions = {"schema": "el-reference-1", "scene_id": config["scene_id"], "scene_family_id": config["scene_family_id"],
                "camera_sha256": sha256(out/"cameras.json"), "config_sha256": sha256(config_path),
                "blend_sha256": frozen_hash, "scene_path": "scene/" + blend_path.name,
                "rgb_encoding": "display_srgb_0_1", "depth_semantics": "positive_optical_axis_camera_z_m",
                "visibility_codes": {"0": "background_or_invalid", "1": "observed", "2": "disoccluded", "3": "outside_input_fov"},
                "visibility_reasons": REASONS, "pixel_order": "top_left row, column", "views": {},
                "array_schema": {
                    "rgb": "H,W,3 float32 display_sRGB RGB [0,1]",
                    "position": "H,W,3 float32 Blender world XYZ metres; NaN if no valid surface",
                    "z": "H,W float32 OpenCV camera axial depth metres; NaN if no valid surface",
                    "object_id": "H,W uint16 index assigned to the mesh; 0 for no mesh",
                    "reference_surface": "H,W bool mesh hit with finite position and positive target z",
                    "visibility": "H,W uint8 input-view first-hit visibility (visibility_codes)",
                    "visibility_reason": "H,W uint8 reason (visibility_reasons)",
                    "first_source_hit_id": "H,W uint16 input BVH first hit; 0 when unavailable",
                    "edge": "H,W bool four-neighbour ID, surface or visibility boundary, square dilation"
                }, "ray_algorithm": "Blender BVHTree.FromPolygons from evaluated opaque world-space loop triangles",
                "ray_triangle_count": int(len(bvh[1])), "primary_seed": cfg["seed"], "repeat_seed": cfg["repeat_seed"]}
    if "formal" in config:
        versions.update({k: config[k] for k in ("formal", "scene_version", "dataset_id", "dataset_hash", "split", "trial_id", "condition_id")})
    geometry_key = "geometric_disocclusion" if "formal" in config else "known_occlusion_pair"
    for name in ["source", *(f"v{i:02d}" for i in range(1, 7))]:
        work.camera = cameras[name]
        work.cycles.seed = cfg["seed"]
        exr = native/f"{name}.exr"
        png = reference/f"{name}.png"
        source_native = validation/"source_blender.png" if name == "source" else None
        with logs.record("render_primary", view=name, seed=cfg["seed"]):
            render(work, exr, source_native)
        with logs.record("export_geometry_and_rgb", view=name):
            linear, pos, obj, channels = read_passes(exr)
            rgb = display_srgb(linear)
            if not np.isfinite(rgb).all() or not np.isfinite(linear).all():
                raise ValueError(f"Invalid combined RGB for {name}")
            geometry, info = geometry_passes(pos, obj, views[name], views["source"], bvh, tol)
            if geometry_key == "known_occlusion_pair":
                info[geometry_key]["meets_width_and_interior"] = (
                    info[geometry_key]["max_contiguous_horizontal_width_px"] >= tol["band_min_width_px"]
                    and info[geometry_key]["interior_pixels"] >= tol["band_min_interior_pixels"])
            result = reference/f"{name}.npz"
            np.savez_compressed(result, rgb=rgb, **geometry)
            save_png(png, rgb)
            versions["views"][name] = {"reference_sha256": sha256(result), "png_sha256": sha256(png),
                                       "linear_exr_sha256": sha256(exr), "width": views[name]["width"],
                                       "height": views[name]["height"], "native_channels": channels}
            summary["view_geometry"][name] = info
            if name == "source":
                input_path = out/"input.png"
                shutil.copyfile(png, input_path)
                versions["input_sha256"] = sha256(input_path)
                summary["source_png_vs_blender_max_code_error"] = int(np.max(np.abs(
                    png_pixels(source_native).astype(int) - png_pixels(input_path))))
                summary["source_png_vs_fp32_max_abs_error"] = float(np.max(np.abs(
                    png_pixels(input_path).astype(np.float64)/255 - rgb)))
                summary["resize"] = source_resize(views["source"], input_path, validation, points, cfg.get("resize_rounding") == "nearest_even")
            if name in cfg["camera"]["development"]:
                primary[name] = rgb
    for name in cfg["camera"]["development"]:
        with logs.record("render_repeat_and_measure_noise", view=name, seed=cfg["repeat_seed"]):
            work.camera = cameras[name]
            work.cycles.seed = cfg["repeat_seed"]
            exr = repeats/f"{name}.exr"
            render(work, exr)
            linear, _, _, _ = read_passes(exr)
            repeated = display_srgb(linear)
            np.savez_compressed(repeats/f"{name}.npz", rgb=repeated)
            save_png(repeats/f"{name}.png", repeated)
            summary["noise"][name] = seed_comparison(primary[name], repeated, tol)
    geo = summary["view_geometry"]
    source_geo = geo["source"]
    summary["tests"] = {
        **fixture["tests"],
        "frozen_blend_unmodified": sha256(blend_path) == frozen_hash,
        "finite_point_camera_calibration": all(p["max_error_px"] <= tol["projection_tolerance_px"]
            and p["right_moves_right"] and p["up_moves_up"] and p["forward_increases_z"]
            and abs(p["rotation_determinant"] - 1) <= 1e-5 for p in projections.values()),
        "actual_camera_translation": all(projections[v]["translation_from_source_m"] > 0 for v in views if v != "source"),
        "rendered_position_alignment": all(g["max_reprojection_px"] is not None and
            g["max_reprojection_px"] <= tol["render_pixel_tolerance_px"] for g in geo.values()),
        "source_geometry_observed": source_geo["surface_pixels"] > 0 and
            source_geo["visibility_counts"][1]/max(1, source_geo["surface_pixels"]) >= tol["source_min_observed_fraction"],
        "measurable_development_occlusion": all(geo[v][geometry_key]["meets_width_and_interior"]
            for v in cfg["camera"]["development"]),
        "native_source_display": summary["source_png_vs_blender_max_code_error"] <= tol["display_tolerance_code_values"],
        "input_bytes_preserved": versions["input_sha256"] == versions["views"]["source"]["png_sha256"],
        "numerical_rgb_to_png": summary["source_png_vs_fp32_max_abs_error"] <= 0.5/255 + 1e-7,
        "resize_calibration": summary["resize"]["max_projection_residual_px"] <= tol["projection_tolerance_px"] and
            summary["resize"]["stored_half_matches_resampler"],
        "two_seed_development_noise": all(n["passed"] for n in summary["noise"].values())}
    failed = [name for name, passed in summary["tests"].items() if not passed]
    summary["passed"] = versions["validation_passed"] = not failed
    summary["failed_checks"] = failed
    write_json(validation/"results.json", summary)
    versions["validation_sha256"] = sha256(validation/"results.json")
    write_json(out/"reference.json", versions)
    logs.emit("reference_acceptance", "failed" if failed else "completed", failed_checks=failed)
    if failed:
        raise RuntimeError("Reference acceptance failed: " + ", ".join(failed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None)
    produce(args.config.resolve(strict=True), args.out.resolve())
