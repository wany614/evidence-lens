"""Read-only verification of a sealed truth payload; optional report goes to a new recorder payload."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import bpy
from mathutils import Vector
import numpy as np
import OpenEXR
from PIL import Image, ImageFilter


VISIBILITY_REASONS = (
    "target_no_surface", "same_first_hit", "nearer_first_hit", "outside_input_image",
    "behind_input_camera", "nonfinite_position", "invalid_target_z", "no_source_ray_hit",
    "first_hit_farther_than_target", "degenerate_input_ray",
)


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_codes(path):
    with Image.open(path) as im:
        require(im.mode == "RGB", f"Expected RGB PNG at {path.name}")
        return np.array(im)


def verify_edges(obj, surface, vis, radius):
    mask = np.zeros_like(surface)
    for values in (obj, surface, vis):
        dy = values[1:] != values[:-1]
        dx = values[:, 1:] != values[:, :-1]
        mask[:-1] |= dy
        mask[1:] |= dy
        mask[:, :-1] |= dx
        mask[:, 1:] |= dx
    return np.array(Image.fromarray(mask.astype("uint8")).filter(ImageFilter.MaxFilter(2*radius+1)), dtype=bool)


def visibility_inventory(data, recorded, name):
    reasons = data["visibility_reason"]
    require(np.all(reasons < len(VISIBILITY_REASONS)), f"{name} unknown visibility reason")
    codes = np.array([0, 1, 2, 3, 3, 0, 0, 0, 0, 0], dtype=np.uint8)
    require(np.array_equal(data["visibility"], codes[reasons]), f"{name} visibility class/reason disagreement")
    background = data["object_id"] == 0
    require(np.array_equal(background, reasons == 0), f"{name} background/mesh ID disagreement")
    require(np.array_equal(data["reference_surface"], ~np.isin(reasons, [0, 5, 6])),
            f"{name} target surface validity disagrees with visibility reasons")
    totals = np.bincount(reasons.ravel(), minlength=len(VISIBILITY_REASONS))
    require(recorded == dict(zip(VISIBILITY_REASONS, totals.tolist())), f"{name} visibility reason totals")
    rgb_valid = np.isfinite(data["rgb"]).all(axis=-1)
    edges = data["edge"]
    unresolved = {}
    for code in range(5, 10):
        mask = reasons == code
        edge_count = int(np.count_nonzero(mask & edges))
        unresolved[str(code)] = {"reason": VISIBILITY_REASONS[code], "count": int(totals[code]),
                                 "edge_count": edge_count, "interior_count": int(totals[code]) - edge_count}
    return {"rgb_valid_count": int(rgb_valid.sum()), "mesh_hit_count": int((~background).sum()),
            "genuine_background_count": int(background.sum()),
            "genuine_background_with_valid_rgb_count": int(np.count_nonzero(background & rgb_valid)),
            "unresolved_mesh_hit_count": int(totals[5:].sum()),
            "unresolved_with_finite_target_geometry_count": int(totals[7:].sum()),
            "unresolved_with_valid_rgb_count": int(np.count_nonzero((reasons >= 5) & rgb_valid)),
            "unresolved_by_reason": unresolved}


def verify_disocclusion(data, settings, recorded, name, certified):
    """Recompute each qualifying same-pair run from sorted disoccluded pixels."""
    obj, blocker, region, edge = (data[k] for k in ("object_id", "first_source_hit_id", "visibility", "edge"))
    ys, xs = np.nonzero((region == 2) & (obj != 0) & (blocker != 0) & (obj != blocker))
    groups = {}
    for y, x in zip(ys.tolist(), xs.tolist()):
        pair = (int(obj[y, x]), int(blocker[y, x]))
        stats = groups.setdefault(pair, [0, 0, 0, -1, -2, 0])
        stats[0] += 1
        stats[1] += int(not edge[y, x])
        run = stats[5]+1 if stats[3] == y and stats[4] == x-1 else 1
        stats[2] = max(stats[2], run)
        stats[3:] = [y, x, run]
    pairs = [{"receiver_id": r, "occluder_id": b, "total_pixels": n, "interior_pixels": m,
              "max_contiguous_horizontal_width_px": width}
             for (r,b), (n,m,width,*_) in sorted(groups.items())
             if width >= settings["band_min_width_px"] and m >= settings["band_min_interior_pixels"]]
    measured = {"pair_count": len(groups), "measurable_pairs": pairs, "meets_width_and_interior": bool(pairs)}
    require(measured == recorded, f"{name} independent geometric disocclusion counts differ")
    if certified:
        require(bool(pairs), f"{name} lacks a measurable geometric disocclusion band")
    return measured


def native_camera(scene, graph, name, view, tolerance):
    obj = scene.objects[name]
    camera, render = obj.data, scene.render
    width = render.resolution_x * render.resolution_percentage // 100
    height = render.resolution_y * render.resolution_percentage // 100
    require([width, height] == [view["width"], view["height"]], f"{name} frozen render resolution")
    require(camera.type == "PERSP" and view["projection"] == "pinhole" and camera.sensor_fit == view["sensor_fit"],
            f"{name} frozen camera projection/sensor fit")
    actual = {"lens_mm": camera.lens, "sensor_width_mm": camera.sensor_width, "sensor_height_mm": camera.sensor_height,
              "lens_shift": [camera.shift_x, camera.shift_y], "clip_m": [camera.clip_start, camera.clip_end],
              "pixel_aspect": [render.pixel_aspect_x, render.pixel_aspect_y]}
    for field, measured in actual.items():
        require(np.allclose(measured, view[field], atol=1e-7, rtol=1e-7), f"{name} frozen camera {field}")
    # Blender's native perspective matrix maps to NDC. Convert its x-right/y-up viewport
    # to top-left integer pixel centres and the OpenCV optical axes.
    projection = np.array(obj.calc_matrix_camera(graph, x=width, y=height,
                          scale_x=render.pixel_aspect_x, scale_y=render.pixel_aspect_y))
    k = np.array([[width/2*projection[0, 0], -width/2*projection[0, 1], width/2*(1-projection[0, 2])-0.5],
                  [-height/2*projection[1, 0], height/2*projection[1, 1], height/2*(1+projection[1, 2])-0.5],
                  [0, 0, 1]])
    k_error = float(np.abs(k-view["K"]).max())
    require(k_error <= tolerance, f"{name} native Blender projection matrix differs by {k_error:.6g} pixels")
    c2w = np.asarray(obj.matrix_world) @ np.diag([1, -1, -1, 1])
    require(np.allclose(c2w, view["c2w"], rtol=0, atol=1e-7), f"{name} frozen camera pose")
    return k, c2w, {**actual, "width": width, "height": height, "resolution_percentage": render.resolution_percentage,
                    "K_integer_from_blender_projection": k.tolist(), "max_intrinsic_error_px": k_error}


def native_ray_witnesses(data, source, scene, graph, validation, seed):
    """Independently call Scene.ray_cast on examples of each stored visibility/edge stratum."""
    rng = np.random.default_rng(seed)
    flat_vis = data["visibility"].ravel()
    flat_edge = data["edge"].ravel()
    flat_surface = data["reference_surface"].ravel()
    flat_pos = data["position"].reshape(-1, 3)
    origin = Vector(np.asarray(source["c2w"])[:3, 3])
    w2c, k = np.asarray(source["w2c"]), np.asarray(source["K"])
    mismatches, resolved_unknowns, counts = [], [], {}
    flat_reasons = data["visibility_reason"].ravel()
    for region in range(4):
        for near_edge in (False, True):
            possible = np.flatnonzero((flat_vis == region) & (flat_edge == near_edge))
            picked = rng.choice(possible, min(32, len(possible)), replace=False)
            counts[f"{region}_{'edge' if near_edge else 'interior'}"] = len(picked)
            for i in picked:
                expected = 0
                point = flat_pos[i].astype(np.float64)
                if flat_surface[i]:
                    cv = w2c[:3, :3] @ point + w2c[:3, 3]
                    if cv[2] <= 0:
                        expected = 3
                    else:
                        q = k @ cv
                        x, y = q[:2]/q[2]
                        if not (-0.5 <= x < source["width"]-0.5 and -0.5 <= y < source["height"]-0.5):
                            expected = 3
                        else:
                            vector = Vector(point) - origin
                            eps = max(validation["ray_epsilon_abs_m"], validation["ray_epsilon_rel"] * vector.length)
                            hit, location, *_ = scene.ray_cast(graph, origin, vector.normalized())
                            if hit:
                                signed = vector.length - (location-origin).length
                                expected = 1 if abs(signed) <= eps else 2 if signed > eps else 0
                if expected != region:
                    if region == 0 and flat_reasons[i] >= 5:
                        resolved_unknowns.append({"pixel_flat_index": int(i), "stored_reason": int(flat_reasons[i]),
                                                  "native_class": expected, "edge": bool(flat_edge[i])})
                    else:
                        mismatches.append([int(i), expected, region])
    return {"strata_sample_counts": counts, "mismatches": mismatches,
            "independently_resolved_unknown_witnesses": resolved_unknowns}


def verify(payload):
    manifest = read_json(payload/"reference.json")
    cameras = read_json(payload/"cameras.json")
    summary = read_json(payload/"validation"/"results.json")
    source_config = read_json(payload/"scene"/"reference_config.json")
    cfg = source_config["reference"]
    vcfg = cfg["validation"]
    pixel_limit = vcfg["render_pixel_tolerance_px"]
    require(np.isfinite(pixel_limit) and pixel_limit >= 0 and
            summary["thresholds"]["render_pixel_tolerance_px"] == pixel_limit,
            "Recorded Position reprojection tolerance disagrees with the frozen config")
    require(manifest["schema"] == "el-reference-1" and cameras["schema"] == "el-camera-1", "Unknown schema")
    require(cameras["unit"] == "m" and cameras["pixel_center"] == "integer", "Unexpected camera convention")
    require(hash_file(payload/"cameras.json") == manifest["camera_sha256"], "Camera hash mismatch")
    require(hash_file(payload/"validation"/"results.json") == manifest["validation_sha256"], "Validation hash mismatch")
    require(hash_file(payload/"input.png") == manifest["input_sha256"] == manifest["views"]["source"]["png_sha256"],
            "Model input bytes disagree with the primary source render")
    family = source_config["scene_family_id"]
    require(re.fullmatch(r"scene_\d{3}_[a-z0-9_]+", family) and
            source_config["scene_id"] == manifest["scene_id"] == family+"/v001" and
            manifest["scene_family_id"] == family, "Frozen scene family identity disagrees")
    blend_name = family + "_v001.blend"
    require(manifest["scene_path"] == f"scene/{blend_name}", "Frozen scene pathname disagrees with family")
    blend = payload/"scene"/blend_name
    require(hash_file(blend) == manifest["blend_sha256"], "Frozen Blender file hash mismatch")
    bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False, use_scripts=False)
    scene = bpy.context.scene
    require(scene.name == blend.stem, "Native scene name disagrees with frozen identity")
    formal = source_config.get("formal")
    if formal:
        require(all(manifest[k] == source_config[k] for k in
                    ("formal", "scene_version", "dataset_id", "dataset_hash", "split", "trial_id", "condition_id")),
                "Formal family and dataset lineage disagree")
    graph = bpy.context.evaluated_depsgraph_get()
    names = ["source", *(f"v{i:02d}" for i in range(1, 7))]
    require(set(names) == set(manifest["views"]) == set(cameras["views"]), "View inventory mismatch")
    checks = {}
    for camera_index, name in enumerate(names):
        v = cameras["views"][name]
        native_k, native_c2w, native_calibration = native_camera(scene, graph, name, v, vcfg["projection_tolerance_px"])
        raw = payload/"reference"/"linear"/f"{name}.exr"
        files = {"reference_sha256": payload/"reference"/f"{name}.npz",
                 "png_sha256": payload/"reference"/f"{name}.png", "linear_exr_sha256": raw}
        for key, file in files.items():
            require(hash_file(file) == manifest["views"][name][key], f"{name} {key} mismatch")
        with np.load(files["reference_sha256"], allow_pickle=False) as handle:
            a = {key: handle[key] for key in handle.files}
        h, w = v["height"], v["width"]
        for key in ["rgb", "position", "z", "object_id", "visibility", "visibility_reason", "first_source_hit_id",
                    "reference_surface", "edge"]:
            require(a[key].shape == ((h, w, 3) if key in ("rgb", "position") else (h, w)), f"{name} {key} shape")
        for key, dtype in {"rgb": np.float32, "position": np.float32, "z": np.float32, "object_id": np.uint16,
                           "visibility": np.uint8, "visibility_reason": np.uint8, "first_source_hit_id": np.uint16,
                           "reference_surface": np.bool_, "edge": np.bool_}.items():
            require(a[key].dtype == dtype, f"{name} {key} dtype")
        rgb = a["rgb"]
        require(np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1, f"{name} RGB domain")
        require(np.array_equal(read_codes(files["png_sha256"]), np.floor(rgb*255+0.5).astype(np.uint8)),
                f"{name} PNG values disagree with independent rounding of FP32")
        surface = a["reference_surface"]
        require(np.array_equal(surface, np.isfinite(a["z"]) & (a["z"]>0) &
                               np.isfinite(a["position"]).all(-1) & (a["object_id"]>0)), f"{name} surface validity")
        require(np.isnan(a["position"][~surface]).all() and np.isnan(a["z"][~surface]).all(), f"{name} invalid depth")
        with OpenEXR.File(str(raw), separate_channels=True) as exr:
            ch = exr.channels()
            pos = np.stack([ch[f"Reference.Position.{i}"].pixels for i in "XYZ"], axis=-1)
            require(np.array_equal(pos[surface], a["position"][surface]), f"{name} native Position disagreement")
            require(np.array_equal(ch["Reference.IndexOB.X"].pixels, a["object_id"]), f"{name} native index disagreement")
            linear = np.stack([ch[f"Reference.Combined.{i}"].pixels for i in "RGB"], axis=-1).astype(np.float64)
            linear = np.maximum(linear, 0)
            display = np.clip(np.where(linear < 0.0031308, 12.92*linear, 1.055*linear**(1/2.4)-0.055), 0, 1)
            require(np.max(np.abs(display-rgb)) < 5e-7, f"{name} native linear-to-display disagreement")
        world = pos[surface].astype(np.float64)
        c2w = np.array(v["c2w"])
        w2c = np.linalg.inv(c2w)
        require(np.allclose(w2c, v["w2c"], atol=1e-7, rtol=0), f"{name} inverse pose mismatch")
        cv = world @ w2c[:3, :3].T + w2c[:3, 3]
        q = cv @ np.array(v["K"]).T
        ys, xs = np.nonzero(surface)
        pixel_centers = np.column_stack((xs, ys))
        require(len(ys) > 0, f"{name} has no valid surface to verify P1 Position alignment")
        err = np.linalg.norm(q[:, :2]/q[:, 2:3] - pixel_centers, axis=-1)
        native_w2c = np.linalg.inv(native_c2w)
        native_cv = world @ native_w2c[:3, :3].T + native_w2c[:3, 3]
        native_q = native_cv @ native_k.T
        native_err = np.linalg.norm(native_q[:, :2]/native_q[:, 2:3] - pixel_centers, axis=-1)
        max_px, max_native_px = float(err.max()), float(native_err.max())
        require(max_px <= pixel_limit and max_native_px <= pixel_limit,
                f"{name} Position reprojection exceeds {pixel_limit} px: supplied K {max_px}, native K {max_native_px}")
        max_z = float(np.abs(cv[:, 2]-a["z"][surface]).max())
        require(max_z < 1e-5, f"{name} recomputed optical depth disagrees")
        counts = np.bincount(a["visibility"].ravel(), minlength=4).tolist()
        require(counts == summary["view_geometry"][name]["visibility_counts"], f"{name} visibility counts")
        inventory = visibility_inventory(a, summary["view_geometry"][name]["visibility_reasons"], name)
        require(np.array_equal(a["edge"], verify_edges(a["object_id"], surface, a["visibility"], vcfg["edge_radius_px"])),
                f"{name} edge mask")
        hits = native_ray_witnesses(a, cameras["views"]["source"], scene, graph, vcfg, 15173+camera_index)
        require(not hits["mismatches"], f"{name} independent Scene.ray_cast disagrees at {hits['mismatches'][:5]}")
        checks[name] = {"surface_count": len(ys), "max_camera_z_error_m": max_z,
                        "max_pixel_reprojection_error_px": max_px, "max_native_pixel_reprojection_error_px": max_native_px,
                        "render_pixel_tolerance_px": pixel_limit, "native_camera": native_calibration,
                        "visibility_inventory": inventory, "native_scene_ray_check": hits}
        if formal:
            checks[name]["geometric_disocclusion"] = verify_disocclusion(
                a, vcfg, summary["view_geometry"][name]["geometric_disocclusion"], name,
                summary["passed"] and name in cfg["camera"]["development"])
        if name in cfg["camera"]["development"]:
            with np.load(payload/"validation"/"second_seed"/f"{name}.npz", allow_pickle=False) as other:
                per_pixel = np.abs(rgb.astype(np.float64)-other["rgb"]).mean(axis=-1)
            noise = summary["noise"][name]
            number = int(np.count_nonzero(per_pixel > vcfg["noise_pixel_mae_threshold"]))
            require(abs(per_pixel.mean()-noise["mae"]) < 1e-8 and number == noise["exceeding_pixel_count"],
                    f"{name} recorded repeat metrics cannot be reproduced")
            checks[name]["repeat_mae"] = float(per_pixel.mean())
            checks[name]["repeat_exceeding_count"] = number
    original = read_codes(payload/"input.png")
    native_png = payload/"validation"/"source_blender.png"
    native_codes = read_codes(native_png)
    require(native_codes.shape == original.shape, "Native source PNG and input dimensions disagree")
    code_delta = np.abs(native_codes.astype(np.int16) - original.astype(np.int16))
    max_code_error = int(code_delta.max())
    require(max_code_error <= vcfg["display_tolerance_code_values"] and
            max_code_error == summary["source_png_vs_blender_max_code_error"],
            f"Native Blender PNG and input PNG differ by {max_code_error} code values")
    source_png_readback = {"native_file_sha256": hash_file(native_png), "input_file_sha256": manifest["input_sha256"],
                           "max_decoded_code_error": max_code_error,
                           "allowed_code_error": vcfg["display_tolerance_code_values"],
                           "differing_pixel_count": int(np.any(code_delta != 0, axis=-1).sum()),
                           "differing_channel_count": int(np.count_nonzero(code_delta)),
                           "pixel_count": original.shape[0] * original.shape[1]}
    box_means = original.reshape(256, 2, 256, 2, 3).mean(axis=(1, 3))
    half = (np.rint(box_means) if formal else np.floor(box_means+0.5)).astype(np.uint8)
    require(np.array_equal(half, read_codes(payload/"validation"/"source_256.png")), "Half-size input mismatches")
    with np.load(payload/"validation"/"decoded_input.npz", allow_pickle=False) as stored:
        require(np.array_equal(original, stored["rgb_u8"]), "Stored input decode mismatches")
    require(manifest["validation_passed"] == summary["passed"] == all(summary["tests"].values()), "Acceptance mismatch")
    return {"schema": "el-reference-readback-1", "verification_version": "1.2" if formal else "1.1",
            "reference_json_sha256": hash_file(payload/"reference.json"),
            **({"scene_family_id": family, "dataset_hash": manifest["dataset_hash"], "formal": formal} if formal else {}),
            "integrity_checked": True, "scientific_acceptance": summary["passed"],
            "source_png_readback": source_png_readback,
            "repeat_measurement": {"primary_seed": cfg["seed"], "repeat_seed": cfg["repeat_seed"],
                                   "denoiser": scene.cycles.denoiser, "scope": "stochastic variation with fixed denoiser",
                                   "systematic_denoiser_bias": "unmeasured"},
            "failed_scientific_checks": summary["failed_checks"], "views": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="Fresh recorder output directory for the readback report")
    args = parser.parse_args()
    report = verify(args.payload.resolve(strict=True))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        dest = args.out/"reference_readback.json"
        with dest.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(report, allow_nan=False), flush=True)
    raise SystemExit(0 if report["scientific_acceptance"] else 1)
