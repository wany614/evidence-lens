"""Read-only verification of a sealed truth payload; optional report goes to a new recorder payload."""
import argparse
import hashlib
import json
from pathlib import Path

import bpy
from mathutils import Vector
import numpy as np
import OpenEXR
from PIL import Image, ImageFilter


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


def native_ray_witnesses(data, source, scene, graph, validation, seed):
    """Independently call Scene.ray_cast on examples of each stored visibility/edge stratum."""
    rng = np.random.default_rng(seed)
    flat_vis = data["visibility"].ravel()
    flat_edge = data["edge"].ravel()
    flat_surface = data["reference_surface"].ravel()
    flat_pos = data["position"].reshape(-1, 3)
    origin = Vector(np.asarray(source["c2w"])[:3, 3])
    w2c, k = np.asarray(source["w2c"]), np.asarray(source["K"])
    mismatches, counts = [], {}
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
                    mismatches.append([int(i), expected, region])
    return {"strata_sample_counts": counts, "mismatches": mismatches}


def verify(payload):
    manifest = read_json(payload/"reference.json")
    cameras = read_json(payload/"cameras.json")
    summary = read_json(payload/"validation"/"results.json")
    cfg = read_json(payload/"scene"/"reference_config.json")["reference"]
    vcfg = cfg["validation"]
    require(manifest["schema"] == "el-reference-1" and cameras["schema"] == "el-camera-1", "Unknown schema")
    require(cameras["unit"] == "m" and cameras["pixel_center"] == "integer", "Unexpected camera convention")
    require(hash_file(payload/"cameras.json") == manifest["camera_sha256"], "Camera hash mismatch")
    require(hash_file(payload/"validation"/"results.json") == manifest["validation_sha256"], "Validation hash mismatch")
    require(hash_file(payload/"input.png") == manifest["input_sha256"] == manifest["views"]["source"]["png_sha256"],
            "Model input bytes disagree with the primary source render")
    blend = payload/"scene"/"scene_001_tabletop_v001.blend"
    require(hash_file(blend) == manifest["blend_sha256"], "Frozen Blender file hash mismatch")
    bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False, use_scripts=False)
    scene = bpy.context.scene
    graph = bpy.context.evaluated_depsgraph_get()
    names = ["source", *(f"v{i:02d}" for i in range(1, 7))]
    require(set(names) == set(manifest["views"]) == set(cameras["views"]), "View inventory mismatch")
    checks = {}
    for camera_index, name in enumerate(names):
        v = cameras["views"][name]
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
        world = a["position"][surface].astype(np.float64)
        c2w = np.array(v["c2w"])
        native_c2w = np.asarray(bpy.data.objects[name].matrix_world) @ np.diag([1, -1, -1, 1])
        require(np.allclose(c2w, native_c2w, rtol=0, atol=1e-7), f"{name} frozen calibration mismatch")
        w2c = np.linalg.inv(c2w)
        require(np.allclose(w2c, v["w2c"], atol=1e-7, rtol=0), f"{name} inverse pose mismatch")
        cv = world @ w2c[:3, :3].T + w2c[:3, 3]
        q = cv @ np.array(v["K"]).T
        ys, xs = np.nonzero(surface)
        err = np.linalg.norm(q[:, :2]/q[:, 2:3] - np.column_stack((xs, ys)), axis=-1)
        max_z = float(np.abs(cv[:, 2]-a["z"][surface]).max())
        require(max_z < 1e-5, f"{name} recomputed optical depth disagrees")
        counts = np.bincount(a["visibility"].ravel(), minlength=4).tolist()
        require(counts == summary["view_geometry"][name]["visibility_counts"], f"{name} visibility counts")
        require(np.array_equal(a["edge"], verify_edges(a["object_id"], surface, a["visibility"], vcfg["edge_radius_px"])),
                f"{name} edge mask")
        hits = native_ray_witnesses(a, cameras["views"]["source"], scene, graph, vcfg, 15173+camera_index)
        require(not hits["mismatches"], f"{name} independent Scene.ray_cast disagrees at {hits['mismatches'][:5]}")
        checks[name] = {"surface_count": len(ys), "max_camera_z_error_m": max_z,
                        "max_pixel_reprojection_error_px": float(err.max()), "native_scene_ray_check": hits}
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
    half = np.floor(original.reshape(256, 2, 256, 2, 3).mean(axis=(1, 3))+0.5).astype(np.uint8)
    require(np.array_equal(half, read_codes(payload/"validation"/"source_256.png")), "Half-size input mismatches")
    with np.load(payload/"validation"/"decoded_input.npz", allow_pickle=False) as stored:
        require(np.array_equal(original, stored["rgb_u8"]), "Stored input decode mismatches")
    require(manifest["validation_passed"] == summary["passed"] == all(summary["tests"].values()), "Acceptance mismatch")
    return {"schema": "el-reference-readback-1", "reference_json_sha256": hash_file(payload/"reference.json"),
            "integrity_checked": True, "scientific_acceptance": summary["passed"],
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
