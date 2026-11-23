"""Read-only Phase 1 audit of signed study inputs, PLY bytes, prior probes and native Blender camera geometry."""

import argparse
import json
from pathlib import Path

import numpy as np
from inspect_sharp_ply import read_ply
from phase1_geometry_oracle import NAMES, sha
from PIL import Image

from evidence_lens.formal_inputs import PinnedRun


def load(path):
    return json.loads(path.read_bytes())


def legacy(data):
    settings = data["settings"]
    ztol = settings["depth_probe_abs_tolerance_m"]
    errors = []
    for s, xy, alpha in zip(
        data["samples"], ([16, 16], [28, 10]), (0.4, 0.75), strict=True
    ):
        errors.append(
            {
                "xy": s["xy"],
                "expected_xy": xy,
                "expected_Z_m": 2,
                "ED_m": s["ED"],
                "ED_error_m": s["ED"] - 2,
                "D_minus_alphaZ_m": s["D"] - s["alpha"] * 2,
                "alpha_error": s["alpha"] - alpha,
                "linear_red_error": s["red"] - 0.25 * s["alpha"],
            }
        )
    assert all(
        v["xy"] == list(v["expected_xy"])
        and abs(v["ED_error_m"]) <= ztol
        and abs(v["D_minus_alphaZ_m"]) <= ztol
        and abs(v["alpha_error"]) <= settings["alpha_probe_abs_tolerance"]
        and abs(v["linear_red_error"]) <= settings["color_probe_abs_tolerance"]
        for v in errors
    )
    assert data["empty_pixel_alpha"] == 0
    assert (
        abs(data["translated_probe_alpha"] - 0.4)
        <= settings["alpha_probe_abs_tolerance"]
    )
    assert (
        data["translation_passed"]
        and data["off_axis_plane_passed"]
        and data["alpha_normalization_passed"]
        and data["linear_blend_passed"]
    )
    return {
        "samples": errors,
        "translated_alpha": data["translated_probe_alpha"],
        "empty_alpha": data["empty_pixel_alpha"],
        "historical_translated_Z": "Only the combined pass boolean and moved alpha were serialized; individual moved ED is unavailable.",
    }


def native(blend, protocol):
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    bpy.ops.wm.open_mainfile(filepath=str(blend.resolve()))
    scene = bpy.context.scene
    source = scene.objects["source"]
    # Resolve original metric points with Blender's native OpenGL camera axes, independent of the adapter's CV transform.
    points = [
        source.matrix_world @ Vector((p[0], -p[1], -p[2])) for p in protocol["means_m"]
    ]
    geom = {}
    for fixture in protocol["scenes"][:2]:
        for name, c in fixture["cameras"].items():
            expected = fixture["expected"][name]
            camera = scene.objects[name]
            measurements = []
            for world, expect in zip(points, expected, strict=True):
                projected = world_to_camera_view(scene, camera, world)
                uv = np.array(
                    [
                        projected.x * c["width"] - 0.5,
                        (1 - projected.y) * c["height"] - 0.5,
                    ]
                )
                measurements.append(
                    {
                        "xy_native": uv.tolist(),
                        "expected": expect["uv"],
                        "xy_max_error_px": float(np.max(np.abs(uv - expect["uv"]))),
                        "Z_native_m": float(projected.z),
                        "expected_Z_m": expect["z_m"],
                        "Z_error_m": float(projected.z - expect["z_m"]),
                    }
                )
            geom[fixture["name"] + "/" + name] = measurements
    assert (
        max(v["xy_max_error_px"] for vals in geom.values() for v in vals)
        < protocol["tolerance"]["centroid_px"]
    )
    assert (
        max(abs(v["Z_error_m"]) for vals in geom.values() for v in vals)
        < protocol["tolerance"]["metric_m"]
    )
    bg = next(n for n in scene.world.node_tree.nodes if n.type == "BACKGROUND")
    return {
        "blender_version": bpy.app.version_string,
        "units": scene.unit_settings.system,
        "metres_per_unit": float(scene.unit_settings.scale_length),
        "world_strength": float(bg.inputs["Strength"].default_value),
        "world_linear_rgba": list(bg.inputs["Color"].default_value),
        "film_transparent": scene.render.film_transparent,
        "display": scene.display_settings.display_device,
        "view": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "exposure": scene.view_settings.exposure,
        "gamma": scene.view_settings.gamma,
        "geometry": geom,
        "no_render_performed": True,
    }


def audit(store, plan, protocol, cold=None, frozen_index=None):
    all_cases = load(plan)
    fixture = load(protocol)
    result = {
        "schema": "el-phase1-archived-camera-audit-1",
        "plan_sha256": sha(plan),
        "protocol_sha256": sha(protocol),
        "cases": [],
    }
    originals = {}
    cache = {}
    consumed = {}

    class AuditedRun(PinnedRun):
        def open(self, rel):
            verified = super().open(rel)
            consumed.setdefault(self.pin["run_id"], {})[rel] = self.files[rel]["sha256"]
            return verified

    def recorded(pin):
        source = (
            store if (store / "runs" / pin["run_id"]).is_dir() or cold is None else cold
        )
        rid = pin["run_id"]
        if rid not in cache:
            if frozen_index is not None:
                assert (
                    sha(frozen_index / "manifests" / (rid + ".json"))
                    == pin["manifest_sha256"]
                )
            cache[rid] = AuditedRun(source, pin)
        assert (
            cache[rid].pin == pin and cache[rid].run["execution_status"] == "succeeded"
        )
        return cache[rid]

    def bound(child, owner, paths):
        rows = load(child.open("inputs.json"))["inputs"]
        hashes = {
            item["sha256"]
            for row in rows
            if row["parent_run_id"] == owner.run["run_id"]
            for item in row["files"]
        }
        assert {owner.files[p]["sha256"] for p in paths} <= hashes

    for case in all_cases["cases"]:
        f, c, trial = case["family"], case["condition"], case["trial"]
        mark = (
            frozen_index / "selections" / (trial + ".json")
            if frozen_index is not None
            else store / "trials" / trial / "trial_selection.json"
        )
        assert sha(mark) == case["selection_sha256"]
        selected = case["selected"]
        owner = selected["preparation"] if c == "R1" else selected["reference"]
        gen_pin = selected.get("producer", selected["generation"])
        refs = recorded(selected["reference"])
        sr = recorded(owner)
        gen = recorded(gen_pin)
        original_gen = recorded(selected["generation"])
        rend = recorded(selected["render"])
        checked = recorded(selected["readback"])
        bound(original_gen, sr, ("payload/input.png", "payload/cameras.json"))
        bound(rend, sr, ("payload/input.png", "payload/cameras.json"))
        bound(rend, gen, ("payload/scene.ply", "payload/generation.json"))
        snapshots = {}
        for label, run, name in (
            ("learned_generator", original_gen, "generator_adapter.py"),
            ("renderer", rend, "renderer_adapter.py"),
        ):
            code = run.open("source/src/evidence_lens/" + name)
            digest = sha(code, True)
            assert digest == fixture["normalized_source_sha256"]["evidence_lens"][name]
            snapshots[label] = digest
        camfile = sr.open("payload/cameras.json")
        cam = load(camfile)
        assert (cam["unit"], cam["pixel_center"], cam["axis_convention"]) == (
            "m",
            "integer",
            "OpenCV x right, y down, z forward",
        )
        base = load(refs.open("payload/cameras.json"))
        for n in NAMES:
            assert cam["views"][n]["c2w"] == base["views"][n]["c2w"]
            if n != "source" or c != "R1":
                assert cam["views"][n] == base["views"][n]
        v = cam["views"]["source"]
        w, h = v["width"], v["height"]
        image = sr.open("payload/input.png")
        with Image.open(image) as im:
            assert im.size == (w, h) and im.format == "PNG" and im.mode == "RGB"
        if c == "R1":
            assert [w, h] == [256, 256] and np.allclose(
                np.array([[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1]])
                @ base["views"]["source"]["K"],
                v["K"],
                atol=1e-12,
                rtol=0,
            )
        gen_path = gen.open("payload/generation.json")
        g = load(gen_path)
        inference = load(original_gen.open("payload/generation.json"))
        if c == "Q1":
            assert (
                g["schema"] == "el-gaussian-derivation-1"
                and g["forward_calls_in_this_stage"] == 0
            )
            assert g["parent_generation_run_id"] == selected["generation"]["run_id"]
            assert (
                g["parent_ply_sha256"]
                == original_gen.files["payload/scene.ply"]["sha256"]
            )
            bound(gen, original_gen, ("payload/scene.ply", "payload/generation.json"))
            snapshots["colour_roundtrip"] = sha(
                gen.open("source/scripts/stress_condition.py"), True
            )
        real_ply = gen.open("payload/scene.ply")
        pred = load(rend.open("payload/prediction.json"))
        probe = load(rend.open("payload/depth-probe.json"))
        assert g["input_sha256"] == pred["input_sha256"] == sha(image) and g[
            "camera_sha256"
        ] == pred["camera_sha256"] == sha(camfile)
        assert g["producer_run_id"] == pred["generation_run_id"] == gen_pin["run_id"]
        assert inference["source_size_wh"] == [w, h] and inference[
            "network_input_wh"
        ] == [1536, 1536]
        assert (
            inference["generator_commit"] == fixture["sharp_commit"]
            and inference["forward_seconds"] > 0
        )
        assert inference["native_frame"] == "source CV camera, metres"
        assert inference["runtime"]["gsplat"] == pred["runtime"]["gsplat"] == "1.5.3"
        assert abs(inference["focal_length_px"] - v["K"][0][0]) < 1e-10 and g[
            "ply_sha256"
        ] == pred["ply_sha256"] == sha(real_ply)
        assert pred["depth_validation_receipt_sha256"] == sha(
            rend.open("payload/depth-probe.json")
        )
        assert (
            pred["rgb_encoding"] == "display_srgb_0_1"
            and pred["depth_semantics"] == "expected_camera_z_m"
        )
        vertex, extra = read_ply(real_ply)
        ply_k = extra["intrinsic"].reshape(3, 3)
        k = np.asarray(v["K"])
        ke = k.copy()
        ke[:2, 2] += 0.5
        assert np.allclose(ply_k, ke, atol=2e-4, rtol=0) and list(
            extra["image_size"]
        ) == [w, h]
        assert np.array_equal(extra["extrinsic"].reshape(4, 4), np.eye(4))
        assert extra["color_space"].tolist() == [0]
        assert len(vertex) == g["gaussian_count"]
        shapes = {}
        for n in NAMES:
            path = rend.open("payload/" + n + ".npz")
            vv = cam["views"][n]
            assert sha(path) == pred["views"][n]["sha256"]
            with np.load(path, allow_pickle=False) as z:
                alpha = z["alpha"]
                depth = z["z"]
                rgb = z["rgb"]
                miss = alpha == 0
                assert alpha.shape == depth.shape == (
                    vv["height"],
                    vv["width"],
                ) and rgb.shape == (
                    *depth.shape,
                    3,
                )
                assert np.isnan(depth[miss]).all()
                png = rend.open("payload/" + n + ".png")
                with Image.open(png) as preview:
                    assert preview.mode == "RGB" and preview.size == (
                        vv["width"],
                        vv["height"],
                    )
                    encoded = np.rint(255 * np.nan_to_num(rgb)).astype(np.uint8)
                    assert np.array_equal(np.asarray(preview), encoded)
                shapes[n] = {
                    "width": vv["width"],
                    "height": vv["height"],
                    "alpha_zero": int(miss.sum()),
                    "missing_z": int(np.count_nonzero(~np.isfinite(depth))),
                    "preview_uint8_exact": True,
                }
        ref = load(refs.open("payload/reference.json"))
        rb = load(checked.open("payload/reference_readback.json"))
        if f not in originals:
            blend_path = next(
                k
                for k in refs.files
                if k.startswith("payload/") and k.endswith(".blend")
            )
            blend = refs.open(blend_path)
            assert sha(blend) == ref["blend_sha256"]
            orig = native(blend, fixture)
            assert orig["units"] == "METRIC" and orig["metres_per_unit"] == 1
            originals[f] = {
                "reference": selected["reference"],
                "readback": selected["readback"],
                "blend_sha256": sha(blend),
                "reference_rgb": ref["rgb_encoding"],
                "reference_depth": ref["depth_semantics"],
                "native": orig,
                "archived_max_Position_reprojection_px": max(
                    x["max_native_pixel_reprojection_error_px"]
                    for x in rb["views"].values()
                ),
                "archived_max_Position_Z_error_m": max(
                    x["max_camera_z_error_m"] for x in rb["views"].values()
                ),
            }
        result["cases"].append(
            {
                "family": f,
                "trial": trial,
                "condition": c,
                "producer": gen_pin,
                "renderer": selected["render"],
                "selection_sha256": sha(mark),
                "generation_receipt_sha256": sha(gen_path),
                "input_sha256": sha(image),
                "camera_sha256": sha(camfile),
                "ply_sha256": sha(real_ply),
                "width_height": [w, h],
                "K_integer": v["K"],
                "f_px_recorded": inference["focal_length_px"],
                "f_over_width_derived": inference["focal_length_px"] / w,
                "network_size_declared": inference["network_input_wh"],
                "forward_seconds_recorded": inference["forward_seconds"],
                "native_frame_recorded": inference["native_frame"],
                "original_learned_run": selected["generation"],
                "producer_type": g["schema"],
                "zero_forward_Q1": g.get("forward_calls_in_this_stage")
                if c == "Q1"
                else None,
                "executed_source_normalized_sha256": snapshots,
                "checkpoint_sha256": inference["checkpoint_sha256"],
                "archived_runtime_gaps": [
                    "No serialized internal tensor, measured NDC coordinates, or observed disparity_factor in historical learned runs."
                ],
                "ply_header_K": ply_k.tolist(),
                "ply_header_colour_space": "sRGB/SH0 (code 0)",
                "ply_vertex_count": len(vertex),
                "actual_prediction_grids": shapes,
                "prior_calibration": legacy(probe),
                "background": pred["render_settings"]["background"],
            }
        )
    result["native_references"] = originals
    result["verified_archive_members"] = consumed
    result["passed"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "plan", "protocol", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cold-store", type=Path)
    args = parser.parse_args()
    data = audit(args.store, args.plan, args.protocol, args.cold_store)
    with args.out.open("x") as fp:
        json.dump(data, fp, indent=2, allow_nan=False)
        fp.write("\n")
