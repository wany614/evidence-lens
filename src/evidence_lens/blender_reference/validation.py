"""Small rendered and analytic witnesses for the reference producer."""
import bpy
import numpy as np
from PIL import Image

from scene import camera, camera_record, configure, evaluated_bvh, material, project
from passes import (display_srgb, half_camera, png_codes, png_pixels, read_passes,
                    reduce_half, sha256, visibility)


def render(scene, exr_path, native_png=None):
    settings = scene.render.image_settings
    scene.render.filepath = str(exr_path)
    if bpy.ops.render.render(write_still=True, scene=scene.name) != {"FINISHED"}:
        raise RuntimeError(f"Blender did not finish {exr_path.name}")
    if native_png is None:
        return
    settings.file_format = "PNG"
    settings.color_depth = "8"
    settings.color_mode = "RGB"
    try:
        bpy.data.images["Render Result"].save_render(str(native_png), scene=scene)
    finally:
        settings.file_format = "OPEN_EXR_MULTILAYER"
        settings.color_depth = "32"
        settings.color_mode = "RGBA"
        settings.exr_codec = "ZIP"


def calibration_fixture(cfg, root):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure(scene, cfg, 96, 96)
    scene.cycles.samples = 4
    scene.cycles.use_denoising = False
    scene.camera = camera(scene, "Calibration camera", cfg["camera"], (0, 0, 0))
    colors = [[0.18, 0.50, 0.0031308], [0.70, 0.04, 0.24]]
    # Tabulated independently from IEC 61966-2-1 for the calibration swatches.
    expected_colors = [[0.46135613, 0.73535698, 0.04044994], [0.85430583, 0.22091636, 0.52711279]]
    for i, (loc, size) in enumerate([((0, 0, -2), (4, 4)), ((0.17, 0.03, -1), (0.18, 0.19))]):
        bpy.ops.mesh.primitive_plane_add(size=1, location=loc)
        obj = bpy.context.object
        obj.name, obj.scale, obj.pass_index = f"Swatch {i+1}", (*size, 1), i+1
        obj.data.materials.append(material(obj.name, colors[i]))
    bpy.context.view_layer.update()
    view = camera_record(scene, scene.camera, "calibration")
    exr, native = root/"calibration.exr", root/"calibration_blender.png"
    render(scene, exr, native)
    linear, pos, ids, names = read_passes(exr)
    rgb = display_srgb(linear)
    native_codes = png_pixels(native)
    uv, z = project(pos, view)
    whole_image_error = int(np.abs(png_codes(rgb).astype(int) - native_codes).max())
    measurements = []
    for point, obj_id in [([0, 0, -2], 1), ([0.64, 0.55, -2], 1),
                          ([-0.58, -0.49, -2], 1), ([0.17, 0.03, -1], 2)]:
        q, expected_z = project([point], view)
        x, y = np.floor(q[0] + 0.5).astype(int)
        color = expected_colors[obj_id-1]
        measurements.append({"known_point_m": point, "pixel_xy": [int(x), int(y)],
                             "object_expected": obj_id, "object_read": int(ids[y, x]),
                             "expected_z_m": float(expected_z[0]), "measured_z_m": float(z[y, x]),
                             "ray_distance_m": float(np.linalg.norm(pos[y, x])),
                             "expected_display_rgb": color, "measured_display_rgb": rgb[y, x].tolist(),
                             "native_blender_codes": native_codes[y, x].tolist(),
                             "reprojection_error_px": float(np.linalg.norm(uv[y, x] - [x, y])),
                             "color_abs_max": float(np.max(np.abs(rgb[y, x] - color)))})
    # These inspect first hits against actual fixture triangles. Target validity is supplied explicitly.
    witnesses = np.array([[[0, 0, -2], [0.34, 0.06, -2], [1.5, 0, -2], [0, 0, 2],
                           [0, 0, -1.5], [np.nan, 0, -2]]], dtype=np.float32)
    _, vis, reason, hit = visibility(witnesses, np.ones((1, 6), dtype=np.uint16),
                                     np.ones((1, 6)), view, evaluated_bvh(scene), cfg["validation"])
    half = reduce_half(native_codes, cfg.get("resize_rounding") == "nearest_even")
    Image.fromarray(half).save(root/"calibration_half.png")
    resize_landmarks = {}
    marker = np.array(expected_colors[1]) * 255
    background = np.array(expected_colors[0]) * 255
    for name, image, v in [("full", native_codes, view), ("half", half, half_camera(view))]:
        distance_marker = np.linalg.norm(image-marker, axis=-1)
        distance_background = np.linalg.norm(image-background, axis=-1)
        ys, xs = np.nonzero(distance_marker < distance_background)
        if not len(ys):
            raise ValueError("The independently rendered foreground color swatch is missing")
        measured = np.array([(xs.min()+xs.max())/2, (ys.min()+ys.max())/2])
        predicted, _ = project([[0.17, 0.03, -1]], v)
        resize_landmarks[name] = {"projected_xy": predicted[0].tolist(), "image_marker_xy": measured.tolist(),
                                  "error_px": float(np.linalg.norm(measured - predicted[0]))}
    t = cfg["validation"]
    max_dz = max(abs(m["measured_z_m"]/m["expected_z_m"]-1) for m in measurements)
    tests = {"plane_depth": max_dz <= t["plane_relative_tolerance"],
             "separates_radial_and_axial": measurements[1]["ray_distance_m"] > measurements[1]["measured_z_m"] + 0.05,
             "plane_object_ids": all(m["object_expected"] == m["object_read"] for m in measurements),
             "known_swatches": max(m["color_abs_max"] for m in measurements) <= t["display_tolerance_code_values"]/255,
             "native_color_transform": whole_image_error <= t["display_tolerance_code_values"],
             "known_visible_occluded_outside_invalid": vis.tolist() == [[1, 2, 3, 3, 0, 0]]
                and reason.tolist() == [[1, 2, 3, 4, 8, 5]] and hit.tolist()[0][:2] == [1, 2],
             "pixel_center": max(m["reprojection_error_px"] for m in measurements) <= t["render_pixel_tolerance_px"],
             "resize_marker": max(m["error_px"] for m in resize_landmarks.values()) <= t["render_pixel_tolerance_px"]}
    np.savez_compressed(root/"calibration.npz", rgb=rgb, position=pos, z=z.astype(np.float32), object_id=ids)
    return {"tests": tests, "camera": view, "exr_channels": names, "swatches": measurements,
            "max_relative_depth_error": max_dz, "max_native_png_code_difference": whole_image_error,
            "first_hit_witness_order": ["observed", "occluded", "outside", "behind", "farther_hit", "nonfinite"],
            "visibility": vis.tolist()[0], "reason": reason.tolist()[0], "hit_ids": hit.tolist()[0],
            "actual_resized_landmarks": resize_landmarks}


def source_resize(source, path, root, world_points, ties_to_even=False):
    full = png_pixels(path)
    half = reduce_half(full, ties_to_even)
    image_path = root/"source_256.png"
    Image.fromarray(half).save(image_path)
    small_camera = half_camera(source)
    high, _ = project(world_points, source)
    low, _ = project(world_points, small_camera)
    np.savez_compressed(root/"decoded_input.npz", rgb_u8=full)
    return {"source_png_sha256": sha256(path), "half_png_sha256": sha256(image_path),
            "color_encoding": "display_srgb", "source_bit_depth": 8, "half_camera": small_camera,
            "method": ("2x2 box average of decoded display-sRGB code values, ties to even once, no crop/pad"
                       if ties_to_even else "2x2 box average of decoded display-sRGB code values, round half up once, no crop/pad"),
            "max_projection_residual_px": float(np.linalg.norm((high+0.5)/2-0.5-low, axis=1).max()),
            "stored_half_matches_resampler": bool(np.array_equal(png_pixels(image_path), half))}


def seed_comparison(primary_rgb, second_rgb, settings):
    delta = np.mean(np.abs(primary_rgb-second_rgb), axis=-1, dtype=np.float64)
    count = int(np.count_nonzero(delta > settings["noise_pixel_mae_threshold"]))
    mae = float(np.mean(delta, dtype=np.float64))
    return {"mae": mae, "exceeding_pixel_count": count, "pixel_count": delta.size,
            "fraction": count/delta.size, "pixel_threshold": settings["noise_pixel_mae_threshold"],
            "passed": mae <= settings["noise_mae_max"] and count/delta.size <= settings["noise_exceedance_fraction_max"]}
