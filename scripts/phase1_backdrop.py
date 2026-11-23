"""Check the camera background along the actual Blender sensor footprint and an exposed World control."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np

REF = Path(__file__).resolve().parents[1] / "src/evidence_lens/blender_reference"
sys.path.insert(0, str(REF))
from passes import half_camera, read_passes, sha256, write_json
from scene import camera_record
from validation import render


def solid(scene, index):
    obj, = [o for o in scene.objects if o.type == "MESH" and o.pass_index == index]
    mat, = obj.data.materials
    surface = mat.node_tree.nodes.get("Material Output").inputs["Surface"]
    if (not obj.visible_camera or obj.hide_render or len(surface.links) != 1 or
            surface.links[0].from_node.type != "BSDF_DIFFUSE" or len(obj.data.vertices) != 8):
        raise ValueError("The studio enclosure must contain opaque, camera-visible boxes")
    matrix = np.array(obj.matrix_world)
    vertices = np.array([v.co[:] for v in obj.data.vertices]) @ matrix[:3, :3].T + matrix[:3, 3]
    lo, hi = vertices.min(0), vertices.max(0)
    if not np.all((vertices == lo) | (vertices == hi)) or len(np.unique(vertices, axis=0)) != 8:
        raise ValueError("Expected an axis-aligned box")
    return lo, hi


def covered(view, wall, floor_box, padding):
    lo, hi = wall
    flo, fhi = floor_box
    camera = np.array(view["c2w"])
    pos = camera[:3, 3]
    width, height = view["width"], view["height"]
    rectangle = np.array([[x, y, 1.] for x in (-0.5-padding, width-0.5+padding)
                          for y in (-0.5-padding, height-0.5+padding)])
    directions = rectangle @ np.linalg.inv(view["K"]).T @ camera[:3, :3].T
    near, far = view["clip_m"]
    if min(directions[:, 1]) <= 0 or not pos[1] < lo[1]:
        raise ValueError("Camera does not face the studio wall")
    with np.errstate(divide="raise", invalid="raise"):
        distance = ((lo[1] + hi[1])/2 - pos[1]) / directions[:, 1]
    end = pos + distance[:, None] * directions
    start = pos + near * directions
    footprint = np.vstack([end, start, pos])
    margins = {
        "wall_left_m": float(end[:, 0].min() - lo[0]), "wall_right_m": float(hi[0] - end[:, 0].max()),
        "wall_ceiling_m": float(hi[2] - end[:, 2].max()),
        "floor_xy_m": float(min((footprint[:, :2] - flo[:2]).min(), (fhi[:2] - footprint[:, :2]).min())),
        "near_above_floor_m": float(start[:, 2].min() - fhi[2]),
        "wall_after_near_m": float(distance.min() - near), "wall_before_far_m": float(far - distance.max()),
    }
    if min(margins.values()) <= 0 or fhi[2] < lo[2]:
        raise ValueError(f"Open primary camera rays cannot be excluded: {margins}, floor/wall {fhi[2]}, {lo[2]}")
    return margins


def paired_control(scene, work, spec):
    def pixels(key):
        path = work / (key + ".exr")
        render(scene, path)
        linear, position, index, _ = read_passes(path)
        return linear, position, index

    original = pixels("closed-world")
    scene.world.cycles_visibility.camera = False
    hidden = pixels("closed-black")
    delta = float(np.max(np.abs(original[0] - hidden[0])))
    equal_geometry = (np.array_equal(original[1], hidden[1], equal_nan=True) and
                      np.array_equal(original[2], hidden[2]))
    for obj in scene.objects:
        if obj.type == "MESH":
            obj.hide_render = True
    scene.render.resolution_x = scene.render.resolution_y = 32
    scene.cycles.use_denoising = False
    black = pixels("open-black")
    scene.world.cycles_visibility.camera = True
    lit = pixels("open-world")
    open_max, open_min = float(np.max(np.abs(black[0]))), float(np.min(lit[0]))
    result = {"closed_rgb_max_abs_linear": delta, "closed_geometry_arrays_equal": equal_geometry,
              "open_camera_black_max_abs_linear": open_max, "open_camera_world_min_linear": open_min,
              "open_object_hits": int(np.count_nonzero(black[2]) + np.count_nonzero(lit[2]))}
    if (delta > spec["max_closed_linear_rgb_change"] or not equal_geometry or result["open_object_hits"] or
            open_max > spec["max_open_corrected_linear_rgb"] or open_min < spec["min_open_original_linear_rgb"]):
        raise ValueError(f"Backdrop control failed {result}")
    return result


def audit(evidence, spec, out):
    inventory = json.loads((evidence / "inventory.json").read_text())
    actual = []
    total_surface = total = 0
    with tempfile.TemporaryDirectory() as scratch:
        for i, item in enumerate(inventory["references"]):
            folder = evidence / item["family"]
            for name, pin in item["files"].items():
                if sha256(folder / name) != pin:
                    raise ValueError("Archived backdrop evidence differs from its input pin")
            bpy.ops.wm.open_mainfile(filepath=str(folder / "scene.blend"), load_ui=False, use_scripts=False)
            scene = bpy.context.scene
            if scene.render.engine != "CYCLES" or scene.cycles.filter_width > spec["padding_pixels"]:
                raise ValueError("Unexpected engine or filter extent")
            wall, floor_box = solid(scene, 1), solid(scene, 2)
            saved = json.loads((folder / "cameras.json").read_text())["views"]
            validation = json.loads((folder / "validation.json").read_text())
            family_rows = {}
            for name, view in saved.items():
                measured = camera_record(scene, scene.objects[name], view["purpose"])
                for key in ("K", "c2w", "w2c", "clip_m"):
                    if not np.allclose(measured[key], view[key], atol=1e-6, rtol=0):
                        raise ValueError("Loaded Blender camera and sealed camera differ")
                family_rows[name] = covered(view, wall, floor_box, spec["padding_pixels"])
                n = validation["view_geometry"][name]["surface_pixels"]
                if n != view["width"] * view["height"]:
                    raise ValueError("Archive reports uncovered central samples")
                total_surface += n
                total += view["width"] * view["height"]
            if item["split"] == "test":
                family_rows["R1-source"] = covered(half_camera(saved["source"]), wall, floor_box, spec["padding_pixels"])
            row = {"family": item["family"], "reference": item["parent"], "world_visible_to_camera": bool(scene.world.cycles_visibility.camera),
                   "sensor_bounds": family_rows, "floor_top_m": float(floor_box[1][2]), "wall_bottom_m": float(wall[0][2])}
            if i == 0:
                row["rendered_control"] = paired_control(scene, Path(scratch), spec)
            actual.append(row)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "backdrop.json", {"schema": "el-phase1-backdrop-evidence-1", "families": actual,
                                       "covered_B0_primary_pixels": total_surface, "total_B0_primary_pixels": total,
                                       "checked_sensor_grids": sum(len(x["sensor_bounds"]) for x in actual), "passed": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evidence", "protocol", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    audit(args.evidence, json.loads(args.protocol.read_text()), args.out)
