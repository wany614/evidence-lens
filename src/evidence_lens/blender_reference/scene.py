"""Construct the procedural world and measure calibration from actual Blender objects."""
import os
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def configure(scene, cfg, width, height):
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    render = scene.render
    render.engine = cfg["engine"]
    render.resolution_x, render.resolution_y = width, height
    render.resolution_percentage = 100
    render.pixel_aspect_x = render.pixel_aspect_y = 1.0
    render.film_transparent = False
    render.use_motion_blur = False
    render.use_compositing = False
    render.use_sequencer = False
    render.dither_intensity = 0.0
    render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    render.image_settings.color_depth = "32"
    render.image_settings.color_mode = "RGBA"
    render.image_settings.exr_codec = "ZIP"
    render.image_settings.color_management = "FOLLOW_SCENE"
    scene.display_settings.display_device = cfg["color"]["display"]
    view = scene.view_settings
    view.view_transform = cfg["color"]["view"]
    view.look = cfg["color"]["look"]
    view.exposure = cfg["color"]["exposure"]
    view.gamma = cfg["color"]["gamma"]
    view.use_curve_mapping = False
    view.use_white_balance = False
    view.use_hdr_view = False
    layer = scene.view_layers[0]
    layer.name = "Reference"
    layer.use_pass_position = True
    layer.use_pass_object_index = True
    cycles = scene.cycles
    cycles.device = cfg["device"]
    cycles.samples = cfg["samples"]
    cycles.seed = cfg["seed"]
    cycles.use_animated_seed = False
    cycles.use_adaptive_sampling = False
    cycles.use_denoising = True
    cycles.denoiser = cfg["denoiser"]
    cycles.denoising_use_gpu = False
    cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cycles.denoising_prefilter = "ACCURATE"
    cycles.max_bounces = cfg["max_bounces"]
    cycles.diffuse_bounces = cfg["diffuse_bounces"]
    cycles.pixel_filter_type = cfg["pixel_filter"]
    cycles.filter_width = cfg["filter_width"]


def camera(scene, name, settings, location, aim=None):
    data = bpy.data.cameras.new(name)
    data.type = "PERSP"
    data.lens = settings["lens_mm"]
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = settings["sensor_mm"]
    data.clip_start, data.clip_end = settings["clip_m"]
    data.dof.use_dof = False
    data.shift_x = data.shift_y = 0.0
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = location
    if aim is not None:
        obj.rotation_euler = (Vector(aim) - obj.location).to_track_quat("-Z", "Y").to_euler()
    return obj


def material(name, color, roughness=None, checker=None, checker_scale=None):
    result = bpy.data.materials.new(name)
    result.diffuse_color = (*color, 1.0)
    result.use_nodes = True
    tree = result.node_tree
    tree.nodes.clear()
    emission = roughness is None
    shader = tree.nodes.new("ShaderNodeEmission" if emission else "ShaderNodeBsdfDiffuse")
    shader.inputs["Color"].default_value = (*color, 1.0)
    if not emission:
        shader.inputs["Roughness"].default_value = roughness
    if checker is not None:
        coords = tree.nodes.new("ShaderNodeTexCoord")
        pattern = tree.nodes.new("ShaderNodeTexChecker")
        pattern.inputs["Color1"].default_value = (*color, 1.0)
        pattern.inputs["Color2"].default_value = (*checker, 1.0)
        pattern.inputs["Scale"].default_value = checker_scale
        tree.links.new(coords.outputs["Generated"], pattern.inputs["Vector"])
        tree.links.new(pattern.outputs["Color"], shader.inputs["Color"])
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(shader.outputs[0], output.inputs["Surface"])
    return result


def make_object(spec, mat):
    if "size" in spec:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=spec["center"])
        obj = bpy.context.object
        obj.dimensions = spec["size"]
    else:
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=spec["radius"], depth=spec["depth"],
                                            location=spec["center"])
        obj = bpy.context.object
    obj.name = spec["name"]
    obj.pass_index = spec["id"]
    obj.data.materials.append(mat)
    return obj


def build_world(cfg):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "scene_001_tabletop_v001"
    configure(scene, cfg, cfg["width"], cfg["height"])
    mats = {name: material(name, spec["linear"], cfg["material_roughness"],
                           spec.get("checker_linear"), spec.get("scale"))
            for name, spec in cfg["materials"].items()}
    for spec in cfg["objects"]:
        make_object(spec, mats[spec["material"]])
    world = bpy.data.worlds.new("Diffuse skylight")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (*cfg["world_color_linear"], 1.0)
    background.inputs["Strength"].default_value = cfg["world_strength"]
    scene.world = world
    spec = cfg["area_light"]
    light = bpy.data.lights.new("Soft area light", "AREA")
    light.shape = "DISK"
    light.size = spec["size_m"]
    light.energy = spec["energy_w"]
    light.color = spec["color_linear"]
    obj = bpy.data.objects.new(light.name, light)
    scene.collection.objects.link(obj)
    obj.location = spec["location"]
    obj.rotation_euler = (Vector(spec["aim"]) - obj.location).to_track_quat("-Z", "Y").to_euler()
    cam = cfg["camera"]
    cameras = {name: camera(scene, name, cam, pos, cam["aim"]) for name, pos in cam["positions"].items()}
    scene.camera = cameras["source"]
    bpy.context.view_layer.update()
    return scene, cameras


def camera_record(scene, obj, purpose):
    width, height = scene.render.resolution_x, scene.render.resolution_y
    focal = obj.data.lens * width / obj.data.sensor_width
    k = np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]])
    k_edge = k.copy()
    k_edge[:2, 2] += 0.5
    # Blender looks along local -Z with local +Y up.
    c2w = np.array(obj.matrix_world, dtype=np.float64) @ np.diag([1, -1, -1, 1])
    return {"width": width, "height": height, "K": k.tolist(), "K_edge": k_edge.tolist(),
            "c2w": c2w.tolist(), "w2c": np.linalg.inv(c2w).tolist(),
            "purpose": purpose, "projection": "pinhole", "lens_mm": obj.data.lens,
            "sensor_fit": obj.data.sensor_fit, "sensor_width_mm": obj.data.sensor_width,
            "sensor_height_mm": obj.data.sensor_height, "lens_shift": [obj.data.shift_x, obj.data.shift_y],
            "pixel_aspect": [scene.render.pixel_aspect_x, scene.render.pixel_aspect_y],
            "distortion": "none", "clip_m": [obj.data.clip_start, obj.data.clip_end]}


def project(points, view):
    world = np.asarray(points, dtype=np.float64)
    w2c = np.asarray(view["w2c"])
    cv = world @ w2c[:3, :3].T + w2c[:3, 3]
    image = cv @ np.asarray(view["K"]).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return image[..., :2] / image[..., 2:3], cv[..., 2]


def validate_camera(scene, obj, view, points):
    expected = []
    for world in points:
        mapped = world_to_camera_view(scene, obj, Vector(world))
        expected.append([mapped.x * view["width"] - 0.5, (1 - mapped.y) * view["height"] - 0.5])
    actual, z = project(points, view)
    native_error = np.linalg.norm(actual - expected, axis=1)
    local = np.array([[0, 0, -2], [0.2, 0, -2], [0, 0.2, -2], [0, 0, -3]], dtype=np.float64)
    blender_pose = np.asarray(obj.matrix_world)
    world = local @ blender_pose[:3, :3].T + blender_pose[:3, 3]
    uv, depth = project(world, view)
    return {"point_count": len(points), "max_error_px": float(native_error.max()),
            "min_calibration_point_z_m": float(z.min()),
            "right_moves_right": bool(uv[1, 0] > uv[0, 0]), "up_moves_up": bool(uv[2, 1] < uv[0, 1]),
            "forward_increases_z": bool(depth[3] > depth[0] > 0),
            "rotation_determinant": float(np.linalg.det(np.asarray(view["c2w"])[:3, :3]))}


def evaluated_bvh(scene):
    """Build first-hit acceleration from evaluated, world-space triangles of the frozen opaque meshes."""
    graph = bpy.context.evaluated_depsgraph_get()
    vertices, triangles, owner = [], [], []
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            base = len(vertices)
            vertices.extend(evaluated.matrix_world @ v.co for v in mesh.vertices)
            triangles.extend(tuple(base + i for i in tri.vertices) for tri in mesh.loop_triangles)
            owner.extend([obj.pass_index] * len(mesh.loop_triangles))
        finally:
            evaluated.to_mesh_clear()
    return BVHTree.FromPolygons(vertices, triangles, all_triangles=True), np.asarray(owner, dtype=np.uint16)


def ocio_config_path():
    if os.environ.get("OCIO"):
        return Path(os.environ["OCIO"]).resolve(strict=True)
    directory = bpy.utils.system_resource("DATAFILES", path="colormanagement")
    if directory and (Path(directory) / "config.ocio").is_file():
        return Path(directory) / "config.ocio"
    raise RuntimeError("Cannot identify the active Blender OpenColorIO configuration")


def render_settings(scene):
    c, v, r = scene.cycles, scene.view_settings, scene.render
    return {"blender_version": bpy.app.version_string, "build_hash": bpy.app.build_hash.decode(),
            "ocio_version": list(bpy.app.ocio.version), "engine": r.engine, "device": c.device,
            "samples": c.samples, "adaptive_sampling": c.use_adaptive_sampling,
            "denoising": c.use_denoising, "denoiser": c.denoiser, "denoising_gpu": c.denoising_use_gpu,
            "denoising_input_passes": c.denoising_input_passes, "denoising_prefilter": c.denoising_prefilter,
            "max_bounces": c.max_bounces, "diffuse_bounces": c.diffuse_bounces,
            "pixel_filter": c.pixel_filter_type, "filter_width": c.filter_width, "seed": c.seed,
            "display": scene.display_settings.display_device, "view": v.view_transform, "look": v.look,
            "exposure": v.exposure, "gamma": v.gamma, "curves": v.use_curve_mapping,
            "white_balance": v.use_white_balance, "hdr": v.use_hdr_view, "dither": r.dither_intensity,
            "motion_blur": r.use_motion_blur, "film_transparent": r.film_transparent,
            "camera_dof": scene.camera.data.dof.use_dof, "unit": scene.unit_settings.system,
            "meters_per_unit": scene.unit_settings.scale_length}
