"""Prepare the registered R1 source bundle or Q1 colour-rounded copy of an accepted B0 PLY."""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from inspect_sharp_ply import VERTICES, read_ply

COLOUR_FIELDS = ("f_dc_0", "f_dc_1", "f_dc_2")
CAMERA_FIELDS = {"K", "K_edge", "c2w", "w2c", "width", "height", "clip_m", "distortion", "lens_mm",
                 "lens_shift", "pixel_aspect", "projection", "purpose", "sensor_fit", "sensor_height_mm",
                 "sensor_width_mm"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def half_source(rgb, cameras):
    height, width, channels = rgb.shape
    if channels != 3 or rgb.dtype != np.uint8 or width % 2 or height % 2:
        raise ValueError("R1 expects an even-sized RGB8 source")
    src = cameras["views"]["source"]
    if set(src) - CAMERA_FIELDS:
        raise ValueError(f"unrecognized source camera field: {sorted(set(src) - CAMERA_FIELDS)}")
    if (cameras["schema"] != "el-camera-1" or cameras["pixel_center"] != "integer"
            or (src["width"], src["height"]) != (width, height)):
        raise ValueError("input dimensions or camera schema differ")
    # Integer sums divided by four are exact in binary. rint implements ties to even.
    sums = rgb.reshape(height // 2, 2, width // 2, 2, 3).sum(axis=(1, 3), dtype=np.uint16)
    reduced = np.rint(sums / 4).astype(np.uint8)
    q, r = np.divmod(sums, 4)
    if not np.array_equal(reduced, q + ((r > 2) | ((r == 2) & ((q % 2) == 1)))):
        raise ValueError("independent integer R1 readback differs")
    edge = np.asarray(src["K_edge"], dtype=np.float64)
    centre_to_edge = np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]])
    if not np.allclose(edge, centre_to_edge @ src["K"], atol=1e-8, rtol=0):
        raise ValueError("source centre and edge intrinsics disagree")
    half_edge = np.diag([.5, .5, 1.]) @ edge
    new_source = dict(src, width=width // 2, height=height // 2, K_edge=half_edge.tolist(),
                      K=np.linalg.solve(centre_to_edge, half_edge).tolist())
    return reduced, dict(cameras, views=dict(cameras["views"], source=new_source))


def reduce_input(image_path, camera_path, reference_path, output, producer):
    reference = load(reference_path)
    source_sha, camera_sha = sha(image_path), sha(camera_path)
    if not (reference.get("validation_passed") is True and reference["input_sha256"] == source_sha
            and reference["camera_sha256"] == camera_sha):
        raise ValueError("R1 inputs differ from their accepted B0 reference")
    with Image.open(image_path) as image:
        if image.format != "PNG" or image.mode != "RGB" or {"icc_profile", "gamma", "srgb", "exif"} & image.info.keys():
            raise ValueError("R1 requires the original RGB8 PNG")
        reduced, cameras = half_source(np.asarray(image), load(camera_path))
    if (output / "input.png").exists():
        raise FileExistsError(output / "input.png")
    Image.fromarray(reduced).save(output / "input.png")
    write(output / "cameras.json", cameras)
    asset = {"schema": "el-r1-source-1", "condition_id": "R1", "producer_run_id": producer,
             "algorithm": "2x2 box mean in display-sRGB code values, ties to nearest even uint8, no crop",
             "parent_input_sha256": source_sha, "parent_camera_sha256": camera_sha,
             "parent_reference_sha256": sha(reference_path), "input_sha256": sha(output / "input.png"),
             "camera_sha256": sha(output / "cameras.json"), "source_size_wh": list(reduced.shape[:2][::-1])}
    write(output / "source_derivation.json", asset)
    # Evaluation-only derivative. The six target view hashes and Blender validation stay pinned.
    bound_reference = dict(reference, input_sha256=asset["input_sha256"], camera_sha256=asset["camera_sha256"],
                           source_condition="R1", source_ground_truth_scope="Targets only; the B0 source truth raster has its original camera and dimensions.",
                           source_derivation_sha256=sha(output/"source_derivation.json"),
                           parent_reference_sha256=asset["parent_reference_sha256"])
    write(output / "reference.json", bound_reference)


def field_digest(vertices, name):
    return hashlib.sha256(np.ascontiguousarray(vertices[name]).tobytes()).hexdigest()


def envelope(path, vertices):
    with path.open("rb") as stream:
        header = stream.read(vertices.offset)
        stream.seek(vertices.offset + vertices.nbytes)
        tail = stream.read()
    return hashlib.sha256(header).hexdigest(), hashlib.sha256(tail).hexdigest()


def round_colour(ply, generation_path, output, producer):
    start = time.monotonic()
    base = load(generation_path)
    data, _ = read_ply(ply)
    if data.dtype.names != tuple(VERTICES):
        raise ValueError("unsupported vertex layout")
    if (base.get("schema") != "el-generation-1" or base["ply_sha256"] != sha(ply)
            or base["gaussian_count"] != len(data) or base["ply_bytes"] != ply.stat().st_size):
        raise ValueError("Q1 needs the exact accepted B0 PLY and generation metadata")
    changes = {}
    converted = {}
    for name in COLOUR_FIELDS:
        source = data[name]
        with np.errstate(over="ignore", invalid="ignore"):
            result = source.astype("<f2").astype("<f4")
        if not np.isfinite(source).all() or not np.isfinite(result).all():
            raise ValueError(f"nonfinite or FP16 overflow in {name}")
        converted[name] = result
        changes[name] = {"values_changed": int(np.count_nonzero(result != source)),
                         "max_abs_change": float(np.max(np.abs(result - source)))}
    target = output / "scene.ply"
    with ply.open("rb") as source, target.open("xb") as destination:
        shutil.copyfileobj(source, destination, 1024 * 1024)
    rounded = np.memmap(target, dtype=data.dtype, mode="r+", offset=data.offset, shape=data.shape)
    for name, result in converted.items():
        rounded[name] = result
    rounded.flush()
    del rounded
    checked, _ = read_ply(target)
    if envelope(ply, data) != envelope(target, checked):
        raise ValueError("PLY header or supplementary data changed")
    fields = {}
    for name in VERTICES:
        before, after = field_digest(data, name), field_digest(checked, name)
        if name not in COLOUR_FIELDS and before != after:
            raise ValueError(f"protected field changed: {name}")
        fields[name] = {"input_sha256": before, "output_sha256": after, **changes.get(name, {})}
    info = {"schema": "el-gaussian-derivation-1", "condition_id": "Q1", "producer_run_id": producer,
            "parent_generation_run_id": base["producer_run_id"], "parent_ply_sha256": base["ply_sha256"],
            "generator_commit": base["generator_commit"], "checkpoint_sha256": base["checkpoint_sha256"],
            "input_sha256": base["input_sha256"], "camera_sha256": base["camera_sha256"],
            "ply_sha256": sha(target), "ply_bytes": target.stat().st_size, "gaussian_count": len(checked),
            "schema_fields": fields, "header_and_metadata_sha256": list(envelope(target, checked)),
            "colour_fields": list(COLOUR_FIELDS), "rounding": "binary32 -> binary16 ties-even -> binary32",
            "forward_calls_in_this_stage": 0, "derivation_seconds": time.monotonic() - start}
    write(output / "generation.json", info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="condition", required=True)
    for condition, inputs in (("R1", ["input", "cameras", "reference"]), ("Q1", ["ply", "generation"])):
        sub = subs.add_parser(condition)
        for key in (*inputs, "out"):
            sub.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    producer = os.environ.get("EL_RUN_ID")
    if not producer:
        parser.error("use the recorder to assign the scientific derivation ID")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.condition == "R1":
        reduce_input(args.input, args.cameras, args.reference, args.out, producer)
    else:
        round_colour(args.ply, args.generation, args.out, producer)


if __name__ == "__main__":
    main()
