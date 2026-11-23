"""Read the pinned SHARP PLY layout and check its declared camera and colour encoding."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

VERTICES = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
SUPPLEMENTS = (("extrinsic", 16, "float", "<f4"), ("intrinsic", 9, "float", "<f4"),
               ("image_size", 2, "uint", "<u4"), ("frame", 2, "int", "<i4"),
               ("disparity", 2, "float", "<f4"), ("color_space", 1, "uchar", "u1"),
               ("version", 3, "uchar", "u1"))


def read_ply(path):
    """Return read-only vertices and metadata for the registered binary layout."""
    with path.open("rb") as stream:
        def line():
            return stream.readline(256).decode("ascii").strip()
        if (line(), line()) != ("ply", "format binary_little_endian 1.0"):
            raise ValueError("expected binary little-endian PLY 1.0")
        element, kind, count = line().split()
        if (element, kind) != ("element", "vertex") or not count.isdigit() or int(count) < 1:
            raise ValueError("expected the SHARP vertex count")
        for key in VERTICES:
            if line() != "property float " + key:
                raise ValueError(f"unexpected SHARP field layout at {key}")
        for name, n, declaration, _ in SUPPLEMENTS:
            if (line(), line()) != (f"element {name} {n}", f"property {declaration} {name}"):
                raise ValueError(f"unexpected SHARP supplemental layout at {name}")
        if line() != "end_header":
            raise ValueError("unexpected extra header fields")
        offset = stream.tell()
        vertex_dtype = np.dtype([(key, "<f4") for key in VERTICES])
        byte_count = vertex_dtype.itemsize * int(count)
        expected_size = offset + byte_count + sum(n * np.dtype(dt).itemsize for _, n, _, dt in SUPPLEMENTS)
        if path.stat().st_size != expected_size:
            raise ValueError("actual and declared Gaussian file sizes disagree")
        vertices = np.memmap(path, mode="r", dtype=vertex_dtype, offset=offset, shape=(int(count),))
        stream.seek(offset + byte_count)
        extras = {name: np.fromfile(stream, dtype=dt, count=n) for name, n, _, dt in SUPPLEMENTS}
    return vertices, extras


def inspect(ply, cameras, generation):
    vertices, extras = read_ply(ply)
    meta = json.loads(generation.read_text(encoding="utf-8"))
    camera = json.loads(cameras.read_text(encoding="utf-8"))["views"]["source"]
    with ply.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with cameras.open("rb") as stream:
        camera_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    matrix = extras["intrinsic"].reshape(3, 3)
    k_error = float(np.abs(matrix - np.asarray(camera["K_edge"])).max())
    if (digest != meta["ply_sha256"] or camera_hash != meta["camera_sha256"]
            or len(vertices) != meta["gaussian_count"] or ply.stat().st_size != meta["ply_bytes"]
            or extras["image_size"].tolist() != [camera["width"], camera["height"]]
            or extras["color_space"].tolist() != [0] or k_error > 1e-4
            or not np.array_equal(extras["extrinsic"].reshape(4, 4), np.eye(4))):
        raise ValueError("PLY provenance, camera or sRGB colour-space metadata differs")
    fields = {}
    for key in VERTICES:
        data = np.ascontiguousarray(vertices[key])
        finite = np.isfinite(data)
        fields[key] = {"storage_dtype": "<f4", "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                       "finite": int(finite.sum()), "nan": int(np.isnan(data).sum()),
                       "positive_infinite": int(np.isposinf(data).sum()), "negative_infinite": int(np.isneginf(data).sum()),
                       "min_finite": float(data[finite].min()) if finite.any() else None,
                       "max_finite": float(data[finite].max()) if finite.any() else None}
    return {"schema": "el-sharp-ply-readback-1", "generation_run_id": meta["producer_run_id"],
            "ply_sha256": digest, "camera_sha256": camera_hash, "count": len(vertices),
            "ply_bytes": ply.stat().st_size, "fields": fields,
            "source_frame": "identity extrinsic in source camera coordinates",
            "camera_K_edge_max_difference_px": k_error,
            "color": "tag 0, sRGB represented by the three degree-zero SH fields",
            "metadata": {k: v.tolist() for k, v in extras.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("ply", "cameras", "generation", "out"):
        parser.add_argument("--" + role, required=True, type=Path)
    args = parser.parse_args()
    receipt = inspect(args.ply, args.cameras, args.generation)
    args.out.mkdir(exist_ok=True, parents=True)
    with (args.out / "gaussian-readback.json").open("x", encoding="utf-8") as f:
        json.dump(receipt, f, allow_nan=False, indent=2)
        f.write("\n")
