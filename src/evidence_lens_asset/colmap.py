"""Independent reader of the COLMAP 3.10 camera/image exchange layout."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from evidence_lens_bundle.common import Unsupported, require

MODELS = {"SIMPLE_PINHOLE": 3, "PINHOLE": 4}
MODEL_IDS = {0: "SIMPLE_PINHOLE", 1: "PINHOLE"}


def unique(dest, key, value, kind):
    require(type(key) is int and key >= 0 and key not in dest, "duplicate or invalid COLMAP " + kind + " ID")
    dest[key] = value


def intrinsics(model, width, height, params):
    require(model in MODELS, "unsupported or distorted COLMAP camera " + str(model), Unsupported)
    require(
        width > 0 and height > 0 and len(params) == MODELS[model],
        "invalid COLMAP camera dimensions or parameters",
    )
    require(np.isfinite(params).all(), "non-finite camera parameters")
    if model == "SIMPLE_PINHOLE":
        fx, cx, cy = params
        fy = fx
    else:
        fx, fy, cx, cy = params
    require(min(fx, fy) > 0, "non-positive focal length")
    k = [[fx, 0, cx - 0.5], [0, fy, cy - 0.5], [0, 0, 1]]
    return {"width": width, "height": height, "K": k, "projection": "pinhole", "distortion": "none"}


def pose(q, t, scale):
    q, t = np.asarray(q, dtype=np.float64), np.asarray(t, dtype=np.float64)
    require(
        q.shape == (4,)
        and t.shape == (3,)
        and np.isfinite(q).all()
        and np.isfinite(t).all()
        and abs(np.linalg.norm(q) - 1) < 1e-5,
        "invalid COLMAP world-to-camera pose",
    )
    w, x, y, z = q / np.linalg.norm(q)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    result = np.eye(4)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ (t * scale)
    return result.tolist()


def text_tables(root):
    cameras, images = {}, {}
    for line in (root / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        a = line.split()
        require(len(a) >= 4, "truncated cameras.txt row")
        unique(cameras, int(a[0]), intrinsics(a[1], int(a[2]), int(a[3]), list(map(float, a[4:]))), "camera")
    lines = iter((root / "images.txt").read_text(encoding="utf-8").splitlines())
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        a = line.split()
        require(
            len(a) == 10, "COLMAP 3.10 images.txt requires a single filename without whitespace", Unsupported
        )
        unique(
            images, int(a[0]), (int(a[8]), list(map(float, a[1:5])), list(map(float, a[5:8])), a[9]), "image"
        )
        points = next(lines, None)
        require(
            points is not None and not points.lstrip().startswith("#"), "missing images.txt observation line"
        )
        points = points.split()
        require(len(points) % 3 == 0, "invalid images.txt observations")
        for i in range(0, len(points), 3):
            require(
                np.isfinite([float(points[i]), float(points[i + 1])]).all(), "nonfinite image observation"
            )
            int(points[i + 2])
    return cameras, images


def take(f, n):
    raw = f.read(n)
    require(len(raw) == n, "truncated COLMAP binary member")
    return raw


def fields(f, spec):
    fmt = struct.Struct("<" + spec)
    return fmt.unpack(take(f, fmt.size))


def binary_tables(root):
    cameras, images = {}, {}
    with (root / "cameras.bin").open("rb") as f:
        for _ in range(fields(f, "Q")[0]):
            i, model, w, h = fields(f, "iiQQ")
            require(
                model in MODEL_IDS,
                "unsupported or distorted COLMAP camera model ID " + str(model),
                Unsupported,
            )
            kind = MODEL_IDS[model]
            unique(cameras, i, intrinsics(kind, w, h, fields(f, "d" * MODELS[kind])), "camera")
        require(not f.read(1), "unconsumed cameras.bin data")
    with (root / "images.bin").open("rb") as f:
        for _ in range(fields(f, "Q")[0]):
            vals = fields(f, "i7di")
            chars = bytearray()
            while (c := take(f, 1)) != b"\0":
                chars += c
                require(len(chars) <= 65535, "overlong image name")
            unique(images, vals[0], (vals[8], vals[1:5], vals[5:8], chars.decode("utf-8")), "image")
            n = fields(f, "Q")[0]
            # The 24-byte observations contain x, y and int64 point ID; they are not deployment labels.
            require(n <= (root / "images.bin").stat().st_size // 24, "oversized image observation table")
            f.seek(n * 24, 1)
            require(f.tell() <= (root / "images.bin").stat().st_size, "truncated image observation table")
        require(not f.read(1), "unconsumed images.bin data")
    return cameras, images


def load(root, encoding):
    root = Path(root)
    require(encoding in ("text", "binary"), "specify COLMAP text or binary explicitly")
    suffix = "txt" if encoding == "text" else "bin"
    for n in ("cameras", "images"):
        p = root / f"{n}.{suffix}"
        require(p.is_file() and not p.is_symlink(), "missing or symlink COLMAP " + p.name)
    c, i = (text_tables if encoding == "text" else binary_tables)(root)
    require(c and i and all(v[0] in c for v in i.values()), "missing registered COLMAP camera/image")
    return c, i
