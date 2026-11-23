"""Construct independently projected, source-frame metric fixtures for the phase-one GPU probe."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("source", *(f"v{i:02d}" for i in range(1, 7)))
POINTS = [[-0.36, -0.21, 1.65], [0.47, -0.11, 2.35], [-0.14, 0.40, 3.20]]
SOURCE_FILES = {
    "sharp": [
        "cli/predict.py",
        "utils/gaussians.py",
        "utils/color_space.py",
        "models/predictor.py",
        "models/initializer.py",
        "models/composer.py",
    ],
    "gsplat": ["rendering.py", "cuda/csrc/RasterizeToPixels3DGSFwd.cu", "cuda/include/Utils.cuh"],
    "evidence_lens": ["generator_adapter.py", "renderer_adapter.py", "_worker.py"],
}


def sha(path, text=False):
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if text else raw).hexdigest()


def project(point, source, target):
    """Transform via physical camera origins and world-space basis vectors; no inverse view matrix."""
    a, b = (np.asarray(c["c2w"], dtype=np.float64) for c in (source, target))
    world = a[:3, 3] + sum(a[:3, i] * point[i] for i in range(3))
    target_vector = world - b[:3, 3]
    x, y, z = [np.dot(target_vector, b[:3, i]) for i in range(3)]
    k = target["K"]
    return {
        "uv": [k[0][0] * x / z + k[0][2], k[1][1] * y / z + k[1][2]],
        "z_m": z,
        "range_m": np.linalg.norm(target_vector),
    }


def original_fixture_rgb(w, h):
    yy, xx = np.indices((h, w))
    return np.stack(((3 * xx + yy) % 251, (xx + 5 * yy) % 241, (7 * xx + 11 * yy) % 239), axis=-1).astype(
        np.uint8
    )


def rotate(axis, degrees):
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def custom(w, h, f, c2w):
    return {
        "width": w,
        "height": h,
        "K": [[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]],
        "c2w": c2w.tolist(),
        "clip_m": [0.05, 30],
    }


def registered(case, cameras):
    source = cameras["source"]
    return {
        "name": case,
        "cameras": cameras,
        "expected": {v: [project(p, source, c) for p in POINTS] for v, c in cameras.items()},
    }


def fixture(case, w, h, f):
    rgb = original_fixture_rgb(w, h)
    samples = []
    for ix, iy in ((0, 0), (1535, 1535), (441, 867), (1174, 327), (691, 1250)):
        # Independent scalar interpolation in the input grid, including both endpoints.
        x, y = ix * (w - 1) / 1535, iy * (h - 1) / 1535
        x0, y0 = int(x), int(y)
        dx, dy = x - x0, y - y0
        val = sum(
            float(ax * ay) * rgb[min(y0 + j, h - 1), min(x0 + i, w - 1)].astype(float)
            for i, ax in enumerate((1 - dx, dx))
            for j, ay in enumerate((1 - dy, dy))
        )
        samples.append({"internal_xy": [ix, iy], "rgb": (val / 255).tolist()})
    pts = np.asarray(POINTS, float)
    return {
        "name": case,
        "width": w,
        "height": h,
        "f_px": f,
        "disparity_factor": f / w,
        "native_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "ndc_means": (pts * np.array([2 * f / w, 2 * f / h, 1])).tolist(),
        "samples": samples,
    }


def specification():
    official = ROOT / "configs/formal-cameras-v1.json"
    cameras = json.loads(official.read_text())["views"]
    brief = {k: {n: c[n] for n in ("width", "height", "K", "c2w", "clip_m")} for k, c in cameras.items()}
    half = copy.deepcopy(brief)
    v = half["source"]
    v.update(width=256, height=256)
    v["K"] = (np.array([[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1]]) @ v["K"]).tolist()
    src = np.eye(4)
    src[:3, :3] = rotate("z", 90)
    src[:3, 3] = [1.25, -0.75, 0.40]
    relative = np.eye(4)
    relative[:3, :3] = rotate("y", 12) @ rotate("z", -7)
    relative[:3, 3] = [0.11, -0.07, -0.16]
    asym = {"source": custom(96, 64, 80, src), "oblique": custom(112, 72, 90, src @ relative)}
    asym["oblique"]["K"] = [[90, 0, 54.2], [0, 77, 31.7], [0, 0, 1]]
    locations = {
        "sharp": ROOT / ".research/upstream/ml-sharp/src/sharp",
        "gsplat": ROOT / ".research/upstream/gsplat/gsplat",
        "evidence_lens": ROOT / "src/evidence_lens",
    }
    sha_sets = {
        group: {f: sha(locations[group] / f, True) for f in paths} for group, paths in SOURCE_FILES.items()
    }
    return {
        "schema": "el-phase1-geometry-probe-1",
        "protocol_version": "1.0",
        "registered_utc": "2026-11-23T08:28:10Z",
        "purpose": "Bypass learned weights, test official preprocessing/unprojection and metric PLY/renderer/ED. Expected values are pre-GPU.",
        "camera_domain": "integer pixel centres, x right, y down, z positive forward; physical lengths in metres",
        "formal_camera_sha256": sha(official),
        "baseline_config_sha256": sha(ROOT / "configs/windows-b0.json"),
        "sharp_commit": "aed6527499ef91cba3b54c18d49a870f25947190",
        "gsplat_commit": "937e29912570c372bed6747a5c9bf85fed877bae",
        "normalized_source_sha256": sha_sets,
        "means_m": POINTS,
        "scales_m": [[0.037, 0.044, 0.032], [0.056, 0.048, 0.040], [0.067, 0.079, 0.053]],
        "wxyz": [[1, 0, 0, 0], [0.8, 0, 0.6, 0], [0.6, 0, 0, 0.8]],
        "linear_rgb": [[0.17, 0.43, 0.74], [0.71, 0.26, 0.39], [0.31, 0.78, 0.12]],
        "opacity": [0.38, 0.61, 0.74],
        "tolerance": {
            "metric_m": 2e-4,
            "centroid_px": 0.05,
            "opacity": 0.002,
            "linear_rgb": 0.002,
            "colour_srgb": 0.002,
            "ply_field": 3e-6,
            "preprocess": 1e-4,
            "f_over_width": 2e-7,
        },
        "tolerance_basis": "Legacy 0.2 mm ED and 0.002 alpha/colour. 0.05 pixel on isolated log-Gaussian fits separates a half-pixel error tenfold. 3e-6 accounts for FP32 log/exp and sRGB round trips. 1e-4 RGB absorbs FP32 interpolation over a 1536-grid.",
        "scenes": [registered("B0", brief), registered("R1", half), registered("asymmetric", asym)],
        "official_predict_image_witness": [
            fixture("B0", 512, 512, brief["source"]["K"][0][0]),
            fixture("R1", 256, 256, v["K"][0][0]),
            fixture("rectangular", 96, 64, 80),
            fixture("odd", 95, 63, 67),
        ],
        "layered": {
            "camera": asym["source"],
            "uv": [31, 24],
            "z_m": [1.5, 3.5],
            "opacity": [0.25, 0.5],
            "linear_rgb": [[0.2, 0.5, 0.9], [0.8, 0.15, 0.3]],
            "expected": {
                "alpha": 0.625,
                "D": 1.6875,
                "ED": 2.7,
                "linear_rgb": [0.35, 0.18125, 0.3375],
                "empty_alpha": 0,
                "empty_z": "NaN",
            },
        },
        "negative_control": "Deliberately remove 0.5 from the custom target principal point. Measured minus correct centre must be (-0.5,-0.5) within 0.05px on both axes.",
        "missing_depth": "Native renderer writes NaN for zero-alpha background. Forced unverified calibration semantics must write all-Z NaN, while native colour and alpha remain identical. The forced gate tests branch behavior only.",
    }


def centroid(opacity):
    """Estimate a subpixel centre from the measured log of a single isolated Gaussian silhouette."""
    y, x = np.nonzero(opacity > np.max(opacity) * 0.05)
    if len(x) < 6:
        raise ValueError("insufficient independent alpha samples")
    x0, y0 = x.mean(), y.mean()
    dx, dy = x - x0, y - y0
    design = np.column_stack((np.ones(x.shape), dx, dy, dx * dx, dx * dy, dy * dy))
    fit, _, rank, _ = np.linalg.lstsq(design, np.log(opacity[y, x].astype(float)), rcond=None)
    hessian = np.array([[2 * fit[3], fit[4]], [fit[4], 2 * fit[5]]])
    if rank != 6 or np.max(np.linalg.eigvalsh(hessian)) >= 0:
        raise ValueError("indeterminate alpha centre")
    return np.array([x0, y0]) - np.linalg.solve(hessian, fit[1:3])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    with args.out.open("x") as fp:
        json.dump(specification(), fp, indent=2, allow_nan=False)
        fp.write("\n")
