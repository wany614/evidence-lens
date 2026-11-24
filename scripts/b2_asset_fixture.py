"""Author the native bead study, two calibrated COLMAP serializations and three asset files."""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path
import numpy as np
from PIL import Image

from evidence_lens_bundle.common import sha, write

IDS = ("41", "5", "70", "109", "13")
MODELS = {17: ("PINHOLE", 203.0, 211.0, 128.25, 126.75), 93: ("SIMPLE_PINHOLE", 217.0, 128.75, 129.25)}


def camera_rig():
    cameras = {}
    poses = {}
    rig = [
        (-0.3125, 0.0625, 0.0, 0.11),
        (0.375, -0.03125, 0.03125, -0.13),
        (-0.5, -0.09375, 0.0625, 0.16),
        (0.5, 0.09375, -0.03125, -0.18),
        (0.0625, 0.1875, 0.03125, -0.04),
    ]
    for i, (name, values) in enumerate(zip(IDS, rig)):
        x, y, z, angle = values
        center = np.array([x, y, z])
        c, s = np.cos(angle), np.sin(angle)
        r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        pose = np.eye(4)
        pose[:3, :3] = r
        pose[:3, 3] = center
        cid = (17, 93)[i % 2]
        _, *f = MODELS[cid]
        fx, cx, cy = f if len(f) == 3 else (f[0], f[2], f[3])
        fy = fx if len(f) == 3 else f[1]
        cameras[name] = {
            "width": 256,
            "height": 256,
            "K": [[fx, 0.0, cx - 0.5], [0.0, fy, cy - 0.5], [0.0, 0.0, 1.0]],
            "c2w": pose.tolist(),
            "projection": "pinhole",
            "distortion": "none",
        }
        filename = (
            ("acquired/left-lens.png", "acquired/endpoint.png")[i] if i < 2 else "unobserved/" + name + ".png"
        )
        poses[name] = (
            cid,
            [np.cos(angle / 2), 0.0, -np.sin(angle / 2), 0.0],
            (-r.T @ (center * 100.0)).tolist(),
            filename,
        )
    return cameras, poses


def write_colmap(directory, poses):
    for mode in ("text", "binary"):
        (directory / mode).mkdir(parents=True)
    lines = []
    blob = bytearray(struct.pack("<Q", len(MODELS)))
    for cid, (model, *f) in MODELS.items():
        lines.append(f"{cid} {model} 256 256 " + " ".join(map(str, f)))
        blob.extend(
            struct.pack("<iiQQ" + "d" * len(f), cid, {"PINHOLE": 1, "SIMPLE_PINHOLE": 0}[model], 256, 256, *f)
        )
    (directory / "text/cameras.txt").write_text("\n".join(lines) + "\n")
    (directory / "binary/cameras.bin").write_bytes(blob)
    lines = []
    blob = bytearray(struct.pack("<Q", len(poses)))
    for name, (cid, q, t, filename) in poses.items():
        lines.extend([name + " " + " ".join(format(v, ".17g") for v in (*q, *t)) + f" {cid} " + filename, ""])
        blob.extend(
            struct.pack("<i7di", int(name), *q, *t, cid) + filename.encode() + b"\0" + struct.pack("<Q", 0)
        )
    (directory / "text/images.txt").write_text("\n".join(lines) + "\n")
    (directory / "binary/images.bin").write_bytes(blob)
    (directory / "text/points3D.txt").write_bytes(b"# Original authored cameras; no sparse tracks.\n")
    (directory / "binary/points3D.bin").write_bytes(struct.pack("<Q", 0))


def native_values():
    rng = np.random.Generator(np.random.PCG64(9312))
    n = 145
    t = np.linspace(0, 5 * np.pi, n)
    ring = 0.43 + 0.18 * np.cos(t * 0.39)
    centers = (
        np.round(
            np.column_stack((ring * np.cos(t), 0.73 * np.linspace(-1, 1, n), 2.40 + ring * np.sin(t))) * 128
        )
        / 128
    )
    quats = rng.normal(size=(n, 4))
    quats /= np.linalg.norm(quats, axis=1)[:, None]
    radius = rng.uniform(0.022, 0.070, (n, 3))
    center_color = np.column_stack(
        (0.47 + 0.30 * np.cos(t), 0.47 + 0.30 * np.sin(t * 0.77), 0.48 + 0.24 * np.cos(t * 0.39))
    )
    sh = rng.uniform(-0.035, 0.035, (n, 16, 3))
    sh[:, 0, :] = (center_color - 0.5) / np.sqrt(1 / (4 * np.pi))
    return {
        k: np.ascontiguousarray(v, np.float32)
        for k, v in {
            "centers_m": centers,
            "radii_m": radius,
            "quaternions_wxyz": quats,
            "alpha": rng.uniform(0.38, 0.88, n),
            "linear_color": center_color,
            "sh": sh,
        }.items()
    }


def source_picture(camera, a, path):
    # A distinct optical source: ray intersections with opaque spheres of the authored bead study.
    k = np.asarray(camera["K"])
    pose = np.asarray(camera["c2w"])
    y, x = np.indices((256, 256))
    ray = (
        np.stack(((x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1], np.ones_like(x)), axis=-1)
        @ pose[:3, :3].T
    )
    ray /= np.linalg.norm(ray, axis=-1)[..., None]
    z = np.full((256, 256), np.inf)
    rgb = np.zeros((256, 256, 3))
    for i, center in enumerate(a["centers_m"]):
        v = pose[:3, 3] - center
        sc = np.max(a["radii_m"][i])
        b = ray @ v
        disc = b * b - (v @ v - sc * sc)
        t = -b - np.sqrt(np.maximum(disc, 0))
        keep = (disc >= 0) & (t > 0) & (t < z)
        if keep.any():
            n = (v + ray[keep] * t[keep, None]) / sc
            rgb[keep] = (
                a["linear_color"][i]
                * np.clip(0.35 + 0.65 * (n @ np.array([-0.36, -0.48, -0.8])), 0.25, 1)[:, None]
            )
            z[keep] = t[keep]
    display = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.maximum(rgb, 0.0031308) ** (1 / 2.4) - 0.055)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(display, 0, 1) * 255).astype("u1")).save(path)


def save_assets(out, a):
    means = a["centers_m"] * 100
    ls = np.log(a["radii_m"] * 100)
    op = np.log(a["alpha"] / (1 - a["alpha"]))
    col = a["linear_color"]
    logits = np.log(col / (1 - col))
    np.savez(
        out / "project.npz",
        means=means,
        log_scales=ls,
        quats=a["quaternions_wxyz"],
        opacity_logits=op,
        color_logits=logits,
    )
    for degree in (0, 3):
        columns = [
            *["x", "y", "z", "nx", "ny", "nz"],
            *(f"f_dc_{i}" for i in range(3)),
            *(f"f_rest_{i}" for i in range(45 if degree else 0)),
            "opacity",
            *(f"scale_{i}" for i in range(3)),
            *(f"rot_{i}" for i in range(4)),
        ]
        groups = [means, np.zeros_like(means), a["sh"][:, 0, :]]
        if degree:
            groups.append(a["sh"][:, 1:, :].transpose(0, 2, 1).reshape(len(means), 45))
        groups.extend([op[:, None], ls, a["quaternions_wxyz"]])
        matrix = np.ascontiguousarray(np.concatenate(groups, axis=1), dtype="<f4")
        fmt = "binary_little_endian" if degree else "ascii"
        header = "\n".join(
            [
                "ply",
                "format " + fmt + " 1.0",
                "comment Original Evidence Lens helical bead study",
                f"element vertex {len(means)}",
                *["property float " + n for n in columns],
                "end_header",
                "",
            ]
        ).encode()
        with (out / f"graphdeco-sh{degree}.ply").open("wb") as f:
            f.write(header)
            if degree:
                f.write(matrix.tobytes())
            else:
                np.savetxt(f, matrix, fmt="%.9g")


def build(out):
    cams, poses = camera_rig()
    arr = native_values()
    write(out / "native-cameras.json", cams)
    np.savez(out / "native.npz", **arr)
    write_colmap(out / "colmap", poses)
    save_assets(out, arr)
    for n in IDS[:2]:
        source_picture(cams[n], arr, out / "images" / poses[n][3])
    centers = [np.asarray(cams[n]["c2w"])[:3, 3] for n in IDS[:2]]
    plan = {
        "schema": "el-asset-view-plan-1",
        "input_domain": "Original helical bead study, analytic surface photographs and Gaussian approximations",
        "sources": [{"id": n, "image_id": int(n)} for n in IDS[:2]],
        "targets": [{"id": n, "image_id": int(n)} for n in IDS[2:]],
        "primary_source_id": "41",
        "endpoint_source_ids": ["41", "5"],
        "metric": {
            "kind": "surveyed_source_baseline",
            "source": "Known designed metric geometry of the original camera rig",
            "meters_per_unit": 0.01,
            "source_ids": ["41", "5"],
            "distance_m": float(np.linalg.norm(centers[0] - centers[1])),
            "tolerance_m": 1e-8,
        },
    }
    write(out / "plan.json", plan)
    files = {
        p.relative_to(out).as_posix(): {"sha256": sha(p.read_bytes()), "bytes": p.stat().st_size}
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }
    write(
        out / "fixture.json",
        {
            "schema": "el-b2-original-asset-1",
            "run_id": os.getenv("EL_RUN_ID"),
            "sources": "Original analytic ray intersections with independently shaded spheres of the same bead study",
            "scene": "original-helical-beads-v1",
            "files": files,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    build(p.parse_args().out)
