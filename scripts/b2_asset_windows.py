"""CUDA 1.5.3 readback of the original metric scene against six independently parsed asset paths."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image, ImageDraw
from evidence_lens_bundle.common import json_bytes, require, sha, write
from evidence_lens_asset.render import device, render

CASES = {
    "graphdeco-sh0": ("graphdeco-v1", "graphdeco-sh0.ply"),
    "graphdeco-sh3": ("graphdeco-v1", "graphdeco-sh3.ply"),
    "project-npz": ("evidence-lens-npz-v1", "project.npz"),
}


def basis(xyz, degree):
    """Real orthonormal SH in (l,m=-l..+l) order, via associated Legendre recurrence."""
    x, y, z = xyz.T
    phi = np.arctan2(y, x)
    sin = np.sqrt(np.maximum(0, 1 - z * z))
    columns = []
    for l in range(degree + 1):
        for m in range(-l, l + 1):
            k = abs(m)
            p = np.ones_like(z)
            for j in range(1, k + 1):
                p *= -(2 * j - 1) * sin
            if l > k:
                older = p
                p = z * (2 * k + 1) * p
                for j in range(k + 2, l + 1):
                    older, p = p, ((2 * j - 1) * z * p - (j + k - 1) * older) / (j - k)
            norm = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - k) / math.factorial(l + k))
            if m:
                trig = (np.sin if m < 0 else np.cos)(k * phi)
                columns.append(math.sqrt(2) * norm * p * trig)
            else:
                columns.append(norm * p)
    return np.stack(columns, axis=1)


def native(torch, a, cam, case):
    from gsplat import rasterization

    t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device="cuda")
    k = np.asarray(cam["K"], np.float64).copy()
    k[0, 2] += 0.5
    k[1, 2] += 0.5
    pose = np.asarray(cam["c2w"], np.float64)
    if case == "project-npz":
        colors = a["linear_color"]
    else:
        degree = 3 if case == "graphdeco-sh3" else 0
        delta = a["centers_m"].astype(np.float64) - pose[:3, 3]
        direction = delta / np.linalg.norm(delta, axis=1)[:, None]
        h = basis(direction, degree)
        colors = np.maximum(0.0, 0.5 + np.einsum("nk,nkc->nc", h, a["sh"][:, : h.shape[1], :]))
    image, alpha, _ = rasterization(
        means=t(a["centers_m"]),
        quats=t(a["quaternions_wxyz"]),
        scales=t(a["radii_m"]),
        opacities=t(a["alpha"]),
        colors=t(colors),
        viewmats=t(np.linalg.inv(pose)[None]),
        Ks=t(k[None]),
        width=cam["width"],
        height=cam["height"],
        near_plane=0.01,
        far_plane=100.0,
        eps2d=0.3,
        packed=False,
        rasterize_mode="classic",
        sh_degree=None,
        render_mode="RGB+ED",
        camera_model="pinhole",
    )
    raw = image[0].cpu().numpy()
    alpha = alpha[0, ..., 0].cpu().numpy()
    rgb = np.maximum(0.0, raw[:, :, :3])
    if case == "project-npz":
        rgb = np.where(
            rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.maximum(rgb, 0.0031308) ** np.float32(1 / 2.4) - 0.055
        )
    z = raw[:, :, 3].copy()
    z[alpha <= 0] = np.nan
    return {"rgb": np.clip(rgb, 0, 1), "alpha": alpha, "z": z}


def compare(a, b, tol):
    rows = {}
    for k in ("rgb", "z", "alpha"):
        x, y = a[k], b[k]
        require(x.shape == y.shape and x.dtype == y.dtype == np.float32, "native frame layout changed")
        finite = np.isfinite(x) & np.isfinite(y)
        diff = np.abs(x[finite].astype(np.float64) - y[finite])
        mx = float(diff.max(initial=0))
        support_diff = int(np.count_nonzero(np.isfinite(x) != np.isfinite(y)))
        rows[k] = {
            "n": int(finite.sum()),
            "total": int(x.size),
            "support_difference": support_diff,
            "max_abs": mx,
            "mean_abs": float(np.mean(diff)) if diff.size else None,
            "tolerance": tol[k],
            "passed": bool(diff.size and not support_diff and mx <= tol[k]),
        }
    return rows


def panel(a, b, out):
    def image(x):
        return Image.fromarray(np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8))

    def col(x):
        return np.repeat(x[:, :, None], 3, axis=2)

    frames = [
        ("Native RGB", a["rgb"]),
        ("Imported RGB", b["rgb"]),
        ("RGB absolute delta x1000", np.abs(a["rgb"] - b["rgb"]) * 1000),
        ("Native ED, 0..4 m", col(np.nan_to_num(a["z"]) / 4)),
        ("Imported ED, 0..4 m", col(np.nan_to_num(b["z"]) / 4)),
        ("Imported opacity, 0..1", col(b["alpha"])),
    ]
    page = Image.new("RGB", (768, 568), "#181c24")
    g = ImageDraw.Draw(page)
    for i, (label, x) in enumerate(frames):
        u, v = (i % 3) * 256, (i // 3) * 284
        page.paste(image(x), (u, v + 25))
        g.text((u + 8, v + 6), label, fill="#ffffff")
    page.save(out)


def execute(fixture, protocol, out):
    expected = json_bytes((fixture / "fixture.json").read_bytes())
    for n, s in expected["files"].items():
        raw = (fixture / n).read_bytes()
        require(sha(raw) == s["sha256"] and len(raw) == s["bytes"], "changed fixture: " + n)
    p = json_bytes(protocol.read_bytes())
    require(p["schema"] == "el-asset-acceptance-protocol-1", "wrong comparison protocol")
    tol = p["comparisons"]["absolute_tolerance"]
    tols = {"rgb": tol["rgb"], "alpha": tol["alpha"], "z": tol["expected_camera_Z_m"]}
    torch = device()
    cams = json_bytes((fixture / "native-cameras.json").read_bytes())
    rows = []
    timings = []
    with np.load(fixture / "native.npz", allow_pickle=False) as z:
        native_values = dict(z)
    torch.cuda.reset_peak_memory_stats()
    for case, (profile, name) in CASES.items():
        ref = out / "native" / case
        ref.mkdir(exist_ok=True, parents=True)
        references = {}
        start = perf_counter()
        with torch.no_grad():
            for v in p["comparisons"]["views"]:
                a = native(torch, native_values, cams[v], case)
                references[v] = a
                np.savez_compressed(ref / (v + ".npz"), **a)
        torch.cuda.synchronize()
        timings.append({"case": case, "stage": "render_native_reference", "seconds": perf_counter() - start})
        for fmt in ("text", "binary"):
            dest = out / "bundles" / case / fmt
            start = perf_counter()
            validation = render(
                fixture / "colmap" / fmt,
                fixture / "images",
                fixture / name,
                fixture / "plan.json",
                fmt,
                profile,
                dest,
            )
            torch.cuda.synchronize()
            timings.append(
                {"case": case, "stage": "inspect_render_validate_" + fmt, "seconds": perf_counter() - start}
            )
            c = json_bytes((dest / "cameras.json").read_bytes())
            producer = json_bytes((dest / "producer.json").read_bytes())
            for v in p["comparisons"]["views"]:
                file = producer["frames"][v]["file"]
                raw = (dest / file["path"]).read_bytes()
                require(sha(raw) == file["sha256"], "rendered bytes differ")
                with np.load(dest / file["path"], allow_pickle=False) as z:
                    imported = dict(z)
                numeric = compare(references[v], imported, tols)
                geometry = float(
                    max(
                        np.max(np.abs(np.asarray(c["views"][v][key]) - np.asarray(cams[v][key])))
                        for key in ("K", "c2w")
                    )
                )
                rows.append(
                    {
                        "asset": case,
                        "colmap": fmt,
                        "view_id": v,
                        "native_sha256": sha((ref / (v + ".npz")).read_bytes()),
                        "import_sha256": sha(raw),
                        "bundle_sha256": validation["bundle_sha256"],
                        "camera_max_abs": geometry,
                        "channels": numeric,
                        "passed": geometry <= tol["camera_K_and_pose"]
                        and all(s["passed"] for s in numeric.values()),
                    }
                )
                if v == "70" and fmt == "binary":
                    panel(references[v], imported, out / (case + "-comparison.png"))
    receipt = {
        "schema": "el-b2-gaussian-parity-1",
        "run_id": os.getenv("EL_RUN_ID"),
        "fixture_sha256": sha((fixture / "fixture.json").read_bytes()),
        "protocol_sha256": sha(protocol.read_bytes()),
        "source_code_sha256": sha(Path(__file__).read_bytes()),
        "rows": rows,
        "timings": timings,
        "device": torch.cuda.get_device_name(0),
        "gsplat": "1.5.3",
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "passed": bool(len(rows) == 30 and all(r["passed"] for r in rows)),
    }
    write(out / "parity.json", receipt)
    require(receipt["passed"], "one or more registered numerical comparisons failed; see parity.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("fixture", "protocol", "out"):
        parser.add_argument("--" + n, type=Path, required=True)
    a = parser.parse_args()
    execute(a.fixture, a.protocol, a.out)
