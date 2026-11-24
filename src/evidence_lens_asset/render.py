"""Optional gsplat 1.5.3 CUDA rasterizer; it does not run or fit a risk model."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import time
import uuid
from pathlib import Path
import numpy as np
from evidence_lens.scoring_io import new_output
from evidence_lens_bundle.bundle import Bundle
from evidence_lens_bundle.common import (
    PROFILE,
    RELEASE_INDEX_SHA256,
    SEMANTICS,
    Unsupported,
    encoded,
    profile,
    require,
    row,
    sha,
    write,
)
from .prepare import inspect


def device():
    try:
        import torch

        version = importlib.metadata.version("gsplat")
    except (ImportError, importlib.metadata.PackageNotFoundError) as e:
        raise Unsupported("asset rendering requires CUDA PyTorch and gsplat 1.5.3") from e
    require(
        version == "1.5.3" and torch.cuda.is_available(),
        "asset rendering requires CUDA and gsplat 1.5.3",
        Unsupported,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def layer(torch, values, camera, mode="RGB+ED", direct_rgb=None):
    from gsplat import rasterization

    as_tensor = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device="cuda")
    k = np.array(camera["K"], dtype=np.float64)
    k[:2, 2] += 0.5
    w2c = np.linalg.inv(np.asarray(camera["c2w"], dtype=np.float64))
    a = values
    if direct_rgb is not None:
        colors = as_tensor(direct_rgb)
        degree = None
    elif "sh" in a:
        colors = as_tensor(a["sh"])
        degree = 3 if a["sh"].shape[1] == 16 else 0
    else:
        colors = as_tensor(a["color_logits"]).sigmoid()
        degree = None
    result, opacity, _ = rasterization(
        means=as_tensor(a["means"]),
        quats=torch.nn.functional.normalize(as_tensor(a["quats"]), dim=-1),
        scales=as_tensor(a["log_scales"]).exp(),
        opacities=as_tensor(a["opacity_logits"]).sigmoid(),
        colors=colors,
        viewmats=as_tensor(w2c[None]),
        Ks=as_tensor(k[None]),
        width=camera["width"],
        height=camera["height"],
        near_plane=0.01,
        far_plane=100.0,
        eps2d=0.3,
        packed=False,
        rasterize_mode="classic",
        camera_model="pinhole",
        render_mode=mode,
        sh_degree=degree,
        backgrounds=None,
    )
    rgb = result[0, ..., :3].clamp_min(0)
    if "color_logits" in a and direct_rgb is None:
        rgb = torch.where(
            rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb.clamp_min(0.0031308).pow(1 / 2.4) - 0.055
        )
    alpha = opacity[0, ..., 0].cpu().numpy()
    z = result[0, ..., 3].cpu().numpy().copy()
    z[alpha <= 0] = np.nan
    require(
        np.isfinite(alpha).all()
        and (alpha >= 0).all()
        and (alpha <= 1).all()
        and np.all(np.isfinite(z[alpha > 0]))
        and np.all(z[alpha > 0] > 0),
        "invalid renderer alpha/depth",
    )
    return {"rgb": rgb.clamp(0, 1).cpu().numpy(), "alpha": alpha, "z": z}


def depth_probe(torch):
    points = np.asarray([[0, 0, 2], [0, 0, 4], [0.5, 0.25, 2], [1, 0.5, 4]], np.float32)
    a = {
        "means": points,
        "quats": np.tile(np.array([1, 0, 0, 0], np.float32), (4, 1)),
        "log_scales": np.full((4, 3), np.log(0.025), np.float32),
        "opacity_logits": np.zeros(4, np.float32),
        "color_logits": np.zeros((4, 3), np.float32),
    }
    cam = {"K": [[64, 0, 32], [0, 64, 32], [0, 0, 1]], "c2w": np.eye(4), "height": 65, "width": 65}
    ed = layer(torch, a, cam)
    acc = layer(torch, a, cam, "RGB+D")
    rows = []
    for x, y in ((32, 32), (48, 40)):
        alpha = float(ed["alpha"][y, x])
        z = float(ed["z"][y, x])
        depth_sum = float(acc["z"][y, x])
        require(
            abs(alpha - 0.75) <= 1e-4 and abs(z - 8 / 3) <= 1e-4 and abs(z - depth_sum / alpha) <= 1e-4,
            "renderer failed off-axis metric expected camera-Z probe",
        )
        rows.append({"x": x, "y": y, "alpha": alpha, "expected_camera_z_m": z, "weighted_depth_m": depth_sum})
    return {
        "schema": "el-asset-cuda-depth-probe-1",
        "renderer": "gsplat 1.5.3 CUDA",
        "passed": True,
        "samples": rows,
    }


def render(model_dir, images_dir, gaussian_path, plan_path, encoding, gaussian_profile, output):
    started = time.perf_counter()
    stats, cam, photos, arr = inspect(
        model_dir, images_dir, gaussian_path, plan_path, encoding, gaussian_profile
    )
    torch = device()
    rid = os.getenv("EL_RUN_ID") or str(uuid.uuid4())
    with new_output(output) as dest, torch.no_grad():

        def save(name, raw):
            p = dest / name
            p.parent.mkdir(exist_ok=True, parents=True)
            p.write_bytes(raw)
            return row(name, raw)

        detail = save("asset-input.json", encoded(stats))
        c = save("cameras.json", encoded(cam))
        probe = save("depth-probe.json", encoded(depth_probe(torch)))
        sources = {name: save(f"sources/{i:03d}.png", raw) for i, (name, raw) in enumerate(photos.items())}
        frames = {}
        for i, v in enumerate(stats["source_order"] + stats["target_order"]):
            path = f"renders/{i:03d}.npz"
            (dest / path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / path, **layer(torch, arr, cam["views"][v]))
            frames[v] = {"status": "ready", "file": row(path, (dest / path).read_bytes())}
        torch.cuda.synchronize()
        identity = save(
            "renderer.json",
            encoded(
                {
                    "schema": "el-gaussian-asset-renderer-1",
                    "run_id": rid,
                    "asset_sha256": stats["asset_sha256"],
                    "inspection_sha256": detail["sha256"],
                    "cameras_sha256": c["sha256"],
                    "depth_probe_sha256": probe["sha256"],
                    "renderer": "gsplat",
                    "version": "1.5.3",
                    "device": torch.cuda.get_device_name(0),
                    "torch": str(torch.__version__),
                    "python": platform.python_version(),
                    "frames": frames,
                    "inspect_and_render_seconds": time.perf_counter() - started,
                    "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
                    "reserved_peak_bytes": torch.cuda.max_memory_reserved(),
                }
            ),
        )
        producer = {
            "schema": "el-render-evidence-1",
            "generator": {
                "run_id": rid,
                "receipt_sha256": detail["sha256"],
            },
            "renderer": {"run_id": rid, "receipt_sha256": identity["sha256"]},
            "checkpoint_sha256": stats["asset_sha256"],
            "source_rgb_sha256": {k: v["sha256"] for k, v in sources.items()},
            "source_order": stats["source_order"],
            "target_order": stats["target_order"],
            "camera_sha256": c["sha256"],
            "semantics": SEMANTICS,
            "depth_verified": True,
            "depth_probe_sha256": probe["sha256"],
            "frames": frames,
        }
        b = {
            k: stats[k]
            for k in (
                "input_domain",
                "condition",
                "source_order",
                "target_order",
                "primary_source_id",
                "endpoint_source_ids",
            )
        }
        b.update(
            schema="el-evidence-bundle-1",
            creation={
                "run_id": rid,
                "implementation": "evidence-lens versioned asset import / gsplat 1.5.3",
                "code_sha256": sha(Path(__file__).read_bytes()),
                "config_sha256": stats["plan_sha256"],
                "original_case_sha256": None,
            },
            camera=c,
            geometry={
                "prepared_camera_and_z_unit": "m",
                "meters_per_original_unit": stats["metric"]["meters_per_unit"],
                "metric_evidence": stats["metric"]["source"],
                "crop": "No crop",
                "resize": "Native 256 x 256",
                "undistortion": "COLMAP pinhole (distortion-free)",
                "scoring_resampling": "none",
            },
            sources=sources,
            producer=save("producer.json", encoded(producer)),
            risk_profile={
                "id": PROFILE,
                "sha256": sha(encoded(profile())),
                "release_index_sha256": RELEASE_INDEX_SHA256,
            },
        )
        write(dest / "bundle.json", b)
        result = Bundle(dest).validate()
    return result
