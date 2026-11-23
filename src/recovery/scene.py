"""FP32 pinhole projection, degree-zero color and the source photometric objective."""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
import platform

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization

from evidence_lens._bundle import encoded, load
from evidence_lens._verify import require
from recovery.contract import PARAMS, WIDTHS


def setup():
    require(importlib.metadata.version("gsplat") == "1.5.3" and torch.cuda.device_count() == 1,
            "Use gsplat 1.5.3 and one visible CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_device(0)
    # device_count() can use NVML without initializing PyTorch's allocator.
    # Create the context and allocator before asking it to reset its counters.
    torch.empty(1, dtype=torch.float32, device="cuda:0")
    torch.cuda.reset_peak_memory_stats(0)
    return torch.device("cuda:0")


def runtime():
    return {"python": platform.python_version(), "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "gsplat": importlib.metadata.version("gsplat"), "device": torch.cuda.get_device_name(0),
            "torch_allocated_peak_bytes": torch.cuda.max_memory_allocated(0),
            "torch_reserved_peak_bytes": torch.cuda.max_memory_reserved(0)}


def cameras(path, ids, device):
    doc = load(path)
    require(doc["schema"] == "el-camera-1" and doc["pixel_center"] == "integer" and doc["unit"] == "m"
            and doc["axis_convention"] == "OpenCV x right, y down, z forward" and set(doc["views"]) == set(ids),
            "Unsupported camera contract or view roles")
    views = {}
    for key in ids:
        cam = doc["views"][key]
        width, height = cam["width"], cam["height"]
        k = np.asarray(cam["K"], dtype=np.float64)
        pose = np.asarray(cam["c2w"], dtype=np.float64)
        near, far = cam["clip_m"]
        require(type(width) is int and width > 0 and type(height) is int and height > 0 and
                k.shape == (3, 3) and pose.shape == (4, 4) and np.isfinite(k).all() and np.isfinite(pose).all() and
                k[0, 0] > 0 and k[1, 1] > 0 and np.allclose(k[2], [0, 0, 1]) and
                abs(k[0, 1]) < 1e-6 and abs(k[1, 0]) < 1e-6 and np.allclose(pose[3], [0, 0, 0, 1]) and
                np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5) and
                abs(np.linalg.det(pose[:3, :3]) - 1) < 1e-5 and np.isfinite([near, far]).all() and 0 < near < far,
                "Invalid pinhole camera " + key)
        k[:2, 2] += 0.5
        views[key] = (torch.tensor(k[None], device=device, dtype=torch.float32),
                      torch.tensor(np.linalg.inv(pose)[None], device=device, dtype=torch.float32),
                      width, height, near, far)
    centers = [np.asarray(doc["views"][s]["c2w"], dtype=np.float64)[:3, 3] for s in ("s00", "s03")]
    baseline = float(np.linalg.norm(centers[0] - centers[1]))
    require(math.isfinite(baseline) and baseline > 0, "Shared endpoint baseline must be positive in metres")
    calibration = {}
    for key in ids:
        cam = doc["views"][key]
        physical = {name: np.asarray(cam[name], dtype=np.float64).tolist() for name in ("K", "c2w", "clip_m")}
        physical.update(width=cam["width"], height=cam["height"])
        calibration[key] = hashlib.sha256(encoded(physical)).hexdigest()
    frame = {s: np.asarray(doc["views"][s]["c2w"], dtype=np.float64).tolist() for s in ("s00", "s03")}
    return views, baseline, calibration, hashlib.sha256(encoded(frame)).hexdigest()


def init_parameters(path, cap, device):
    with np.load(path, allow_pickle=False) as data:
        require(len(data.files) == len(PARAMS) and set(data.files) == set(PARAMS), "Initialization array contract")
        tensors = {}
        n = len(data["means"])
        require(1 <= n <= cap, "Initialization exceeds the registered point budget")
        for name, internal in PARAMS.items():
            v = data[name]
            shape = (n,) if WIDTHS[name] is None else (n, WIDTHS[name])
            require(v.dtype == np.dtype("<f4") and v.shape == shape and np.isfinite(v).all(),
                    "Invalid FP32 init field " + name)
            tensors[internal] = torch.nn.Parameter(torch.as_tensor(v, device=device))
    require(bool((torch.linalg.vector_norm(tensors["quats"], dim=1) > 1e-9).all()), "Zero initialization quaternion")
    return tensors


def source_image(path, view, device):
    with Image.open(path) as image:
        require(image.format == "PNG" and image.mode == "RGB" and image.size == (view[2], view[3]),
                "Source must be an RGB PNG at calibrated resolution")
        return torch.from_numpy(np.array(image, dtype=np.float32) / 255).to(device)


def draw(params, view, eps, depth=False):
    k, w2c, width, height, near, far = view
    image, alpha, info = rasterization(
        means=params["means"], quats=F.normalize(params["quats"], dim=-1), scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(), colors=params["colors"].sigmoid(), viewmats=w2c, Ks=k,
        width=width, height=height, near_plane=near, far_plane=far, eps2d=eps, packed=False,
        backgrounds=k.new_zeros((1, 3)), camera_model="pinhole", rasterize_mode="classic", sh_degree=None,
        render_mode="RGB+ED" if depth else "RGB")
    linear = image[0, ..., :3].clamp_min(0)
    display = torch.where(linear <= .0031308, 12.92 * linear,
                          1.055 * linear.clamp_min(.0031308).pow(1 / 2.4) - .055).clamp(0, 1)
    return display, alpha[0, ..., 0], image[0, ..., 3] if depth else None, info


def photometric(prediction, reference, weights):
    l1 = F.l1_loss(prediction, reference)
    if weights["one_minus_ssim"] == 0:
        return l1, l1, None
    require(min(prediction.shape[:2]) >= 11, "Full valid SSIM window needs at least 11 pixels")
    x, y = (a.permute(2, 0, 1).unsqueeze(0) for a in (prediction, reference))
    axis = torch.arange(-5, 6, device=x.device, dtype=x.dtype)
    w = torch.exp(-axis.square() / (2 * 1.5**2))
    w = w / w.sum()
    kernel = torch.outer(w, w).expand(3, 1, 11, 11).contiguous()
    # Valid 11x11 population moments on each display-sRGB channel. Data range is one.
    def mean(a):
        return F.conv2d(a, kernel, groups=3)
    mx, my = mean(x), mean(y)
    vx, vy = (mean(t.square()) - mu.square() for t, mu in ((x, mx), (y, my)))
    cov = mean(x * y) - mx * my
    ssim = (((2 * mx * my + .01**2) * (2 * cov + .03**2)) /
            ((mx.square() + my.square() + .01**2) * (vx + vy + .03**2))).mean()
    return weights["display_l1"] * l1 + weights["one_minus_ssim"] * (1 - ssim), l1, ssim
