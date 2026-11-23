"""Generate metric Gaussians from one RGB PNG with the pinned upstream predictor."""

import argparse
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ._worker import digest, measured, read, require_cuda, require_sharp, run_id, runtime_info, write


def generate(config, input_path, camera_path, weights, out):
    cfg = read(config)
    identity = run_id()
    expected = cfg["generator"]["checkpoint_sha256"]
    if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("Freeze the official checkpoint SHA-256 before this run.")
    if weights.stat().st_size != cfg["generator"]["checkpoint_bytes"] or digest(weights) != expected:
        raise ValueError("Checkpoint byte count or SHA-256 differs from frozen configuration.")

    commit = require_sharp(cfg["generator"]["commit"])
    device = require_cuda(cfg)
    import torch
    from sharp.cli.predict import predict_image
    from sharp.models import PredictorParams, create_predictor
    from sharp.utils.gaussians import save_ply

    cameras = read(camera_path)
    if (cameras["schema"], cameras["unit"], cameras["pixel_center"]) != ("el-camera-1", "m", "integer"):
        raise ValueError("Unsupported cameras.")
    source = cameras["views"]["source"]
    with Image.open(input_path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (source["width"], source["height"]):
            raise ValueError("Expected one RGB PNG at the actual source-camera resolution.")
        rgb = np.array(image, dtype=np.uint8)
    k = np.asarray(source["K"], dtype=float)
    f = float(k[0, 0])
    h, w = rgb.shape[:2]
    if not np.isfinite(f) or f <= 0 or not np.allclose(k, [[f, 0, (w-1)/2], [0, f, (h-1)/2], [0, 0, 1]], atol=1e-5, rtol=0):
        raise ValueError("SHARP single-focal-length input requires centred square pixels.")
    if cfg["generator"]["internal_size"] != [1536, 1536]:
        raise ValueError("The pinned SHARP predictor uses a fixed 1536-square resize.")
    stats = {}
    meta = {"schema": "el-generation-1", "producer_run_id": identity, "generator_commit": commit,
            "checkpoint_sha256": expected, "input_sha256": digest(input_path),
            "camera_sha256": digest(camera_path), "source_size_wh": [w, h], "focal_length_px": f,
            "network_input_wh": [1536, 1536], "resize": "bilinear align_corners=True, no crop or padding",
            "channels": "RGB uint8 display sRGB / 255", "native_frame": "source CV camera, metres",
            "ply_color": "sRGB SH degree 0; upstream reader converts to linear RGB for blending",
            "runtime": runtime_info(device), "phases": stats}
    out.mkdir(parents=True, exist_ok=True)
    try:
        with measured("load_model", stats, device):
            params = torch.load(weights, map_location="cpu", weights_only=True)
            model = create_predictor(PredictorParams())
            model.load_state_dict(params)
            del params
            model.eval().to(device)
        # Time the model separately without replacing its published preprocessing or unprojection.
        stamps = []
        def stamp(*args):
            torch.cuda.synchronize(device)
            stamps.append(time.perf_counter())
        before = model.register_forward_pre_hook(stamp)
        after = model.register_forward_hook(stamp)
        try:
            with measured("preprocess_forward_unproject", stats, device), torch.inference_mode():
                t0 = time.perf_counter()
                gs = predict_image(model, rgb, f, device)
                torch.cuda.synchronize(device)
                t1 = time.perf_counter()
        finally:
            before.remove()
            after.remove()
        if len(stamps) == 2:
            meta["preprocess_seconds"], meta["forward_seconds"], meta["unproject_seconds"] = (
                stamps[0]-t0, stamps[1]-stamps[0], t1-stamps[1])
        if not all(torch.isfinite(v).all().item() for v in gs):
            raise ValueError("Non-finite model output.")
        with measured("export_ply", stats, device):
            save_ply(gs, f, (h, w), out / "scene.ply")
        meta["gaussian_count"] = gs.mean_vectors.shape[1]
        meta["ply_sha256"] = digest(out / "scene.ply")
        meta["ply_bytes"] = (out / "scene.ply").stat().st_size
    finally:
        write(out / "generation.json", meta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "input", "cameras", "weights", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    generate(a.config, a.input, a.cameras, a.weights, a.out)


if __name__ == "__main__":
    main()
