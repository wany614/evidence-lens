"""Render a SHARP PLY at seven fixed cameras using gsplat on the Windows worker."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ._worker import digest, measured, read, require_cuda, require_sharp, run_id, runtime_info, write

VIEWS = ("source", "v01", "v02", "v03", "v04", "v05", "v06")


def render_geometry(gs, view, src_c2w, mode, eps, device):
    import torch
    from gsplat import rasterization

    # SHARP writes Gaussians in the source camera's metric right/down/forward frame.
    world_to_target = np.linalg.inv(np.asarray(view["c2w"], dtype=np.float64))
    relative = world_to_target @ src_c2w
    k = np.array(view["K"], dtype=np.float32)
    # gsplat v1.5.3 samples at (column + .5, row + .5).
    k[:2, 2] += 0.5
    colors, alpha, _ = rasterization(
        means=gs.mean_vectors[0], quats=gs.quaternions[0], scales=gs.singular_values[0],
        opacities=gs.opacities.reshape(-1), colors=gs.colors[0],
        viewmats=torch.as_tensor(relative[None], device=device, dtype=torch.float32),
        Ks=torch.as_tensor(k[None], device=device), width=view["width"], height=view["height"],
        near_plane=view.get("clip_m", [0.05, 30.0])[0],
        far_plane=view.get("clip_m", [0.05, 30.0])[1],
        render_mode=mode, rasterize_mode="classic", packed=False, eps2d=eps)
    return colors[0], alpha[0, ..., 0]


def depth_probe(cfg, device):
    import torch
    from sharp.utils.gaussians import Gaussians3D

    def tensor(x):
        return torch.tensor(x, device=device, dtype=torch.float32)
    # Two coplanar splats at integer pixel centres, with different opacities.
    centres = [[0., 0., 2.], [1., -0.5, 2.]]
    gs = Gaussians3D(tensor([centres]), tensor([[[0.04]*3]*2]),
                     tensor([[[1., 0., 0., 0.]]*2]), tensor([[[0.25, 0.5, 0.75]]*2]),
                     tensor([[0.4, 0.75]]))
    src = np.eye(4)
    view = {"c2w": src.tolist(), "K": [[24, 0, 16], [0, 24, 16], [0, 0, 1]],
            "height": 33, "width": 33}
    data, alpha = render_geometry(gs, view, src, "RGB+ED", cfg["eps2d"], device)
    acc, _ = render_geometry(gs, view, src, "RGB+D", cfg["eps2d"], device)
    samples = []
    for y, x, opacity in [(16, 16, 0.4), (10, 28, 0.75)]:
        samples.append({"xy": [x,y], "z_expected_m": 2.0, "opacity": opacity,
                        "alpha": float(alpha[y,x]), "ED": float(data[y,x,3]),
                        "D": float(acc[y,x,3]), "red": float(data[y,x,0])})
    translated = dict(view)
    target = np.eye(4)
    target[0, 3] = 0.25
    translated["c2w"] = target.tolist()
    moved, a = render_geometry(gs, translated, src, "RGB+ED", cfg["eps2d"], device)
    moved_alpha = float(a[16,13])
    alpha_tol, z_tol = cfg["alpha_probe_abs_tolerance"], cfg["depth_probe_abs_tolerance_m"]
    return {"off_axis_plane_passed": all(abs(t["ED"]-2.) <= z_tol for t in samples),
            "alpha_normalization_passed": all(abs(t["alpha"]-t["opacity"]) <= alpha_tol
                                              and abs(t["D"]-2*t["alpha"]) <= z_tol for t in samples),
            "translation_passed": abs(moved_alpha-.4) <= alpha_tol and abs(float(moved[16,13,3])-2.) <= z_tol,
            "linear_blend_passed": all(abs(t["red"]-0.25*t["opacity"]) <= cfg["color_probe_abs_tolerance"] for t in samples),
            "samples": samples, "translated_probe_alpha": moved_alpha,
            "empty_pixel_alpha": float(alpha[0,0]), "settings": cfg}


def render(config, input_path, camera_path, ply, generation_meta, out):
    cfg = read(config)
    rid = run_id()
    if cfg["renderer"]["mode"] != "RGB+ED" or cfg["renderer"]["background"] != "black":
        raise ValueError("Calibration applies to RGB+ED with a black background.")
    generation = read(generation_meta)
    camera_hash, input_hash = digest(camera_path), digest(input_path)
    if (generation["camera_sha256"], generation["input_sha256"], generation["ply_sha256"]) != (camera_hash, input_hash, digest(ply)):
        raise ValueError("PLY, cameras or exact input differ from the generating run.")
    require_sharp(cfg["generator"]["commit"])
    device = require_cuda(cfg)
    from sharp.utils.color_space import linearRGB2sRGB
    from sharp.utils.gaussians import load_ply

    cameras = read(camera_path)
    if cameras.get("axis_convention") != "OpenCV x right, y down, z forward":
        raise ValueError("Camera-to-world matrices must use the calibrated OpenCV axes.")
    views = cameras["views"]
    if (cameras["schema"], cameras["unit"], cameras["pixel_center"], set(views)) != (
            "el-camera-1", "m", "integer", set(VIEWS)):
        raise ValueError("Unsupported camera file.")
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    with measured("calibration", stats, device):
        probe = depth_probe(cfg["renderer"], device)
    write(out / "depth-probe.json", probe)
    if not (probe["translation_passed"] and probe["linear_blend_passed"] and probe["empty_pixel_alpha"] == 0.):
        raise ValueError("The camera or colour probe failed. See depth-probe.json.")
    verified_depth = probe["off_axis_plane_passed"] and probe["alpha_normalization_passed"]
    with measured("load_ply", stats, device):
        gs, meta = load_ply(ply)
        if meta.color_space != "linearRGB" or tuple(meta.resolution_px) != (
                views["source"]["width"], views["source"]["height"]):
            raise ValueError("The predicted PLY has incompatible colour or source size.")
        gs = gs.to(device)
    index = {"schema": "el-prediction-1", "producer_run_id": rid, "input_sha256": input_hash,
             "camera_sha256": camera_hash, "generation_run_id": generation["producer_run_id"],
             "ply_sha256": generation["ply_sha256"], "rgb_encoding": "display_srgb_0_1",
             "depth_semantics": "expected_camera_z_m", "alpha_semantics": "accumulated_opacity",
             "depth_validation": probe, "depth_validation_receipt_sha256": digest(out/"depth-probe.json"),
             "runtime": runtime_info(device), "render_settings": cfg["renderer"],
             "array_layout": {"rgb": "H,W,RGB FP32 display sRGB", "z": "H,W FP32 metres along target +Z; NaN at zero alpha or unverified semantics",
                              "alpha": "H,W FP32 dimensionless accumulated opacity"},
             "views": {}, "phases": stats}
    with Image.open(input_path) as image:
        input_rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.
    try:
        for vid in VIEWS:
            with measured("view_"+vid, stats, device):
                colors, opacity = render_geometry(gs, views[vid], np.array(views["source"]["c2w"]),
                                                  cfg["renderer"]["mode"], cfg["renderer"]["eps2d"], device)
                rgb = linearRGB2sRGB(colors[..., :3]).clamp(0, 1).cpu().numpy().astype(np.float32)
                z = colors[..., 3].cpu().numpy().astype(np.float32)
                alpha = opacity.cpu().numpy().astype(np.float32)
                z[(alpha <= 0) | ~np.isfinite(alpha) | (z <= 0)] = np.nan
                if not verified_depth:
                    z.fill(np.nan)
                dst = out / f"{vid}.npz"
                np.savez_compressed(dst, rgb=rgb, z=z, alpha=alpha)
                # Non-finite RGB remains present in the measurement array.
                Image.fromarray(np.rint(255*np.nan_to_num(rgb)).astype("uint8")).save(out/f"{vid}.png")
                index["views"][vid] = {"sha256": digest(dst), "bytes": dst.stat().st_size,
                                       "nonfinite_rgb_pixels": int((~np.isfinite(rgb).all(-1)).sum())}
                if vid == "source":
                    diff = np.abs(rgb-input_rgb).mean(-1)
                    finite = np.isfinite(diff)
                    index["source_rgb_check"] = {"mae": float(diff[finite].mean()) if finite.any() else None,
                                                "finite_pixels": int(finite.sum()), "pixels": int(diff.size)}
    finally:
        write(out/"prediction.json", index)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "input", "cameras", "ply", "generation", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    render(a.config, a.input, a.cameras, a.ply, a.generation, a.out)


if __name__ == "__main__":
    main()
