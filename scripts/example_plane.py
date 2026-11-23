"""Render the project's redistributable example from a single authored image and cameras on CPU."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from evidence_lens.scoring_io import Camera, ReadAudit, VIEWS, file_hash, load_cameras, load_source_png, write_json


def composite(texture, opacity):
    """One planar sample composited over black in linear light, then returned in display sRGB."""
    linear = np.where(texture <= .04045, texture / 12.92, ((texture + .055) / 1.055) ** 2.4)
    result = linear * opacity
    return np.where(result <= .0031308, 12.92 * result, 1.055 * np.maximum(result, 0) ** (1 / 2.4) - .055)


def project(texture, source, target, plane_z, opacity=1.):
    """Back-project target pixel-centre rays to the plane z_source=plane_z."""
    yy, xx = np.indices(target.shape, dtype=np.float64)
    local = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ np.linalg.inv(target.K).T
    relative = np.linalg.solve(source.c2w, target.c2w)
    rays = local @ relative[:3, :3].T
    origin = relative[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (plane_z - origin[2]) / rays[..., 2]
        hit = (origin + depth[..., None] * rays) @ source.K.T
        uv = hit[..., :2] / hit[..., 2:3]
    ok = (np.isfinite(depth) & (depth > 0) & np.isfinite(uv).all(axis=-1)
          & (uv[..., 0] >= -.5) & (uv[..., 0] < source.width - .5)
          & (uv[..., 1] >= -.5) & (uv[..., 1] < source.height - .5))
    rgb = np.zeros((*target.shape, 3), dtype=np.float32)
    x, y = np.floor(uv[ok] + .5).astype(int).T
    rgb[ok] = np.clip(composite(texture[y, x], opacity), 0, 1)
    alpha = np.where(ok, opacity, 0).astype(np.float32)
    expected_z = np.where(alpha > 0, depth, np.nan).astype(np.float32)
    return {"rgb": rgb, "z": expected_z, "alpha": alpha}


def check_depth():
    K = np.array([[9., 0., 7.], [0., 9., 7.], [0., 0., 1.]])
    source = Camera(15, 15, K, np.eye(4))
    pose = np.eye(4)
    pose[0, 3], pose[2, 3] = .6, .25
    target = Camera(15, 15, K, pose)
    rgb = np.ones((15, 15, 3), dtype=np.float32)
    rows = []
    for opacity in (.4, .75):
        result = project(rgb, source, target, 2., opacity)
        for y, x in ((7, 7), (4, 9)):
            rows.append({"opacity": opacity, "measured": float(result["alpha"][y,x]),
                         "expected_z_m": 1.75, "depth_m": float(result["z"][y,x]),
                         "display_red": float(result["rgb"][y,x,0])})
    empty = result["alpha"] == 0
    depth_ok = all(abs(r["expected_z_m"]-r["depth_m"]) < 1e-6 for r in rows)
    opacity_ok = all(abs(r["measured"]-r["opacity"]) < 1e-6 for r in rows)
    if not (depth_ok and opacity_ok and empty.any() and np.isnan(result["z"][empty]).all()):
        raise ValueError("single-plane depth, opacity or empty pixel calibration failed")
    return {"off_axis_plane_passed": bool(depth_ok), "alpha_normalization_passed": bool(opacity_ok),
            "empty_depth_proved": True, "plane_in_source_camera_m": 2.0, "probes": rows}


def resized(texture, layout, stride):
    """Area-average sRGB8 samples; transform pixel edges before centre conversion."""
    if stride < 1 or any(s % stride for s in texture.shape[:2]):
        raise ValueError("example stride must divide the two source dimensions")
    h, w = texture.shape[:2]
    arr = texture.reshape(h//stride, stride, w//stride, stride, 3)
    small = np.rint(arr.mean(axis=(1, 3), dtype=np.float64)).astype(np.uint8)
    views = {}
    for view, v in layout["views"].items():
        if v["width"] % stride or v["height"] % stride:
            raise ValueError(f"{view} dimensions do not divide by the example stride")
        edge = np.diag([1/stride, 1/stride, 1.]) @ v["K_edge"]
        K = np.array([[1, 0, -.5], [0, 1, -.5], [0, 0, 1.]]) @ edge
        views[view] = dict(v, width=v["width"]//stride, height=v["height"]//stride,
                           K_edge=edge.tolist(), K=K.tolist())
    return small, dict(layout, views=views)


def produce(source_image, cameras, out, plane_z, stride, producer):
    if not np.isfinite(plane_z) or plane_z <= 0:
        raise ValueError("plane distance must be finite and positive")
    audit = ReadAudit()
    base = load_cameras(cameras, audit)
    load_source_png(source_image, base["source"], audit)
    with Image.open(source_image) as im:
        texture, layout = resized(np.asarray(im.convert("RGB")), json.loads(cameras.read_bytes()), stride)
    exp = out / "example"
    pred = exp / "predictions"
    pred.mkdir(parents=True)
    Image.fromarray(texture).save(exp/"input.png")
    write_json(exp/"cameras.json", layout)
    cooked = load_cameras(exp/"cameras.json", ReadAudit())
    normalized = texture.astype(np.float32)/np.float32(255)
    probe = check_depth()
    write_json(pred/"depth-probe.json", probe)
    views = {}
    for view in VIEWS:
        data = project(normalized, cooked["source"], cooked[view], plane_z)
        file = pred/(view+".npz")
        np.savez_compressed(file, **data)
        views[view] = {"sha256": file_hash(file), "bytes": file.stat().st_size}
        if view == "v01":
            Image.fromarray(np.rint(255*data["rgb"]).astype(np.uint8)).save(exp/"v01-prediction.png")
        if view == "source" and np.max(np.abs(data["rgb"]-normalized)) > 1e-6:
            raise ValueError("the rendered source differs from its supplied texture")
    write_json(pred/"prediction.json", {"schema":"el-prediction-1", "producer_run_id":producer, "views":views,
               "adapter":"evidence-lens-textured-plane-1", "input_sha256":file_hash(exp/"input.png"),
               "camera_sha256":file_hash(exp/"cameras.json"), "rgb_encoding":"display_srgb_0_1",
               "alpha_semantics":"accumulated_opacity", "depth_semantics":"expected_camera_z_m", "failed_views":{},
               "depth_validation":probe, "depth_validation_receipt_sha256":file_hash(pred/"depth-probe.json")})
    write_json(exp/"provenance.json", {"schema":"el-original-example-1", "run_id":producer, "adapter":"evidence-lens-textured-plane-1",
               "source_png_sha256":file_hash(source_image), "source_cameras_sha256":file_hash(cameras), "source_stride":stride,
               "hypothesized_plane_m":plane_z, "predictions":"Inverse pinhole ray intersection with one opaque plane textured by the sole input.",
               "purpose":"Redistributable inference and image-coordinate demonstration. The formal experiment evaluates SHARP."})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("input","cameras","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--plane-z-m",type=float,default=2.)
    p.add_argument("--stride",type=int,default=4)
    a=p.parse_args()
    rid=os.environ.get("EL_RUN_ID")
    if not rid:
        p.error("a scientific run ID is required from the recorder")
    produce(a.input,a.cameras,a.output,a.plane_z_m,a.stride,rid)


if __name__=="__main__":
    main()
