"""Analytically calibrated off-centre pinhole and opacity-normalized Z probe."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from evidence_lens._bundle import save
from evidence_lens._verify import require
from recovery.scene import cameras, draw, runtime, setup

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
out = p.parse_args().out
started = time.perf_counter()
device = setup()
k = np.array([[26., 0., 13.25], [0., 29., 14.75], [0., 0., 1.]])
cam = {"K": k.tolist(), "width": 33, "height": 33, "clip_m": [.05, 8.]}
other = np.eye(4)
other[0, 3] = 1.
doc = {"schema": "el-camera-1", "unit": "m", "axis_convention": "OpenCV x right, y down, z forward", "pixel_center": "integer",
       "views": {"s00": {**cam, "c2w": np.eye(4).tolist()}, "s03": {**cam, "c2w": other.tolist()}}}
path = out / "analytic-cameras.json"
save(path, doc)
views, scale, _, _ = cameras(path, ["s00", "s03"], device)
x, y = 22, 12
zs = np.array([.41, .91])
positions = np.linalg.solve(k, [x, y, 1.])[None] * zs[:, None]
op = np.array([.4, .65])
color = np.array([.2, .45, .62])
to = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
params = {"means": to(positions), "scales": to(np.log(np.full((2, 3), .0018))), "quats": to([[1, 0, 0, 0]] * 2),
          "opacities": to(np.log(op / (1 - op))), "colors": to(np.tile(np.log(color / (1 - color)), (2, 1)))}
with torch.no_grad():
    rgb, alpha, z, info = draw(params, views["s00"], .3, depth=True)
weights = np.array([op[0], (1 - op[0]) * op[1]])
expected_z = float(weights @ zs / weights.sum())
expected_linear = weights.sum() * color
expected_rgb = np.where(expected_linear <= .0031308, 12.92 * expected_linear, 1.055 * expected_linear ** (1 / 2.4) - .055)
errors = {"projection_edge_px": float(np.max(np.abs(info["means2d"].cpu().numpy()[0] - [x + .5, y + .5]))),
          "expected_camera_z_m": abs(float(z[y,x]) - expected_z), "alpha": abs(float(alpha[y,x]) - weights.sum()),
          "display_rgb": float(np.max(np.abs(rgb[y,x].cpu().numpy() - expected_rgb)))}
passed = (scale == 1. and errors["projection_edge_px"] < 1e-4 and errors["expected_camera_z_m"] < 1e-4 and
          errors["alpha"] < 5e-4 and errors["display_rgb"] < 5e-4 and float(alpha[0,0]) == 0 and
          not bool(torch.count_nonzero(rgb[0,0])))
save(out / "ed-probe.json", {"passed": bool(passed), "errors": errors, "expected_z_m": expected_z,
     "actual_z_m": float(z[y,x]), "expected_alpha": float(weights.sum()), "actual_alpha": float(alpha[y,x]),
     "expected_display_rgb": expected_rgb.tolist(), "actual_display_rgb": rgb[y,x].cpu().tolist(),
     "native_edge_means2d": info["means2d"].cpu().tolist(), "resources": runtime(), "wall_seconds": time.perf_counter() - started})
require(passed, "Independent projection, ED, alpha, display RGB or empty background fixture")
