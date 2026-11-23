"""Build only an analytic engineering cell with two source PNGs and metadata-only target views."""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from evidence_lens._bundle import load, save

p = argparse.ArgumentParser()
p.add_argument("--candidate", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
if os.environ.get("EL_RUN_KIND") != "dataset_build":
    raise RuntimeError("Record the engineering fixture as dataset_build")
out = a.out
(out / "sources").mkdir()
rng = np.random.default_rng(90417)
n = 48
means = rng.uniform([-.68, -.53, 2.3], [.68, .53, 3.0], (n, 3))
scales = rng.uniform(.02, .07, (n, 3))
scales[::2] *= .10
quat = rng.normal(size=(n, 4))
quat /= np.linalg.norm(quat, axis=1, keepdims=True)
def logit(x):
    return np.log(x / (1 - x))
np.savez(out / "init.npz", means=means.astype("<f4"), log_scales=np.log(scales).astype("<f4"),
         quats=quat.astype("<f4"), opacity_logits=logit(rng.uniform(.2, .5, n)).astype("<f4"),
         color_logits=logit(rng.uniform(.15, .75, (n, 3))).astype("<f4"))
save(out / "init.json", {"actual_method": "analytic-engineering-grid-v1", "scene_scale_m": 1.})
width, height = 48, 48
yy, xx = np.mgrid[:height, :width]
views = {}
for i, (key, center) in enumerate([("s00", -.5), ("s03", .5)] + [(f"v{k:02d}", -.48 + .14*k) for k in range(1, 7)]):
    pose = np.eye(4)
    pose[0, 3] = center
    views[key] = {"width": width, "height": height, "clip_m": [.05, 20],
                  "K": [[43., 0., 22.1], [0., 45., 24.1], [0., 0., 1.]], "c2w": pose.tolist()}
    if key.startswith("s"):
        raster = np.stack(((xx * 4 + i * 7) % 255, (yy * 5 + 20) % 255, (xx + yy + 30) % 255), -1)
        Image.fromarray(raster.astype(np.uint8)).save(out / "sources" / (key + ".png"))
meta = {"schema": "el-camera-1", "unit": "m", "pixel_center": "integer",
        "axis_convention": "OpenCV x right, y down, z forward"}
save(out / "source_cameras.json", {**meta, "views": {k: views[k] for k in ("s00", "s03")}})
save(out / "render_cameras.json", {**meta, "views": views})
recipe = load(a.candidate)
recipe.update(checkpoints=[0, 4, 8, 12, 500, 1500, 3000, 5000])
recipe["topology"].update(start=1, stop=8, every=2, reset_steps=[3], grow_grad2d=1e-9, prune_scale_from=5000)
save(out / "fixture-recipe.json", recipe)
save(out / "timeout-recipe.json", {**recipe, "wall_seconds": 1})
save(out / "fixture.json", {"profile": "analytic-only", "object_id": "generated-fixture", "seed": 90417,
                            "reference_target_images": 0, "resume_at": 4, "end_step": 12,
                            "absolute_parameter_and_moment_tolerance": 0.0001, "relative_tolerance": 0.0001,
                            "absolute_display_loss_tolerance": 0.00003})
