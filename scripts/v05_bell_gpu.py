"""Render original M4 Gaussians on registered analysis grids. Receives no photographs or native scene."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(root, out):
    import torch

    run_id = os.environ["EL_RUN_ID"]
    ticket = json.loads((root / "ticket.json").read_bytes())
    if ticket["schema"] != "el-v05-bell-gpu-ticket-1" or ticket["allowed_photographic_inputs"] != []:
        raise ValueError("incorrect Gaussian-only source role")
    for name, expected in ticket["files"].items():
        if digest(root/name) != expected:
            raise ValueError("changed ticket member " + name)
    sys.path.insert(0, str(root / "runtime"))
    import stage3_gaussians as native

    device = native.check_gpu()
    receipt = native.read(root / "reconstruction.json")
    budget = native.read(root / "render-config.json")["optimizer"]["render"]
    parameters = root / "scene.npz"
    native.require(digest(parameters) == receipt["checkpoint_sha256"] == ticket["checkpoint_sha256"], "wrong original M4 fit")
    with np.load(parameters, allow_pickle=False) as raw:
        native.require(set(raw.files) == set(native.PARAMS) and native.digest_values(raw) == receipt["final_parameter_sha256"], "wrong parameter bytes")
        params = {k: torch.as_tensor(raw[k], device=device) for k in native.PARAMS}
    count = params["means"].shape[0]
    native.require(count == receipt["final_gaussian_count"] and count > 0 and all(t.shape[0] == count for t in params.values()), "inconsistent actual Gaussian cardinality")
    native.write(out / "depth-probe.json", native.analytic_probe(device, budget))
    grids = native.read(root / "projections.json")
    frames = {}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        activated = native.native_parameters(params)
        for tag, grid in grids.items():
            frame_dir = out / tag
            frame_dir.mkdir()
            frames[tag] = {}
            x0,y0,w,h = grid["active_xywh"]
            W,H = grid["analysis_size"]
            for name in ticket["source_order"] + ticket["target_order"]:
                camera = grid["render_views"][name]
                native.require([camera["width"], camera["height"]] == [w,h], "native camera grid differs")
                view = native.view_tensor(camera, device)
                val, opacity = native.raster(*activated, view, "RGB+ED", budget["eps2d"])
                rgb = native.display(val[..., :3]).cpu().numpy().astype(np.float32)
                alpha = opacity.cpu().numpy().astype(np.float32)
                z = val[..., 3].cpu().numpy().astype(np.float32)
                native.require(np.isfinite(rgb).all() and np.isfinite(alpha).all() and ((alpha >= 0) & (alpha <= 1)).all(), "invalid native color or alpha")
                z[(alpha <= 0) | ~np.isfinite(z) | (z <= 0)] = np.nan
                if (W,H) != (w,h):
                    dest = np.s_[y0:y0+h,x0:x0+w]
                    rgb2, a2, z2 = np.zeros((H,W,3), np.float32), np.zeros((H,W), np.float32), np.full((H,W), np.nan, np.float32)
                    rgb2[dest], a2[dest], z2[dest] = rgb, alpha, z
                    rgb,alpha,z = rgb2,a2,z2
                path = frame_dir / (name+".npz")
                np.savez_compressed(path,rgb=rgb,alpha=alpha,z=z)
                frames[tag][name] = {"path": tag+"/"+path.name,"sha256":digest(path),"bytes": path.stat().st_size}
    torch.cuda.synchronize()
    native.write(out / "predictions.json", {"schema":"el-v05-bell-render-1","run_id": run_id,"frames":frames,
                 "registered_packet_run":ticket["run_id"],"checkpoint_sha256":digest(parameters),"gaussians":count,
                 "depth_probe_sha256":digest(out/"depth-probe.json"),"worker_seconds":time.perf_counter()-started,
                 "runtime":native.runtime(),"new_optimization_steps":0,"source_photo_reads":0,"target_reference_reads":0,
                 "model":"original Stage C 4096-Gaussian bell M4 model", "render_core_sha256":ticket["files"]["runtime/stage3_gaussians.py"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    run(Path(__file__).resolve().parent, parser.parse_args().out)
