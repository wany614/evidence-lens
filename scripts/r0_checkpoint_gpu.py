"""Probe two sealed Gaussian checkpoints with gsplat 1.5.3, without an optimizer."""

import argparse
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from evidence_lens import stage3_gaussians as g

EPS = (0.0, 1e-12)


def source_photo(path, expected):
    import torch

    g.require(g.sha(path) == expected, "Source photograph SHA differs")
    with Image.open(path) as im:
        g.require(im.mode == "RGB" and im.size == (256, 256), "Unexpected source grid")
        return torch.as_tensor(np.asarray(im, dtype=np.float32) / 255, device="cuda:0")


def count_rows(a):
    return {str(e): int(np.count_nonzero(a > e)) for e in EPS}


def probe(p, camera, eps, target):
    import torch
    from gsplat import rasterization

    means, scales, quats, opacity, colours = g.native_parameters(p)
    k, pose, width, height, clip = g.view_tensor(camera, means.device)
    rgb, alpha, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacity,
        colors=colours,
        viewmats=pose,
        Ks=k,
        width=width,
        height=height,
        near_plane=clip[0],
        far_plane=clip[1],
        radius_clip=0.0,
        eps2d=eps,
        rasterize_mode="classic",
        packed=False,
        render_mode="RGB",
        backgrounds=None,
    )
    # For direct RGB with fixed geometry this derivative equals the sum of the
    # per-pixel alpha weights after front-to-back transmittance and raster cutoffs.
    (deriv,) = torch.autograd.grad(rgb[..., 0].sum(), colours, retain_graph=target is not None)
    w = deriv[:, 0].detach().cpu().numpy()
    alpha_sum = float(alpha.detach().double().sum())
    weight_sum = float(np.sum(w, dtype=np.float64))
    g.require(
        torch.isfinite(deriv).all().item()
        and torch.count_nonzero(deriv[:, 1:]).item() == 0
        and np.all(w >= 0)
        and abs(alpha_sum - weight_sum) <= max(0.05, alpha_sum * 2e-4),
        "Colour-probe weights disagree with accumulated opacity",
    )
    n = len(w)
    radii = meta["radii"].detach().cpu().numpy().reshape(n, -1).copy()
    tiles = meta["tiles_per_gauss"].detach().cpu().numpy().reshape(n).copy()
    positive_r = (radii > 0).all(axis=1)
    g.require(
        radii.shape[1] in (1, 2)
        and np.all(tiles >= 0)
        and not np.any((w > 0) & (tiles == 0))
        and not np.any((tiles > 0) & ~positive_r),
        "Raster and tile support disagree",
    )
    data = {"radii": radii, "tiles": tiles, "weight": w}
    result = {
        "radius_xy_dimensions": radii.shape[1],
        "positive_radius": int(positive_r.sum()),
        "intersected_tiles": int(np.sum(tiles, dtype=np.int64)),
        "tiled_rows": int(np.count_nonzero(tiles)),
        "colour_weight_rows": count_rows(w),
        "colour_weight_sum": weight_sum,
        "alpha_sum": alpha_sum,
        "photometric": None,
    }
    if target is not None:
        loss = (g.display(rgb[0]) - target).abs().mean()
        groups = torch.autograd.grad(loss, [p[k] for k in g.PARAMS])
        norms = np.stack(
            [v.detach().reshape(n, -1).double().norm(dim=1).cpu().numpy() for v in groups], axis=1
        )
        g.require(np.isfinite(norms).all(), "Nonfinite source task gradient")
        data["task_row_l2"] = norms
        result["photometric"] = {
            "loss_display_mae": float(loss.detach()),
            "nonzero_any_rows": count_rows(norms.max(axis=1)),
            "groups": {
                k: {
                    "row_counts": count_rows(norms[:, i]),
                    "whole_group_l2": float(np.linalg.norm(norms[:, i])),
                }
                for i, k in enumerate(g.PARAMS)
            },
        }
    return data, result


def run(a):
    import torch

    started = time.perf_counter()
    device = g.check_gpu()
    cfg = g.read(a.config)
    ids = g.SETS[cfg["condition_id"]]
    sources = dict(a.source)
    g.require(
        tuple(cfg["source_ids"]) == ids
        and set(sources) == set(ids)
        and len(a.source) == len(ids)
        and g.sha(a.cameras) == cfg["camera_sha256"]
        and cfg["optimization_steps"] == 0,
        "R0 input contract mismatch",
    )
    views = g.valid_cameras(g.read(a.cameras), ids + g.TARGETS)
    images = {v: source_photo(Path(sources[v]), cfg["source_sha256"][v]) for v in ids}
    result = {
        "schema": "el-r0-cuda-checkpoint-1",
        "run_id": os.environ["EL_RUN_ID"],
        "object": cfg["object_id"],
        "condition": cfg["condition_id"],
        "job_sha256": cfg["formal_handoff_sha256"],
        "source_sha256": cfg["source_sha256"],
        "camera_sha256": cfg["camera_sha256"],
        "parents": cfg["originals"],
        "optimizer_steps": 0,
        "thresholds": list(EPS),
        "rows": 4096,
        "task_gradient_parameters": list(g.PARAMS),
        "historical_per_row_gradient": None,
        "checkpoints": {},
    }
    torch.cuda.reset_peak_memory_stats()
    for phase, file in (("initial", a.initial), ("final", a.final)):
        spec = cfg["checkpoints"][phase]
        g.require(g.sha(file) == spec["file_sha256"], "Checkpoint SHA differs")
        with np.load(file, allow_pickle=False) as z:
            g.require(
                set(z.files) == set(g.PARAMS) and g.digest_values(z) == spec["parameter_sha256"],
                "Checkpoint tensors differ",
            )
            p = {k: torch.tensor(z[k], device=device, requires_grad=True) for k in g.PARAMS}
        stage = {"file_sha256": g.sha(file), "parameter_sha256": spec["parameter_sha256"], "views": {}}
        for v in ids + g.TARGETS:
            data, row = probe(p, views[v], cfg["eps2d"], images.get(v))
            if v in ids:
                expected = spec["loss_by_source"][v]
                g.require(
                    abs(row["photometric"]["loss_display_mae"] - expected) < 3e-5,
                    "Original source loss mismatch",
                )
                row["photometric"]["original_source_mae"] = expected
            f = a.output / (phase + "-" + v + ".npz")
            np.savez_compressed(f, **data)
            row["file"] = {"path": f.name, "bytes": f.stat().st_size, "sha256": g.sha(f)}
            stage["views"][v] = row
        after = g.digest_values({k: x.detach().cpu().numpy() for k, x in p.items()})
        g.require(after == spec["parameter_sha256"], "Diagnostic mutated the checkpoint")
        stage["after_probe_parameter_sha256"] = after
        result["checkpoints"][phase] = stage
    result["runtime"] = g.runtime()
    result["elapsed_s"] = time.perf_counter() - started
    g.write(a.output / "r0-gpu.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("config", "initial", "final", "cameras", "output"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--source", action="append", nargs=2, required=True, metavar=("ID", "PNG"))
    run(p.parse_args())
