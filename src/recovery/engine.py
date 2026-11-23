"""Data-plane kernels. Inputs are individual recorder substitutions, never a cell root."""
from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from recovery.contract import HORIZON, PARAMS, SOURCES, TARGETS, WIDTHS, phase, validate_recipe
from recovery.scene import cameras, draw, init_parameters, photometric, runtime, setup, source_image
from recovery.topology import BoundedRefinement, assert_adam


def tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [tree(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(tree(v, device) for v in value)
    return value


def rng_snapshot():
    typ, keys, pos, has_gauss, cache = np.random.get_state()
    return {"torch_cpu": torch.get_rng_state(), "torch_cuda0": torch.cuda.get_rng_state(0),
            "python": random.getstate(), "numpy": [typ, keys.tolist(), pos, has_gauss, cache]}


def restore_rng(record):
    torch.set_rng_state(record["torch_cpu"].cpu())
    torch.cuda.set_rng_state(record["torch_cuda0"].cpu(), 0)
    random.setstate(record["python"])
    typ, keys, pos, has_gauss, cache = record["numpy"]
    np.random.set_state((typ, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cache))


def rates(recipe, update, scale):
    progress = (update - 1) / (HORIZON - 1)
    return {PARAMS[k]: math.exp((1 - progress) * math.log(start) + progress * math.log(end)) *
            (scale if k == "means" else 1.) for k, (start, end) in recipe["learning_rates"].items()}


def optimizers(params, recipe, scale):
    lr = rates(recipe, 1, scale)
    return {k: torch.optim.Adam([v], lr=lr[k], betas=tuple(recipe["adam"]["betas"]),
                                eps=recipe["adam"]["eps"], foreach=False, fused=False) for k, v in params.items()}


def valid_params(params, cap):
    n = len(params["means"])
    require(set(params) == set(PARAMS.values()) and 1 <= n <= cap, "Checkpoint parameter roster or count")
    for field, k in PARAMS.items():
        expected = (n,) if WIDTHS[field] is None else (n, WIDTHS[field])
        p = params[k]
        require(p.dtype == torch.float32 and p.shape == expected and bool(torch.isfinite(p).all()),
                "Invalid Gaussian tensor " + k)
    require(bool((params["quats"].norm(dim=1) > 1e-9).all()) and
            bool(torch.isfinite(params["scales"].exp()).all()) and bool((params["scales"].exp() > 0).all()),
            "Invalid quaternion or scale")


def load_point(path, digest, ident):
    require(sha(path) == digest, "Issued checkpoint SHA-256 differs")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    require(ck["schema"] == "el-recovery-checkpoint-1" and ck["identity"] == ident, "Checkpoint identity differs")
    validate_recipe(ck["recipe"])
    require(ck["horizon"] == HORIZON and ck["step"] in ck["recipe"]["checkpoints"], "Checkpoint schedule")
    valid_params(ck["params"], ck["recipe"]["topology"]["cap"])
    return ck


def generation(config, inputs, out):
    begun = time.perf_counter()
    device = setup()
    cfg = load(config)
    ident = cfg["science"]
    ids = SOURCES[ident["condition"]]
    require(set(inputs) == {"recipe", "init", "initmeta", "cameras", *ids} |
            ({"resume"} if cfg.get("resume_origin") else set()), "Wrong optimizer input roles")
    for role, digest in cfg["role_hashes"].items():
        require(sha(inputs[role]) == digest, "Changed optimizer role " + role)
    recipe = validate_recipe(load(inputs["recipe"]))
    require(cfg["role_hashes"]["recipe"] == ident["recipe_sha256"], "Recipe identity")
    method = load(inputs["initmeta"])
    views, scale, source_calibration, frame = cameras(inputs["cameras"], ids, device)
    require(isinstance(method["actual_method"], str) and method["actual_method"].strip() and
            math.isclose(scale, method["scene_scale_m"], rel_tol=1e-5, abs_tol=1e-8), "Endpoint baseline or init method")
    source = {s: source_image(inputs[s], views[s], device) for s in ids}
    random.seed(recipe["seed"])
    np.random.seed(recipe["seed"])
    torch.manual_seed(recipe["seed"])
    torch.cuda.manual_seed_all(recipe["seed"])
    first = 0
    counts = {s: 0 for s in ids}
    totals = Counter()
    strategy = BoundedRefinement(recipe["topology"], scale)
    if cfg.get("resume_origin"):
        ck = load_point(inputs["resume"], cfg["role_hashes"]["resume"], ident)
        first = ck["step"]
        require(first == cfg["resume_origin"]["step"] and ck["producer_run_id"] == cfg["resume_origin"]["run_id"]
                and ck["recipe"] == recipe and ck["source_order"] == list(ids)
                and ck["next_source_index"] == first % len(ids)
                and ck["counts"] == {s: (first + len(ids) - 1 - i) // len(ids) for i, s in enumerate(ids)}
                and ck["endpoint_frame_sha256"] == frame and ck["source_calibration"] == source_calibration
                and math.isclose(ck["scene_scale_m"], scale, rel_tol=1e-12), "Resume cursor/parent differs")
        params = {k: torch.nn.Parameter(x.to(device)) for k, x in ck["params"].items()}
        opts = optimizers(params, recipe, scale)
        for k, opt in opts.items():
            opt.load_state_dict(ck["adam"][k])
        strategy.state = tree(ck["strategy"], device)
        counts, totals = dict(ck["counts"]), Counter(ck["topology_totals"])
        restore_rng(ck["rng"])
        del ck
    else:
        params = init_parameters(inputs["init"], recipe["topology"]["cap"], device)
        opts = optimizers(params, recipe, scale)
    valid_params(params, recipe["topology"]["cap"])
    assert_adam(params, opts, first)
    end = cfg["end_step"]
    require(first < end <= recipe["stop_step"] and end in recipe["checkpoints"], "Invalid continuation endpoint")
    points = {}
    run_id = os.environ["EL_RUN_ID"]
    (out / "checkpoints").mkdir()
    def issue(step):
        torch.cuda.synchronize()
        valid_params(params, recipe["topology"]["cap"])
        assert_adam(params, opts, step)
        target = out / "checkpoints" / f"step-{step:04d}.pt"
        temp = target.with_suffix(".part")
        blob = {"schema": "el-recovery-checkpoint-1", "producer_run_id": run_id, "identity": ident, "recipe": recipe,
                "step": step, "horizon": HORIZON, "phase": phase(step, recipe), "scene_scale_m": scale,
                "actual_method": method["actual_method"], "endpoint_frame_sha256": frame, "source_calibration": source_calibration,
                "source_order": list(ids), "next_source_index": step % len(ids),
                "counts": counts, "topology_totals": dict(totals), "params": tree(params, "cpu"),
                "adam": {k: tree(opt.state_dict(), "cpu") for k, opt in opts.items()},
                "strategy": tree(strategy.state, "cpu"), "rng": rng_snapshot(),
                "next_lr": rates(recipe, min(step + 1, HORIZON), scale)}
        with temp.open("xb") as stream:
            torch.save(blob, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temp.rename(target)
        points[str(step)] = {"step": step, "phase": phase(step, recipe), "gaussians": len(params["means"]),
                             "checkpoint_sha256": sha(target), "bytes": target.stat().st_size,
                             "path": target.relative_to(out).as_posix(), "producer_run_id": run_id}
        save(out / "issued.json", {"run_id": run_id, "identity": ident, "checkpoints": points})
    if not cfg.get("resume_origin"):
        issue(0)
    started = time.perf_counter()
    with (out / "steps.jsonl").open("x", encoding="utf8", buffering=1) as log:
        for step in range(first + 1, end + 1):
            sid = ids[(step - 1) % len(ids)]
            n = len(params["means"])
            lr = rates(recipe, step, scale)
            for k, opt in opts.items():
                opt.param_groups[0]["lr"] = lr[k]
                opt.zero_grad(set_to_none=True)
            pred, alpha, _, info = draw(params, views[sid], recipe["eps2d"])
            require(bool(torch.isfinite(alpha).all()) and float(alpha.max()) > 0, "Empty/nonfinite projected support")
            strategy.retain(params, opts, step, info)
            loss, l1, ssim = photometric(pred, source[sid], recipe["loss"])
            require(bool(torch.isfinite(loss)), "Nonfinite source photometric loss")
            loss.backward()
            strategy.observe(params, step, info)
            pre = {k: v.detach().clone() for k, v in params.items()}
            gradients = {k: float(v.grad.norm()) for k, v in params.items()}
            require(all(math.isfinite(a) for a in gradients.values()), "Nonfinite Gaussian gradient")
            for opt in opts.values():
                opt.step()
            change = {k: float((v.detach() - pre[k]).norm()) for k, v in params.items()}
            require(all(math.isfinite(a) for a in change.values()), "Nonfinite parameter update")
            events = strategy.refine(params, opts, step)
            for ev in events:
                if "selected" in ev:
                    totals[ev["operation"]] += ev["selected"]
            counts[sid] += 1
            radii = info["radii"].detach()
            visible = (radii > 0).all(-1)
            xy = info["means2d"].detach()[visible]
            row = {"step": step, "source": sid, "phase": phase(step, recipe), "source_counts": dict(counts),
                   "pixels": views[sid][2] * views[sid][3], "n_before": n, "n_after": len(params["means"]),
                   "loss": float(loss.detach()), "display_l1": float(l1.detach()),
                   "ssim": None if ssim is None else float(ssim.detach()), "lr": lr,
                   "gradient_l2": gradients, "parameter_update_l2_before_topology": change, "topology": events,
                   "visible": int(visible.sum()), "max_radius_px": float(radii.max()),
                   "max_abs_visible_means2d_px": float(xy.abs().max()) if xy.numel() else None}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if step in recipe["checkpoints"]:
                issue(step)
    save(out / "reconstruction.json", {"schema": "el-recovery-reconstruction-1", "run_id": run_id,
         "status": "completed" if end == cfg["planned_stop_step"] else "engineering_segment", "identity": ident,
         "job_stop_step": cfg["planned_stop_step"], "recipe_stop_step": recipe["stop_step"], "schedule_horizon": HORIZON,
         "resume_origin": cfg.get("resume_origin"), "first_completed_step": first, "last_completed_step": end,
         "updates_in_process": end - first, "sampling_counts": counts, "topology_totals": dict(totals),
         "actual_method": method["actual_method"], "scene_scale_m": scale, "endpoint_frame_sha256": frame, "checkpoints": points,
         "trace_sha256": sha(out / "steps.jsonl"), "loop_wall_seconds": time.perf_counter() - started,
         "worker_wall_seconds": time.perf_counter() - begun, "resources": runtime()})


@torch.no_grad()
def render(config, inputs, out):
    begun = time.perf_counter()
    device = setup()
    cfg = load(config)
    ident = cfg["science"]
    ids = SOURCES[ident["condition"]]
    require(set(inputs) == {"cameras", *("c" + s for s in cfg["selected"])} and
            sha(inputs["cameras"]) == cfg["render_camera_sha256"], "Renderer inputs or camera digest")
    views, scale, calibration, frame = cameras(inputs["cameras"], ids + TARGETS, device)
    points = {}
    for key, receipt in sorted(cfg["selected"].items(), key=lambda kv: int(kv[0])):
        ck = load_point(inputs["c" + key], receipt["checkpoint_sha256"], ident)
        require(ck["step"] == int(key) and ck["producer_run_id"] == receipt["producer_run_id"] and
                ck["endpoint_frame_sha256"] == frame and ck["source_calibration"] == {s: calibration[s] for s in ids}
                and math.isclose(ck["scene_scale_m"], scale, rel_tol=1e-5, abs_tol=1e-8), "Wrong checkpoint, frame or source calibration")
        params = {k: t.to(device) for k, t in ck["params"].items()}
        directory = out / f"step-{int(key):04d}"
        directory.mkdir()
        outputs = {}
        for sid in ids + TARGETS:
            rgb, alpha, depth, _ = draw(params, views[sid], ck["recipe"]["eps2d"], depth=True)
            a, z, rgb = (t.cpu().numpy().astype(np.float32, copy=True) for t in (alpha, depth, rgb))
            require(np.isfinite(rgb).all() and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(), "Invalid RGB or alpha")
            z[(a <= 0) | ~np.isfinite(z) | (z <= 0)] = np.nan
            path, png = directory / (sid + ".npz"), directory / (sid + ".png")
            np.savez_compressed(path, rgb=rgb, z=z, alpha=a)
            Image.fromarray(np.floor(rgb * 255 + .5).astype(np.uint8)).save(png)
            outputs[sid] = {"path": path.relative_to(out).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size,
                            "png_path": png.relative_to(out).as_posix(), "png_sha256": sha(png), "png_bytes": png.stat().st_size}
        points[key] = {"step": int(key), "phase": ck["phase"], "gaussians": len(params["means"]),
                       "checkpoint_sha256": receipt["checkpoint_sha256"], "producer_run_id": receipt["producer_run_id"], "views": outputs}
        del params, ck
    save(out / "outputs.json", {"schema": "el-recovery-outputs-1", **ident, "render_camera_sha256": cfg["render_camera_sha256"],
         "reconstruction_run_id": cfg["generation"]["run_id"], "endpoint_frame_sha256": frame,
         "scene_scale_m": scale, "status": "completed", "checkpoints": points,
         "depth_semantics": "opacity-normalized expected camera Z in metres", "rgb_encoding": "display_srgb_float32",
         "worker_wall_seconds": time.perf_counter() - begun, "resources": runtime()})
