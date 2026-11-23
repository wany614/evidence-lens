"""Authenticate and measure old D1 checkpoints, original photos and 057 cameras without fitting."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from evidence_lens._bundle import encoded, load, sha
from evidence_lens.formal_inputs import PinnedRun, require
from evidence_lens import stage3_gaussians as g
from r0_geometry import covariance, project, project_summary, quantiles


def csvfile(path, rows):
    with path.open("x", newline="", encoding="utf8") as f:
        dw = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        dw.writeheader()
        dw.writerows(rows)


def picture(rgb):
    return Image.fromarray(np.clip(np.floor(np.asarray(rgb) * 255 + 0.5), 0, 255).astype(np.uint8))


def board(out, group, views):
    canvas = Image.new("RGB", (4 * 256, 34 + len(views) * 282), "white")
    dr = ImageDraw.Draw(canvas)
    for col, title in enumerate(
        (
            "Original photo",
            "Saved source self-render" if group == "sources" else "Saved target render",
            "RGB MAE (0 to .30)",
            "Accumulated opacity",
        )
    ):
        dr.text((col * 256 + 7, 9), title, fill="black")
    for i, (v, src, pred, err, alpha) in enumerate(views):
        row = 34 + i * 282
        dr.text((8, row), f"{v}   |   full crop MAE {float(err.mean(dtype=np.float64)):.5f}", fill="#111111")
        for col, im in enumerate(
            (
                src,
                pred,
                np.tile(np.clip(err / 0.30, 0, 1)[..., None], (1, 1, 3)),
                np.tile(alpha[..., None], (1, 1, 3)),
            )
        ):
            canvas.paste(picture(im), (col * 256, row + 20))
    canvas.save(out)


def pinned(store, shelf, pin):
    r = PinnedRun(store if (store / "runs" / pin["run_id"]).is_dir() else shelf, pin)
    require(r.run["execution_status"] == "succeeded", "Unsuccessful original")
    return r


def scene(run, name):
    path = run.open("payload/" + name + ".npz")
    with np.load(path, allow_pickle=False) as z:
        vals = {k: z[k] for k in g.PARAMS}
        require(
            set(z.files) == set(g.PARAMS)
            and all(v.dtype == np.float32 and len(v) == 4096 and np.isfinite(v).all() for v in vals.values()),
            "Unexpected checkpoint",
        )
    return vals, sha(path)


def describe_state(obj, co, phase, s):
    scale = np.exp(s["log_scales"].astype("f8"))
    op = 1 / (1 + np.exp(-s["opacity_logits"].astype("f8")))
    return {
        "object": obj,
        "condition": co,
        "phase": phase,
        "count": len(op),
        "scale_m": quantiles(scale),
        "anisotropy_max_over_min": quantiles(scale.max(axis=1) / scale.min(axis=1)),
        "opacity": quantiles(op),
        "mean_coordinates_m": {ax: quantiles(s["means"][:, i]) for i, ax in enumerate("xyz")},
    }


def projection(out, cells, rows, obj, co, phase, state, cams, eps):
    cov = covariance(state)
    union = {
        k: np.zeros(len(state["means"]), bool)
        for k in ("inside_clip", "centre_inside", "ellipse_intersects", "aabb_intersects")
    }
    for name, cam in cams.items():
        d = project(state, cov, cam, eps)
        rec = project_summary(d, cam)
        f = out / "row-diagnostics" / f"{obj}-{co}-{phase}-{name}.npz"
        np.savez_compressed(f, **d)
        simple = {k: v for k, v in rec.items() if not isinstance(v, dict)}
        rows.append(
            {
                "object": obj,
                "condition": co,
                "phase": phase,
                "view": name,
                "role": "source" if name.startswith("s") else "target",
                **simple,
                **{label + "_" + q: vs[q] for label, vs in rec.items() if isinstance(vs, dict) for q in vs},
            }
        )
        if name.startswith("s"):
            for k in union:
                union[k] |= d[k]
    result = {k: int(v.sum()) for k, v in union.items()}
    cells.append({"object": obj, "condition": co, "phase": phase, **result})
    return union


def main(a):
    cfg = load(a.config)
    out = a.output
    for sub in ("boards", "row-diagnostics", "public"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    states = []
    changes = []
    quality = []
    projrows = []
    projunions = []
    training = []
    details = {}
    baseline = None
    synthetic_modes = set()
    for h in cfg["arms"]:
        obj, co = h["object"], h["condition"]
        ids = g.SETS[co]
        parents = {k: pinned(a.store, a.shelf, pin) for k, pin in h["parents"].items()}
        gen, ren = parents["generation"], parents["render"]
        r = load(gen.open("payload/reconstruction.json"))
        p = load(ren.open("payload/prediction.json"))
        source = parents["sources"]
        reference = parents["reference"]
        rig = parents["rig"]
        budget = load(parents["budget"].open("payload/optimizer.json"))
        sindex = load(source.open("payload/source-images.json"))
        tindex = load(reference.open("payload/references.json"))
        cameras = rig.open(f"payload/condition/{co}/render_cameras.json")
        cams = g.valid_cameras(load(cameras), ids + g.TARGETS)
        eps = budget["render"]["eps2d"]
        require(
            tuple(r["generator_source_ids"]) == ids
            and p["generator_source_ids"] == list(ids)
            and p["target_ids"] == list(g.TARGETS)
            and p["reconstruction_run_id"] == gen.pin["run_id"]
            and sha(cameras) == p["render_camera_sha256"]
            and r["trace_sha256"] == sha(gen.open("payload/optimizer.jsonl"))
            and p["depth_verified"] is True,
            "Wrong original renderer or optimization",
        )
        initial, initsha = scene(gen, "initial")
        final, finalsha = scene(gen, "scene")
        require(
            initsha == r["initial_checkpoint_sha256"]
            and finalsha == r["checkpoint_sha256"] == p["checkpoint_sha256"]
            and g.digest_values(initial) == r["initial_parameter_sha256"]
            and g.digest_values(final) == r["final_parameter_sha256"],
            "Checkpoint or parameter digest differs",
        )
        regenerated = g.initial_values(budget["initialization"])
        require(
            all(np.array_equal(initial[n].view("u4"), regenerated[n].view("u4")) for n in g.PARAMS),
            "Seeded initial tensors differ",
        )
        if baseline is None:
            baseline = g.digest_values(initial)
        require(baseline == g.digest_values(initial), "The four random initializations differ")
        log = [
            json.loads(line)
            for line in gen.open("payload/optimizer.jsonl").read_text().splitlines()
            if line.strip()
        ]
        require(len(log) == r["gaussian_optimization_steps"] == 256, "Wrong main budget")
        for i, x in enumerate(log):
            v = ids[i % len(ids)]
            require(
                x["update"] == i + 1
                and x["source"] == v
                and x["image_sha256"] == r["source_rgb_sha256"][v]
                and x["pixels"] == 65536
                and set(x["gradient_l2"]) == set(g.PARAMS)
                and all(np.isfinite(gv) and gv >= 0 for gv in x["gradient_l2"].values()),
                "Wrong historical trace",
            )
        for v in ids:
            subset = [x for x in log if x["source"] == v]
            training.append(
                {
                    "object": obj,
                    "condition": co,
                    "source": v,
                    "updates": len(subset),
                    "loss_initial": r["loss_initial_by_source"][v],
                    "loss_final": r["loss_final_by_source"][v],
                    **{
                        k + "_whole_group_nonzero_steps": sum(t["gradient_l2"][k] > 0 for t in subset)
                        for k in g.PARAMS
                    },
                    "historical_per_row_gradients": "unavailable",
                    "historical_per_row_transmittance": "unavailable",
                }
            )
        dval = {k: (final[k].astype("f8") - initial[k].astype("f8")).reshape(4096, -1) for k in g.PARAMS}
        numeric = {k: np.any(v != 0, axis=1) for k, v in dval.items()}
        bits = {k: (initial[k].view("u4") != final[k].view("u4")).reshape(4096, -1).any(axis=1) for k in g.PARAMS}
        union = np.logical_or.reduce(list(numeric.values()))
        changes.append(
            {
                "object": obj,
                "condition": co,
                "n": 4096,
                "any_numeric": int(union.sum()),
                "any_bits": int(np.logical_or.reduce(list(bits.values())).sum()),
                **{k + "_numeric_rows": int(v.sum()) for k, v in numeric.items()},
                **{k + "_bit_rows": int(v.sum()) for k, v in bits.items()},
                **{
                    "position_change_m_" + pct: value
                    for pct, value in quantiles(np.linalg.norm(dval["means"], axis=1)).items()
                },
                **{
                    "changed_position_m_" + pct: value
                    for pct, value in quantiles(np.linalg.norm(dval["means"][numeric["means"]], axis=1)).items()
                },
            }
        )
        np.savez_compressed(
            out / "row-diagnostics" / f"{obj}-{co}-deltas.npz",
            **{k: np.linalg.norm(dv, axis=1) for k, dv in dval.items()},
            any_change=union,
        )
        for phase, state in [("initial", initial), ("final", final)]:
            states.append(describe_state(obj, co, phase, state))
            projection(out, projunions, projrows, obj, co, phase, state, cams, eps)
        if co not in synthetic_modes:
            synthetic_modes.add(co)
            camfile = parents["synthetic"].open(f"payload/CASE/{co}/source_cameras.json")
            projection(
                out,
                projunions,
                projrows,
                "StageC-057",
                co,
                "initial",
                initial,
                g.valid_cameras(load(camfile), ids),
                eps,
            )
        allpanels = {"sources": [], "targets": []}
        for v in ids + g.TARGETS:
            src = v in ids
            owner = source if src else reference
            ix = (sindex if src else tindex)["views"][v]["rgb"]
            img = owner.open("payload/" + ix["path"])
            require(
                sha(img) == ix["sha256"] and (not src or sha(img) == r["source_rgb_sha256"][v]),
                "Wrong primary view",
            )
            f = ren.open("payload/" + v + ".npz")
            require(sha(f) == p["views"][v]["sha256"], "Rendered view changed")
            with Image.open(img) as im:
                photo = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            with np.load(f, allow_pickle=False) as z:
                rgb = z["rgb"]
                alpha = z["alpha"]
                zz = z["z"]
            require(
                photo.shape == rgb.shape == (256, 256, 3)
                and np.all(np.isfinite(rgb))
                and np.isfinite(alpha).all()
                and np.all((alpha >= 0) & (alpha <= 1)),
                "Invalid image grid",
            )
            err = np.mean(abs(rgb - photo), axis=2, dtype=np.float32)
            if src:
                require(
                    abs(float(err.mean(dtype=np.float64)) - r["loss_final_by_source"][v]) < 3e-5,
                    "Saved self-render does not reproduce the old source MAE",
                )
            quality.append(
                {
                    "object": obj,
                    "condition": co,
                    "view": v,
                    "role": "source" if src else "target",
                    "crop_pixels": 65536,
                    "mean_rgb_mae": float(err.mean(dtype=np.float64)),
                    "pixels_strict_mae_gt_0_05": int((err > 0.05).sum()),
                    "positive_alpha": int((alpha > 0).sum()),
                    "positive_depth": int((np.isfinite(zz) & (zz > 0)).sum()),
                    "alpha_sum": float(np.sum(alpha, dtype=np.float64)),
                    "photo_sha256": sha(img),
                    "prediction_sha256": sha(f),
                }
            )
            allpanels["sources" if src else "targets"].append((v, photo, rgb, err, alpha))
        for group, viewrows in allpanels.items():
            board(out / "boards" / f"{obj}-{co}-{group}.png", group, viewrows)
        details[obj + "-" + co] = {
            "parents": h["parents"],
            "initial_file_sha256": initsha,
            "final_file_sha256": finalsha,
            "initial_parameter_sha256": baseline,
            "final_parameter_sha256": g.digest_values(final),
            "original_gpu_environment": r["runtime"],
            "original_trace_sha256": r["trace_sha256"],
            "optimizer_sha256": sha(parents["budget"].open("payload/optimizer.json")),
            "initialization": budget["initialization"],
            "learning_rates": budget["optimization"]["learning_rates"],
            "source_camera_sha256": r["source_camera_sha256"],
            "render_camera_sha256": sha(cameras),
            "clip_m": {v: cams[v]["clip_m"] for v in cams},
            "backend_eps2d_px2": eps,
        }
    for name, rows in [
        ("parameter-rows", changes),
        ("projected-views", projrows),
        ("source-union", projunions),
        ("image-quality", quality),
        ("historical-log", training),
    ]:
        csvfile(out / "public" / (name + ".csv"), rows)
    (out / "public/state-quantiles.json").write_bytes(
        encoded({"schema": "el-r0-state-quantiles-1", "states": states})
    )
    summary = {
        "schema": "el-r0-cpu-readback-1",
        "run_id": os.environ["EL_RUN_ID"],
        "arms": details,
        "same_initial_parameter_sha256": baseline,
        "rng_regenerated_all_raw_tensors_bitwise": True,
        "optimization_steps": 0,
        "historical_per_row_gradient": None,
        "staged_cameras": "released source cameras of family 057 used with the same saved initial arrays",
        "private_boards": {f.name: sha(f) for f in sorted((out / "boards").iterdir())},
        "public": {f.name: sha(f) for f in sorted((out / "public").iterdir())},
    }
    (out / "r0-cpu.json").write_bytes(encoded(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("config", "store", "shelf", "output"):
        parser.add_argument("--" + n, type=Path, required=True)
    main(parser.parse_args())
