"""Compute recovery quality curves and two source-only appearance references without reading risk predictions."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dtu_score import VIEWS
from evidence_lens.formal_inputs import require
from evidence_lens.scoring_io import new_output, write_json, file_hash
from recovery_eval_io import Originals, Outputs, SHAPE, STEPS, SYNTH, authority, recipe, permit_frozen
from stage3_risk_boards import heat, rgb
from vision_sources import signed


def measurement(ref, pred, valid):
    require(
        ref.dtype == np.float32 and valid.dtype == bool and ref.shape == (*valid.shape, 3),
        "wrong reference domain",
    )
    v = valid & np.isfinite(ref).all(2) & ((ref >= 0) & (ref <= 1)).all(2)
    if pred is not None:
        require(
            pred.dtype == np.float32 and pred.shape == ref.shape, "wrong predicted RGB precision or shape"
        )
    e = (
        np.zeros_like(v)
        if pred is None
        else v & np.isfinite(pred).all(2) & ((pred >= 0) & (pred <= 1)).all(2)
    )
    err = np.full(valid.shape, np.nan, np.float32)
    y = np.full(valid.shape, 255, np.uint8)
    if e.any():
        err[e] = np.abs(pred[e] - ref[e]).mean(1, dtype=np.float32)
        y[e] = err[e] > np.float32(0.05)
    V, E = int(v.sum()), int(e.sum())
    Y = int(np.count_nonzero(y == 1))
    errsum = float(err[e].sum(dtype=np.float64))
    return {
        "V": V,
        "E": E,
        "V_minus_E": V - E,
        "Y": Y,
        "rgb_mae_sum": errsum,
        "MAE": errsum / E if E else None,
        "Y_over_E": Y / E if E else None,
        "E_over_V": E / V if V else None,
    }, (v, e, y, err)


def closest_axis(cameras, sources, view):
    def forward(name):
        z = cameras[name].c2w[:3, 2]
        return z / np.linalg.norm(z)

    axes = {s: float(np.arccos(np.clip(np.dot(forward(s), forward(view)), -1, 1))) for s in sorted(sources)}
    return min(axes, key=lambda s: (axes[s], s)), axes


def source_mean(sources):
    require(
        bool(sources)
        and all(
            a.dtype == np.float32
            and a.shape == (*SHAPE, 3)
            and np.isfinite(a).all()
            and ((a >= 0) & (a <= 1)).all()
            for a in sources.values()
        ),
        "missing or illegal source RGB",
    )
    count = sum(a.shape[0] * a.shape[1] for a in sources.values())
    mean = sum((a.sum((0, 1), dtype=np.float64) for a in sources.values()), np.zeros(3, np.float64)) / count
    return np.broadcast_to(mean.astype(np.float32), (*SHAPE, 3)), mean.tolist()


def table(path, headings, rows, title):
    if not rows:
        return
    w = 256
    h = 256
    label = 22
    stride = 266
    start = 58
    rowheight = 293
    left = 12
    page = Image.new(
        "RGB", (2 * left + stride * len(headings) - 10, start + len(rows) * rowheight), (247, 246, 240)
    )
    pen = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=13)
    pen.text((left, 6), title, fill=(18, 50, 60), font=ImageFont.load_default(size=18))
    pen.text(
        (left, 33),
        "Native display-sRGB. Purple is outside the comparable domain. Grey means a missing image.",
        fill=(18, 50, 60),
        font=font,
    )
    xbar = page.width - 274
    values = np.broadcast_to(np.linspace(0, 0.2, 160, dtype=np.float32), (11, 160))
    page.paste(Image.fromarray(heat(values, 0.2, np.ones(values.shape, bool))), (xbar, 33))
    pen.text((xbar - 19, 30), "0", fill=(18, 50, 60), font=font)
    pen.text((xbar + 165, 30), "0.20+ MAE", fill=(18, 50, 60), font=font)
    for iy, (view, images) in enumerate(rows):
        for ix, (name, img) in enumerate(zip(headings, images, strict=True)):
            x = left + ix * stride
            yy = start + iy * rowheight
            pen.text((x, yy), view + "  " + name, fill=(18, 50, 60), font=font)
            if img is None:
                pen.rectangle((x, yy + label, x + w - 1, yy + label + h - 1), fill=(113, 118, 127))
            else:
                page.paste(Image.fromarray(img), (x, yy + label))
    path.parent.mkdir(parents=True, exist_ok=True)
    page.save(path)


def rgb_image(a):
    return None if a is None else rgb(a)


def sum_views(rows):
    if not rows:
        return None
    c = {k: sum(r[k] for r in rows) for k in ("V", "E", "V_minus_E", "Y", "rgb_mae_sum")}
    ms = [r["MAE"] for r in rows if r["MAE"] is not None]
    return {
        **c,
        "MAE": c["rgb_mae_sum"] / c["E"] if c["E"] else None,
        "Y_over_E": c["Y"] / c["E"] if c["E"] else None,
        "macro_view_MAE": float(np.mean(ms, dtype=np.float64)) if ms else None,
        "defined_views": len(ms),
        "scheduled_views": len(rows),
    }


def work(roots, req, out, rid):
    require(req["schema"] == "el-recovery-quality-request-1", "wrong quality request")
    authority(roots, req)
    recipe(roots, req)
    if req["object_id"] == "set6":
        frozen = permit_frozen(roots, req)
    else:
        frozen = None
    old = Originals(roots, req)
    outputs = Outputs(roots, req, old.case)
    data = outputs.data
    allowed = old.case.generator
    targets = tuple(VIEWS)
    selected = [frozen["stop_step"]] if frozen else list(STEPS)
    if frozen and data["status"] in ("failed", "truncated"):
        # Retain measured intermediate diagnostics. They never stand in for the frozen endpoint.
        reached = {int(k) for k in data["checkpoints"]}
        selected = sorted(
            {frozen["stop_step"]} | {s for s in STEPS if s in reached and s < frozen["stop_step"]}
        )
    steps = {
        str(s): {
            "step": s,
            "status": "not_reached",
            "reason": data.get("failure_reason")
            or data.get("reason")
            or "not present in archived trajectory",
        }
        for s in selected
    }
    constant, mean = source_mean(old.sources)
    comparators = {}
    old_target = {v: old.case.frame(v) for v in targets}
    old_source = {s: old.case.frame(s) for s in allowed}
    info = {}
    photos = {}
    for v in targets:
        s, ang = closest_axis(old.case.cameras, allowed, v)
        comparators[v] = {"source_id": s, "source_axis_angle_radians": ang}
        photos[v] = old.sources[s]
        frame = old_target[v]
        op = None if frame is None else frame["rgb"]
        registries = {"old": op, "mean_color": constant, "nearest_axis_copy": photos[v]}
        info[v] = {}
        for dom, valid in [("full_crop", old.masks[v]), ("object", old.object_masks[v])]:
            if valid is None:
                info[v][dom] = {"status": "unavailable", "reason": old.object_reason[v]}
                continue
            info[v][dom] = {k: measurement(old.refs[v], x, valid)[0] for k, x in registries.items()}
    refs = {v: old.hashes[v] for v in targets}
    original_source = {s: old.case.meta["source_images"][s]["sha256"] for s in allowed}
    old_renders = {s: old.case.meta["renders"].get(s, {}).get("sha256") for s in (*allowed, *targets)}
    boards = {}
    legacy_sources = {
        s: measurement(
            old.sources[s], old_source[s]["rgb"] if old_source[s] is not None else None, np.ones(SHAPE, bool)
        )[0]
        for s in allowed
    }
    with new_output(out) as dest:
        for s in selected:
            if str(s) not in data["checkpoints"]:
                continue
            step = data["checkpoints"][str(s)]
            target_rows = []
            source_rows = []
            stats = {}
            srcstats = {}
            for v in targets:
                new = outputs.rgb(s, v)
                ref = old.refs[v]
                before = old_target[v]["rgb"] if old_target[v] is not None else None
                valid = old.masks[v]
                now, b = measurement(ref, new, valid)
                _, a = measurement(ref, before, valid)
                pair = a[1] & b[1]
                pold, _ = measurement(ref, before, pair)
                pnew, _ = measurement(ref, new, pair)
                obj = old.object_masks[v]
                stats[v] = {
                    "prediction_sha256": step["views"][v].get("sha256"),
                    "reference_sha256": refs[v],
                    "full_crop": now,
                    "object": measurement(ref, new, obj)[0] if obj is not None else None,
                    "object_reason": old.object_reason[v],
                    "paired_comparable": {
                        "planned_reference_V": now["V"],
                        "paired_E": int(pair.sum()),
                        "old": pold,
                        "new": pnew,
                    },
                }
                target_rows.append(
                    (
                        v,
                        [
                            rgb(ref),
                            rgb_image(before),
                            rgb_image(new),
                            heat(a[3], 0.2, a[1]),
                            heat(b[3], 0.2, b[1]),
                            rgb(photos[v]),
                        ],
                    )
                )
            for v in allowed:
                new = outputs.rgb(s, v)
                ref = old.sources[v]
                before = old_source[v]["rgb"] if old_source[v] is not None else None
                valid = np.ones(SHAPE, bool)
                now, b = measurement(ref, new, valid)
                _, a = measurement(ref, before, valid)
                srcstats[v] = {
                    **now,
                    "prediction_sha256": step["views"][v].get("sha256"),
                    "source_sha256": original_source[v],
                }
                source_rows.append(
                    (
                        v,
                        [
                            rgb(ref),
                            rgb_image(before),
                            rgb_image(new),
                            heat(a[3], 0.2, a[1]),
                            None if new is None else heat(b[3], 0.2, b[1]),
                        ],
                    )
                )
            title = f"{req['object_id']}  {req['condition']}  {int(s)} updates  |  "
            for name, head, images in [
                (
                    "target",
                    [
                        "Reference",
                        "Old prediction",
                        "New prediction",
                        "Old RGB MAE",
                        "New RGB MAE",
                        "Closest-axis source",
                    ],
                    target_rows,
                ),
                (
                    "source",
                    ["Original source", "Old self-render", "New self-render", "Old RGB MAE", "New RGB MAE"],
                    source_rows,
                ),
            ]:
                pp = dest / "boards" / f"{name}-{s:04}.png"
                table(pp, head, images, title + name)
                boards[str(pp.relative_to(dest))] = {"bytes": pp.stat().st_size, "sha256": file_hash(pp)}
            steps[str(s)] = {
                "step": s,
                "status": "measured",
                "endpoint_role": (
                    "interim_diagnostic"
                    if frozen and s != frozen["stop_step"]
                    else "frozen_endpoint"
                    if frozen
                    else "development_or_control_curve"
                ),
                "checkpoint_sha256": step["checkpoint_sha256"],
                "phase": step["phase"],
                "gaussians": step["gaussians"],
                "source": srcstats,
                "targets": stats,
                "source_summary": sum_views(list(srcstats.values())),
                "target_summary": sum_views([z["full_crop"] for z in stats.values()]),
                "target_summary_paired_old": sum_views(
                    [z["paired_comparable"]["old"] for z in stats.values()]
                ),
                "target_summary_paired_new": sum_views(
                    [z["paired_comparable"]["new"] for z in stats.values()]
                ),
            }
        doc = {
            "schema": "el-recovery-quality-1",
            "run_id": rid,
            "object_id": req["object_id"],
            "condition": req["condition"],
            "role": "synthetic_control"
            if req["object_id"] == SYNTH
            else ("development_selection" if req["object_id"] == "set1" else "already_exposed_regression"),
            "recipe_id": req["recipe_id"],
            "recipe_sha256": data["recipe_sha256"],
            "protocol_sha256": data["protocol_sha256"],
            "execution_status": data["status"],
            "completed_steps": data["completed_steps"],
            "schedule_steps": data["schedule_steps"],
            "output_reason": data.get("reason") or data.get("failure_reason"),
            "steps": steps,
            "old_and_zero_training_targets": info,
            "old_source": legacy_sources,
            "angular_source_choices": comparators,
            "source_mean_rgb": mean,
            "original_target_hashes": refs,
            "original_source_hashes": original_source,
            "old_render_hashes": old_renders,
            "original_case_sha256": old.case.sha256,
            "old_checkpoint_sha256": old.case.meta["checkpoint_sha256"],
            "old_reconstruction_run_id": old.case.meta["reconstruction_run_id"],
            "boards": boards,
            "input_pins": {
                k: req[k]
                for k in (
                    "protocol",
                    "recipe",
                    "outputs",
                    "original_case",
                    "references",
                    "reference_views",
                    "original_provenance",
                    "freeze",
                )
                if k in req
            },
            "risk_inputs_read": 0,
        }
        write_json(dest / "quality.json", doc)
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for x in ("store", "shelf", "request", "output"):
        p.add_argument("--" + x, required=True, type=Path)
    p.add_argument("--request-sha256", required=True)
    a = p.parse_args()
    j = work((a.store, a.shelf), signed(a.request, a.request_sha256), a.output, os.environ.get("EL_RUN_ID"))
    print(json.dumps({k: j[k] for k in ("object_id", "condition", "execution_status", "completed_steps")}))
