"""Measure source-camera extent of the frozen recovery Gaussian checkpoints on CPU."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from r0_geometry import covariance, project, quantiles

CELLS = {"scene_057_offset_bell_cage/M4": "057", "set1/M2": "set1m2", "set1/M4": "set1m4"}
CHECKPOINTS = (0, 3000, 5000)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def native_log(trace, sid, checkpoint):
    candidates = [r for r in trace if r["source"] == sid]
    # Training logs radii from the rasterization that precedes the logged Adam update.
    entry = min(candidates, key=lambda r: (abs((r["step"] - 1) - checkpoint), r["step"]))
    return {
        "native_forward_logged_update": entry["step"],
        "native_forward_state_step": entry["step"] - 1,
        "native_state_step_minus_checkpoint": entry["step"] - 1 - checkpoint,
        "native_gsplat_max_radius_px": entry["max_radius_px"],
        "native_visible": entry["visible"],
    }


def inspect_state(tensors, camera, scales, cov, eps):
    d = project(tensors, cov, camera, eps)
    z, axis, uv = d["camera_z_m"], d["semiaxes_3sigma_px"][:, 0], d["centre_uv"]
    summary = {
        "camera_near_m": camera["clip_m"][0],
        "camera_far_m": camera["clip_m"][1],
        "N": len(z),
        "behind_or_at_camera": int((z <= 0).sum()),
        "positive_at_or_before_near": int(((z > 0) & (z <= camera["clip_m"][0])).sum()),
        **{"z_all_m_" + k: v for k, v in quantiles(z).items()},
    }
    extreme = []
    for key, mask in (
        ("in_clip", d["inside_clip"]),
        ("center_inside", d["centre_inside"]),
        ("ellipse_intersects", d["ellipse_intersects"]),
    ):
        indexes = np.flatnonzero(mask)
        radii = axis[indexes]
        summary.update(
            {
                key + "_n": len(indexes),
                key + "_major_gt64": int((radii > 64).sum()),
                key + "_major_gt256": int((radii > 256).sum()),
                key + "_major_px_p99": float(np.quantile(radii, 0.99)) if len(radii) else None,
                key + "_major_px_max": float(radii.max()) if len(radii) else None,
            }
        )
        if len(indexes):
            i = int(indexes[np.argmax(radii)])
            extreme.append(
                {
                    "domain": key,
                    "checkpoint_row": i,
                    "analytic_major_3sigma_px": float(axis[i]),
                    "camera_z_m": float(z[i]),
                    "center_x_px": float(uv[i, 0]),
                    "center_y_px": float(uv[i, 1]),
                    **{"scale_" + a + "_m": float(scales[i, j]) for j, a in enumerate("xyz")},
                    "anisotropy": float(scales[i].max() / scales[i].min()),
                    "opacity": float(np.exp(-np.logaddexp(0.0, -tensors["opacity_logits"][i]))),
                    "center_inside": bool(d["centre_inside"][i]),
                    "ellipse_intersects": bool(d["ellipse_intersects"][i]),
                }
            )
    return summary, extreme


def worker(config, args, out):
    import torch

    cfg = read(config)
    checkpoints = tuple(cfg.get("checkpoints", CHECKPOINTS))
    check(checkpoints in (CHECKPOINTS, (3000, 5000)), "Unregistered checkpoint set")
    bindings = dict(pair.split("=", 1) for pair in args)
    expected = {role for cell in cfg["cells"] for role in cell["members"]}
    check(set(bindings) == expected and len(args) == len(expected), "Incorrect diagnostic input roles")
    rows, states, examples = [], [], []
    for cell in cfg["cells"]:
        name = cell["object_id"] + "/" + cell["condition"]
        check(name in CELLS and CELLS[name] == cell["short"], "Cell outside registered diagnostic")
        src = {}
        for role, spec in cell["members"].items():
            path = Path(bindings[role])
            check(path.stat().st_size == spec["bytes"] and digest(path) == spec["sha256"], "Changed " + role)
            src[role.removeprefix(cell["short"] + "-")] = path
        cams = read(src["camera"])
        ids = ["s00", "s03"] if cell["condition"] == "M2" else ["s00", "s01", "s02", "s03"]
        check(
            cams["unit"] == "m" and cams["pixel_center"] == "integer" and set(cams["views"]) == set(ids),
            "Expected source-only metric cameras",
        )
        trace = [json.loads(line) for line in src["trace"].read_text().splitlines()]
        check(
            all(x["step"] == i and x["source"] == ids[(i - 1) % len(ids)] for i, x in enumerate(trace, 1)),
            "Unexpected native trace or source order",
        )
        for step in checkpoints:
            ck = torch.load(src["p" + str(step)], map_location="cpu", weights_only=True)
            ident = ck["identity"]
            check(
                ck["schema"] == "el-recovery-checkpoint-1"
                and ck["producer_run_id"] == cell["generation"]["run_id"]
                and ck["step"] == step
                and ck["horizon"] == 5000
                and ident["object_id"] == cell["object_id"]
                and ident["condition"] == cell["condition"]
                and ident["source_camera_sha256"] == digest(src["camera"])
                and ident["recipe_sha256"] == cell["recipe_sha256"]
                and ident["generator_source_ids"] == ids,
                "Checkpoint identity, source cameras or step differs",
            )
            arrays = {}
            for k, arg, tail in (
                ("means", "means", (3,)),
                ("log_scales", "scales", (3,)),
                ("quats", "quats", (4,)),
                ("opacity_logits", "opacities", ()),
            ):
                t = ck["params"][arg]
                check(
                    t.device.type == "cpu"
                    and t.dtype == torch.float32
                    and tuple(t.shape[1:]) == tail
                    and torch.isfinite(t).all().item(),
                    "Invalid Gaussian " + arg,
                )
                arrays[k] = t.detach().numpy()
            n = len(arrays["means"])
            check(
                all(len(a) == n for a in arrays.values()) and n == cell["counts"][str(step)],
                "Wrong population",
            )
            scale = np.exp(arrays["log_scales"].astype("f8"))
            check(np.isfinite(scale).all() and (scale > 0).all(), "Nonfinite scale")
            cov = covariance(arrays)
            ident_row = {
                "object_id": cell["object_id"],
                "condition": cell["condition"],
                "step": step,
                "phase": ck["phase"],
                "checkpoint_sha256": digest(src["p" + str(step)]),
            }
            stats = {
                "all_axis_scale_m": scale.ravel(),
                "row_largest_scale_m": scale.max(axis=1),
                "row_smallest_scale_m": scale.min(axis=1),
                "row_anisotropy": scale.max(axis=1) / scale.min(axis=1),
            }
            states.append(
                {
                    **ident_row,
                    "N": n,
                    **{field + "_" + q: v for field, a in stats.items() for q, v in quantiles(a).items()},
                }
            )
            for sid in ids:
                cam = cams["views"][sid]
                check(cam["width"] == cam["height"] == 256, "Unexpected source resolution")
                statistics, extreme = inspect_state(arrays, cam, scale, cov, ck["recipe"]["eps2d"])
                which = {**ident_row, "source_id": sid}
                rows.append({**which, **statistics, **native_log(trace, sid, step)})
                examples.extend({**which, **r} for r in extreme)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "source-projection.csv", rows)
    write_csv(out / "checkpoint-scales.csv", states)
    write_csv(out / "extreme-points.csv", examples)
    save(
        out / "projection.json",
        {
            "schema": "el-recovery-source-extent-1",
            "run_id": os.environ["EL_RUN_ID"],
            "checkpoints": list(checkpoints),
            "inputs": cfg["cells"],
            "sources": sum(len(c["members"]) for c in cfg["cells"]),
            "optimizer_updates_executed": 0,
            "projection": "FP64 pinhole covariance Jacobian and unconstrained local 3-sigma major semiaxis using the original epsilon. Counts use strict >64 and >256 pixels in each named domain. The source image is 256 x 256.",
            "timing": "The native maximum and visible count came from the nearest same-source training draw. Its row gives the actual pre-update state step and signed offset from the checkpoint.",
            "interpretation": "The CPU major semiaxis and gsplat's rasterizer support radius have different definitions. Native per-point radii and compositing weights were unrecorded for these fits. Checkpoint row indices identify only their own saved tensors.",
            "native_arrays_read": "Source cameras, three guarded CPU checkpoints and source optimizer trace per condition. No image files.",
            "files": [
                {"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
                for p in sorted(out.glob("*.csv"))
            ],
        },
    )


def record(args):
    from evidence_lens._bundle import asset_path
    from evidence_lens.formal_inputs import PinnedRun

    project = Path(__file__).resolve().parents[1]
    store = args.store.resolve()
    cells, paths, parents = [], {}, []
    checkpoints = (3000, 5000) if args.late_only else CHECKPOINTS
    for definition in args.cell:
        name, rid, manifest = definition.split("=")
        check(
            name in CELLS and name not in [c["object_id"] + "/" + c["condition"] for c in cells],
            "Use each approved source cell once",
        )
        obj, cond = name.split("/")
        pin = {"run_id": rid, "manifest_sha256": manifest}
        run = PinnedRun(store, pin)
        check(
            run.run["execution_status"] == "succeeded"
            and run.run["run_kind"] == "generation"
            and run.run["experiment_id"] == "reconstruction-recovery-v1",
            "Expected completed scientific recovery fit",
        )
        gen = read(run.open("payload/reconstruction.json"))
        ident = gen["identity"]
        check(
            (ident["object_id"], ident["condition"]) == (obj, cond)
            and gen["run_id"] == rid
            and gen["last_completed_step"] == gen["schedule_horizon"] == 5000,
            "Generation does not match the named diagnostic cell",
        )
        inputs = read(run.open("inputs.json"))
        c_input = next(r for r in inputs["inputs"] if r["role"] == "cameras")
        check(
            c_input["storage"] == "shared"
            and len(c_input["files"]) == 1
            and c_input["files"][0]["sha256"] == ident["source_camera_sha256"],
            "Expected identical shared cameras",
        )
        short = CELLS[name]
        items = {
            "camera": asset_path(store, ident["source_camera_sha256"]),
            "trace": run.open("payload/steps.jsonl"),
        }
        for step in checkpoints:
            issued = gen["checkpoints"][str(step)]
            check(
                issued["path"] == f"checkpoints/step-{step:04d}.pt" and issued["producer_run_id"] == rid,
                "Diagnostic requires original issued checkpoint",
            )
            path = run.open("payload/" + issued["path"])
            check(digest(path) == issued["checkpoint_sha256"], "Changed issued checkpoint")
            items["p" + str(step)] = path
        members = {}
        for role, path in items.items():
            full = short + "-" + role
            paths[full] = (path, rid)
            members[full] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        check(
            members[short + "-camera"]["sha256"] == ident["source_camera_sha256"]
            and members[short + "-trace"]["sha256"] == gen["trace_sha256"],
            "Camera or trace differs from frozen fit",
        )
        cells.append(
            {
                "object_id": obj,
                "condition": cond,
                "short": short,
                "recipe_sha256": ident["recipe_sha256"],
                "generation": pin,
                "members": members,
                "counts": {str(s): gen["checkpoints"][str(s)]["gaussians"] for s in checkpoints},
            }
        )
        parents.append(rid + "=" + manifest)
    check(
        {c["short"] for c in cells} in ({"057"}, {"set1m2", "set1m4"}),
        "Use the control or the two original set1 conditions",
    )
    recipes = {c["recipe_sha256"] for c in cells}
    check(len(recipes) == 1, "Use a single registered recipe for each diagnostic")
    recipe_prefix = next(iter(recipes))[:12]
    for existing in sorted((store / "runs").glob("*/payload/projection.json")):
        label = read(existing.parent.parent / "run.json")
        if label["experiment_id"] != "recovery-source-extent-readback-v1" or set(label["parent_run_ids"]) != {
            c["generation"]["run_id"] for c in cells
        }:
            continue
        prev = read(existing)
        if prev.get("schema") != "el-recovery-source-extent-1":
            continue
        if prev["inputs"] == cells and tuple(prev.get("checkpoints", CHECKPOINTS)) == checkpoints:
            old = PinnedRun(
                store,
                {
                    "run_id": existing.parent.parent.name,
                    "manifest_sha256": digest(existing.parent.parent / "manifest.json"),
                },
            )
            check(
                old.run["execution_status"] == "succeeded"
                and old.open("payload/projection.json") == existing,
                "Cannot reuse a changed or unsuccessful diagnostic",
            )
            for item in prev["files"]:
                check(
                    digest(old.open("payload/" + item["path"])) == item["sha256"],
                    "Changed prior numerical table",
                )
            print(json.dumps({"status": "reused", **old.pin, "projection_sha256": digest(existing)}))
            return 0
    with tempfile.TemporaryDirectory(prefix="extent-", dir=store / "work") as td:
        temp = Path(td)
        engine = temp / "source/scripts"
        engine.mkdir(parents=True)
        for f in ("recovery_extent_probe.py", "r0_geometry.py"):
            shutil.copyfile(project / "scripts" / f, engine / f)
        cfg = {
            "schema": "el-recovery-source-extent-job-1",
            "experiment_id": "recovery-source-extent-readback-v1",
            "trial_id": "r-source-extent-" + "-".join(c["short"] for c in cells) + "-" + recipe_prefix,
            "cells": cells,
            "checkpoints": list(checkpoints),
            "recording": {"expected_output_bytes": 2 * 2**20},
        }
        save(temp / "config.json", cfg)
        cmd = [
            sys.executable,
            "-m",
            "evidence_lens.recorder",
            "--root",
            str(store),
            "--kind",
            "archive_verify",
            "--config",
            str(temp / "config.json"),
            "--source-root",
            str(engine.parent),
            "--storage-policy",
            str(project / "configs/storage.json"),
        ]
        for p in parents:
            cmd += ["--parent", p]
        for role, (path, rid) in paths.items():
            cmd += ["--shared-input", role + "=" + str(path), "--input-parent", role + "=" + rid]
        cmd += [
            "--",
            str(args.cpu_python.absolute()),
            "-B",
            "{source}/scripts/recovery_extent_probe.py",
            "worker",
            "--config",
            "{config}",
            "--out",
            "{out}",
        ]
        for role in paths:
            cmd += ["--input", role + "={input:" + role + "}"]
        return subprocess.run(cmd).returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    r = sub.add_parser("record")
    r.add_argument("--store", type=Path, required=True)
    r.add_argument("--cpu-python", type=Path, default=Path(".venv-vision-cpu/bin/python"))
    r.add_argument(
        "--late-only", action="store_true", help="Compare the two late checkpoints of the scale-decay recipe."
    )
    r.add_argument(
        "--cell", action="append", required=True, metavar="OBJECT/M2_OR_M4=GENERATION_ID=MANIFEST_SHA"
    )
    w = sub.add_parser("worker")
    w.add_argument("--config", required=True, type=Path)
    w.add_argument("--out", required=True, type=Path)
    w.add_argument("--input", action="append", required=True)
    cli = parser.parse_args()
    if cli.action == "record":
        sys.exit(record(cli))
    worker(cli.config, cli.input, cli.out)
