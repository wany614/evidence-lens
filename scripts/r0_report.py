"""Independent quantitative checks of R0 CPU geometry, native CUDA measurements and original D1 errors."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil

import numpy as np
from evidence_lens._bundle import encoded, load, sha
from evidence_lens.formal_inputs import require
from evidence_lens import stage3_gaussians as g
from r0_cpu import csvfile, pinned
from r0_geometry import quantiles


def read_csv(path):
    with path.open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def maps(run, path):
    with np.load(run.open("payload/" + path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def count(x):
    return int(np.count_nonzero(x))


def run(a):
    cfg = load(a.config)
    cpu = pinned(a.store, a.shelf, cfg["cpu"])
    core = load(cpu.open("payload/r0-cpu.json"))
    ctr = pinned(a.store, a.shelf, cfg["centres"])
    centres = load(ctr.open("payload/centres.json"))
    require(
        centres["run_id"] == ctr.pin["run_id"]
        and centres["optimization_steps"] == 0
        and centres["photographs"] == 0,
        "Centres readback does not identify zero-fit geometry",
    )
    out = a.output
    pub = out / "public"
    pub.mkdir(exist_ok=True)
    checks = []
    flat = []
    arms = []
    per_gt = []
    history = []
    synth = {}
    centres_seen = set()
    for n, d in core["public"].items():
        src = cpu.open("payload/public/" + n)
        require(sha(src) == d, "CPU summary SHA mismatch")
        shutil.copyfile(src, pub / n)
    quality = read_csv(pub / "image-quality.csv")
    for key, d in core["arms"].items():
        o, co = key.split("-")
        ids = g.SETS[co]
        runpin = cfg["gpu"][key]
        gpu = pinned(a.store, a.shelf, runpin)
        report = load(gpu.open("payload/r0-gpu.json"))
        gen = pinned(a.store, a.shelf, d["parents"]["generation"])
        evalrun = pinned(a.store, a.shelf, cfg["evaluations"][key])
        ev = load(evalrun.open("payload/evaluation.json"))
        require(
            evalrun.run["run_kind"] == "evaluation" and ev["object_id"] == o and ev["condition"] == co,
            "Wrong old D1 evaluation",
        )
        for v in g.TARGETS:
            q = next(x for x in quality if x["object"] == o and x["condition"] == co and x["view"] == v)
            vrow = next(x for x in ev["rows"] if x["view"] == v)
            ref = vrow["domains"]["full_crop"]["MS-L1"]["counts"]
            require(
                int(q["crop_pixels"]) == ref["E"] == ref["V"]
                and int(q["pixels_strict_mae_gt_0_05"]) == ref["Y"]
                and abs(float(q["mean_rgb_mae"]) - ref["rgb_mae_sum"] / ref["E"]) < 3e-11,
                "D1 RGB readback differs",
            )
            per_gt.append(
                {
                    "object": o,
                    "condition": co,
                    "view": v,
                    "E": ref["E"],
                    "Y": ref["Y"],
                    "mean_rgb_mae": float(q["mean_rgb_mae"]),
                    "old_run_id": evalrun.pin["run_id"],
                    "exact_high_error_count": True,
                }
            )
        require(
            report["optimizer_steps"] == 0
            and report["run_id"] == gpu.pin["run_id"]
            and report["object"] == o
            and report["condition"] == co
            and report["parents"]["generation"] == d["parents"]["generation"]
            and report["camera_sha256"] == d["render_camera_sha256"]
            and report["task_gradient_parameters"] == list(g.PARAMS),
            "R0 native run has different original source",
        )
        histfirst = json.loads(gen.open("payload/optimizer.jsonl").read_text().splitlines()[0])
        require(histfirst["source"] == ids[0] and histfirst["update"] == 1, "Wrong historical first loss")
        with np.load(cpu.open("payload/row-diagnostics/" + key + "-deltas.npz"), allow_pickle=False) as delta:
            changed = delta["any_change"]
        for phase, ph in report["checkpoints"].items():
            expected = d["initial_parameter_sha256" if phase == "initial" else "final_parameter_sha256"]
            require(
                ph["parameter_sha256"] == ph["after_probe_parameter_sha256"] == expected,
                "R0 marker or original arrays changed",
            )
            masks = {
                k: np.zeros(4096, bool)
                for k in (
                    "radius",
                    "tile",
                    "weight",
                    "weight_gt_1e_minus12",
                    "task",
                    "task_gt_1e_minus12",
                    *g.PARAMS,
                )
            }
            for v, entry in ph["views"].items():
                path = entry["file"]["path"]
                arr = maps(gpu, path)
                require(
                    sha(gpu.open("payload/" + path)) == entry["file"]["sha256"]
                    and set(arr)
                    == (
                        {"radii", "tiles", "weight", "task_row_l2"}
                        if v in ids
                        else {"radii", "tiles", "weight"}
                    ),
                    "Wrong probe file inventory",
                )
                radius = (arr["radii"] > 0).all(axis=1)
                tile = arr["tiles"] > 0
                w = arr["weight"]
                require(
                    w.shape == tile.shape == radius.shape == (4096,)
                    and np.isfinite(w).all()
                    and np.all(w >= 0)
                    and count(radius) == entry["positive_radius"]
                    and count(tile) == entry["tiled_rows"]
                    and count(w > 0) == entry["colour_weight_rows"]["0.0"]
                    and count(w > 1e-12) == entry["colour_weight_rows"]["1e-12"]
                    and int(arr["tiles"].sum()) == entry["intersected_tiles"]
                    and np.all(~(w > 0) | tile),
                    "Wrong per-row raster support",
                )
                total = float(np.sum(w, dtype=np.float64))
                alpha = entry["alpha_sum"]
                require(
                    abs(total - entry["colour_weight_sum"]) < 1e-7
                    and abs(alpha - total) <= max(0.05, 2e-4 * alpha),
                    "Unverified transmittance weights",
                )
                delta_alpha = None
                if phase == "final":
                    old = next(q for q in quality if (q["object"], q["condition"], q["view"]) == (o, co, v))
                    delta_alpha = abs(float(old["alpha_sum"]) - alpha)
                    require(
                        delta_alpha <= max(0.1, 2e-5 * alpha),
                        "Old renderer opacity differs from native R0 readback",
                    )
                task = None
                task_12 = None
                loss = None
                if v in ids:
                    norms = arr["task_row_l2"]
                    require(
                        norms.shape == (4096, len(g.PARAMS))
                        and np.isfinite(norms).all()
                        and np.all(norms >= 0),
                        "Wrong task grads",
                    )
                    for i, k in enumerate(g.PARAMS):
                        nn = count(norms[:, i] > 0)
                        signed = entry["photometric"]["groups"][k]
                        mag = float(np.linalg.norm(norms[:, i]))
                        require(
                            nn == signed["row_counts"]["0.0"]
                            and abs(mag - signed["whole_group_l2"]) <= max(2e-14, mag * 1e-10),
                            "Task grad re-read differs",
                        )
                        masks[k] |= norms[:, i] > 0
                        if phase == "initial" and v == ids[0]:
                            before = float(histfirst["gradient_l2"][k])
                            err = abs(mag - before)
                            require(err <= max(5e-8, 2e-4 * before), "First historical backward norm differs")
                            history.append(
                                {
                                    "object": o,
                                    "condition": co,
                                    "first_source": v,
                                    "parameter": k,
                                    "historical_group_l2": before,
                                    "same_checkpoint_probe_l2": mag,
                                    "abs_difference": err,
                                }
                            )
                    anytask = norms.max(axis=1)
                    task = count(anytask > 0)
                    task_12 = count(anytask > 1e-12)
                    loss = entry["photometric"]["loss_display_mae"]
                    require(
                        task == entry["photometric"]["nonzero_any_rows"]["0.0"]
                        and task_12 == entry["photometric"]["nonzero_any_rows"]["1e-12"],
                        "Whole-source task count differs",
                    )
                    for k, m in [
                        ("radius", radius),
                        ("tile", tile),
                        ("weight", w > 0),
                        ("weight_gt_1e_minus12", w > 1e-12),
                        ("task", anytask > 0),
                        ("task_gt_1e_minus12", anytask > 1e-12),
                    ]:
                        masks[k] |= m
                sorted_mass = np.cumsum(np.sort(w.astype("f8"))[::-1])
                native_size = arr["radii"][radius].max(axis=1)
                flat.append(
                    {
                        "object": o,
                        "condition": co,
                        "phase": phase,
                        "view": v,
                        "role": "source" if v in ids else "target",
                        "radii_positive": count(radius),
                        "rows_assigned_tiles": count(tile),
                        "gaussian_tile_pairs": int(arr["tiles"].sum()),
                        "rows_positive_weight": count(w > 0),
                        "rows_weight_gt_1e_minus12": count(w > 1e-12),
                        "native_bounds_wider_or_taller_than_grid": count(
                            (2 * arr["radii"] > 256).any(axis=1)
                        ),
                        **{
                            "positive_native_radius_max_axis_px_" + k: n
                            for k, n in quantiles(native_size).items()
                        },
                        "rows_for_half_the_weight": int(np.searchsorted(sorted_mass, 0.5 * total) + 1)
                        if total
                        else None,
                        "rows_for_90pct_weight": int(np.searchsorted(sorted_mass, 0.9 * total) + 1)
                        if total
                        else None,
                        "largest_weight_share": float(np.max(w) / total) if total else None,
                        "sum_colour_weights": total,
                        "sum_alpha": alpha,
                        "mass_readback_abs": abs(total - alpha),
                        "final_abs_diff_original_alpha_sum": delta_alpha,
                        "source_task_nonzero_rows": task,
                        "source_task_gt_1e_minus12_rows": task_12,
                        "source_task_mae": loss,
                    }
                )
                name = f"{key}-{phase}-{v}.npz"
                checks.append(parity(cpu, ctr, centres, name, arr["radii"]))
                centres_seen.add(name)
            arms.append(
                {
                    "object": o,
                    "condition": co,
                    "phase": phase,
                    "all_rows": 4096,
                    "any_parameter_changed": count(changed),
                    **{("source_union_" + k): count(m) for k, m in masks.items()},
                    "changed_and_source_weight": count(changed & masks["weight"]),
                    "changed_without_source_weight_here": count(changed & ~masks["weight"]),
                    "unchanged_with_source_weight_here": count(~changed & masks["weight"]),
                    "changed_and_source_task_here": count(changed & masks["task"]),
                }
            )
        if co not in synth:
            syn = pinned(a.store, a.shelf, d["parents"]["synthetic"])
            rec = load(syn.open(f"payload/CASE/{co}/reconstruction.json"))
            pred = load(syn.open(f"payload/CASE/{co}/prediction.json"))
            require(
                rec["initial_parameter_sha256"] == core["same_initial_parameter_sha256"]
                and rec["initial_checkpoint_sha256"] == d["initial_file_sha256"]
                and pred["checkpoint_sha256"] == rec["checkpoint_sha256"] != rec["initial_checkpoint_sha256"],
                "Original synthetic checkpoint identity differs",
            )
            synth[co] = {
                "recorded_initial_source_mae": rec["loss_initial_by_source"],
                "original_initial_render_grid_sha256": None,
                "original_final_render_checkpoint_sha256": pred["checkpoint_sha256"],
                "source_camera_sha256": rec["source_camera_sha256"],
            }
            for v in ids:
                name = f"StageC-057-{co}-initial-{v}.npz"
                checks.append(parity(cpu, ctr, centres, name, None))
                centres_seen.add(name)
    require(centres_seen == set(centres["files"]), "Wrong native projection roster")
    for n, rows in [
        ("native-views", flat),
        ("native-source-union", arms),
        ("projection-parity", checks),
        ("original-target-error-parity", per_gt),
        ("first-step-group-gradient-parity", history),
    ]:
        csvfile(pub / (n + ".csv"), rows)
    summary = {
        "schema": "el-r0-evidence-report-1",
        "run_id": os.environ["EL_RUN_ID"],
        "cpu": cfg["cpu"],
        "gpu": cfg["gpu"],
        "centres": cfg["centres"],
        "original_057": synth,
        "original_evaluations": cfg["evaluations"],
        "original_initial_parameter_sha256": core["same_initial_parameter_sha256"],
        "raw_initial_arrays_reproduced": core["rng_regenerated_all_raw_tensors_bitwise"],
        "new_gaussian_or_risk_training_steps": 0,
        "source_backward_views": sum(r["role"] == "source" for r in flat),
        "native_all_view_checkpoints": len(flat),
        "native_camera_checks": len(checks),
        "native_supported_rows_across_checks": sum(r["valid_backend_rows"] for r in checks),
        "native_centre_max_abs_px": max(r["centre_max_abs_px"] for r in checks),
        "native_camera_z_max_abs_m": max(r["z_max_abs_m"] for r in checks),
        "first_historical_backward_groups": len(history),
        "old_individual_gaussian_gradient_by_step": None,
        "private_boards": core["private_boards"],
        "independent_d1_reference_views": len(per_gt),
        "files": {f.name: {"sha256": sha(f), "bytes": f.stat().st_size} for f in sorted(pub.iterdir())},
    }
    (pub / "summary.json").write_bytes(encoded(summary))


def parity(cpu, ctr, centres, name, native_radii):
    sp = centres["files"][name]
    path = ctr.open("payload/" + name)
    require(sha(path) == sp["sha256"], "Native projection file changed")
    native = maps(ctr, name)
    fp = maps(cpu, "row-diagnostics/" + name)
    ids = native["index"]
    require(
        np.all(np.diff(ids) > 0) and len(ids) == sp["rows"] and np.all(fp["inside_clip"][ids]),
        "Unexpected native support indices",
    )
    if native_radii is not None:
        require(
            np.array_equal(ids, np.flatnonzero((native_radii > 0).all(axis=1)))
            and np.array_equal(native["radii"], native_radii[ids]),
            "Fused projection and actual tile pipeline radius differs",
        )
    xy = native["means2d_edge"].astype("f8") - 0.5
    expected = fp["centre_uv"][ids]
    err = abs(xy - expected)
    zd = abs(native["camera_z_m"].astype("f8") - fp["camera_z_m"][ids])
    require(
        np.all(err <= 0.003 + 5e-6 * abs(expected)) and np.all(zd < 1e-5),
        "Analytic CPU and independent CUDA camera disagree",
    )
    return {
        "original_camera_and_phase": name[:-4],
        "valid_backend_rows": len(ids),
        "centre_max_abs_px": float(err.max()),
        "centre_abs_p99_px": float(np.quantile(err, 0.99)),
        "z_max_abs_m": float(zd.max()),
        "z_abs_p99_m": float(np.quantile(zd, 0.99)),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("config", "store", "shelf", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    run(p.parse_args())
