"""Locate projection residuals on the already recorded native support. No rasterization or fitting."""

import argparse
from pathlib import Path
import os

import numpy as np

from evidence_lens._bundle import encoded, load, sha
from evidence_lens._verify import require
from r0_cpu import csvfile, pinned
from r0_report import maps


def fma32(a, b, c):
    """Emulate a single correctly rounded FP32 multiply-add on these finite real-world values."""
    return (np.asarray(a, dtype="f8") * np.asarray(b, dtype="f8") + c).astype("f4")


def pixel_routes(means, cam, native_z):
    pose = np.linalg.inv(np.asarray(cam["c2w"], dtype="f8"))
    k = np.asarray(cam["K"], dtype="f8")
    p64 = means.astype("f8")
    xyz = p64 @ pose[:3, :3].T + pose[:3, 3]
    focal, principal = k.diagonal()[:2], k[:2, 2]
    uv = xyz[:, :2] * focal / xyz[:, 2:3] + principal
    pose32 = pose.astype("f4")
    focal32 = focal.astype("f4")
    edge32 = principal.astype("f4") + np.float32(0.5)
    q = p64 @ pose32[:3, :3].astype("f8").T + pose32[:3, 3].astype("f8")
    quantized = q[:, :2] * focal32.astype("f8") / q[:, 2:3] + edge32.astype("f8") - 0.5

    def project32(local, fused=False):
        rz = np.float32(1) / local[:, 2:3]
        product = local[:, :2] * focal32
        edge = fma32(product, rz, edge32) if fused else product * rz + edge32
        return edge.astype("f8") - 0.5, local[:, 2].astype("f8")

    # gsplat 1.5.3 writes R*p+t and fx*x*(1/z)+cx. BLAS and scalar paths can round differently.
    x, y, z = (means[:, i:i + 1] for i in range(3))
    r0, r1, r2, trans = pose32[:3].T
    separate = ((x * r0 + y * r1) + z * r2) + trans
    contracted = fma32(z, r2, fma32(y, r1, x * r0)) + trans
    return {
        "fp64_reference": (uv, xyz[:, 2]),
        "fp32_camera_only_fp64_arithmetic": (quantized, q[:, 2]),
        "fp64_with_observed_cuda_z": (xyz[:, :2] * focal / native_z[:, None] + principal, native_z),
        "fp32_numpy_matmul": project32(means @ pose32[:3, :3].T + trans),
        "fp32_separate_left_to_right": project32(separate),
        "fp32_contracted_left_to_right": project32(contracted, fused=True),
    }


def run(a):
    cfg = load(a.config)
    core = pinned(a.store, a.shelf, cfg["report"])
    cpu = pinned(a.store, a.shelf, cfg["cpu"])
    ctr = pinned(a.store, a.shelf, cfg["centres"])
    original = load(core.open("config.resolved.yaml"))
    require(core.run["experiment_id"] == "r0-checkpoint-evidence-report"
            and original["cpu"] == cpu.pin and original["centres"] == ctr.pin, "Wrong original projection report")
    limits = original["readback_tolerances"]
    centre_cfg = load(ctr.open("config.resolved.yaml"))
    native_index = load(ctr.open("payload/centres.json"))
    inputs = {s["role"]: s for s in load(ctr.open("inputs.json"))["inputs"]}

    def input_file(role):
        ref = inputs[role]
        f = ctr.open(ref["path"])
        require(len(ref["files"]) == 1 and sha(f) == centre_cfg["roles"][role]["sha256"], "Original input differs")
        return f

    all_rows = []
    by_route = {}
    seen = set()
    threshold = cfg["review_absolute_px"]
    for cell in centre_cfg["cells"]:
        cameras = load(input_file(cell["camera_role"]))["views"]
        for phase, role in cell["phase_roles"].items():
            with np.load(input_file(role), allow_pickle=False) as z:
                means = z["means"]
            for view, cam in cameras.items():
                filename = f"{cell['cell']}-{phase}-{view}.npz"
                entry = native_index["files"][filename]
                require(filename not in seen and sha(ctr.open("payload/" + filename)) == entry["sha256"], "Invalid native entry")
                seen.add(filename)
                n = maps(ctr, filename)
                raw = maps(cpu, "row-diagnostics/" + filename)
                ids = n["index"]
                expected, depth = raw["centre_uv"][ids], raw["camera_z_m"][ids]
                measured = n["means2d_edge"].astype("f8") - 0.5
                cuda_z = n["camera_z_m"].astype("f8")
                inside = raw["centre_inside"][ids]
                require(np.isfinite(expected).all() and (depth > cam["clip_m"][0]).all()
                        and np.all(np.abs(measured - expected) <= limits["centre_absolute_px"] + limits["centre_relative"] * abs(expected))
                        and np.all(np.abs(cuda_z - depth) < limits["z_absolute_m"]), "Original acceptance changed")
                routes = pixel_routes(means[ids], cam, cuda_z)
                uv, z = routes["fp64_reference"]
                require(np.allclose(uv, expected, rtol=0, atol=1e-10) and np.allclose(z, depth, rtol=0, atol=1e-13),
                        "Saved original FP64 reference differs")
                routes["fp64_reference"] = (expected, depth)
                row = {
                    "name": filename[:-4], "phase": phase, "view": view, "near": cam["clip_m"][0],
                    "dims": [cam["width"], cam["height"]], "ids": ids, "inside": inside,
                    "measured": measured, "cuda_z": cuda_z, "routes": routes,
                }
                all_rows.append(row)
                for method, (xy, dz) in routes.items():
                    by_route.setdefault(method, []).append((measured - xy, cuda_z - dz))
    require(seen == set(native_index["files"]) and len(seen) == cfg["view_checkpoints"], "Incomplete roundoff readback")
    flags = np.concatenate([r["inside"] for r in all_rows])
    table = []
    for name, pairs in by_route.items():
        e = np.concatenate([xy for xy, _ in pairs])
        dz = np.concatenate([z for _, z in pairs])
        ae = abs(e)
        table.append({
            "projection": name, "supported_rows": len(e), "inside_rows": int(flags.sum()),
            "absolute_max_px": float(ae.max()), "inside_absolute_max_px": float(ae[flags].max()),
            "median_absolute_axis_px": float(np.median(ae)), "signed_mean_x_px": float(e[:, 0].mean()),
            "signed_mean_y_px": float(e[:, 1].mean()), "rows_above_review_absolute_px": int((ae.max(axis=1) > threshold).sum()),
            "axes_above_review_absolute_px": int((ae > threshold).sum()), "inside_rows_above_review_absolute_px": int((ae[flags].max(axis=1) > threshold).sum()),
            "z_max_absolute_m": float(abs(dz).max()),
        })

    def worst(in_frame):
        maximum = -1
        info = {}
        for row in sorted(all_rows, key=lambda r: r["name"]):
            ref, fp_z = row["routes"]["fp64_reference"]
            err = np.abs(row["measured"] - ref)
            if in_frame:
                err = np.where(row["inside"][:, None], err, -1.0)
            i, axis = np.unravel_index(np.argmax(err), err.shape)
            if err[i, axis] <= maximum:
                continue
            maximum = float(err[i, axis])
            info = {
                "cell_phase_view": row["name"], "point_zero_based": int(row["ids"][i]), "axis": "xy"[axis],
                "inside_frame": bool(row["inside"][i]), "image_width_height": row["dims"], "abs_axis_residual_px": maximum,
                "measured_xy_integer_centres": row["measured"][i].tolist(), "reference_xy_integer_centres": ref[i].tolist(),
                "measured_z_m": float(row["cuda_z"][i]), "reference_z_m": float(fp_z[i]),
                "near_m": row["near"], "reference_z_above_near_m": float(fp_z[i] - row["near"]),
                "original_allowed_axis_residual_px": float(limits["centre_absolute_px"] + limits["centre_relative"] * abs(ref[i, axis])),
                "routes": {n: {"xy_integer_centres": xy[i].tolist(), "z_m": float(z[i]),
                               "abs_axis_residual_px": float(abs(xy[i, axis] - row["measured"][i, axis]))}
                           for n, (xy, z) in row["routes"].items()},
            }
        return info

    csvfile(a.output / "roundoff-routes.csv", table)
    info = {
        "schema": "el-r0-roundoff-readback-1", "run_id": os.environ["EL_RUN_ID"],
        "parents": {k: cfg[k] for k in ("cpu", "report", "centres")}, "view_checkpoints": len(seen),
        "original_readback_tolerances": limits, "review_absolute_px": threshold,
        "support": "Rows with both positive native CUDA radii at each named camera and checkpoint. Frame uses the original FP64 integer-centre grid.",
        "scope": "Post hoc localization of a previously accepted residual, using existing checkpoint and projection arrays.",
        "routes": "Arithmetic variants compare the same coordinates. Substituting measured CUDA Z isolates depth sensitivity. Scalar contraction is a comparison and does not identify the compiled instructions.",
        "global_fp64_witness": worst(False), "in_frame_fp64_witness": worst(True),
        "rows": len(flags), "inside": int(flags.sum()), "route_table_sha256": sha(a.output / "roundoff-routes.csv"),
        "optimizer_steps": 0, "photos_read": 0,
    }
    (a.output / "roundoff.json").write_bytes(encoded(info))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("store", "shelf", "config", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    run(p.parse_args())
