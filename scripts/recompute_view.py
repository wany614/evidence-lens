"""Independently recompute P2 labels, denominators and sampled camera/L0 decisions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def expected_reason(x, y, target, source, source_cam, target_cam, depth_ok):
    a, z = target["alpha"][y,x], target["z"][y,x]
    if not np.isfinite(a) or not 0 <= a <= 1:
        return 1
    if a < 0.5:
        return 2
    if not depth_ok:
        return 3
    if not np.isfinite(z) or z <= 0:
        return 4
    camera_point = np.linalg.solve(target_cam["K"], [x, y, 1.0])*float(z)
    world = np.asarray(target_cam["c2w"]) @ np.append(camera_point, 1.)
    p = np.linalg.solve(source_cam["c2w"], world)[:3]
    if p[2] <= 0:
        return 5
    projection = np.asarray(source_cam["K"]) @ p
    qx, qy = np.floor(projection[:2]/projection[2] + 0.5).astype(int)
    if not (0 <= qx < source_cam["width"] and 0 <= qy < source_cam["height"]):
        return 5
    a0, z0 = source["alpha"][qy,qx], source["z"][qy,qx]
    if not np.isfinite(a0) or not 0 <= a0 <= 1:
        return 6
    if a0 < 0.5:
        return 7
    if not np.isfinite(z0) or z0 <= 0:
        return 8
    return 9 if np.float32(max(0., p[2]-z0)/max(p[2],1.e-6)) > np.float32(0.10) else 0


def arrays(path, names):
    with np.load(path, allow_pickle=False) as data:
        return {n:data[n] for n in names}


def check(a):
    cameras = json.loads(a.cameras.read_text(encoding="utf8"))["views"]
    meta = json.loads(a.metadata.read_text(encoding="utf8"))
    depth_ok = all(meta["depth_validation"][k] for k in ("off_axis_plane_passed","alpha_normalization_passed"))
    src = arrays(a.source_prediction, ("z","alpha"))
    target = arrays(a.prediction, ("rgb","z","alpha"))
    ref = arrays(a.reference, ("rgb", "z", "position"))
    score = arrays(a.score, ("l0","l0_reason"))
    labels = arrays(a.labels, ("rgb_error", "rgb_high_error", "depth_evaluable",
                              "signed_depth_error", "depth_high_error"))
    v = np.isfinite(ref["rgb"]).all(-1)
    e = v & np.isfinite(target["rgb"]).all(-1)
    error = np.full(v.shape, np.nan, np.float32)
    error[e] = np.abs(ref["rgb"][e] - target["rgb"][e]).mean(axis=1)
    y = e & (error > np.float32(0.05))
    if not np.array_equal(labels["rgb_error"], error, equal_nan=True):
        raise ValueError("stored continuous RGB error differs")
    if not np.array_equal(labels["rgb_high_error"], np.where(e, y, 255).astype(np.uint8)):
        raise ValueError("stored label or missing code differs")
    valid_surface = np.isfinite(ref["position"]).all(-1) & np.isfinite(ref["z"]) & (ref["z"] > 0)
    measured_depth = (valid_surface & depth_ok & np.isfinite(target["z"]) & (target["z"] > 0)
                      & np.isfinite(target["alpha"]) & (target["alpha"] >= 0.5) & (target["alpha"] <= 1))
    signed_depth = np.full(v.shape, np.nan, np.float32)
    signed_depth[measured_depth] = target["z"][measured_depth] - ref["z"][measured_depth]
    tolerance_m = np.maximum(np.float32(0.02), np.float32(0.02) * ref["z"])
    depth_error = measured_depth & (np.abs(signed_depth) > tolerance_m)
    if (not np.array_equal(labels["depth_evaluable"], measured_depth)
            or not np.array_equal(labels["signed_depth_error"], signed_depth, equal_nan=True)
            or not np.array_equal(labels["depth_high_error"], np.where(measured_depth, depth_error, 255))):
        raise ValueError("stored camera-Z mask, signed error or tolerance label differs")
    reasons = score["l0_reason"]
    expected_l0 = np.full(v.shape, np.nan, dtype=np.float32)
    expected_l0[reasons == 0] = 0
    expected_l0[np.isin(reasons, (2, 5, 7, 9))] = 1
    if np.any(reasons > 9) or not np.array_equal(score["l0"], expected_l0, equal_nan=True):
        raise ValueError("stored L0 risk differs from the declared reason mapping")
    high = e & (score["l0"] == 1)
    low = e & (score["l0"] == 0)
    unknown = e & ~np.isfinite(score["l0"])
    if (high | low | unknown).sum() != e.sum():
        raise ValueError("incomplete score partition")
    masks = {
        ("output","prediction_valid_fraction"):(e,v),
        ("output","invalid_output_fraction"):(v & ~e,v),
        ("output","rgb_error_fraction_gt_0.05"):(y,e),
        ("output","depth_error_fraction"):(depth_error,measured_depth),
        ("L0","high_risk_recall"):(high & y,y), ("L0","warning_precision"):(high & y,high),
        ("L0","low_risk_error_rate"):(low & y,low), ("L0","overall_low_risk_coverage"):(low,v),
        ("L0","unknown_fraction"):(unknown,e)}
    with a.metrics.open(encoding="utf8",newline="") as f:
        rows = {}
        for row in csv.DictReader(f):
            if row["region_name"] == "all" and row["camera_id"] == a.view:
                key = row["method_id"], row["metric_name"]
                if key in rows:
                    raise ValueError(f"duplicate metric for fixed view: {key}")
                rows[key] = row
    comparisons = {}
    for key, (num, den) in masks.items():
        n, d = int(num.sum()), int(den.sum())
        row = rows[key]
        if float(row["numerator"]) != n or int(row["denominator"]) != d:
            raise ValueError(f"numerator/denominator mismatch: {key}")
        if (d and abs(float(row["value"])-n/d)>1e-12) or (not d and row["value"]):
            raise ValueError(f"undefined or numeric ratio mismatch: {key}")
        comparisons["/".join(key)] = [n,d]
    h, w = v.shape
    checked = 0
    for y0 in np.linspace(0,h-1,min(h,17),dtype=int):
        for x0 in np.linspace(0,w-1,min(w,17),dtype=int):
            reason = expected_reason(x0,y0,target,src,cameras["source"],cameras[a.view],depth_ok)
            if reason != int(score["l0_reason"][y0,x0]):
                raise ValueError(f"scalar L0 differs at {(int(x0), int(y0))}")
            checked += 1
    return {"view":a.view, "scalar_l0_witnesses":checked, "pixel_labels_checked":int(v.size),
            "signed_camera_z_pixels":int(measured_depth.sum()),
            "ratios_numerator_denominator":comparisons, "passed":True}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--view",default="v01")
    for name in ("cameras","metadata","source-prediction","prediction","reference","score","labels","metrics","out"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    result=check(args)
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/"independent-view-check.json").open("x",encoding="utf8") as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write("\n")
