"""Compare production crop, ray and feature geometry with independent analytic observations."""

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path

import numpy as np

from evidence_lens.multisource import camera_rays, extract_ordered
from evidence_lens.scoring_io import Camera
from evidence_lens_bundle.grids import PROFILE_SIZES, build_grids, validate_grids


def run(registration):
    rows = []
    for profile, (w, h) in PROFILE_SIZES.items():
        ow, oh = 977, 673
        fx, fy, cx, cy = 1029.25, 897.125, 441.6, 364.125
        k = [[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]
        original = {"width": ow, "height": oh, "K": k, "distortion": "none"}
        grids = build_grids(profile, {"camera": original})
        view = grids["views"]["camera"]
        kw = np.array(view["analysis_K"])
        validate_grids(grids, {"views": {"camera":{"width":w,"height":h,"K":kw.tolist()}}}, ["camera"])
        # Derive test rays from raw optics and the exact integer crop, independently of the reported affine.
        scale = min(Fraction(ow, w), Fraction(oh, h))
        cw, ch = [int(Fraction(n) * scale + Fraction(1, 2)) for n in (w, h)]
        left, top = (ow-cw)//2, (oh-ch)//2
        if view["crop_xywh"] != [left, top, cw, ch]:
            raise ValueError("centre crop differs from rational reference")
        pose = np.eye(4)
        pose[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        pose[:3, 3] = [1.6, -0.2, 0.3]
        cam = Camera(w, h, kw, pose)
        yy, xx = np.indices((h, w), dtype=np.float64)
        ox = left + (xx + 0.5) * cw / w - 0.5
        oy = top + (yy + 0.5) * ch / h - 0.5
        oracle = np.stack(((ox-cx)/fx, (oy-cy)/fy, np.ones_like(ox)), -1)
        z_metres = 1.37 + 0.02*np.mod(xx+yy, 25)
        rays = camera_rays(cam)
        ray_error = float(np.max(np.abs((rays-oracle)*z_metres[...,None])))
        q = oracle @ kw.T
        uv_error = float(max(np.max(np.abs(q[...,0]-xx)), np.max(np.abs(q[...,1]-yy))))
        # Near-half-pixel silhouette. A second source is unnecessary for this single-source oracle.
        silhouette = xx < (w*0.54+0.37)
        rgb = np.full((h,w,3), 0.4, dtype=np.float32)
        alpha = np.ones((h,w), np.float32)
        target_z = np.full((h,w), 4., dtype=np.float32)
        target_alpha = alpha.copy()
        target_z[0,0] = np.nan
        target_alpha[0,0] = 0
        target_alpha[0,1] = np.nan
        src_z = np.where(silhouette, 2., 4.).astype(np.float32)
        src_z[-1,-1] = np.nan
        src_alpha = alpha.copy()
        src_alpha[-1,-1] = 0
        src_alpha[-2,-1] = np.float32(0.125)
        target = dict(z=target_z,alpha=target_alpha,rgb=rgb)
        source = dict(z=src_z,alpha=src_alpha,rgb=rgb)
        result = extract_ordered(target,cam,[("source", (rgb,source,cam))])
        support = ~silhouette
        support[-2:,-1] = False
        conflict = silhouette.copy()
        conflict[0,:2] = False
        why = np.zeros((h,w), np.uint8)
        why[0,0], why[0,1], why[-1,-1] = 4,3,6
        expected = (("support_count",support), ("depth_conflict_count",conflict),("unknown_reason",why))
        differences = {key:int(np.count_nonzero(result[key] != arr)) for key,arr in expected}
        bounds = registration["geometry_checks"]
        if (ray_error > bounds["unprojection_abs_tol_m"] or uv_error > bounds["projection_abs_tol_px"]
                or any(differences.values())):
            raise ValueError(f"{profile}: ray={ray_error} projection={uv_error} masks={differences}")
        rows.append(dict(profile=profile, analysis_wh=[w,h],original_wh=[ow,oh],cropped_xywh=view["crop_xywh"],
            pixels=w*h, projection_abs_px=uv_error, unprojection_abs_m=ray_error,
            expected_conflicts=int(conflict.sum()),expected_supported=int(support.sum()),
            invalid_target_depth=1,invalid_target_alpha=1,no_comparable_source=1,
            depth_and_mask_mismatch_counts=differences))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    rows=run(json.loads(a.registration.read_bytes()))
    a.out.write_text(json.dumps({"schema":"el-v05-analytic-geometry-1","run_id":os.environ.get("EL_RUN_ID"),
                                 "rows":rows},indent=2,allow_nan=False)+"\n",encoding="utf-8")
