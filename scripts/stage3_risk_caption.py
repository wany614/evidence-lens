"""Correct a raster figure legend and verify every original scientific risk/error tile and archived array."""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from evidence_lens.formal_inputs import require, verified_file
from evidence_lens.multisource import METHODS, SCOPES
from evidence_lens.scoring_io import file_hash, new_output, write_json
from stage3_risk_boards import draw_header
from stage3_risk_sources import asset
from vision_sources import signed

COL = np.array(
    [[23, 71, 96], [37, 138, 151], [137, 200, 162], [245, 205, 123], [207, 63, 68]], dtype=np.float64
)
INVALID = (127, 55, 150)
UNKNOWN = (113, 118, 127)
LOW = (75, 168, 184)
HIGH = (230, 99, 67)


def mapped(a, mask, limit):
    finite = np.isfinite(a)
    t = np.clip(np.where(finite, a / limit, 0), 0, 1).astype(np.float64) * 4
    lo = np.minimum(np.floor(t).astype(int), 3)
    f = (t - lo)[..., None]
    colors = np.rint(COL[lo] + f * (COL[lo + 1] - COL[lo])).astype(np.uint8)
    colors[~finite] = UNKNOWN
    colors[~mask] = INVALID
    return colors


def state_colors(risk, mask, tau):
    result = np.full(risk.shape + (3,), UNKNOWN, np.uint8)
    finite = np.isfinite(risk)
    below = risk.astype(np.float64) < tau
    result[finite & below] = LOW
    result[finite & ~below] = HIGH
    result[~mask] = INVALID
    return result


def reissue(source, index_sha, output, rid):
    original = signed(source / "expected-example.json", index_sha)
    require(
        original["schema"] == "el-stage3-public-expected-example-1" and len(original["cases"]) == 12,
        "wrong original 057 case study",
    )
    with new_output(output) as dest:
        saved = set()

        def copy(spec):
            name = spec["path"]
            if name not in saved:
                path = verified_file(source, name, spec)
                p = dest / name
                p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, p)
                require(file_hash(p) == spec["sha256"], "changed original scientific member")
                saved.add(name)
            return dest / name

        indices = {}
        for method in METHODS:
            spec = original["scores"][method]
            j = json.loads(copy(spec).read_bytes())
            indices[method] = j
            for row in j["views"]:
                for m in row["modes"].values():
                    if m["file"]:
                        copy(m["file"])
        cases = []
        proof = []
        for orig in original["cases"]:
            r = dict(orig)
            c, v = r["condition"], r["view"]
            ep = copy(r["error"])
            with np.load(ep, allow_pickle=False) as z:
                e = z["E"]
                require(e.dtype == bool, "expected boolean RGB evaluation mask")
                rgb_error = z["rgb_mae"]
                zz = z["depth_evaluable"]
                dz = z["depth_abs_error_m"]
            old = verified_file(source, r["board"]["path"], r["board"])
            with Image.open(old) as im:
                img = im.convert("RGB")
            pre = np.asarray(img).copy()
            h, w = e.shape
            stride = w + 6

            def exact(col, row, pixels):
                left = 12 + col * stride
                top = 90 + row * (w + 22 + 6) + 22
                require(
                    np.array_equal(pre[top : top + h, left : left + w], pixels),
                    f"original pixel colours differ: {c}/{v}/{col}/{row}",
                )

            exact(6, 0, mapped(rgb_error, e, 0.2))
            exact(7, 0, mapped(dz, zz, 0.2))
            low = high = unknown = invalid = 0
            for ii, scope in enumerate(SCOPES, 1):
                for jj, m in enumerate(METHODS):
                    j = indices[m]
                    view = next(x for x in j["views"] if x["condition"] == c and x["view"] == v)
                    sp = view["modes"][scope]["file"]
                    with np.load(copy(sp), allow_pickle=False) as z:
                        risk = z["risk"]
                        state = z["state"]
                    exact(2 * jj, ii, mapped(risk, e, 1.0))
                    exact(2 * jj + 1, ii, state_colors(risk, e, j["threshold"]))
                    low += int(((state == 0) & e).sum())
                    high += int(((state == 1) & e).sum())
                    unknown += int(((state == 255) & e).sum())
                    invalid += int((~e).sum())
            draw_header(img, original["family"], c, v, w, h, "ready")
            np.testing.assert_array_equal(np.asarray(img)[90:], pre[90:])
            out = dest / r["board"]["path"]
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out, format="PNG")
            r["board"] = asset(dest, out)
            cases.append(r)
            proof.append(
                {
                    "condition": c,
                    "view": v,
                    "original_board_sha256": orig["board"]["sha256"],
                    "board": r["board"],
                    "image_width": img.width,
                    "image_height": img.height,
                    "header_height": 90,
                    "unaltered_raster_pixels_below_header": int((img.height - 90) * img.width),
                    "checked_error_and_risk_and_state_tiles": 26,
                    "low_state_pixels_across_tiles": low,
                    "high_state_pixels_across_tiles": high,
                    "unknown_state_pixels_across_tiles": unknown,
                    "outside_E_state_pixels_across_tiles": invalid,
                }
            )
        receipt = {
            "schema": "el-stage3-figure-legend-reissue-1",
            "run_id": rid,
            "original_run_id": original["run_id"],
            "original_expected_example_sha256": index_sha,
            "source_arrays_and_score_indices_with_identical_bytes": len(saved),
            "legend": {
                "zero": COL[0].astype(int).tolist(),
                "upper_or_higher": COL[-1].astype(int).tolist(),
                "low": LOW,
                "high": HIGH,
                "unknown": UNKNOWN,
                "outside_rgb_or_depth_evaluation": INVALID,
                "rgb_mae_limit": 0.20,
                "depth_mae_m_limit": 0.20,
                "risk_limit": 1.0,
            },
            "boards": proof,
        }
        write_json(dest / "figure-legend-qa.json", receipt)
        updated = {
            **original,
            "run_id": rid,
            "cases": cases,
            "display_reissue": {
                "original_run_id": original["run_id"],
                "original_expected_example_sha256": index_sha,
                "qa": asset(dest, dest / "figure-legend-qa.json"),
            },
        }
        write_json(dest / "expected-example.json", updated)
    return updated


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(reissue(a.source, a.source_sha256, a.output, os.environ["EL_RUN_ID"])["display_reissue"])
