"""Freeze original evidence identities, then rerun the phase-one camera audit from that sealed index."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from phase1_camera_audit import audit
from phase1_geometry_oracle import sha

from evidence_lens.formal_inputs import PinnedRun

SHARP = "aed6527499ef91cba3b54c18d49a870f25947190"
GSPLAT = "937e29912570c372bed6747a5c9bf85fed877bae"
REPORT = {
    "run_id": "20261123T050620457525Z-a311c1632b7d4b1b",
    "manifest_sha256": "209fb7f045a7e79aaa2113eb50a8823c4f0dbf66098e058ae4b201a826653aba",
}
# Ranges are one-based inclusive. The files and line digests are pinned before the archive replay.
SECTIONS = (
    (
        "project",
        "src/evidence_lens/generator_adapter.py",
        [(19, 51), (55, 84)],
        "RGB8 source size, centred integer K, direct focal input and timed neural forward, official PLY export.",
    ),
    (
        "sharp",
        "src/sharp/cli/predict.py",
        [(159, 205)],
        "Original focal pixels divided by source width, endpoint-aligned bilinear 1536-square RGB, row-scaled boundary K and identity unprojection.",
    ),
    (
        "sharp",
        "src/sharp/models/predictor.py",
        [(126, 143)],
        "Learned monocular disparity enters the initial depth calculation through focal/width divided by clamped disparity.",
    ),
    (
        "sharp",
        "src/sharp/models/initializer.py",
        [(256, 271)],
        "Initial x/y sample centres start half a stride inside the boundary and increase right/down.",
    ),
    (
        "sharp",
        "src/sharp/models/composer.py",
        [(188, 209)],
        "Network combines the initial inverse depth with its learned update and emits (Z*x_ndc, Z*y_ndc, Z).",
    ),
    (
        "sharp",
        "src/sharp/utils/gaussians.py",
        [
            (69, 97),
            (113, 121),
            (243, 265),
            (294, 299),
            (307, 343),
            (353, 378),
            (402, 436),
            (459, 462),
        ],
        "Executable inverse NDC–intrinsics transform leaves depth in source coordinates. Official PLY writes a boundary-centred K, identity pose and RGB SH0; its reader restores linear colour, scale and opacity.",
    ),
    (
        "sharp",
        "src/sharp/utils/color_space.py",
        [(18, 28), (46, 58), (77, 88)],
        "The PLY colour-space code 0 means sRGB. The official piecewise functions convert between sRGB and linear channels.",
    ),
    (
        "project",
        "src/evidence_lens/renderer_adapter.py",
        [(14, 32), (41, 69), (75, 114), (119, 139)],
        "The selected target receives inverse(target camera-to-world) times source camera-to-world. Both principal coordinates gain 0.5 once. Depth validation, black background, display conversion, NaN and preview paths.",
    ),
    (
        "gsplat",
        "gsplat/cuda/include/Utils.cuh",
        [(500, 536)],
        "Perspective projection evaluates (fx X/Z + cx, fy Y/Z + cy) in the rasterizer boundary coordinate convention.",
    ),
    (
        "gsplat",
        "gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu",
        [(55, 74), (105, 120), (201, 208)],
        "The dense classic projection kernel reads row-major target rotation/translation, computes the camera mean and stores its optical-axis Z.",
    ),
    (
        "gsplat",
        "gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu",
        [(60, 65), (142, 169), (177, 187)],
        "A pixel samples at (column+0.5,row+0.5). Front-to-back opacity times remaining transmittance weights each Gaussian contribution.",
    ),
    (
        "gsplat",
        "gsplat/rendering.py",
        [(614, 633), (760, 770)],
        "A depth channel accumulates target-camera Z. Expected depth divides the accumulated depth by accumulated opacity, with a numerical floor.",
    ),
    (
        "project",
        "src/evidence_lens/blender_reference/scene.py",
        [(13, 35), (147, 155), (171, 201)],
        "The Cycles scene sets one metre per unit, an illuminated World, and opaque film. The camera export converts Blender to OpenCV axes and integer principal coordinates.",
    ),
    (
        "project",
        "src/evidence_lens/blender_reference/passes.py",
        [(175, 204)],
        "Finite Position mesh hits are projected to positive optical-axis Z in the target camera. Valid display RGB is stored separately.",
    ),
    (
        "project",
        "src/evidence_lens/blender_reference/verify_reference.py",
        [(110, 137)],
        "Independent native Blender camera projection and archived position checks use the pixel-centre conversion.",
    ),
    (
        "project",
        "scripts/stress_condition.py",
        [(37, 60), (103, 147)],
        "R1 halves the source resolution, focal values and principal-point coordinates with the integer-centre map. Q1 changes only stored SH0 colour coefficients using the B0 PLY parent.",
    ),
)


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(path.read_bytes())


def file_index(folder):
    return {
        path.relative_to(folder).as_posix(): sha(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def pins(plan):
    unique = {}
    for case in plan["cases"]:
        for role in (
            "reference",
            "readback",
            "generation",
            "producer",
            "preparation",
            "render",
        ):
            pin = case["selected"].get(role)
            if pin:
                assert unique.setdefault(pin["run_id"], pin) == pin
    return unique


def prepare(out, root, plan_file, protocol, store, cold):
    """Unsealed intake reads only original manifests, selections, source, and the explicitly supplied plan."""
    out.mkdir()
    shutil.copyfile(plan_file, out / "plan.json")
    shutil.copyfile(protocol, out / "geometry-protocol.json")
    shutil.copyfile(root / "protocol/phase1-camera-chain-v1.md", out / "call-path.md")
    plan = read(plan_file)
    assert Counter(c["condition"] for c in plan["cases"]) == {
        "B0": 24,
        "R1": 4,
        "Q1": 4,
    }
    (out / "selections").mkdir()
    (out / "manifests").mkdir()
    for case in plan["cases"]:
        path = store / "trials" / case["trial"] / "trial_selection.json"
        assert sha(path) == case["selection_sha256"]
        shutil.copyfile(path, out / "selections" / (case["trial"] + ".json"))
    for rid, pin in pins(plan).items():
        shelf = store if (store / "runs" / rid).is_dir() else cold
        run = PinnedRun(shelf, pin)
        assert run.run["execution_status"] == "succeeded"
        shutil.copyfile(run.root / "manifest.json", out / "manifests" / (rid + ".json"))

    sources = {
        "project": (root, None, None),
        "sharp": (
            root / ".research/upstream/ml-sharp",
            SHARP,
            "https://github.com/apple-aiml-research/ml-sharp",
        ),
        "gsplat": (
            root / ".research/upstream/gsplat",
            GSPLAT,
            "https://github.com/nerfstudio-project/gsplat",
        ),
    }
    traces = []
    for owner, path, intervals, meaning in SECTIONS:
        repository, revision, url = sources[owner]
        content = (repository / path).read_bytes()
        if revision:
            committed = subprocess.check_output(
                ["git", "-C", repository, "show", revision + ":" + path]
            )
            assert content.replace(b"\r\n", b"\n") == committed
        dest = out / "raw-source" / owner / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        lines = content.replace(b"\r\n", b"\n").splitlines(keepends=True)
        segments = []
        for start, end in intervals:
            assert 0 < start <= end <= len(lines), (path, start, end, len(lines))
            fragment = b"".join(lines[start - 1 : end])
            segments.append(
                {
                    "first_line": start,
                    "last_line": end,
                    "lf_sha256": hashlib.sha256(fragment).hexdigest(),
                }
            )
        traces.append(
            {
                "owner": owner,
                "path": path,
                "revision": revision,
                "file_sha256": sha(dest),
                "lf_sha256": sha(dest, True),
                "segments": segments,
                "finding": meaning,
                "source_url": f"{url}/blob/{revision}/{path}" if revision else None,
            }
        )
    write(
        out / "source-index.json",
        {
            "schema": "el-phase1-camera-source-index-1",
            "original_report": REPORT,
            "files": traces,
        },
    )


def seal(bundle, report_path, out):
    """Save the frozen registry and its hashes before any numerical replay."""
    report = read(report_path)
    assert report["run_id"] == REPORT["run_id"] and report["kind"] == "heldout_study"
    plan = read(bundle / "plan.json")
    expected = {
        (c["family"], c["condition"])
        for c in plan["cases"]
        if c["family"] in {row["scene_family_id"] for row in report["selected_trials"]}
    }
    observed = {
        (row["scene_family_id"], row["condition_id"])
        for row in report["selected_trials"]
    }
    assert len(observed) == 12 and expected == observed
    assert len(list((bundle / "selections").glob("*.json"))) == 32
    for rid, pin in pins(plan).items():
        assert sha(bundle / "manifests" / (rid + ".json")) == pin["manifest_sha256"]
    for case in plan["cases"]:
        assert (
            sha(bundle / "selections" / (case["trial"] + ".json"))
            == case["selection_sha256"]
        )
    dest = out / "audit-index"
    shutil.copytree(bundle, dest)
    write(
        out / "intake.json",
        {
            "schema": "el-phase1-camera-intake-1",
            "run_id": os.environ["EL_RUN_ID"],
            "original_report": REPORT,
            "original_report_json_sha256": sha(report_path),
            "cases": 32,
            "archived_stage_pins": len(pins(plan)),
            "files": file_index(dest),
        },
    )


def summarize(data):
    cases = data["cases"]
    refs = data["native_references"].values()
    native = [
        p for r in refs for values in r["native"]["geometry"].values() for p in values
    ]
    reference = data["native_references"].values()
    old = [xy for c in cases for xy in c["prior_calibration"]["samples"]]
    groups = {}
    for cond in ("B0", "R1", "Q1"):
        rows = [row for row in cases if row["condition"] == cond]
        grids = [g for c in rows for g in c["actual_prediction_grids"].values()]
        groups[cond] = {
            "cells": len(rows),
            "source_wh": rows[0]["width_height"],
            "focal_px": rows[0]["f_px_recorded"],
            "f_over_input_width": rows[0]["f_over_width_derived"],
            "prediction_grids": len(grids),
            "alpha_zero_pixels": sum(g["alpha_zero"] for g in grids),
            "unavailable_z_pixels": sum(g["missing_z"] for g in grids),
        }
    return {
        "schema": "el-phase1-camera-summary-1",
        "run_id": os.environ["EL_RUN_ID"],
        "actual_report": REPORT,
        "passed": data["passed"],
        "groups": groups,
        "reference_blends": len(data["native_references"]),
        "unique_historical_learned_runs": len(
            {c["original_learned_run"]["run_id"] for c in cases}
        ),
        "neural_generations_in_audit": 0,
        "reference_renders_in_audit": 0,
        "Q1_zero_forward_derivations": sum(c["zero_forward_Q1"] == 0 for c in cases),
        "actual_npz_and_png_pairs": sum(
            len(c["actual_prediction_grids"]) for c in cases
        ),
        "verified_unique_archive_members": sum(
            map(len, data["verified_archive_members"].values())
        ),
        "native_point_projections": len(native),
        "native_point_max_axis_residual_px": max(p["xy_max_error_px"] for p in native),
        "native_point_max_z_residual_m": max(abs(p["Z_error_m"]) for p in native),
        "legacy_probe_max_ED_residual_m": max(abs(v["ED_error_m"]) for v in old),
        "legacy_probe_max_D_minus_alphaZ_m": max(
            abs(v["D_minus_alphaZ_m"]) for v in old
        ),
        "real_Position_max_sample_reprojection_px": max(
            r["archived_max_Position_reprojection_px"] for r in reference
        ),
        "real_Position_max_Z_residual_m": max(
            r["archived_max_Position_Z_error_m"] for r in reference
        ),
    }


def replay(bundle, receipt_path, report_path, out, store, cold):
    receipt = read(receipt_path)
    assert receipt["files"] == file_index(bundle) and receipt[
        "original_report_json_sha256"
    ] == sha(report_path)
    assert receipt["original_report"] == REPORT
    data = audit(
        store, bundle / "plan.json", bundle / "geometry-protocol.json", cold, bundle
    )
    for rid, members in data["verified_archive_members"].items():
        rows = read(bundle / "manifests" / (rid + ".json"))["files"]
        indexed = {row["path"]: row["sha256"] for row in rows}
        assert all(indexed[path] == value for path, value in members.items())
    data["intake_run_id"] = receipt["run_id"]
    data["study_report"] = REPORT
    write(out / "camera-audit.json", data)
    summary = summarize(data)
    summary.update(
        intake_run_id=receipt["run_id"],
        plan_sha256=data["plan_sha256"],
        camera_audit_sha256=sha(out / "camera-audit.json"),
        source_index_sha256=sha(bundle / "source-index.json"),
        intake_receipt_sha256=sha(receipt_path),
    )
    write(out / "camera-summary.json", summary)
    shutil.copyfile(bundle / "source-index.json", out / "source-index.json")
    shutil.copyfile(bundle / "call-path.md", out / "call-path.md")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "seal", "replay"))
    for name in (
        "out",
        "project",
        "plan",
        "protocol",
        "bundle",
        "store",
        "cold",
        "report",
        "receipt",
    ):
        p.add_argument("--" + name, type=Path, required=(name == "out"))
    a = p.parse_args()
    if a.mode == "prepare":
        prepare(a.out, a.project, a.plan, a.protocol, a.store, a.cold)
    elif a.mode == "seal":
        seal(a.bundle, a.report, a.out)
    else:
        replay(a.bundle, a.receipt, a.report, a.out, a.store, a.cold)


if __name__ == "__main__":
    main()
