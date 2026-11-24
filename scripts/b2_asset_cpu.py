"""Read original interchange inputs on CPU and compare with immutable native cameras and Gaussian parameters."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

import numpy as np

from evidence_lens_bundle.common import Unsupported, require, sha, write
from evidence_lens_asset.prepare import inspect
from evidence_lens_asset import gaussian


def run(a):
    require(
        sha(a.colmap_reader.read_bytes())
        == "a732ea7f4cb0382188106b3dd31c0b58a121c4eeafe985404d3f81a331bc8acd",
        "unrecognized COLMAP 3.10 reader",
    )
    spec = importlib.util.spec_from_file_location("pinned_colmap_310_reader", a.colmap_reader)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    fixture = a.fixture
    proto = json.loads(a.protocol.read_bytes())
    camera_gold = json.loads((fixture / "native-cameras.json").read_bytes())
    results = []
    with np.load(fixture / "native.npz", allow_pickle=False) as z:
        gold = dict(z)
    parsed = []
    for fmt, ext in (("text", ".txt"), ("binary", ".bin")):
        model = fixture / "colmap" / fmt
        cams, ims, points = official.read_model(model, ext=ext)
        require(
            len(ims) == 5 and len(cams) == 2 and len(points) == 0,
            "official COLMAP reader did not recover the rig",
        )
        for view, im in ims.items():
            expected_name = {"41": "acquired/left-lens.png", "5": "acquired/endpoint.png"}.get(
                str(view), "unobserved/" + str(view) + ".png"
            )
            require(im.name == expected_name, "official COLMAP filename differs")
            r = official.qvec2rotmat(im.qvec)
            actual = np.eye(4)
            actual[:3, :3] = r.T
            actual[:3, 3] = -0.01 * r.T @ im.tvec
            require(
                np.allclose(actual, camera_gold[str(view)]["c2w"], rtol=0, atol=1e-8),
                "official camera readback differs",
            )
        for file, profile in (
            ("graphdeco-sh0.ply", "graphdeco-v1"),
            ("graphdeco-sh3.ply", "graphdeco-v1"),
            ("project.npz", "evidence-lens-npz-v1"),
        ):
            began = perf_counter()
            stats, cal, photos, arr = inspect(
                model, fixture / "images", fixture / file, fixture / "plan.json", fmt, profile
            )
            require(
                list(photos) == ["41", "5"] and stats["target_order"] == ["70", "109", "13"],
                "opaque ID order changed",
            )
            cd = max(
                float(np.abs(np.asarray(c[k]) - np.asarray(camera_gold[name][k])).max())
                for name, c in cal["views"].items()
                for k in ("K", "c2w")
            )
            require(
                cd <= proto["comparisons"]["absolute_tolerance"]["camera_K_and_pose"],
                "metric camera mismatch",
            )
            actual = {
                "means": arr["means"],
                "radii": np.exp(arr["log_scales"]),
                "alpha": 1 / (1 + np.exp(-arr["opacity_logits"])),
                "quaternions": arr["quats"],
            }
            ref = {
                "means": gold["centers_m"],
                "radii": gold["radii_m"],
                "alpha": gold["alpha"],
                "quaternions": gold["quaternions_wxyz"],
            }
            if "sh" in arr:
                actual["SH"] = arr["sh"]
                ref["SH"] = gold["sh"][:, : arr["sh"].shape[1]]
            else:
                actual["linear_color"] = 1 / (1 + np.exp(-arr["color_logits"]))
                ref["linear_color"] = gold["linear_color"]
            error = {k: float(np.max(np.abs(v - ref[k]))) for k, v in actual.items()}
            require(all(x < 1e-6 for x in error.values()), "wrong native Gaussian attributes")
            results.append(
                {
                    "asset": file,
                    "encoding": fmt,
                    "sources": list(photos),
                    "targets": stats["target_order"],
                    "camera_max_abs": cd,
                    "parameter_max_abs": error,
                    "elapsed_seconds": perf_counter() - began,
                    "input_sha256": stats["asset_sha256"],
                    "gaussians": stats["gaussians"],
                }
            )
            parsed.append(stats)
    faults = {}
    with tempfile.TemporaryDirectory(dir=a.out.parent) as work:
        tmp = Path(work)
        bad = tmp / "points.ply"
        bad.write_bytes(
            b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n"
        )
        distorted = tmp / "model"
        shutil.copytree(fixture / "colmap/text", distorted)
        p = distorted / "cameras.txt"
        p.chmod(0o600)
        p.write_text(p.read_text().replace("17 PINHOLE", "17 OPENCV"))
        missing = tmp / "unknown-scale.json"
        plan = json.loads((fixture / "plan.json").read_bytes())
        del plan["metric"]
        write(missing, plan)
        partial = tmp / "incomplete.ply"
        partial.write_bytes(
            (fixture / "graphdeco-sh3.ply").read_bytes().replace(b"f_rest_23\n", b"f_rest_53\n", 1)
        )
        checks = {
            "sparse_PLY": lambda: gaussian.load(bad, "graphdeco-v1", 0.01),
            "incomplete_SH3": lambda: gaussian.load(partial, "graphdeco-v1", 0.01),
            "distortion": lambda: inspect(
                distorted,
                fixture / "images",
                fixture / "project.npz",
                fixture / "plan.json",
                "text",
                "evidence-lens-npz-v1",
            ),
            "no_metric_scale": lambda: inspect(
                fixture / "colmap/text",
                fixture / "images",
                fixture / "project.npz",
                missing,
                "text",
                "evidence-lens-npz-v1",
            ),
        }
        for name, func in checks.items():
            try:
                func()
            except Unsupported as e:
                faults[name] = str(e)
            else:
                raise ValueError("unsupported input succeeded: " + name)
    write(
        a.out / "cpu-readback.json",
        {
            "schema": "el-b2-asset-cpu-readback-1",
            "run_id": os.environ["EL_RUN_ID"],
            "protocol_sha256": sha(a.protocol.read_bytes()),
            "fixture_sha256": sha((fixture / "fixture.json").read_bytes()),
            "official_COLMAP_reader_sha256": sha(a.colmap_reader.read_bytes()),
            "cases": results,
            "typed_unsupported": faults,
            "target_photos_present_or_opened": 0,
            "gpu_calls": 0,
            "passed": True,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("fixture", "protocol", "colmap-reader", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    run(p.parse_args())
