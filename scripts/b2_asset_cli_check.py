"""Exercise successful CPU import and explicit unsupported/invalid CLI outcomes in a fresh installed environment."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from PIL import Image
from bundle_build import environment
from bundle_original import check, put, sha


def shell(cli, argv):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)
    p = subprocess.run([str(cli), *map(str, argv)], env=env, capture_output=True, text=True)
    data = json.loads(p.stdout if p.returncode == 0 else p.stderr)
    return p.returncode, data


def run(a):
    with tempfile.TemporaryDirectory(dir=a.out.parent) as td:
        stage = Path(td)
        cli, packages = environment(stage, a.wheel)
        args = lambda fixture, model, encoding, plan: [
            "--colmap",
            fixture / "colmap" / encoding,
            "--encoding",
            encoding,
            "--images",
            fixture / "images",
            "--gaussians",
            fixture / model,
            "--gaussian-profile",
            "evidence-lens-npz-v1" if model.endswith(".npz") else "graphdeco-v1",
            "--plan",
            plan,
        ]
        rows = {}
        for mode in ("text", "binary"):
            for scene in ("graphdeco-sh0.ply", "graphdeco-sh3.ply", "project.npz"):
                dst = stage / (scene + mode)
                rc, result = shell(
                    cli, ["inspect", *args(a.fixture, scene, mode, a.fixture / "plan.json"), "--out", dst]
                )
                check(
                    rc == 0
                    and result["source_order"] == ["41", "5"]
                    and result["target_order"] == ["70", "109", "13"]
                    and result["renderer"]["device"] == "cuda"
                    and result["scorer"]["device"] == "cpu",
                    "failed successful CPU inspection",
                )
                rows[scene + "-" + mode] = {
                    "status": "validated",
                    "inspection_sha256": sha((dst / "inspection.json").read_bytes()),
                    "cameras_sha256": sha((dst / "cameras.json").read_bytes()),
                    "sh_degree": result["sh_degree"],
                    "entries": result["gaussians"],
                }
        failures = {}
        modes = (
            "sparse_cloud",
            "missing_opacity",
            "unsupported_sh_order",
            "camera_distortion",
            "missing_scale",
            "contradictory_scale",
            "duplicate_source",
            "absent_source_photo",
            "inconsistent_source_size",
            "no_CUDA",
        )
        for name in modes:
            f = shutil.copytree(a.fixture, stage / name, copy_function=shutil.copyfile)
            model = "project.npz"
            plan = f / "plan.json"
            j = json.loads(plan.read_bytes())
            command = "inspect"
            if name in ("sparse_cloud", "missing_opacity", "unsupported_sh_order"):
                model = "graphdeco-sh0.ply"
                p = f / model
                raw = p.read_bytes()
                if name == "sparse_cloud":
                    p.write_bytes(
                        b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n"
                    )
                elif name == "missing_opacity":
                    p.write_bytes(raw.replace(b"property float opacity", b"property float lossfac"))
                else:
                    p.write_bytes(raw.replace(b"end_header", b"property float f_rest_0\nend_header"))
            elif name == "camera_distortion":
                p = f / "colmap/text/cameras.txt"
                p.write_text(p.read_text().replace("17 PINHOLE", "17 OPENCV"))
            elif name == "missing_scale":
                del j["metric"]
            elif name == "contradictory_scale":
                j["metric"]["meters_per_unit"] = 1
            elif name == "duplicate_source":
                j["sources"][1] = j["sources"][0]
            elif name == "absent_source_photo":
                (f / "images/acquired/left-lens.png").unlink()
            elif name == "inconsistent_source_size":
                Image.new("RGB", (128, 256)).save(f / "images/acquired/left-lens.png")
            else:
                command = "render-asset"
            put(plan, j)
            dst = stage / ("out-" + name)
            rc, result = shell(cli, [command, *args(f, model, "text", plan), "--out", dst])
            expected = 2 if name in ("duplicate_source", "absent_source_photo") else 3
            check(
                rc == expected
                and result["status"] == ("invalid" if rc == 2 else "unsupported")
                and not dst.exists(),
                name + " failure disposition incorrect",
            )
            failures[name] = {
                "exit": rc,
                "status": result["status"],
                "message": result["message"].replace(str(f), "<fixture>"),
            }
        put(
            a.out / "cli-check.json",
            {
                "schema": "el-b2-asset-installed-cli-check-1",
                "run_id": os.environ.get("EL_RUN_ID"),
                "installed": packages,
                "wheel_sha256": sha(a.wheel.read_bytes()),
                "fixture_sha256": sha((a.fixture / "fixture.json").read_bytes()),
                "positive": rows,
                "negative": failures,
                "target_photo_reads": 0,
                "cuda_scientific_operations": 0,
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("wheel", "fixture", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    run(p.parse_args())
