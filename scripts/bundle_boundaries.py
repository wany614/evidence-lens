"""Record controlled faults on private copies of the public example and conserve their canvas counts."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from bundle_build import environment, run
from bundle_original import check, put, sha


def faults(a):
    with tempfile.TemporaryDirectory(dir=a.out.parent) as tmp:
        t = Path(tmp)
        cli, lib = environment(t, a.wheel)
        seen = {}
        for name, code in (
            ("target_reference_role", 2),
            ("wrong_depth_semantics", 3),
            ("wrong_unit", 3),
            ("wrong_camera_pose", 2),
            ("three_sources", 3),
            ("wrong_operating_profile", 3),
            ("absent_ready_frame", 2),
            ("swapped_source_bytes", 2),
            ("same_shape_wrong_input_pin", 2),
            ("optional_model_root_required", 3),
        ):
            bundle = t / name
            shutil.copytree(a.bundle, bundle, copy_function=shutil.copyfile)
            b = json.loads((bundle / "bundle.json").read_bytes())
            p = json.loads((bundle / "producer.json").read_bytes())
            c = json.loads((bundle / "cameras.json").read_bytes())
            command = [cli, "validate", "--bundle", bundle]
            if name == "target_reference_role":
                b["references"] = {"path": "ReferenceBundle.json"}
            if name == "wrong_depth_semantics":
                p["semantics"]["z"] = "euclidean_ray_distance"
            if name == "wrong_unit":
                b["geometry"]["prepared_camera_and_z_unit"] = "arbitrary_units"
            if name == "wrong_camera_pose":
                c["views"][b["target_order"][0]]["c2w"][0][0] += 0.41
            if name == "three_sources":
                b["source_order"] = b["source_order"][:3]
            if name == "wrong_operating_profile":
                b["risk_profile"]["id"] = "new_thresholds"
            if name == "absent_ready_frame":
                (bundle / p["frames"][b["target_order"][0]]["file"]["path"]).unlink()
            if name == "swapped_source_bytes":
                s = b["source_order"]
                (bundle / b["sources"][s[0]]["path"]).write_bytes(
                    (bundle / b["sources"][s[1]]["path"]).read_bytes()
                )
            if name == "same_shape_wrong_input_pin":
                command += ["--bundle-sha256", "7" * 64]
            if name == "optional_model_root_required":
                command = [cli, "score", "--model", "MS-ft", "--bundle", bundle, "--out", t / "unwritten"]
            if name == "wrong_camera_pose":
                put(bundle / "cameras.json", c)
                raw = (bundle / "cameras.json").read_bytes()
                b["camera"].update(bytes=len(raw), sha256=sha(raw))
                p["camera_sha256"] = sha(raw)
            put(bundle / "producer.json", p)
            raw = (bundle / "producer.json").read_bytes()
            b["producer"].update(bytes=len(raw), sha256=sha(raw))
            put(bundle / "bundle.json", b)
            sub = subprocess.run(
                [str(x) for x in command],
                capture_output=True,
                text=True,
                env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            )
            status = json.loads(sub.stderr.strip().splitlines()[-1])
            check(
                sub.returncode == code and status["status"] == ("unsupported" if code == 3 else "invalid"),
                name,
            )
            seen[name] = {"cli_exit": code, "status": status["status"], "message": status["message"]}
        valid = {}
        for label in ("one_render_failed", "unverified_depth"):
            bundle = t / label
            shutil.copytree(a.bundle, bundle, copy_function=shutil.copyfile)
            b = json.loads((bundle / "bundle.json").read_bytes())
            first = b["target_order"][0]
            p = json.loads((bundle / "producer.json").read_bytes())
            if label == "one_render_failed":
                (bundle / p["frames"][first]["file"]["path"]).unlink()
                p["frames"][first] = {
                    "status": "render_failed",
                    "reason": "Recorded engineering test of failed target disposition",
                }
            else:
                p.update(depth_verified=False, depth_probe_sha256=None)
            put(bundle / "producer.json", p)
            raw = (bundle / "producer.json").read_bytes()
            b["producer"].update(bytes=len(raw), sha256=sha(raw))
            put(bundle / "bundle.json", b)
            out = t / ("scores-" + label)
            run([cli, "score", "--bundle", bundle, "--out", out])
            result = json.loads((out / "scores.json").read_bytes())
            c = result["counts"]
            firstrow = result["views"][0]
            reason = "render_failed" if label == "one_render_failed" else "depth_unverified"
            check(
                c["canvas"] == 6 * 256**2
                and firstrow["unknown_reasons"][reason] == 256**2
                and firstrow["counts"]["scoreable"] == 0,
                "incorrect unavailable canvas",
            )
            if label == "unverified_depth":
                check(c["unknown"] == c["canvas"], "unverified depth produced finite predictions")
            valid[label] = {
                "counts": c,
                "first_target": firstrow["counts"],
                "unknown_reasons": firstrow["unknown_reasons"],
            }
        put(
            a.out / "boundaries.json",
            {
                "schema": "el-bundle-boundary-qa-1",
                "run_id": os.getenv("EL_RUN_ID"),
                "intentional_engineering_cases": True,
                "installed_distributions": lib,
                "rejections": seen,
                "honest_unknown_outcomes": valid,
                "scientific_observations": 0,
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("bundle", "wheel", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    faults(p.parse_args())
