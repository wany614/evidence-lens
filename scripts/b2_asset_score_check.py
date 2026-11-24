"""Replay the installed CPU CLI and read back the archived numerical maps and optional calibration layer."""

import argparse
from pathlib import Path
import json
import os
import tempfile
import numpy as np
from bundle_build import environment, run
from bundle_original import put, check, sha


def work(a):
    with tempfile.TemporaryDirectory(dir=a.out.parent) as td:
        cli, inventory = environment(Path(td), a.wheel)
        h = sha((a.bundle / "bundle.json").read_bytes())
        validate = json.loads(run([cli, "validate", "--bundle", a.bundle, "--bundle-sha256", h]))
        dest = a.out / "scores"
        cmd = [cli, "score", "--bundle", a.bundle, "--bundle-sha256", h, "--out", dest]
        if a.profile is not None:
            cmd += ["--calibration-profile", a.profile, "--calibration-sha256", sha(a.profile.read_bytes())]
        if a.scope is not None:
            cmd += ["--scope-proof", a.scope, "--scope-sha256", sha(a.scope.read_bytes())]
        reply = json.loads(run(cmd))
        score = json.loads((dest / "scores.json").read_bytes())
        check(
            validate["target_order"] == score["target_order"] and reply["counts"] == score["counts"],
            "score receipt mismatch",
        )
        html = a.out / "report.html"
        html_result = json.loads(
            run(
                [
                    cli,
                    "report",
                    "--scores",
                    dest,
                    "--scores-sha256",
                    sha((dest / "scores.json").read_bytes()),
                    "--out",
                    html,
                ]
            )
        )
        actual = score["calibration_profile"]
        if a.profile is not None:
            prof = json.loads(a.profile.read_bytes())
            check(
                actual["status"] == (prof["status"] if a.scope else "outside_calibration_scope"),
                "wrong real profile disposition",
            )
        rows = []
        if a.original:
            old = json.loads((a.original / "scores.json").read_bytes())
            check(
                old["case_sha256"] == score["creation"]["original_case_sha256"]
                and old["checkpoint_sha256"] == score["checkpoint_sha256"],
                "original scorer belongs to different inputs",
            )
            for new in score["views"]:
                v = new["id"]
                entry = old["views"][v]["methods"][score["method"]]["file"]
                gold = a.original / entry["path"]
                fp = dest / new["arrays"]["path"]
                check(
                    sha(gold.read_bytes()) == entry["sha256"]
                    and sha(fp.read_bytes()) == new["arrays"]["sha256"],
                    "numerical member differs",
                )
                with np.load(gold, allow_pickle=False) as p, np.load(fp, allow_pickle=False) as q:
                    check(
                        np.array_equal(q["state"], p["state"])
                        and np.array_equal(q["valid"], p["valid"])
                        and np.array_equal(q["risk"].view(np.uint32), p["risk"].view(np.uint32)),
                        "original release score changed",
                    )
                    rows.append(
                        {
                            "view": v,
                            "pixels": q["state"].size,
                            "finite": int(q["valid"].sum()),
                            "different_float32_words_including_NaN": 0,
                            "different_states": 0,
                            "different_validity": 0,
                            "upstream_sha256": entry["sha256"],
                            "bundle_score_sha256": new["arrays"]["sha256"],
                        }
                    )
        put(
            a.out / "grade.json",
            {
                "schema": "el-b2-asset-score-readback-1",
                "run_id": os.environ["EL_RUN_ID"],
                "wheel_sha256": sha(a.wheel.read_bytes()),
                "installed": inventory,
                "bundle_sha256": h,
                "model": score["model"],
                "counts": score["counts"],
                "calibration": actual,
                "html_sha256": html_result["html_sha256"],
                "original_case_rows": rows,
                "target_reference_photographs_read": 0,
                "passed": True,
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("wheel", "bundle", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("scope", "profile", "original"):
        p.add_argument("--" + n, type=Path)
    work(p.parse_args())
