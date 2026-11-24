"""Exercise the standard wheel on a recovered Case or compare its outputs with sealed original scores."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from bundle_build import environment, run
from bundle_grade import compare, read
from bundle_original import METHODS, check, put, sha
from bundle_score_record import released_models


def score(a):
    with tempfile.TemporaryDirectory(dir=a.out.parent) as scratch:
        tmp = Path(scratch)
        cli, base = environment(tmp, a.wheel)
        old = json.loads((a.case / "case.json").read_bytes())
        case_sha = sha((a.case / "case.json").read_bytes())
        proof_sha = sha((a.proof / "proof.json").read_bytes())
        clean = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

        def failure(args, label):
            result = subprocess.run([str(v) for v in args], text=True, capture_output=True, env=clean)
            check(result.returncode == 2, "expected integrity rejection: " + label)
            return json.loads(result.stderr)

        args = [
            cli,
            "adapt-legacy",
            "--case",
            a.case,
            "--case-sha256",
            case_sha,
            "--input-domain",
            "Private exposed DTU set6 M4; frozen recovery checkpoint",
        ]
        without = failure([*args, "--out", tmp / "no-proof"], "missing producer proof")
        wrong = failure(
            [*args, "--proof", a.proof, "--proof-sha256", "0" * 64, "--out", tmp / "wrong-proof"],
            "changed proof digest",
        )
        tamper = tmp / "tamper"
        shutil.copytree(a.proof, tamper, copy_function=shutil.copyfile)
        n = tamper / "documents/normalized.json"
        oldnorm = json.loads(n.read_bytes())
        oldnorm["original_render"]["run_id"] = oldnorm["original_generation"]["run_id"]
        put(n, oldnorm)
        smuggle = failure(
            [*args, "--proof", tamper, "--proof-sha256", proof_sha, "--out", tmp / "wrong-render"],
            "drawn from a different render",
        )
        b = a.out / "bundle"
        reply = json.loads(run([*args, "--proof", a.proof, "--proof-sha256", proof_sha, "--out", b]))
        validation = json.loads(
            run([cli, "validate", "--bundle", b, "--bundle-sha256", reply["bundle_sha256"]])
        )
        check(validation == reply, "validation changed")
        real = json.loads((b / "producer.json").read_bytes())
        models = tmp / "models"
        released_models(a.cpu, models)
        opt_dir = tmp / "opt"
        opt_dir.mkdir()
        optional, _ = environment(opt_dir, a.wheel)
        python = optional.with_name("python.exe" if os.name == "nt" else "python")
        run(
            [
                shutil.which("uv"),
                "pip",
                "install",
                "--offline",
                "--python",
                python,
                "torch==2.8.0",
                "safetensors==0.8.0",
            ]
        )
        observations = {}
        for method in METHODS:
            exe = optional if method in ("MS-ft", "MS-frozen") else cli
            dest = a.out / "scores" / method
            run(
                [
                    exe,
                    "score",
                    "--model",
                    method,
                    "--bundle",
                    b,
                    "--bundle-sha256",
                    reply["bundle_sha256"],
                    "--models",
                    models,
                    "--out",
                    dest,
                ]
            )
            r = json.loads((dest / "scores.json").read_bytes())
            observations[method] = {
                "receipt_sha256": sha((dest / "scores.json").read_bytes()),
                "counts": r["counts"],
            }
        put(
            a.out / "recovery-compatibility.json",
            {
                "schema": "el-bundle-recovery-deployment-check-1",
                "run_id": os.getenv("EL_RUN_ID"),
                "old_case_sha256": case_sha,
                "proof_sha256": proof_sha,
                "bundle_sha256": reply["bundle_sha256"],
                "source_only": True,
                "conditions": old["condition"],
                "generator": real["generator"],
                "renderer": real["renderer"],
                "original_probe_sha256": real["depth_probe_sha256"],
                "checkpoint_sha256": real["checkpoint_sha256"],
                "tested_rejections": {
                    "missing_proof": without,
                    "changed_proof": wrong,
                    "changed_normalized_renderer": smuggle,
                },
                "scores": observations,
                "base_environment": base,
                "wheel_sha256": sha(a.wheel.read_bytes()),
            },
        )


def grade(a):
    old = json.loads((a.old / "scores.json").read_bytes())
    rows = []
    for method in METHODS:
        base = a.new / method
        r = json.loads((base / "scores.json").read_bytes())
        check(
            r["creation"]["original_case_sha256"] == old["case_sha256"]
            and r["checkpoint_sha256"] == old["checkpoint_sha256"]
            and all(
                r["model"][k] == old["models"][method][k]
                for k in ("threshold", "model_id", "model_index_sha256", "weights_sha256")
            ),
            "model, original Case, threshold or frozen recovery checkpoint differs",
        )
        for v in r["views"]:
            t = v["id"]
            q = old["views"][t]["methods"][method]
            check(v["render"]["file"]["sha256"] == q["target_prediction_sha256"], "target bytes differ")
            with (
                np.load(read(base, v["arrays"]), allow_pickle=False) as n,
                np.load(read(a.old, q["file"]), allow_pickle=False) as o,
            ):
                numeric = compare(n, o)
                same = np.array_equal(n["risk"].view(np.uint32), o["risk"].view(np.uint32))
            rows.append({"method": method, "target": t, "all_float32_bits_identical": same, **numeric})
    ok = len(rows) == 24 and all(
        x["all_float32_bits_identical"] and x["finite_disagreements"] == x["state_disagreements"] == 0
        for x in rows
    )
    put(
        a.out / "recovery-grade.json",
        {
            "schema": "el-bundle-recovered-scores-grade-1",
            "run_id": os.getenv("EL_RUN_ID"),
            "status": "passed" if ok else "failed",
            "physical_object": "set6",
            "condition": "M4",
            "original_case_sha256": old["case_sha256"],
            "original_score_sha256": sha((a.old / "scores.json").read_bytes()),
            "method_view_count": len(rows),
            "canvas": sum(x["canvas"] for x in rows),
            "finite_pairs": sum(x["finite_together"] for x in rows),
            "rows": rows,
        },
    )
    check(ok, "restored historical scientific scores/states changed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for name, fields in (
        ("score", ("wheel", "case", "proof", "cpu", "out")),
        ("grade", ("new", "old", "out")),
    ):
        q = sub.add_parser(name)
        for n in fields:
            q.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    (score if a.action == "score" else grade)(a)
