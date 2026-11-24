"""Run all four released models against authenticated prepared Bundles in isolated CPU environments."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from zipfile import ZipFile

from bundle_build import environment, run
from bundle_original import CPU, INDEX, METHODS, check, put, sha


def released_models(source, dest):
    with source.open("rb") as f:
        check(hashlib.file_digest(f, "sha256").hexdigest() == CPU, "wrong CPU ZIP")
    with ZipFile(source) as z:
        raw = z.read("release.json")
        check(sha(raw) == INDEX, "wrong released model index")
        dest.mkdir()
        (dest / "release.json").write_bytes(raw)
        files = json.loads(raw)["files"]
        for n, spec in files.items():
            if n != "cpu.py" and not n.startswith("models/"):
                continue
            p = dest / n
            check(p.resolve().is_relative_to(dest.resolve()), "unsafe release name")
            b = z.read(n)
            check(len(b) == spec["bytes"] and sha(b) == spec["sha256"], n)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)


def score(a):
    with tempfile.TemporaryDirectory(dir=a.out.parent) as tmp:
        temp = Path(tmp)
        cli, base = environment(temp, a.wheel)
        original = temp / "original"
        released_models(a.cpu, original)
        visu = temp / "visual"
        visu.mkdir()
        opt, _ = environment(visu, a.wheel)
        python = opt.with_name("python.exe" if os.name == "nt" else "python")
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
        optional = json.loads(
            run(
                [
                    python,
                    "-I",
                    "-B",
                    "-c",
                    "import torch,safetensors,json; print(json.dumps({'torch':torch.__version__,'safetensors':safetensors.__version__,'cuda_available':torch.cuda.is_available()}))",
                ]
            )
        )
        support = {}
        for condition in ("M2", "M4"):
            bundle = a.bundles / condition
            bsha = sha((bundle / "bundle.json").read_bytes())
            for method in METHODS:
                exe = opt if method in ("MS-frozen", "MS-ft") else cli
                out = a.out / "scores" / condition / method
                cmd = [
                    exe,
                    "score",
                    "--bundle",
                    bundle,
                    "--bundle-sha256",
                    bsha,
                    "--model",
                    method,
                    "--device",
                    "cpu",
                    "--out",
                    out,
                ]
                if method != "MS-L1":
                    cmd += ["--models", original]
                start = time.perf_counter()
                v = json.loads(run(cmd))
                elapsed = time.perf_counter() - start
                support[condition + "/" + method] = {
                    "score_sha256": sha((out / "scores.json").read_bytes()),
                    "counts": v["counts"],
                    "process_seconds": elapsed,
                }
        html = a.out / "offset-bell-cage-report.html"
        s = a.out / "scores/M4/MS-L1"
        made = json.loads(
            run(
                [
                    cli,
                    "report",
                    "--scores",
                    s,
                    "--scores-sha256",
                    support["M4/MS-L1"]["score_sha256"],
                    "--out",
                    html,
                ]
            )
        )
        put(
            a.out / "outcomes.json",
            {
                "schema": "el-bundle-recorded-cpu-1",
                "run_id": os.getenv("EL_RUN_ID"),
                "model_release_sha256": INDEX,
                "wheel_sha256": sha(a.wheel.read_bytes()),
                "base_installed_distributions": base,
                "optional_environment": optional,
                "device": "cpu",
                "declared_target_reference_inputs": 0,
                "results": support,
                "html": made,
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("wheel", "bundles", "cpu", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    score(p.parse_args())
