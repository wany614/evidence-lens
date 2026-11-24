"""Run the upstream-specific negative controls against the sealed v0.5 wheel in a fresh CPU environment."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from xml.etree import ElementTree

from dual_records import bind, selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens_bundle.common import require


WHEEL = "evidence_lens-0.5.0-py3-none-any.whl"
PACKAGES = ("pytest==9.2.0", "torch==2.8.0", "numpy==2.5.5", "Pillow==12.3.0")


def control(store):
    build = selected(store, "a-release-build")
    wheel = bind(store, build["run_id"], "payload/" + WHEEL)
    return stage(store, "a-upstream-wheel-tests", "archive_verify", {"build": build, "wheel": wheel},
                 {"packages": PACKAGES, "inputs": "generated pinhole bead fixture, tiny checkpoint, sealed wheel"},
                 ["{python}", "-B", "{source}/scripts/v05_upstream_tests.py", "worker", "--build", "{input:build}",
                  "--wheel", "{input:wheel}", "--out", "{out}"], "payload/tests.json", 15000)


def test(build, wheel, out):
    require(wheel.name == WHEEL and load(build)["assets"][WHEEL]["sha256"] == sha(wheel), "incorrect release wheel")
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"}}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", UV_OFFLINE="true")
    with tempfile.TemporaryDirectory(dir=out.parent) as td:
        scratch = Path(td)
        for name in ("tests/test_v05_asset_contract.py", "scripts/b2_asset_fixture.py"):
            dest = scratch / name
            dest.parent.mkdir()
            shutil.copyfile(root / name, dest)
        venv = scratch / "env"
        uv = shutil.which("uv")
        require(uv is not None, "uv is required for the offline wheel test")
        subprocess.run([uv, "venv", "--no-project", "--python", sys.executable, venv], env=env, check=True, cwd=scratch)
        python = venv / "bin/python"
        subprocess.run([uv, "pip", "install", "--offline", "--python", python, *PACKAGES, wheel],
                       cwd=scratch, env=env, check=True)
        classes = "import evidence_lens_asset.gaussian as g; import sys; from pathlib import Path; assert Path(g.__file__).is_relative_to(sys.prefix)"
        subprocess.run([python, "-I", "-B", "-c", classes], cwd=scratch, env=env, check=True)
        xml = out / "tests.xml"
        subprocess.run([python, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml", xml,
                        scratch / "tests/test_v05_asset_contract.py"], check=True, cwd=scratch, env=env)
    cases = list(ElementTree.parse(xml).iter("testcase"))
    require(len(cases) == 3 and all(not any(case.find(k) is not None for k in ("skipped", "failure", "error")) for case in cases),
            "an upstream negative-control suite was absent or failed")
    save(out / "tests.json", {"schema": "el-v05-upstream-installed-tests-1", "run_id": os.environ["EL_RUN_ID"],
         "wheel_sha256": sha(wheel), "wheel_code_commit": load(build)["code_commit"], "pass": [case.attrib["name"] for case in cases],
         "failures": 0, "skips": 0, "packages": PACKAGES, "fixture_source_sha256": sha(root / "scripts/b2_asset_fixture.py"),
         "test_source_sha256": sha(root / "tests/test_v05_asset_contract.py"), "junit_sha256": sha(xml)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "worker"))
    for key in ("store", "build", "wheel", "out"):
        parser.add_argument("--" + key, type=Path)
    args = parser.parse_args()
    if args.action == "worker":
        test(args.build, args.wheel, args.out)
    else:
        print(json.dumps(control(args.store.resolve())))
