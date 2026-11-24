"""Run the distributed source regressions in one fresh offline environment."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from xml.etree import ElementTree
from zipfile import ZipFile

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require


SUITES = (
    "test_b2_asset", "test_bundle", "test_replay", "test_resident",
    "test_t1_confirmation_controls", "test_grid_display", "test_rectangular_scoring",
    "test_rectangular_vision", "test_v05_bell_evaluate",
)
PACKAGES = (
    "numpy==2.5.5", "Pillow==12.3.0", "torch==2.8.0",
    "safetensors==0.8.0", "psutil==7.2.2", "pyarrow==26.0.1", "pytest==9.2.0",
)


def exercise(build, archive, out):
    require(load(build)["assets"][archive.name]["sha256"] == sha(archive), "different source archive")
    env = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"}}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1")
    uv = shutil.which("uv")
    require(bool(uv), "uv is required for this recorded offline regression")
    with TemporaryDirectory(dir=out.parent) as scratch:
        work = Path(scratch)
        project = work / "source"
        with ZipFile(archive) as z:
            z.extractall(project)
        venv = work / "env"
        subprocess.run([uv, "venv", "--no-project", "--python", sys.executable, venv], check=True, env=env)
        python = venv / "bin/python"
        subprocess.run([uv, "pip", "install", "--offline", "--python", python, *PACKAGES], check=True, env=env)
        env["PYTHONPATH"] = os.pathsep.join(str(project / k) for k in ("src", "scripts"))
        xml = out / "tests.xml"
        subprocess.run([python, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml", xml,
                        *["tests/" + name + ".py" for name in SUITES]], cwd=project, env=env, check=True)
    cases = list(ElementTree.parse(xml).iter("testcase"))
    require(cases and all(not any(c.find(k) is not None for k in ("failure", "error", "skipped"))
                          for c in cases), "a registered source regression was absent or failed")
    save(out / "tests.json", dict(schema="el-v05-source-tests-1", run_id=os.environ["EL_RUN_ID"],
         code_commit=load(build)["code_commit"], source_archive_sha256=sha(archive),
         suites=SUITES, packages=PACKAGES, passed=len(cases), skipped=0, junit_sha256=sha(xml)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("store", "build", "archive", "out"):
        parser.add_argument("--" + key, type=Path)
    args = parser.parse_args()
    if args.store:
        from dual_records import bind, selected, stage

        build = selected(args.store, "a-release-build")
        source = bind(args.store, build["run_id"], "payload/evidence-lens-source-v0.5.0.zip")
        print(stage(args.store, "a-immutable-source-tests", "archive_verify", {"build": build, "source": source},
              {"suites": SUITES, "packages": PACKAGES, "network": "offline installation"},
              ["{python}", "scripts/v05_source_tests.py", "--build", "{input:build}", "--archive",
               "{input:source}", "--out", "{out}"], "payload/tests.json", 60000))
    else:
        exercise(args.build, args.archive, args.out)
