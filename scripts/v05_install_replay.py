"""Install old and new wheels in the same locked CPU environment and replay the actual arrays."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import venv
import zipfile


ORIGINAL = "0ed367c24019004d5126c823d899db3941f0fe9dd24473394c3cd6b581c7cfa0"
GOLD = "7aed41037e7c4ec7f0a8583d33bdbe5291d6b02bf0ef8df325a13d337490f9cc"
OLD_WHEEL = "d9ae3890245d91d86d47ec19b654272773846acb0436b0a41eb498ef0e2e6281"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def call(args, env):
    completed = subprocess.run([str(x) for x in args], capture_output=True, text=True, env=env)
    if completed.returncode:
        raise RuntimeError(f"{Path(args[0]).name} exit {completed.returncode}: "
                           f"{completed.stderr[-2200:]} {completed.stdout[-400:]}")
    return completed.stdout


def extract(archive, dest):
    with zipfile.ZipFile(archive) as z:
        seen = set()
        for name in z.namelist():
            p = Path(name)
            if not name or p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name or name.casefold() in seen:
                raise ValueError("invalid or duplicate archive member")
            seen.add(name.casefold())
        z.extractall(dest)


def arrays(grade, methods):
    result = {}
    for m in methods:
        folder = grade/m/"scores"
        receipt = json.loads((folder/"scores.json").read_bytes())
        if [v["id"] for v in receipt["views"]] != [f"v{i:02d}" for i in range(1,7)]:
            raise ValueError("old-grid target roster differs")
        for view in receipt["views"]:
            spec = view["arrays"]
            path = folder/spec["path"]
            if sha(path) != spec["sha256"]:
                raise ValueError("score output bytes differ from receipt")
            with zipfile.ZipFile(path) as z:
                if set(z.namelist()) != {"risk.npy","valid.npy","state.npy","unknown_reason.npy"}:
                    raise ValueError("unexpected returned array set")
                result[m+"/"+view["id"]] = {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}
    return result


def replay(a):
    if sha(a.original) != ORIGINAL or sha(a.old_wheel) != OLD_WHEEL:
        raise ValueError("different historic fixture or wheel")
    if sha(a.wheel) != a.wheel_sha256 or sha(a.examples) != a.examples_sha256:
        raise ValueError("new wheel or example bytes differ")
    a.out.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if k not in ("PYTHONHOME","PYTHONPATH","VIRTUAL_ENV")}
    env.update(PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="-1",
               PIP_DISABLE_PIP_VERSION_CHECK="1")
    uv = None if a.wheelhouse else shutil.which("uv")
    if not uv and not a.wheelhouse:
        raise ValueError("provide the local wheelhouse or use uv with a populated offline cache")
    neural_args = (a.neural_source, a.neural_worker, a.protocol)
    if any(neural_args) and (not all(neural_args) or "vision" not in a.groups):
        raise ValueError("supply all three official verification inputs and the vision installation")
    groups = {}
    with tempfile.TemporaryDirectory(dir=a.out.parent) as scratch:
        tmp=Path(scratch)
        extract(a.original,tmp/"old-examples")
        extract(a.examples,tmp/"new-examples")
        if sha(tmp/"old-examples/expected.json") != GOLD:
            raise ValueError("different original oracle")
        for group in a.groups:
            methods=["MS-L1"] if group=="default" else ["B-L1-transfer","MS-L1","MS-frozen","MS-ft"]
            location = tmp / ("env-"+group)
            if uv:
                call([uv,"venv","--no-project","--python",sys.executable,location],env)
            else:
                venv.EnvBuilder(with_pip=True).create(location)
            exe=location/("Scripts/python.exe" if os.name=="nt" else "bin/python")
            install=([uv,"pip","install","--offline","--python",exe] if uv else
                     [exe,"-I","-m","pip","install","--no-index","--find-links",a.wheelhouse])
            deps=["numpy==2.5.5","Pillow==12.3.0"]
            if group=="vision":
                if not a.models:
                    raise ValueError("real full CPU model package required")
                deps+=["torch==2.8.0","safetensors==0.8.0","psutil==7.2.2"]
            inventory=("import importlib.metadata as m,json;print(json.dumps({d.metadata['Name'].lower():"
                       "d.version for d in m.distributions()}))")
            grades, ids, times, versions = {}, {}, {}, {}
            for version,wheel in [("0.4.0",a.old_wheel),("0.5.0",a.wheel)]:
                tick=time.perf_counter()
                call([*install,*deps,wheel] if version=="0.4.0" else [*install,"--no-deps",wheel],env)
                installed=json.loads(call([exe,"-I","-c",inventory],env))
                if installed.pop("evidence-lens") != version:
                    raise ValueError("wrong installed version")
                if group=="default" and set(installed) - {"numpy","pillow","pip","setuptools","wheel"}:
                    raise ValueError("unexpected default dependency")
                times[version]={"install_seconds":time.perf_counter()-tick}
                versions[version]=installed
                keep = a.keep_scores and version == "0.5.0" and (group == "vision" or len(a.groups) == 1)
                dest=(a.out if keep else tmp)/f"legacy-{group}-{version}"
                cmd=[exe,"-I","-B",tmp/"old-examples/accept.py","--root",tmp/"old-examples","--sequence","bell-M4",
                     "--method","MS-L1" if group=="default" else "all","--expected-sha256",GOLD,"--out",dest]
                if group=="vision":
                    cmd+=["--models",a.models,"--threads","4"]
                tick=time.perf_counter()
                call(cmd,env)
                times[version]["original_seconds"]=time.perf_counter()-tick
                ids[version]=arrays(dest,methods)
                grades[version]=json.loads((dest/"acceptance.json").read_bytes())
                save(a.out/f"legacy-{group}-{version}.json",grades[version])
            if versions["0.4.0"]!=versions["0.5.0"] or ids["0.4.0"]!=ids["0.5.0"]:
                raise ValueError("same-host legacy numerical arrays or dependency versions changed")
            dest=(a.out if a.keep_scores and (group=="vision" or len(a.groups)==1) else tmp)/("profiles-"+group)
            command=[exe,"-I","-B",a.worker,"run","--bundles",tmp/"new-examples","--out",dest,"--methods",*methods]
            if group=="vision":
                command+=["--models",a.models]
            tick=time.perf_counter()
            call(command,env)
            receipt=json.loads((dest/"acceptance.json").read_bytes())
            if len(receipt["cells"]) != 6*len(methods) or any(len(c["views"])!=6 for c in receipt["cells"]):
                raise ValueError("incomplete finite-shape matrix")
            save(a.out/f"profiles-{group}.json",receipt)
            groups[group]={"versions":versions["0.5.0"],"legacy_method_targets_exact":len(ids["0.5.0"]),
                "legacy_array_pairs_exact":4*len(ids["0.5.0"]),"legacy_array_digests":ids["0.5.0"],
                "profiles_method_targets":6*len(receipt["cells"]),"profiles_sha256":sha(a.out/f"profiles-{group}.json"),
                "legacy_grades":{v:sha(a.out/f"legacy-{group}-{v}.json") for v in grades},
                "install_and_original_times":times,"profiles_wall_seconds":time.perf_counter()-tick}
            if group == "vision" and all(neural_args):
                call([exe, "-I", "-B", a.neural_worker, "--official-source", a.neural_source,
                      "--models", a.models, "--protocol", a.protocol, "--threads", "4",
                      "--out", a.out/"neural"], env)
                groups[group]["official_numerical_sha256"] = sha(a.out/"neural/neural.json")
    save(a.out/"installation.json",{"schema":"el-v05-installed-acceptance-1","run_id":os.getenv("EL_RUN_ID"),
         "platform":platform.platform(),"arch":platform.machine(),"python":platform.python_version(),
         "wheel_sha256":sha(a.wheel),"original_wheel_sha256":OLD_WHEEL,"original_examples_sha256":ORIGINAL,
         "examples_sha256":a.examples_sha256,"groups":groups})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("old-wheel","wheel","original","examples","worker","out","models","wheelhouse"):
        p.add_argument("--"+key,type=Path,required=key not in {"models","wheelhouse"})
    for key in ("wheel-sha256","examples-sha256"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--groups",nargs="+",choices=("default","vision"),default=["default","vision"])
    for key in ("neural-source", "neural-worker", "protocol"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--keep-scores",action="store_true")
    replay(p.parse_args())
