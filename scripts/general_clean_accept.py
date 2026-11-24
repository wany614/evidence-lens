"""Install the exact downloaded wheel in new environments and replay the original released arrays."""
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


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def call(args, env):
    process=subprocess.run([str(v) for v in args],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if process.returncode:
        raise RuntimeError(f"{Path(args[0]).name} exited {process.returncode}: {process.stderr[-1800:]} {process.stdout[-450:]}")
    return process.stdout


def verify_files(folder, manifest_sha):
    path=folder/"release-manifest.json"
    if sha(path)!=manifest_sha:
        raise ValueError("unexpected release manifest")
    doc=read(path)
    if doc["schema"]!="el-general-release-1":
        raise ValueError("unsupported release manifest")
    assets={}
    for name,spec in doc["assets"].items():
        if spec["role"] in ("wheel","original_replays"):
            if Path(name).name!=name or Path(name).is_absolute():
                raise ValueError("invalid release filename")
            f=folder/name
            if f.stat().st_size!=spec["bytes"] or sha(f)!=spec["sha256"]:
                raise ValueError("different release asset: "+name)
            assets[spec["role"]]=f
    return doc,assets


def acceptance(a):
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError("use a new output folder")
    a.out.mkdir(parents=True,exist_ok=True)
    doc,assets=verify_files(a.downloads,a.manifest_sha256)
    env={k:v for k,v in os.environ.items() if k not in ("PYTHONPATH","PYTHONHOME","VIRTUAL_ENV")}
    env.update(PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="-1",PIP_DISABLE_PIP_VERSION_CHECK="1")
    checks={}
    with tempfile.TemporaryDirectory(dir=a.out.parent) as td:
        tmp=Path(td)
        examples=tmp/"examples"
        with zipfile.ZipFile(assets["original_replays"]) as z:
            z.extractall(examples)
        if sha(examples/"expected.json")!=doc["original_gold_sha256"]:
            raise ValueError("unexpected replay oracle")
        for group in (["default"] if a.default_only else ["default","vision"]):
            path=tmp/group
            uv=shutil.which("uv") if not a.wheelhouse else None
            if uv:
                call([uv,"venv","--no-project","--python",sys.executable,path],env)
            else:
                venv.EnvBuilder(with_pip=True).create(path)
            exe=path/("Scripts/python.exe" if os.name=="nt" else "bin/python")
            installer=([uv,"pip","install","--python",exe] if uv else [exe,"-I","-m","pip","install"])
            if a.wheelhouse:
                installer+=["--no-index","--find-links",a.wheelhouse.resolve()]
            elif a.offline and uv:
                installer+=["--offline"]
            elif a.offline:
                raise ValueError("offline installation requires uv's cache or a wheelhouse")
            deps=["numpy==2.5.5","Pillow==12.3.0"]
            if group=="vision":
                if a.models is None:
                    raise ValueError("supply the complete original models for optional replay")
                deps+=["torch==2.8.0","safetensors==0.8.0","psutil==7.2.2"]
            before=time.monotonic()
            call([*installer,*deps,assets["wheel"].resolve()],env)
            inventory=json.loads(call([exe,"-I","-c","import importlib.metadata as m,json; print(json.dumps({d.metadata['Name'].lower():d.version for d in m.distributions()}))"],env))
            if inventory["evidence-lens"]!=doc["tag"][1:] or (group=="default" and set(inventory)-{"numpy","pillow","evidence-lens","pip","setuptools","wheel"}):
                raise ValueError("wrong environment or unrequested heavy dependency in default replay")
            install_s=time.monotonic()-before
            method="MS-L1" if group=="default" else "all" if a.four else "MS-ft"
            dest=a.out/group
            cmd=[exe,"-I","-B",examples/"accept.py","--root",examples,"--sequence","bell-M4",
                 "--method",method,"--out",dest.resolve(),"--expected-sha256",doc["original_gold_sha256"]]
            if group=="vision":
                cmd+=["--models",a.models.resolve(),"--threads","4"]
            before=time.monotonic()
            call(cmd,env)
            p=dest/"acceptance.json";r=read(p)
            if r["installed_distribution"]!=inventory["evidence-lens"] or r["gold_index_sha256"]!=doc["original_gold_sha256"]:
                raise ValueError("acceptance is for a different software or original")
            checks[group]={"method":method,"status":r["status"],"bit_exact":r["bit_exact"],
                           "acceptance_sha256":sha(p),"bundle_sha256":r["bundle_sha256"],"install_seconds":install_s,
                           "grade_seconds_including_HTML":time.monotonic()-before,"distributions":inventory}
    result={"schema":"el-general-clean-acceptance-1","run_id":os.getenv("EL_RUN_ID"),"tag":doc["tag"],
            "code_commit":doc["code_commit"],"wheel_sha256":sha(assets["wheel"]),"archive_sha256":sha(assets["original_replays"]),
            "platform":{"os":platform.system(),"version":platform.release(),"arch":platform.machine(),"python":platform.python_version()},
            "cuda_visible_devices":"-1","groups":checks}
    (a.out/"acceptance.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("downloads","out"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--manifest-sha256",required=True)
    p.add_argument("--models",type=Path);p.add_argument("--wheelhouse",type=Path)
    p.add_argument("--offline",action="store_true");p.add_argument("--four",action="store_true")
    p.add_argument("--default-only",action="store_true")
    print(json.dumps(acceptance(p.parse_args()),sort_keys=True))
