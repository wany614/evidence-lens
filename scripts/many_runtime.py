"""Build the separately versioned source-bank distribution and provision a verified CPU environment."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile

from dual_records import bind, selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require


WHEEL="evidence_lens_many_source-0.1.0.dev0-py3-none-any.whl"
BASE_WHEEL="evidence_lens-0.5.0-py3-none-any.whl"
PACKAGES=("numpy==2.5.5","Pillow==12.3.0","torch==2.8.0","safetensors==0.8.0","psutil==7.2.2","pyarrow==26.0.1")


def controller(a):
    s=a.store
    pins={"study":selected(s,"b-study-registration"),"baseline":selected(s,"a-baseline")}
    if a.action=="build":
        root=Path("prototype/many-source").resolve()
        paths=[root/"pyproject.toml", root/"LICENSE",root/"NOTICE",*sorted((root/"src/evidence_lens_many").glob("*.py"))]
        paths+=[p for p in [root/"README.md"] if p.is_file()]
        require(all(f.is_file() for f in paths) and (root/"src/evidence_lens_many/cli.py") in paths,
                "incomplete prototype or rights inventory")
        extra={"f"+str(i):f for i,f in enumerate(paths)}
        names={k:v.relative_to(root).as_posix() for k,v in extra.items()}
        return stage(s,a.label or "b-score-core-build","archive_verify",pins,{"files":names,"version":"0.1.0.dev0"},
            ["{python}","scripts/many_runtime.py","build-worker","--config","{config}","--out","{out}",
             "--files",*["{input:"+r+"}" for r in extra]],"payload/build.json",500000,extra)
    build=selected(s,a.build_label)
    pins["prototype"]=bind(s,build["run_id"],"payload/"+WHEEL)
    pins["basewheel"]=bind(s,selected(s,"a-release-build")["run_id"],"payload/"+BASE_WHEEL)
    env=s/"work/many-env"/pins["prototype"]["sha256"][:20]
    return stage(s,a.label or "b-score-runtime","archive_verify",pins,{"exact_dependencies":PACKAGES,"installation":"offline isolated CPU"},
        ["{python}","scripts/many_runtime.py","env-worker","--base","{input:basewheel}","--prototype","{input:prototype}",
         "--directory",str(env),"--out","{out}"],"payload/runtime.json",30000)


def build_worker(a):
    cfg=load(a.config)
    with tempfile.TemporaryDirectory(dir=a.out.parent) as temp:
        root=Path(temp)/"project"
        for i,path in enumerate(a.files):
            role="f"+str(i)
            name=cfg["files"][role]
            require(sha(path)==cfg["extra_sha256"][role],"different preregistered source "+name)
            dest=root/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest)
        subprocess.run(["uv","build","--offline","--wheel","--out-dir",a.out,root],check=True,
                       env={**os.environ,"SOURCE_DATE_EPOCH":"946684800"})
        wheel=a.out/WHEEL
        with ZipFile(wheel) as z:
            files=z.namelist()
            require(len(files)==len(set(files)) and
                    all(n.startswith(("evidence_lens_many/","evidence_lens_many_source-0.1.0.dev0.dist-info/")) for n in files),
                    "unregistered namespace in wheel")
            expected={f.relative_to(root/"src").as_posix():f for f in (root/"src").rglob("*.py")}
            require(set(expected)=={n for n in files if n.startswith("evidence_lens_many/")} and
                    all(z.read(n)==f.read_bytes() for n,f in expected.items()),"changed installed module")
            for f in ("LICENSE","NOTICE"):
                names=[n for n in files if n.endswith("/"+f)]
                require(len(names)==1 and z.read(names[0])==(root/f).read_bytes(),"missing wheel notice "+f)
            rows=list(csv.reader(io.StringIO(z.read(next(n for n in files if n.endswith("/RECORD"))).decode())))
            require({r[0] for r in rows}==set(files),"wheel record is incomplete")
            members={n:hashlib.sha256(z.read(n)).hexdigest() for n in files if n.startswith("evidence_lens_many/")}
    save(a.out/"build.json",dict(schema="el-many-source-build-1",run_id=os.environ["EL_RUN_ID"],wheel=WHEEL,
         bytes=wheel.stat().st_size,sha256=sha(wheel),dependencies=["evidence-lens==0.5.0"],source_members=members,
         complete_input_sha256={name:cfg["extra_sha256"][key] for key,name in cfg["files"].items()}))


def env_worker(a):
    env={k:v for k,v in os.environ.items() if k not in ("PYTHONPATH","PYTHONHOME","VIRTUAL_ENV")}
    env.update(PYTHONNOUSERSITE="1",CUDA_VISIBLE_DEVICES="-1",PYTHONDONTWRITEBYTECODE="1")
    a.directory.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(["uv","venv","--no-project","--allow-existing","--python",sys.executable,a.directory],check=True,env=env)
    py=a.directory/"bin/python"
    subprocess.run(["uv","pip","install","--offline","--python",py,a.base,a.prototype,*PACKAGES],check=True,env=env)
    script="""import hashlib,importlib.metadata as m,json,pathlib,sys,zipfile
import evidence_lens_many as many
import evidence_lens_many.evaluation
for wheel in sys.argv[1:]:
    with zipfile.ZipFile(wheel) as z:
        for n in z.namelist():
            if '.dist-info/' in n: continue
            p=pathlib.Path(m.distribution('evidence-lens-many-source').locate_file(n))
            assert p.read_bytes()==z.read(n),n
print(json.dumps({'packages':{d.metadata['Name']:d.version for d in m.distributions()},
                  'feature_schema_sha256':many.FEATURE_SCHEMA_SHA256,'columns':list(many.FEATURE_NAMES)}))"""
    raw=subprocess.check_output([py,"-I","-B","-c",script,a.base,a.prototype],env=env,text=True)
    data=json.loads(raw)
    for n in ("evidence-lens","evidence-lens-bank"):
        subprocess.run([a.directory/"bin"/n,"--help"],check=True,stdout=subprocess.DEVNULL,env=env)
    save(a.out/"runtime.json",dict(schema="el-many-verified-cpu-runtime-1",run_id=os.environ["EL_RUN_ID"],
        base_wheel_sha256=sha(a.base),prototype_wheel_sha256=sha(a.prototype),python=sys.version,
        relative_executable="work/many-env/"+a.directory.name+"/bin/python",**data))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["build","env","build-worker","env-worker"])
    for key in ("store","config","out","directory","base","prototype"): p.add_argument("--"+key,type=Path)
    p.add_argument("--files",nargs="+",type=Path)
    p.add_argument("--build-label",default="b-score-core-build")
    p.add_argument("--label")
    a=p.parse_args()
    if a.store:print(controller(a))
    elif a.action=="build-worker":build_worker(a)
    else:env_worker(a)
