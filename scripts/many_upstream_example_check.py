"""Check both published external-adapter fixtures with the registered installed prototype environment."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

from dual_records import bind, resolve, selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from evidence_lens.archive import unpack


def control(s):
    examples=selected(s,"b-upstream-adapter-examples");rec=load(resolve((s,s/"history"),examples))
    runtime=selected(s,"b-score-runtime");env=load(resolve((s,s/"history"),runtime))
    pins={"examples":examples,"runtime":runtime}
    for name in ("complete","incomplete"):
        pins[name]=bind(s,examples["run_id"],"payload/"+rec[name]["path"])
    script=Path(__file__).resolve().parents[1]/"prototype/many-source/upstream/prepare_bank.py"
    c=["{python}","-B","{source}/scripts/many_upstream_example_check.py","worker","--out","{out}","--config","{config}",
       "--python",str(s/env["relative_executable"]),"--input","adapter","{input:adapter}"]
    for name in pins:c+=["--input",name,"{input:"+name+"}"]
    return stage(s,"b-external-import-acceptance","archive_verify",pins,{"scope":"installed independent clean positive and missing expected camera-Z rejection"},
                 c,"payload/acceptance.json",15000,extra_files={"adapter":script})


def worker(cfg,files,python,out):
    a=load(files["examples"]);positive,negative=a["complete"],a["incomplete"]
    require(sha(files["complete"])==positive["sha256"] and sha(files["incomplete"])==negative["sha256"],"changed external fixture")
    with tempfile.TemporaryDirectory(dir=out.parent) as td:
        scratch=Path(td);unpack(files["complete"],scratch/"positive");unpack(files["incomplete"],scratch/"negative")
        part=scratch/"positive";args=[str(python),"-I","-B",str(files["adapter"])]
        env={k:v for k,v in os.environ.items() if k not in {"PYTHONHOME","PYTHONPATH","VIRTUAL_ENV"}}
        env["PYTHONNOUSERSITE"]="1"
        proc=subprocess.run([*args,"prepare","--sources",part/"sources.zip","--asset",part/"native","--out",scratch/"rebuilt",
              "--trusted-asset-sha256",positive["trust_native"],"--trusted-source-sha256",positive["trust_source"],
              "--trusted-lineage-sha256",positive["trust_lineage"]],capture_output=True,text=True,check=True,cwd=scratch,env=env)
        rebuilt=json.loads(proc.stdout);h=sha(scratch/"rebuilt/bank.json")
        require(h==positive["expected_bank"],"standalone exact verified bank reconstruction differs")
        bad=subprocess.run([*args,"inspect","--asset",scratch/"negative/missing-camera-z.npz"],capture_output=True,text=True,check=True,cwd=scratch,env=env)
        actual=json.loads(bad.stdout)
        require(actual["status"]=="incomplete" and "expected RGB, alpha and optical-axis Z" in actual["missing"][0],"missing-depth artifact was accepted")
    save(out/"acceptance.json",{"schema":"el-many-external-adapter-acceptance-1","run_id":os.environ["EL_RUN_ID"],"parents":cfg["sources"],
         "complete":{"bank_sha256":h,"matching_science_bank":True,"inspection":rebuilt},"incomplete":actual,
         "standalone_converter_sha256":sha(files["adapter"]),"native_checkpoint":positive["native_checkpoint"],"photographs_in_negative_fixture":0})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("record","worker"))
    for n in ("store","out","config","python"):p.add_argument("--"+n,type=Path)
    p.add_argument("--input",action="append",nargs=2,default=[]);a=p.parse_args()
    if a.action=="worker":worker(load(a.config),{k:Path(v) for k,v in a.input},a.python,a.out)
    else:print(json.dumps(control(a.store.resolve())))
