"""Reproduce the metadata-only DTU camera registration or independent checkerboard audit in the recorder."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

from evidence_lens._bundle import sha
from dtu_handoff import BASELINE,RIG,payload,pin,snapshot
from stage3_reference import perform,located


def run(a):
    store,project=a.store.resolve(),a.project.resolve()
    control=store.parent/"dtu-case-scratch";control.mkdir(parents=True,exist_ok=True)
    metadata=payload(store,a.metadata)/"members/SampleSet"
    matrices=metadata/"MVS Data/Calibration/cal18"
    with tempfile.TemporaryDirectory(dir=control) as scratch:
        code=Path(scratch)/"engine"
        if a.action=="register":
            parents={"metadata":a.metadata,"index":a.index,"unit":a.evaluator,"baseline":a.baseline}
            inputs={"readme":(metadata/"ReadMe.txt","metadata"),"matrices":(matrices,"metadata"),
                    "directory":(payload(store,a.index)/"directory.json","index"),
                    "evaluator":(payload(store,a.evaluator)/"members/SampleSet/Matlab evaluation code/BaseEval2Obj_web.m","unit"),
                    "baseline":(payload(store,a.baseline)/"cameras.json","baseline")}
            script="scripts/dtu_camera.py";member="camera-plan.json"
            config={"experiment_id":"d1-dtu-rig-registration","trial_id":"d1-camera-plan-v1","dataset_id":"official-dtu-sampleset-2018",
                    "dataset_hash":sha(inputs["directory"][0]),"camera_set_id":"dtu-crop256-v1","recording":{"expected_output_bytes":120000}}
        else:
            parents={"metadata":a.metadata,"rig":a.rig}
            inputs={"calibration":(matrices,"metadata"),"rig":(payload(store,a.rig),"rig")}
            script="scripts/dtu_camera_checkerboard.py";member="checkerboard.json"
            config={"experiment_id":"d1-dtu-independent-camera-audit","trial_id":"d1-dtu-49-checkerboards","recording":{"expected_output_bytes":60000}}
        snapshot(project,code,(script,"src/evidence_lens/dtu_camera.py"))
        cmd=[sys.executable,"{source}/"+script,"--out","{out}"]
        for role in inputs:cmd+=["--"+role,"{input:"+role+"}"]
        result=perform(store,project,code,config,"archive_verify",inputs,parents,cmd)
        return {"pin":result,"member":"payload/"+member,"sha256":sha(located(store,result)/"payload"/member)}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    for action in ("register","checkerboards"):
        q=sub.add_parser(action)
        q.add_argument("--store",type=Path,default=Path(".research"));q.add_argument("--project",type=Path,default=Path("."))
        q.add_argument("--metadata",type=pin,required=True)
        if action=="register":
            q.add_argument("--index",type=pin,required=True);q.add_argument("--evaluator",type=pin,required=True);q.add_argument("--baseline",type=pin,default=BASELINE)
        else:q.add_argument("--rig",type=pin,default=RIG)
    print(json.dumps(run(p.parse_args()),sort_keys=True))
