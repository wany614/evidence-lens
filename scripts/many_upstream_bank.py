"""Accept a native external fit, materialize a complete bank, and preserve explicit verification of its source roles."""

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np

from dual_records import bind, resolve, selected, stage
from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens._verify import require
from many_upstream_control import imported

ROOT=Path(__file__).resolve().parents[1]


def control(s,tag,side):
    ticket=selected(s,"b-fit-ticket-"+tag)
    dispatch=load(resolve((s,s/"history"),ticket))
    rid,manifest,status=imported(s,side,ticket,dispatch["packet"]["sha256"],dispatch["trial"],"model_fit")
    if status["status"]!="succeeded":
        from dtu_eval import read_pinned
        original=next((p for p in (s/"runs"/rid/"run.json",s/"quarantine"/rid/"run.json") if p.is_file()),None)
        require(original is not None,"the unsuccessful original run record is unavailable")
        native_pin={"run_id":rid,"manifest_sha256":manifest,"member":"run.json","sha256":sha(original)}
        read_pinned((s,s/"history"),native_pin,success=False)
        cmd=["{python}","-B","{source}/scripts/many_upstream_bank.py","failed-worker","--config","{config}","--out","{out}","--store",str(s),
             "--input","transport","{input:return_receipt}","--input","ticket","{input:ticket}"]
        details={"native_record":native_pin,"status":status["status"],"transport_sha256":status["sha256"]}
        pins={"ticket":ticket}
        if dispatch.get("model_freeze") is not None:
            require(dispatch["model_freeze"]==selected(s,"b-model-freeze"),"failed fit ticket differs from official confirmation freeze")
            pins["model_freeze"]=dispatch["model_freeze"]
            cmd+=["--input","model_freeze","{input:model_freeze}"]
        return stage(s,"b-terminal-failure-"+tag,"archive_verify",pins,details,cmd,"payload/failure.json",5000,
                     extra_files={"return_receipt":side})
    asset=bind(s,rid,"payload/asset.json");doc=load(resolve((s,s/"history"),asset))
    require(doc["ticket_run"]==ticket["run_id"] and doc["source_package"]==dispatch["data"] and doc["run_id"]==rid
            and doc["R"]==dispatch["R"] and doc["H"]==dispatch["H_poses_only"],"native fit used different authorized source data")
    pins={"ticket":ticket,"data":dispatch["data"],"asset":asset,"sfm":dispatch["sfm"]}
    for k,role in (("checkpoint","checkpoint"),("training","training"),("parity","native_parity"),("probe","depth_probe")):
        pins[k]=bind(s,rid,"payload/"+doc[role]["path"]);require(pins[k]["sha256"]==doc[role]["sha256"],"native "+role+" hash differs")
    for i,k in enumerate(doc["R"]+doc["H"]):
        row=doc["frames"][k];name=f"frame-{i:04d}"
        pins[name]=bind(s,rid,"payload/"+row["path"]);require(pins[name]["sha256"]==row["sha256"],"changed native prediction")
    runtime=selected(s,"b-score-runtime")
    incoming=load(s/"runs"/runtime["run_id"]/"inputs.json")["inputs"]
    row,=[item for item in incoming if item["role"]=="prototype"]
    wheel=bind(s,row["parent_run_id"],"payload/"+Path(row["path"]).name)
    require(wheel["sha256"]==load(resolve((s,s/"history"),runtime))["prototype_wheel_sha256"],"different scorer wheel")
    pins.update(scorer_runtime=runtime,scorer_wheel=wheel)
    parent=ROOT/"prototype/many-source"
    extra={"adapter":parent/"upstream/prepare_bank.py","transport":side}
    cmd=["{python}","-B","{source}/scripts/many_upstream_bank.py","worker","--config","{config}","--out","{out}"]
    for k in [*pins,*extra]:cmd+=["--input",k,"{input:"+k+"}"]
    return stage(s,"b-bank-"+tag,"dataset_build",pins,{"native_run":rid,"native_manifest_sha256":manifest,
                 "native_container_sha256":status["sha256"],"native_container_bytes":status["bytes"]},cmd,"payload/bank.json",460_000_000,extra_files=extra)


def worker(cfg,inputs,out):
    doc=load(inputs["asset"]);fit=load(inputs["training"]);parity=load(inputs["parity"]);probe=load(inputs["probe"])
    require(sha(inputs["checkpoint"])==doc["checkpoint"]["sha256"],"wrong actual native checkpoint")
    native_ids=doc["R"]+doc["H"]
    with tempfile.TemporaryDirectory(dir=out.parent) as td:
        temp=Path(td);shutil.copyfile(inputs["adapter"],temp/"prepare_bank.py")
        sys.path[:0]=[td,str(inputs["scorer_wheel"])]
        from prepare_bank import materialize, frames
        first=inputs["frame-0000"].read_bytes()
        with np.load(io.BytesIO(first),allow_pickle=False) as r:
            deficient=io.BytesIO();np.savez_compressed(deficient,rgb=r["rgb"],alpha=r["alpha"])
        try:frames(deficient.getvalue())
        except ValueError as exc:negative=str(exc)
        else:raise ValueError("missing expected camera-Z was accepted")
        evidence={k:cfg["sources"][k] for k in ("ticket","data","asset","sfm","checkpoint","parity","probe","training")}
        present={k:inputs[f"frame-{i:04d}"] for i,k in enumerate(native_ids)}
        inspect=materialize(inputs["data"],doc,fit,parity,probe,lambda key,spec:present[key].read_bytes(),out,evidence)
    src=doc["source_self"]
    save(out/"upstream-evidence.json",{"schema":"el-many-upstream-bank-proof-1","run_id":os.environ["EL_RUN_ID"],
         "native":{"run_id":cfg["native_run"],"manifest_sha256":cfg["native_manifest_sha256"],"container_sha256":cfg["native_container_sha256"],"bytes":cfg["native_container_bytes"]},
         "bank_sha256":sha(out/"bank.json"),"verified":inspect,"checkpoint_sha256":doc["checkpoint"]["sha256"],"gaussians":doc["gaussians"],
         "parameters":fit["native_parameters"],"input_source_visits":fit["source_visits"],"source_self":src,"source_only_mean":doc["source_only_mean_rgb"],
         "mean_source_self_mae":float(np.mean([r["mae"] for r in src])),"mean_source_colour_baseline_mae":float(np.mean([r["source_mean_rgb_baseline_mae"] for r in src])),
         "runtime":doc["runtime"],"native_frame_parity":parity["frames"],"native_registry":evidence,"partial_input_rejection":negative})


def failed(cfg,inputs,out,store):
    from dtu_eval import read_pinned
    _,original=read_pinned((store,store/"history"),cfg["native_record"],success=False)
    run=load(original);dispatch=load(inputs["ticket"]);side=load(inputs["transport"])
    freeze=cfg["sources"].get("model_freeze")
    require(freeze==dispatch.get("model_freeze"),"fit failure scope differs from its dispatched model freeze")
    if freeze is not None:
        require(load(inputs["model_freeze"])["schema"]=="el-many-confirmation-model-freeze-1"
                and sha(inputs["model_freeze"])==freeze["sha256"],"unverified confirmation model freeze")
    require(sha(original)==cfg["native_record"]["sha256"] and run["run_id"]==cfg["native_record"]["run_id"]
            and side["manifest_sha256"]==cfg["native_record"]["manifest_sha256"]
            and run["execution_status"]==side["status"]==cfg["status"] and side["worker_exit_code"]!=0
            and run["trial_id"]==dispatch["trial"] and dispatch["permit_one_fit"] is True,
            "the unsuccessful native record and immutable fit ticket disagree")
    save(out/"failure.json",{"schema":"el-many-terminal-failure-1","run_id":os.environ["EL_RUN_ID"],
         "family":dispatch["scene_id"],"R":len(dispatch["R"]),"target_ids":dispatch["H_poses_only"],
         "failed_native_run":cfg["native_record"],"reason":run["failure_reason"],"status":run["execution_status"],
         "worker_exit_code":side["worker_exit_code"],"elapsed_seconds":run["elapsed_seconds_monotonic"],
         "planned_fit_consumed":1,"ticket":cfg["sources"]["ticket"],"transport_sha256":side["sha256"],
         "model_freeze":freeze,"model_freeze_sha256":freeze["sha256"] if freeze else None})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("receive","worker","failed-worker"))
    for k in ("store","receipt","config","out"):p.add_argument("--"+k,type=Path)
    p.add_argument("--tag");p.add_argument("--input",action="append",nargs=2,default=[]);a=p.parse_args()
    if a.action=="receive":print(json.dumps(control(a.store.resolve(),a.tag,a.receipt)))
    elif a.action=="failed-worker":failed(load(a.config),{k:Path(v) for k,v in a.input},a.out,a.store.resolve())
    else:worker(load(a.config),{k:Path(v) for k,v in a.input},a.out)
