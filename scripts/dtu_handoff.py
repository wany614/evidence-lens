"""Record D1 conversions or create source-only paired Windows jobs from explicit sealed runs."""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from evidence_lens._bundle import encoded,load,sha
from evidence_lens._verify import require,verify_bundle
from dtu_convert import opening
from make_handoff import publish,sha_bytes,zip_bytes
from stage3_jobs import CORE,signature
from stage3_reference import located,perform
from dtu_reconstruct_windows import OPT_SHA,SETS,TARGETS

BASELINE={"run_id":"20261123T154624482736Z-1429371f24844d16","manifest_sha256":"7cb8ab96c18f204ea5fd3b7792e9a56e120a6fcb4a2c833c4cd3175c1f176690"}
RIG={"run_id":"20261123T193804778622Z-622e5e9ca6ef434b","manifest_sha256":"8fe4dde3b7e95e5b49d6d43bbe9f57e123c00954e4fe3b49e41860880b7269ac"}


def pin(value):
    parts=value.split("=")
    require(len(parts)==2,"Expected RUN_ID=MANIFEST_SHA256")
    return {"run_id":parts[0],"manifest_sha256":parts[1]}


def payload(store,p):
    root=located(store,p)
    actual=verify_bundle(root)
    require(actual["manifest_sha256"]==p["manifest_sha256"] and actual["run"]["execution_status"]=="succeeded","Invalid sealed parent")
    return root/"payload"



def recorded_opening(store, p, original_path, expected, role):
    require(p is not None,"Physical holdout access requires the explicit sealed authority parent")
    stage=payload(store,p)
    filename="opening.json" if role=="sources" else "reference-gate.json"
    original=stage/filename
    marker=opening("set6",original_path,expected,role)
    admin=load(stage.parent/"run.json")
    require(sha(original)==expected and marker["run_id"]==p["run_id"] and
            admin["experiment_id"]=="d1-dtu-authority" and admin["trial_id"]==("d1-dtu-opening" if role=="sources" else "d1-dtu-reference"),
            "A different recorder produced the proposed external access marker")
    registered=store/"dtu-authority"/"dtu-crop256-v1"
    freeze=load(registered/"freeze.json")
    require(sha(registered/filename)==expected and sha(registered/"freeze.json")==marker["freeze_sha256"] and
            freeze["camera_plan_sha256"]==sha(payload(store,RIG)/"camera-plan.json") and freeze["protocol"]["external_objects"]==["set6"],
            "External access does not match the single registered freeze")
    return original


def snapshot(project,src,names):
    for name in names:
        target=src/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(project/name,target)


def convert(a):
    project,store=a.project.resolve(),a.store.resolve()
    op=opening(a.object_id,a.opening,a.opening_sha256,a.role)
    if op: recorded_opening(store,a.opening_pin,a.opening,a.opening_sha256,a.role)
    rig=payload(store,RIG); plan=load(rig/"camera-plan.json")
    raw=payload(store,a.download)
    reg=load(raw/"members.json")
    roles=plan["source_view_ids"] if a.role=="sources" else plan["target_view_ids"]
    names={k:"SampleSet/MVS Data/Rectified/scan"+a.object_id[3:]+"/rect_%03d_3_r5000.png"%i for k,i in roles.items()}
    require(set(names.values())=={r["member"] for r in reg["files"]},"Different explicit DTU acquisition or role")
    pins={"rig":RIG,"download":a.download}
    inputs={"plan":(rig/"camera-plan.json","rig"),"crop":(rig/"crop.json","rig"),"members":(raw/"members.json","download")}
    inputs.update({v:(raw/"members"/name,"download") for v,name in names.items()})
    args=["{python}","{source}/scripts/dtu_convert.py","--object-id",a.object_id,"--role",a.role,"--out","{out}"]
    for n in ("plan","crop","members"):args+=["--"+n,"{input:"+n+"}"]
    for n in roles:args+=["--image",n,"{input:"+n+"}"]
    if op:
        require(a.opening_pin,"Need sealed opening parent")
        orig=recorded_opening(store,a.opening_pin,a.opening,a.opening_sha256,a.role)
        pins["opening"]=a.opening_pin;inputs["opening"]=(orig,"opening")
        args+=["--opening","{input:opening}","--opening-sha256",a.opening_sha256]
    cfg={"experiment_id":"d1-dtu-crop256","trial_id":"d1-"+a.object_id+"-"+a.role+"-v1","scene_family_id":a.object_id,
         "dataset_id":"official-dtu-sampleset-2018","dataset_hash":reg["archive"]["directory_sha256"],
         "role":a.role,"camera_set_id":"dtu-crop256-v1","camera_set_hash":plan["camera_sha256"],"opening_sha256":a.opening_sha256,
         "recording":{"expected_output_bytes":2400000}}
    scratch=store.parent/"dtu-case-scratch";scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as work:
        src=Path(work)/"engine";snapshot(project,src,("scripts/dtu_convert.py","src/evidence_lens/dtu_camera.py"))
        proof=perform(store,project,src,cfg,"derivation",inputs,pins,args)
    return {"pin":proof,"file_sha256":sha(located(store,proof)/"payload"/("source-images.json" if a.role=="sources" else "references.json"))}


def build(a):
    project,store=a.project.resolve(),a.store.resolve()
    opening(a.object_id,a.opening,a.opening_sha256)
    if a.object_id=="set6": recorded_opening(store,a.opening_pin,a.opening,a.opening_sha256,"sources")
    rig=payload(store,RIG); src=payload(store,a.source); base=payload(store,BASELINE)
    plan=load(rig/"camera-plan.json"); srcmeta=load(src/"source-images.json"); plan_sha=sha(rig/"camera-plan.json")
    require(srcmeta["object_id"]==a.object_id and srcmeta["camera_plan_sha256"]==plan_sha and srcmeta["opening_sha256"]==a.opening_sha256,"Source conversion does not match")
    require(sha(base/"optimizer.json")==OPT_SHA,"Original production optimizer changed")
    names=["scripts/worker_jobs.py","scripts/dtu_reconstruct_windows.py","scripts/dtu_render.py","configs/storage.json",
           "src/evidence_lens/stage3_gaussians.py",*("src/evidence_lens/"+n+".py" for n in CORE)]
    worker={n:(project/n).read_bytes() for n in names}
    wh=sha_bytes(zip_bytes(worker)); packages={};entries=[]
    for c,ids in SETS.items():
        tid="d1-dtu-"+a.object_id+"-"+c+"-256"
        cams={n+".json":(rig/"condition"/c/(n+".json")).read_bytes() for n in ("source_cameras","render_cameras")}
        for n in cams:
            require(signature(cams[n])==plan["condition_cameras"][c][n.removesuffix(".json")],"Registered condition cameras changed")
        src_images={}
        for s in ids:
            name="sources/"+s+".png"
            meta=srcmeta["views"][s]["rgb"];img=(src/name).read_bytes()
            require(meta=={"path":name,**signature(img)},"Original source conversion differs")
            src_images[name]=img
        assets={**cams,**src_images,"optimization.json":(base/"optimizer.json").read_bytes()}
        identity={"dataset_hash":srcmeta["archive"]["directory_sha256"],"dataset_id":"official-dtu-sampleset-2018",
                  "scene_id":a.object_id,"scene_family_id":a.object_id,"scene_version":"SampleSet-cal18-r5000",
                  "scene_hash":sha(src/"source-images.json"),"split":"development" if a.object_id=="set1" else "test",
                  "camera_set_id":"dtu-crop256-v1","camera_set_hash":plan["condition_cameras"][c]["render_cameras"]["sha256"],"condition_id":c}
        job={"schema":"el-dtu-job-1","trial_id":tid,"experiment_id":"d1-dtu-source-only-256","condition":c,"object_id":a.object_id,
             "identity":identity,"generator_source_ids":ids,"target_ids":TARGETS,"camera_plan_sha256":plan_sha,
             "reference":a.source,"readback":RIG,"budget":BASELINE,"worker_sha256":wh,"optimization_steps":256,
             "opening":a.opening_pin,"opening_sha256":a.opening_sha256,"files":{k:signature(v) for k,v in assets.items()}}
        raw=encoded(job); container=zip_bytes({**assets,"job.json":raw});name="jobs/"+tid+".zip"
        packages[name]=container
        entries.append({"trial_id":tid,"condition":c,"path":name,"job_sha256":sha_bytes(raw),**signature(container)})
    manifest={"schema":"el-dtu-worker-batch-1","batch":a.batch,"object_id":a.object_id,"camera_plan_sha256":plan_sha,"worker_sha256":wh,
              "worker_files":{n:signature(b) for n,b in worker.items()},"jobs":entries}
    b=encoded(manifest); container=zip_bytes({"batch.json":b,**packages,**{"worker/"+k:v for k,v in worker.items()}})
    dest=a.out
    publish(dest/"batch.json",b);publish(dest/(a.batch+".zip"),container)
    result={"batch":a.batch,"batch_sha256":sha_bytes(b),"container_sha256":sha_bytes(container),"container_bytes":len(container),
            "worker_sha256":wh,"camera_plan_sha256":plan_sha,"source_conversion":a.source,"rig":RIG,"optimizer":BASELINE,
            "opening":a.opening_pin,"opening_sha256":a.opening_sha256,"jobs":entries}
    publish(dest/"delivery.json",encoded(result))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    p=parser.add_subparsers(dest="action",required=True)
    for action in ("convert","batch"):
        q=p.add_parser(action)
        for n,d in (("store",".research"),("project",".")):q.add_argument("--"+n,type=Path,default=Path(d))
        q.add_argument("--object-id",required=True,choices=("set1","set6"));q.add_argument("--opening",type=Path)
        q.add_argument("--opening-sha256");q.add_argument("--opening-pin",type=pin)
        if action=="convert":
            q.add_argument("--role",required=True,choices=("sources","references"));q.add_argument("--download",type=pin,required=True)
        else:
            q.add_argument("--batch",required=True);q.add_argument("--source",type=pin,required=True);q.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(convert(args) if args.action=="convert" else build(args),sort_keys=True))
