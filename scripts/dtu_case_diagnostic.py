"""Compare set1 source and target endpoints for the registered 256/1,024-step pair."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from evidence_lens._bundle import encoded,load,sha
from evidence_lens.dtu_camera import CONDITIONS,TARGETS,require,write
from evidence_lens import stage3_gaussians as native
from dtu_case import INIT,OPT,assemble,gpu_probe


def numeric(args):
    # This development-only operation is deliberately incapable of reading an arbitrary object roster.
    request=load(args.request)
    require(request["object_id"]=="set1" and request["conditions"]==["M2","M4"] and request["decision"]["sha256"]==
            "e5ff509f7079b42c0d742c774e5b469dfa7e73258eca24e66391419f3395a927","Incorrect diagnostic authority")
    src=load(args.sources/"source-images.json");ref=load(args.references/"references.json");plan=load(args.rig/"camera-plan.json")
    require(src["object_id"]==ref["object_id"]=="set1" and src["camera_plan_sha256"]==ref["camera_plan_sha256"]==sha(args.rig/"camera-plan.json") and
            all(src[k]==ref[k]==plan[k] for k in ("camera_sha256","view_set_sha256","crop_sha256","lighting_id","source_view_ids","target_view_ids")),"Wrong set1 crop/lighting/reference")
    out={"schema":"el-dtu-paired-set1-budget-diagnostic-1","run_id":os.environ["EL_RUN_ID"],"decision":request["decision"],"object_id":"set1",
         "crop_sha256":ref["crop_sha256"],"reference_sha256":sha(args.references/"references.json"),"event":"mean absolute display-sRGB RGB difference > 0.05 (float32 pixel arithmetic)",
         "conditions":{},"models_scored":0}
    for c in ("M2","M4"):
        cells={}
        for updates in (256,1024):
            label=f"{c}-{updates}"
            row=request["cells"][label]
            require(row["object_id"]=="set1" and row["condition"]==c and row["trial_id"]=="d1-dtu-set1-"+c+("-256" if updates==256 else "-1024-diagnostic"),"Diagnostic cell mismatch")
            gen=getattr(args,"gen_"+c+"_"+str(updates));render=getattr(args,"render_"+c+"_"+str(updates))
            m=load(gen/"reconstruction.json");p=load(render/"prediction.json")
            require(m["run_id"]==row["generation"]["run_id"] and p["run_id"]==row["render"]["run_id"] and p["reconstruction_run_id"]==m["run_id"]
                    and sha(gen/"reconstruction.json")==p["reconstruction_sha256"] and sha(gen/"scene.npz")==p["checkpoint_sha256"]==m["checkpoint_sha256"]
                    and m["sampled_source_pixels"]==updates*256**2 and m["gaussian_optimization_steps"]==updates and
                    m["initial_gaussian_count"]==m["final_gaussian_count"]==4096 and m["topology_events"]==[] and m["optimizer_sha256"]==OPT,
                    "Incorrect fixed-budget selected pair")
            with np.load(gen/"initial.npz",allow_pickle=False) as a,np.load(gen/"scene.npz",allow_pickle=False) as b:
                require(native.digest_values(a)==m["initial_parameter_sha256"]==INIT and native.digest_values(b)==m["final_parameter_sha256"],"Changed checkpoint")
            trace=[json.loads(x) for x in (gen/"optimizer.jsonl").read_text().splitlines()]
            ids=CONDITIONS[c]
            require(len(trace)==updates and sha(gen/"optimizer.jsonl")==m["trace_sha256"] and
                    all(r["update"]==i+1 and r["source"]==ids[i%len(ids)] and r["pixels"]==256**2 and
                        r["image_sha256"]==src["views"][r["source"]]["rgb"]["sha256"] for i,r in enumerate(trace)),"Missing or unauthorized source update")
            # Reuse the supported deployment reader privately, then discard its repeated input bytes.
            with tempfile.TemporaryDirectory() as temp:
                case=assemble(Path(temp),c,args.sources/"sources",args.rig/"condition"/c,render,render)
                probe=gpu_probe(render,case)
                source={}
                for s,(im,frame,_) in case.sources("all").items():
                    measured=float(np.mean(abs(im-frame["rgb"]),dtype=np.float64))
                    require(abs(measured-m["loss_final_by_source"][s])<1e-5 and m["source_sampling"][s]["samples"]==updates//len(ids) and
                            all(v>0 for v in m["source_sampling"][s]["gradient_nonzero_steps"].values()),"Independent source endpoint check failed")
                    source[s]={"mae":measured,"initial":m["loss_initial_by_source"][s],**m["source_sampling"][s]}
                views={}
                for v in TARGETS:
                    info=ref["views"][v]
                    require(info["valid_mask"]=={"kind":"all_true","width":256,"height":256} and info["rgb"]["path"]=="targets/"+v+".png","Different reference domain")
                    f=args.references/info["rgb"]["path"];require(sha(f)==info["rgb"]["sha256"],"Different target photograph")
                    with Image.open(f) as im:truth=np.asarray(im,dtype=np.float32)/np.float32(255)
                    pred=case.frame(v);error=np.mean(np.abs(pred["rgb"]-truth),axis=-1,dtype=np.float32)
                    finite=np.isfinite(error)
                    views[v]={"V":int(error.size),"E":int(finite.sum()),"Y":int((finite & (error>np.float32(.05))).sum()),
                              "rgb_absolute_error_sum":float(error[finite].sum(dtype=np.float64)),
                              "target_mae":float(error[finite].mean(dtype=np.float64)) if finite.any() else None,
                              "positive_alpha":int(np.sum(pred["alpha"]>0)),"finite_ED":int(np.isfinite(pred["z"]).sum()),
                              "target_rgb_sha256":info["rgb"]["sha256"],"prediction_sha256":p["views"][v]["sha256"]}
            E=sum(v["E"] for v in views.values());Y=sum(v["Y"] for v in views.values());err=sum(v["rgb_absolute_error_sum"] for v in views.values())
            cells[str(updates)]={"generation":row["generation"],"render":row["render"],"checkpoint_sha256":m["checkpoint_sha256"],
                   "same_initial_parameter_sha256":INIT,"gaussians":4096,"updates":updates,"source_pixels":updates*256**2,
                   "sources":source,"target_views":views,"target_pooled":{"V":sum(v["V"] for v in views.values()),"E":E,"Y":Y,"Y_over_E":Y/E if E else None,"MAE":err/E if E else None},
                   "source_mean_MAE":float(np.mean([v["mae"] for v in source.values()])),"optimization_s":m["optimization_seconds"],"GPU_probe":probe}
        a,b=cells["256"],cells["1024"]
        require(all(a["sources"][s]["initial"]==b["sources"][s]["initial"] for s in a["sources"]),"Initial source loss differs at the two registered budgets")
        out["conditions"][c]={"endpoints":cells,"delta_1024_minus_256":{"mean_source_mae":b["source_mean_MAE"]-a["source_mean_MAE"],
               "pooled_target_mae":b["target_pooled"]["MAE"]-a["target_pooled"]["MAE"],"target_incidence":b["target_pooled"]["Y_over_E"]-a["target_pooled"]["Y_over_E"]}}
    write(args.out/"diagnostic.json",encoded(out))


def recorded(a):
    from dtu_handoff import payload,pin,RIG,snapshot
    from stage3_reference import perform,located
    from dtu_case_record import CODE
    store=a.store.resolve();project=a.project.resolve()
    root=store/"dtu/intake"
    decision=pin("20261123T194928787292Z-adc6aa0155074a27=dfc00645ab730246790c6e5762035e6415c7b1699f005615c82cc2b00f93badf")
    dpath=payload(store,decision)/"diagnostic-decision.json"
    sources=pin("20261123T194119974535Z-2c998bf23a1a43e8=071ff4efbee311cf66ef09b8f0f164c663a2592dce755f929a1e15c51bea8182")
    refs=pin("20261123T194129171008Z-0fb45a3c1ed94dc1=e5d531f992a36c6c7a809a8a1277ae350dc44198f3326d19485084de024e8c83")
    parents={"decision":decision,"sources":sources,"references":refs,"rig":RIG}
    inputs={n:(payload(store,pin_),n) for n,pin_ in parents.items() if n!="decision"};inputs["decision"]=(dpath,"decision")
    scratch=store.parent/"dtu-case-scratch";scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temp:
        work=Path(temp);code=work/"code";snapshot(project,code,CODE+["scripts/dtu_case_diagnostic.py"])
        request={"object_id":"set1","conditions":["M2","M4"],"decision":{**decision,"sha256":sha(dpath)},"cells":{}}
        cmd=["{python}","{source}/scripts/dtu_case_diagnostic.py","worker","--request","{input:roster}","--out","{out}"]
        for n in ("rig","sources","references"):cmd+=["--"+n,"{input:"+n+"}"]
        for c in ("M2","M4"):
            for u,batch in ((256,"d1-dtu-set1-256"),(1024,"d1-dtu-set1-1024-diagnostic")):
                path=root/batch/c/"selected.json";sel=load(path);require(sel["object_id"]=="set1" and sel["condition"]==c,"Wrong diagnostic selection")
                request["cells"][f"{c}-{u}"]={**sel,"selection_sha256":sha(path)}
                for name,entry in (("gen","generation"),("render","render")):
                    n=name+"_"+c+"_"+str(u)
                    parents[n]=sel[entry];inputs[n]=(payload(store,sel[entry]),n);cmd+=["--"+n,"{input:"+n+"}"]
        req=work/"roster.json";req.write_bytes(encoded(request));inputs["roster"]=(req,None)
        cfg={"experiment_id":"d1-dtu-development-budget-diagnostic","trial_id":"set1-paired-256-1024-native-rgb-error",
             "diagnostic_decision":decision,"recording":{"expected_output_bytes":100000}}
        result=perform(store,project,code,cfg,"evaluation",inputs,parents,cmd)
        return {**result,"diagnostic_sha256":sha(located(store,result)/"payload/diagnostic.json")}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    q=sub.add_parser("record");q.add_argument("--store",type=Path,default=Path(".research"));q.add_argument("--project",type=Path,default=Path("."))
    q=sub.add_parser("worker")
    for n in ("rig","sources","references","request","out",*(f"{k}_{c}_{u}" for c in ("M2","M4") for u in (256,1024) for k in ("gen","render"))):
        q.add_argument("--"+n,required=True,type=Path)
    a=p.parse_args()
    if a.action=="record":print(json.dumps(recorded(a)))
    else:numeric(a)
