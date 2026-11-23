"""Build released-format source-only cases and independently audit native D1 scientific returns."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from evidence_lens._bundle import load,sha
from evidence_lens.dtu_camera import CONDITIONS,SOURCES,TARGETS,convert_rgb,crop_geometry,encoded,record,require,write
from evidence_lens.multisource_io import Case
from evidence_lens.scoring_io import write_json
from evidence_lens import stage3_gaussians as native

INIT="4c0bc6c98be6006de05b07ddbfd6468137d644b458a3739e099b2a8e499422d4"
OPT="2d1a704a8c93922326663edb3e27b047e05b1916bc47795d9a43771900fdf122"
OLD_CASES={"M2":"f9065e54062f75c7d38b2f062a2cab4d243f5cdf4c9ff3ecac924db07800c396",
           "M4":"8ab6d51ec5c348fc61121478631a7ead7a9299659337cc8a8a5811922ec1d598"}


def put(data,destination,name,expected=None):
    meta={"path":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
    if expected:require(meta["sha256"]==expected,"Wrong member for "+name)
    write(destination/name,data)
    return meta


def assemble(destination, condition, image_dir, camera_dir, render_dir, prediction_files, certify_identity_rgb=False):
    ids=CONDITIONS[condition]
    p=load(render_dir/"prediction.json")
    require(p["generator_source_ids"]==list(ids) and p["target_ids"]==list(TARGETS) and p["depth_verified"] is True
            and p["rgb_encoding"]=="display_srgb_0_1" and p["depth_semantics"]=="expected_camera_z_m" and
            p["alpha_semantics"]=="accumulated_opacity","Unverified deployed channel or source set")
    m={"schema":"el-stage3-deploy-case-1","condition":condition,"generator_source_ids":list(ids),"target_ids":list(TARGETS),
       "reconstruction_run_id":p["reconstruction_run_id"],"checkpoint_sha256":p["checkpoint_sha256"],
       "prediction":put((render_dir/"prediction.json").read_bytes(),destination,"prediction.json"),"source_images":{},"renders":{}}
    for s in ids:
        data=(image_dir/(s+".png")).read_bytes()
        if certify_identity_rgb:data=convert_rgb(data,crop_geometry(256,256))
        m["source_images"][s]=put(data,destination,"sources/"+s+".png",p["source_rgb_sha256"][s])
    for n,x in (("source_cameras","source_camera_sha256"),("render_cameras","render_camera_sha256")):
        m[n]=put((camera_dir/(n+".json")).read_bytes(),destination,n+".json",p[x])
    for k,row in p["views"].items():
        m["renders"][k]=put((prediction_files/(k+".npz")).read_bytes(),destination,"predictions/"+k+".npz",row["sha256"])
    write_json(destination/"case.json",m)
    return Case(destination,sha(destination/"case.json"))


def published057(a):
    cases={}
    for c,expected in OLD_CASES.items():
        base=a.original/c
        old=Case(base,expected)
        new=assemble(a.out/"cases"/c,c,base/"sources",base,base,base/"predictions",True)
        require(new.sha256==expected,"Published case receipt changed through the external reader")
        views={}
        for v in TARGETS:
            scopes={}
            for scope in ("all","one"):
                old_rgb,of=old.features(v,scope)
                nrgb,nf=new.features(v,scope)
                require(of.keys()==nf.keys() and np.array_equal(nrgb,old_rgb),"Different prediction after external read")
                for field,x in of.items():
                    require(x.shape==nf[field].shape and x.dtype==nf[field].dtype and np.array_equal(x,nf[field],equal_nan=True),
                            "Published deployment feature changed: "+field)
                scopes[scope]={"feature_shape":list(nf["X"].shape),"scoreable":int(nf["scoreable"].sum()),"fields_bitwise_matched":len(nf)}
            views[v]=scopes
        cases[c]={"case_sha256":expected,"views":views,"original_all_deploy_members_byte_identical":True}
    write(a.out/"regression.json",encoded({"schema":"el-dtu-057-adapter-regression-1","passed":True,"cases":cases,
          "measured_max_feature_gap":0.0,"comparison":"Rebuilt through D1 source/camera/prediction adapter, SHA-identical original published cases. Compared RGB, every extracted numeric feature, flags and scoreability for all and s00."}))


def lineage(job,jobsha,stage,files,pins,gen):
    cfg=load(files/(stage+"-config.json")); inputs=load(files/(stage+"-inputs.json"))
    meta=load(files/(stage+"-run.json"));kind="generation" if stage=="gen" else "render"
    require(meta["run_id"]==pins[stage]["run_id"] and meta["run_kind"]==kind and meta["trial_id"]==job["trial_id"] and
            meta["experiment_id"]==job["experiment_id"] and meta["execution_status"]=="succeeded" and
            all(meta[k]==v for k,v in job["identity"].items()) and cfg["formal_handoff_sha256"]==jobsha
            and cfg["optimizer_sha256"]==OPT and cfg["optimization_steps"]==256 and cfg["worker_sha256"]==job["worker_sha256"]
            and cfg["camera_plan_sha256"]==job["camera_plan_sha256"],"A Windows stage does not belong to the registered input ticket")
    parents=[job["readback"],job["reference"],job["budget"]]
    if job.get("opening"):parents.append(job["opening"])
    if stage=="render":parents.append(pins["gen"])
    require(inputs["parents"]==parents,"Windows stage parent list differs")
    required={"optimizer":(job["budget"],OPT),"cameras":(job["readback"],job["files"][("source" if stage=="gen" else "render")+"_cameras.json"]["sha256"])}
    ids=CONDITIONS[job["condition"]]
    if stage=="gen":
        required.update({s:(job["reference"],job["files"]["sources/"+s+".png"]["sha256"]) for s in ids})
    else:
        required.update(checkpoint=(pins["gen"],gen["checkpoint_sha256"]),reconstruction=(pins["gen"],sha(files/"reconstruction.json")))
    actual={v["role"]:v for v in inputs["inputs"]}
    require(set(actual)==set(required),"Unexpected scientific file or target photograph")
    for role,(owner,identity) in required.items():
        v=actual[role]
        require(v["parent_run_id"]==owner["run_id"] and len(v["files"])==1 and v["files"][0]["sha256"]==identity,
                "Wrong scientific input file or parent: "+role)
    return cfg


def gpu_probe(render,case):
    new=load(render/"dtu-camera-probe.json");old=load(render/"depth-probe.json")
    require(new["passed"] is True and old["passed"] is True and new["prediction_sha256"]==sha(render/"prediction.json") and
            new["render_camera_sha256"]==case.pred["render_camera_sha256"] and
            new["original_C_probe_sha256"]==case.pred["depth_probe_sha256"]==sha(render/"depth-probe.json"),"Missing measured GPU calibration and depth proofs")
    require(set(new["cameras"])==set(case.cameras),"Unverified camera entered actual deployment")
    max_a=max_ed=max_sum=0.
    for k,cam in case.cameras.items():
        q=new["cameras"][k]; expected=cam.K.copy();expected[:2,2]+=.5
        require(q["passed"] and np.array_equal(q["integer_K"],cam.K) and np.array_equal(q["c2w"],cam.c2w) and
                np.max(abs(q["measured_gsplat_edge_K"]-expected))<.0001 and q["measured_empty_alpha"]==0 and len(q["cases"])==2,"Wrong camera provenance or pixel origin")
        z0=(.4*.41+.6*.65*.91)/(.4+.6*.65)
        for x,(z,a,d,xy) in zip(q["cases"],[(z0,.79,z0*.79,[41,61]),(.73,.72,.73*.72,[199,154])],strict=True):
            require(x["xy"]==xy and abs(x["expected_ED_m"]-z)<1e-12 and abs(x["expected_alpha"]-a)<1e-12 and abs(x["expected_D_m"]-d)<1e-12 and
                    x["actual_alpha"]-x["neighbour_max_alpha"]>.004,"GPU probe coordinates, compositing or point centres differ")
            max_a=max(max_a,abs(x["actual_alpha"]-a));max_ed=max(max_ed,abs(x["actual_ED_m"]-z));max_sum=max(max_sum,abs(x["actual_D_m"]-d))
    require(max_a<=.0005 and max_ed<=.0001 and max_sum<=.0001,"Actual off-axis GPU depth or alpha tolerance exceeded")
    return {"passed":True,"cameras":len(case.cameras),"max_alpha_error":max_a,"max_ED_error_m":max_ed,"max_D_error_m":max_sum,
            "proof_sha256":sha(render/"dtu-camera-probe.json"),"C_depth_probe_sha256":sha(render/"depth-probe.json")}


def verify_scene(gen,render,case,cfg):
    m=load(gen/"reconstruction.json");p=case.pred
    require(m["run_id"]==p["reconstruction_run_id"] and m["checkpoint_sha256"]==p["checkpoint_sha256"]==sha(gen/"scene.npz") and
            sha(gen/"reconstruction.json")==p["reconstruction_sha256"] and m["initial_checkpoint_sha256"]==sha(gen/"initial.npz") and
            m["schema"]=="el-stage3-reconstruction-1" and m["backend_version"]=="1.5.3" and m["precision"]=="float32" and
            m["source_camera_sha256"]==p["source_camera_sha256"] and m["source_rgb_sha256"]==p["source_rgb_sha256"] and
            m["initial_gaussian_count"]==m["final_gaussian_count"]==4096 and m["topology_events"]==[] and m["gaussian_optimization_steps"]==256 and
            m["sampled_source_pixels"]==256**3 and m["optimizer_sha256"]==OPT and m["trace_sha256"]==sha(gen/"optimizer.jsonl"),
            "Reconstruction provenance, fixed budget or source binding changed")
    expected=native.initial_values(cfg["optimizer"]["initialization"])
    unchanged=np.ones(4096,dtype=bool)
    with np.load(gen/"initial.npz",allow_pickle=False) as initial,np.load(gen/"scene.npz",allow_pickle=False) as final:
        require(native.digest_values(initial)==native.digest_values(expected)==m["initial_parameter_sha256"]==INIT
                and native.digest_values(final)==m["final_parameter_sha256"] and set(initial.files)==set(final.files)==set(native.PARAMS),"Numerical checkpoint identity differs")
        for k in native.PARAMS:
            a,b=initial[k],final[k]
            require(a.dtype==b.dtype==np.float32 and a.shape==b.shape and np.isfinite(b).all() and np.array_equal(a,expected[k]),"Changed metric Gaussian geometry, seed or precision")
            delta=np.linalg.norm((a-b).astype(np.float64))
            require(delta>0 and np.isclose(delta,m["parameter_delta_l2"][k],rtol=1e-8),"Unmeasured geometry or appearance update")
            unchanged &= np.all((a==b).reshape(4096,-1),axis=1)
    logs=[json.loads(x) for x in (gen/"optimizer.jsonl").read_text().splitlines()]
    ids=tuple(case.generator); counts={s:Counter() for s in ids}
    require(len(logs)==256,"Incomplete update trace")
    for i,row in enumerate(logs):
        s=ids[i%len(ids)]
        require(row["update"]==i+1 and row["source"]==s and row["image_sha256"]==p["source_rgb_sha256"][s] and
                row["pixels"]==256**2 and np.isfinite(row["loss_display_srgb_mae"]) and set(row["gradient_l2"])==set(native.PARAMS),"Invalid source update")
        for k,g in row["gradient_l2"].items():
            require(np.isfinite(g) and g>=0,"Invalid gradient trace")
            counts[s][k]+=int(g>0)
    self_losses={}
    sources=case.sources("all")
    for s in ids:
        src,pred,_=sources[s]
        measured=float(np.mean(abs(src-pred["rgb"]),dtype=np.float64))
        st=m["source_sampling"][s]
        require(st["samples"]==256//len(ids) and st["pixels"]==256**3//len(ids) and dict(counts[s])==st["gradient_nonzero_steps"]
                and all(counts[s][k]>0 for k in native.PARAMS) and abs(measured-m["loss_final_by_source"][s])<1e-5,"Independent source sampling, gradient or self-render MAE mismatch")
        self_losses[s]={"source_updates":st["samples"],"source_pixels":st["pixels"],"nonzero_gradient_updates":dict(counts[s]),
                        "initial":m["loss_initial_by_source"][s],"optimizer_final":m["loss_final_by_source"][s],"rendered_final":measured}
    quality={}
    for view in tuple(case.generator)+TARGETS:
        f=case.frame(view);a,z=f["alpha"],f["z"]
        require(np.isfinite(f["rgb"]).all() and np.isfinite(a).all() and np.isnan(z[a==0]).all() and np.all(np.isfinite(z[a>0])) and np.all(z[a>0]>0),
                "Implicit depth code or invalid color in a registered view")
        valid=np.isfinite(z)
        quality[view]={"pixels":a.size,"alpha_positive":int((a>0).sum()),"alpha_gt_0p05":int((a>.05).sum()),
                       "alpha_min":float(a.min()),"alpha_max":float(a.max()),"depth_positive":int(valid.sum()),
                       "min_ED_m":float(z[valid].min()) if valid.any() else None,"max_ED_m":float(z[valid].max()) if valid.any() else None}
    return {"initial_parameter_sha256":INIT,"final_parameter_sha256":m["final_parameter_sha256"],"all_parameters_unchanged_gaussians":int(unchanged.sum()),
            "initial_gaussians":4096,"final_gaussians":4096,"steps":256,"sampled_source_pixels":256**3,"sources":self_losses,
            "views":quality,"optimization_seconds":m["optimization_seconds"],"render_seconds":p["stage_seconds"],"runtime":m["runtime"],
            "geometry_frame":"DTU metric frame anchored at the two common endpoint axes, fixed C initialization and learning rates in metres"}


def build(a):
    job=load(a.packet/"job.json"); pins=load(a.packet/"pins.json")
    g=load(a.gen/"reconstruction.json")
    gc= lineage(job,pins["job_sha256"],"gen",a.packet,pins,g)
    lineage(job,pins["job_sha256"],"render",a.packet,pins,g)
    src=load(a.sources/"source-images.json"); plan=load(a.rig/"camera-plan.json")
    c=job["condition"]
    require(src["object_id"]==job["object_id"] and sha(a.rig/"camera-plan.json")==job["camera_plan_sha256"]==src["camera_plan_sha256"] and
            {k:src[k] for k in ("camera_sha256","view_set_sha256","crop_sha256","source_view_ids","target_view_ids","lighting_id")}
            =={k:plan[k] for k in ("camera_sha256","view_set_sha256","crop_sha256","source_view_ids","target_view_ids","lighting_id")},
            "Case source conversion and camera registration disagree")
    case=assemble(a.out/"case",c,a.sources/"sources",a.rig/"condition"/c,a.render,a.render)
    require(case.pred["run_id"]==pins["render"]["run_id"] and g["run_id"]==pins["gen"]["run_id"],"Result receipts inconsistent")
    proof=gpu_probe(a.render,case)
    quality=verify_scene(a.gen,a.render,case,gc)
    write(a.out/"reconstruction-readback.json",encoded({"schema":"el-dtu-reconstruction-readback-1","passed":True,**quality,"GPU_probe":proof}))
    keys=("object_id","camera_plan_sha256","camera_sha256","view_set_sha256","crop_sha256","source_view_ids","target_view_ids","lighting_id")
    common={k:src[k] for k in keys}
    provenance={"schema":"el-dtu-case-provenance-1","run_id":os.environ["EL_RUN_ID"],"condition":c,"optimization_steps":256,"arm":"main-256","case_sha256":case.sha256,**common,
                "source_conversion":job["reference"],"rig_registration":job["readback"],"generation":pins["gen"],"render":pins["render"],
                "handoff_sha256":pins["job_sha256"],"opening_sha256":job.get("opening_sha256"),"budget":job["budget"],"source_images":{k:src["views"][k] for k in case.generator},
                "reconstruction_readback_sha256":sha(a.out/"reconstruction-readback.json"),"GPU_probe_sha256":proof["proof_sha256"]}
    write(a.out/"case-provenance.json",encoded(provenance))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest="action",required=True)
    q=subs.add_parser("build")
    for n in ("packet","rig","sources","gen","render","out"):q.add_argument("--"+n,type=Path,required=True)
    q=subs.add_parser("published057")
    for n in ("original","out"):q.add_argument("--"+n,type=Path,required=True)
    args=parser.parse_args()
    if args.action=="build":build(args)
    else:published057(args)
