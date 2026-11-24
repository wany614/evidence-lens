"""Register the external trainer, prepare a source-only fit, and verify a native Windows return."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

from dual_records import bind, resolve, selected, stage
from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens._verify import require, verify_bundle
from evidence_lens.archive import unpack
from make_handoff import publish
from many_upstream_patch import program, TRAINER_SHA
from many_upstream_sfm import RULE, source_inputs
from many_upstream_vendor import BASELINE, COMMIT
from t0_reconstruct_intake import receive_verified_zip


ROOT=Path(__file__).resolve().parents[1]
RECIPE={"updates":6000,"native_checkpoint_step_zero_based":5999,"initialization":"R-only calibrated COLMAP triangulation",
        "sfm":RULE,"seed":42,"native_method":"gsplat simple_trainer MCMC","strategy":{"cap_max":60000,
        "refine_start_iter":500,"refine_stop_iter":4500,"refine_every":100,"noise_lr":500000.,"min_opacity":.005},
        "loss":{"display_l1":.8,"valid_ssim":.2,"sigma":1.5,"window":11,"opacity_reg":.01,"scale_reg":.01},
        "init_opa":.5,"init_scale":.1,"sh_degree":3,"sh_degree_interval":1000,"normalization":"identity",
        "near_far":[.01,1e10],"depth_alpha_epsilon":1e-6,"camera":"pinhole OpenCV integer centres; add 0.5 to both principal point coordinates for gsplat",
        "native_adapter_max_error_scale":2e-5,"per_fit_wall_limit_seconds":3600,"training_guard_seconds":3390,
        "renderer":"classic gsplat 1.5.3 RGB+ED; combined display SH3; all source and target poses",
        "preapproved_fits":15,"hard_fits":20,"cumulative_gpu_seconds":28800}


def cmd(action,pins):
    c=["{python}","-B","{source}/scripts/many_upstream_control.py",action,"--out","{out}","--config","{config}"]
    for name in pins: c+=["--input",name,"{input:"+name+"}"]
    return c


def inventory(s,side):
    vendor=selected(s,"b-upstream-vendor")
    dispatch=load(resolve((s,s/"history"),vendor))
    raw=load(side);archive=side.parent/raw["package"]
    require(raw["status"]=="succeeded" and raw["worker_exit_code"]==0 and raw["bytes"]==archive.stat().st_size and sha(archive)==raw["sha256"],
            "Windows external environment unavailable or damaged")
    pins={"vendor":vendor,"packet":bind(s,vendor["run_id"],"payload/inventory-gpu.zip")}
    c=cmd("inventory-worker",pins)+["--input","native","{input:native}","--input","transport","{input:transport}"]
    return stage(s,"b-upstream-runtime","archive_verify",pins,{"original_native_manifest_sha256":raw["manifest_sha256"],"container_sha256":raw["sha256"],
           "stock_direct_parent_status":"quarantined: input assets missing from declared parent: kit",
           "method":"verify original native manifest and every embedded role, then verify extracted directory members against issued ZIP individually"},
           c,"payload/runtime.json",12_000,extra_files={"native":archive,"transport":side})


def imported(s,side,ticket,packet_sha,trial,kind):
    doc=load(side)
    rid=doc.get("run_id",doc.get("run"));manifest=doc["manifest_sha256"]
    archive=side.parent/doc["package"]
    require(doc["package"]==rid+".zip" and archive.stat().st_size==doc["bytes"] and sha(archive)==doc["sha256"],"native container hash or byte count differs")
    with zipfile.ZipFile(archive) as z:
        run=json.loads(z.read("run.json"));inputs=json.loads(z.read("inputs.json"))
        require(run["run_id"]==rid and run["run_kind"]==kind and run["trial_id"]==trial and run["experiment_id"]=="dual-track-b"
            and run["execution_status"]==doc["status"] and (doc["worker_exit_code"]==0)==(doc["status"]=="succeeded"),"unexpected Windows experiment or disposition")
        recorded={f["sha256"] for i in inputs["inputs"] for f in i["files"]}
        require(packet_sha in recorded and inputs["parents"]==[{"run_id":ticket["run_id"],"manifest_sha256":ticket["manifest_sha256"]}],
                "native stage did not register this exact immutable parent and full packet")
    parent=verify_bundle(s/"runs"/ticket["run_id"])
    require(parent["manifest_sha256"]==ticket["manifest_sha256"],"Mac-issued external parent has changed")
    result=receive_verified_zip(archive,s,doc["sha256"],ROOT/"configs/storage.json",{ticket["run_id"]:parent},ticket)
    require(result["status"] in ("accepted","identical") and result["run_id"]==rid and result["manifest_sha256"]==manifest,"native intake failed: "+str(result))
    return rid,manifest,doc


def recipe(s):
    vendor=selected(s,"b-upstream-vendor")
    pins={"study":selected(s,"b-study-registration"),"roles":selected(s,"b-data-registration"),"vendor":vendor,"runtime":selected(s,"b-upstream-runtime"),
          "objective_control":selected(s,"b-upstream-objectives"),
          "trainer":bind(s,vendor["run_id"],"payload/official/examples/simple_trainer.py"),
          "utils":bind(s,vendor["run_id"],"payload/official/examples/utils.py"),"license":bind(s,vendor["run_id"],"payload/official/LICENSE")}
    return stage(s,"b-upstream-recipe","archive_verify",pins,{"recipe":RECIPE,"baseline_sha256":BASELINE},cmd("recipe-worker",pins),"payload/recipe.json",900000)


def prep(s,label,n,exe,freeze=None):
    src=selected(s,label); raw=load(resolve((s,s/"history"),src))
    key=f"R{n}";r=raw["packages"][key]
    z=bind(s,src["run_id"],"payload/"+r["path"])
    require(z["sha256"]==r["sha256"],"wrong source archive member")
    group=label.removeprefix("b-source-");study=load(resolve((s,s/"history"),selected(s,"b-study-registration")))
    require(study["schema"]=="el-many-source-registration-1" and study["baseline"]=="sha256:"+BASELINE,
            "unexpected source-role registration or A baseline")
    protocol=study["study"]
    scope=protocol["family_roles"]
    known=[v for vs in scope.values() for v in vs]
    scene=next((k for k in known if k.replace("_","-")==group),None)
    require(scene is not None,"unregistered source cohort")
    sizes=([protocol["external"][scene][1]] if scene in protocol["external"] else protocol["authored_checkpoint_conditions"])
    require(n in sizes,"unregistered source cardinality for this scene")
    pins={"recipe":selected(s,"b-upstream-recipe"),"source_role":src,"data":z}
    if scene in scope["confirmation"]:
        require(freeze=="b-model-freeze","confirmation fit requires the registered common model freeze")
        pins["model_freeze"]=selected(s,freeze)
        lock=load(resolve((s,s/"history"),pins["model_freeze"]))
        require(lock["schema"]=="el-many-confirmation-model-freeze-1" and lock["run_id"]==pins["model_freeze"]["run_id"]
                and lock["all_confirmation_references_closed"] is True and lock["upstream_recipe"]==pins["recipe"]
                and lock["parents"]["pilot_budget"]==selected(s,"b-pilot-budget")
                and lock["parents"]["development_diagnostics"]==selected(s,"b-development-diagnostics"),
                "confirmation requires the exact development-frozen upstream recipe, budget and reference gate")
    m=f"{group}-r{n}"
    program_pin=bind(s,pins["recipe"]["run_id"],"payload/sfm-program.py")
    pins["sfm_program"]=program_pin
    sfm=stage(s,"b-sfm-"+m,"dataset_build",pins,{"maximum_points":15000,"real_R":n,"scene":scene,"rule":RULE},
          [str(exe),"-B","{input:sfm_program}","--data","{input:data}","--out","{out}"],"payload/sfm.json",28_000_000)
    pins["sfm"]=sfm
    for f in ("cameras","images","points3D"):pins[f]=bind(s,sfm["run_id"],"payload/sparse/0/"+f+".txt")
    pins["runtime_archive"]=bind(s,pins["recipe"]["run_id"],"payload/trainer-runtime.zip")
    return stage(s,"b-fit-ticket-"+m,"dataset_build",pins,{"scene":scene,"n_R":n,"trial":"b-fit-"+m,"recipe":RECIPE},
                 cmd("packet-worker",pins),"payload/dispatch.json",85_000_000)


def pack(cfg,files,out):
    model=load(files["recipe"]);sfm=load(files["sfm"]);rows={}
    with zipfile.ZipFile(files["data"]) as z:
        scene, _=source_inputs(z)
        R=scene["R"];H=scene["H"]
        require(len(R)==cfg["n_R"] and scene["scene_id"]==cfg["scene"] and R==sfm["source_ids"] and sfm["official_points_or_tracks_read"]==0,
                "SfM or photograph authorization differs")
        for i,k in enumerate(R,1):
            spec=scene["source_rgb"][k];require(sfm["photo_sha256"][k]==spec["sha256"],"SfM/photo identity differs")
            rows[f"train/images/{i:04d}.png"]=z.read(spec["path"])
    cameras={k:{key:scene["views"][k][key] for key in ("width","height","K","c2w")} for k in R+H}
    rows["train/roles.json"]=encoded({"schema":"el-many-native-train-1","R":R,"cameras":{k:cameras[k] for k in R},"photos":sfm["photo_sha256"],
                 "initial_points":sfm["initial_points"],"time_limit_seconds":RECIPE["training_guard_seconds"]})
    rows["projection.json"]=encoded({"R":R,"H":H,"views":cameras,**{k:scene[k] for k in ("scene_id","configuration","frame_receipt","unit")}})
    for f in ("cameras","images","points3D"):
        require(sha(files[f])==sfm["components"]["sparse/0/"+f+".txt"],"changed source-only COLMAP component")
        rows["train/sparse/0/"+f+".txt"]=files[f].read_bytes()
    require(sha(files["runtime_archive"])==model["runtime"]["sha256"],"changed official adapted runtime")
    with zipfile.ZipFile(files["runtime_archive"]) as z:
        for name in z.namelist():rows[name]=z.read(name)
    ticket={"schema":"el-many-upstream-fit-ticket-1","run_id":os.environ["EL_RUN_ID"],"official_commit":COMMIT,"original_trainer_sha256":TRAINER_SHA,
            "source_package":cfg["sources"]["data"],"recipe":RECIPE,"mcmc_canonical_lf_sha256":model["mcmc_canonical_lf_sha256"],
            "files":{n:hashlib.sha256(b).hexdigest() for n,b in rows.items()},"registered_H_input":"poses only"}
    freeze=cfg["sources"].get("model_freeze")
    if freeze is not None:
        ticket["model_freeze"]=freeze
    rows["ticket.json"]=encoded(ticket)
    f=out/"fit.zip"
    with zipfile.ZipFile(f,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,b in sorted(rows.items()):z.writestr(name,b)
    dispatch={"schema":"el-many-native-fit-dispatch-1","run_id":os.environ["EL_RUN_ID"],"trial":cfg["trial"],
         "experiment_id":"dual-track-b","kind":"model_fit","packet":{"path":f.name,"bytes":f.stat().st_size,"sha256":sha(f)},
         "scene_id":scene["scene_id"],"configuration":scene["configuration"],"R":R,"H_poses_only":H,"initial_points":sfm["initial_points"],
         "sfm":cfg["sources"]["sfm"],"recipe":cfg["sources"]["recipe"],"data":cfg["sources"]["data"],"max_wall_seconds":3600,
         "native_command":"python -I -B run_gpu.py --out EMPTY_RECORDER_PAYLOAD","permit_one_fit":True}
    if freeze is not None:
        dispatch["model_freeze"]=freeze
    save(out/"dispatch.json",dispatch)


def worker(action,cfg,paths,out):
    if action=="inventory-worker":
        side=load(paths["transport"]);vendor=cfg["sources"]["vendor"]
        with tempfile.TemporaryDirectory(dir=out.parent) as temp:
            native=Path(temp)/"original"
            unpack(paths["native"],native)
            checked=verify_bundle(native)
            require(checked["manifest_sha256"]==side["manifest_sha256"] and checked["run"]["execution_status"]=="succeeded"
                    and checked["run"]["run_kind"]=="archive_verify" and checked["run"]["trial_id"]=="b-upstream-inventory"
                    and checked["inputs"]["parents"]==[{"run_id":vendor["run_id"],"manifest_sha256":vendor["manifest_sha256"]}],
                    "unexpected native environment inventory")
            roles={x["role"]:x for x in checked["inputs"]["inputs"]}
            require(roles["packet"]["files"][0]["sha256"]==cfg["sources"]["packet"]["sha256"]
                    and roles["dispatch"]["files"][0]["sha256"]==vendor["sha256"],"wrong external packet or dispatch")
            with zipfile.ZipFile(paths["packet"]) as z:
                expected={k:hashlib.sha256(z.read(k)).hexdigest() for k in z.namelist()}
            actual={x["path"].rsplit("/",1)[-1]:x["sha256"] for x in roles["kit"]["files"]}
            require(expected==actual,"extracted native inventory kit differs from exact issued ZIP members")
            doc=load(native/"payload/inventory.json")
            ref={"run_id":checked["run"]["run_id"],"manifest_sha256":checked["manifest_sha256"],
                 "member":"payload/inventory.json","sha256":sha(native/"payload/inventory.json")}
        require(doc["matches_official_mcmc"] and doc["commit"]==COMMIT and doc["fits"]==doc["photo_reads"]==0 and doc["cap"]==60000,"incorrect CUDA inventory")
        save(out/"runtime.json",{"schema":"el-many-official-runtime-readback-1","run_id":os.environ["EL_RUN_ID"],"native":ref,"inventory":doc,
             "stock_direct_parent_status":cfg["stock_direct_parent_status"],"independently_verified_zip_kit_members":expected})
    elif action=="recipe-worker":
        original,utils=paths["trainer"].read_bytes(),paths["utils"].read_bytes()
        trainer,helpers,patch=program(original,utils)
        files={"official/simple_trainer.py":original,"official/utils.py":utils,"official/LICENSE":paths["license"].read_bytes(),
               "runtime/simple_trainer.py":trainer,"runtime/upstream_math.py":helpers,"headless-trainer.patch":patch,
               "runtime/many_upstream_native.py":(ROOT/"scripts/many_upstream_native.py").read_bytes(),
               "runtime/many_upstream_adapter.py":(ROOT/"scripts/many_upstream_adapter.py").read_bytes(),"run_gpu.py":(ROOT/"scripts/many_upstream_gpu.py").read_bytes()}
        receipt=load(paths["runtime"])["inventory"]
        require(len(receipt["packages"])>0 and receipt["matches_official_mcmc"],"unverified external runtime")
        f=out/"trainer-runtime.zip"
        with zipfile.ZipFile(f,"x",compression=zipfile.ZIP_DEFLATED) as z:
            for name,b in sorted(files.items()):z.writestr(name,b)
        (out/"headless-trainer.patch").write_bytes(patch)
        (out/"sfm-program.py").write_bytes((ROOT/"scripts/many_upstream_sfm.py").read_bytes())
        save(out/"recipe.json",{"schema":"el-many-upstream-recipe-1","run_id":os.environ["EL_RUN_ID"],"parents":cfg["sources"],"recipe":cfg["recipe"],
             "baseline_id":"sha256:"+BASELINE,"commit":COMMIT,"original_trainer_sha256":TRAINER_SHA,"mcmc_canonical_lf_sha256":receipt["mcmc_module_canonical_lf_sha256"],
             "headless_patch_sha256":sha(out/"headless-trainer.patch"),"sfm_program_sha256":sha(out/"sfm-program.py"),
             "runtime":{"path":f.name,"bytes":f.stat().st_size,"sha256":sha(f)},"files":{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}})
    else:pack(cfg,paths,out)


def publish_packet(s,pin,exchange):
    f=resolve((s,s/"history"),pin);c=load(f);n=c["packet"]["path"]
    target=exchange/c["trial"]
    for name,b in ((n,(f.parent/n).read_bytes()),("dispatch.json",f.read_bytes()),("pin.json",encoded(pin))):publish(target/name,b)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("inventory","recipe","prep","inventory-worker","recipe-worker","packet-worker"))
    for k in ("store","receipt","out","config","sfm-python","exchange"):p.add_argument("--"+k,type=Path)
    p.add_argument("--source-stage");p.add_argument("--n",type=int);p.add_argument("--freeze");p.add_argument("--input",action="append",nargs=2,default=[])
    a=p.parse_args()
    if a.action.endswith("-worker"):worker(a.action,load(a.config),{k:Path(v) for k,v in a.input},a.out)
    else:
        s=a.store.resolve()
        if a.action=="inventory":pin=inventory(s,a.receipt)
        elif a.action=="recipe":pin=recipe(s)
        else:
            pin=prep(s,a.source_stage,a.n,a.sfm_python,a.freeze)
            if a.exchange:publish_packet(s,pin,a.exchange)
        print(json.dumps(pin))
