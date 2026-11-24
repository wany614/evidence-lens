"""Independently align each sealed fit's actual native parser rows and inference cameras, then account for checkpoint updates."""

import argparse
import ast
from collections import Counter
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from PIL import Image


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    path.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def register(s, mode, executable):
    from dual_records import bind, resolve, selected, stage
    from many_upstream_diagnostics import COHORT, PILOT
    tags=[tag for tag,_,_ in COHORT]
    pins={"recipe":selected(s,"b-upstream-recipe"),"freeze":selected(s,"b-model-freeze"),
          "development_queue":selected(s,"b-training-development-fit-queue"),"confirmation_queue":selected(s,"b-confirmation-fit-queue")}
    if mode=="packets":
        pilot=selected(s,"b-bank-"+PILOT)
        proof=read(resolve((s,s/"history"),bind(s,pilot["run_id"],"payload/upstream-evidence.json")))
        pins["probe"]=proof["native_registry"]["probe"]
        for tag in tags:
            pin=selected(s,"b-fit-ticket-"+tag);row=read(resolve((s,s/"history"),pin))
            pins["ticket-"+tag]=pin
            pins["packet-"+tag]=bind(s,pin["run_id"],"payload/"+row["packet"]["path"])
            pins["source-"+tag]=row["data"]
        label,member="b-fit-input-alignment","payload/alignment.json"
    else:
        pins["alignment"]=selected(s,"b-fit-input-alignment")
        pins["upstream"]=selected(s,"b-upstream-closeout")
        recipe=read(resolve((s,s/"history"),pins["recipe"]))
        pins["original_trainer"]=recipe["parents"]["trainer"]
        pins["adapted_runtime"]=bind(s,pins["recipe"]["run_id"],"payload/trainer-runtime.zip")
        for tag in tags:
            if (s/"work/controller"/("b-terminal-failure-"+tag)/"selected.json").is_file():
                pins["failure-"+tag]=selected(s,"b-terminal-failure-"+tag)
                continue
            bank=selected(s,"b-bank-"+tag)
            proof=read(resolve((s,s/"history"),bind(s,bank["run_id"],"payload/upstream-evidence.json")))
            pins["fit-"+tag]=proof["native_registry"]["training"]
            pins["probe-"+tag]=proof["native_registry"]["probe"]
        label,member="b-checkpoint-update-attribution","payload/updates.json"
    cmd=[str(executable),"-B","{source}/scripts/many_upstream_alignment.py",mode+"-worker","--config","{config}","--out","{out}"]
    for role in pins:
        cmd+=["--input",role,"{input:"+role+"}"]
    return stage(s,label,"archive_verify",pins,{"tags":tags,"access":"issued R-only photographs and camera calibration; no H observations",
        "numeric_camera_check":"replay the sealed CPU Parser and Dataset. Extract original projection camera statements and remap only the device to CPU."},
        cmd,member,2_000_000)


def camera_program(raw, name):
    """Obtain actual pre-raster K and c2w from either recorded inference path, without a CUDA call."""
    tree=ast.parse(raw)
    if name=="independent_adapter":
        target=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="render")
    else:
        targets=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
                 and [getattr(i,"id",None) for i in n.target.elts]==["idx","v"]]
        need(len(targets)==1,"could not identify the recorded native projection loop")
        target=targets[0]
    statements=[]
    for n in target.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ("k","pose"):
            statements.append(copy.deepcopy(n))
        elif isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Subscript) and getattr(n.target.value,"id",None)=="k":
            statements.append(copy.deepcopy(n))
    need(len(statements)==3 and isinstance(statements[1],ast.AugAssign),"unsupported camera preparation source")
    for n in statements:
        for item in ast.walk(n):
            if isinstance(item,ast.keyword) and item.arg=="device":
                need(isinstance(item.value,ast.Constant) and item.value.value=="cuda","unexpected inference device")
                item.value.value="cpu"
    block=ast.Module(body=statements,type_ignores=[])
    code=compile(block,name,"exec")
    import torch
    def apply(cam):
        env={"torch":torch,"cam":cam}
        exec(code,env)
        return env["k"].numpy(),env["pose"].numpy()
    return apply


def quat_pose(row):
    """An independent wxyz outer-product construction of COLMAP world-to-camera and its inverse."""
    w,x,y,z=np.fromiter(map(float,row[1:5]),dtype=float)
    q=np.array([w,x,y,z]);w,x,y,z=q/np.linalg.norm(q)
    v=np.array([x,y,z]);cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    rotation=(w*w-v@v)*np.eye(3)+2*np.outer(v,v)+2*w*cross
    mat=np.eye(4);mat[:3,:3]=rotation;mat[:3,3]=list(map(float,row[5:8]))
    return np.linalg.inv(mat)


def maxdiff(left,right):
    return float(np.max(np.abs(np.asarray(left,np.float64)-np.asarray(right,np.float64)),initial=0))


def probe(data):
    need(data["schema"]=="el-many-native-expected-z-probe-1" and data["passed"] is True and
         [(r["x"],r["y"]) for r in data["samples"]]==[(32,32),(48,40)],"unverified original on/off-axis depth probe")
    for r in data["samples"]:
        need(abs(r["alpha"]-.75)<1e-4 and abs(r["expected_z"]-8/3)<1e-4 and abs(r["allocated_depth_sum"]-2)<1e-4,
             "unexpected native expected optical-axis depth or opacity")
    return {"samples":data["samples"],"max_abs_expected_z_error":max(abs(r["expected_z"]-8/3) for r in data["samples"])}


def projection_pixels(k0,p0,k1,p1,grid):
    # Unit-optical-depth plane in the FP32 deployment camera. Evaluate quantization in FP64.
    relative=np.linalg.solve(p0.astype(float),p1.astype(float))
    plane=relative[:3,:3].copy();plane[:,2]+=relative[:3,3]
    h=k0.astype(float) @ plane @ np.linalg.inv(k1.astype(float))
    xy=h @ grid
    need(np.isfinite(xy).all() and (xy[2]>0).all(),"invalid projected source plane")
    return maxdiff(xy[:2]/xy[2],grid[:2])


def check_case(tag,paths,recipe,grid):
    import torch
    ticket=read(paths["ticket-"+tag]);p=paths["packet-"+tag];s=paths["source-"+tag]
    need(file_digest(p)==ticket["packet"]["sha256"] and file_digest(s)==ticket["data"]["sha256"],"wrong original fit or source container")
    rows=[]
    with zipfile.ZipFile(p) as z,zipfile.ZipFile(s) as src,tempfile.TemporaryDirectory() as t:
        root=Path(t);inner=json.loads(z.read("ticket.json"));roles=json.loads(z.read("train/roles.json"))
        rendering=json.loads(z.read("projection.json"));scene=json.loads(src.read("scene.json"))
        R=roles["R"];H=rendering["H"]
        need(R==scene["A"]==scene["R"]==rendering["R"]==ticket["R"] and H==scene["H"]==ticket["H_poses_only"]
             and set(R).isdisjoint(H) and len(R)==len(set(R)) and set(roles["photos"])==set(roles["cameras"])==set(R)
             and inner["source_package"]==ticket["data"] and set(scene["source_rgb"])==set(R),"case roster disagreement")
        photo_names=[f"train/images/{i:04d}.png" for i in range(1,len(R)+1)]
        train_names=set(photo_names)|{"train/roles.json","train/sparse/0/images.txt","train/sparse/0/cameras.txt","train/sparse/0/points3D.txt"}
        need({name for name in z.namelist() if name.startswith("train/")}==train_names,"unregistered training namespace")
        needed_programs=["runtime/many_upstream_native.py","runtime/many_upstream_adapter.py","run_gpu.py","runtime/simple_trainer.py"]
        for name in [*train_names,*needed_programs]:
            raw=z.read(name)
            need(digest(raw)==inner["files"][name],"fit packet manifest differs for "+name)
            if name in needed_programs:
                need(digest(raw)==recipe["files"][name],"different frozen upstream implementation")
            target=root/name;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(raw)
        spec=importlib.util.spec_from_file_location("recorded_training_parser",root/needed_programs[0])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        parser=module.Parser(str(root/"train"),1,False,0)
        data=module.Dataset(parser,"train",None,False)
        need(parser.ids==R and len(data)==len(R),"actual sealed Parser/Dataset source order differs")
        native_camera=camera_program(z.read("run_gpu.py"),"official_runner_input")
        adapter_camera=camera_program(z.read("runtime/many_upstream_adapter.py"),"independent_adapter")
        cams=[line.split() for line in z.read("train/sparse/0/cameras.txt").decode().splitlines()]
        images=z.read("train/sparse/0/images.txt").decode().splitlines()
        need(len(cams)==len(R) and len(images)==2*len(R),"COLMAP roster size differs")
        for i,key in enumerate(R):
            name=photo_names[i];header=images[2*i].split();crm=cams[i]
            camera=rendering["views"][key];declared={k:scene["views"][key][k] for k in ("width","height","K","c2w")}
            need(roles["cameras"][key]==camera==declared and (camera["width"],camera["height"])==(378,252),"camera identity differs")
            need(crm[:4]==[str(i+1),"PINHOLE","378","252"] and header[0]==header[8]==str(i+1)
                 and header[9]==Path(name).name,"source index, PNG basename or COLMAP camera id differ")
            b=z.read(name);base=scene["source_rgb"][key];authorized=src.read(base["path"])
            need(b==authorized and digest(b)==base["sha256"]==roles["photos"][key]==inner["files"][name],"source PNG identity differs")
            with Image.open(io.BytesIO(authorized)) as im:
                need(im.size==(378,252) and im.mode=="RGB","invalid source observation")
                source_pixels=np.asarray(im)
            actual=data[i]
            need(actual["image_id"]==i and actual["image"].dtype==torch.uint8
                 and np.array_equal(actual["image"].numpy(),source_pixels),"native Dataset image differs from authorized PNG")
            k0,p0=actual["K"].numpy(),actual["camtoworld"].numpy()
            k1,p1=native_camera(camera);ka,pa=adapter_camera(camera)
            need(np.array_equal(k1,ka) and np.array_equal(p1,pa),"render camera boundaries differ")
            fx,fy,cx,cy=map(float,crm[4:8]);colmap_k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
            independent=quat_pose(header);original=np.array(camera["c2w"],float)
            edge=np.asarray(camera["K"],float).copy();edge[:2,2]+=.5
            need(np.array_equal(k0,colmap_k.astype(np.float32)) and np.array_equal(p1,original.astype(np.float32))
                 and maxdiff(independent,original)<=2e-5 and maxdiff(k0,k1)<=1e-4
                 and maxdiff(p0,p1)<=2e-5,"native training or inference camera differs outside the recorded tolerance")
            delta={"source_png_uint8":0,"dataset_k_vs_colmap_cast":maxdiff(k0,colmap_k.astype(np.float32)),
                   "colmap_k_fp64_vs_original_plus_half":maxdiff(colmap_k,edge),"train_vs_render_k_fp32":maxdiff(k0,k1),
                   "independent_quaternion_c2w_fp64_vs_declared":maxdiff(independent,original),
                   "dataset_c2w_fp32_vs_independent_fp64":maxdiff(p0,independent),
                   "train_vs_render_c2w_rotation_fp32":maxdiff(p0[:3,:3],p1[:3,:3]),
                   "train_vs_render_c2w_translation_fp32":maxdiff(p0[:3,3],p1[:3,3]),
                   "two_native_routes_k_fp32":maxdiff(k1,ka),"two_native_routes_c2w_fp32":maxdiff(p1,pa),
                   "unit_depth_plane_reprojection_analysis_px":projection_pixels(k0,p0,k1,p1,grid)}
            rows.append({"source":key,"colmap_image_id":i+1,"filename":header[9],"source_png_sha256":base["sha256"],
                         "matched_rgb_pixels":source_pixels.shape[0]*source_pixels.shape[1],"max_abs":delta})
    return {"scene_id":ticket["scene_id"],"fit_ticket":ticket["run_id"],"fit_zip_sha256":ticket["packet"]["sha256"],
            "registered_source_zip_sha256":ticket["data"]["sha256"],"native_loader_sha256":recipe["files"][needed_programs[0]],
            "source_views":len(rows),"matched_rgb_pixels":sum(r["matched_rgb_pixels"] for r in rows),
            "all_k_entries_numerically_equal_fp32":sum(r["max_abs"]["train_vs_render_k_fp32"]==0 for r in rows),
            "all_pose_entries_numerically_equal_fp32":sum(r["max_abs"]["train_vs_render_c2w_rotation_fp32"]==0 and
                  r["max_abs"]["train_vs_render_c2w_translation_fp32"]==0 for r in rows),
            "max_abs":{k:max(r["max_abs"][k] for r in rows) for k in rows[0]["max_abs"]},"rows":rows}


def run_packets(cfg,paths,out):
    import torch
    torch.set_num_threads(1)
    y,x=np.mgrid[0:252,0:378]
    grid=np.vstack((x.ravel()+.5,y.ravel()+.5,np.ones(x.size)))
    recipe=read(paths["recipe"])
    result={tag:check_case(tag,paths,recipe,grid) for tag in cfg["tags"]}
    counts=sum(v["source_views"] for v in result.values());pixels=sum(v["matched_rgb_pixels"] for v in result.values())
    need(len(result)==15 and counts==1053 and pixels==counts*378*252,"incomplete 15-checkpoint source census")
    write(out/"alignment.json",{"schema":"el-many-fit-input-alignment-1","run_id":os.environ["EL_RUN_ID"],"parents":cfg["sources"],
          "checkpoints":len(result),"views":counts,"matched_rgb_pixels":pixels,"matched_uint8_channel_values":3*pixels,
          "exact_k_views_fp32":sum(v["all_k_entries_numerically_equal_fp32"] for v in result.values()),
          "exact_pose_views_fp32":sum(v["all_pose_entries_numerically_equal_fp32"] for v in result.values()),
          "max_abs":{k:max(v["max_abs"][k] for v in result.values()) for k in next(iter(result.values()))["max_abs"]},
          "native_pilot_depth":probe(read(paths["probe"])),"cpu_torch":torch.__version__,"numpy":np.__version__,
          "method":"Native frozen Parser and Dataset.__getitem__ replay. Both native source camera-preparation statements are extracted from the sealed programs and evaluated on CPU in FP32. The FP64 reprojection comparison uses all analysis pixels on one plane at unit deployment camera-Z. Other scene depths can give different translational residuals.",
          "source_access":"only authorized R images, known camera poses and the source-only triangulation; held-out photos decoded: 0",
          "cases":result})


def checkpoint_program_order(raw):
    tree=ast.parse(raw)
    runner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="Runner")
    train=next(n for n in runner.body if isinstance(n,ast.FunctionDef) and n.name=="train")
    loop=next(n for n in ast.walk(train) if isinstance(n,ast.For) and getattr(n.target,"id",None)=="step"
              and getattr(n.iter,"id",None)=="pbar")
    calls=[n for n in ast.walk(loop) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    def positions(instance,method):
        return sorted(n.lineno for n in calls if n.func.attr==method and getattr(n.func.value,"id",None)==instance)
    backward=positions("loss","backward");save=positions("torch","save");steps=positions("optimizer","step")
    need(len(backward)==len(save)==1 and steps and backward[0]<save[0]<steps[0],"official native checkpoint/update ordering differs")
    return {"loss_backward_line":backward[0],"checkpoint_torch_save_line":save[0],"optimizer_step_lines":steps}


def verified_sampler(fit, recipe):
    """Reconstruct indices from the pinned CPU draw and native sampler. Name step 5998 only after an exact original digest match."""
    import torch
    roster=fit["R"]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(recipe["recipe"]["seed"])
        torch.rand((fit["initial_gaussians"],4),dtype=torch.float32,device="cpu")
        loader=torch.utils.data.DataLoader(range(len(roster)),batch_size=1,shuffle=True,num_workers=0,
                                          persistent_workers=False,pin_memory=False)
        it=iter(loader);order=[];sequence=hashlib.sha256()
        for step in range(fit["training_steps"]):
            try:
                item=next(it)
            except StopIteration:
                it=iter(loader);item=next(it)
            key=roster[int(item.item())];order.append(key)
            sequence.update((str(step)+" "+key+"\n").encode())
    matches=sequence.hexdigest()==fit["ordered_actual_source_step_sha256"]
    if matches:
        need(all(order[row["step"]]==row["source"] for row in fit["trace"]) and dict(Counter(order))==fit["source_visits"],
             "recorded sparse trace and full source-order digest disagree")
    return {"original_sha256":fit["ordered_actual_source_step_sha256"],"replayed_sha256":sequence.hexdigest(),
            "full_order_hash_matches":matches,"last_update_in_saved_model_zero_based":5998,
            "last_update_in_saved_model_source":order[5998] if matches else None,
            "method":"Replay seeded CPU quaternion initializer draw and stock PyTorch shuffled DataLoader on integer row IDs. Publish the unlogged penultimate identity only when the digest of every one of the 6000 (step, source) rows equals its original native witness."}


def run_attribution(cfg,paths,out):
    align,up=read(paths["alignment"]),read(paths["upstream"]);need(align["checkpoints"]==15 and up["issued_fit_count"]==15,"incomplete registered series")
    recipe=read(paths["recipe"]);original=paths["original_trainer"].read_bytes()
    with zipfile.ZipFile(paths["adapted_runtime"]) as runtime:
        active=runtime.read("runtime/simple_trainer.py")
    need(digest(original)==recipe["original_trainer_sha256"] and digest(active)==recipe["files"]["runtime/simple_trainer.py"],
         "different original or executed trainer bytes")
    program={name:{"sha256":digest(raw),**checkpoint_program_order(raw)} for name,raw in (("original",original),("executed",active))}
    cases={}
    for tag in cfg["tags"]:
        if "failure-"+tag in paths:
            cases[tag]={"status":"unavailable","failure":cfg["sources"]["failure-"+tag]}
            continue
        fit=read(paths["fit-"+tag]);all_visits=fit["source_visits"];last=[r for r in fit["trace"] if r["step"]==5999]
        need(fit["native_checkpoint_step_zero_based"]==5999 and fit["training_steps"]==sum(all_visits.values())==6000
             and fit["checkpoint_timing"]=="official save before optimizer update of zero-based step 5999"
             and len(last)==1 and last[0]["source"] in all_visits and [r["source"] for r in align["cases"][tag]["rows"]]==fit["R"]
             and fit["native_checkpoint"]["sha256"]==up["cases"][tag]["checkpoint_sha256"],"real native trace or checkpoint identity differs")
        applied=all_visits.copy();applied[last[0]["source"]]-=1
        need(sum(applied.values())==5999 and min(applied.values())>0,"missing source exposure before checkpoint")
        cases[tag]={"status":"succeeded","checkpoint_sha256":fit["native_checkpoint"]["sha256"],"training":cfg["sources"]["fit-"+tag],
            "all_computed_source_samples":all_visits,"checkpoint_updates_by_source":applied,"checkpoint_updates":sum(applied.values()),
            "last_computed_sample_source":last[0]["source"],"last_computed_sample_zero_based":5999,
            "verified_full_order_replay":verified_sampler(fit,recipe),
            "native_depth":probe(read(paths["probe-"+tag])),"depth_probe":cfg["sources"]["probe-"+tag]}
    ok=[c for c in cases.values() if c["status"]=="succeeded"]
    write(out/"updates.json",{"schema":"el-many-checkpoint-update-attribution-1","run_id":os.environ["EL_RUN_ID"],"parents":cfg["sources"],
          "successful_checkpoints":len(ok),"total_source_loss_computations":sum(sum(c["all_computed_source_samples"].values()) for c in ok),
          "exact_6000_source_order_digest_matches":sum(c["verified_full_order_replay"]["full_order_hash_matches"] for c in ok),
          "total_source_updates_in_saved_checkpoints":sum(c["checkpoint_updates"] for c in ok),"trainer_program_order":program,
          "semantics":"Each fit computed the last source loss and gradients at zero-based step 5999. Official Runner.train saved the model before optimizer.step. The final in-memory optimizer update was performed after the saved checkpoint.",
          "wddm":"Device-wide nvidia-smi memory includes desktop/background allocations; WDDM per-process driver counters can be unavailable. PyTorch allocator data are independent process-local framework counters. Missing per-process driver counters remain null in the upstream resource ledger.",
          "successful_cases_without_observable_driver_process_peak":[tag for tag,r in up["cases"].items() if r["status"]=="succeeded"
                                                                    and r["telemetry"]["observed_peak_driver_process_bytes"] is None],
          "heldout_observations_decoded":0,"cases":cases})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("packets","attribution","packets-worker","attribution-worker"))
    for k in ("store","cpu-python","config","out"):p.add_argument("--"+k,type=Path)
    p.add_argument("--input",action="append",nargs=2,default=[]);a=p.parse_args()
    if a.mode.endswith("-worker"):
        files={k:Path(v) for k,v in a.input};cfg=read(a.config)
        if a.mode=="packets-worker":run_packets(cfg,files,a.out)
        else:run_attribution(cfg,files,a.out)
    else:
        print(json.dumps(register(a.store.resolve(),a.mode,a.cpu_python.absolute())))
