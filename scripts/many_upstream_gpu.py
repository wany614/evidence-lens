"""Train the pinned official single-GPU simple_trainer and bind its checkpoint to native source/target predictions."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def write(path,doc):
    path.write_text(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def record(path,root):
    return {"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":checksum(path)}


def run(root,out):
    start=time.monotonic();cpu=time.process_time()
    import torch
    ticket=json.loads((root/"ticket.json").read_bytes())
    if ticket["schema"]!="el-many-upstream-fit-ticket-1" or ticket["recipe"]["updates"]!=6000:
        raise ValueError("unexpected native training recipe")
    for n,h in ticket["files"].items():
        if checksum(root/n)!=h:
            raise ValueError("changed native packet member "+n)
    sys.path.insert(0,str(root/"runtime"))
    from many_upstream_adapter import PARAMS,depth_probe,fields,load_checkpoint,need,render
    import simple_trainer as official
    from gsplat.strategy import MCMCStrategy
    import gsplat.strategy.mcmc as official_strategy
    implementation=hashlib.sha256(Path(official_strategy.__file__).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    need(implementation==ticket["mcmc_canonical_lf_sha256"] and importlib.metadata.version("gsplat")=="1.5.3","unexpected gsplat strategy")
    need(torch.cuda.is_available() and torch.cuda.device_count()==1,"one native CUDA device required")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision("highest");torch.set_num_threads(2)
    torch.cuda.set_device(0);torch.empty(1,device="cuda");torch.cuda.reset_peak_memory_stats()
    roles=json.loads((root/"train/roles.json").read_bytes())
    train_start=time.monotonic()
    with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        cfg=official.Config(disable_viewer=True,disable_video=True,normalize_world_space=False,test_every=0,data_factor=1,
                 result_dir=scratch,data_dir=str(root/"train"),max_steps=6000,save_steps=[6000],eval_steps=[],ply_steps=[],tb_every=0,
                 sh_degree=3,sh_degree_interval=1000,init_type="sfm",init_opa=.5,init_scale=.1,opacity_reg=.01,scale_reg=.01,
                 packed=False,ssim_lambda=.2,batch_size=1,near_plane=.01,far_plane=1e10,
                 strategy=MCMCStrategy(cap_max=60000,refine_start_iter=500,refine_stop_iter=4500,refine_every=100,verbose=False))
        runner=official.Runner(0,0,1,cfg)
        runner.train()
        torch.cuda.synchronize()
        dataset=runner.trainset
        fit_s=time.monotonic()-train_start
        counts={k:int(dataset.visits[k]) for k in roles["R"]}
        need(set(dataset.visits)==set(roles["R"]) and sum(counts.values())==6000 and max(counts.values())-min(counts.values())<=1,
             "actual training image set or balanced traversal differs from registered R")
        path=out/"checkpoint.pt";shutil.copyfile(Path(scratch)/"ckpts/ckpt_5999_rank0.pt",path)
        native=load_checkpoint(path); need(native["step"]==5999,"not the official final native checkpoint")
        runner.splats.load_state_dict(native["splats"],strict=True)
        raw={k:v.cuda() for k,v in native["splats"].items()}
        equal={k:{"shape":list(v.shape),"raw_sha256":hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest(),
                  "equal_to_loaded_native":torch.equal(v,runner.splats[k].detach().cpu())} for k,v in native["splats"].items()}
        need(set(equal)==set(PARAMS) and all(v["equal_to_loaded_native"] for v in equal.values()),"parameter parity failed")
        n=len(raw["means"])
        underflow=raw["opacities"].sigmoid().detach().cpu().numpy()
        scales=torch.exp(raw["scales"]).detach().cpu().numpy()
        write(out/"training.json",{"schema":"el-many-official-fit-1","run_id":os.environ["EL_RUN_ID"],"native_checkpoint":record(path,out),
              "official_commit":ticket["official_commit"],"original_trainer_sha256":ticket["original_trainer_sha256"],"adapted_trainer_sha256":ticket["files"]["runtime/simple_trainer.py"],
              "R":roles["R"],"source_visits":counts,"ordered_actual_source_step_sha256":dataset.sequence.hexdigest(),
              "training_steps":sum(counts.values()),"native_checkpoint_step_zero_based":native["step"],
              "checkpoint_timing":"official save before optimizer update of zero-based step 5999","initial_gaussians":roles["initial_points"],
              "gaussians":n,"alpha_above_0_005":int((underflow>.005).sum()),"opacity_percentiles":np.percentile(underflow,[0,25,50,75,100]).tolist(),
              "scale_percentiles":np.percentile(scales,[0,25,50,75,100]).tolist(),"native_parameters":equal,"trace":dataset.trace,
              "fit_seconds":fit_s,"train_scene_scale":float(runner.scene_scale),"normalization":"identity","dataset_read":"R-only rebuilt COLMAP",
              "source_photo_opens":len(counts),"target_photo_opens":0,"internal_holdout":0,"lpips_evaluations":0})
        scene=json.loads((root/"projection.json").read_bytes())
        need(scene["R"]==roles["R"] and set(scene["R"]).isdisjoint(scene["H"]),"render roster and trainer differ")
        folder=out/"frames";folder.mkdir()
        frames={};parity={};all_sources=[];source_only_mean=torch.stack([p.float().mean((0,1)) for p in runner.parser.images]).mean(0).numpy()/255
        with torch.no_grad():
            probe=depth_probe();write(out/"depth-probe.json",probe)
            for idx,v in enumerate(scene["R"]+scene["H"]):
                cam=scene["views"][v]
                k=torch.tensor(cam["K"],dtype=torch.float32,device="cuda");k[:2,2]+=.5
                pose=torch.tensor(cam["c2w"],dtype=torch.float32,device="cuda")
                val,alpha,info=runner.rasterize_splats(pose[None],k[None],cam["width"],cam["height"],sh_degree=3,
                       render_mode="RGB+ED",near_plane=cfg.near_plane,far_plane=cfg.far_plane)
                expected=fields(val,alpha)
                actual=render(raw,cam,(cfg.near_plane,cfg.far_plane))
                deltas={}
                for chan in ("rgb","alpha","z"):
                    x,y=actual[chan],expected[chan];fx=np.isfinite(x);fy=np.isfinite(y)
                    need(np.array_equal(fx,fy),"external/native depth support mask differs")
                    maximum=float(np.abs(x[fx]-y[fx]).max(initial=0))
                    need(maximum<=2e-5*max(1,float(np.abs(y[fy]).max(initial=0))),"native adapter parity failed "+chan)
                    deltas[chan]={"max_abs":maximum,"common_finite":int(fx.sum()),"exact":x[fx].tobytes()==y[fy].tobytes()}
                parity[v]=deltas
                f=folder/f"{idx:04d}.npz";np.savez_compressed(f,**actual);frames[v]=record(f,out)
                if idx<len(scene["R"]):
                    source=runner.parser.images[idx].numpy().astype(np.float32)/255
                    paint=float(np.abs(source-actual["rgb"]).mean())
                    mean=float(np.abs(source-source_only_mean).mean())
                    radii=info["radii"].detach().cpu().numpy()
                    all_sources.append({"source":v,"mae":paint,"source_mean_rgb_baseline_mae":mean,
                                        "alpha_gt_0_5":int((actual["alpha"]>.5).sum()),
                                        "finite_z":int(np.isfinite(actual["z"]).sum()),"projected_gaussians":int((radii>0).all(-1).sum()),
                                        "max_projected_radius_px":float(radii.max(initial=0))})
    write(out/"parity.json",{"schema":"el-many-native-checkpoint-parity-1","native_fields":equal,"frames":parity,
          "native_renderer":"official Runner.rasterize_splats at registered commit",
          "compared_renderer":"independent adapter loading native SH tensors and calling gsplat.rasterization",
          "source_camera_count":len(scene["R"]),"heldout_poses_only":len(scene["H"]),"native_depth_probe":record(out/"depth-probe.json",out)})
    need(time.monotonic()-start<3600,"single reconstruction exceeded one hour")
    write(out/"asset.json",{"schema":"el-many-external-asset-1","run_id":os.environ["EL_RUN_ID"],"ticket_run":ticket["run_id"],
          "source_package":ticket["source_package"],"scene_id":scene["scene_id"],"configuration":scene["configuration"],
          "official_commit":ticket["official_commit"],"R":scene["R"],"A":scene["R"],"H":scene["H"],"unit":scene["unit"],
          "frame_receipt":scene["frame_receipt"],"views":scene["views"],"frames":frames,"source_self":all_sources,
          "source_only_mean_rgb":source_only_mean.tolist(),"checkpoint":record(path,out),"gaussians":n,
          "native_parity":record(out/"parity.json",out),"depth_probe":record(out/"depth-probe.json",out),"training":record(out/"training.json",out),
          "rgb_encoding":"native SH display_srgb_0_1 directly fitted to the source RGB8 values; terminal clamp [0,1]",
          "z_semantics":"opacity-normalized expected OpenCV optical-axis Z in frame units; alpha <= 1e-6 is NaN",
          "source_camera_count":len(scene["R"]),"target_pose_count":len(scene["H"]),"target_reference_reads":0,
          "runtime":{"gpu":torch.cuda.get_device_name(0),"torch":str(torch.__version__),"cuda":torch.version.cuda,
                "gsplat":importlib.metadata.version("gsplat"),"fit_seconds":fit_s,"job_wall_seconds":time.monotonic()-start,
                "process_cpu_seconds":time.process_time()-cpu,"peak_torch_allocated_bytes":torch.cuda.max_memory_allocated(),
                "peak_torch_reserved_bytes":torch.cuda.max_memory_reserved()}})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,required=True)
    run(Path(__file__).resolve().parent,p.parse_args().out)
