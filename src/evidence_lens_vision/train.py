"""Bounded, paired vision risk fitting. Only train and development manifests are accepted."""
import collections
import hashlib
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from .base import verified_base
from .contract import BRANCHES, HASH, RUN, SEEDS, Dataset, digest, file_record, json_bytes, output_dir, preprocessing, put, read, require, transform
from .deploy import export, open_model, score
from .configure import config
from .model import RiskModel, image_tensor

GRAD_WITNESS=("backbone.blocks.10.attn.qkv.weight","backbone.blocks.11.attn.qkv.weight",
              "backbone.norm.weight","projection.weight","head.2.weight")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=False)


def tensor_sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def hashes(model):
    return {n:tensor_sha(t) for n,t in model.state_dict().items()}


def delta_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def restore_delta(model, partial):
    names={n for n,p in model.named_parameters() if p.requires_grad}
    require(set(partial)==names,"resume or selected state has another trainable roster")
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in names:
                require(isinstance(partial[n],torch.Tensor) and partial[n].shape==p.shape and
                        partial[n].dtype==p.dtype and torch.isfinite(partial[n]).all(), "invalid selected tensor "+n)
                p.copy_(partial[n])


def change_record(model, initial, original_sha, branch):
    final=hashes(model)
    frozen={n:d for n,d in original_sha.items() if n not in initial}
    require(all(final[n]==d for n,d in frozen.items()),"a frozen tensor or state changed")
    changes={}
    for name,original in initial.items():
        current=model.get_parameter(name).detach().cpu()
        diff=current-original
        changes[name]={"before_sha256":original_sha[name],"after_sha256":final[name],"elements":diff.numel(),
                       "changed":int(torch.count_nonzero(diff)),"max_abs_change":float(diff.abs().max()),
                       "l2_change":float(torch.linalg.vector_norm(diff.double()))}
    groups=["projection.","head."]
    if branch=="L2-vision-ft":
        groups+=["backbone.blocks.10.","backbone.blocks.11.","backbone.norm."]
    require(all(any(n.startswith(g) and r["changed"]>0 for n,r in changes.items()) for g in groups),
            "one of the required trained groups has no parameter change")
    return {"frozen_before":frozen,"frozen_after":{n:final[n] for n in frozen},"frozen_identical":True,"trainable":changes}


def optimizer_for(model, cfg):
    named=dict(model.named_parameters())
    roles={"projection_head":[n for n,p in named.items() if p.requires_grad and not n.startswith("backbone.")]}
    if cfg["branch"]=="L2-vision-ft":
        roles["pretrained_suffix"]=[n for n,p in named.items() if p.requires_grad and n.startswith("backbone.")]
    records=[{"role":k,"names":names,"scalars":sum(named[n].numel() for n in names),
              "lr":cfg["backbone_lr"] if k=="pretrained_suffix" else cfg["head_lr"],"weight_decay":0.01} for k,names in roles.items()]
    opt=torch.optim.AdamW([{"params":[named[n] for n in r["names"]],"lr":r["lr"]} for r in records],
                          betas=(0.9,0.999),eps=1e-8,weight_decay=0.01,foreach=False,fused=False)
    return opt,records


def inventory(train):
    loaded=[]
    counts=collections.Counter()
    for row in sorted(train.rows,key=lambda r:(r["family"],r["view"])):
        ix,x,y=train.samples(row)
        if ix.size:
            loaded.append((row,ix,x,y))
            counts[row["family"]]+=1
    require(loaded,"training has no supervised view")
    # Every family contributes 1/F, divided over its supervised views and their fixed sampled pixels.
    weights=[1/(len(counts)*counts[r["family"]]) for r,*_ in loaded]
    return loaded,weights,dict(counts)


def epoch_order(seed,epoch,count):
    return np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(count).tolist()


def development(model, ds, pre, device, choose=False, score_dir=None):
    from evidence_lens.formal_metrics import rank, ranking_ap
    from evidence_lens.l1_freeze import threshold_for

    views={}
    score_rows=[]
    aps=collections.defaultdict(list)
    denominators=collections.Counter()
    lists=collections.defaultdict(list)
    for row in ds.rows:
        f,v=row["family"],row["view"]
        with np.load(ds.file(row["full"]["truth"]),allow_pickle=False) as z:
            V,E,labels=z["reference_valid"],z["rgb_evaluable"],z["rgb_high_error"]
        shape=(row["height"],row["width"])
        require(V.dtype==E.dtype==np.bool_ and labels.dtype==np.uint8 and V.shape==E.shape==labels.shape==shape
                and not np.any(E & ~V) and np.all(labels[E]<=1) and np.all(labels[~E]==255),"invalid independent truth domain")
        denominators[f]+=int(V.sum())
        prediction_info={"family":f,"condition":"B0","view":v,"rgb_sha256":row["rgb"]["sha256"] if row["rgb"] else None,
                         "deploy_sha256":row["full"]["deploy"]["sha256"] if row["full"]["deploy"] else None}
        if row["status"]=="render_failed":
            prediction_info.update(status="render_failed",reason=row["reason"],score=None)
            require(row["full"]["deploy"] is None,"ineligible render carries a replacement output")
            record={"status":"render_failed","n":0,"ap":None,"V":int(V.sum()),"E":int(E.sum())}
        else:
            with np.load(ds.file(row["full"]["deploy"]),allow_pickle=False) as z:
                require(set(z.files)=={"X","scoreable","l0","alpha_score"},"unknown deploy feature role")
                raw,ok,l0,alpha=z["X"],z["scoreable"],z["l0"],z["alpha_score"]
            arrays,_=score(model,ds.rgb(row),raw,ok,pre,device)
            if score_dir:
                path=score_dir/f/"B0"/(v+".npz")
                path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,risk=arrays["risk"],valid=arrays["valid"])
                prediction_info.update(status="scored",reason=None,path=path.relative_to(score_dir).as_posix())
            common=E & ok & np.isfinite(l0) & np.isfinite(alpha) & arrays["valid"]
            record,_=rank(arrays["risk"],labels==1,common)
            record["ap"]=ranking_ap(record)
            record.update(V=int(V.sum()),E=int(E.sum()))
            if record["ap"] is not None:
                aps[f].append(record["ap"])
            if choose:
                lists[f].append(arrays["risk"][E & arrays["valid"]])
        views.setdefault(f,{})[v]=record
        if score_dir: score_rows.append(prediction_info)
    families={f:float(np.mean(aps[f])) if aps[f] else None for f in sorted(ds.families)}
    defined=[a for a in families.values() if a is not None]
    require(defined,"no two-class development view on the registered shared domain")
    result={"family_equal_ap":float(np.mean(defined)),"family_ap":families,"views":views,
            "defined_families":len(defined),"registered_families":len(families)}
    if choose:
        allowed={f:np.concatenate(lists[f]) if lists[f] else np.array([],np.float32) for f in ds.families}
        result["threshold"]=threshold_for(allowed,denominators)
        result["threshold"]["domain"]="development-B0"
        if score_dir:
            tau=result["threshold"]["tau"]
            for r in score_rows:
                if r["status"]=="scored":
                    p=score_dir/r.pop("path")
                    with np.load(p,allow_pickle=False) as z:
                        risk,valid=z["risk"],z["valid"]
                    state=np.full(risk.shape,255,np.uint8)
                    state[valid]=(risk[valid].astype(np.float64)>=tau).astype(np.uint8)
                    np.savez_compressed(p,risk=risk,valid=valid,state=state)
                    r["score"]=file_record(p,score_dir)
            result["score_rows"]=score_rows
    return result


def train_group(model, opt, pre, train, loaded, weights, ids, device, state):
    opt.zero_grad(set_to_none=True)
    sums=0.0
    for i in ids:
        row,ix,x,y=loaded[i]
        weight=weights[i]*len(loaded)/len(ids)
        encoded,fraction,frame,observed=image_tensor(train.rgb(row),device)
        require(observed.ravel()[ix].all(),"sample includes invalid target RGB")
        logits,supported=model.at(model.project(encoded),fraction,frame,ix,torch.from_numpy(transform(x,pre)).to(device))
        require(supported.all().item() and torch.isfinite(logits).all().item(),"training sample lacks finite support")
        loss=F.binary_cross_entropy_with_logits(logits,torch.from_numpy(y.astype(np.float32)).to(device))
        require(torch.isfinite(loss).item(),"nonfinite task BCE")
        witness=len(state["witnesses"])<8
        grads={}
        hooks=[]
        if witness:
            def hook_for(n):
                def capture(g):
                    unscaled=g.detach().float()/weight
                    require(torch.isfinite(unscaled).all().item(),"nonfinite data-only gradient")
                    grads[n]={"l2_norm":float(torch.linalg.vector_norm(unscaled)),"max_abs":float(unscaled.abs().max()),
                              "nonzero_coordinates":int(torch.count_nonzero(unscaled))}
                return capture
            for n,p in model.named_parameters():
                if p.requires_grad and n in GRAD_WITNESS:
                    hooks.append(p.register_hook(hook_for(n)))
        try:
            (loss*weight).backward()
        finally:
            for handle in hooks:
                handle.remove()
        value=float(loss.detach())
        if witness:
            state["witnesses"].append({"family":row["family"],"view":row["view"],"epoch":state["epoch"],
                "upcoming_optimizer_step":state["steps"]+1,"samples_sha256":row["samples"]["sha256"],
                "n":len(y),"positive":int(y.sum()),"bce":value,"gradients_of_unregularized_view_mean_bce":grads})
        sums+=value*weights[i]
    for p in model.parameters():
        if p.requires_grad:
            require(p.grad is not None and torch.isfinite(p.grad).all().item(),"missing or nonfinite task gradient")
    opt.step()
    return sums


def save_state(out, model, opt, state, best, bound_sha, run_id):
    snap={"schema":"el-vision-resume-1","configuration_sha256":bound_sha,"origin_run_id":run_id,
          "live":delta_state(model),"optimizer":opt.state_dict(),"state":state,"best":best,
          "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    tmp=out/"training-state.pt.tmp"
    torch.save(snap,tmp)
    os.replace(tmp,out/"training-state.pt")


def run_fit(args, phase):
    started=time.perf_counter()
    out=output_dir(args.output)
    cfg=read(args.config)
    if os.environ.get("EL_RUN_ID") and args.run_id:
        require(os.environ["EL_RUN_ID"]==args.run_id,"explicit run ID differs from current recorder")
    config(cfg,phase,args.lr_selection)
    run_id=args.run_id or os.environ.get("EL_RUN_ID")
    require(isinstance(run_id,str) and RUN.fullmatch(run_id),"use a recorder-created run ID")
    require(digest(args.preprocessing)==cfg["preprocessing_sha256"] and digest(args.base_pin)==cfg["base_pin_sha256"],
            "preprocessing or base differs from the registered input")
    train=Dataset(args.train,"train",cfg["training_dataset_sha256"])
    dev=None
    if phase=="pilot":
        require(not args.development and set(cfg["pilot_families"])==set(train.families) and len(train.families)<=4,
                "pilot needs an explicit training-only roster of at most four families")
    else:
        require(args.development,"supply the pinned development manifest")
        dev=Dataset(args.development,"development",cfg["development_dataset_sha256"])
        require(not (train.families.keys() & dev.families.keys()) and dev.schema==train.schema,
                "training/development overlap or schema mismatch")
    if "training_family_ids" in cfg:
        require(sorted(train.families)==cfg["training_family_ids"] and
                (dev is None or sorted(dev.families)==cfg["development_family_ids"]), "registered family roster changed")
    pre=preprocessing(args.preprocessing,train_sha=train.sha,feature_sha=train.schema)
    seed_everything(cfg["seed"])
    device=torch.device(args.device)
    if device.type=="cuda":
        require(torch.cuda.is_available(),"requested CUDA training device unavailable")
        torch.cuda.reset_peak_memory_stats(device)
    official,audit=verified_base(args.official_source,args.base,args.base_pin)
    put(out/"base-verification.json",audit)
    model=RiskModel()
    model.backbone.load_state_dict(official.state_dict(),strict=True)
    del official
    # Isolate head initialization from upstream constructors, device and branch.
    torch.manual_seed(cfg["seed"])
    model.projection.reset_parameters()
    for layer in model.head:
        if isinstance(layer,torch.nn.Linear):
            layer.reset_parameters()
    model.configure(cfg["branch"])
    initial_hash=hashes(model)
    initial=delta_state(model)
    head_initial={n:p for n,p in initial.items() if n.startswith(("projection.","head."))}
    save_file(head_initial,str(out/"initial-head.safetensors"))
    model.to(device)
    opt,groups=optimizer_for(model,cfg)
    loaded,weights,counts=inventory(train)
    identity={"training_config_sha256":digest(args.config),"training_dataset_sha256":train.sha,
              "development_dataset_sha256":None if not dev else dev.sha,"feature_schema_digest":train.schema,
              "preprocessing_sha256":digest(args.preprocessing),"base_pin_sha256":digest(args.base_pin),
              "base_checkpoint_sha256":audit["base"]["checkpoint_sha256"]}
    source_dir=Path(__file__).resolve().parent
    worker_root=source_dir.parents[1]
    source_hash={p.name:digest(p) for p in sorted(source_dir.glob("*.py"))}
    handoff=worker_root/"worker-source.json"
    if handoff.is_file():
        from .contract import bound
        handoff_doc=read(handoff)
        require(handoff_doc["schema"]=="el-vision-worker-source-1","wrong transported worker inventory")
        for n,entry in handoff_doc["files"].items():
            bound(worker_root,dict(entry,path=n))
        source_hash["worker-source.json"]=digest(handoff)
    for name in ("scripts/vision_worker.py","configs/vision-base-v1.json"):
        source_hash[name]=digest(worker_root/name)
    pairing={"epochs":cfg["epochs"],"patience":cfg["patience"],"accumulate":cfg["accumulate"],
             "microbatch":cfg["microbatch"],"weight_decay":cfg["weight_decay"],"precision":cfg["precision"],
             "seed":cfg["seed"],"training":train.sha,"development":dev.sha if dev else None,"ordering":"numpy SeedSequence(seed,epoch)"}
    pairing_sha=hashlib.sha256(json_bytes(pairing)).hexdigest()
    bound_sha=hashlib.sha256(json_bytes({"identity":identity,"source":source_hash})).hexdigest()
    state={"epoch":0,"cursor":0,"steps":0,"epoch_weighted_loss":0.0,"best_ap":None,"best_epoch":None,
           "bad_epochs":0,"curve":[],"witnesses":[],"done":False,"stop_reason":None}
    best=None
    resumed=None
    if args.resume:
        require(args.resume_sha256 and digest(args.resume)==args.resume_sha256,"resume requires the original checkpoint hash")
        snap=torch.load(args.resume,map_location="cpu",weights_only=True)
        require(snap["schema"]=="el-vision-resume-1" and snap["configuration_sha256"]==bound_sha,
                "resume source, configuration, data or base differs")
        restore_delta(model,snap["live"])
        opt.load_state_dict(snap["optimizer"])
        state,best=snap["state"],snap["best"]
        torch.set_rng_state(snap["torch_rng"])
        if device.type=="cuda":
            torch.cuda.set_rng_state_all(snap["cuda_rng"])
        resumed={"run_id":snap["origin_run_id"],"checkpoint_sha256":args.resume_sha256,"from_steps":state["steps"]}
    elif args.resume_sha256:
        raise ValueError("resume hash supplied without a checkpoint file")
    evidence={"schema":"el-vision-training-setup-1","run_id":run_id,"stage":phase,"config":cfg,"source_sha256":source_hash,
              "identity":identity,"optimizer":{"algorithm":"AdamW","betas":[0.9,0.999],"epsilon":1e-8,"schedule":"constant","groups":groups},
              "initial_tensors":initial_hash,"initial_head_sha256":digest(out/"initial-head.safetensors"),
              "microbatch":1,"accumulate":4,"supervised_views_by_family":counts,"supervised_views":len(loaded),
              "epoch_objective":"equal supervised families, then supervised views, then sampled pixels; view-mean BCE",
              "precision":{"forward":"float32","gradient":"float32","optimizer":"float32","TF32":False,"AMP":False},
              "torch":str(torch.__version__),"numpy":np.__version__,"safetensors":__import__("safetensors").__version__,"device":str(device),"resume":resumed,"frozen_feature_cache":"none"}
    if device.type=="cuda":
        evidence.update(gpu=torch.cuda.get_device_name(device),cuda=str(torch.version.cuda))
    put(out/"setup.json",evidence)
    while not state["done"]:
        epoch=state["epoch"]
        order=epoch_order(cfg["seed"],epoch,len(loaded))
        cursor=state["cursor"]
        while cursor<len(order):
            ids=order[cursor:cursor+4]
            state["epoch_weighted_loss"]+=train_group(model,opt,pre,train,loaded,weights,ids,device,state)
            state["steps"]+=1
            cursor+=len(ids)
            state["cursor"]=cursor
            if phase=="pilot" and state["steps"]>=cfg["max_updates"]:
                state["done"]=True
                state["stop_reason"]="registered_pilot_update_limit"
                break
            if state["steps"]%8==0 and cursor<len(order):
                save_state(out,model,opt,state,best,bound_sha,run_id)
        row={"epoch":epoch+1,"steps":state["steps"],"processed_views":cursor,
             "processed_epoch_fraction":cursor/len(order),"sum_weighted_view_bce":state["epoch_weighted_loss"],
             "order_sha256":hashlib.sha256(json_bytes(order)).hexdigest()}
        if dev is not None:
            val=development(model,dev,pre,device)
            row["development"]=val
            ap=val["family_equal_ap"]
            if state["best_ap"] is None or ap>state["best_ap"]:
                state.update(best_ap=ap,best_epoch=epoch+1,bad_epochs=0)
                best=delta_state(model)
            else:
                state["bad_epochs"]+=1
            if state["bad_epochs"]>=cfg["patience"]:
                state.update(done=True,stop_reason="development_patience")
        state["curve"].append(row)
        state.update(epoch=epoch+1,cursor=0,epoch_weighted_loss=0.)
        if state["epoch"]>=cfg["epochs"] and not state["done"]:
            state.update(done=True,stop_reason="registered_epoch_limit")
        save_state(out,model,opt,state,best,bound_sha,run_id)
        put(out/"curves.json",state["curve"])
    if best is not None:
        restore_delta(model,best)
    required_names=[n for n in GRAD_WITNESS if n in initial]
    require(len(state["witnesses"])>=3 and all(sum(w["gradients_of_unregularized_view_mean_bce"][n]["l2_norm"]>0
            for w in state["witnesses"])>=3 for n in required_names),"insufficient nonzero task-BCE gradient evidence")
    changes=change_record(model,initial,initial_hash,cfg["branch"])
    put(out/"parameter-changes.json",changes)
    put(out/"gradient-witnesses.json",state["witnesses"])
    selected=development(model,dev,pre,device,choose=True,score_dir=out/"development-scores") if dev else None
    selected_rows=None
    if selected:
        selected_rows=selected.pop("score_rows")
        require(selected["family_equal_ap"]==state["best_ap"],"restored selected checkpoint differs from its development AP")
        put(out/"development.json",selected)
    tau=selected["threshold"]["tau"] if selected else None
    model_id=run_id
    package=out/"package"
    export(model,package,args.preprocessing,args.base_pin,Path(args.official_source)/"LICENSE",
           dict(identity,branch=cfg["branch"],model_id=model_id,seed=cfg["seed"],phase=phase),tau)
    if selected is not None:
        put(out/"development-scores/scores.json",{"schema":"el-vision-scores-1","method":cfg["branch"],
            "model_sha256":digest(package/"model.json"),"dataset_sha256":dev.sha,"views":selected_rows,
            "device":str(device),"threshold":tau,"selected_epoch":state["best_epoch"]})
    # Cross-device export check reads only one authorized training prediction and its sampled deployment features.
    sample_row,ix,x,_=loaded[0]
    rgb=train.rgb(sample_row)
    readback,reader_pre,_=open_model(package,device="cpu")
    def sampled(net,p,d):
        net.eval()
        with torch.inference_mode():
            img,mask,fr,_=image_tensor(rgb,d)
            logits,supported=net.at(net.project(img),mask,fr,ix,torch.from_numpy(transform(x,p)).to(d))
            return torch.sigmoid(logits).cpu().numpy(),supported.cpu().numpy()
    source,sup=sampled(model,pre,device)
    cpu,cpu_sup=sampled(readback,reader_pre,"cpu")
    require(np.array_equal(sup,cpu_sup) and np.isfinite(cpu).all() and np.allclose(source,cpu,rtol=2e-5,atol=2e-4),
            "reloaded CPU package disagrees with training inference")
    result={"schema":"el-vision-training-1","status":"completed","phase":phase,"run_id":run_id,"branch":cfg["branch"],
            "seed":cfg["seed"],"head_lr":cfg["head_lr"],"backbone_lr":cfg["backbone_lr"],**identity,
            "source_sha256":source_hash,"paired_design_sha256":pairing_sha,"optimizer_steps":state["steps"],"epochs":state["epoch"],"best_epoch":state["best_epoch"],
            "best_development_family_equal_ap":state["best_ap"],"stop_reason":state["stop_reason"],"threshold":tau,
            "initial_head_sha256":evidence["initial_head_sha256"],"package":file_record(package/"model.json",out),
            "frozen_identical":True,"task_gradient_minibatches":len(state["witnesses"]),
            "cpu_export":{"max_abs_error":float(np.max(np.abs(source-cpu))),"support_equal":True,"counts":len(ix),
                          "sample_sha256":sample_row["samples"]["sha256"],"rgb_sha256":sample_row["rgb"]["sha256"]},
            "peak_cuda_allocated":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
            "peak_cuda_reserved":torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None,
            "wall_seconds":time.perf_counter()-started,"resume":resumed}
    files=["setup.json","curves.json","base-verification.json","initial-head.safetensors","gradient-witnesses.json","parameter-changes.json",
           "training-state.pt"]+(["development.json","development-scores/scores.json"] if selected else [])
    # A completed-checkpoint replay recreates the new receipt and the curve in its own output.
    put(out/"curves.json",state["curve"])
    if not (out/"training-state.pt").exists():
        shutil.copyfile(args.resume,out/"training-state.pt")
    result["files"]={n:file_record(out/n,out) for n in files}
    put(out/"training.json",result)
    return result
