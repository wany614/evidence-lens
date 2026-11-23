"""Full-backbone inference takes one target prediction and its authorized numerical features."""
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .contract import COLUMNS, bound, digest, file_record, output_dir, preprocessing, put, read, require, transform
from .model import ARCH, RiskModel, image_tensor


def synchronize(device):
    if torch.device(device).type=="cuda":
        torch.cuda.synchronize(device)


def score(model, rgb, raw, scoreable, pre, device="cpu", threshold=None):
    require(raw.shape==(*rgb.shape[:2], len(COLUMNS)) and raw.dtype==np.float32 and
            scoreable.shape==rgb.shape[:2] and scoreable.dtype==np.bool_,"malformed native deploy feature/mask")
    synchronize(device)
    start=time.perf_counter()
    with torch.inference_mode():
        encoded,fraction,frame,observed=image_tensor(rgb,device)
        synchronize(device)
        prep=time.perf_counter()
        projected=model.project(encoded)
        synchronize(device)
        backbone=time.perf_counter()
        risk=np.full(frame.height*frame.width,np.nan,np.float32)
        native=raw.reshape(-1,len(COLUMNS))
        ids=np.flatnonzero((scoreable & observed).ravel())
        for begin in range(0,len(ids),16384):
            chunk=ids[begin:begin+16384]
            values=torch.from_numpy(transform(native[chunk],pre)).to(device)
            logits,valid=model.at(projected,fraction,frame,chunk,values)
            require(torch.isfinite(logits[valid]).all().item(),"nonfinite risk on a supported target pixel")
            vals=torch.sigmoid(logits).cpu().numpy()
            supported=valid.cpu().numpy()
            risk[chunk[supported]]=vals[supported]
        synchronize(device)
    risk=risk.reshape(frame.height,frame.width)
    valid=np.isfinite(risk)
    arrays={"risk":risk,"valid":valid}
    if threshold is not None:
        states=np.full(risk.shape,255,np.uint8)
        states[valid]=(risk[valid].astype(np.float64)>=threshold).astype(np.uint8)
        arrays["state"]=states
    times={"target_preprocess_seconds":prep-start,"complete_backbone_and_projection_seconds":backbone-prep,
           "features_sampling_and_head_seconds":time.perf_counter()-backbone}
    return arrays, {"frame":frame.json(),"finite_native_rgb_pixels":int(observed.sum()),"supported_pixels":int(valid.sum()),
                    "unknown_pixels":int((~valid).sum()),"threshold":threshold,"timings":times}


def export(model, out, pre_path, base_pin, license_path, identity, threshold):
    out=output_dir(out)
    state={name:value.detach().cpu().contiguous() for name,value in model.state_dict().items()}
    save_file(state,str(out/"weights.safetensors"),metadata={"format":"pt"})
    put(out/"architecture.json",ARCH)
    for src,dest in ((pre_path,"preprocessing.json"),(base_pin,"base.json"),(license_path,"DINOv2-LICENSE")):
        shutil.copyfile(src,out/dest)
    package=Path(__file__).parent
    target=out/"runtime/evidence_lens_vision"
    target.mkdir(parents=True)
    for name in ("__init__.py","contract.py","model.py","deploy.py","native.py"):
        shutil.copyfile(package/name,target/name)
    # The native entry recomputes camera geometry and legal features from the original prediction.
    import evidence_lens
    legacy=Path(evidence_lens.__file__).resolve().parent
    core=out/"runtime/evidence_lens"
    core.mkdir()
    for name in ("__init__.py","features.py","l1.py","risk_model.py","scoring_io.py"):
        shutil.copyfile(legacy/name,core/name)
    project=package.parents[1]
    if not (project/"LICENSE").is_file():
        project=legacy.parents[1]
    for name in ("LICENSE","NOTICE"):
        shutil.copyfile(project/name,out/name)
    native_launcher='''from pathlib import Path
import sys
import time
boot=time.perf_counter()
sys.path.insert(0,str(Path(__file__).resolve().parent/"runtime"))
from evidence_lens_vision.native import standalone
standalone(Path(__file__).resolve().parent,boot)
'''
    (out/"score-native.py").write_text(native_launcher,encoding="utf-8")
    launcher='''from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"runtime"))
from evidence_lens_vision.deploy import standalone
standalone(Path(__file__).resolve().parent)
'''
    (out/"infer.py").write_text(launcher,encoding="utf-8")
    (out/"requirements.txt").write_text("numpy>=1.26,<3\ntorch>=2.8,<3\nsafetensors>=0.5,<1\nPillow>=11,<13\npsutil>=6,<8\n",encoding="utf-8")
    files={p.relative_to(out).as_posix():file_record(p,out) for p in sorted(out.rglob("*")) if p.is_file()}
    index={"schema":"el-vision-model-1",**identity,"threshold":threshold,"files":files,"dtype":"float32",
           "complete_backbone_included":True,"total_parameters":sum(v.numel() for v in model.parameters()),
           "weights_bytes":(out/"weights.safetensors").stat().st_size,
           "score_interpretation":"sigmoid risk score for display-sRGB error; calibration unassessed"}
    put(out/"model.json",index)
    return index


def open_model(root, expected=None, device="cpu"):
    root=Path(root)
    index=read(root/"model.json")
    require((expected is None or digest(root/"model.json")==expected) and index["schema"]=="el-vision-model-1"
            and index["complete_backbone_included"] is True and index["dtype"]=="float32","wrong frozen model identity")
    for name,entry in index["files"].items():
        require(name==entry["path"],"model package file key differs")
        bound(root,entry)
    required={"weights.safetensors","architecture.json","preprocessing.json","base.json","DINOv2-LICENSE"}
    require(required<=index["files"].keys() and read(root/"architecture.json")==ARCH,"unsupported architecture or package")
    pre=preprocessing(root/"preprocessing.json",train_sha=index["training_dataset_sha256"],feature_sha=index["feature_schema_digest"])
    model=RiskModel()
    weights=load_file(str(root/"weights.safetensors"),device="cpu")
    require(all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in weights.values()),"invalid deployment tensor")
    model.load_state_dict(weights,strict=True)
    tau=index["threshold"]
    require(tau is None or type(tau) in (int,float) and np.isfinite(tau) and 0<=tau<=1.000000000000001,
            "invalid frozen development threshold")
    return model.to(device).eval(),pre,index


def infer(package, rgb_path, deploy, output, device="cpu", expected=None):
    out=output_dir(output)
    start=time.perf_counter()
    model,pre,idx=open_model(package,expected,device)
    loaded=time.perf_counter()
    rgb=np.load(rgb_path,allow_pickle=False)
    with np.load(deploy,allow_pickle=False) as z:
        require(set(z.files)=={"X","scoreable","l0","alpha_score"}, "only the deployment feature bundle may be supplied")
        raw,ok=z["X"],z["scoreable"]
    inputs=time.perf_counter()
    arrays,details=score(model,rgb,raw,ok,pre,device,idx["threshold"])
    np.savez_compressed(out/"scores.npz",**arrays)
    receipt={"schema":"el-vision-inference-1","run_id":os.environ.get("EL_RUN_ID"),"model_id":idx["model_id"],"branch":idx["branch"],
             "model_manifest_sha256":digest(Path(package)/"model.json"),"rgb_sha256":digest(rgb_path),
             "deploy_sha256":digest(deploy),"output":file_record(out/"scores.npz",out),"device":str(device),**details}
    receipt["timings"].update(package_load_seconds=loaded-start,read_arrays_seconds=inputs-loaded,
                             wall_through_output_seconds=time.perf_counter()-start)
    put(out/"inference.json",receipt)
    return receipt


def standalone(package):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("rgb","deploy","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--model-sha256")
    p.add_argument("--device",default="cpu",choices=("cpu","cuda"))
    a=p.parse_args()
    infer(package,a.rgb,a.deploy,a.output,a.device,a.model_sha256)
