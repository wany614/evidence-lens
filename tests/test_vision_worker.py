"""Synthetic data only. No external pretrained checkpoint, study family or GPU is needed."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/"src"))
torch=pytest.importorskip("torch")
pytest.importorskip("safetensors")
import evidence_lens_vision.train as training
from evidence_lens_vision.contract import COLUMNS, Dataset, digest, file_record, put, read, transform
from evidence_lens_vision.deploy import infer, open_model
from evidence_lens_vision.model import Backbone, Frame, RiskModel, image_tensor
from evidence_lens_vision.selection import choose

torch.set_num_threads(2)


def preprocess_doc(train_sha):
    from evidence_lens.l1 import schema_digest
    return {"schema":"el-vision-preprocessing-1","columns":COLUMNS,"training_dataset_sha256":train_sha,
            "feature_schema_digest":schema_digest(),"medians":[0]*11,"means":[0]*11,"scales":[1]*11}


def matrix(rng,shape):
    x=np.zeros((*shape,26),np.float32)
    x[...,:11]=rng.uniform(0,1,size=(*shape,11))
    x[...,12]=1
    x[...,10]=np.nan
    x[...,25]=1
    return x


def make_data(path,split,seed):
    from evidence_lens.l1 import schema_digest
    path.mkdir()
    family="synthetic-"+split
    h,w=8,12
    rows=[]
    rng=np.random.default_rng(seed)
    for n in range(6):
        v=f"v{n+1:02d}"
        dest=path/family/"B0"/v
        dest.mkdir(parents=True)
        rgb=rng.random((h,w,3),dtype=np.float32)
        rgb[0,0]=np.nan
        np.save(dest/"rgb.npy",rgb)
        x=matrix(rng,(h,w))
        ix=np.arange(1,h*w,dtype=np.int32)
        y=(x[...,2]>.55).astype(np.uint8)
        np.savez_compressed(dest/"samples.npz",index=ix,X=x.reshape(-1,26)[ix],Y=y.ravel()[ix])
        full=None
        if split!="train":
            ok=np.ones((h,w),bool);ok[0,0]=False
            np.savez_compressed(dest/"deploy.npz",X=x,scoreable=ok,l0=np.zeros((h,w),np.float32),alpha_score=x[...,0])
            y[~ok]=255
            np.savez_compressed(dest/"truth.npz",reference_valid=np.ones_like(ok),rgb_evaluable=ok,rgb_high_error=y)
            full={k:file_record(dest/(k+".npz"),path) for k in ("deploy","truth")}
        rows.append(dict(family=family,condition="B0",view=v,height=h,width=w,status="completed",reason=None,
                         eligible=len(ix),sample_count=len(ix),rgb=file_record(dest/"rgb.npy",path),
                         samples=file_record(dest/"samples.npz",path),full=full))
    put(path/"dataset.json",dict(schema="el-vision-dataset-1",split=split,columns=COLUMNS,feature_schema_digest=schema_digest(),
                                 label_version="el-labels-1",views=rows))
    return path/"dataset.json"


def test_native_pixel_centre_mapping_and_padding_support():
    torch.manual_seed(3)
    m=RiskModel()
    data=np.zeros((17,103,3),np.float32);data[:,2]=np.nan
    image,mask,frame,native_ok=image_tensor(data,"cpu")
    assert frame==Frame(17,103,42,252,105,0)
    assert np.all(~native_ok[:,2]) and mask.shape==(1,1,18,18)
    assert np.isclose(float(mask.sum()),42*252/196)
    assert torch.count_nonzero(image[:,:,:105])==0
    ids=np.array([0,2,102,824,1750],np.int32)
    patch_tokens=torch.randn(1,32,18,18,requires_grad=True)
    raw=torch.randn(len(ids),26)
    logits,supported=m.at(patch_tokens,mask,frame,ids,raw)
    grid=frame.grid(ids,"cpu")
    cfg=dict(mode="bilinear",align_corners=False,padding_mode="border")
    frac=torch.nn.functional.grid_sample(mask,grid,**cfg)
    expected=torch.nn.functional.grid_sample(patch_tokens*mask,grid,**cfg)/frac
    assert torch.allclose(logits,m.head(torch.cat((expected[0,:,:,0].T,raw),1)).flatten(),atol=2e-6)
    assert supported.all()
    logits.sum().backward()
    assert torch.isfinite(patch_tokens.grad).all() and patch_tokens.grad.abs().sum()>0


def test_trainable_groups_and_malformed_features():
    m=RiskModel()
    m.configure("L2-frozen")
    assert all(not p.requires_grad for p in m.backbone.parameters())
    assert set(training.witness_parameters(m,"L2-frozen"))=={"head.2.weight","projection.weight"}

    m.configure("L2-vision-ft")
    assert all(p.requires_grad==n.startswith(("backbone.blocks.10.","backbone.blocks.11.","backbone.norm.","head.","projection."))
               for n,p in m.named_parameters())
    assert set(training.witness_parameters(m,"L2-vision-ft"))==set(training.GRAD_WITNESS)
    m.backbone.norm.weight.requires_grad_(False)
    with pytest.raises(ValueError,match="disagree with trainable"):
        training.witness_parameters(m,"L2-vision-ft")
    assert 21_000_000 < sum(p.numel() for p in m.parameters()) < 25_000_000
    raw=matrix(np.random.default_rng(4),(15,))
    p=preprocess_doc("a"*64)
    assert np.isfinite(transform(raw,p)).all()
    raw[0,25]=0
    with pytest.raises(ValueError,match="feature flag"):
        transform(raw,p)


def test_bounded_choice_ties_pairing_and_no_seed_shopping():
    def row(branch,seed,lr,ap,n):
        return {"schema":"el-vision-training-1","phase":"train","status":"completed","run_id":"run"+str(n),"manifest_sha256":"b"*64,
                "model_sha256":"c"*64,"training_dataset_sha256":"d"*64,"development_dataset_sha256":"e"*64,"preprocessing_sha256":"f"*64,
                "base_pin_sha256":"9"*64,"source_sha256":{"worker":"a"*64},"initial_head_sha256":str(seed),"paired_design_sha256":str(seed),
                "branch":branch,"seed":seed,"best_epoch":1,"best_development_family_equal_ap":ap,"optimizer_steps":2,"frozen_identical":True,
                "task_gradient_minibatches":6,"head_lr":lr if branch=="L2-frozen" else 3e-4,
                "backbone_lr":lr if branch=="L2-vision-ft" else 0.,"threshold":.45}
    rows=[row("L2-frozen",1729,1e-4,.6,0),row("L2-frozen",1729,3e-4,.6,1),
          row("L2-vision-ft",1729,1e-5,.5,2),row("L2-vision-ft",1729,3e-5,.7,3)]
    result=choose(rows,"learning-rate")
    assert result["chosen"]["L2-frozen"]["head_lr"]==1e-4 and result["chosen"]["L2-vision-ft"]["backbone_lr"]==3e-5
    for seed in (2718,3141):
        rows.extend([row("L2-frozen",seed,1e-4,.99,len(rows)),row("L2-vision-ft",seed,3e-5,.98,len(rows)+1)])
    assert all(r["seed"]==1729 for r in choose(rows,"primary")["chosen"].values())
    rows[-1]["backbone_lr"]=1e-5
    with pytest.raises(ValueError,match="paired seeds"):
        choose(rows,"primary")


def test_reject_test_material_before_read(monkeypatch,tmp_path):
    from evidence_lens_vision import batch
    import types
    pkg=dict(branch="L2-vision-ft",model_id="unit",phase="train",threshold=.2,seed=1729)
    monkeypatch.setattr(batch,"open_model",lambda *a:(None,{},pkg))
    method=tmp_path/"freeze.json";put(method,{"unit":True})
    marker=types.ModuleType("vision_evaluate")
    def closed(*args):
        raise ValueError("closed independent gate")
    marker.test_access=closed
    marker.model_matches=lambda *a: True
    monkeypatch.setitem(sys.modules,"vision_evaluate",marker)
    args=SimpleNamespace(output=tmp_path/"out",package="ignored",model_sha256="a"*64,device="cpu",split="test",
                         dataset=tmp_path/"hidden-do-not-open",dataset_sha256="b"*64,store=tmp_path,shelf=tmp_path,
                         catalog=tmp_path/"catalog.json",catalog_sha256="c"*64,method_freeze=method,method_sha256=digest(method))
    with pytest.raises(ValueError,match="closed independent gate"):
        batch.dataset_scores(args)


def test_synthetic_finetune_restore_export_and_blind_inference(tmp_path,monkeypatch):
    train=make_data(tmp_path/"train","train",2)
    dev=make_data(tmp_path/"dev","development",3)
    prep=tmp_path/"preprocessing.json";put(prep,preprocess_doc(digest(train)))
    pin=tmp_path/"base.json";put(pin,{"synthetic_weight_identity":"fixture only"})
    src=tmp_path/"synthetic-source";src.mkdir();(src/"LICENSE").write_text("synthetic test")
    cfg={"schema":"el-vision-run-1","phase":"train","branch":"L2-vision-ft","seed":1729,"epochs":2,"patience":3,
         "head_lr":.0003,"backbone_lr":.00001,"weight_decay":.01,"microbatch":1,"accumulate":4,"precision":"float32",
         "base_pin_sha256":digest(pin),"preprocessing_sha256":digest(prep),"training_dataset_sha256":digest(train),
         "development_dataset_sha256":digest(dev)}
    config_file=tmp_path/"config.json";put(config_file,cfg)
    # Manufactured random initialization is isolated to this synthetic subprocess-free regression.
    def synthetic(*args):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(771)
            base=Backbone()
        return base,{"base":{"checkpoint_sha256":"1"*64},"fixture":True}
    monkeypatch.setattr(training,"verified_base",synthetic)
    rid="20260101T000000000000Z-0123456789abcdef"
    a=SimpleNamespace(config=config_file,base=tmp_path/"fake",base_pin=pin,official_source=src,train=train,development=dev,
                      preprocessing=prep,lr_selection=None,resume=None,resume_sha256=None,output=tmp_path/"fit",device="cpu",run_id=rid)
    result=training.run_fit(a,"train")
    assert result["optimizer_steps"]==4 and result["task_gradient_minibatches"]==8 and result["frozen_identical"]
    assert result["cpu_export"]["max_abs_error"]<1e-6
    assert result["cpu_export"]["all_tensor_sha256_equal"] and result["cpu_export"]["tensor_count"]==181
    setup=read(a.output/"setup.json")
    flags=setup["precision"]["backend_readback"]
    assert flags["deterministic_algorithms"] and not any(flags[n] for n in ("cudnn_benchmark","cuda_matmul_allow_tf32","cudnn_allow_tf32","deterministic_warn_only"))
    assert flags["CUBLAS_WORKSPACE_CONFIG"]==os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    assert set(setup["gradient_witness_names"])==set(training.GRAD_WITNESS)
    assert {"src/evidence_lens/formal_metrics.py","src/evidence_lens/l1_freeze.py","src/evidence_lens_vision/train.py"}<=result["source_sha256"].keys()
    checkpoint=torch.load(a.output/"training-state.pt",weights_only=True,map_location="cpu")
    assert json.loads(json.dumps(checkpoint["state"]["curve"],allow_nan=False))==read(a.output/"curves.json")
    assert checkpoint["state"]["curve"][0]["development"]["views"]["synthetic-development"]["v01"]["status"]=="two_class"

    frozen_dev=read(a.output/"development-scores/scores.json")
    assert len(frozen_dev["views"])==6 and frozen_dev["model_sha256"]==result["package"]["sha256"]
    for vr in frozen_dev["views"]:
        with np.load(a.output/"development-scores"/vr["score"]["path"],allow_pickle=False) as z:
            valid=z["valid"]; expected=np.full(valid.shape,255,np.uint8)
            expected[valid]=(z["risk"][valid].astype(np.float64)>=result["threshold"]).astype(np.uint8)
            assert np.array_equal(expected,z["state"])
    package=a.output/"package"
    changes=read(a.output/"parameter-changes.json")
    full_backbone,_,_=open_model(package,result["package"]["sha256"])
    assert training.tensor_readback(full_backbone,changes)["tensor_table_sha256"]==result["cpu_export"]["tensor_table_sha256"]
    with torch.no_grad():
        full_backbone.head[2].bias.add_(1)
    with pytest.raises(ValueError,match="differ from the selected"):
        training.tensor_readback(full_backbone,changes)
    del full_backbone
    assert changes["frozen_before"]==changes["frozen_after"] and all(changes["trainable"][n]["changed"]>0 for n in training.GRAD_WITNESS)
    assert (a.output/"training-state.pt").is_file() and 80_000_000<(package/"weights.safetensors").stat().st_size<110*2**20
    full=dev.parent/"synthetic-development/B0/v01"
    # The single-view deployment path functions with no training/test label files present.
    (full/"truth.npz").rename(tmp_path/"unmounted-truth.npz")
    inf=infer(package,full/"rgb.npy",full/"deploy.npz",tmp_path/"infer","cpu",result["package"]["sha256"])
    with np.load(tmp_path/"infer/scores.npz",allow_pickle=False) as z:
        assert z["risk"].shape==(8,12) and np.isnan(z["risk"][0,0]) and z["state"][0,0]==255
        assert np.array_equal(np.isfinite(z["risk"]),z["valid"])
    assert inf["threshold"]==result["threshold"]
    # The exported native reader reconstructs camera features from pixels and predictions in isolation.
    from PIL import Image
    native=tmp_path/"native-only";native.mkdir()
    src_png=native/"input.png";Image.new("RGB",(12,8),(45,80,25)).save(src_png)
    camera={"width":12,"height":8,"K":[[10,0,5.5],[0,10,3.5],[0,0,1]],"c2w":np.eye(4).tolist()}
    cams=native/"cameras.json";put(cams,{"schema":"el-camera-1","unit":"m","pixel_center":"integer",
        "views":{k:camera for k in ["source",*[f"v{i:02d}" for i in range(1,7)]]}})
    pred=native/"predictions";pred.mkdir()
    for k in ("source","v01"):
        np.savez_compressed(pred/(k+".npz"),rgb=np.full((8,12,3),.3,np.float32),z=np.full((8,12),2.,np.float32),
                            alpha=np.ones((8,12),np.float32))
    put(pred/"prediction.json",{"schema":"el-prediction-1","rgb_encoding":"display_srgb_0_1","depth_semantics":"expected_camera_z_m",
        "alpha_semantics":"accumulated_opacity","input_sha256":digest(src_png),"camera_sha256":digest(cams),
        "producer_run_id":"synthetic","depth_validation":{"off_axis_plane_passed":True,"alpha_normalization_passed":True}})
    measured=tmp_path/"timed-native"
    proc=subprocess.run([sys.executable,"-I",str(package/"score-native.py"),"--input",str(src_png),"--cameras",str(cams),
                         "--predictions",str(pred),"--model-sha256",result["package"]["sha256"],"--view","v01",
                         "--warm-repeats","1","--output",str(measured)],cwd=tmp_path,capture_output=True,text=True,
                         env=dict(os.environ,OMP_NUM_THREADS="2"))
    assert proc.returncode==0,proc.stderr
    cost=read(measured/"native-cost.json")
    assert cost["all_measured_views_scored"] and cost["warm_views"]==1 and cost["hot_p95_seconds"]>0
    assert not cost["all_measured_views_512"] and cost["process_tree_peak_rss_bytes"]>0
    (tmp_path/"unmounted-truth.npz").rename(full/"truth.npz")
    old_setup=read(a.output/"setup.json")
    a.resume=a.output/"training-state.pt";a.resume_sha256=digest(a.resume)
    a.output=tmp_path/"replayed"
    replay=training.run_fit(a,"train")
    assert result["package"]["sha256"]==replay["package"]["sha256"] and replay["optimizer_steps"]==4
    assert old_setup["initial_tensors"]==read(a.output/"setup.json")["initial_tensors"]

def test_development_fails_on_boolean_support_or_duplicate_view(tmp_path,monkeypatch):
    manifest=make_data(tmp_path/"dev","development",93)
    ds=Dataset(manifest,"development",digest(manifest))
    pre=preprocess_doc("7"*64)
    def scored(*args):
        risk=np.full((8,12),.25,np.float32)
        return {"risk":risk,"valid":np.isfinite(risk)},{}
    monkeypatch.setattr(training,"score",scored)
    with pytest.raises(ValueError,match="original scoreable support"):
        training.development(None,ds,pre,"cpu")
    def legal(*args):
        risk=np.full((8,12),.25,np.float32);risk[0,0]=np.nan
        return {"risk":risk,"valid":np.isfinite(risk)},{}
    monkeypatch.setattr(training,"score",legal)
    ds.rows.append(ds.rows[0])
    with pytest.raises(ValueError,match="non-B0 view"):
        training.development(None,ds,pre,"cpu")
    ds.rows.pop()
    wrong=tmp_path/"dev/invalid-scoreable.npz"
    original=ds.file(ds.rows[0]["full"]["deploy"])
    with np.load(original,allow_pickle=False) as values:
        np.savez(wrong,**{k:values[k].astype(np.uint8) if k=="scoreable" else values[k] for k in values.files})
    ds.rows[0]["full"]["deploy"]=file_record(wrong,manifest.parent)
    with pytest.raises(ValueError,match="scoreable bool"):
        training.development(None,ds,pre,"cpu")
