"""Materialize six authored, reference-free deployment bundles from native sealed producers."""

import argparse
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from dual_records import bind, resolve, selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens_asset.permissions import source_grant
from evidence_lens_bundle.bundle import Bundle
from evidence_lens_bundle.common import PROFILE, RELEASE_INDEX_SHA256, SEMANTICS, encoded, profile, require, row
from evidence_lens_bundle.common import sha as digest
from evidence_lens_bundle.grids import build_grids, resample_photo
from v05_bell_common import CHECKPOINT, OLD, SIZES, SOURCES, TARGETS, projections
from v05_bell_control import historical


def portable(folder, name, raw):
    p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    return row(name,raw)


def assemble(store):
    roots=store,store/"history"
    rb=selected(store,"a-bell-gpu-readback");data=load(resolve(roots,rb));native=data["native"]
    pins={"readback":rb,"native":native,"probe":data["depth_probe"],"photos":selected(store,"a-bell-source-observations"),
          "registration":selected(store,"a-bell-register-six"),"reconstruction":historical(store,"reconstruction"),"cameras":historical(store,"cameras")}
    nr=load(resolve(roots,native));src=load(resolve(roots,pins["photos"]))
    for tag, frames in nr["frames"].items():
        for view,f in frames.items():
            pins["frame-"+tag+"-"+view]=bind(store,native["run_id"],"payload/"+f["path"])
            require(pins["frame-"+tag+"-"+view]["sha256"]==f["sha256"],"Windows frame differs from its receipt")
    for tag, photos in src["products"].items():
        for view,f in photos.items():
            pins["photo-"+tag+"-"+view]=bind(store,pins["photos"]["run_id"],"payload/"+f["rgb8"]["path"])
    cmd=["{python}","-B","{source}/scripts/v05_bell_bundles.py","worker","--config","{config}","--out","{out}"]
    for k in pins:cmd += ["--input",k,"{input:"+k+"}"]
    cmd += ["--input","profile","{input:profile}"]
    profile_file=Path(__file__).resolve().parents[1]/"src/evidence_lens_bundle/assets/profile.json"
    originals=[]
    for path in sorted((store/"work/controller/a-bell-bundles").glob("attempt-*.jsonl")):
        rec=json.loads(path.read_text().splitlines()[-1]);originals.append({k:rec[k] for k in ("run_id","manifest_sha256","execution_status")})
    return stage(store,"a-bell-bundle-publication","dataset_build",pins,{"expected_images":20,"expected_frames":60,"score_inputs_only":True,
                 "profile_sha256":sha(profile_file),"prior_packaging_attempts":originals},cmd,"payload/bundles.json",220_000_000,{"profile":profile_file})


def one(out, tag, grid, reg, photos, native, rec, cfg, inputs):
    root=out/"bundles"/tag
    w,h=grid["analysis_size"]
    legacy=tag==OLD
    observations={}
    frames={}
    evidence={}
    source_order=source_grant(reg["new_source_acquisition"],"fixed-M4")
    require(source_order==SOURCES and photos["role"]=="sources" and photos["target_reference_renders"]==0,
            "unexpected photo access")
    grids=None if legacy else build_grids(f"analysis-{w}x{h}-v1",grid["original_views"])
    for name in SOURCES+TARGETS:
        if grids is not None:
            require(np.array_equal(np.asarray(grids["views"][name]["analysis_K"],np.float32),
                                   np.asarray(grid["analysis_views"][name]["K"],np.float32)),"actual CUDA intrinsics disagree with photo grid")
        spec=native["frames"][tag][name]
        raw=inputs["frame-"+tag+"-"+name].read_bytes()
        require(digest(raw)==spec["sha256"],"changed original native frame")
        frames[name]={"status":"ready","file":portable(root,"renders/"+name+".npz",raw)}
        if name not in source_order:continue
        kind=grid["source_shape"]
        src=photos["products"][kind][name]
        p=inputs["photo-"+kind+"-"+name]
        require(src["camera"]==grid["original_views"][name] and sha(p)==src["rgb8"]["sha256"],"source observation/camera differs")
        with Image.open(p) as im:
            if legacy:
                require(im.mode=="RGB" and im.size==(896,504),"wrong landscape original")
                sampled=Image.new("RGB",(256,256));sampled.paste(im.resize((256,144),Image.Resampling.LANCZOS),(0,56))
            else:
                sampled=resample_photo(im,grids["views"][name],(w,h))
        buf=io.BytesIO();sampled.save(buf,format="PNG")
        observations[name]=portable(root,"sources/"+name+".png",buf.getvalue())
        evidence[name]={"original_camera":src["camera"],"original_rgb8_sha256":src["rgb8"]["sha256"],
                        "analysis_png_sha256":observations[name]["sha256"],"original_fit_source_sha256":rec["source_rgb_sha256"][name]}
    camera={"schema":"el-camera-1","unit":"m","pixel_center":"integer","image_origin":"top_left",
            "axis_convention":"OpenCV x right, y down, z forward","views":grid["analysis_views"]}
    camera_spec=portable(root,"cameras.json",encoded(camera))
    pf=cfg["sources"]
    p={"schema":"el-render-evidence-1" if legacy else "el-render-evidence-2",
       "generator":{"run_id":rec["run_id"],"receipt_sha256":pf["reconstruction"]["sha256"]},
       "renderer":{"run_id":native["run_id"],"receipt_sha256":pf["native"]["sha256"]},
       "checkpoint_sha256":CHECKPOINT,"source_rgb_sha256":{k:v["sha256"] for k,v in observations.items()},
       "source_order":SOURCES,"target_order":TARGETS,"camera_sha256":camera_spec["sha256"],"semantics":SEMANTICS,
       "depth_verified":True,"depth_probe_sha256":pf["probe"]["sha256"],"frames":frames}
    if grids is not None:p["grids_sha256"]=digest(encoded(grids))
    b={"schema":"el-evidence-bundle-1" if legacy else "el-evidence-bundle-2", "input_domain":"authored offset-bell-cage, previously released development reconstruction",
       "creation":{"run_id":os.environ["EL_RUN_ID"],"implementation":"v0.5 native M4 bell source and renderer assembly","code_sha256":sha(Path(__file__)),
                   "config_sha256":digest(encoded(cfg)),"original_case_sha256":None},
       "condition":"M4","source_order":SOURCES,"primary_source_id":"s00","endpoint_source_ids":["s00","s03"],"target_order":TARGETS,
       "camera":camera_spec,"geometry":{"prepared_camera_and_z_unit":"m","meters_per_original_unit":1.0,"metric_evidence":"Original designed metric Blender camera rig",
           "crop":"Registered original whole camera field","resize":"256 × 144 landscape inside 256 × 256" if legacy else "2× original source to native analysis size",
           "undistortion":"Original pinhole; no distortion","scoring_resampling":"none"},
       "sources":observations,"producer":portable(root,"producer.json",encoded(p)),
       "risk_profile":{"id":PROFILE,"sha256":digest(encoded(profile())),"release_index_sha256":RELEASE_INDEX_SHA256}}
    if grids is not None:b["grids"]=grids
    spec=portable(root,"bundle.json",encoded(b))
    checked=Bundle(root,spec["sha256"]).validate()
    require(checked["status"]=="valid" and checked["render_failed"]=={},"incomplete prepared bell evidence")
    proof={"scope":"The four original fit photos were 256-square; the scoring source views are new native observations at the same poses.",
           "license":"CC BY 4.0, wany614","original_observations":evidence,"observation_run":pf["photos"],"native_render":pf["native"],
           "source_resampling":"display-sRGB8 Pillow LANCZOS; old path pads measured landscape after resampling" if legacy else grids["photo_resampling"],
           "analysis_size":grid["analysis_size"],"active_xywh":grid["active_xywh"],"registered_camera_views":grid["original_views"],
           "evaluation_reference_reads":0}
    proof_spec=portable(root,"source-observations.json",encoded(proof))
    return {"path":"bundles/"+tag+"/bundle.json","sha256":spec["sha256"],"analysis_size":[w,h],"active_xywh":grid["active_xywh"],
            "source_observation_sha256":proof_spec["sha256"],"validation":checked}


def worker(cfg,inputs,out):
    import evidence_lens_bundle.common as common

    require(sha(inputs["profile"])==cfg["profile_sha256"],"changed released risk profile")
    common.ASSETS=inputs["profile"].parent
    reg=load(inputs["registration"]);native=load(inputs["native"]);photos=load(inputs["photos"]);rec=load(inputs["reconstruction"])
    require(reg["source_order"]==SOURCES and photos["registration"]==cfg["sources"]["registration"] and
            rec["checkpoint_sha256"]==native["checkpoint_sha256"]==CHECKPOINT and native["gaussians"]==4096
            and native["target_reference_reads"]==0 and load(inputs["probe"])["passed"],"incorrect source/map/render registration")
    grids=projections(load(inputs["cameras"])["views"])
    routes={tag:one(out,tag,grid,reg,photos,native,rec,cfg,inputs) for tag,grid in grids.items()}
    for name,role in (("native-render.json","native"),("original-reconstruction.json","reconstruction"),("depth-probe.json","probe")):
        portable(out,"proofs/"+name,inputs[role].read_bytes())
    index={"schema":"el-v05-bell-bundles-1","run_id":os.environ["EL_RUN_ID"],"registration":cfg["sources"]["registration"],
           "sources":SOURCES,"targets":TARGETS,"routes":routes,"readback":cfg["sources"]["readback"],"source_observations":cfg["sources"]["photos"],
           "checkpoint_sha256":CHECKPOINT,"native_frames":60,"reference_images_in_archive":0,
           "license":{"original_bell_photographs_and_derived_renders":"CC-BY-4.0","attribution":"wany614, Evidence Lens","url":"https://creativecommons.org/licenses/by/4.0/"}}
    save(out/"index.json",index)
    portable(out,"README.md",b"# Evidence Lens authored bell\n\nSix fixed-M4 scoring bundles share the original Gaussian scene and camera poses. Each bundle has four observed source photos and six target predictions. See index.json and each source-observations.json for camera grids, model origin and resampling. All included original data and derived images are CC BY 4.0, wany614, Evidence Lens. https://creativecommons.org/licenses/by/4.0/\n")
    archive=out/"v05-bell-bundles.zip"
    files=sorted(p for p in out.rglob("*") if p.is_file())
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in files:z.write(p,p.relative_to(out).as_posix())
    save(out/"bundles.json",{**index,"archive":{"path":archive.name,"bytes":archive.stat().st_size,"sha256":sha(archive)},"archive_index_sha256":sha(out/"index.json")})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("assemble","worker"))
    for key in ("config","out","store"):p.add_argument("--"+key,type=Path)
    p.add_argument("--input",action="append",nargs=2,default=[])
    a=p.parse_args()
    if a.action=="worker":worker(load(a.config),{k:Path(v) for k,v in a.input},a.out)
    else:print(json.dumps(assemble(a.store.resolve())))
