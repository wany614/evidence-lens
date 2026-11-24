"""Render only the registered role in the original, self-contained authored Blender scene."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import bpy
import numpy as np

from evidence_lens._bundle import load, save, sha
from evidence_lens_asset.permissions import source_grant
from evidence_lens_bundle.common import require
from v05_bell_common import OLD, SIZES, SOURCES, TARGETS, camera_on_grid


def run(role, cfg, inputs, out):
    reg = load(inputs["registration"])
    for k, v in inputs.items():
        require(sha(v) == cfg["sources"][k]["sha256"], "changed registered input " + k)
    require(reg["schema"] == "el-v05-bell-processing-1" and reg["source_order"] == SOURCES
            and reg["target_order"] == TARGETS and reg["analysis_wh"] == {k:list(v) for k,v in SIZES.items()}, "unregistered native views")
    if role == "references":
        op = load(inputs["opening"])
        require(op.get("schema") == "el-v05-bell-reference-opening-1" and op.get("registered_scores_sealed") is True
                and op.get("terminal_score_count") == 48 and op.get("targets") == TARGETS
                and op.get("conditions") == ["landscape", OLD] and op.get("registration") == cfg["sources"]["registration"],
                "complete sealed landscape scoring gate required before the six independent measurements")
        names, shapes = TARGETS, ["landscape"]
    else:
        require(role == "sources" and "opening" not in inputs, "unexpected source authority")
        names, shapes = source_grant(reg["new_source_acquisition"], "fixed-M4"), list(SIZES)
        require(names == SOURCES, "wrong static M4 photos")
    helpers = Path(__file__).resolve().parents[1] / "src/evidence_lens/blender_reference"
    sys.path.insert(0, str(helpers))
    from passes import display_srgb, read_passes, save_png
    from scene import camera_record
    from validation import render

    all_cameras = load(inputs["cameras"])["views"]
    settings = load(inputs["settings"])
    bpy.ops.wm.open_mainfile(filepath=str(inputs["world"]), load_ui=False)
    scene = bpy.context.scene
    require(not len(bpy.data.libraries) and not bpy.utils.blend_paths(absolute=False, packed=True, local=True),
            "original bell world is not self-contained")
    require((scene.render.resolution_x,scene.render.resolution_y,scene.render.resolution_percentage)==(256,256,100), "changed original scene grid")
    for n in names:
        require(camera_record(scene,bpy.data.objects[n],all_cameras[n]["purpose"]) == all_cameras[n], "native scene camera differs from the signed metadata")
    scene.cycles.seed = settings["seed"]
    scene.cycles.device = "CPU"
    rows = {}
    with tempfile.TemporaryDirectory() as td, (out/"access.jsonl").open("x") as log:
        for shape in shapes:
            w,h = SIZES[shape]
            scene.render.resolution_x,scene.render.resolution_y = 2*w,2*h
            folder = out/shape
            folder.mkdir()
            rows[shape] = {}
            for n in names:
                clock = time.perf_counter()
                scene.camera=bpy.data.objects[n]
                measured=camera_record(scene,scene.camera,all_cameras[n]["purpose"])
                expected=camera_on_grid(all_cameras[n],2*w,2*h)
                require(np.allclose(measured["K"],expected["K"],rtol=0,atol=1e-9)
                        and expected["c2w"]==measured["c2w"], "physical lens or pose changed")
                exr=Path(td)/(shape+"-"+n+".exr")
                log.write(json.dumps({"event":"start_native_render","role":role,"shape":shape,"view":n,"time_ns":time.time_ns()})+"\n")
                log.flush()
                render(scene,exr)
                linear,_,_,_=read_passes(exr)
                rgb=display_srgb(linear)
                require(rgb.shape==(2*h,2*w,3) and np.isfinite(rgb).all(),"invalid native display photo")
                f=folder/(n+".png");save_png(f,rgb)
                row={"view":n,"camera":expected,"rgb8":{"path":str(f.relative_to(out)),"sha256":sha(f),"bytes":f.stat().st_size},
                     "native_wall_s":time.perf_counter()-clock,"license":"CC-BY-4.0"}
                if role=="references":
                    fp=folder/(n+"-display-fp32.npz");np.savez_compressed(fp,rgb=rgb)
                    row["display_fp32"]={"path":str(fp.relative_to(out)),"sha256":sha(fp),"bytes":fp.stat().st_size}
                rows[shape][n]=row
                exr.unlink()
    save(out/"photos.json", {"schema":"el-v05-bell-photos-1","run_id":os.environ["EL_RUN_ID"],"role":role,"products":rows,
         "registration":cfg["sources"]["registration"],"opening":cfg["sources"].get("opening"),"world":cfg["sources"]["world"],
         "camera_origin":cfg["sources"]["cameras"],"bpy":bpy.app.version_string,"cycles_device":scene.cycles.device,
         "cycles_samples":scene.cycles.samples,"seed":scene.cycles.seed,"original_size_multiplier":2,
         "source_renders":len(shapes)*len(names) if role=="sources" else 0,
         "target_reference_renders":len(shapes)*len(names) if role=="references" else 0})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("role",choices=("sources","references"));p.add_argument("--config",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--input",action="append",nargs=2,required=True)
    a=p.parse_args();run(a.role,load(a.config),{k:Path(v) for k,v in a.input},a.out)
