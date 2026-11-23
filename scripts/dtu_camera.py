"""Select the DTU rig from official metadata and seal the physical coordinate checks."""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from evidence_lens.dtu_camera import (CONDITIONS, SOURCES, TARGETS, camera_outputs, digest,
                                      encoded, record, require, write)

BOUGUET="https://robots.stanford.edu/cs223b04/JeanYvesCalib/htmls/parameters.html"
C_COLOR="SampleSet/MVS Data/Rectified/scan%d/rect_%03d_3_r5000.png"


def build(a):
    readme=a.readme.read_bytes()
    evaluator=a.evaluator.read_bytes()
    calmat=loadmat(a.matrices/"Calib_Results_right.mat",squeeze_me=True,
                   variable_names=["dX","dY","X_1","nx","ny","alpha_c"])
    notes=readme.decode("cp1252",errors="replace")
    eval_text=evaluator.decode("ascii",errors="replace")
    for term in ("corrected for radial distortion","most diffuse lighting","color images are supplied by the right camera",
                 "same frame of reference as the structured light reconstruction","0-10 mm"):
        require(term in notes,"The sealed DTU documentation changed: "+term)
    require("0-10 mm" in eval_text and "dist_tresshold=10;" in eval_text.replace(" ","")
            and "min(BaseEval.Ddata,dist_tresshold)/dist_tresshold" in eval_text
            and "min(BaseEval.Dstl,dist_tresshold)/dist_tresshold" in eval_text,
            "No explicit DTU world-distance to millimetre equivalence")
    w,h=int(calmat["nx"]),int(calmat["ny"])
    require((w,h)==(1600,1200) and int(calmat["dX"])==int(calmat["dY"])==11 and float(calmat["alpha_c"])==0
            and np.array_equal(np.unique(calmat["X_1"][0]),np.arange(0,122,11)),
            "Calibration grid, image origin source or original image dimensions changed")
    names={r["name"] for r in json.loads(a.directory.read_bytes())}
    ids=[i for i in range(1,65) if all(C_COLOR%(obj,i) in names for obj in (1,6))]
    require(len(ids)>=10,"Too few available rectified cameras")
    projections={i:np.loadtxt(a.matrices/f"pos_{i:03d}.txt") for i in ids}
    ref=a.baseline.read_bytes()
    require(digest(ref)=="1b6cb179e40b14c5874e4bc89ff8eecef6aa3fa94836cbf37d2b25f9f637267e","Stage C layout pin changed")
    result=camera_outputs(projections,w,h,0.001,0,json.loads(ref))
    viewset={"schema":"el-dtu-view-set-1","source_view_ids":{k:result["ids"][k] for k in SOURCES},
             "target_view_ids":{k:result["ids"][k] for k in TARGETS},"available_common_capture_ids":ids,
             "objects":["set1","set6"],"lighting_id":3,"selection":result["fit"]}
    out=a.out
    write(out/"view-set.json",encoded(viewset))
    for name,key in (("crop","crop"),("cameras","cameras"),("metric-frame","frame")):
        write(out/(name+".json"),encoded(result[key]))
    write(out/"projection-checks.json",encoded({"schema":"el-dtu-projection-checks-1","passed":True,"views":result["checks"],
          "maximum_abs_px":max(x["max_abs_reprojection_px"] for x in result["checks"].values()),
          "threshold_px":0.005,"axial_threshold_m":1e-10}))
    for cond,src in CONDITIONS.items():
        for stem,ids_ in (("source_cameras",src),("render_cameras",src+TARGETS)):
            write(out/"condition"/cond/(stem+".json"),encoded({**result["cameras"],"views":{k:result["cameras"]["views"][k] for k in ids_}}))
    evidence={"readme":record(readme),"world_distance_evaluator":record(evaluator),"right_calibration":record((a.matrices/"Calib_Results_right.mat").read_bytes()),
              "projections":{str(i):record((a.matrices/f"pos_{i:03d}.txt").read_bytes()) for i in ids},
              "archive_member_name_index":record(a.directory.read_bytes()),"stage3_layout":record(ref),
              "metric_derivation":"DTU supplies the right-colour projections in STL world coordinates. Its original MATLAB evaluator maps the raw nearest-neighbour distances to a 0–10 mm colour scale with literal cutoff 10. The right calibration plate coordinates advance by 11 native units.",
              "pixel_frame_authority":BOUGUET,"pixel_origin":"Bouguet camera model, upper-left pixel centre [0,0]. Native pos_N matrices are used directly; rectification intrinsics are obtained from those matrices, not substituted from the original distorted right calibration.",
              "rectification":"DTU Rectified colour PNG matched to cal18 pos_N text. No second undistortion."}
    plan={"schema":"el-dtu-camera-plan-1","converter":"dtu-crop256-v1","objects":["set1","set6"],"lighting_id":3,
          "source_view_ids":viewset["source_view_ids"],"target_view_ids":viewset["target_view_ids"],"camera_sha256":digest((out/"cameras.json").read_bytes()),
          "crop_sha256":digest((out/"crop.json").read_bytes()),"view_set_sha256":digest((out/"view-set.json").read_bytes()),
          "metric_frame_sha256":digest((out/"metric-frame.json").read_bytes()),"projection_checks_sha256":digest((out/"projection-checks.json").read_bytes()),
          "evidence":evidence,"condition_cameras":{c:{n:record((out/"condition"/c/(n+".json")).read_bytes()) for n in ("source_cameras","render_cameras")} for c in CONDITIONS}}
    write(out/"camera-plan.json",encoded(plan))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("readme","evaluator","matrices","directory","baseline","out"):p.add_argument("--"+n,type=Path,required=True)
    build(p.parse_args())
