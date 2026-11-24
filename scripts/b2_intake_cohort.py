"""Register one geometry-only eligibility pool and apportion C6, T1 and reserves."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from evidence_lens.dtu_camera import CLIP_M, decompose
from t0_dtu_archive import check, digest, write_json

S0 = (24,22)
FIXED = (24,23,17,22)
TARGET = (16,18,25,37,35,36)
POOL = (7,10,13,17,20,23,31,40)
UNION = tuple(sorted(set(FIXED)|set(TARGET)|set(POOL)))
SEED = 137031
DOCUMENTED_POOL = (32,34,37,42,45,48,52,54,61,62,75,76,77,103,110,118)


def retrieve(path):
    return json.loads(path.read_bytes())


def register(config,out):
    check(len(UNION)==16 and len(set(POOL))==8 and not (set(TARGET)&(set(POOL)|set(FIXED)))
          and not (set(POOL)&set(S0)) and set(FIXED)-set(S0)<=set(POOL),"incompatible image roles")
    write_json(out/"allocation-plan.json",{
        "schema":"el-b2-dtu-allocation-plan-1","run_id":os.environ["EL_RUN_ID"],"utc":config["registered_utc"],
        "seed":SEED,"random_priority":"Lower SHA-256 digest of UTF-8 evidence-lens-b2-split-v1|137031| followed by the frozen physical_family_id. Ties by numeric official scan ID.",
        "eligible_universe":list(DOCUMENTED_POOL),"upstream":config["pinned"],"source_endpoints":list(S0),"fixed_sources":list(FIXED),
        "evaluation_targets":list(TARGET),"candidate_cameras":list(POOL),"complete_unique_capture_ids":list(UNION),"lighting_id":3,
        "common_eligibility":"Same 16 distinct positive-sized official Rectified members, positive native capture IDs, accepted method/CRC/offset, verified right-camera native P and shared metric frame for every object. Retain original subject/rotation uncertainty decisions.",
        "allocation":"Sort the entire eligible family population by priority. First six C6, next six T1, remainder in reserve rank order. Technical missing members or invalid calibration exclude a family before assigning either role. Preserve every exclusion. If fewer than twelve pass, assign first six C6 then subsequent T1 and report the shortfall. No post-assignment replacement based on any image, feature count, reconstruction, prediction or outcome. Any subsequent documented accessibility failure remains in that arm.",
        "authority":{"C6":"fixed registered source members only after the cohort is sealed; evaluation references await a complete prediction/score matrix and calibration method lock",
                     "T1":"no photographs or derived observations in batch two","candidates":"metadata only for C6/T1 allocation; Q_probe is virtual, selector receives no evaluation-target poses"},
        "transfer_budget":{"C6_sources_unique":24,"C6_targets_unique":36,"T1_photo_downloads":0,"max_response_bytes_per_data_cell":32*1024**2},
        "archival_notice":"No scene photographs, reference geometry, observations or performance were read to allocate the cohort."})


def calibrated_cameras(roles, plan, crop, frame, files, old):
    affine=np.asarray(crop["affine"],dtype=np.float64)
    transform=np.asarray(frame["dtu_metres_to_canonical_metres"],dtype=np.float64)
    factor=frame["metres_per_native_unit"]
    check(factor==.001 and frame["endpoint_ids"]==list(S0),"wrong canonical rig basis")
    mapping={**plan["source_view_ids"],**plan["target_view_ids"]}
    result={}
    for n in roles:
        p=files[n]; check(digest(p.read_bytes())==plan["evidence"]["projections"][str(n)]["sha256"],"changed original projection")
        proj=np.loadtxt(p);native_k,pose=decompose(proj,factor,0);k=affine@native_k
        check(abs(k[0,1])<.003 and abs(k[1,0])<=1e-9,"unsupported skew")
        k[0,1]=k[1,0]=0; c2w=transform@pose
        check(np.isfinite(c2w).all() and np.allclose(c2w[3],[0,0,0,1]) and float(np.linalg.det(c2w[:3,:3]))>0,"invalid pose")
        cam={"width":256,"height":256,"K":k.tolist(),"c2w":c2w.tolist(),"clip_m":CLIP_M}
        for logical,number in mapping.items():
            if n==number:check(cam==old["views"][logical],"reproduced fixed rig differs")
        xy=np.asarray([[0.,0.],[255.,0.],[0.,255.],[255.,255.],[127.5,127.5],[52.25,101.5]])
        pix=np.c_[xy,np.ones(len(xy))]
        maxerr=0.
        for z in (.21,.87,2.43):
            xyz=np.linalg.solve(k,pix.T).T*z
            native=(pose[:3,:3]@xyz.T).T+pose[:3,3]
            h=(proj@np.c_[native/factor,np.ones(len(native))].T).T
            projected=(affine@(h/h[:,2,None]).T).T
            maxerr=max(maxerr,float(np.max(np.abs(projected[:,:2]-xy))))
        check(maxerr<.005,"native-to-crop reprojection exceeds frozen threshold")
        result[n]={"official_view_id":n,"projection_sha256":digest(p.read_bytes()),"max_reprojection_px":maxerr,"calibration":cam}
    return result


def allocate(args,config):
    reg=retrieve(args.plan);prior=retrieve(args.roster);archive=retrieve(args.archive)
    check(reg["seed"]==SEED and reg["eligible_universe"]==list(DOCUMENTED_POOL) and reg["complete_unique_capture_ids"]==list(UNION),"allocation protocol changed")
    check(archive==prior["archive"] and prior["lighting_id"]==3,"wrong archive or light")
    index={r["name"]:r for r in retrieve(args.members)}
    rows=[r for r in prior["inventory"] if r["disposition"]=="eligible_beyond_registered_budget"]
    check(tuple(r["scan_id"] for r in rows)==DOCUMENTED_POOL and len({r["physical_family_id"] for r in rows})==len(rows),"registered physical pool changed")
    plan=retrieve(args.camera);crop=retrieve(args.crop);frame=retrieve(args.frame)
    original={int(k):Path(p) for k,p in args.projection}
    geo=calibrated_cameras(UNION,plan,crop,frame,original,retrieve(args.fixed))
    cal_hash=digest(args.camera.read_bytes())
    allrows=[]
    for r in rows:
        n=r["scan_id"];members={vi:index.get(f"Rectified/scan{n}/rect_{vi:03}_3_r5000.png") for vi in UNION}
        absent=[vi for vi,m in members.items() if not m or m["bytes"]<=0 or m["compressed"]<=0 or m["offset"]<=0 or m["method"] not in (0,8) or len(m["crc32"])!=8]
        priority=digest(f"evidence-lens-b2-split-v1|{SEED}|{r['physical_family_id']}".encode())
        allrows.append({"object_id":r["object_id"],"scan_id":n,"physical_family_id":r["physical_family_id"],"primary_subject":r["primary_subject"],
            "documented_or_possible_shared_group":r["documented_or_possible_shared_group"],"declared_rotation_group":r["declared_rotation_group"],
            "priority_sha256":priority,"eligible":not absent,"missing_or_incompatible_capture_ids":absent,"members":members,
            "source_members":{k:members[v] for k,v in plan["source_view_ids"].items()},"target_members":{k:members[v] for k,v in plan["target_view_ids"].items()}})
    ordered=sorted((r for r in allrows if r["eligible"]),key=lambda r:(r["priority_sha256"],r["scan_id"]))
    for i,r in enumerate(ordered):
        r.update(rank=i+1,role="C6" if i<6 else "T1" if i<12 else "reserve")
    for r in allrows:
        r.setdefault("role","ineligible");r.setdefault("rank",None)
    split={name:[r["object_id"] for r in ordered if r["role"]==name] for name in ("C6","T1","reserve")}
    camera={**prior["camera"],"plan_sha256":cal_hash}
    doc={"schema":"el-b2-dtu-cohort-1","run_id":os.environ["EL_RUN_ID"],"allocation_plan":config["pinned"]["allocation"],
         "upstream_roster":config["pinned"]["roster"],"seed":SEED,"archive":archive,"camera":camera,"conditions":["M2","M4"],
         "source_view_ids":plan["source_view_ids"],"target_view_ids":plan["target_view_ids"],"target_ids":list(plan["target_view_ids"]),
         "endpoints":list(S0),"candidate_cameras":list(POOL),"required_capture_ids":list(UNION),"lighting_id":3,"splits":split,
         "objects":allrows,"budget":reg["transfer_budget"],"permissions":reg["authority"],"photo_bodies_read":0}
    write_json(args.out/"cohort.json",doc)
    write_json(args.out/"camera-eligibility.json",{"schema":"el-b2-native-rig-audit-1","required_capture_ids":list(UNION),
        "reprojection_tolerance_px":.005,"max_reprojection_px":max(r["max_reprojection_px"] for r in geo.values()),"cameras":geo})
    # This file is suitable for a selector. The eligibility audit also contains held-out camera geometry and stays out of any selector package.
    safe=sorted(set(POOL)|set(S0))
    write_json(args.out/"candidate-cameras.json",{"schema":"el-b2-candidate-rig-1","camera_convention":"OpenCV integer centre; metres",
        "endpoint_ids":list(S0),"candidate_ids":list(POOL),"cameras":{i:geo[i]["calibration"] for i in safe}})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("register","allocate"))
    for k in ("config","out","plan","roster","archive","members","camera","crop","frame","fixed"):p.add_argument("--"+k,type=Path,required=k in ("config","out"))
    p.add_argument("--projection",nargs=2,action="append",default=[])
    a=p.parse_args();cfg=retrieve(a.config)
    if a.action=="register":register(cfg,a.out)
    else:allocate(a,cfg)
