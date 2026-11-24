"""Check the new empirical arithmetic on synthetic cases and the previously exposed DTU set1."""
import argparse
import json
import os
import runpy
from pathlib import Path
import sys
import numpy as np
from b2_calibration import ObjectData,fit
from t0_dtu_archive import check,write_json


def run(a):
    cfg=json.loads(a.config.read_bytes())
    tests=runpy.run_path(str(Path(__file__).resolve().parents[1]/"tests/test_b2_calibration.py"))
    completed=[]
    for name,func in tests.items():
        if name.startswith("test_"):
            func();completed.append(name)
    actual=[]
    for s,l in a.view:
        with np.load(s,allow_pickle=False) as scores,np.load(l,allow_pickle=False) as labels:
            actual.append((scores["risk"],labels["full_crop_V"],labels["full_crop_E"],labels["full_crop_Y"]))
    check(len(actual)==6,"D fixture needs its full exposed view set")
    group=ObjectData.from_views("set1",actual)
    f=fit([group]);max_diff=0
    for row in f["curves"]:
        tau=row["tau"];low=[e & np.isfinite(r) & (r.astype(np.float64)<tau) for r,v,e,y in actual]
        l=sum(int(mask.sum()) for mask in low)
        ly=sum(int((mask&(y==1)).sum()) for mask,(_,v,e,y) in zip(low,actual))
        official=group.count(tau)
        check((l,ly)==(official["L"],official["LY"]),"201-candidate D threshold counts disagree with direct masks")
    write_json(a.out/"calibration-validation.json",{"schema":"el-b2-calibration-validation-1","run_id":os.environ["EL_RUN_ID"],"passed":True,
               "development_object":"set1","historical_arm":"original published D1 256-step fixed M2 interface","selected_new_parameters":False,
               "D_targets":len(actual),"D_valid_pixels":group.V,"D_evaluable_pixels":group.E,"D_deployment_finite":len(group.deployment),
               "D_thresholds_verified_against_direct_boolean_masks":len(f["curves"]),"synthetic_cases":completed,
               "C6_target_reads":0,"T1_photo_reads":0})

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("config","out"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--view",nargs=2,action="append",required=True);run(p.parse_args())
