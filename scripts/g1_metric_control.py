"""Check registered denominator, failure, tie and ranking rules on small generated arrays before opening."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from dtu_eval_metrics import labels, observation
from dtu_score import METHODS
from evidence_lens._bundle import save
from evidence_lens._verify import require
from g1_fixed_score import selected
from g1_recount import check_ap, compare_anchors
from general_records import stage
from t0_evaluate import population


def csv_row(row):
    return {k: "" if v is None else str(v) for k, v in row.items()}


def worker(out):
    obj, cond, view = "generated-control", "M2", "v01"
    shape = (15, 17)
    yy, xx = np.indices(shape)
    gt = np.zeros((*shape,3), np.float32)
    src = np.broadcast_to(np.where((yy+xx)%4==0,.14,.04).astype(np.float32)[...,None],gt.shape).copy()
    fg = np.ones(shape,bool)
    tau = dict.fromkeys(METHODS,.5)
    msg = f"el-recovery-coverage-v1|{obj}|{cond}|{view}".encode()
    seed = int.from_bytes(hashlib.sha256(msg).digest()[:8],"big")
    perm = np.random.Generator(np.random.PCG64(seed)).permutation(fg.size)
    rows=[]
    for name in ("mixed_with_ties","all_scores_tied","missing_scores","all_negative","all_positive","missing_prediction"):
        p = src.copy()
        if name == "all_negative":
            p.fill(0)
        if name == "all_positive":
            p.fill(.2)
        if name == "missing_prediction":
            p = None
        v,e,y,err,aux=labels(gt,p,fg)
        r={m:((yy+i*xx)%3).astype(np.float32)/2 for i,m in enumerate(METHODS,1)}
        if name in ("missing_prediction","missing_scores"):
            for f in r.values():
                f.fill(np.nan)
        elif name == "all_scores_tied":
            for f in r.values():
                f.fill(.5)
        else:
            r[METHODS[0]][(xx+yy)%9==0]=np.nan
        scores,_=observation(v,e,y,err,aux,r,tau)
        common,quotas=population(v,e,y,r,tau,obj,cond,view)
        anchor=int((common&(r["MS-L1"].astype(np.float64)<tau["MS-L1"])).sum())
        for m in METHODS:
            row=scores[m]; own=e&np.isfinite(r[m])
            for kind,eligible in [("own",own),("common",common)]:
                rec=row[kind]
                rec={kind+"_AP":rec["ap"], kind+"_prevalence":rec["positive"]/rec["n"] if rec["n"] else None}
                check_ap(csv_row(rec),kind,r[m],eligible,y==1)
            compare_anchors([csv_row(q) for q in quotas if q["method"]==m],r[m],common,y==1,perm,anchor,int(v.sum()))
            if name == "missing_prediction":
                require(row["counts"]["V"]==fg.size and row["counts"]["E"]==row["counts"]["Y"]==0,
                        "a missing render must leave the complete reference denominator and zero comparable pixels")
        rows.append(dict(control=name, reference_pixels=int(v.sum()), comparable_pixels=int(e.sum()),
                         common_score_pixels=int(common.sum()), ranking_checks=8, quota_checks=len(quotas)))
    save(out/"control.json",{"schema":"el-t1-metric-control-1","run_id":os.environ["EL_RUN_ID"],
                            "data_role":"generated numerical unit controls","target_photographs_read":0,"controls":rows})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("action",choices=["record","worker"])
    for k in ("store","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.action=="worker":
        worker(a.out)
    else:
        r=a.store; ep=selected(r,"g0-entry")
        print(json.dumps(stage(r,r/"history","t1-metric-control","evaluation",{"entry":ep},
              {"data":"deterministic generated arrays; reference gate closed"},
              ["{python}","{source}/scripts/g1_metric_control.py","worker","--out","{out}"],"payload/control.json",5000)))
