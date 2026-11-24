"""Reopen the native T1 arrays and independently recount denominators, rankings and exact-k ties."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from evidence_lens.formal_inputs import verified_file
from general_records import resolve, stage
from g1_fixed_score import selected


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def count(mask):
    return int(np.count_nonzero(mask))


def compare_float(record,key,value):
    got=record[key]
    require((got=="" if value is None else got!="" and abs(float(got)-value)<=1e-12),
            f"independent value differs for {key}")


def check_ap(record,kind,risk,eligible,y):
    values=risk[eligible]
    events=y[eligible].astype(np.int64)
    n=len(values); positives=int(events.sum())
    expected=None
    if positives and positives<n:
        # Count the entire tied group at once in descending order.
        order=np.argsort(-values,kind="stable")
        sorted_scores=values[order]
        ends=np.r_[np.flatnonzero(np.diff(sorted_scores)!=0),n-1]
        tp=np.cumsum(events[order])[ends]
        expected=float(np.sum(np.diff(np.r_[0,tp])*(tp/(ends+1)))/positives)
    compare_float(record,kind+"_AP",expected)
    compare_float(record,kind+"_prevalence",positives/n if n else None)


def compare_anchors(records,score,common,events,perm,threshold_anchor,total):
    ids=perm[common.ravel()[perm]]
    ordered=ids[np.argsort(score.ravel()[ids],kind="stable")]
    vals=score.ravel()[ordered]; y=events.ravel()[ordered]
    for row in records:
        a=row["anchor"]
        k=threshold_anchor if a=="new_frozen_MS-L1" else (2*int(a.split("%")[0])*total+99)//200
        require(k==int(row["requested_k"]) and len(vals)==int(row["eligible"]) and count(y)==int(row["eligible_positive"]),
                "different matched population or budget")
        if k>len(vals):
            require(row["reason"]=="outside_support" and row["accepted"]=="", "out-of-support quota changed")
        elif k==0:
            require(row["reason"]=="empty_low" and row["accepted"]=="0" and row["risk"]=="", "empty low group changed")
        else:
            cut=vals[k-1]; lower=vals<cut; tied=vals==cut
            nb=count(lower); same=count(tied); need=k-nb
            before=count(y[lower]); good=count(y[tied])
            expected={"accepted":k,"accepted_positive":count(y[:k]),"cutoff_group_size":same,
                      "accepted_in_cutoff_group":need,"min_possible_accepted_positive":before+max(0,need-(same-good)),
                      "max_possible_accepted_positive":before+min(need,good)}
            require(all(int(row[col])==v for col,v in expected.items()),"accepted pixels or cutoff tie differs")
            require(np.float32(row["cutoff_score"])==cut, "different FP32 cutoff")


def audit(store,cfg,out):
    roots=(store,store/"history"); pins=cfg["sources"]
    summary_path=resolve(roots,pins["evaluation"])
    report=load(summary_path)
    op=load(resolve(roots,pins["opening"]))
    require(report["opening"]==pins["opening"] and report["matrix"]==op["matrix"],"unbound evaluator output")
    eligible=load(resolve(roots,op["matrix"]))
    require(all(report["numeric_outputs"][n] == sha(summary_path.parent/n)
                for n in ("method-views.csv","matched-views.csv")), "recorded evaluation table has changed")
    table=rows(summary_path.parent/"method-views.csv")
    matched=rows(summary_path.parent/"matched-views.csv")
    require(len(table)==288 and len(matched)==288*9, "missing evaluation rows")
    keyed={(q["object_id"],q["condition"],q["view"],q["method"]):q for q in table}
    quotas={}
    for q in matched:
        k=(q["object_id"],q["condition"],q["view"],q["method"])
        quotas.setdefault(k,[]).append(q)
    require(len(keyed)==288 and all(len(v)==9 for v in quotas.values()),"duplicated or missing score rows")
    n=0
    for cell in eligible["fixed_cells"]:
        obj=cell["object_id"]; cond=cell["condition"]
        reference_path=resolve(roots,pins["reference-"+obj])
        refs=load(reference_path)
        require(refs["opening"]==pins["opening"], "wrong photos in recount")
        native=scoredef=None
        if cell["status"]=="succeeded":
            cp=resolve(roots,cell["case"]); native=load(cp)
            sp=resolve(roots,cell["scores"]); scoredef=load(sp)
        for view,ref in refs["views"].items():
            path=verified_file(reference_path.parent,ref["rgb"]["path"],ref["rgb"])
            with Image.open(path) as photo:
                target=np.asarray(photo,dtype=np.float32)/np.float32(255)
            V=(np.isfinite(target)&(target>=0)&(target<=1)).all(axis=-1)
            E=np.zeros_like(V); errors=np.zeros_like(V); errsum=0.
            prediction_digest=""
            if native and view in native["renders"]:
                unit=native["renders"][view]; predpath=verified_file(cp.parent,unit["path"],unit)
                with np.load(predpath,allow_pickle=False) as f:
                    prediction=f["rgb"]
                    require(prediction.dtype==np.float32 and prediction.shape==target.shape==(256,256,3),
                            "native prediction violates the frozen FP32 RGB contract")
                prediction_digest=unit["sha256"]
                E=V&(np.isfinite(prediction)&(prediction>=0)&(prediction<=1)).all(axis=-1)
                err=np.abs(prediction-target).mean(axis=-1,dtype=np.float32)
                errors=E&(err>np.float32(0.05));errsum=float(err[E].sum(dtype=np.float64))
            methods=("B-L1-transfer","MS-L1","MS-frozen","MS-ft")
            risks={}
            for method in methods:
                if scoredef and scoredef["views"][view]["status"]=="ready":
                    spec=scoredef["views"][view]["methods"][method]["file"]
                    with np.load(verified_file(sp.parent,spec["path"],spec),allow_pickle=False) as f:
                        risks[method]=f["risk"]
                else:
                    risks[method]=np.full(V.shape,np.nan,np.float32)
            common=E & np.all([np.isfinite(z) for z in risks.values()],axis=0)
            msg=f"el-recovery-coverage-v1|{obj}|{cond}|{view}".encode()
            seed=int.from_bytes(hashlib.sha256(msg).digest()[:8],"big")
            perm=np.random.Generator(np.random.PCG64(seed)).permutation(V.size)
            threshold_anchor=count(common&(risks["MS-L1"].astype(np.float64)<float(keyed[obj,cond,view,"MS-L1"]["threshold"])))
            for method,risk in risks.items():
                k=(obj,cond,view,method); rec=keyed[k]
                require(rec["reference_sha256"]==ref["rgb"]["sha256"] and rec["prediction_sha256"]==prediction_digest,
                        "native input digest differs from evaluation")
                valid=E&np.isfinite(risk); lo=valid&(risk.astype(np.float64)<float(rec["threshold"]))
                hi=valid&~lo; unknown=E&~valid
                groups={"V":V,"E":E,"Y":errors,"S":valid,"SY":valid&errors,"C":common,"CY":common&errors,
                        "L":lo,"LY":lo&errors,"H":hi,"HY":hi&errors,"U":unknown,"UY":unknown&errors}
                require(all(count(mask)==int(rec[key]) for key,mask in groups.items()),"native risk denominators differ")
                compare_float(rec,"rgb_mae_sum",errsum)
                check_ap(rec,"own",risk,valid,errors);check_ap(rec,"common",risk,common,errors)
                compare_anchors(quotas[k],risk,common,errors,perm,threshold_anchor,count(V))
                n+=1
    require(n == 288, "incomplete independent native recount")
    save(out/"recount.json",{"schema":"el-t1-native-independent-recount-1","run_id":os.environ["EL_RUN_ID"],
         "primary":pins["evaluation"],"opening":pins["opening"],"view_method_rows":n,"exact_k_rows":len(matched),
         "original_reference_photos":36,"counts_checked":["V","E","Y","S","SY","C","CY","L","LY","H","HY","U","UY"],
         "also_checked":["native RGB MAE sums","own and within_new AP and prevalence","accepted errors",
                         "exact-k and complete cutoff tie bounds"],"passed":True,"risk_training":0,"threshold_refits":0})


def main(a):
    s=a.store;h=s/"history"
    if a.action=="worker":return audit(s,load(a.config),a.out)
    pins={"evaluation":selected(s,"t1-fixed-evaluation"),"opening":selected(s,"t1-opening")}
    op=load(resolve((s,h),pins["opening"]))
    pins.update({"reference-"+o:selected(s,"t1-reference-"+o) for o in op["object_ids"]})
    return stage(s,h,"t1-fixed-recount","archive_verify",pins,{"purpose":"independent native recount"},
                 ["{python}","{source}/scripts/g1_recount.py","worker","--store",str(s),"--config","{config}",
                  "--out","{out}"],"payload/recount.json",5000)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["record","worker"])
    for k in ("store","config","out"):p.add_argument("--"+k,type=Path)
    print(json.dumps(main(p.parse_args()),sort_keys=True))
