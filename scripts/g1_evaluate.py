"""Independent fixed-layout T1 evaluation with the frozen metrics and one collective opening."""

import argparse
from collections import defaultdict
import html
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np
from dtu_eval import photo, score_map
from dtu_eval_board import board
from dtu_eval_metrics import labels, observation
from dtu_score import METHODS, VIEWS
from evidence_lens._bundle import load, save, sha, utc
from evidence_lens._verify import require
from evidence_lens.multisource_io import Case
from g1_fixed_score import selected
from general_records import resolve, stage
from recovery_eval_coverage import PERCENTS
from recovery_eval_risk import coverage_totals, reasons
from t0_evaluate import COUNTS, mask_id, population, ratio, reduce_rows, table
from t1_access import H, OBJECTS


def originals(roots,obj,opening,pin,cohort):
    path=resolve(roots,pin)
    ref=load(path)
    subject,=[s for s in cohort["objects"] if s["object_id"]==obj and s["role"]=="T1"]
    require(ref["schema"]=="el-t1-references-1" and ref["opening"]==opening and ref["object_id"]==obj
            and ref["physical_family_id"]==subject["physical_family_id"] and set(ref["views"])==set(VIEWS),
            "reference is outside the joint opening")
    frames={}
    for v,cam in zip(VIEWS,H,strict=True):
        r=ref["views"][v]; registered=subject["members"][str(cam)]
        require(r["official_view_id"]==cam and r["archive_member"]==registered["name"]
                and r["original_crc32"]==registered["crc32"] and r["original_bytes"]==registered["bytes"]
                and r["valid_mask"]=={"kind":"all_true","width":256,"height":256}
                and r["object_mask"] is None, "unexpected reference member or mask")
        frames[v]=photo(path.parent,r["rgb"])
    return ref,frames


def equal_objects(object_rows,matched):
    rows=[]
    for cond in ("M2","M4"):
        for method in METHODS:
            group=[r for r in object_rows if (r["condition"],r["method"])==(cond,method)]
            require(len(group)==len(OBJECTS),"missing object in equal-weight summary")
            metrics=("MAE","error_rate","coverage","low_error","unknown","common_AP","common_prevalence",
                     "own_AP","own_prevalence")
            values={k:[r[k] for r in group if r[k] is not None] for k in metrics}
            for anchor in [f"{n}%_V" for n in PERCENTS]+["new_frozen_MS-L1"]:
                values["within_new_risk_"+anchor]=[r["risk"] for r in matched if
                    (r["condition"],r["method"],r["anchor"])==(cond,method,anchor) and r["risk"] is not None]
            for metric,vs in values.items():
                rows.append(dict(condition=cond,method=method,metric=metric,objects_scheduled=len(group),
                                 objects_defined=len(vs),object_equal_mean=mean(vs) if vs else None,
                                 minimum=min(vs,default=None),maximum=max(vs,default=None)))
    return rows


def evaluate(store,cfg,out):
    roots=(store,store/"history")
    pins=cfg["sources"]
    opened=load(resolve(roots,pins["opening"]))
    require(opened["schema"]=="el-t1-joint-opening-1" and opened["object_ids"]==OBJECTS
            and opened["matrix"]==pins["matrix"] and opened["fixed_score_count"]==288,"wrong study opening")
    entry=load(resolve(roots,opened["entry"]))
    freeze=load(resolve(roots,entry["freeze"]))
    cohort=load(resolve(roots,freeze["cohort"]))
    tau={m:freeze["released_models"][m]["threshold"] for m in METHODS}
    cells=load(resolve(roots,pins["matrix"]))["fixed_cells"]
    public=out/"public"; private=out/"private"
    public.mkdir(); private.mkdir()
    all_views=[]; matches=[]; causes=[]; object_rows=[]; matched_objects=[]; source_rows=[]; boards=[]
    first_read=utc(); read=[]
    for obj in OBJECTS:
        ref,photos=originals(roots,obj,pins["opening"],pins["reference-"+obj],cohort)
        read.extend(dict(object_id=obj,view=v,**{k:row[k] for k in ("official_view_id","original_sha256")},
                         crop_sha256=row["rgb"]["sha256"]) for v,row in ref["views"].items())
        for c in [x for x in cells if x["object_id"]==obj]:
            cond=c["condition"]; case=None; sc={}; source_images={}
            if c["status"]=="succeeded":
                case=Case(resolve(roots,c["case"]).parent,c["case"]["sha256"])
                ss=resolve(roots,c["scores"]); sc=load(ss)
                for sid,(image,frame,_) in case.sources("all").items():
                    source_images[sid]=image
                    valid,e,y,err,_=labels(image,frame["rgb"] if frame else None,np.ones((256,256),bool))
                    source_rows.append(dict(object_id=obj,condition=cond,view=sid,V=int(valid.sum()),E=int(e.sum()),
                                            Y=int((y==1).sum()),source_MAE=ratio(float(err[e].sum(dtype=np.float64)),int(e.sum()))))
            by_method=defaultdict(list); quotas=defaultdict(list)
            for v in VIEWS:
                frame=case.frame(v) if case else None
                ready=frame is not None and sc["views"][v]["status"]=="ready"
                reference=photos[v]; rgb=frame["rgb"] if frame else None
                valid,e,y,err,aux=labels(reference,rgb,np.ones((256,256),bool))
                risks={m:score_map(ss.parent,sc,m,v,case,tau[m]) if ready else
                       np.full(valid.shape,np.nan,np.float32) for m in METHODS}
                result,_=observation(valid,e,y,err,aux,risks,tau)
                common,exact=population(valid,e,y,risks,tau,obj,cond,v)
                matches.extend(exact)
                for row in exact:
                    quotas[row["method"],row["anchor"]].append(row)
                if ready:
                    unknown,_,_=reasons(case,v,frame,risks,e,y)
                    for method,groups in unknown.items():
                        causes.extend(dict(object_id=obj,condition=cond,view=v,method=method,reason=reason,**counts)
                                      for reason,counts in groups.items())
                for method,res in result.items():
                    record=dict(object_id=obj,condition=cond,view=v,method=method,status="scored" if ready else c["status"] if not case else "render_failed",
                                threshold=tau[method],V_mask_sha256=mask_id(valid),E_mask_sha256=mask_id(e),
                                common_mask_sha256=mask_id(common),S_mask_sha256=mask_id(e&np.isfinite(risks[method])),
                                label_sha256=mask_id(y),reference_sha256=ref["views"][v]["rgb"]["sha256"],
                                prediction_sha256=case.meta["renders"][v]["sha256"] if frame else None)
                    record.update({k:res["counts"][k] for k in COUNTS})
                    for key in ("own","common"):
                        group=res[key]
                        record[key+"_AP"]=group["ap"]
                        record[key+"_prevalence"]=ratio(group["positive"],group["n"])
                        record[key+"_AP_reason"]=group["missing_reason"]
                    all_views.append(record);by_method[method].append(record)
                name=f"{obj}-{cond}-{v}.png"
                board(private/name,obj,cond,v,source_images,rgb,reference,(valid,e,y,err,aux),risks,tau,
                      frame["alpha"] if frame else None)
                boards.append(name)
            for m,group in by_method.items():
                object_rows.append(dict(object_id=obj,physical_family_id=ref["physical_family_id"],condition=cond,
                                        method=m,status=c["status"],views_scheduled=len(VIEWS),
                                        views_scored=sum(r["status"]=="scored" for r in group),threshold=tau[m],**reduce_rows(group)))
            for (m,anchor),items in quotas.items():
                matched_objects.append(dict(object_id=obj,condition=cond,method=m,domain="within_new",
                                            anchor=anchor,**coverage_totals(items)))
    require(len(all_views)==288 and len(object_rows)==48 and len(boards)==72 and len(read)==36,
            "incomplete fixed-layout independent evaluation")
    for name,rows in (("method-views",all_views),("method-conditions",object_rows),("matched-views",matches),
                      ("matched-conditions",matched_objects),("object-equal",equal_objects(object_rows,matched_objects)),
                      ("unknown-reasons",causes),("source-fit",source_rows)):
        table(public,name+".csv",rows)
    (private/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>T1 private panels</title>'
        '<h1>Frozen T1 independent evaluation</h1><ul>'+''.join(
        f'<li><a href="{html.escape(n)}">{html.escape(n[:-4])}</a></li>' for n in boards)+'</ul>',encoding="utf-8")
    save(public/"evaluation.json",dict(schema="el-t1-fixed-evaluation-1",run_id=os.environ["EL_RUN_ID"],
         opening=pins["opening"],matrix=pins["matrix"],first_evaluator_read_utc=first_read,actual_references=read,
         objects=OBJECTS,scheduled=288,scored=sum(v["status"]=="scored" for v in all_views),
         label="FP32 mean absolute display-sRGB error strictly above 0.05",matched_domain="within_new",
         calibration={n:{"status":p["status"],"tau":p["tau"]} for n,p in freeze["layout_profiles"].items()},
         verified_object_mask=None,new_risk_training=0,threshold_refits=0,
         numeric_outputs={p.name:sha(p) for p in public.glob("*.csv")}))


def main(a):
    s=a.store; roots=(s,s/"history")
    if a.action=="worker":
        return evaluate(s,load(a.config),a.out)
    opening=selected(s,"t1-opening")
    op=load(resolve(roots,opening))
    pins={"opening":opening,"matrix":op["matrix"]}
    pins.update({"reference-"+o:selected(s,"t1-reference-"+o) for o in OBJECTS})
    return stage(*roots,"t1-fixed-evaluation","evaluation",pins,{"domain":"full_center_crop","risk_events":"FP32 RGB > 0.05"},
                 ["{python}","{source}/scripts/g1_evaluate.py","worker","--store",str(s),"--config","{config}",
                  "--out","{out}"],"payload/public/evaluation.json",130000000)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["record","worker"])
    for name in ("store","config","out"):
        p.add_argument("--"+name,type=Path)
    print(json.dumps(main(p.parse_args()),sort_keys=True))
