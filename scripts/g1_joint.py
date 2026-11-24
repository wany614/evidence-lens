"""Freeze all T1 terminal slots and grant one joint independent reference opening."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from dtu_score import METHODS, SCOPES, VIEWS
from evidence_lens._bundle import load, save, sha, utc
from evidence_lens._verify import require
from evidence_lens.multisource_io import Case
from g1_fixed_score import fixed_case, selected, verify_scores
from general_records import bind, resolve, stage
from t1_access import CONDITIONS, FIXED, H, OBJECTS, S0, SLOTS, STRATEGIES


def exposure(store):
    rows=[]
    for root in (store/"runs",store/"quarantine"):
        for p in root.glob("*/payload/access.json"):
            row=load(p)
            if row.get("object_id") in OBJECTS:
                require(row["observed_reference_count"] == 0 and row["logical_condition"] in CONDITIONS,
                        "prior T1 reference access before joint seal")
                native_run=load(p.parent.parent/"run.json")
                rows.append({"run_id":native_run["run_id"],"started_at_utc":native_run["started_at_utc"],
                             "object_id":row["object_id"],
                             "condition":row["logical_condition"],"sha256":sha(p)})
        require(not any(root.glob("*/payload/references.json")), "a reference acquisition already exists")
    return sorted(rows,key=lambda r:r["run_id"])


def source_identity(src):
    slots=src["source_ids"]
    return (src["object_id"],src["condition"],src["source_camera_sha256"],src["render_camera_sha256"],
            [(src["sources"][s]["official_view_id"],src["sources"][s]["original_sha256"],
              src["sources"][s]["rgb"]["sha256"]) for s in slots])


def matrix(store, cfg, out):
    roots=(store,store/"history")
    pins=cfg["sources"]
    g=load(resolve(roots,pins["entry"]))
    f=load(resolve(roots,g["freeze"]))
    t=load(resolve(roots,pins["terminals"]))
    cells=t["cells"]
    require(isinstance(cells,list) and len(cells)==42 and
            [(c["object_id"],c["condition"]) for c in cells]==[(o,b) for o in OBJECTS for b in CONDITIONS],
            "all 42 terminals required in the registered order")
    require(set(f["layout_profiles"])=={"M2","M4"} and all(v["tau"] is None and v["status"]=="no_feasible_operating_point"
                for v in f["layout_profiles"].values()),"C6 profile changed")
    roster={ (c["object_id"],c["condition"]):c for c in cells}
    fixed=[]; indices=[]; observations=[]; outcomes=[]; sealed_at={}
    for obj in OBJECTS:
        dp=pins["decisions-"+obj]
        choice=load(resolve(roots,dp))
        require(choice["object_id"]==obj and set(choice["results"])==set(STRATEGIES)
                and set(choice["selections"])==set(STRATEGIES), "missing five decisions")
        completed = choice.get("schema") == "el-b2-joint-nbv-choices-1"
        if completed:
            require(all(isinstance(v,list) and len(v)==2 and len(set(v))==2 for v in choice["results"].values()),
                    "unproven camera pair")
        else:
            require(choice.get("schema")=="el-t1-probe-terminal-1" and choice.get("status")=="failed"
                    and choice.get("completed_pair_decisions")==0 and all(v is None for v in choice["results"].values()),
                    "missing selections have no registered frozen-probe failure")
        stopped = store/"runs"/dp["run_id"]
        sealed = load(stopped/"completion.json")["sealed_at_utc"]
        sealed_at[obj]=sealed
        for strategy in STRATEGIES:
            require(sha(stopped/"payload"/(strategy+".json"))==choice["selections"][strategy],
                    "strategy terminal file differs from the joint register")
        for name in CONDITIONS:
            c=roster[obj,name]; is_fixed=name.endswith("-fixed")
            alias=c.get("alias_of")
            if not completed and name!="M2-fixed":
                require(c["status"]=="failed" and c.get("reason") and c["disposition"] and
                        all(c.get(k) is None for k in ("source","initialization","ticket","readback","case","alias_of")),
                        "unopened branch carries unauthorized evidence")
                resolve(roots,c["disposition"])
                if not is_fixed:
                    strategy=name[3:]
                    observations.append({"object_id":obj,"strategy":strategy,"status":"failed","selected":None,
                                         "decision":bind(store,dp["run_id"],"payload/"+strategy+".json"),
                                         "observation":None,"reason":c["reason"]})
            else:
                src=load(resolve(roots,c["source"]))
                slots=SLOTS[name[:2]]
                order=S0 if name=="M2-fixed" else FIXED if is_fixed else [24,*choice["results"][name[3:]],22]
                require(src["object_id"]==obj and src["logical_condition"]==name and src["entry"]==pins["entry"]
                        and src["source_ids"]==slots and [src["sources"][v]["official_view_id"] for v in slots]==order
                        and src["reference_photo_reads"]==0 and src["other_branch_candidate_observations"]==0,
                        "incorrect branch exposure or ordered source")
                if name!="M2-fixed":
                    observed=load(store/"runs"/c["source"]["run_id"]/"run.json")["started_at_utc"]
                    require(src["decisions"]==dp and observed>=sealed,"candidate acquisition preceded five sealed choices")
                if not is_fixed:
                    own=src["own_decision"]
                    require(own["run_id"]==dp["run_id"] and own["sha256"]==choice["selections"][name[3:]],
                            "different trajectory authorization")
                    observations.append({"object_id":obj,"strategy":name[3:],"status":"observed","selected":order[1:3],
                                         "decision":own,"observation":c["source"]})
                if alias:
                    original=roster[obj,alias]
                    require(not original.get("alias_of") and
                            all(c[k]==original[k] for k in ("case","readback","ticket","initialization"))
                            and source_identity(src)==source_identity(load(resolve(roots,original["source"]))),
                            "alias is not the same ordered immutable computation")
            if c["status"]=="succeeded":
                rb=load(resolve(roots,c["readback"]))
                require(rb["object_id"]==obj and rb["logical_condition"]==(alias or name)
                        and rb["status"]=="succeeded" and rb["verified"] and rb["case_sha256"]==c["case"]["sha256"]
                        and rb["initialization"]==c["initialization"] and
                        rb["origins"]["dispatch"]=={k:c["ticket"][k] for k in ("run_id","manifest_sha256")}
                        and rb["completed_steps"]==5000 and rb["recipe_sha256"]==g["recipe_sha256"]
                        and rb["target_reference_files_opened"]==0,"unverified T1 immutable endpoint")
                native=Case(resolve(roots,c["case"]).parent,c["case"]["sha256"])
                for view in VIEWS:
                    if view not in native.failed:
                        require(native.frame(view) is not None,"target prediction is absent")
            else:
                require(c["status"] in ("failed","not_executed") and c.get("reason")
                        and (c.get("disposition") or c.get("readback")),"unproven terminal failure")
                resolve(roots,c.get("disposition") or c["readback"])
            outcomes.append(c)
            if not is_fixed:
                continue
            index={"object_id":obj,"condition":name[:2],"status":c["status"],"case":c.get("case"),
                   "readback":c.get("readback"),"reason":c.get("reason"),"scores":None}
            if c["status"]=="succeeded":
                _,cp,case=fixed_case(roots,obj,name[:2],c["readback"],g,f)
                require(cp==c["case"],"different case in score register")
                score=pins["score-"+obj+"-"+name]
                rows=verify_scores(roots,score,case,f["released_models"])
                index["scores"]=score
            else:
                rows=[{"view":v,"method":m,"status":c["status"],"reason":c["reason"],"scope":SCOPES[m],
                       "score_file":None} for v in VIEWS for m in METHODS]
            fixed.extend({"object_id":obj,"condition":name[:2],**r} for r in rows)
            indices.append(index)
    require(len(fixed)==288 and len(observations)==30 and len(indices)==12,"incomplete independent matrix")
    access = exposure(store)
    blocked_objects = {v["object_id"] for v in observations if v["status"]=="failed"}
    require(all(r["condition"]=="M2-fixed" for r in access if r["object_id"] in blocked_objects),
            "photographic receipt for an unauthorized trajectory")
    require(all(r["condition"]=="M2-fixed" or r["started_at_utc"]>=sealed_at[r["object_id"]] for r in access),
            "a photographic acquisition preceded the five sealed choices")
    save(out/"sealed-matrix.json",{"schema":"el-t1-closed-matrix-1","run_id":os.environ["EL_RUN_ID"],
        "utc":utc(),"entry":pins["entry"],"freeze":g["freeze"],"terminals":pins["terminals"],"cells":outcomes,
        "objects":OBJECTS,"observations":observations,"fixed_cells":indices,"fixed_score_terminals":fixed,
        "fixed_score_statuses":dict(Counter(r["status"] for r in fixed)),"prior_acquisitions":access,
        "strategy_statuses":dict(Counter(o["status"] for o in observations)),
        "independent_reference_reads":0,"additional_selected_descriptive_scores":0})


def open_once(store,cfg,out):
    pins=cfg["sources"]
    roots=(store,store/"history")
    book=load(resolve(roots,pins["matrix"]))
    require(book["schema"]=="el-t1-closed-matrix-1" and book["entry"]==pins["entry"]
            and book["objects"]==OBJECTS and len(book["cells"])==42 and len(book["fixed_score_terminals"])==288
            and len(book["observations"])==30 and book["prior_acquisitions"]==exposure(store),
            "unexpected matrix or photographic access after terminal seal")
    lock=load(resolve(roots,pins["evaluation_program"]))
    project=Path(__file__).resolve().parents[1]
    require(all(sha(project/n)==digest for n,digest in lock["scientific_evaluation_files"].items()),
            "evaluation or opening software differs from its pre-opening seal")
    save(out/"opening.json",{"schema":"el-t1-joint-opening-1","utc":utc(),"run_id":os.environ["EL_RUN_ID"],
        "entry":pins["entry"],"matrix":pins["matrix"],"evaluation_program":pins["evaluation_program"],
        "object_ids":OBJECTS,"target_ids":list(VIEWS),"official_targets":H,"condition_count":42,
        "fixed_score_count":288,"fixed_statuses":book["fixed_score_statuses"],"predictions_and_scores_sealed":True,
        "permitted_reader":"independent_target_evaluator","before_opening_reference_reads":0,
        "authorized_operation":"One collective photographic opening over the six registered objects and target views.",
        "new_risk_training":0,"threshold_refits":0})


def main(a):
    s=a.store; h=s/"history"
    if a.action.endswith("-worker"):
        cfg=load(a.config)
        return (matrix if a.action=="matrix-worker" else open_once)(s,cfg,a.out)
    entry=selected(s,"g0-entry")
    if a.action=="matrix":
        terminals=selected(s,"t1-terminals")
        pins={"entry":entry,"terminals":terminals}
        for obj in OBJECTS:
            pins["decisions-"+obj]=selected(s,"t1-decisions-"+obj)
        book=load(resolve((s,h),terminals))
        for c in book["cells"]:
            if c["condition"].endswith("-fixed") and c["status"]=="succeeded":
                k=c["object_id"]+"-"+c["condition"]
                pins["score-"+k]=selected(s,"t1-score-"+k)
        name,member="t1-closed-matrix","sealed-matrix.json"
    else:
        pins={"entry":entry,"matrix":selected(s,"t1-closed-matrix"),
              "evaluation_program":selected(s,"t1-evaluation-program")}
        name,member="t1-opening","opening.json"
    return stage(s,h,name,"archive_verify",pins,{"phase":a.action,"reference_read_in_this_worker":False},
                 ["{python}","{source}/scripts/g1_joint.py",a.action+"-worker","--store",str(s),
                  "--config","{config}","--out","{out}"],"payload/"+member,1500000)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["matrix","open","matrix-worker","open-worker"])
    p.add_argument("--store",type=Path,required=True)
    for k in ("out","config"):
        p.add_argument("--"+k,type=Path)
    print(json.dumps(main(p.parse_args()),sort_keys=True))
