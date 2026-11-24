"""Lock the C6 empirical method, commit label-free candidates and fit only after its collective opening."""
import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import os
import platform
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

import numpy as np
from b2_calibration import ANCHORS,ObjectData,candidates,fit,leave_one_out
from b2_intake_photos import valid_opening
from b2_intake_record import locate
from dtu_eval import photo,score_map
from dtu_eval_metrics import labels,observation
from dtu_score import METHODS,RELEASE_SHA,SCOPES,VIEWS
from evidence_lens.multisource_io import Case
from t0_dtu_archive import check,digest,write_json

RECIPE="57730c706aaaa2f88c7f39ff179ba04ff971f31e599bf3409a3d545c61d72c98"
ROOT=Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(p.read_bytes())


def checked(stores,pin):
    return read(locate(stores,pin))


def table(path,rows):
    if not rows:raise ValueError("empty planned output")
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader()
        for row in rows:
            w.writerow({k:(";".join(v) if isinstance(v,list) else v) for k,v in row.items()})


def make_lock(cfg,out):
    pins=cfg["pinned"];roster=checked(cfg["roots"],pins["cohort"]);release=checked(cfg["roots"],pins["release"])
    old=checked(cfg["roots"],pins["prior_freeze"]);validation=checked(cfg["roots"],pins["validation"])
    check(release["models"]==old["models"] and old["release_index_sha256"]==pins["release"]["sha256"]==RELEASE_SHA and
          old["recipe_sha256"]==RECIPE and old["stop_step"]==5000 and validation["passed"] and validation["C6_target_reads"]==0,"unverified deployment or validation identity")
    recon=checked(cfg["roots"],pins["reconstruction"])
    check(recon["recipe_sha256"]==RECIPE and recon["optimizer_seed"]==43171 and len(recon["worker_tree_sha256"])==64,"wrong reconstruction release")
    actual={p:digest((ROOT/p).read_bytes()) for p in cfg["scientific_files"]}
    check(actual==cfg["scientific_files"],"method software changed before lock")
    scope={"layout":"DTU-fixed","primary_source_id":"s00","lighting_id":3,"resolution":[256,256],"rgb_encoding":"display_srgb_0_1",
           "depth_semantics":"expected_camera_z_m","alpha_semantics":"accumulated_opacity","recipe_sha256":RECIPE,"stop_step":5000,
           "reconstructor":{"name":"r2-scale-decay-02","worker_tree_sha256":recon["worker_tree_sha256"],"gaussian_cap":20000,"optimizer_seed":43171},
           "renderer":{"name":"gsplat","version":"1.5.3"},"release_index_sha256":RELEASE_SHA,
           "model_sha256":release["models"]["MS-L1"]["weights_sha256"],"model_index_sha256":release["models"]["MS-L1"]["model_index_sha256"],
           "feature_schema_sha256":release["feature_schema_sha256"],"preprocessing_sha256":release["files"]["models/MS-L1/preprocessing.json"]["sha256"],
           "feature_columns":39}
    artifact={"schema":"el-b2-calibration-method-1","run_id":os.environ["EL_RUN_ID"],"cohort":pins["cohort"],"scientific_files":actual,
              "source_validation":pins["validation"],"prior_release":pins["release"],"prior_freeze":pins["prior_freeze"],"reconstruction":pins["reconstruction"],"target_rgb_reads":0,
              "science_roles":{"only_calibration":"C6","T1":"unopened","previous_T0":"retains its original frozen confirmation role"},
              "scope":scope,"objects":roster["splits"]["C6"],"conditions":["M2","M4"],"models":release["models"],"method":"MS-L1",
              "candidates":{"count":201,"q":"j/200; j=0..200","sample":"Finite deployment MS-L1 scores at six targets; pool each object's native-grid scores across its six views; no references, E or Y used.",
               "weighting":"Each nonempty object contributes equal total probability, each of its scores weight 1/N_g; calculate inverse CDF by exact integer comparisons.",
               "ties":"j=0 uses smallest observed score as strict empty cutoff. j>0 chooses smallest FP32 value with inclusive mixture mass >=j/200; tau=nextafter(float64(value),+infinity). All equal observations are classified together. Keep all repeated anchors and number of distinct thresholds.",
               "empty":"Generate using available finite-score objects, record number contributing. If all are empty retain 201 null anchors. All planned objects stay in the fit feasibility test."},
              "target":{"maximum_mean_low_error":.05,"minimum_mean_coverage":.20,"require_every_object_nonempty":True},
              "definitions":{"epsilon":.05,"label":"strict mean absolute display RGB FP32 error > 0.05 on evaluable RGB",
                             "V":"valid registered reference pixels, whole crop","E":"V and valid RGB prediction","S":"E and finite method score",
                             "L":"S and float64(score) < tau","LY":"L with error label 1","U_E":"E minus S","missing_render":"known V with E=0 and no observed error labels"},
              "selection":"All planned C6 objects require L>0. Compare exact rational mean across objects: LY/L <= 1/20, L/V >= 1/5. Among feasible candidates maximize mean coverage, minimize mean low-error, then minimum tau, then lowest repeated anchor. Inspect full nonmonotone grid. Empty L gives null low-error.",
              "holdout":"Six leave-one-object-out fits per condition regenerate all candidates from the other five deployment distributions, select against their Y, then evaluate the held-out object once if the training fold returns a threshold. No feasible point remains a reported outcome.",
              "calibration_scope":"Only empirical, fixed frozen method, same native source/target layout, camera, light, renderer and optimizer. Source-dependent NBV/other layouts outside scope.",
              "environment":{"python":platform.python_version(),"numpy":np.__version__}}
    write_json(out/"calibration-method.json",artifact)


@dataclass
class Deploy:
    object_id:str
    deployment:np.ndarray


def inspect_method(stores,pins):
    method=checked(stores,pins["method"])
    check(method["schema"]=="el-b2-calibration-method-1" and method["cohort"]==pins["cohort"] and
          method["target_rgb_reads"]==0 and all(digest((ROOT/p).read_bytes())==h for p,h in method["scientific_files"].items()),
          "unbound or altered C6 fitting implementation")
    return method


def cells_of(matrix):
    if "terminal_cells" in matrix:return matrix["terminal_cells"]
    rows=[]
    for name,row in matrix["cells"].items():
        obj,cond=name.rsplit("-",1)
        check(row.get("object_id",obj)==obj and row.get("condition",cond)==cond,"conflicting cell identity")
        rows.append({"object_id":obj,"condition":cond,**row})
    return rows


def fetch_evidence(stores,method,cohort,pins,cells,references=None,single=None):
    by={(x["object_id"],x["condition"]):x for x in cells}
    ids=cohort["splits"]["C6"]
    expected={(i,c) for i in ids for c in ("M2","M4")}
    if single:
        check(single in expected and references is None,"only registered source-side cells may be monitored")
        expected={single}
    check(len(by)==len(cells)==len(expected) and set(by)==expected,"wrong C6 result matrix")
    original_tau={m:method["models"][m]["threshold"] for m in METHODS}
    groups={"M2":[],"M4":[]};views=[]; original=[];object_metrics=[]
    for oid in ids:
        truth={}
        if references is not None:
            meta=checked(stores,references[oid]); refroot=locate(stores,references[oid]).parent
            check(meta["schema"]=="el-b2-dtu-references-1" and meta["object_id"]==oid and meta["opening"]==pins["opening"] and
                  meta["cohort"]==pins["cohort"] and meta["calibration_method"]==pins["method"] and set(meta["views"])==set(VIEWS),
                  "wrong independently opened target truth")
            registration,=[r for r in cohort["objects"] if r["object_id"]==oid]
            for v,row in meta["views"].items():
                check(row["official_view_id"]==cohort["target_view_ids"][v] and row["archive_member"]==registration["target_members"][v]["name"] and
                      row["valid_mask"]=={"kind":"all_true","width":256,"height":256},"target identity, calibration or validity differs")
                truth[v]=photo(refroot,row["rgb"])
        for condition in groups:
            if (oid,condition) not in by:continue
            cell=by[oid,condition]; case=None;sc=None;sr=None
            if cell["status"]!="scored":
                check(cell["status"] in ("failed","not_executed") and cell["reason"],"unknown scientific terminal")
                disp=checked(stores,cell["disposition"])
                check(disp["object_id"]==oid and disp["condition"]==condition and disp["status"]==cell["status"],"incorrect terminal disposition")
            if cell.get("case") and cell.get("readback"):
                cp=locate(stores,cell["case"]);case=Case(cp.parent,cell["case"]["sha256"]);rb=checked(stores,cell["readback"])
                check(rb["object_id"]==oid and rb["condition"]==case.condition==condition and rb["case_sha256"]==case.sha256 and
                      rb["verified"] is True and rb["recipe_sha256"]==RECIPE and rb["completed_steps"]==5000 and
                      rb["cohort"]==pins["cohort"] and rb["freeze"]==method["reconstruction"] and rb["layout"]==condition+"-fixed" and rb["optimizer_seed"]==43171 and cell["case"]["run_id"]==cell["readback"]["run_id"] and rb["target_reference_files_opened"]==0,
                      "case does not belong to registered C6 fixed reconstruction")
                source=checked(stores,rb["source"])
                check(source["schema"]=="el-b2-dtu-source-1" and source["object_id"]==oid and source["condition"]==condition
                      and source["cohort"]==pins["cohort"] and source["source_ids"]==list(case.generator)
                      and all(case.meta["source_images"][s]["sha256"]==source["sources"][s]["rgb"]["sha256"] for s in case.generator),
                      "original reconstruction source is not the approved fixed C6 capture")
                check(case.meta["source_cameras"]["sha256"]==cohort["camera"]["conditions"][condition]["source_cameras"]["sha256"] and
                      case.meta["render_cameras"]["sha256"]==cohort["camera"]["conditions"][condition]["render_cameras"]["sha256"],"non-fixed calibration")
            if cell["status"]=="scored":
                check(case is not None,"scored cell lacks original case")
                sp=locate(stores,cell["scores"]);sc=read(sp);sr=sp.parent
                check(sc["schema"]=="el-dtu-frozen-scores-1" and sc["case_sha256"]==case.sha256 and sc["condition"]==condition
                      and sc["checkpoint_sha256"]==case.meta["checkpoint_sha256"] and sc["reconstruction_run_id"]==case.meta["reconstruction_run_id"]
                      and sc["release_sha256"]==RELEASE_SHA and sc["models"]==method["models"] and sc["method_scopes"]==SCOPES and set(sc["views"])==set(VIEWS),
                      "unverified archived score, checkpoint or release")
            raw=[];labelled=[];counts={m:[] for m in METHODS}
            for v in VIEWS:
                ready=sc is not None and v not in case.failed
                if ready:check(sc["views"][v]["status"]=="ready" and set(sc["views"][v]["methods"])==set(METHODS),"missing scored method")
                elif sc is not None:check(sc["views"][v]["status"]=="render_failed" and v in case.failed,"unexpected missing score")
                scores={m:score_map(sr,sc,m,v,case,original_tau[m]) if ready else np.full((256,256),np.nan,np.float32) for m in METHODS}
                risk=scores["MS-L1"]
                raw.append(risk[np.isfinite(risk)])
                views.append({"object_id":oid,"condition":condition,"view":v,"terminal_status":cell["status"],"scored":ready,
                              "deployment_finite":int(np.isfinite(risk).sum()),"risk_float32_sha256":digest(np.asarray(risk,dtype="<f4",order="C").tobytes()),
                              "score_files":{m:sc["views"][v]["methods"][m]["file"]["sha256"] for m in METHODS} if ready else {},
                              "case":cell.get("case"),"scores":cell.get("scores")})
                if references is not None:
                    frame=case.frame(v) if case is not None else None
                    vv,ee,yy,err,aux=labels(truth[v],frame["rgb"] if frame else None,np.ones((256,256),bool))
                    labelled.append((risk,vv,ee,yy))
                    obs,_=observation(vv,ee,yy,err,aux,scores,original_tau)
                    for m in METHODS:
                        x=obs[m]["counts"];own=obs[m]["own"];common=obs[m]["common"]
                        record={"object_id":oid,"condition":condition,"view":v,"method":m,"threshold":original_tau[m],
                                **{k:x[k] for k in ("V","E","Y","S","SY","C","CY","L","LY","H","HY","U","UY","V_minus_E")},
                                "own_AP":own.get("ap"),"common_AP":common.get("ap"),"own_prevalence":obs[m]["rates"]["own_prevalence"]["value"],"common_prevalence":obs[m]["rates"]["common_prevalence"]["value"]}
                        original.append(record);counts[m].append(record)
            if references is None:
                groups[condition].append(Deploy(oid,np.sort(np.concatenate(raw).astype(np.float64),kind="stable")))
            else:
                groups[condition].append(ObjectData.from_views(oid,labelled))
                for m,rr in counts.items():
                    sums={k:sum(t[k] for t in rr) for k in ("V","E","Y","S","L","LY","H","HY","U","V_minus_E")}
                    object_metrics.append({"object_id":oid,"condition":condition,"method":m,**sums,"coverage":sums["L"]/sums["V"] if sums["V"] else None,
                                           "low_error":sums["LY"]/sums["L"] if sums["L"] else None})
    return groups,views,original,object_metrics


def candidate_map(groups):
    result={}
    for c,objects in groups.items():
        result[c]={"full":candidates(objects)}
        for left in objects:result[c]["held-out-"+left.object_id]=candidates([o for o in objects if o.object_id!=left.object_id])
    return result


def perform(a,cfg):
    stores=cfg["roots"];pins=cfg["pinned"];roster=checked(stores,pins["cohort"]);method=inspect_method(stores,pins)
    if a.action=="fingerprint":
        obj,condition=cfg["object_id"],cfg["condition"]
        groups,views,_,_=fetch_evidence(stores,method,roster,pins,[{"object_id":obj,"condition":condition,**cfg["cell"]}],single=(obj,condition))
        values=groups[condition][0].deployment.astype("<f4")
        write_json(a.out/"fingerprint.json",{"schema":"el-b2-risk-fingerprint-1","run_id":os.environ["EL_RUN_ID"],
                   "cohort":pins["cohort"],"calibration_method":pins["method"],"object_id":obj,"condition":condition,"views":views,
                   "cell":cfg["cell"],"sorted_deployment_float32_sha256":digest(values.tobytes()),"finite_scores":values.size,
                   "C6_reference_reads":0,"T1_photo_reads":0})
        return
    if a.action=="candidates":
        matrix=checked(stores,pins["matrix"])
        groups,views,_,_=fetch_evidence(stores,method,roster,pins,cells_of(matrix))
        reports={};index={(o.object_id,c):o for c,rows in groups.items() for o in rows}
        for (obj,condition),o in index.items():
            key=obj+"-"+condition;fp=pins["fp"+key];r=checked(stores,fp);subset=[v for v in views if (v["object_id"],v["condition"])==(obj,condition)]
            check(r["views"]==subset and r["sorted_deployment_float32_sha256"]==digest(o.deployment.astype("<f4").tobytes()) and
                  r["finite_scores"]==len(o.deployment) and r["cohort"]==pins["cohort"] and r["calibration_method"]==pins["method"],
                  "original deployment differs from its individually archived pre-label evidence")
            reports[key]=fp
        artifact={"schema":"el-b2-score-candidates-1","fingerprints":reports,"run_id":os.environ["EL_RUN_ID"],"cohort":pins["cohort"],"calibration_method":pins["method"],
                  "matrix":pins["matrix"],"candidate_thresholds":candidate_map(groups),"views":views,"C6_reference_reads":0,"T1_photo_reads":0}
        write_json(a.out/"score-candidates.json",artifact);return
    op=checked(stores,pins["opening"]);valid_opening(roster,pins["cohort"],pins["method"],op)
    score_only=checked(stores,pins["candidates"])
    check(op["score_candidates"]==pins["candidates"] and score_only["cohort"]==pins["cohort"] and score_only["calibration_method"]==pins["method"]
          and score_only["C6_reference_reads"]==0,"candidate commitment differs from opening")
    refpins={o:pins["ref"+o] for o in roster["splits"]["C6"]}
    data,views,orig,objstats=fetch_evidence(stores,method,roster,pins,cells_of(op),refpins)
    check(views==score_only["views"],"scored views differ from zero-label commitment")
    curves=[];perobj=[];foldrows=[];profiles={};summaries={}
    (a.out/"profiles").mkdir()
    for c,objects in data.items():
        full=fit(objects);loo=leave_one_out(objects)
        fits=[("full",full)]+[("held-out-"+x["held_out"],x["training"]) for x in loo]
        for fold,f in fits:
            check([r["tau"] for r in f["curves"]]==score_only["candidate_thresholds"][c][fold],"candidate set changed after labels")
            curves.extend({"condition":c,"fold":fold,**r} for r in f["curves"]);perobj.extend({"condition":c,"fold":fold,**r} for r in f["counts"])
        for x in loo:foldrows.append({"condition":c,"held_out":x["held_out"],"training_ids":x["training_ids"],"fit_status":x["training"]["status"],
            "selected_anchor":x["training"]["selected_anchor"],"tau":x["training"]["tau"],"held_out_status":x["held_out_status"],**x["held_out_result"]})
        calibrated=full["selected"]
        profile={"schema":"el-b2-calibration-profile-1","method":"MS-L1","condition":c,"profile_version":1,"status":full["status"],"tau":full["tau"],
                 "target":method["target"],"comparison":"score < tau","original_tau":method["models"]["MS-L1"]["threshold"],"epsilon":.05,
                 "scope":{**method["scope"],"source_ids":["s00","s03"] if c=="M2" else ["s00","s01","s02","s03"],
                          "source_camera_sha256":roster["camera"]["conditions"][c]["source_cameras"]["sha256"],
                          "render_camera_sha256":roster["camera"]["conditions"][c]["render_cameras"]["sha256"]},
                 "selected_anchor":full["selected_anchor"],"selected_estimates":calibrated,"candidate_anchors":201,"distinct_thresholds":full["distinct_thresholds"],
                 "evaluated_on_holdout":False,"holdout_target_met":None,"calibration_object_ids":[o.object_id for o in objects],
                 "lineage":{"fit_run_id":os.environ["EL_RUN_ID"],"cohort":pins["cohort"],"algorithm":pins["method"],"score_candidates":pins["candidates"],
                            "opening":pins["opening"],"references":refpins},"C6_selected_threshold_counts":[o.count(full["tau"]) for o in objects],
                 "epsilon_definition":method["definitions"]["label"],"raw_counts_file":"../candidate-object-counts.csv",
                 "remaining_scope":"Empirical calibration on C6 only. Fixed M4 does not establish suitability for M4-selected or imported assets."}
        dest=a.out/"profiles"/(c+".json");write_json(dest,profile);profiles[c]={"path":"profiles/"+c+".json","sha256":digest(dest.read_bytes())}
        sufficient=[r for r in full["curves"] if r["nonempty_objects"]==len(objects) and r["mean_coverage"] is not None and r["mean_coverage"]>=.2]
        lowerr=[r for r in full["curves"] if r["nonempty_objects"]==len(objects) and r["mean_low_error"] is not None and r["mean_low_error"]<=.05]
        summaries[c]={"status":full["status"],"tau":full["tau"],"distinct_thresholds":full["distinct_thresholds"],"feasible_anchors":sum(r["feasible"] for r in full["curves"]),
                      "minimum_error_among_all_nonempty_coverage_at_least_20_percent":min((r["mean_low_error"] for r in sufficient),default=None),
                      "highest_coverage_among_all_nonempty_error_at_most_5_percent":max((r["mean_coverage"] for r in lowerr),default=None),
                      "loo_feasible_folds":sum(r["training"]["tau"] is not None for r in loo),"loo_folds":len(loo)}
    for fn,rows in (("candidate-curves.csv",curves),("candidate-object-counts.csv",perobj),("leave-one-out.csv",foldrows),
                    ("original-method-views.csv",orig),("original-method-object-counts.csv",objstats)):table(a.out/fn,rows)
    write_json(a.out/"calibration-results.json",{"schema":"el-b2-calibration-results-1","run_id":os.environ["EL_RUN_ID"],"cohort":pins["cohort"],
               "calibration_method":pins["method"],"opening":pins["opening"],"score_candidates":pins["candidates"],"profiles":profiles,
               "outcomes":summaries,"raw_scored_views_methods":len(orig),"views":len(views),"fitted_conditions":2,"leave_one_object_out_fits":12,
               "candidate_rows":len(curves),"per_object_candidate_rows":len(perobj),"references":refpins,"T1_access":0,"new_risk_models":0,
               "numerical_files":{p.name:digest(p.read_bytes()) for p in sorted(a.out.glob("*.csv"))}})


def close_matrix(cfg, out):
    matrix=cfg["matrix_snapshot"]
    check(matrix["schema"]=="el-b2-c6-closed-matrix-1" and matrix["cohort"]==cfg["pinned"]["cohort"],
          "wrong calibration score-only matrix")
    original=bytes.fromhex(cfg["matrix_bytes_hex"])
    check(read_document(original)==matrix and digest(original)==cfg["matrix_sha256"],"changed root matrix")
    (out/"closed-matrix.json").write_bytes(original)


def read_document(raw):
    return json.loads(raw)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("lock","matrix","fingerprint","candidates","fit"))
    p.add_argument("--config",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();cfg=read(a.config);cfg["roots"]=[Path(r) for r in cfg["read_only_roots"]]
    if a.action=="lock":make_lock(cfg,a.out)
    elif a.action=="matrix":close_matrix(cfg,a.out)
    else:perform(a,cfg)
