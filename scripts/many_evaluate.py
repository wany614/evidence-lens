"""Evaluate one sealed multi-source checkpoint against an independently authorized reference package."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from many_head import TRAIN, check, digest, load, save
from many_matrix import checked_file, prediction_hash
from many_data_access import EvaluatorTruth


DEV=("b04-layered-louvers","delivery_area")
CONFIRM=("b05-perforated-gallery","b06-corner-alcoves","living_room","bridge")
RULES=("coverage","alpha_missing","residual")


def verify_role(scores,truth,opening):
    family=scores["family"]
    role="train" if family in TRAIN else "development" if family in DEV else "confirmation" if family in CONFIRM else None
    check(role is not None and scores["role"]==role and scores["references_accessed"]==0, "data role differs")
    check(truth["schema"]=="el-many-evaluator-truth-1" and truth["scene_id"]==family and truth["split"]==role
          and set(truth["targets"])==set(scores["target_ids"]),
          "different reference cameras")
    if role=="confirmation":
        check(opening is not None and opening["phase"]=="confirmation" and opening["schema"]=="el-many-h-opening-1"
              and set(opening["allowed_targets"])==set(CONFIRM) and
              list(opening["allowed_targets"][family])==list(scores["target_ids"]), "confirmation access missing")
        row=opening["terminals"][family][str(scores["R"])]
        check(row["status"]=="succeeded" and row["scores_run"]==scores["run_id"],
              "scores had no terminal receipt before opening H")


def open_results(folder,view):
    ranks={}; geometry={}
    for cell,rec in view["cells"].items():
        f=checked_file(folder,{"path":rec["file"],"sha256":rec["sha256"]})
        with np.load(f,allow_pickle=False) as a:
            ranks[cell]={n[5:]:a[n] for n in a.files if n.startswith("rank_")}
            geometry[cell]={n:a[n] for n in a.files if not n.startswith("rank_")}
        check(len(rec["queries"])==len(set(rec["queries"]))==rec["actual_k"],"Q contains a duplicate")
    return ranks,geometry


def ratio(a,b):
    return a/b if b else None


def scope_metrics(small,full,valid,high):
    correct=valid & ~high
    missing=small["incomplete"] | full["incomplete"]
    invalid=small["invalid_geometry"] | full["invalid_geometry"]
    comparable=full["comparable"]>0
    gap=~missing & ~invalid & (small["comparable"]==0) & comparable
    absent=~missing & ~invalid & ~comparable
    n=int(correct.sum()); g=int((correct&gap).sum())
    check(np.all(full["witness"][gap]>=0),"audited gap is missing its actual witness")
    audited=not np.any(full["incomplete"])
    query_missing=int((correct&small["incomplete"]).sum())
    query_invalid_only=int((correct&small["invalid_geometry"]&~full["invalid_geometry"]).sum())
    issue=("full_bank_incomplete" if not audited else "query_incomplete" if query_missing else
           "query_only_invalid_geometry" if query_invalid_only else "no_correct_labels" if not n else None)
    return dict(reference_labelled=int(valid.sum()),all_reference_correct=n,correct_with_gap=g,
        correct_gap_rate=ratio(g,n) if issue is None else None,audit_complete=audited,
        correct_query_incomplete=query_missing,correct_query_only_invalid_geometry=query_invalid_only,undefined=issue,
        selected_gap=int(gap.sum()),labelled_selected_gap=int((gap&valid).sum()),
        library_no_comparable=int(absent.sum()),library_incomplete=int((missing&~invalid).sum()),
        invalid_geometry=int(invalid.sum()),query_has_comparable=int((small["comparable"]>0).sum()),
        query_contains_supported=int((small["support"]>0).sum()),
        depth_conflicts=int((small["depth_conflict"]>0).sum()),colour_conflicts=int((small["colour_conflict"]>0).sum()),
        gap_witness_sources={str(int(i)):int(n) for i,n in zip(*np.unique(full["witness"][gap],return_counts=True))})


def visibility(bank,archive,cache,truth,view,cells,geometries):
    if not truth.get("z"):
        return None
    zt=archive.optical_z(view)
    tgt=bank.cameras[view]
    y,x=np.indices(zt.shape,dtype=np.float64)
    cam=np.stack(((x-tgt.K[0,2])/tgt.K[0,0],(y-tgt.K[1,2])/tgt.K[1,1],np.ones_like(x)),axis=-1)
    good=np.isfinite(zt)&(zt>0)
    pts=(cam*np.where(good,zt,0)[...,None])@tgt.c2w[:3,:3].T+tgt.c2w[:3,3]
    count={n:np.zeros(zt.shape,np.uint16) for n in cells}
    for sid in bank.ids:
        use=[n for n,c in cells.items() if sid in c["queries"]]
        if not use: continue
        source=bank.cameras[sid]
        if sid not in cache:
            cache[sid]=archive.optical_z(sid)
        z=cache[sid]
        local=(pts-source.c2w[:3,3])@source.c2w[:3,:3]
        dz=local[...,2]
        forward=good&(dz>0)
        norm=np.where(forward,dz,1)
        u=np.floor(source.K[0,0]*local[...,0]/norm+source.K[0,2]+.5).astype(np.int32)
        v=np.floor(source.K[1,1]*local[...,1]/norm+source.K[1,2]+.5).astype(np.int32)
        inside=forward&(u>=0)&(u<378)&(v>=0)&(v<252)
        measured=z[np.clip(v,0,251),np.clip(u,0,377)]
        # Nearest true source sample has a finite footprint; relative 1% plus 1 mm for authored metre-scale scenes.
        visible=inside & np.isfinite(measured) & (measured>0) & (np.abs(measured-dz)<=.001+.01*np.abs(dz))
        for key in use: count[key]+=visible
    report={}
    for key,val in count.items():
        tv=val>0; pred=geometries[key]["comparable"]>0
        report[key]=dict(independent_target_surface=int(good.sum()),true_visible=int((good&tv).sum()),
            model_comparable=int((good&pred).sum()),both=int((good&tv&pred).sum()),
            only_model=int((good&~tv&pred).sum()),only_true=int((good&tv&~pred).sum()),
            neither=int((good&~tv&~pred).sum()))
    return report


def exact_visibility(probe, view, cells, geometry):
    result={}
    for name,cell in cells.items():
        d=probe.target(view,cell["queries"])
        x,y=np.moveaxis(d["xy"],-1,0)
        surface=d["foreground"]
        physical=d["visible"]
        comparable=geometry[name]["comparable"][y,x]>0
        counts=dict(probe_pixels=int(surface.size),independent_target_surface=int(surface.sum()),
            in_any_source_frame=int((surface&d["in_frame"]).sum()),true_visible=int((surface&physical).sum()),
            model_comparable=int((surface&comparable).sum()),both=int((surface&physical&comparable).sum()),
            only_model=int((surface&~physical&comparable).sum()),only_true=int((surface&physical&~comparable).sum()),
            neither=int((surface&~physical&~comparable).sum()))
        check(counts["independent_target_surface"]==sum(counts[k] for k in ("both","only_model","only_true","neither")),
              "visibility contingency does not sum to all physical foreground samples")
        result[name]=counts
    return result


def evaluate(a):
    from evidence_lens_many import Bank
    from evidence_lens_many.evaluation import labels,mask_id,opening_priority,ranking_metrics,quota
    cpu=time.process_time();wall=time.perf_counter()
    scores=load(a.scores); truth=load(a.truth)
    opening=load(a.opening) if a.opening else None
    verify_role(scores,truth,opening)
    if opening:
        check(truth["opening_sha256"]==digest(a.opening),"reference authorized by a different opening")
        terminal=opening["terminals"][scores["family"]][str(scores["R"])]
        check(terminal["receipt"]["sha256"]==digest(a.scores) and
              a.freeze and opening["model_freeze"]["sha256"]==digest(a.freeze),"unsealed rank or model opening")
    b=Bank(a.bank,scores["bank_sha256"])
    archive=EvaluatorTruth(a.truth,digest(a.truth),bank=b)
    probe=None
    if a.visibility:
        from many_data_visibility import VisibilityTruth
        probe=VisibilityTruth(a.visibility,digest(a.visibility))
        probe.verify(archive)
    check(bool(probe)==(scores["mode"]=="matrix" and scores["family"].startswith("b0")),
          "incorrect physical-geometry evaluation scope")
    cache={}
    check(list(b.target_ids)==scores["target_ids"] and b.meta["checkpoint_sha256"]==scores["checkpoint_sha256"],
          "different camera or checkpoint")
    if scores["mode"]=="development":
        check(scores["family"] in DEV,"development ranks on a different partition")
    else:
        check(a.freeze is not None,"independent evaluator requires the model freeze")
        frozen=load(a.freeze)
        check(frozen["schema"]=="el-many-confirmation-model-freeze-1" and frozen["strong_simple_rule"]==a.rule
              and a.rule in RULES and scores["models"]["B-CPU-LR"]==frozen["final_model"]["sha256"],
              "different frozen simple rule or model")
    rows={}; comparisons=[]
    for view in scores["target_ids"]:
        rec=scores["views"][view]
        reference=archive.target_rgb(view)
        pred=b.target(view)
        check(rec["prediction"]==prediction_hash(b,view),"target prediction changed between budgets")
        rgb=pred["rgb"] if pred is not None else np.full_like(reference,np.nan)
        ref,v,high,err=labels(rgb,reference)
        entry=dict(status=rec["status"],reference=int(ref.sum()),labelled=int(v.sum()),errors=int(high.sum()),
                   error_rate=ratio(int(high.sum()),int(v.sum())),prediction_unavailable=int((ref&~v).sum()),
                   mae_sum=float(err[v].sum(dtype=np.float64)),prediction_sha256=rec["prediction"],
                   source_reference_sha256=truth["targets"][view]["analysis_rgb"]["sha256"],
                   labels_sha256=mask_id(high),valid_labels_sha256=mask_id(v))
        rows[view]=entry
        if rec["status"]!="ready":
            entry.update(p1=None,p2=None,missing_reason=rec["status"])
            continue
        ranks,geom=open_results(a.scores.parent,rec)
        k16="coverage-16"
        check(k16 in ranks,"registered K16 missing")
        ranks16=ranks[k16]
        if scores["mode"]=="development":
            check(set(ranks16)==set(RULES)|{"lr-0.001","lr-0.01","lr-0.1"},"unexpected model in development domain")
            common=v & np.logical_and.reduce([np.isfinite(r) for r in ranks16.values()])
            entry["selection"]={m:quota(r,common,high,20) for m,r in ranks16.items()}
            entry["selection_common"]=int(common.sum())
            continue
        entire=v & np.logical_and.reduce([np.isfinite(r) for scoreset in ranks.values() for r in scoreset.values()])
        method="B-CPU-LR"; simple=a.rule
        common=v & np.isfinite(ranks16[method]) & np.isfinite(ranks16[simple])
        new=quota(ranks16[method],common,high,20); baseline=quota(ranks16[simple],common,high,20)
        check(new["selected"]==baseline["selected"],"unpaired P2 quota")
        entry["p2"]=dict(common=int(common.sum()),quota=new["selected"],new_errors=new["errors"],
            undefined=(None if new["selected"] else "quota_rounds_to_zero" if common.any() else "empty_common_domain"),
            rule=simple,rule_errors=baseline["errors"],new_error_rate=new["error_rate"],
            rule_error_rate=baseline["error_rate"],quota_over_reference=ratio(new["selected"],entry["reference"]),
            new_minus_rule=ratio(new["errors"]-baseline["errors"],new["selected"]),
            common_hash=mask_id(common),new_selection=new["pixel_indices_sha256"],rule_selection=baseline["pixel_indices_sha256"])
        full=geom["coverage-"+str(len(b.ids))]
        mechanisms={c:scope_metrics(g,full,v,high) for c,g in geom.items()}
        entry["mechanism"]=mechanisms
        entry["queries"]={k:dict(actual_k=c["actual_k"],nominal_budgets=c["nominal_budgets"],source_ids=c["queries"],
            query_cost=c["query_cost"],selection_seconds=c["selection_seconds"],head_seconds=c["head_seconds"],
            all_three_simple_rules_seconds=c["all_three_simple_rules_seconds"]) for k,c in rec["cells"].items()}
        cg,fg,larger=(mechanisms[c]["correct_gap_rate"] for c in ("coverage-4","fixed-4","coverage-16"))
        entry["p1"]=dict(coverage4_minus_fixed4=cg-fg if cg is not None and fg is not None else None,
            coverage16_minus_coverage4=larger-cg if cg is not None and larger is not None else None)
        entry["nearest_depth_visibility_approximation"]=visibility(b,archive,cache,truth,view,rec["cells"],geom)
        entry["visibility"]=exact_visibility(probe,view,rec["cells"],geom) if probe else None
        entry["priority"]={}
        cost=rec["cells"][k16]
        for method_name,rank in ranks16.items():
            tick=time.perf_counter()
            priority=opening_priority(rank)
            priority.update(summary_seconds=time.perf_counter()-tick,
                shared_selector_seconds=cost["selection_seconds"],
                shared_evidence_seconds=cost["query_cost"]["elapsed_through_snapshot_seconds"],
                sources_projected=cost["query_cost"]["source_geometry_projections"],
                head_seconds=cost["head_seconds"].get(method_name),
                all_three_simple_rules_seconds=cost["all_three_simple_rules_seconds"])
            entry["priority"][method_name]=priority
        for cell,scoreset in ranks.items():
            pair=v & np.logical_and.reduce([np.isfinite(r) for r in scoreset.values()])
            for m,r in scoreset.items():
                reasons=geom[cell].get("reason_"+m,geom[cell]["reason"])
                unknown={str(int(ix)):int(n) for ix,n in zip(*np.unique(reasons[~np.isfinite(r)],return_counts=True))}
                for name,mask in [("own",None),("same_query_methods",pair),("all_checkpoint_ranks",entire)]:
                    metric=ranking_metrics(r,ref,v,high,domain=mask)
                    comparisons.append(dict(target=view,cell=cell,method=m,domain=name,
                                            unknown_reasons=unknown,**metric))
    save(a.out/"rank-domains.json",dict(schema="el-many-domain-metrics-1",rows=comparisons))
    save(a.out/"evaluation.json",dict(schema="el-many-checkpoint-evaluation-1",run_id=os.getenv("EL_RUN_ID"),
        scene=scores["family"],role=scores["role"],R=scores["R"],mode=scores["mode"],rule=a.rule,
        scores_sha256=digest(a.scores),bank_sha256=b.sha256,truth_sha256=digest(a.truth),
        opening_sha256=digest(a.opening) if a.opening else None,visibility_sha256=digest(a.visibility) if probe else None,views=rows,
        domain_results_sha256=digest(a.out/"rank-domains.json"),cpu_core_seconds=time.process_time()-cpu,
        wall_seconds=time.perf_counter()-wall))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("scores","bank","truth","opening","freeze","visibility","out"):
        p.add_argument("--"+n,type=Path,required=n not in ("opening","freeze","visibility"))
    p.add_argument("--rule",choices=RULES)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    evaluate(args)
