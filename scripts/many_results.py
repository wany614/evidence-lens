"""Reduce sealed per-view results with equal target, checkpoint and scene weights."""

import argparse
import csv
import os
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from dual_records import selected,stage,resolve
from evidence_lens._bundle import load,save,sha
from evidence_lens._verify import require
from many_science_control import CONFIRM, DEV, ROSTER, slug


def table(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x",encoding="utf-8",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=keys,lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean_or_missing(values):
    return sum(values)/len(values) if values and all(v is not None for v in values) else None


def format_num(value):
    return "N/A" if value is None else f"{value:.4%}"


def summarize(data,out,freeze):
    scene_rows=[];view_rows=[];scopes=[];visibility=[];depth_approx=[];ranks=[];curves=[];proxies=[];reason_rows=[];query_rows=[]
    domain_collect={}
    for scene,configs in ROSTER.items():
        part="confirmation" if scene in CONFIRM else "development" if scene in DEV else "train"
        checkpoints=[]
        for n in configs:
            d,rankfile=data[(scene,n)]
            require(d["scene"]==scene and d["R"]==n and d["role"]==part,"different registered checkpoint")
            planned_views=6 if scene.startswith("b0") else 10
            require(set(d["views"])=={f"h{i:02d}" for i in range(planned_views)},"omitted planned targets")
            cells=["fixed-4","nearest-4"]+["coverage-"+str(k) for k in sorted({4,8,16,min(32,n),n})]
            for cell in cells:
                methods=["coverage","alpha_missing","residual","B-CPU-LR"]
                if cell=="coverage-4":methods+=["MS-L1","MS-ft"]
                for method in methods:
                    for dom in ("own","same_query_methods","all_checkpoint_ranks"):
                        domain_collect[scene,n,cell,method,dom]=[]
            if rankfile is not None:
                require(sha(rankfile)==d["domain_results_sha256"],"changed signed domain table")
                domain=load(rankfile)
                for row in domain["rows"]:
                    ident=dict(scene=scene,R=n,view=row["target"],cell=row["cell"],method=row["method"],domain=row["domain"])
                    basic={k:row[k] for k in ("canvas","reference","labelled","reference_errors","finite_scores",
                                               "denominator","positive","ap","base","ap_undefined")}
                    append={**ident,**basic}
                    for q in row["quota"]:
                        prefix=q["direction"]+"_"+str(q["percent"])+"_"
                        append.update({prefix+k:q[k] for k in ("selected","errors","error_rate","share_full_reference","error_capture")})
                        total=row["reference_errors"]
                        append[prefix+"capture_of_all_labelled_errors"]=q["errors"]/total if total else None
                        append[prefix+"undefined_quota"]=None if q["selected"] else (
                            "quota_rounds_to_zero" if q["eligible"] else "empty_domain")
                    ranks.append(append)
                    for q in row["coverage_curve"]:curves.append({**ident,**q})
                    if row["domain"]=="own":
                        for code,count in row["unknown_reasons"].items():reason_rows.append({**ident,"reason_code":code,"pixels":count})
                    key=(scene,n,row["cell"],row["method"],row["domain"])
                    domain_collect.setdefault(key,[]).append((row["ap"],row["base"],row["positive"],row["denominator"]))
            per=[]
            for v,r in d["views"].items():
                ident=dict(scene=scene,R=n,view=v,role=part)
                vals={k:r.get(k) for k in ("status","reference","labelled","errors","prediction_unavailable",
                                            "prediction_sha256","labels_sha256","valid_labels_sha256")}
                p2=r.get("p2") or {};p1=r.get("p1") or {}
                value={**ident,**vals,**p2,**p1}
                view_rows.append(value);per.append(value)
                for cell,sc in r.get("mechanism",{}).items():
                    scopes.append({**ident,"cell":cell,**{k:v for k,v in sc.items() if k!="gap_witness_sources"}})
                for cell,sc in r.get("queries",{}).items():
                    query_rows.append({**ident,"cell":cell,"actual_k":sc["actual_k"],
                        "nominal_budgets":" ".join(map(str,sc["nominal_budgets"])),"query_ids":" ".join(sc["source_ids"]),
                        "selection_seconds":sc["selection_seconds"],"head_seconds":sc["head_seconds"]["B-CPU-LR"],
                        "all_three_simple_rules_seconds":sc["all_three_simple_rules_seconds"],**sc["query_cost"]})
                for cell,sc in (r.get("visibility") or {}).items(): visibility.append({**ident,"cell":cell,**sc})
                for cell,sc in (r.get("nearest_depth_visibility_approximation") or {}).items():
                    depth_approx.append({**ident,"cell":cell,**sc})
            checkpoints.append({k:mean_or_missing([v.get(k) for v in per]) for k in
                                ("new_minus_rule","new_error_rate","rule_error_rate","quota_over_reference",
                                 "coverage4_minus_fixed4","coverage16_minus_coverage4")})
            if scene in ("delivery_area","living_room","bridge"):
                ids=list(d["views"])
                for method in ("B-CPU-LR","coverage","alpha_missing","residual"):
                    def key(v):
                        rec=d["views"][v].get("priority",{}).get(method)
                        if not rec or rec["no_scores"]:return (0,0,-1,v)
                        return (1,-rec["top_decile_mean"],-rec["unknown_fraction"],v)
                    tick=time.perf_counter()
                    priority=sorted(ids,key=key)
                    sorting_seconds=time.perf_counter()-tick
                    chosen=priority[:2]
                    all_err=sum(r.get("errors",0) for r in d["views"].values())
                    hit=sum(d["views"][v].get("errors",0) for v in chosen)
                    observed=[entry["priority"][method] for entry in d["views"].values() if entry.get("priority")]
                    proxies.append(dict(scene=scene,R=n,method=method,planned=len(ids),opened=len(chosen),
                        first=chosen[0],second=chosen[1],captured=hit,omitted=all_err-hit,total_label_errors=all_err,
                        labelled=sum(r.get("labelled",0) for r in d["views"].values()),
                        full_reference=sum(r.get("reference",0) for r in d["views"].values()),
                        prediction_unavailable=sum(r["prediction_unavailable"] for r in d["views"].values()),
                        opened_prediction_unavailable=sum(d["views"][v]["prediction_unavailable"] for v in chosen),
                        no_score_views=len(ids)-sum(not r["no_scores"] for r in observed),
                        unknown_target_count=sum(d["views"][v].get("status")!="ready" for v in ids),
                        query_source_projections=sum(r["sources_projected"] for r in observed),
                        shared_selection_seconds=sum(r["shared_selector_seconds"] for r in observed),
                        shared_K16_evidence_seconds=sum(r["shared_evidence_seconds"] for r in observed),
                        summary_seconds=sum(r["summary_seconds"] for r in observed),ordering_seconds=sorting_seconds,
                        head_seconds=sum(r["head_seconds"] for r in observed) if method=="B-CPU-LR" else None,
                        shared_all_three_rules_seconds=sum(r["all_three_simple_rules_seconds"] for r in observed)))
        all_views=[v for v in view_rows if v["scene"]==scene]
        mechanism_total={}
        for cell,key in (("fixed-4","fixed4"),("coverage-4","coverage4"),("coverage-16","coverage16")):
            values=[sc for sc in scopes if sc["scene"]==scene and sc["cell"]==cell]
            complete=len(values)==len(all_views) and all(v["correct_gap_rate"] is not None for v in values)
            mechanism_total[key+"_gap_pixels"]=sum(v["correct_with_gap"] for v in values) if complete else None
            mechanism_total[key+"_correct_label_denominator"]=sum(v["all_reference_correct"] for v in values) if complete else None
        scene_rows.append(dict(scene=scene,role=part,checkpoints=len(configs),views=len(all_views),
            complete_p2_views=sum(v.get("new_minus_rule") is not None for v in all_views),
            reference_pixels=sum(v["reference"] for v in all_views),labelled_pixels=sum(v["labelled"] for v in all_views),
            error_pixels=sum(v["errors"] for v in all_views),common=sum(v.get("common",0) for v in all_views),
            quota=sum(v.get("quota",0) for v in all_views),new_errors=sum(v.get("new_errors",0) for v in all_views),
            rule_errors=sum(v.get("rule_errors",0) for v in all_views),**mechanism_total,**{
                k:mean_or_missing([q[k] for q in checkpoints]) for k in checkpoints[0]}))
    metrics=[]
    for (scene,n,cell,method,domain),items in domain_collect.items():
        planned=6 if scene.startswith("b0") else 10
        complete=len(items)==planned
        metrics.append(dict(scene=scene,R=n,cell=cell,method=method,domain=domain,planned_views=planned,
            measured_views=len(items),omitted_views=planned-len(items),
            views_with_defined_ap=sum(x[0] is not None for x in items),
            ap_target_equal=mean_or_missing([x[0] for x in items]) if complete else None,
            base_target_equal=mean_or_missing([x[1] for x in items]) if complete else None,
            errors=sum(v[2] for v in items),denominator=sum(v[3] for v in items)))
    scene_domains=[]
    lookup={(r["scene"],r["R"],r["cell"],r["method"],r["domain"]):r for r in metrics}
    for scene,configs in ROSTER.items():
        for strategy,nominal in (("fixed",4),("nearest",4),*(("coverage",n) for n in (4,8,16,32,"all"))):
            for method in ("B-CPU-LR","coverage","alpha_missing","residual","MS-L1","MS-ft"):
                if method.startswith("MS-") and (strategy,nominal)!=("coverage",4):continue
                for domain in ("own","same_query_methods","all_checkpoint_ranks"):
                    rows=[lookup[scene,n,strategy+"-"+str(n if nominal=="all" else min(n,nominal)),method,domain] for n in configs]
                    scene_domains.append(dict(scene=scene,strategy=strategy,nominal_k=nominal,method=method,domain=domain,
                        checkpoints=len(configs),planned_views=sum(d["planned_views"] for d in rows),
                        measured_views=sum(d["measured_views"] for d in rows),
                        views_with_defined_ap=sum(d["views_with_defined_ap"] for d in rows),
                        ap_target_then_checkpoint_equal=mean_or_missing([d["ap_target_equal"] for d in rows]),
                        base_target_then_checkpoint_equal=mean_or_missing([d["base_target_equal"] for d in rows]),
                        pooled_positive=sum(d["errors"] for d in rows),pooled_denominator=sum(d["denominator"] for d in rows)))
    outputs={"scenes":scene_rows,"views":view_rows,"comparability":scopes,"geometric-visibility":visibility,
             "nearest-depth-approximation":depth_approx,"query-costs":query_rows,
             "view-priority":proxies,"domain-ranks":ranks,"domain-averages":metrics,"scene-domain-averages":scene_domains,
             "coverage-curves":curves,"unknown-reasons":reason_rows}
    for name,rows in outputs.items():table(out/(name+".csv"),rows)
    external=[r for r in scene_rows if r["scene"] in ("living_room","bridge")]
    confirms=[r for r in scene_rows if r["role"]=="confirmation"]
    sufficient=all(x["complete_p2_views"]==x["views"] for x in scene_rows)
    response="blocked_or_incomplete" if not sufficient else (
        "completed_supported" if all(x["new_minus_rule"]<0 for x in external) else "completed_negative")
    result=dict(schema="el-many-scientific-endpoints-1",run_id=os.environ["EL_RUN_ID"],comparison_status=response,
        cost_and_implementation_acceptance="separately required for final B status",strong_simple_rule=freeze["strong_simple_rule"],
        scenes=scene_rows,external_confirm_scene_equal_delta=mean_or_missing([r["new_minus_rule"] for r in external]),
        all_confirm_scene_equal_delta=mean_or_missing([r["new_minus_rule"] for r in confirms]),
        heldout_external_scenes=len(external),all_families=len(ROSTER),checkpoints=len(data),
        rows={k:len(v) for k,v in outputs.items()},
        files={name+".csv":{"bytes":(out/(name+".csv")).stat().st_size,"sha256":sha(out/(name+".csv"))} for name in outputs})
    save(out/"endpoints.json",result)
    lines=["# Frozen multi-source evidence study","","Two ETH3D scenes supply the external confirmation. "
           "The primary comparison uses the same K16 sources and the same available pixels for both rankings. "
           "Each target accepts the lowest-scoring fifth of this common domain, rounded down. "
           "The scene-level effects below average targets equally within each frozen checkpoint.","",
           f"The development-selected simple rule was **{freeze['strong_simple_rule']}**. "
           f"The registered primary endpoint returned **{response}**. The downstream release records engineering and budget acceptance separately.","",
           "| Scene | Role | Labelled errors / pixels | New errors / accepted | Rule errors / accepted | New minus rule, target-weighted | Coverage Q4 minus fixed Q4 gap |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for r in scene_rows:
        lines.append(f"| {r['scene']} | {r['role']} | {r['error_pixels']:,} / {r['labelled_pixels']:,} | "
                     f"{r['new_errors']:,} / {r['quota']:,} | {r['rule_errors']:,} / {r['quota']:,} | "
                     f"{format_num(r['new_minus_rule'])} | {format_num(r['coverage4_minus_fixed4'])} |")
    lines+=["","## Missing comparable evidence","","| Scene | Fixed Q4 gaps / correct | Coverage Q4 gaps / correct | Coverage Q16 gaps / correct |",
            "|---|---:|---:|---:|"]
    for r in scene_rows:
        def gap(k):
            return "N/A" if r[k+"_gap_pixels"] is None else f"{r[k+'_gap_pixels']:,} / {r[k+'_correct_label_denominator']:,}"
        lines.append(f"| {r['scene']} | {gap('fixed4')} | {gap('coverage4')} | {gap('coverage16')} |")
    lines+=["","The P1 gap requires a correct independent RGB label, no comparable source in the selection, and a comparable source in the complete audited bank. "
            "Its denominator is all correctly predicted labelled pixels in that target. The P2 table includes the accepted share of the complete reference image. "
            "For the authored scenes, geometric-visibility.csv compares the model to independent mesh line-of-sight on the registered 54 × 36 grid. "
            "nearest-depth-approximation.csv reports the separate dense calculation from sampled depth. "
            "Error capture uses eligible positives. The column capture_of_all_labelled_errors includes the errors outside that score domain. "
            "View-priority times distinguish shared source selection and evidence, joint simple-rule timing, head execution and target-level aggregation.","",
            "[Per-view primary denominators](views.csv) · [Each method and domain](domain-ranks.csv) · "
            "[Scene comparability](comparability.csv) · [Two-view priority check](view-priority.csv) · [All measured results](endpoints.json)"]
    (out/"results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    with ZipFile(out/"many-source-results.zip","x",compression=ZIP_DEFLATED,compresslevel=6) as z:
        for f in [*(out/(name+".csv") for name in outputs),out/"endpoints.json",out/"results.md"]:
            z.write(f,f.name)


def control(s):
    pins={"frozen":selected(s,"b-model-freeze"),"opening":selected(s,"b-confirmation-h-opening")}
    cmd=["{python}","scripts/many_results.py","--freeze","{input:frozen}","--out","{out}"]
    for f,ns in ROSTER.items():
        for n in ns:
            p=selected(s,"b-eval-"+slug(f,n)); k="case"+str(len(pins)-2);pins[k]=p
            path=resolve((s,s/"history"),p)
            cmd+=["--evaluation",str(path)]
    return stage(s,"b-scientific-results","report",pins,{"comparison":"P1 and P2, all 15 required checkpoints"},
                 cmd,"payload/endpoints.json",65_000_000)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("store","out","freeze"):p.add_argument("--"+n,type=Path)
    p.add_argument("--evaluation",type=Path,action="append")
    a=p.parse_args()
    if a.store:print(control(a.store))
    else:
        all_data={}
        for path in a.evaluation:
            d=load(path);key=(d["scene"],d["R"])
            require(key not in all_data,"duplicate checkpoint")
            rank=path.parent/"rank-domains.json"
            all_data[key]=(d,rank if rank.is_file() else None)
        require(set(all_data)=={(f,n) for f,ns in ROSTER.items() for n in ns},"incomplete checkpoint ledger")
        summarize(all_data,a.out,load(a.freeze))
