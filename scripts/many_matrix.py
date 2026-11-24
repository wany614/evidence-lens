"""Extract training samples or seal target-level ranks without giving the scorer reference access."""

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from many_head import TRAIN, check, digest, load, sample_ids, save


def checked_file(root,spec):
    import re
    rel=Path(spec["path"])
    check(not rel.is_absolute() and ".." not in rel.parts and "\\" not in spec["path"]
          and re.fullmatch(r"[a-f0-9]{64}",spec["sha256"]), "nonportable registered file")
    f=root/rel
    check(f.is_file() and not f.is_symlink() and digest(f)==spec["sha256"], "different input "+spec["path"])
    return f


def prediction_hash(bank,view):
    item=bank.meta["targets"][view]["render"]
    return item["sha256"] if item else None


def sample(bank,a):
    from evidence_lens_many import FEATURE_NAMES,FEATURE_SCHEMA_SHA256,select_sources,stream_features
    from evidence_lens_many.evaluation import labels,mask_id

    tick=time.perf_counter();cpu=time.process_time()
    check(a.family in TRAIN and a.count in (16,128) and a.truth is not None, "authored training role required")
    truth=load(a.truth)
    check(truth["schema"]=="el-many-evaluator-truth-1" and truth["scene_id"]==a.family
          and truth["split"]=="train" and sorted(truth["targets"])==sorted(bank.target_ids),
          "unexpected reference schema or target list")
    from many_data_access import EvaluatorTruth
    reference=EvaluatorTruth(a.truth,digest(a.truth),bank=bank)
    ys=[]; xs=[]; views=[]; pixels=[]; budgets=[]; hashes={}; counts={}
    for i,view in enumerate(bank.target_ids):
        pred=bank.target(view)
        check(pred is not None,"training prediction missing "+view)
        ref=reference.target_rgb(view)
        _,valid,high,_=labels(pred["rgb"],ref)
        chosen=sample_ids(a.family,a.count,view,valid)
        plan=select_sources(bank,view,strategy="coverage")
        hashes[view]={"event":mask_id(high),"valid":mask_id(valid),"prediction":prediction_hash(bank,view),
                      "reference":truth["targets"][view]["analysis_rgb"]["sha256"]}
        counts[view]={}
        for step in stream_features(bank,view,plan,budgets=(4,8,16,32,"all")):
            idx=chosen[step.finite.ravel()[chosen]]
            xs.append(step.X.reshape(-1,len(FEATURE_NAMES))[idx].copy())
            ys.append(high.ravel()[idx].astype(np.uint8)); pixels.append(idx)
            views.append(np.full(len(idx),i,np.uint8)); budgets.append(np.full(len(idx),step.actual_k,np.int32))
            counts[view][str(step.actual_k)]=len(idx)
    dest=a.out/"samples.npz"
    np.savez_compressed(dest,X=np.concatenate(xs),y=np.concatenate(ys),pixel=np.concatenate(pixels),
                        view=np.concatenate(views),k=np.concatenate(budgets))
    save(a.out/"sample.json",dict(schema="el-many-authored-training-sample-1",run_id=os.getenv("EL_RUN_ID"),
        role="train",family=a.family,R=a.count,bank_sha256=bank.sha256,target_ids=list(bank.target_ids),
        feature_schema_sha256=FEATURE_SCHEMA_SHA256,columns=list(FEATURE_NAMES),sampled=counts,
        truth_sha256=digest(a.truth),label_hashes=hashes,samples_sha256=digest(dest),rows=sum(map(len,pixels)),
        cpu_core_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-tick))


def counters(bank):
    return {key:value.item() if isinstance(value,np.generic) else value for key,value in bank.counters.items()}


def score_case(bank,a):
    from evidence_lens_many import FEATURE_SCHEMA_SHA256,legacy_q4,score_lr,select_sources,simple_ranks,stream_features
    from evidence_lens_many.schema import REASONS

    check(a.truth is None and a.model, "scoring accepts frozen parameters and deployment evidence")
    models=[(path,load(path)) for path in a.model]
    development=a.mode=="development"
    if development:
        check(len(models)==3 and {d["lambda_l2"] for _,d in models}=={.001,.01,.1}
              and a.family in ("delivery_area","b04-layered-louvers"),"three registered development candidates required")
    else:
        check(len(models)==1 and a.legacy_models,"registered new and old models required")
    model_ids={("lr-"+format(d["lambda_l2"],"g") if development else "B-CPU-LR"):digest(f) for f,d in models}
    cases={}; process_start=time.process_time(); wall=time.perf_counter()
    for view in bank.target_ids:
        fdir=a.out/view; fdir.mkdir()
        pred=bank.target(view)
        if pred is None:
            cases[view]=dict(status="missing_target_prediction",prediction=None,cells={})
            continue
        cases[view]=dict(status="ready",prediction=prediction_hash(bank,view),cells={})
        strategies=("coverage",) if development else ("coverage","fixed","nearest")
        for strategy in strategies:
            t=time.perf_counter()
            before=counters(bank)
            plan=select_sources(bank,view,strategy=strategy)
            selection_seconds=time.perf_counter()-t
            requested=(16,) if development else (4,8,16,32,"all") if strategy=="coverage" else (4,)
            for step in stream_features(bank,view,plan,budgets=requested):
                name=f"{strategy}-{step.actual_k}"
                start=time.perf_counter()
                scores=simple_ranks(step)
                simple_seconds=time.perf_counter()-start
                arrays={"rank_"+k:v for k,v in scores.items()}
                rank_times={}
                for f,model in models:
                    key="lr-"+format(model["lambda_l2"],"g") if development else "B-CPU-LR"
                    start=time.perf_counter()
                    arrays["rank_"+key]=score_lr(step,model)
                    rank_times[key]=time.perf_counter()-start
                arrays.update(finite=step.finite,reason=step.reason,invalid_geometry=step.invalid_geometry,
                    incomplete=step.incomplete,comparable=step.comparable_count,support=step.support_count,
                    in_frame=step.in_frame_count,depth_conflict=step.depth_conflict_count,
                    colour_conflict=step.colour_conflict_count,witness=step.first_witness)
                q=list(step.queried_source_ids); legacy={}
                if not development and strategy=="coverage" and step.actual_k==4:
                    for method in ("MS-L1","MS-ft"):
                        old=legacy_q4(bank,view,q,method,a.legacy_models,threads=a.threads)
                        rank=old["rank"]
                        check(rank.shape==step.finite.shape and np.array_equal(np.isfinite(rank),old["finite"]),
                              "frozen model rank domain differs")
                        arrays["rank_"+method]=rank
                        if "reason" in old: arrays["reason_"+method]=old["reason"]
                        legacy[method]={k:v for k,v in old.items() if not isinstance(v,np.ndarray)}
                file=fdir/(name+".npz")
                np.savez_compressed(file,**arrays)
                cases[view]["cells"][name]=dict(strategy=strategy,actual_k=step.actual_k,
                    nominal_budgets=list(step.nominal_budgets),queries=q,file=file.relative_to(a.out).as_posix(),
                    bytes=file.stat().st_size,sha256=digest(file),selection_seconds=selection_seconds,
                    head_seconds=rank_times,all_three_simple_rules_seconds=simple_seconds,
                    elapsed_from_selection_start=time.perf_counter()-t,query_cost=step.cost,
                    bank_counters_before_selection=before,bank_counters_now=counters(bank),legacy=legacy)
    import psutil
    mem=psutil.Process().memory_info()
    if sys.platform=="darwin" or sys.platform.startswith("linux"):
        import resource
        high=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)
    else:
        high=getattr(mem,"peak_wset",None)
    save(a.out/"scores.json",dict(schema="el-many-checkpoint-ranks-1",run_id=os.getenv("EL_RUN_ID"),
         role=a.role,family=a.family,R=a.count,A=len(bank.ids),source_ids=list(bank.ids),target_ids=list(bank.target_ids),
         mode=a.mode,bank_sha256=bank.sha256,checkpoint_sha256=bank.meta["checkpoint_sha256"],
         feature_schema_sha256=FEATURE_SCHEMA_SHA256,models=model_ids,reason_names=list(REASONS),
         rank_space="uncalibrated head; conditional old Q4 transfer",references_accessed=0,views=cases,
         wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-process_start,
         process_peak_rss_bytes=high,counters=counters(bank)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["sample","development","matrix"])
    for n in ("bank","truth","out","legacy-models"):
        parser.add_argument("--"+n,type=Path,required=n in ("bank","out"))
    parser.add_argument("--bank-sha",required=True)
    parser.add_argument("--family",required=True)
    parser.add_argument("--count",required=True,type=int)
    parser.add_argument("--role",choices=["train","development","confirmation"],required=True)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--model",type=Path,action="append")
    a=parser.parse_args()
    from evidence_lens_many import Bank
    b=Bank(a.bank,a.bank_sha)
    check(b.meta["scene_id"]==a.family and len(b.reconstruction_ids)==a.count and len(b.ids)==a.count
          and b.meta["reconstruction_provenance"]=="verified","wrong or unverified R/A bank")
    a.out.mkdir(parents=True,exist_ok=True)
    if a.mode=="sample": sample(b,a)
    else: score_case(b,a)
