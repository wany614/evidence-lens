"""Advance registered fit, ranking and evaluation stages using original terminal recorder members."""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from dual_records import bind, resolve, selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from many_head import TRAIN, LAMBDAS


DEV={"b04-layered-louvers":(16,128),"delivery_area":(34,)}
CONFIRM={"b05-perforated-gallery":(16,128),"b06-corner-alcoves":(16,128),
         "living_room":(55,), "bridge":(100,)}
ROSTER={**{f:(16,128) for f in TRAIN},**DEV,**CONFIRM}
RULES=("coverage","alpha_missing","residual")
FROZEN_FILES=("scripts/many_matrix.py","scripts/many_evaluate.py","scripts/many_head.py",
              "scripts/many_data_access.py","scripts/many_data_visibility.py","scripts/many_data_visibility.cpp",
              "scripts/many_data_terminal.py","scripts/many_results.py","scripts/many_science_control.py")


def slug(f,n):
    return f.replace("_","-")+"-r"+str(n)


def suite(s):
    return {"study":selected(s,"b-study-registration"),"roles":selected(s,"b-data-registration"),
            "runtime":selected(s,"b-score-runtime")}


def source_path(s, pin):
    return resolve((s,s/"history"),pin)


def members(s, pin, filenames):
    return {k:bind(s,pin["run_id"],"payload/"+v) for k,v in filenames.items()}


def runtime(s,pins):
    r=load(source_path(s,pins["runtime"]))
    path=s/r["relative_executable"]
    require(path.is_file(),"verified research CPU installation unavailable")
    return str(path)

def cpu_command(py, script):
    return ["env","OPENBLAS_NUM_THREADS=1","VECLIB_MAXIMUM_THREADS=1","MKL_NUM_THREADS=1",
            "OMP_NUM_THREADS=1",py,"-I","-B","{source}/scripts/"+script]


def case_stage(a):
    s=a.store; family=a.family; n=a.count
    require(n in ROSTER[family],"unregistered checkpoint")
    unit=slug(family,n)
    role="train" if family in TRAIN else "development" if family in DEV else "confirmation"
    pins=suite(s)
    pins["bank"]=selected(s,"b-bank-"+unit)
    bank=source_path(s,pins["bank"]).parent
    py=runtime(s,pins)
    if a.action in ("sample","dev-score","matrix"):
        mode={"sample":"sample","dev-score":"development","matrix":"matrix"}[a.action]
        cmd=cpu_command(py,"many_matrix.py")+[mode,"--bank",str(bank),"--bank-sha",
             pins["bank"]["sha256"],"--family",family,"--count",str(n),"--role",role,"--out","{out}"]
        if mode=="sample":
            require(role=="train","external or confirmation reference excluded from training")
            truth=selected(s,"b-truth-"+family.replace("_","-"));pins["truth"]=truth
            pins.update(members(s,truth,{"truth_archive":"truth.zip"}))
            cmd+=["--truth",str(source_path(s,truth))]
            return stage(s,"b-sample-"+unit,"features",pins,{"split":role,"family":family,"R":n,"uses_H":"training labels only"},
                         cmd,"payload/sample.json",12500000)
        if mode=="development":
            require(role=="development","candidate selection only on registered development families")
            fit=selected(s,"b-head-candidates");pins["candidate_fit"]=fit
            for i,lam in enumerate(LAMBDAS):
                label="candidate"+str(i)
                pins[label]=bind(s,fit["run_id"],"payload/candidate-"+format(lam,"g")+".json")
                cmd+=["--model","{input:"+label+"}"]
            return stage(s,"b-dev-ranks-"+unit,"model_score",pins,{"split":role,"family":family,"R":n,"H_access":False},
                         cmd,"payload/scores.json",40000000)
        pins["model_freeze"]=selected(s,"b-model-freeze")
        frozen=load(source_path(s,pins["model_freeze"]))
        require(all(sha(Path(f))==v for f,v in frozen["science_programs"].items()),
                "registered confirmation or scoring program changed")
        pins["model"]=frozen["final_model"]
        pins["old"]=bind(s/"history","20261124T041052425868Z-d308c134e69d4d4b","payload/cpu/release.json")
        cmd+=["--model","{input:model}","--legacy-models",str(source_path(s,pins["old"]).parent)]
        return stage(s,"b-matrix-"+unit,"model_score",pins,
                     {"split":role,"family":family,"R":n,"H_access":False,"source_bank_audit_requested":True,
                      "new_head":pins["model"]["sha256"],"old_models":frozen["old_models"]},
                     cmd,"payload/scores.json",165000000)
    require(a.action in ("dev-eval","evaluate"),"unknown evaluation")
    dev=a.action=="dev-eval"
    require(not dev or role=="development","development-only measurements on different partition")
    truth=selected(s,"b-truth-"+family.replace("_","-"));pins["truth"]=truth
    pins.update(members(s,truth,{"truth_archive":"truth.zip"}))
    pins["ranks"]=selected(s,("b-dev-ranks-" if dev else "b-matrix-")+unit)
    cmd=cpu_command(py,"many_evaluate.py")+["--scores",str(source_path(s,pins["ranks"])),
         "--bank",str(bank),"--truth",str(source_path(s,truth)),"--out","{out}"]
    if not dev:
        pins["model_freeze"]=selected(s,"b-model-freeze")
        freeze=load(source_path(s,pins["model_freeze"]))
        require(all(sha(Path(f))==v for f,v in freeze["science_programs"].items()),"frozen measurement code changed")
        cmd+=["--rule",freeze["strong_simple_rule"],"--freeze","{input:model_freeze}"]
        if family.startswith("b0"):
            probe=selected(s,"b-exact-visibility-"+family)
            pins["visibility"]=probe
            pins.update(members(s,probe,{"visibility_archive":"visibility.zip"}))
            cmd+=["--visibility",str(source_path(s,probe))]
    if role=="confirmation":
        pins["opening"]=selected(s,"b-confirmation-h-opening")
        cmd+=["--opening","{input:opening}"]
    return stage(s,("b-dev-eval-" if dev else "b-eval-")+unit,"evaluation",pins,
                 {"role":role,"R":n,"scene":family,"early_candidate_evaluation":dev},cmd,"payload/evaluation.json",15000000)


def head_stage(a):
    s=a.store;pins=suite(s)
    py=runtime(s,pins)
    cmd=cpu_command(py,"many_head.py")+["--out","{out}"]
    for f in TRAIN:
        for n in (16,128):
            name="s"+str(2*TRAIN.index(f)+(n==128))
            pins[name]=selected(s,"b-sample-"+slug(f,n))
            pins[name+"_arrays"]=bind(s,pins[name]["run_id"],"payload/samples.npz")
            cmd+=["--sample",str(source_path(s,pins[name]))]
    if a.action=="refit":
        pins["choice"]=selected(s,"b-development-selection")
        pins["candidate"]=load(source_path(s,pins["choice"]))["chosen_candidate"]
        cmd+=["--selected","{input:candidate}"]
    return stage(s,"b-head-"+("final" if a.action=="refit" else "candidates"),"model_fit",pins,
        {"families":TRAIN,"max_rows":200000,"real_numpy_head":True,"neural_updates":0,
         "candidate_count":1 if a.action=="refit" else len(LAMBDAS),"data_split":"original_train_only"},
        cmd,"payload/fits.json",30000)


def aggregate_scores(families):
    data={}; quotas=[]
    for f,checkpoints in families.items():
        cells=[]
        for d in checkpoints:
            vals=[]
            for name,v in d["views"].items():
                require(v.get("selection") and all(measure["selected"]>0 for measure in v["selection"].values()),
                        "unavailable development K16 paired quota")
                vals.append({m:q["error_rate"] for m,q in v["selection"].items()})
                quotas.append(dict(scene=f,R=d["R"],view=name,reference=v["reference"],common=v["selection_common"],
                                   results=v["selection"]))
            cells.append({m:sum(v[m] for v in vals)/len(vals) for m in vals[0]})
        data[f]={m:sum(v[m] for v in cells)/len(cells) for m in cells[0]}
    means={m:sum(v[m] for v in data.values())/len(data) for m in next(iter(data.values()))}
    return data,means,quotas


def choose_worker(cfg, files,out):
    evaluations={f:[] for f in DEV}
    for key,path in files.items():
        if key.startswith("dev"):
            d=load(path)
            require(d["mode"]=="development" and d["scene"] in DEV and d["R"] in DEV[d["scene"]],"bad dev identity")
            evaluations[d["scene"]].append(d)
    require(all(sorted(item["R"] for item in ds)==sorted(DEV[f]) for f,ds in evaluations.items()),"dev coverage incomplete")
    per,means,quotas=aggregate_scores(evaluations)
    rule=min(RULES,key=lambda r:(means[r],RULES.index(r)))
    lam=min(LAMBDAS,key=lambda v:(means["lr-"+format(v,"g")],-v))
    pin=cfg["sources"]["candidate"+str(LAMBDAS.index(lam))]
    save(out/"selection.json", dict(schema="el-many-development-choice-1",run_id=os.environ["EL_RUN_ID"],
        chosen_candidate=pin,lambda_l2=lam,strong_simple_rule=rule,per_scene=per,scene_equal_error_rates=means,
        actual_quotas=quotas,reference_families=list(DEV),confirmation_observations=0))


def gate(a):
    s=a.store;pins=suite(s)
    details={"operation":a.action}
    extras={}
    if a.action=="choose":
        for idx,(f,n) in enumerate((f,n) for f,ns in DEV.items() for n in ns):
            pins["dev"+str(idx)]=selected(s,"b-dev-eval-"+slug(f,n))
        fit=selected(s,"b-head-candidates")
        pins.update(members(s,fit,{"candidate"+str(i):"candidate-"+format(v,"g")+".json" for i,v in enumerate(LAMBDAS)}))
        name="b-development-selection";receipt="selection.json"
    elif a.action=="freeze":
        pins["choice"]=selected(s,"b-development-selection")
        pins["final_fit"]=selected(s,"b-head-final")
        pins.update(members(s,pins["final_fit"],{"model":"model.json"}))
        pins["upstream_recipe"]=selected(s,a.recipe_label)
        pins["pilot_budget"]=selected(s,a.budget_label)
        pins["development_diagnostics"]=selected(s,a.diagnostics_label)
        pins["bank_build"]=selected(s,"b-score-core-build")
        pins["geometry_protocol"]=selected(s,"b-visibility-probe-registration")
        pins["development_interface"]=selected(s,"b-development-science-interface")
        pins["legacy"]=bind(s/"history","20261124T041052425868Z-d308c134e69d4d4b","payload/cpu/release.json")
        details["science_programs"]={f:sha(Path(f)) for f in FROZEN_FILES}
        commit=subprocess.check_output(["git","rev-parse","HEAD^{commit}"],text=True).strip()
        for filename,hash_value in details["science_programs"].items():
            blob=subprocess.check_output(["git","show",commit+":"+filename])
            require(hashlib.sha256(blob).hexdigest()==hash_value,"scientific program differs from source commit")
        details["science_code_commit"]=commit
        name="b-model-freeze";receipt="freeze.json"
    else:
        require(a.action=="opening","unrecognized gate")
        pins["model_freeze"]=selected(s,"b-model-freeze")
        statuses={}
        for i,(f,n) in enumerate((f,n) for f,ns in CONFIRM.items() for n in ns):
            key="terminal"+str(i)
            name="b-matrix-"+slug(f,n)
            if (s/"work/controller"/name/"selected.json").is_file():
                pin=selected(s,name)
            else:
                # Only the recorded terminal worker can stand in for a missing checkpoint.
                pin=selected(s,"b-terminal-failure-"+slug(f,n))
            pins[key]=pin
            statuses[key]={"family":f,"R":n}
        details["terminals"]=statuses
        name="b-confirmation-h-opening";receipt="opening.json"
    command=["{python}","scripts/many_science_control.py",a.action+"-worker","--config","{config}","--out","{out}"]
    if a.action=="opening":command+=["--store",str(s)]
    for role in pins:command+=["--input",role,"{input:"+role+"}"]
    return stage(s,name,"archive_verify",pins,details,command,"payload/"+receipt,125000,extras)


def gate_worker(a):
    cfg=load(a.config); files={k:Path(v) for k,v in a.input}
    if a.action=="choose-worker":return choose_worker(cfg,files,a.out)
    if a.action=="freeze-worker":
        m=load(files["model"]);c=load(files["choice"]);fit=load(files["final_fit"])
        require(fit["fit_count"]==1 and m["lambda_l2"]==c["lambda_l2"] and m["tau"] is None,"different final model")
        old=load(files["legacy"])
        save(a.out/"freeze.json",dict(schema="el-many-confirmation-model-freeze-1",run_id=os.environ["EL_RUN_ID"],
            final_model=cfg["sources"]["model"],strong_simple_rule=c["strong_simple_rule"],feature_schema_sha256=m["feature_schema_sha256"],
            old_models={k:old["models"][k] for k in ("MS-L1","MS-ft")},upstream_recipe=cfg["sources"]["upstream_recipe"],
            science_programs=cfg["science_programs"],science_code_commit=cfg["science_code_commit"],
            parents=cfg["sources"],all_confirmation_references_closed=True,
            upstream_indeterminate_rule="Report missing outputs or empty P2 common domain and the development source fitting, appearance range, opacity and projection diagnostics. A calculable P2 remains reported even if image quality is poor."))
        return
    from many_matrix import checked_file
    frozen=load(files["model_freeze"])
    allowed={f:[f"h{i:02d}" for i in range(6 if f.startswith("b0") else 10)] for f in CONFIRM}
    terminals={f:{} for f in CONFIRM}
    for key,identity in cfg["terminals"].items():
        d=load(files[key]);f=identity["family"];n=identity["R"]
        if d["schema"]=="el-many-checkpoint-ranks-1":
            original=source_path(a.store,cfg["sources"][key])
            require(sha(original)==sha(files[key]),"staged score index differs from native archive")
            inventory={r["path"]:r for r in load(original.parents[1]/"manifest.json")["files"]}
            require(d["role"]=="confirmation" and d["mode"]=="matrix" and d["family"]==f and d["R"]==n and
                    d["target_ids"]==allowed[f] and set(d["views"])==set(allowed[f])
                    and d["references_accessed"]==0 and d["A"]==len(set(d["source_ids"]))==n
                    and d["models"]=={"B-CPU-LR":frozen["final_model"]["sha256"]}
                    and d["feature_schema_sha256"]==frozen["feature_schema_sha256"],"unregistered scored checkpoint")
            ks=sorted(set([4,8,16,min(32,n),n]))
            expected={"coverage-"+str(k) for k in ks}|{"fixed-4","nearest-4"}
            for item in d["views"].values():
                require((item["status"]=="ready" and set(item["cells"])==expected)
                        or item["status"]=="missing_target_prediction","incomplete confirmation score terminal")
                if item["status"]!="ready":
                    require(not item["cells"] and item["prediction"] is None,"missing view contains unsealed ranks")
                    continue
                full=item["cells"]["coverage-"+str(n)]["queries"]
                require(len(full)==len(set(full))==n and set(full)==set(d["source_ids"]),"bank audit scope differs")
                for cell,c in item["cells"].items():
                    k=int(cell.rsplit("-",1)[1])
                    q=c["queries"]
                    require(len(q)==len(set(q))==c["actual_k"]==k and set(q)<=set(full) and
                            c["query_cost"]["source_geometry_projections"]==k,"actual querying differs from registered budget")
                    if cell.startswith("coverage-"):
                        require(q==full[:k],"coverage query is not nested")
                    record=inventory["payload/"+c["file"]]
                    require(record["sha256"]==c["sha256"] and record["bytes"]==c["bytes"],"score array not sealed in native inventory")
                    checked_file(original.parent,{"path":c["file"],"sha256":c["sha256"]})
                q4=item["cells"]["coverage-4"]
                require(set(q4["legacy"])==set(frozen["old_models"]),"missing Q4 transfer model")
                for m,spec in frozen["old_models"].items():
                    p=q4["legacy"][m]["profile"]
                    require(p["method"]==m and p["queried_source_ids"]==q4["queries"]
                            and p["bank_sha256"]==d["bank_sha256"]
                            and all(p["model"].get(k)==v for k,v in spec.items()),"legacy model or query mismatch")
            entry=dict(status="succeeded",scores_run=d["run_id"],receipt=cfg["sources"][key],
                       target_outcomes={v:r["status"] for v,r in d["views"].items()})
        else:
            require(d["schema"]=="el-many-terminal-failure-1" and d["family"]==f and d["R"]==n and
                    d["target_ids"]==allowed[f] and d["failed_native_run"] and
                    d["model_freeze_sha256"]==cfg["sources"]["model_freeze"]["sha256"],"invalid substitution for planned scores")
            entry=dict(status="failed",receipt=cfg["sources"][key],native_failure=d["failed_native_run"])
        terminals[f][str(n)]=entry
    import hashlib,json
    freeze_hash=hashlib.sha256(json.dumps(terminals,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    save(a.out/"opening.json",dict(schema="el-many-h-opening-1",phase="confirmation",run_id=os.environ["EL_RUN_ID"],
        allowed_targets=allowed,terminals=terminals,prediction_freeze_sha256=freeze_hash,
        model_freeze=cfg["sources"]["model_freeze"],first_actual_reference_access="recorded separately by the evaluator"))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action")
    for n in ("store","config","out"):p.add_argument("--"+n,type=Path)
    p.add_argument("--family",choices=ROSTER)
    p.add_argument("--count",type=int)
    p.add_argument("--input",action="append",nargs=2)
    p.add_argument("--recipe-label",default="b-upstream-recipe")
    p.add_argument("--budget-label",default="b-pilot-budget")
    p.add_argument("--diagnostics-label",default="b-development-diagnostics")
    a=p.parse_args()
    if a.action.endswith("-worker"):gate_worker(a)
    elif a.action in ("choose","freeze","opening"):print(gate(a))
    elif a.action in ("candidates","refit"):print(head_stage(a))
    else:print(case_stage(a))
