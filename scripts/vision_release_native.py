"""Prepare or verify the owner-only six-view native case against pre-existing independent scores."""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE=Path(__file__).resolve()
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE.parent))
import vision_release as rel

VIEWS=rel.VIEWS
METHODS=("L1-full",*rel.BRANCHES)
DATA_NAMES={"input.png","cameras.json","predictions/prediction.json",*("predictions/"+v+".npz" for v in ("source",*VIEWS))}
KEYS={"L1-full":"l1_full","L2-frozen":"l2_frozen","L2-vision-ft":"l2_vision_ft"}


def regular(root,name):
    p=root/rel.safe_name(name)
    rel.require(p.is_file() and p.resolve()==root.resolve()/name,"expected regular file "+name)
    return p


def prepared(a):
    import numpy as np
    from evidence_lens.scoring_io import new_output
    roots={}
    def original(spec):
        rid=spec["run_id"]
        if rid not in roots:roots[rid]=rel.Original(a.store,a.shelf,spec)
        rel.require(roots[rid].pin["manifest_sha256"]==spec["manifest_sha256"],"conflicting case source")
        return roots[rid]
    freeze=original(rel.pin(a.method));method=freeze.json("payload/method-freeze.json")
    models=rel.approved_models(method);msha=rel.sha(freeze.file("payload/method-freeze.json"))
    rel.require(freeze.run.run["run_kind"]=="report" and freeze.run.run["experiment_id"]=="stage2-method-freeze"
                and method["run_id"]==freeze.pin["run_id"] and rel.sha(a.catalog)==method["catalog_sha256"],"wrong joint freeze or catalog")
    catalog=rel.read(a.catalog)
    dev=sorted(r["scene_family_id"] for r in catalog["families"] if r["split"]=="development")
    rel.require(catalog["schema"]=="el-vision-family-roster-1" and len(dev)==len(set(dev))==4,"wrong registered new development roster")
    family=dev[0]
    row=next(r for r in catalog["families"] if r["scene_family_id"]==family)
    trial=row["trials"]["B0"]
    choice_path=regular(a.store,"trials/"+trial+"/trial_selection.json")
    choice=rel.read(choice_path)
    rel.require(choice["selection_status"]=="selected" and choice["condition_id"]=="B0" and choice["scene_family_id"]==family
                and choice["trial_id"]==trial,"wrong original native selection")
    ref=original(choice["selected"]["reference"]);render=original(choice["selected"]["render"])
    rel.require(ref.run.run["run_kind"]=="scene_render" and render.run.run["run_kind"]=="render" and all(
        s.run.run["scene_family_id"]==family and s.run.run["split"]=="development" and s.run.run["condition_id"]=="B0"
        for s in (ref,render)),"expected the selected original development reference and prediction")
    rel.same_parent(render.parents(),ref.pin)
    inputs={r["role"]:r for r in freeze.json("inputs.json")["inputs"]}
    rr=freeze.file(inputs["method_roster"]["path"])
    rel.require(rel.sha(rr)==method["input_roster_sha256"],"post-fit input roster differs")
    roster=rel.read(rr);five=original(roster["stages"]["baseline_scores"])
    rel.same_parent(freeze.parents(),five.pin);baseline=five.json("payload/scores.json")
    rel.require(five.run.run["run_kind"]=="model_score" and baseline["method"]=="original-five"
                and baseline["model_sha256"]==models["L1"]["model_sha256"]
                and baseline["dataset_sha256"]==method["development_dataset_sha256"],"wrong independent logistic scores")
    baseline_views={r["view"]:r for r in baseline["views"] if r["condition"]=="B0" and r["family"]==family}
    rel.require(set(baseline_views)==set(VIEWS),"incomplete original family score maps")
    costs={};info={}
    for b in rel.BRANCHES:
        m=method["cpu_costs"][b];s=original(m["pin"])
        rel.same_parent(freeze.parents(),s.pin)
        for p in (models[b],ref.pin,render.pin):rel.same_parent(s.parents(),p)
        audit=s.json("payload/native-cost.json")
        rel.require(s.run.run["run_kind"]=="model_score" and m["family"]==family and m["model_sha256"]==models[b]["model_sha256"]
                    and rel.sha(s.file("payload/native-cost.json"))==m["native_cost_sha256"] and audit["method"]==b
                    and audit["model_sha256"]==models[b]["model_sha256"] and str(audit["platform"]).startswith(("macOS","Darwin"))
                    and audit["device"]=="cpu" and audit["all_measured_views_512"] and audit["all_measured_views_scored"],"wrong Mac native score parent")
        info[b]={r["view"]:r for r in audit["views"] if r["pass"]==0}
        rel.require(set(info[b])==set(VIEWS),"missing independent CPU cold maps")
        costs[b]=s
    with new_output(a.output) as out:
        for n in sorted(DATA_NAMES):
            p=ref.file("payload/"+n) if not n.startswith("predictions/") else render.file("payload/"+n.split("/",1)[1])
            rel.copy(out,"input/"+n,p)
        meta=rel.read(out/"input/predictions/prediction.json")
        rel.require(meta["producer_run_id"]==render.pin["run_id"] and meta["input_sha256"]==rel.sha(out/"input/input.png")
                    and meta["camera_sha256"]==rel.sha(out/"input/cameras.json") and not meta.get("failed_views"),
                    "original prediction source/camera or seven-view completion differs")
        for v in VIEWS:
            source=baseline_views[v]
            rel.require(source["status"]=="scored" and source["score"],"independent B logistic view unavailable "+v)
            f=five.file("payload/"+source["score"]["path"])
            rel.require(rel.sig(f)=={k:source["score"][k] for k in ("bytes","sha256")},"original baseline score hash differs")
            with np.load(f,allow_pickle=False) as z:x=z["l1_full"]
            valid=np.isfinite(x);state=np.full(valid.shape,255,np.uint8)
            state[valid]=(x[valid].astype(np.float64)>=method["thresholds"]["L1-full"]["tau"]).astype(np.uint8)
            dest=out/"expected"/"L1-full"/(v+".npz");dest.parent.mkdir(exist_ok=True,parents=True)
            np.savez_compressed(dest,risk=x,valid=valid,state=state)
            for b in rel.BRANCHES:
                rec=info[b][v]
                n="cold/"+v+".npz"
                src=costs[b].file("payload/"+n)
                rel.require(rel.sha(src)==rec["output_sha256"] and (rec["height"],rec["width"])==(512,512)
                            and rec["status"]=="scored","independent 512-pixel cold score differs "+b+"/"+v)
                consumed=rec["inputs"]
                for key,name in (("input_png","input.png"),("cameras","cameras.json"),("prediction_metadata","predictions/prediction.json"),
                                 ("prediction/source","predictions/source.npz"),("prediction/"+v,"predictions/"+v+".npz")):
                    rel.require(consumed[key]["sha256"]==rel.sha(out/"input"/name),"CPU reference read another formal input "+name)
                rel.copy(out,"expected/"+b+"/"+v+".npz",src)
        doc={"schema":"el-vision-native-release-case-1","family":family,"trial_id":trial,"condition":"B0","split":"development",
             "selection_sha256":rel.sha(choice_path),"catalog_sha256":method["catalog_sha256"],"method":freeze.pin,"method_sha256":msha,
             "shape":[512,512],"views":list(VIEWS),"methods":list(METHODS),"thresholds":{k:method["thresholds"][k] for k in METHODS},
             "models":{k:{j:v[j] for j in ("model_sha256","run_id","manifest_sha256")} for k,v in models.items()},
             "references":{"L1-full":five.pin,**{b:s.pin for b,s in costs.items()}},"reference":ref.pin,"render":render.pin,
             "files":{p.relative_to(out).as_posix():rel.sig(p) for p in sorted(out.rglob("*")) if p.is_file()}}
        rel.write(out,"case.json",rel.entries(doc))
    return {"case_sha256":rel.sha(a.output/"case.json"),"family":family,"method_sha256":msha,"files":len(doc["files"]),
            "direct_parents":[freeze.pin,ref.pin,render.pin,five.pin,*(s.pin for s in costs.values()),*( {k:m[k] for k in ("run_id","manifest_sha256")} for m in models.values())]}


def fingerprint_case(root,sha):
    path=regular(root,"case.json")
    rel.require(rel.sha(path)==sha,"private case differs from its external pin")
    doc=rel.read(path)
    names={"input/"+n for n in DATA_NAMES}|{"expected/"+m+"/"+v+".npz" for m in METHODS for v in VIEWS}
    rel.require(set(doc.get("models",{}))=={"L1",*rel.BRANCHES} and all(
                rel.RID.fullmatch(p["run_id"]) and rel.HASH.fullmatch(p["manifest_sha256"]) and rel.HASH.fullmatch(p["model_sha256"])
                for p in doc["models"].values()),"invalid original model identities for private case")
    rel.require(doc["schema"]=="el-vision-native-release-case-1" and doc["split"]=="development" and doc["condition"]=="B0"
                and doc["shape"]==[512,512] and doc["views"]==list(VIEWS) and doc["methods"]==list(METHODS)
                and set(doc["files"])==names,"unexpected private case scope")
    present={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    rel.require(present==names|{"case.json"},"private case has missing or undeclared assets")
    for n,identity in doc["files"].items():rel.require(rel.sig(regular(root,n))==identity,"changed private case member "+n)
    return doc


def formal_compare(expected,actual,thresholds):
    import numpy as np
    results={}
    for view in VIEWS:
        row={}
        with np.load(actual/(view+".npz"),allow_pickle=False) as predicted:
            for m in METHODS:
                k=KEYS[m]
                with np.load(expected/m/(view+".npz"),allow_pickle=False) as z:
                    rel.require(set(z.files)=={"risk","valid","state"},"unexpected original formal score fields")
                    a,av,ast=z["risk"],z["valid"],z["state"]
                b,bv,bst=(predicted[k],predicted[k+"_valid"],predicted[k+"_state"])
                for x,valid,state in ((a,av,ast),(b,bv,bst)):
                    rel.require(x.dtype==np.float32 and valid.dtype==np.bool_ and state.dtype==np.uint8
                                and x.shape==valid.shape==state.shape==(512,512) and np.array_equal(valid,np.isfinite(x))
                                and np.isnan(x[~valid]).all() and ((x[valid]>=0)&(x[valid]<=1)).all(),"invalid formal grid "+view+"/"+m)
                    want=np.full(x.shape,255,np.uint8);want[valid]=(x[valid].astype(np.float64)>=thresholds[m]["tau"]).astype(np.uint8)
                    rel.require(np.array_equal(state,want),"class states disagree with registered threshold "+view+"/"+m)
                both=av & bv
                delta=np.abs(a[both].astype(np.float64)-b[both].astype(np.float64))
                maximum=float(delta.max()) if delta.size else None
                crossing=int(np.count_nonzero(ast!=bst));support=int(np.count_nonzero(av!=bv))
                tol=.000001 if m=="L1-full" else .0002
                row[m]={"pixels":a.size,"expected_risk_sha256":hashlib.sha256(np.ascontiguousarray(a,dtype="<f4").tobytes()).hexdigest(),
                        "actual_risk_sha256":hashlib.sha256(np.ascontiguousarray(b,dtype="<f4").tobytes()).hexdigest(),"finite_expected":int(av.sum()),"finite_actual":int(bv.sum()),"finite_common":int(both.sum()),
                        "support_difference":support,"max_abs":maximum,"mean_abs":float(delta.mean()) if delta.size else None,
                        "threshold_state_difference":crossing,"absolute_tolerance":tol,
                        "passed":bool(maximum is not None and maximum<=tol and support==0 and crossing==0)}
        results[view]=row
    return results


def audited_scoring(work,pins):
    """Only this child opens deploy inputs. Comparison happens in a separate process afterwards."""
    sys.path.insert(0,str(work/"runner"))
    inputs=work/"allowed-input";models=work/"models";results=work/"output";tmp=work/"tmp"
    allowed={n:regular(inputs,n) for n in DATA_NAMES}
    declared=set(allowed.values());seen=set()
    roots=[p.resolve() for p in (models,results,tmp,work/"runner",Path(sys.prefix),Path(sys.base_prefix))]
    suffixes={".npy",".npz",".png",".exr",".ply",".pt",".pkl",".pth",".safetensors",".parquet",".json",".csv",".yaml"}
    def audit(event,args):
        if event=="socket.connect":raise PermissionError("network access during native inference")
        if event!="open" or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0]))
        if p.suffix.lower() not in suffixes:return
        p=p.resolve()
        if p in declared:
            seen.add(p);return
        if not any(p.is_relative_to(r) for r in roots):
            raise PermissionError("outside isolated deployment inputs: "+p.name)
    sys.addaudithook(audit)
    from smoke import compute
    runtime=compute(inputs,{k:models/k for k in pins},pins,results)
    rel.require(seen==declared,"native deployment did not observe every calibrated input and original view")
    return {"runtime":runtime,"read_roles":{k:rel.sig(p) for k,p in sorted(allowed.items())},
            "data_access":"Named source PNG, cameras, predicted RGB/depth/alpha and selected package. Evaluation readback runs after this process exits."}


def models_for_check(a,work,doc):
    rel.require(bool(a.archive)!=bool(a.l1) and bool(a.l1)==bool(a.frozen)==bool(a.ft),"supply the released ZIP or all three model directories")
    if a.archive:
        from vision_public_smoke import unpack_archive
        bundle=work/"bundle"
        rel.require(a.archive_sha256 and a.release_sha256,"external archive and release pins required")
        release=unpack_archive(a.archive,a.archive_sha256,bundle,a.release_sha256)
        rel.require(release["method_freeze"]["sha256"]==doc["method_sha256"] and
                    release["models"]==doc["models"] and release["catalog_sha256"]==doc["catalog_sha256"],
                    "released archive belongs to another method or cohort")
        return {k:bundle/"models"/k for k in doc["models"]},bundle/"smoke.py",bundle/"requirements.txt"
    rel.require(not a.archive_sha256 and not a.release_sha256,"archive pins require an actual ZIP")
    return {"L1":a.l1,rel.BRANCHES[0]:a.frozen,rel.BRANCHES[1]:a.ft},HERE.parent/"vision_public_smoke.py",HERE.parents[1]/"docs/releases"/rel.TAG/"requirements.txt"


def check(a):
    doc=fingerprint_case(a.case.resolve(),a.case_sha256)
    if a.case_manifest:rel.require(rel.sha(a.case_manifest)==a.case_sha256,"recorder case manifest differs from verified data")
    output=a.output.resolve()
    rel.require(not any(output.is_relative_to(p.resolve()) for p in (a.case,a.l1,a.frozen,a.ft) if p),
                "select output outside the original case and model packages")
    output.mkdir(parents=True,exist_ok=True)
    rel.require(not any(output.iterdir()),"select an empty independent native grade directory")
    run=a.run_id or os.environ.get("EL_RUN_ID")
    if run:
        rel.require(rel.RID.fullmatch(run) and (not a.run_id or not os.environ.get("EL_RUN_ID") or a.run_id==os.environ["EL_RUN_ID"]),
                    "original native grade recorder identity differs")
    with tempfile.TemporaryDirectory(prefix="vision-native-grader-") as temporary:
        work=Path(temporary).resolve()
        packages,smoke_source,_=models_for_check(a,work,doc)
        for key,p in packages.items():
            p=p.resolve();index=regular(p,"model.json")
            rel.require(rel.sha(index)==doc["models"][key]["model_sha256"],"wrong fixed native-case model "+key)
            names=rel.L1 if key=="L1" else rel.NEURAL|{"model.json"}
            for n in names:rel.copy(work,"models/"+key+"/"+n,regular(p,n))
        rel.copy(work,"runner/smoke.py",smoke_source)
        for n in sorted(DATA_NAMES):rel.copy(work,"allowed-input/"+n,regular(a.case.resolve(),"input/"+n))
        for n in ("tmp","output","runner-home"):(work/n).mkdir(exist_ok=True)
        pins={k:v["model_sha256"] for k,v in doc["models"].items()}
        command=[sys.executable,"-I","-B",str(HERE),"audited-scoring","--work",str(work)]
        env={k:v for k,v in os.environ.items() if not k.startswith(("EL_","PYTHON"))}
        env.update(TEMP=str(work/"tmp"),TMPDIR=str(work/"tmp"),TMP=str(work/"tmp"),TORCH_HOME=str(work/"runner-home"))
        proc=subprocess.run(command,cwd=work,env=env,input=json.dumps(pins),text=True,capture_output=True)
        if proc.returncode:
            if any((work/"output").iterdir()):
                shutil.copytree(work/"output",output/"private-maps")
            rel.write(output,"failure.json",rel.entries({"schema":"el-vision-formal-native-failure-1","run_id":run,
                      "case_sha256":a.case_sha256,"phase":"truth-free-native-inference","returncode":proc.returncode}))
            (output/"private-stderr.log").write_text(proc.stderr,encoding="utf-8")
            raise ValueError("formal native inference failed; see private-stderr.log and failure.json")
        audit=json.loads(proc.stdout)
        try:
            rows=formal_compare(a.case/"expected",work/"output",doc["thresholds"])
        except (ValueError,KeyError,OSError) as exc:
            shutil.copytree(work/"output",output/"private-maps")
            rel.write(output,"failure.json",rel.entries({"schema":"el-vision-formal-native-failure-1","run_id":run,
                      "case_sha256":a.case_sha256,"phase":"independent-map-readback","error":str(exc)}))
            raise
        passed=all(r["passed"] for row in rows.values() for r in row.values())
        result={"schema":"el-vision-formal-native-equivalence-1","run_id":run,"passed":passed,"family":doc["family"],
                "trial_id":doc["trial_id"],"condition":"B0","split":"development","case_sha256":a.case_sha256,
                "method":doc["method"],"method_sha256":doc["method_sha256"],"catalog_sha256":doc["catalog_sha256"],
                "selection_sha256":doc["selection_sha256"],"reference":doc["reference"],"render":doc["render"],"references":doc["references"],
                "models":doc["models"],"grid":[512,512],"methods":list(METHODS),"views":rows,"runtime":audit["runtime"],
                "read_roles":audit["read_roles"],"data_access":audit["data_access"],"archive_sha256":a.archive_sha256,
                "release_sha256":a.release_sha256,"grader_sha256":rel.sha(HERE),"smoke_source_sha256":rel.sha(smoke_source)}
        rel.write(output,"native-validation.json",rel.entries(result))
        if not passed or a.keep_private_maps:shutil.copytree(work/"output",output/"private-maps")
        rel.require(passed,"native numerical, finite-support or threshold-state differences recorded in native-validation.json")
    return {"passed":True,"method_sha256":doc["method_sha256"],"family":doc["family"],"methods":3,"views":6,
            "checked_coordinates":512*512*18,"receipt_sha256":rel.sha(output/"native-validation.json")}


def record(a):
    import fcntl
    from vision_record import publish
    doc=fingerprint_case(a.case.resolve(),a.case_sha256)
    project=a.project.resolve();state=a.state.resolve();state.mkdir(exist_ok=True,parents=True)
    modes=doc["models"];parents={}
    for v in (doc["method"],doc["reference"],doc["render"],*doc["references"].values(),*modes.values()):
        old=parents.setdefault(v["run_id"],v["manifest_sha256"])
        rel.require(old==v["manifest_sha256"],"conflicting case parent seal")
    cfg={"config_version":"el-vision-release-native-1","experiment_id":"stage2-release-native",
         "trial_id":"stage2-release-native-"+doc["family"],"case_sha256":a.case_sha256,
         "identity":{"dataset_id":"evidence-lens-stage2","dataset_hash":doc["catalog_sha256"],"condition_id":"B0","split":"development"},
         "recording":{"expected_output_bytes":64*2**20},"parents":parents,
         "grader_sha256":rel.sha(project/"scripts/vision_release_native.py"),"smoke_sha256":rel.sha(project/"scripts/vision_public_smoke.py")}
    with (state/"lock").open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        publish(state/"config.json",cfg)
        def verified(pr):
            r=rel.Original(a.store,a.shelf,pr);d=r.json("payload/native-validation.json")
            rel.require(d["passed"] and d["case_sha256"]==a.case_sha256 and d["method_sha256"]==doc["method_sha256"]
                        and d["grader_sha256"]==cfg["grader_sha256"] and d["smoke_source_sha256"]==cfg["smoke_sha256"],
                        "selected grade differs from this case and source")
            return dict(r.pin,native_grade_sha256=rel.sha(r.file("payload/native-validation.json")))
        if (state/"selected.json").exists():return verified(rel.read(state/"selected.json"))
        logs=sorted(state.glob("attempt-*.log"))
        previous=None
        if logs:
            try:previous=json.loads(logs[-1].read_text().strip().splitlines()[-1])
            except (ValueError,IndexError) as exc:raise ValueError("inspect the interrupted recorded native grade") from exc
            if previous["execution_status"]=="succeeded":
                result=verified(previous);publish(state/"selected.json",result);return result
            rel.require(a.retry_of==previous["run_id"],"a prior failure requires its explicit --retry-of run ID")
        else:rel.require(not a.retry_of,"no failed native verification exists for this case")
        cmd=[sys.executable,"-m","evidence_lens.recorder","--root",str(a.store.resolve()),"--source-root",str(project),
             "--config",str(state/"config.json"),"--kind","archive_verify","--input","case="+str((a.case/"case.json").resolve())]
        paths={}
        for k,v in modes.items():
            r=rel.Original(a.store,a.shelf,v);prefix="model" if k=="L1" else "package"
            paths[k]=r.file("payload/"+prefix+"/model.json").parent
            role={"L1":"logistic","L2-frozen":"frozen","L2-vision-ft":"finetuned"}[k]
            cmd+=["--input",role+"="+str(paths[k]/"model.json"),"--input-parent",role+"="+v["run_id"]]
        for rid,seal in sorted(parents.items()):cmd+=["--parent",rid+"="+seal]
        if previous:cmd+=["--retry-of",previous["run_id"]]
        cmd+=["--",str(a.python.absolute()),"-I","-B","{source}/scripts/vision_release_native.py","check","--case",str(a.case.resolve()),
              "--case-sha256",a.case_sha256,"--case-manifest","{input:case}","--l1",str(paths["L1"]),"--frozen",str(paths["L2-frozen"]),
              "--ft",str(paths["L2-vision-ft"]),"--run-id","{run_id}","--output","{out}"]
        log=state/f"attempt-{len(logs)+1:02d}.log"
        with log.open("xb") as stream:
            p=subprocess.run(cmd,cwd=project,env=dict(os.environ,PYTHONPATH=str(project/"src")),stdout=stream,stderr=stream,pass_fds=(lock.fileno(),))
        out=json.loads(log.read_text().strip().splitlines()[-1])
        rel.require(p.returncode==0 and out["execution_status"]=="succeeded","recorded native verification failed: "+out["run_id"])
        result=verified(out);publish(state/"selected.json",result);return result


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    prep=sub.add_parser("prepare")
    for k in ("catalog","store","output"):prep.add_argument("--"+k,type=Path,required=True)
    prep.add_argument("--shelf",type=Path);prep.add_argument("--method",required=True)
    ck=sub.add_parser("check")
    for k in ("case","output"):ck.add_argument("--"+k,type=Path,required=True)
    for k in ("l1","frozen","ft","archive","case-manifest"):ck.add_argument("--"+k,type=Path)
    for k in ("archive-sha256","release-sha256","run-id"):ck.add_argument("--"+k)
    ck.add_argument("--case-sha256",required=True);ck.add_argument("--keep-private-maps",action="store_true")
    rec=sub.add_parser("record")
    for k in ("case","store","shelf","project","state","python"):rec.add_argument("--"+k,type=Path,required=True)
    rec.add_argument("--case-sha256",required=True);rec.add_argument("--retry-of")
    internal=sub.add_parser("audited-scoring");internal.add_argument("--work",type=Path,required=True)
    return p


if __name__=="__main__":
    if sys.version_info<(3,11):raise SystemExit("The native grade requires Python 3.11 or newer.")
    args=parser().parse_args()
    if args.action in ("prepare","record"):sys.path.insert(0,str(HERE.parents[1]/"src"))
    if args.action=="audited-scoring":result=audited_scoring(args.work.resolve(),json.load(sys.stdin))
    elif args.action=="prepare":result=prepared(args)
    elif args.action=="record":result=record(args)
    else:result=check(args)
    print(json.dumps(result,allow_nan=False,sort_keys=True))
