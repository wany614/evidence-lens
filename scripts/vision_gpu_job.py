"""Prepare and return one immutable recorder-backed vision training job between Mac and Windows."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE=Path(__file__).resolve()
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE.parents[1]/"src"))
CORE="60d73ccc89c36903ba0182de8922e3f6ba0a15a2"
RECORDER=("__init__","_bundle","_capacity","_dedup","_index","_metrics","_snapshot","_telemetry","_verify",
          "_worker","recorder","archive")
ROLES={"train":"payload/dataset.json","development":"payload/dataset.json","preprocessing":"payload/preprocessing.json",
       "base_pin":"payload/base.json","base_check":"payload/base-verification.json","lr_selection":"payload/selection.json"}


def require(ok, message):
    if not ok: raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_bytes())


def encoded(data):
    return (json.dumps(data,sort_keys=True,allow_nan=False,separators=(",",":"))+"\n").encode()


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()


def parse_pin(value):
    rid,sep,h=value.partition("=")
    require(sep and re.fullmatch(r"[0-9]{8}T[0-9]{12}Z-[a-f0-9]{16}",rid) and re.fullmatch("[a-f0-9]{64}",h),
            "use original run_id=manifest_sha256")
    return {"run_id":rid,"manifest_sha256":h}


def valid_path(name):
    require(isinstance(name,str) and all(re.fullmatch("[A-Za-z0-9_][A-Za-z0-9_.-]*",s) for s in name.split("/")),
            "nonportable relative file member")
    return name


def sig(p):
    return {"bytes":p.stat().st_size,"sha256":sha(p)}


def data_inventory(ds, split):
    require(ds["schema"]=="el-vision-dataset-1" and ds["split"]==split and split in ("train","development"),
            "training transport accepts train or development only")
    entries={}
    keys=set()
    for view in ds["views"]:
        ident=(view["family"],view["condition"],view["view"])
        require(ident not in keys and ident[1]=="B0" and re.fullmatch("v0[1-6]",ident[2]),"duplicate or ineligible study view")
        keys.add(ident)
        require((split=="train")== (view["full"] is None) and view["status"] in ("completed","render_failed")
                and (view["rgb"] is not None)==(view["status"]=="completed"),"wrong learning data partition")
        roles={"rgb":view["rgb"],"samples":view["samples"]}
        if split=="development":
            require(set(view["full"])=={"deploy","truth"} and (view["full"]["deploy"] is not None)==(view["status"]=="completed"),
                    "development requires separate features and truth")
            roles.update(view["full"])
        for role,entry in roles.items():
            if entry is None: continue
            n=valid_path(entry["path"])
            expected="/".join((*ident,"rgb.npy" if role=="rgb" else role+".npz"))
            require(n==expected and n not in entries and type(entry["bytes"]) is int and entry["bytes"]>0 and
                    re.fullmatch("[a-f0-9]{64}",entry["sha256"]),"invalid per-view file role or digest")
            entries[n]={k:entry[k] for k in ("bytes","sha256")}
    require(keys and all({v for f,c,v in keys if f==fam}=={f"v{i:02d}" for i in range(1,7)} for fam,_,_ in keys),
            "training dataset dropped a registered target")
    return entries


def new_dir(p):
    p.mkdir(parents=True,exist_ok=True)
    require(not p.is_symlink() and not any(p.iterdir()),"destination must be new or empty")


def code_payload(repo, worker):
    git=["git","-C",str(repo),"show"]
    names=["src/evidence_lens/"+k+".py" for k in RECORDER]+["scripts/worker_jobs.py","configs/storage.json"]
    payload={"code/"+n:subprocess.check_output([*git,CORE+":"+n]) for n in names}
    payload["code/scripts/vision_gpu_job.py"]=HERE.read_bytes()
    with zipfile.ZipFile(worker) as z:
        info=load_zip_json(z,"worker-source.json")
        require(info["schema"]=="el-vision-worker-source-1" and info["public_cpu_core_commit"]==CORE and
                set(z.namelist())==set(info["files"])|{"worker-source.json"},"worker source inventory differs")
        for n,meta in info["files"].items():
            valid_path(n)
            raw=z.read(n)
            require(len(raw)==meta["bytes"] and hashlib.sha256(raw).hexdigest()==meta["sha256"],"changed scientific worker "+n)
            if n.startswith(("src/","scripts/","configs/")):
                key="code/"+n
                require(key not in payload or payload[key]==raw,"scientific and recorder sources conflict: "+n)
                payload[key]=raw
    return payload


def load_zip_json(z,name):
    return json.loads(z.read(name))


def original(store,shelf,pin):
    from evidence_lens.formal_inputs import PinnedRun
    root=next((p for p in (store,shelf) if p and (p/"runs"/pin["run_id"]).is_dir()),None)
    require(root,"original parent is unavailable: "+pin["run_id"])
    r=PinnedRun(root,pin)
    require(r.run["execution_status"]=="succeeded","a required original parent did not succeed")
    return r


def prepare(a):
    from evidence_lens.scoring_io import new_output
    require(sha(a.worker)==a.worker_sha256 and sha(a.config)==a.config_sha256 and a.expected_output_bytes>0,
            "requested worker, config or output reservation differs")
    cfg=load(a.config);phase=cfg["phase"]
    require(cfg["schema"]=="el-vision-run-1" and phase in ("pilot","train"),"unregistered training stage")
    require(bool(a.development)==(phase=="train") and bool(a.lr_selection)==(cfg["seed"]!=1729),
            "unexpected or missing development/seed-A selection parent")
    rates=(str(cfg["backbone_lr"]) if cfg["branch"]=="L2-vision-ft" else str(cfg["head_lr"])).replace(".","p")
    trial=a.trial or f"stage2-{phase}-{cfg['branch'].lower()}-{cfg['seed']}-{rates}"
    require(re.fullmatch("[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}",trial),"invalid trial ID")
    output_dir=a.output
    with new_output(output_dir) as dest:
        inputs=dest/"inputs";inputs.mkdir()
        files={}
        def copy(n,source):
            path=dest/n;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,path);files[n]=sig(path)
        for name,data in code_payload(a.core_repo,a.worker).items():
            path=dest/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);files[name]=sig(path)
        copy("inputs/worker.zip",a.worker)
        copy("inputs/config.json",a.config)
        parents={};datasets={}
        for key,member in ROLES.items():
            value=getattr(a,key)
            if not value: continue
            pin=parse_pin(value)
            parent=original(a.store,a.shelf,pin)
            p=parent.open(member)
            expected_kind="dataset_build" if key in ("train","development") else "features" if key=="preprocessing" else "archive_verify" if key.startswith("base_") else None
            require(not expected_kind or parent.run["run_kind"]==expected_kind,"parent kind differs for "+key)
            payload=load(p)
            if key in ("train","development"):
                require(sha(p)==cfg["training_dataset_sha256" if key=="train" else "development_dataset_sha256"],
                        "dataset parent differs from configuration "+key)
                entries=data_inventory(payload,key)
                for name,expected in entries.items():
                    row=parent.files.get("payload/"+name,{})
                    require(all(row.get(k)==v for k,v in expected.items()),"original dataset manifest cannot supply "+name)
                datasets[key]={"manifest":sig(p),"files":entries,"views":len(payload["views"]),
                              "families":sorted({row["family"] for row in payload["views"]})}
            if key in ("preprocessing","base_pin","lr_selection"):
                require(sha(p)==cfg[key+"_sha256"],"original input differs from run configuration "+key)
            if key=="base_check":
                require(payload["schema"]=="el-vision-base-check-1" and payload["strict_equal_loaded_tensors"] is True and
                        payload["base_pin_sha256"]==cfg["base_pin_sha256"] and payload["parameters"]==22056576
                        and payload["tensors"]==175,"missing authenticated official base verification")
            copy("inputs/"+key+".json",p)
            parents[key]=dict(pin,member=member)
        base=load(inputs/"base_pin.json");check=load(inputs/"base_check.json")
        require(check["base"]==base,"independent base verification describes another original checkpoint")
        require(load(inputs/"preprocessing.json")["training_dataset_sha256"]==cfg["training_dataset_sha256"],
                "preprocessing uses another training dataset")
        if phase=="pilot":
            require(set(cfg["pilot_families"])==set(datasets["train"]["families"]) and len(cfg["pilot_families"])==4,
                    "pilot roster differs")
        job={"schema":"el-vision-gpu-job-1","phase":phase,"trial_id":trial,"experiment_id":"stage2-vision-"+phase,
             "core_commit":CORE,"worker_sha256":a.worker_sha256,"configuration_sha256":a.config_sha256,
             "datasets":datasets,"parents":parents,"files":files,"checkpoint_sha256":base["checkpoint_sha256"],
             "source_revision":base["revision"],"expected_output_bytes":a.expected_output_bytes}
        (dest/"job.json").write_bytes(encoded(job))
    return {"job":output_dir.name,"trial_id":trial,"job_sha256":sha(output_dir/"job.json")}


def check_job(root,expected):
    require(sha(root/"job.json")==expected,"job differs from the independent Mac dispatch")
    job=load(root/"job.json")
    require(job["schema"]=="el-vision-gpu-job-1" and job["phase"] in ("pilot","train"),"wrong GPU job schema")
    actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_symlink() or p.is_file()}
    require(actual==set(job["files"])|{"job.json"},"job contains missing or undeclared files")
    for n,entry in job["files"].items():
        p=root/valid_path(n)
        require(p.resolve().is_relative_to(root.resolve()) and not p.is_symlink() and sig(p)==entry,"changed input "+n)
    return job


def audit_dataset(root, manifest, expected):
    require(root.is_dir() and not root.is_symlink() and sig(root/"dataset.json")==expected["manifest"] and
            sig(manifest)==expected["manifest"],"external and recorded dataset manifests differ")
    names={"dataset.json",*expected["files"]}
    actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    require(actual==names,"external dataset has missing or extra members")
    for n,entry in expected["files"].items():
        p=root/valid_path(n)
        require(p.resolve().is_relative_to(root.resolve()) and not p.is_symlink() and p.is_file() and sig(p)==entry,
                "external dataset member differs: "+n)
    return {"dataset_sha256":expected["manifest"]["sha256"],"views":expected["views"],"files":len(names),
            "bytes":expected["manifest"]["bytes"]+sum(r["bytes"] for r in expected["files"].values()),
            "verified_asset_table_sha256":hashlib.sha256(encoded(expected["files"])).hexdigest()}


def scientific(a):
    # Large data is mounted read-only by the controller. The independent manifest and every file
    # are authenticated inside the recorded process before training. Originals stay in named parents.
    require(os.environ.get("EL_RUN_ID")==a.run_id,"explicit scientific run ID differs from recorder")
    job=load(a.job);cfg=load(a.config)
    require(sha(a.worker_zip)==job["worker_sha256"] and sha(a.config)==job["configuration_sha256"] and cfg["phase"]==job["phase"],
            "frozen worker/config differs from the Mac decision")
    out=a.output;out.mkdir(exist_ok=True,parents=True)
    checked={}
    for key in job["datasets"]:
        root=Path(os.environ["EL_VISION_"+key.upper()+"_ROOT"]).resolve()
        checked[key]=audit_dataset(root,getattr(a,key),job["datasets"][key])
    resume=None
    expected=os.environ.get("EL_VISION_RESUME_SHA")
    if expected:
        from evidence_lens._verify import verify_bundle
        run=load(out.parent/"run.json")
        retry=load(out.parent/"inputs.json")["retry_of"]
        require(retry and run["run_id"]==a.run_id and run["retry_of_run_id"]==retry["run_id"],"resume has no recorded retry edge")
        previous=Path(os.environ["EL_VISION_STORE"])/"runs"/retry["run_id"]
        old=verify_bundle(previous)
        require(old["manifest_sha256"]==retry["manifest_sha256"] and old["run"]["execution_status"]!="succeeded",
                "resume does not point to its immutable failed predecessor")
        candidates=[n for n in ("payload/training-state.pt","payload/fit/training-state.pt") if
                    n in old["artifacts"] and old["artifacts"][n]["sha256"]==expected]
        require(len(candidates)==1,"sealed retry predecessor does not contain the selected recovery state")
        resume=previous/candidates[0]
    report={"schema":"el-vision-gpu-input-audit-1","run_id":a.run_id,"job_sha256":sha(a.job),"datasets":checked,
            "parents":job["parents"],"worker_sha256":sha(a.worker_zip),"config_sha256":sha(a.config),
            "resume_sha256":expected}
    (out/"transport-audit.json").write_bytes(encoded(report))
    with tempfile.TemporaryDirectory(prefix="vision-runtime-") as tmp:
        runtime=Path(tmp)/"source";runtime.mkdir()
        with zipfile.ZipFile(a.worker_zip) as z:
            for info in z.infolist():
                n=valid_path(info.filename)
                require(not info.is_dir() and info.external_attr>>16 & 0o170000 in (0,0o100000),"invalid frozen worker member")
                p=runtime/n;p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as src,p.open("xb") as dst: shutil.copyfileobj(src,dst)
        work=out/"fit"
        train_root=Path(os.environ["EL_VISION_TRAIN_ROOT"]).resolve()
        cmd=[sys.executable,str(runtime/"scripts/vision_worker.py"),cfg["phase"],"--official-source",os.environ["EL_VISION_DINO_ROOT"],
             "--base",str(a.base),"--base-pin",str(a.base_pin),"--config",str(a.config),"--preprocessing",str(a.preprocessing),
             "--train",str(train_root/"dataset.json"),"--device","cuda","--run-id",a.run_id,"--output",str(work)]
        if a.development: cmd.extend(["--development",str(Path(os.environ["EL_VISION_DEVELOPMENT_ROOT"])/"dataset.json")])
        if a.lr_selection: cmd.extend(["--lr-selection",str(a.lr_selection)])
        if resume: cmd.extend(["--resume",str(resume),"--resume-sha256",expected])
        result=subprocess.run(cmd,cwd=runtime,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"))
        if work.is_dir():
            for p in work.iterdir():
                require(not (out/p.name).exists(),"unexpected training-output collision")
                p.rename(out/p.name)
            work.rmdir()
        return result.returncode


def run_windows(a):
    require(os.name=="nt","the scientific runner requires native Windows")
    root=a.job.resolve()
    job=check_job(root,a.job_sha256)
    require(HERE==(root/"code/scripts/vision_gpu_job.py").resolve(),"use the entry point frozen in this exact job")
    code=root/"code"
    sys.path.insert(0,str(code/"scripts"))
    from worker_jobs import frozen_json, matching_runs, outgoing_receipt
    from evidence_lens._index import writer_gate
    from evidence_lens._bundle import RUN_ID, utc
    from evidence_lens.recorder import worker_started
    store,outgoing=a.store.resolve(),a.return_dir.resolve()
    require(bool(a.development_root)==("development" in job["datasets"]) and a.train_root.is_dir() and a.official_source.is_dir(),
            "provide exact roots for the signed learning splits and official checkout")
    for path in (store,outgoing,root,a.train_root,a.official_source,a.base,*([a.development_root] if a.development_root else [])):
        require(path.resolve().drive.casefold()==root.drive.casefold(),"stage learning data, checkpoints and outputs on the research drive")
    for path in (store,outgoing): path.mkdir(exist_ok=True,parents=True)
    location=store/"vision-jobs"/job["trial_id"]
    location.mkdir(parents=True,exist_ok=True)
    with writer_gate(location):
        frozen_json(location/"job.json",{"trial_id":job["trial_id"],"sha256":a.job_sha256})
        cfg=load(root/"inputs/config.json")
        settings={"config_version":"vision-fit-1","formal_handoff_sha256":a.job_sha256,
                  "identity":{"dataset_id":"evidence-lens-stage2","dataset_hash":cfg["catalog_sha256"],"condition_id":"B0","split":"train"},
                  "phase":job["phase"],"neural":cfg,"scientific_source_sha256":job["worker_sha256"],"data_parents":job["parents"],
                  "recording":{"expected_output_bytes":job["expected_output_bytes"]}}
        settings_path=location/"recorder-config.json";frozen_json(settings_path,settings)
        found,pending=matching_runs(store,job,a.job_sha256,"model_fit",a.retry_of)
        require(not pending,"a current recording exists for this exact dispatch")
        if a.resume_sha256:
            require(a.retry_of and re.fullmatch("[a-f0-9]{64}",a.resume_sha256),"resume requires the original failed run and exact state digest")
        if a.retry_of:
            require(RUN_ID.fullmatch(a.retry_of),"invalid failed attempt")
            first=store/"runs"/a.retry_of
            failed=load(first/"run.json")
            require(failed["trial_id"]==job["trial_id"] and failed["experiment_id"]==job["experiment_id"] and
                    failed["run_kind"]=="model_fit" and failed["execution_status"]!="succeeded" and not failed["retry_of_run_id"]
                    and load(first/"config.resolved.yaml")["formal_handoff_sha256"]==a.job_sha256 and worker_started(first,failed),
                    "retry must name the original sealed scientific failure for this exact job")
        else:
            for path in found:
                linked,live=matching_runs(store,job,a.job_sha256,"model_fit",path.name)
                require(not live,"linked retry is still recording")
                success=[p for p in linked if load(p/"run.json")["execution_status"]=="succeeded"]
                require(len(success)<=1,"ambiguous successful retry")
                if success: return outgoing_receipt(store,outgoing,success[0],a.job_sha256)
        scientific_runs=[p for p in found if worker_started(p,load(p/"run.json"))]
        require(len(scientific_runs)<=1,"multiple scientific attempts for a single dispatch identity")
        if scientific_runs: return outgoing_receipt(store,outgoing,scientific_runs[0],a.job_sha256)
        if found and not a.after_preflight: return outgoing_receipt(store,outgoing,found[-1],a.job_sha256)
        if a.after_preflight:
            require(found and found[-1].name==a.after_preflight,"name the latest matching sealed preflight failure")
        label=(a.retry_of or "initial")+("-after-"+a.after_preflight if a.after_preflight else "")
        marker=location/(label+".json")
        require(not marker.exists(),"dispatch reserved without a sealed result; inspect the in-flight recorder before relaunch")
        args=[sys.executable,"-m","evidence_lens.recorder","--root",str(store),"--source-root",str(code),
              "--kind","model_fit","--config",str(settings_path),"--experiment",job["experiment_id"],"--trial",job["trial_id"]]
        for rid,h in sorted({(p["run_id"],p["manifest_sha256"]) for p in job["parents"].values()}):
            args.extend(["--parent",rid+"="+h])
        for role,pin in job["parents"].items():
            args.extend(["--input",role+"="+str(root/"inputs"/(role+".json")),"--input-parent",role+"="+pin["run_id"]])
        args.extend(["--input","job="+str(root/"job.json"),"--input","worker="+str(root/"inputs/worker.zip"),
                     "--input","train_config="+str(root/"inputs/config.json"),"--shared-input","checkpoint="+str(a.base.resolve()),
                     "--input-parent","checkpoint="+job["parents"]["base_check"]["run_id"]])
        if a.retry_of:args.extend(["--retry-of",a.retry_of])
        args.extend(["--","{python}","{source}/scripts/vision_gpu_job.py","scientific","--job","{input:job}","--worker-zip","{input:worker}",
                     "--train","{input:train}","--preprocessing","{input:preprocessing}","--base","{input:checkpoint}","--base-pin","{input:base_pin}",
                     "--config","{input:train_config}","--run-id","{run_id}","--output","{out}"])
        if a.development_root: args.extend(["--development","{input:development}"])
        if "lr_selection" in job["parents"]: args.extend(["--lr-selection","{input:lr_selection}"])
        env=dict(os.environ,EL_VISION_STORE=str(store),PYTHONPATH=str(code/"src"),PYTHONDONTWRITEBYTECODE="1",EL_VISION_TRAIN_ROOT=str(a.train_root.resolve()),
                 EL_VISION_DINO_ROOT=str(a.official_source.resolve()),CUBLAS_WORKSPACE_CONFIG=":4096:8")
        env.pop("EL_VISION_RESUME_SHA",None)
        if a.resume_sha256: env["EL_VISION_RESUME_SHA"]=a.resume_sha256
        if a.development_root: env["EL_VISION_DEVELOPMENT_ROOT"]=str(a.development_root.resolve())
        temp=store/"cache/vision-temp";temp.mkdir(parents=True,exist_ok=True)
        env.update(TEMP=str(temp),TMP=str(temp))
        frozen_json(marker,{"job_sha256":a.job_sha256,"retry_of":a.retry_of,"requested_at_utc":utc()})
        try:
            proc=subprocess.run(args,cwd=code,env=env,stdout=subprocess.PIPE,text=True)
        except OSError:
            marker.unlink()
            raise
        if not proc.stdout.strip():
            again,active=matching_runs(store,job,a.job_sha256,"model_fit",a.retry_of)
            if not active and again==found:
                marker.unlink()
            raise ValueError("recorder exited without a sealed result, exit "+str(proc.returncode))
        outcome=json.loads(proc.stdout)
        path=store/"runs"/outcome["run_id"]
        require(path.resolve()==Path(outcome["path"]).resolve() and sha(path/"manifest.json")==outcome["manifest_sha256"]
                and load(path/"config.resolved.yaml")["formal_handoff_sha256"]==a.job_sha256,"recorder returned an unrelated result")
        return outgoing_receipt(store,outgoing,path,a.job_sha256)


def receive_mac(a):
    from evidence_lens.archive import receive
    root=a.job.resolve();job=check_job(root,a.job_sha256)
    receipt=load(a.receipt)
    require(receipt["schema"]=="el-worker-return-1" and receipt["handoff_sha256"]==a.job_sha256
            and receipt["trial_id"]==job["trial_id"] and receipt["run_kind"]=="model_fit"
            and receipt["package"]==a.archive.name==receipt["run"]+".zip" and sha(a.archive)==receipt["sha256"]
            and a.archive.stat().st_size==receipt["bytes"],"return receipt/container differs from this job")
    with zipfile.ZipFile(a.archive) as z:
        meta=load_zip_json(z,"run.json");settings=load_zip_json(z,"config.resolved.yaml")
        require(meta["run_id"]==receipt["run"] and meta["execution_status"]==receipt["status"]
                and meta["experiment_id"]==job["experiment_id"] and meta["trial_id"]==job["trial_id"]
                and settings["formal_handoff_sha256"]==a.job_sha256 and
                hashlib.sha256(z.read("manifest.json")).hexdigest()==receipt["manifest_sha256"],"returned seal belongs to another experiment")
        if meta["execution_status"]=="succeeded":
            result=load_zip_json(z,"payload/training.json");audit=load_zip_json(z,"payload/transport-audit.json")
            require(result["schema"]=="el-vision-training-1" and result["phase"]==job["phase"] and result["run_id"]==meta["run_id"]
                    and result["training_config_sha256"]==job["configuration_sha256"] and result["frozen_identical"] is True
                    and result["task_gradient_minibatches"]>=3 and audit["job_sha256"]==a.job_sha256
                    and audit["parents"]==job["parents"] and result["base_checkpoint_sha256"]==job["checkpoint_sha256"],
                    "returned scientific evidence differs from the registered dispatch")
            for key,d in job["datasets"].items():
                checked=audit["datasets"][key]
                require(checked["dataset_sha256"]==d["manifest"]["sha256"] and
                        checked["verified_asset_table_sha256"]==hashlib.sha256(encoded(d["files"])).hexdigest(),
                        "external dataset asset-table validation differs")
    result=receive(a.archive,a.store,receipt["sha256"])
    require(result["status"] in ("accepted","identical"),"Mac canonical receipt: "+str(result))
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    def paths(q,required=(),optional=()):
        for n in required:q.add_argument("--"+n,type=Path,required=True)
        for n in optional:q.add_argument("--"+n,type=Path)
    def values(q,required=(),optional=()):
        for n in required:q.add_argument("--"+n,required=True)
        for n in optional:q.add_argument("--"+n)
    prep=sub.add_parser("prepare")
    paths(prep,("store","core-repo","worker","config","output"),("shelf",))
    values(prep,("worker-sha256","config-sha256","train","preprocessing","base-pin","base-check"),("development","lr-selection","trial"))
    prep.add_argument("--expected-output-bytes",type=int,default=256*2**20)
    run=sub.add_parser("run")
    paths(run,("job","store","return-dir","train-root","official-source","base"),("development-root",))
    values(run,("job-sha256",),("retry-of","after-preflight","resume-sha256"))
    inner=sub.add_parser("scientific")
    paths(inner,("job","worker-zip","config","train","preprocessing","base","base-pin","output"),("development","lr-selection"))
    values(inner,("run-id",))
    receive=sub.add_parser("receive")
    paths(receive,("job","store","receipt","archive"));values(receive,("job-sha256",))
    fast=sub.add_parser("verify-dispatch")
    paths(fast,("job",));values(fast,("job-sha256",))
    return p


def main():
    a=parser().parse_args()
    if a.action=="scientific": return scientific(a)
    if a.action=="prepare":data=prepare(a)
    elif a.action=="run":data=run_windows(a)
    elif a.action=="receive":data=receive_mac(a)
    else:
        job=check_job(a.job.resolve(),a.job_sha256)
        data={"job_sha256":a.job_sha256,"trial_id":job["trial_id"],"verified_files":len(job["files"])}
    print(json.dumps(data,sort_keys=True),flush=True)
    return int(data.get("exit_code",0))


if __name__=="__main__":
    raise SystemExit(main())
