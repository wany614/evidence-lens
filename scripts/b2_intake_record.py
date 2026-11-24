"""Record batch-two intake stages with explicit content pins across the three stores."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from evidence_lens._bundle import encoded,load,save,sha,utc
from evidence_lens._verify import verify_bundle
from t0_dtu_archive import check


def sealed_member(stores,run,member):
    for store in stores:
        base=store/"runs"/run
        if (base/"manifest.json").is_file():
            m=sha(base/"manifest.json")
            check(load(base/"completion.json")["manifest_sha256"]==m,"bad completion manifest")
            rows={r["path"]:r for r in load(base/"manifest.json")["files"]}
            check(member in rows and sha(base/member)==rows[member]["sha256"] and load(base/"run.json")["execution_status"]=="succeeded",
                  "missing successful sealed member")
            return {"run_id":run,"manifest_sha256":m,"member":member,"sha256":rows[member]["sha256"]}
    raise ValueError("missing recorded run "+run)


def locate(stores,pin):
    check(sealed_member(stores,pin["run_id"],pin["member"])==pin,"mismatched explicit run/member digest")
    return next(s/"runs"/pin["run_id"]/pin["member"] for s in stores if (s/"runs"/pin["run_id"]/"manifest.json").is_file())


def record_step(stores,name,worker,member,config,pins,options=None,worker_args=(),extra_code=(),estimate=1024**2,kind="report"):
    store=stores[0]; work=store/"work/controller"/name;work.mkdir(parents=True,exist_ok=True)
    srcfiles=["scripts/"+worker,"scripts/t0_dtu_archive.py","scripts/dtu_sample.py","scripts/dtu_gate.py",*extra_code]
    srcfiles=list(dict.fromkeys(srcfiles));src_hash={a:sha(ROOT/a) for a in srcfiles}
    original={k:locate(stores,v) for k,v in pins.items()}
    cfg={**config,"pinned":pins,"source_files":src_hash,"identity":{"experiment_id":"b2-dtu-calibration","trial_id":name},
         "recording":{"expected_output_bytes":estimate}}
    with (work/"lock").open("a+b") as lk:
        fcntl.flock(lk,fcntl.LOCK_EX)
        cfgpath=work/"config.json"
        if cfgpath.exists():check(cfgpath.read_bytes()==encoded(cfg),"conflicting frozen request")
        else:save(cfgpath,cfg)
        done=work/"selected.json"
        if done.exists():
            selected=load(done);locate(stores,selected);return selected
        logs=list(work.glob("attempt-*.json"))
        if logs:
            previous=load(sorted(logs)[-1])
            check(previous["execution_status"]=="succeeded","prior attempt failed; preserve record and identify a versioned recovery")
            selected=sealed_member(stores,previous["run_id"],member);save(done,selected);return selected
        with tempfile.TemporaryDirectory(dir=work,prefix="source-stage-") as scratch:
            source=Path(scratch)
            for name0 in srcfiles:
                file=source/name0;file.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(ROOT/name0,file)
                check(sha(file)==src_hash[name0],"source changed during scientific setup")
            cmd=[sys.executable,"-m","evidence_lens.recorder","--root",str(store),"--source-root",str(source),
                 "--storage-policy",str(ROOT/"configs/storage.json"),"--kind",kind,"--config",str(cfgpath)]
            for rid,manifest in sorted({v["run_id"]:v["manifest_sha256"] for v in pins.values()}.items()):
                cmd+=["--parent",rid+"="+manifest]
            for role,path in original.items():
                cmd+=["--input",role+"="+str(path),"--input-parent",role+"="+pins[role]["run_id"]]
            argv=["{python}","{source}/scripts/"+worker,*worker_args,"--config","{config}","--out","{out}",*(options or ())]
            proc=subprocess.run(cmd+["--",*argv],cwd=ROOT,env={**os.environ,"PYTHONPATH":str(ROOT/"src"),"PYTHONDONTWRITEBYTECODE":"1"},
                                capture_output=True,text=True)
            (work/"recorder.log").write_text(proc.stdout+proc.stderr)
            check(bool(proc.stdout.strip()),"recorder rejected setup: "+proc.stderr)
            result=json.loads(proc.stdout);save(work/"attempt-01.json",result)
            check(proc.returncode==0,"sealed scientific failure: "+result["run_id"]+"; "+str(result["failure_reason"]))
            selected=sealed_member(stores,result["run_id"],member);save(done,selected);return selected
