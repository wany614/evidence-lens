"""Execute, seal and return a SHA-pinned DTU source-only optimizer and separate camera-verified render."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

sys.dont_write_bytecode=True
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/"src"))
from evidence_lens._bundle import encoded, files, load, local_path, sha
from evidence_lens._verify import require, verify_bundle
from worker_jobs import dispatch, frozen_json

SETS={"M2":["s00","s03"],"M4":["s00","s01","s02","s03"]}
TARGETS=[f"v{i:02d}" for i in range(1,7)]
OPT_SHA="2d1a704a8c93922326663edb3e27b047e05b1916bc47795d9a43771900fdf122"


def exact(path,entry):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size==entry["bytes"] and sha(path)==entry["sha256"],"Changed worker member: "+path.name)


def unpack(path,dest,entry):
    import hashlib
    exact(path,entry)
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        require("job.json" in names and len(names)==len(set(names)),"Invalid job members")
        raw=archive.read("job.json")
        require(hashlib.sha256(raw).hexdigest()==entry["job_sha256"],"Job identity changed")
        j=json.loads(raw)
        ids=SETS[j["condition"]]
        require(j["schema"]=="el-dtu-job-1" and j["experiment_id"]=="d1-dtu-source-only-256" and
                j["trial_id"]==entry["trial_id"] and j["generator_source_ids"]==ids and j["optimization_steps"]==256 and
                j["target_ids"]==TARGETS and j["object_id"] in ("set1","set6"),"Unregistered condition or budget")
        if j["object_id"]=="set6":
            require(j.get("opening") and len(j["opening_sha256"])==64,"No registered heldout opening")
        expected={"optimization.json","source_cameras.json","render_cameras.json",*("sources/"+s+".png" for s in ids)}
        require(set(names)==expected|{"job.json"} and set(j["files"])==expected and j["files"]["optimization.json"]["sha256"]==OPT_SHA,
                "Target image, unauthorized source or changed optimizer")
        for member in sorted(expected):
            dst=dest/local_path(member)
            dst.parent.mkdir(parents=True,exist_ok=True)
            dst.write_bytes(archive.read(member))
            exact(dst,j["files"][member])
        sc,rc=load(dest/"source_cameras.json")["views"],load(dest/"render_cameras.json")["views"]
        require(set(sc)==set(ids) and set(rc)==set(ids+TARGETS) and all(sc[x]==rc[x] for x in ids),"Different input and self-render cameras")
    return j


def launch(store,returns,j,stage,folder,jobsha,gen,retry,resume):
    reg,conv,budget=j["readback"],j["reference"],j["budget"]
    parents=[reg,conv,budget]
    if j.get("opening"):parents.append(j["opening"])
    cfg={**j["identity"],"experiment_id":j["experiment_id"],"trial_id":j["trial_id"],"formal_handoff_sha256":jobsha,
         "camera_plan_sha256":j["camera_plan_sha256"],"source_conversion":conv,"camera_registration":reg,"production_budget":budget,
         "opening":j.get("opening"),"opening_sha256":j.get("opening_sha256"),"generator_source_ids":j["generator_source_ids"],
         "source_rgb_sha256":{s:j["files"]["sources/"+s+".png"]["sha256"] for s in j["generator_source_ids"]},
         "source_camera_sha256":j["files"]["source_cameras.json"]["sha256"],"optimizer":load(folder/"optimization.json"),
         "optimizer_sha256":OPT_SHA,"optimization_steps":256,"worker_sha256":j["worker_sha256"],"recording":{"expected_output_bytes":19000000}}
    roles={"optimizer":(folder/"optimization.json",budget)}
    if stage=="generation":
        cmd=["{python}","{source}/src/evidence_lens/stage3_gaussians.py","optimize","--config","{config}","--out","{out}","--cameras","{input:cameras}"]
        roles["cameras"]=(folder/"source_cameras.json",reg)
        for s in j["generator_source_ids"]:
            roles[s]=(folder/"sources"/(s+".png"),conv)
            cmd+=["--source",s,"{input:"+s+"}"]
    else:
        require(gen and gen["status"]=="succeeded","No selected successful optimizer")
        pin={"run_id":gen["run"],"manifest_sha256":gen["manifest_sha256"]}
        generated=store/"runs"/pin["run_id"]
        verified=verify_bundle(generated)
        require(verified["manifest_sha256"]==pin["manifest_sha256"] and verified["run"]["trial_id"]==j["trial_id"] and
                load(generated/"config.resolved.yaml")["formal_handoff_sha256"]==jobsha,"Changed generating parent")
        parents.append(pin)
        payload=generated/"payload"
        cfg.update(reconstruction_run_id=pin["run_id"],reconstruction_manifest_sha256=pin["manifest_sha256"],
                   render_camera_sha256=j["files"]["render_cameras.json"]["sha256"])
        roles.update(cameras=(folder/"render_cameras.json",reg),checkpoint=(payload/"scene.npz",pin),reconstruction=(payload/"reconstruction.json",pin))
        cmd=["{python}","{source}/scripts/dtu_render.py","--config","{config}","--out","{out}","--cameras","{input:cameras}",
             "--checkpoint","{input:checkpoint}","--reconstruction","{input:reconstruction}"]
    cfgpath=store/"dtu-control"/j["trial_id"]/(stage+".json")
    cfgpath.parent.mkdir(parents=True,exist_ok=True)
    frozen_json(cfgpath,cfg)
    def execute():
        args=[sys.executable,"-m","evidence_lens.recorder","--root",str(store),"--source-root",str(CODE),"--storage-policy",str(CODE/"configs/storage.json"),
              "--kind",stage,"--config",str(cfgpath)]
        for pin in parents:args+=["--parent",pin["run_id"]+"="+pin["manifest_sha256"]]
        for n,(path,pin) in roles.items():args+=["--input",n+"="+str(path),"--input-parent",n+"="+pin["run_id"]]
        if retry:args+=["--retry-of",retry]
        proc=subprocess.run([*args,"--",*cmd],capture_output=True,text=True,env={**os.environ,"PYTHONPATH":str(CODE/"src"),"PYTHONDONTWRITEBYTECODE":"1"})
        if proc.stderr:print(proc.stderr,file=sys.stderr,flush=True)
        require(proc.stdout.strip(),"Recorder produced no stage receipt")
        return json.loads(proc.stdout.splitlines()[-1])
    return dispatch(store,returns,j,jobsha,stage,gen["run"] if gen else None,retry,resume,execute)


def run(a):
    require(os.name=="nt","DTU scientific execution requires Windows CUDA")
    root=CODE.parent
    batchpath=root/"batch.json"
    require(sha(batchpath)==a.batch_sha256,"Batch SHA differs")
    batch=load(batchpath)
    paths=files(CODE)
    require(batch["schema"]=="el-dtu-worker-batch-1" and set(paths)==set(batch["worker_files"]),"Different worker source")
    for name,entry in batch["worker_files"].items():exact(paths[name],entry)
    store,returns=a.store.resolve(),a.returns.resolve()
    require(store.drive and store.drive.casefold()==Path(sys.executable).drive.casefold()==root.drive.casefold(),"Cache, runner and store must use registered volume")
    cache=store/"dtu-job-cache";cache.mkdir(parents=True,exist_ok=True);returns.mkdir(parents=True,exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(CODE/"src"))
    results=[]
    for entry in batch["jobs"]:
        if a.trial and a.trial!=entry["trial_id"]:continue
        with tempfile.TemporaryDirectory(dir=cache) as work:
            dest=Path(work)
            j=unpack(root/local_path(entry["path"]),dest,entry)
            require(j["worker_sha256"]==batch["worker_sha256"] and j["camera_plan_sha256"]==batch["camera_plan_sha256"],"Wrong rig or worker")
            runs=[];gen=None
            for stage in ("generation","render"):
                retry=a.retry if a.trial and stage==a.stage else None
                resume=a.resume_preflight if a.trial and stage==a.stage else None
                result=launch(store,returns,j,stage,dest,entry["job_sha256"],gen,retry,resume)
                runs.append(result);print(json.dumps({"trial":j["trial_id"],"return":result}),flush=True)
                if result["status"]!="succeeded":break
                gen=result
            results.append({"trial_id":j["trial_id"],"job_sha256":entry["job_sha256"],"runs":runs})
    require(results,"No registered job matched")
    proof={"schema":"el-dtu-worker-completion-1","batch":batch["batch"],"batch_sha256":a.batch_sha256,"jobs":results}
    primary=returns/(batch["batch"]+"-worker-summary.json")
    if a.trial or primary.exists():
        import hashlib
        primary=returns/(batch["batch"]+"-"+hashlib.sha256(encoded(proof)).hexdigest()[:16]+".json")
    frozen_json(primary,proof)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-sha256",required=True)
    for n in ("store","returns"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--trial");p.add_argument("--stage",choices=("generation","render"));p.add_argument("--retry");p.add_argument("--resume-preflight")
    args=p.parse_args()
    if (args.retry or args.resume_preflight) and not(args.trial and args.stage):p.error("Retry/resume requires exact trial and stage")
    run(args)
