"""Run all seven released methods on the authorized authored plane, including the complete visual backbone."""
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
import time
import venv
import zipfile
from pathlib import Path

sys.dont_write_bytecode=True
VIEWS=tuple(f"v{i:02d}" for i in range(1,7))
BRANCHES=("L2-frozen","L2-vision-ft")
METHODS={"constant":"constant","1-alpha":"alpha_score","L0":"l0","L1-full":"l1_full","L1-geometry":"l1_geometry",
         "L2-frozen":"l2_frozen","L2-vision-ft":"l2_vision_ft"}
CONTROL_INPUTS={"input.png","cameras.json","predictions/prediction.json","predictions/source.npz",*("predictions/"+v+".npz" for v in VIEWS)}


def require(v,text):
    if not v: raise ValueError(text)


def sha(p):
    with Path(p).open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()


def read(p):return json.loads(Path(p).read_bytes())


def put(p,d):
    Path(p).write_text(json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def signature(p):return {"bytes":p.stat().st_size,"sha256":sha(p)}


def check_assets(root, expected=None):
    folder=Path(root)
    require(not expected or sha(folder/"release.json")==expected,"release.json differs from its external pin")
    doc=read(folder/"release.json")
    require(doc["schema"]=="el-vision-public-release-1" and doc["version"]=="v0.2.0-rc.1" and
            doc["repository"]=="wany614/evidence-lens" and doc["attribution"]=="wany614" and
            doc["scientific_status"]=="confirmed_heldout_report_recorded","unrecognized public model/evidence scope")
    names=set(doc["files"])|{"release.json","SHA256SUMS"}
    actual={p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() or p.is_symlink()}
    require(actual==names,f"release inventory differs: missing={sorted(names-actual)} extra={sorted(actual-names)}")
    records={}
    for name in sorted(names-{"SHA256SUMS"}):
        p=folder/name
        require(p.resolve().is_relative_to(folder.resolve()) and not p.is_symlink() and p.is_file(),"invalid package member "+name)
        records[name]=signature(p)
        if name!="release.json":
            require(records[name]==doc["files"][name],"changed package member "+name)
    sums="".join(records[n]["sha256"]+"  "+n+"\n" for n in sorted(records))
    require((folder/"SHA256SUMS").read_bytes()==sums.encode(),"internal checksum list differs")
    return doc


def unpack_archive(archive, expected, output, manifest_sha):
    import re
    require(re.fullmatch(r"[a-f0-9]{64}",expected) and sha(archive)==expected,"downloaded public model archive hash differs")
    require(re.fullmatch(r"[a-f0-9]{64}",manifest_sha),"supply the externally checked release manifest hash")
    output.mkdir()
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        require(names==sorted(set(names)) and len({n.casefold() for n in names})==len(names) and
                0<len(names)<2000 and sum(i.file_size for i in z.infolist())<1024**3,"invalid public ZIP inventory or size")
        for info in z.infolist():
            n=info.filename
            require(info.external_attr>>16==0o100644 and all(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9._-]*",k) for k in n.split("/")),
                    "unsafe release member "+n)
            p=output/n;p.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src,p.open("xb") as dst:
                import shutil
                shutil.copyfileobj(src,dst,2**20)
    return check_assets(output,manifest_sha)


def verify_archive(root,path,expected):
    require(path and expected and sha(path)==expected,"supply the independently verified ZIP hash")
    with zipfile.ZipFile(path) as z:
        actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        require(z.namelist()==sorted(actual),"ZIP member inventory differs")
        for n in z.namelist():
            with z.open(n) as stream:
                require(hashlib.file_digest(stream,"sha256").hexdigest()==sha(root/n),"extracted member differs: "+n)


def compute(example, models, pins, output, *, require_support=True):
    """Build original CPU scores from calibrated images and predictions, without reference images or labels."""
    import numpy as np
    import torch
    require(set(models)==set(pins)=={"L1",*BRANCHES},"three model packages are required")
    for name,root in models.items():
        require(sha(root/"model.json")==pins[name],"mismatched scientific model "+name)
    # The two signed B packages must deploy the same source. Load that source directly even for an isolated interpreter.
    runtimes=[models[b]/"runtime" for b in BRANCHES]
    scripts={p.relative_to(runtimes[0]).as_posix():sha(p) for p in runtimes[0].rglob("*.py")}
    require(scripts and scripts=={p.relative_to(runtimes[1]).as_posix():sha(p) for p in runtimes[1].rglob("*.py")},
            "frozen and adapted packages carry different CPU inference implementations")
    sys.path.insert(0,str(runtimes[0]))
    from evidence_lens.l1 import run as l1_score
    from evidence_lens_vision.deploy import open_model
    from evidence_lens_vision.native import one
    for module in ("evidence_lens.l1","evidence_lens_vision.deploy","evidence_lens_vision.native"):
        loaded=Path(sys.modules[module].__file__).resolve()
        require(loaded.is_relative_to(runtimes[0].resolve()),"CPU module was not loaded from the verified deployment package")
    output.mkdir(parents=True,exist_ok=True)
    source=example/"input.png";cameras=example/"cameras.json";pred=example/"predictions"
    with tempfile.TemporaryDirectory() as temp:
        tmp=Path(temp)
        l1=l1_score(source,cameras,pred,models["L1"],tmp/"l1",model_sha256=pins["L1"])
        require(l1["status"]=="scored" and all(l1["views"][v]["status"]=="scored" for v in VIEWS),
                "authored native example did not yield all L1 outputs")
        for b in BRANCHES:
            model,pre,index=open_model(models[b],pins[b],"cpu")
            require(index["branch"]==b and index["phase"]=="train" and index["threshold"] is not None,
                    "official formal branch and development threshold required")
            dest=tmp/b;dest.mkdir()
            for v in VIEWS:
                try:
                    info=one(model,pre,index,source,cameras,pred,v,dest/(v+".npz"))
                except (OSError,ValueError,RuntimeError,KeyError) as exc:
                    raise ValueError("native inference failed for "+b+"/"+v+": "+str(exc)) from exc
                require(info["status"]=="scored" and (not require_support or info["supported_pixels"]>0),"unscored or empty visual method/view "+b+"/"+v)
            del model
        for v in VIEWS:
            data={}
            with np.load(tmp/"l1"/(v+".npz"),allow_pickle=False) as z:
                for name,key in METHODS.items():
                    if name in BRANCHES: continue
                    for suffix in ("","_valid","_state"):
                        data[key+suffix]=z[key+suffix]
                data["l0_reason"]=z["l0_reason"] if "l0_reason" in z else z["reason_code"]
            for b in BRANCHES:
                with np.load(tmp/b/(v+".npz"),allow_pickle=False) as z:
                    for src,suffix in (("risk",""),("valid","_valid"),("state","_state")):
                        data[METHODS[b]+suffix]=z[src]
            np.savez_compressed(output/(v+".npz"),**data)
    return {"torch":str(torch.__version__),"numpy":np.__version__,"python":platform.python_version(),
            "architecture":platform.machine(),"system":platform.system(),"device":"cpu"}


def control_inputs(case, digest, work, registration):
    from PIL import Image
    root=case.resolve();index=root/"case.json"
    require(index.is_file() and not index.is_symlink() and re.fullmatch(r"[a-f0-9]{64}",digest) and sha(index)==digest,
            "public-plane case.json differs from its independent pin")
    doc=read(index)
    require(doc["schema"]=="el-vision-public-adapter-case-1" and doc["generator"]=="authored-two-metre-plane" and
            doc["family"]==registration["scene_family_id"]=="scene_037_bottle_carrier" and doc["condition"]==registration["condition_id"]=="public-plane" and
            doc["width"]==doc["height"]==512 and registration["plane_z_m"]==2.0 and registration["resolution_stride"]==1,
            "expected the registered native 512 public plane deployment control")
    require(set(doc["files"])==CONTROL_INPUTS,"public-plane deployment requires exactly ten registered inputs")
    entries={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    require(entries==CONTROL_INPUTS|{"case.json"},"additional or missing public-plane assets; keep evaluations in a sibling directory")
    for name in sorted(CONTROL_INPUTS):
        src=root/name;dest=work/name
        require(src.is_file() and src.resolve()==src,"invalid public-plane input "+name)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with src.open("rb") as inp,dest.open("xb") as out:shutil.copyfileobj(inp,out,2**20)
        require(signature(dest)==doc["files"][name],"public-plane input differs: "+name)
    meta=read(work/"predictions/prediction.json")
    require(meta["adapter"]==registration["adapter"] and meta["input_sha256"]==doc["files"]["input.png"]["sha256"] and
            meta["camera_sha256"]==doc["files"]["cameras.json"]["sha256"] and not meta["failed_views"] and
            set(meta["views"])=={"source",*VIEWS},"public-plane producer, inputs or views differ")
    for view,signature_entry in meta["views"].items():
        require(signature_entry==doc["files"]["predictions/"+view+".npz"],"public-plane producer hash differs: "+view)
    cameras=read(work/"cameras.json")["views"]
    require(set(cameras)=={"source",*VIEWS} and all((r["width"],r["height"])==(512,512) for r in cameras.values()),
            "public-plane camera dimensions differ")
    with Image.open(work/"input.png") as frame:require(frame.size==(512,512),"public-plane source dimensions differ")
    return doc


def previews(maps,view,method,risk,valid,state):
    import numpy as np
    from PIL import Image
    image=np.full(valid.shape,255,np.uint8)
    image[valid]=np.rint(254*np.clip(risk[valid],0,1).astype(np.float64)).astype(np.uint8)
    state_map=np.full((*state.shape,3),230,np.uint8)
    state_map[state==0]=(30,115,110)
    state_map[state==1]=(218,100,32)
    for kind,pixels in (("risk",image),("state",state_map)):
        Image.fromarray(pixels).save(maps/(view+"-"+method+"-"+kind+".png"))


def control_result(scores, out, thresholds):
    import numpy as np
    maps=out/"maps";maps.mkdir();views={};files={}
    names={k+s for k in METHODS.values() for s in ("","_valid","_state")}|{"l0_reason"}
    for v in VIEWS:
        f=scores/(v+".npz");rows={}
        with np.load(f,allow_pickle=False) as z:
            require(set(z.files)==names and z["l0_reason"].dtype==np.uint8 and z["l0_reason"].shape==(512,512),"invalid control output schema: "+v)
            for method,key in METHODS.items():
                x,valid,state=(z[key+s] for s in ("","_valid","_state"))
                require(x.dtype==np.float32 and valid.dtype==np.bool_ and state.dtype==np.uint8 and
                        x.shape==valid.shape==state.shape==(512,512) and np.array_equal(valid,np.isfinite(x)) and
                        np.isnan(x[~valid]).all() and ((x[valid]>=0)&(x[valid]<=1)).all(),"invalid control scores: "+v+"/"+method)
                tau=thresholds[method]["tau"]
                require((state[~valid]==255).all() and np.array_equal(state[valid],(x[valid].astype(np.float64)>=tau).astype(np.uint8)),
                        "incorrect frozen control state: "+v+"/"+method)
                rows[method]={"pixels":int(x.size),"finite":int(valid.sum()),"low":int((state==0).sum()),
                              "high":int((state==1).sum()),"unknown":int((state==255).sum()),"tau":tau}
                if method in ("L1-full",*BRANCHES):previews(maps,v,method,x,valid,state)
        views[v]=rows;files[f.name]=signature(f)
    return {"views":views,"score_files":files,"map_files":{p.name:signature(p) for p in sorted(maps.glob("*.png"))}}


def compare(expected,actual,out,thresholds=None):
    import numpy as np
    views={}
    failed=[]
    maps=out/"maps";maps.mkdir()
    for v in VIEWS:
        vals={}
        with np.load(expected/(v+".npz"),allow_pickle=False) as a,np.load(actual/(v+".npz"),allow_pickle=False) as b:
            require(set(a.files)==set(b.files)=={k+s for k in METHODS.values() for s in ("","_valid","_state")}|{"l0_reason"},
                    "unexpected numerical output schema for "+v)
            for method,key in METHODS.items():
                xs,ys=a[key],b[key]
                vx,vy=a[key+"_valid"],b[key+"_valid"]
                sx,sy=a[key+"_state"],b[key+"_state"]
                require(xs.shape==ys.shape==vx.shape==vy.shape==sx.shape==sy.shape==(128,128) and xs.dtype==ys.dtype==np.float32 and
                        vx.dtype==vy.dtype==np.bool_ and sx.dtype==sy.dtype==np.uint8,"invalid native array "+v+"/"+method)
                label=v+"/"+method
                require(np.array_equal(vx,np.isfinite(xs)) and np.array_equal(vy,np.isfinite(ys)) and
                        np.isnan(xs[~vx]).all() and np.isnan(ys[~vy]).all(),"mask differs from score contents "+label)
                for values,valid,state in ((xs,vx,sx),(ys,vy,sy)):
                    require(((values[valid]>=0)&(values[valid]<=1)).all() and (state[~valid]==255).all() and
                            (state[valid]<=1).all(),"risk or eligibility state out of range "+label)
                    if thresholds:
                        tau=thresholds[method]["tau"]
                        require(np.array_equal(state[valid],(values[valid].astype(np.float64)>=tau).astype(np.uint8)),
                                "operating state differs from frozen threshold "+label)
                mask_equal=np.array_equal(vx,vy)
                tol=.0002 if method in BRANCHES else .000001
                both=vx & vy
                maximum=float(np.max(np.abs(xs[both]-ys[both]))) if np.any(both) else None
                state_diff=int(np.count_nonzero(sx!=sy))
                reasons_equal=(method!="L0" or np.array_equal(a["l0_reason"],b["l0_reason"]))
                passed=mask_equal and maximum is not None and maximum<=tol and state_diff==0 and reasons_equal
                vals[method]={"finite_expected":int(vx.sum()),"finite_actual":int(vy.sum()),"finite_mask_equal":bool(mask_equal),
                              "max_absolute_score_difference":maximum,"absolute_tolerance":tol,"state_difference_count":state_diff,
                              "reason_codes_equal":bool(reasons_equal),"passed":bool(passed)}
                if not passed: failed.append(v+"/"+method)
                if method in ("L1-full",*BRANCHES):
                    previews(maps,v,method,ys,vy,sy)
        views[v]=vals
    require(len(list(maps.glob("*.png")))==36,"incomplete native preview output")
    return {"views":views,"failed":failed,"map_files":{p.name:signature(p) for p in sorted(maps.glob("*.png"))}}


def main():
    if sys.version_info<(3,11):raise SystemExit("Evidence Lens needs Python 3.11 or newer.")
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,default=Path(__file__).resolve().parent)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--release-sha256")
    p.add_argument("--archive",type=Path)
    p.add_argument("--archive-sha256")
    p.add_argument("--use-installed",action="store_true")
    p.add_argument("--wheelhouse",type=Path)
    p.add_argument("--case",type=Path,help="registered external 512 public-plane input directory")
    p.add_argument("--case-sha256",help="independently supplied SHA-256 of the external case.json")
    a=p.parse_args();root=a.bundle.resolve();out=a.output.resolve()
    require(bool(a.case)==bool(a.case_sha256),"supply both --case and --case-sha256")
    require(not out.exists() and not out.is_relative_to(root) and (not a.case or not out.is_relative_to(a.case.resolve())),
            "select a new output directory outside the bundle and original case")
    if not a.use_installed:
        with tempfile.TemporaryDirectory() as t:
            env=Path(t)/"venv";venv.EnvBuilder(with_pip=True).create(env)
            exe=env/("Scripts/python.exe" if os.name=="nt" else "bin/python")
            check_assets(root,a.release_sha256)
            install=[str(exe),"-m","pip","install","--disable-pip-version-check"]
            offline=["--no-index","--find-links",str(a.wheelhouse.resolve())] if a.wheelhouse else []
            if platform.system() in ("Linux","Windows") and not offline:
                subprocess.run([*install,"--index-url","https://download.pytorch.org/whl/cpu","torch==2.8.0"],check=True)
            subprocess.run([*install,*offline,"-r",str(root/"requirements.txt")],check=True)
            cmd=[str(exe),"-I","-B",str(root/"smoke.py"),"--bundle",str(root),"--output",str(out),"--use-installed"]
            for name,val in (("--release-sha256",a.release_sha256),("--archive",a.archive.resolve() if a.archive else None),
                             ("--archive-sha256",a.archive_sha256),("--case",a.case.resolve() if a.case else None),("--case-sha256",a.case_sha256)):
                if val:cmd.extend([name,str(val)])
            subprocess.run(cmd,check=True)
        return
    release=check_assets(root,a.release_sha256)
    if a.archive or a.archive_sha256:verify_archive(root,a.archive,a.archive_sha256)
    out.mkdir(parents=True)
    start=time.perf_counter()
    models={n:root/"models"/n for n in ("L1",*BRANCHES)}
    try:
        if a.case:
            registration=root/"configs/vision-public-control-v1.json";pins={n:release["models"][n]["model_sha256"] for n in models}
            with tempfile.TemporaryDirectory(prefix="evidence-lens-public-plane-") as temporary:
                isolated=Path(temporary);case=control_inputs(a.case,a.case_sha256,isolated,read(registration))
                runtime=compute(isolated,models,pins,out/"scores",require_support=False)
            summary=control_result(out/"scores",out,release["thresholds"])
            summary.update(schema="el-vision-public-control-scores-1",version=release["version"],release_sha256=sha(root/"release.json"),
                           archive_sha256=a.archive_sha256,case_sha256=a.case_sha256,registration_sha256=sha(registration),family=case["family"],
                           condition=case["condition"],generator=case["generator"],inputs=case["files"],models=pins,runtime=runtime,
                           scope="Separate authored-plane deployment control",status="scored",elapsed_seconds=time.perf_counter()-start)
            put(out/"control-scores.json",summary)
            print(json.dumps({"status":"scored","case_sha256":a.case_sha256,"control_scores_sha256":sha(out/"control-scores.json"),
                              "views":6,"methods":7,"pngs":len(summary["map_files"])},sort_keys=True));return
        runtime=compute(root/"example",models,{n:release["models"][n]["model_sha256"] for n in models},out/"scores")
        result=compare(root/"expected",out/"scores",out,release["thresholds"])
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        put(out/"failure.json",{"schema":"el-vision-public-smoke-failure-1","version":release["version"],
                                 "error":str(exc),"release_sha256":sha(root/"release.json"),"case_sha256":a.case_sha256})
        raise
    result.update(schema="el-vision-public-smoke-1",version=release["version"],release_sha256=sha(root/"release.json"),
                  archive_sha256=a.archive_sha256,runtime=runtime,elapsed_seconds=time.perf_counter()-start,passed=not result["failed"],
                  example="authored two-metre plane",study_scope="deployment reproduction")
    put(out/"smoke.json",result)
    require(result["passed"],"CPU smoke differences recorded for "+", ".join(result["failed"]))
    print(json.dumps({"passed":True,"views":len(result["views"]),"methods":len(METHODS),"pngs":len(result["map_files"]),
                      "smoke_sha256":sha(out/"smoke.json")},sort_keys=True))


if __name__=="__main__":
    main()
