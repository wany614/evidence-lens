"""Run all seven released methods on the authorized authored plane, including the complete visual backbone."""
import argparse
import hashlib
import json
import os
import platform
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


def verify_archive(root,path,expected):
    require(path and expected and sha(path)==expected,"supply the independently verified ZIP hash")
    with zipfile.ZipFile(path) as z:
        actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        require(z.namelist()==sorted(actual),"ZIP member inventory differs")
        for n in z.namelist():
            with z.open(n) as stream:
                require(hashlib.file_digest(stream,"sha256").hexdigest()==sha(root/n),"extracted member differs: "+n)


def compute(example, models, pins, output):
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
                require(info["status"]=="scored" and info["supported_pixels"]>0,"empty visual method/view "+b+"/"+v)
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


def compare(expected,actual,out,thresholds=None):
    import numpy as np
    from PIL import Image
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
                    image=np.full(vy.shape,255,np.uint8)
                    image[vy]=np.rint(254*np.clip(ys[vy],0,1).astype(np.float64)).astype(np.uint8)
                    state_map=np.full((*sy.shape,3),230,np.uint8)
                    state_map[sy==0]=(30,115,110)
                    state_map[sy==1]=(218,100,32)
                    for kind,pixels in (("risk",image),("state",state_map)):
                        dest=maps/(v+"-"+method+"-"+kind+".png")
                        Image.fromarray(pixels).save(dest)
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
    a=p.parse_args();root=a.bundle.resolve();out=a.output.resolve()
    require(not out.exists() and not out.is_relative_to(root),"select a new output directory outside the bundle")
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
                             ("--archive-sha256",a.archive_sha256)):
                if val:cmd.extend([name,str(val)])
            subprocess.run(cmd,check=True)
        return
    release=check_assets(root,a.release_sha256)
    if a.archive or a.archive_sha256:verify_archive(root,a.archive,a.archive_sha256)
    out.mkdir(parents=True)
    start=time.perf_counter()
    models={n:root/"models"/n for n in ("L1",*BRANCHES)}
    try:
        runtime=compute(root/"example",models,{n:release["models"][n]["model_sha256"] for n in models},out/"scores")
        result=compare(root/"expected",out/"scores",out,release["thresholds"])
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        put(out/"failure.json",{"schema":"el-vision-public-smoke-failure-1","version":release["version"],
                                 "error":str(exc),"release_sha256":sha(root/"release.json")})
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
