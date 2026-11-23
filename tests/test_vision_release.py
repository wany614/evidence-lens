"""Publication gates, bounded public content and fresh CPU execution on an engineering fixture."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/"scripts"))
import vision_public_smoke as smoke
import vision_release as release


def test_report_includes_all_cohorts_conditions_and_failed_rows():
    names={c:[f"{c}-{n}" for n in range(size)] for c,size in (("new_confirmation",8),("historical_regression",4))}
    methods=[*release.BASELINE,*release.BRANCHES]
    rows=[dict(cohort=c,scene_family_id=f,condition_id=cond,method=m,status="failed")
          for c,fs in names.items() for f in fs for cond in release.CONDITIONS for m in methods]
    method={"methods":methods,"test_family_ids":names["new_confirmation"]}
    assert release.report_tables(rows,method)["new_confirmation"]["family_condition_method_rows"]==168
    with pytest.raises(ValueError,match="duplicate"):
        release.report_tables(rows[:-1]+[rows[0]],method)
    with pytest.raises(ValueError,match="missing"):
        release.report_tables([dict(r,condition_id="invented") if i==0 else r for i,r in enumerate(rows)],method)
    pilot=SimpleNamespace(run=SimpleNamespace(run={"run_kind":"model_fit","experiment_id":"stage2-vision-pilot"}))
    with pytest.raises(ValueError,match="genuine formal"):
        release.model_proof("L2-vision-ft",pilot,{})


def test_output_comparison_and_deterministic_zip(tmp_path):
    for dest in ("a","b","result","bad"):
        (tmp_path/dest).mkdir()
    valid=np.ones((128,128),bool);valid[0,0]=False
    score=np.full(valid.shape,.25,np.float32);score[~valid]=np.nan
    state=np.where(valid,0,255).astype(np.uint8)
    values={"l0_reason":np.ones_like(state)}
    for key in smoke.METHODS.values():
        values.update({key:score,key+"_valid":valid,key+"_state":state})
    for v in smoke.VIEWS:
        np.savez_compressed(tmp_path/"a"/(v+".npz"),**values)
        shutil.copyfile(tmp_path/"a"/(v+".npz"),tmp_path/"b"/(v+".npz"))
    thresholds={m:{"tau":.5} for m in smoke.METHODS}
    result=smoke.compare(tmp_path/"a",tmp_path/"b",tmp_path/"result",thresholds)
    assert not result["failed"] and len(result["map_files"])==36 and sum(map(len,result["views"].values()))==42
    values["l2_vision_ft"]=score.copy();values["l2_vision_ft"][1,1]=.26
    np.savez_compressed(tmp_path/"b/v02.npz",**values)
    bad=smoke.compare(tmp_path/"a",tmp_path/"b",tmp_path/"bad",thresholds)
    assert bad["failed"]==["v02/L2-vision-ft"]
    for out in ("one.zip","two.zip"):
        release.archive_dir(tmp_path/"result",tmp_path/out)
    assert release.sig(tmp_path/"one.zip")==release.sig(tmp_path/"two.zip")
    wrong=tmp_path/"unapproved.zip";wrong.write_bytes(b"unit fixture with no source authorization")
    with pytest.raises(ValueError,match="approved public"):
        release.base_example(wrong,tmp_path/"bad-public-assets")


def test_smoke_checks_exact_package_members(tmp_path):
    manifest={"schema":"el-vision-public-release-1","version":release.TAG,"repository":"wany614/evidence-lens",
              "attribution":"wany614","scientific_status":"confirmed_heldout_report_recorded","files":{}}
    (tmp_path/"release.json").write_bytes(release.entries(manifest))
    (tmp_path/"SHA256SUMS").write_text(release.sha(tmp_path/"release.json")+"  release.json\n")
    smoke.check_assets(tmp_path)
    (tmp_path/"leftover.txt").write_text("foreign output")
    with pytest.raises(ValueError,match="extra=.*leftover.txt"):
        smoke.check_assets(tmp_path)


def test_real_authorized_plane_with_synthetic_trained_branches(tmp_path,monkeypatch):
    """Exercise a complete example grade; the temporary synthetic models cannot satisfy formal release lineage."""
    archive=os.getenv("EL_PUBLIC_RC3_ARCHIVE")
    if not archive:
        pytest.skip("set EL_PUBLIC_RC3_ARCHIVE to the published A ZIP to exercise its licensed example")
    torch=pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from test_vision_worker import make_data,preprocess_doc,training,Backbone,put,read
    archive=Path(archive).resolve()
    release.base_example(archive,tmp_path/"approved")
    assert {p.relative_to(tmp_path/"approved").as_posix() for p in (tmp_path/"approved").rglob("*") if p.is_file()}==release.EXAMPLE
    with zipfile.ZipFile(archive) as z:
        for n in release.L1:
            release.write(tmp_path,"L1/"+n,z.read("model/"+n))
    train=make_data(tmp_path/"train","train",16);dev=make_data(tmp_path/"dev","development",5)
    pre=tmp_path/"preprocessing.json";put(pre,preprocess_doc(release.sha(train)))
    base=tmp_path/"fixture-base.json";put(base,{"test_only":True})
    upstream=tmp_path/"fixture-upstream";upstream.mkdir();(upstream/"LICENSE").write_text("unit fixture only")
    def manufactured(*args):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(450)
            net=Backbone()
        return net,{"fixture":True,"base":{"checkpoint_sha256":"1"*64}}
    monkeypatch.setattr(training,"verified_base",manufactured)
    torch.set_num_threads(2)
    packages={"L1":tmp_path/"L1"}
    tau={"L0":{"tau":.5}}
    for m,t in read(packages["L1"]/"thresholds.json")["thresholds"].items():tau[m]={"tau":t}
    for i,b in enumerate(release.BRANCHES):
        run_id="20260101T00000000000"+str(i)+"Z-0123456789abcdef"
        cfg={"schema":"el-vision-run-1","phase":"train","branch":b,"seed":1729,"epochs":2,"patience":3,
             "head_lr":.0003,"backbone_lr":.00001 if i else 0,"weight_decay":.01,"microbatch":1,"accumulate":4,
             "precision":"float32","base_pin_sha256":release.sha(base),"preprocessing_sha256":release.sha(pre),
             "training_dataset_sha256":release.sha(train),"development_dataset_sha256":release.sha(dev)}
        c=tmp_path/(b+"-config.json");put(c,cfg)
        a=SimpleNamespace(output=tmp_path/b,device="cpu",config=c,base=base,base_pin=base,official_source=upstream,
                          train=train,development=dev,preprocessing=pre,lr_selection=None,resume=None,resume_sha256=None,run_id=run_id)
        result=training.run_fit(a,"train");assert result["phase"]=="train" and result["cpu_export"]["tensor_count"]==181
        packages[b]=a.output/"package";tau[b]={"tau":result["threshold"]}
    command=[sys.executable,"-I","-B",str(HERE/"scripts/vision_release.py"),"grade-example","--rc3-archive",str(archive),
             "--run-id","20260101T000000000008Z-0123456789abcdef","--output",str(tmp_path/"engineering-grade")]
    for opt,key in (("l1","L1"),("frozen",release.BRANCHES[0]),("ft",release.BRANCHES[1])):
        command.extend(["--"+opt,str(packages[key]),"--"+opt+"-sha256",release.sha(packages[key]/"model.json")])
    result=subprocess.run(command,text=True,capture_output=True,cwd=tmp_path,env=dict(os.environ,OMP_NUM_THREADS="2"))
    assert result.returncode==0,result.stderr
    assert set(read(tmp_path/"engineering-grade/deployment-example.json")["views"])==set(release.VIEWS)
    resultdir=tmp_path/"grade-compared";resultdir.mkdir()
    scored=tmp_path/"engineering-grade/example-scores"
    runtime=tmp_path/"synthetic-native-reader";runtime.mkdir()
    (tmp_path/"approved").rename(runtime/"example")
    shutil.copytree(scored,runtime/"expected",copy_function=os.link)
    for b,p in packages.items():
        shutil.copytree(p,runtime/"models"/b,copy_function=os.link)
    shutil.copyfile(HERE/"scripts/vision_public_smoke.py",runtime/"smoke.py")
    monkeypatch.setenv("OMP_NUM_THREADS","2")
    release.verify_native(runtime,resultdir,{n:{"model_sha256":release.sha(p/"model.json")} for n,p in packages.items()},tau)
    proof=read(resultdir/"build-smoke.json")
    assert not proof["failed"] and len(proof["map_files"])==36 and proof["runtime"]["device"]=="cpu"


def test_report_projection_requires_sealed_sources_and_denominator_conservation(tmp_path,monkeypatch):
    import csv
    repo=Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip())
    monkeypatch.syspath_prepend(str(repo/"scripts"))
    folders=tmp_path/"raw";folders.mkdir()
    methods=[*release.BASELINE,*release.BRANCHES]
    new=[f"new-{i}" for i in range(8)];old=[f"old-{i}" for i in range(4)]
    scope={"new_confirmation":new,"historical_regression":old}
    m={k:{"tau":.5,"domain":"development-B0"} for k in methods}
    decision={"default":"L1-full"}
    method={"catalog_sha256":"e"*64,"study_sha256":"9"*64,"models":{},"methods":methods,"test_family_ids":new,"thresholds":m,"deployment_rule":decision}
    def ref(i):
        return {"run_id":f"20260101T00000000000{i}Z-0123456789abcdef","manifest_sha256":str(i)*64}
    freeze=SimpleNamespace(pin=ref(1),file=lambda n:folders/"method-freeze.json")
    (folders/"method-freeze.json").write_bytes(release.entries(method))
    sha=release.sha(freeze.file("ignored"))
    cohort={c:{"families":f,"conditions":sorted(release.CONDITIONS),"planned_family_conditions":len(f)*3,"planned_views":len(f)*18,
               "evaluation":ref(i+3)} for i,(c,f) in enumerate(scope.items())}
    parents={p["run_id"]:p["manifest_sha256"] for p in [freeze.pin,*(v["evaluation"] for v in cohort.values())]}
    report=SimpleNamespace(pin=ref(2),run=SimpleNamespace(run={"run_kind":"report","experiment_id":"stage2-vision-confirmation-report"}),
                           file=lambda n:folders/Path(n).name,parents=lambda:parents)
    values={"study-summary.json":{"schema":"el-vision-public-study-1","method_sha256":sha,"method_freeze":{**freeze.pin,"sha256":sha},
            "run_id":report.pin["run_id"],"study_sha256":method["study_sha256"],"catalog_sha256":method["catalog_sha256"],"cohorts":cohort,"methods":methods},
            "selected-method.json":{"schema":"el-vision-public-method-1","method_sha256":sha,"default_method":"L1-full","thresholds":m,
            "models":{},"deployment_rule":decision},"execution-costs.json":{"fixture":True}}
    for n,v in values.items():(folders/n).write_bytes(release.entries(v))
    (folders/"paired-changes.csv").write_text("cohort,metric\n")
    rows=[dict(cohort=c,scene_family_id=f,condition_id=z,method=k,status="render_failed",views_planned=6,views_render_failed=6,
               views_rendered=0,V=36,E=0,H=0,L=0,U=0,Y=0,HY=0,LY=0,UY=0,S5=0,S7=0)
          for c,fs in scope.items() for f in fs for k in methods for z in release.CONDITIONS]
    def files():
        with (folders/"family-metrics.csv").open("w",newline="") as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        table={n:release.sig(folders/n) for n in release.REPORTS}
        (folders/"manifest.json").write_bytes(release.entries({"schema":"el-vision-public-numerical-1","method_sha256":sha,
                                                            "report_run_id":report.pin["run_id"],"files":table}))
    files()
    assert release.copy_report(tmp_path/"public",method,freeze,report)["historical_regression"]["family_condition_method_rows"]==84
    rows[0]["H"]=1;files()
    with pytest.raises(ValueError,match="RGB-error counts"):
        release.copy_report(tmp_path/"invalid",method,freeze,report)
