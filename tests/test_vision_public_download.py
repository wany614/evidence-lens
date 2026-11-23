import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE/"scripts"))
import vision_release as rel
import vision_release_native as native
import verify_vision_public as public


class PublicBytes(io.BytesIO):
    def geturl(self):return "https://release-assets.githubusercontent.com/readback"


class Reader:
    def __init__(self,raw):self.raw=raw
    def open(self,req,timeout):
        assert "Authorization" not in req.headers and "Cookie" not in req.headers
        return PublicBytes(self.raw)


def test_anonymous_asset_digest_and_unchanged_failure_evidence(tmp_path):
    data=b"small authorized unit asset"
    spec={"name":"unit.bin","browser_download_url":"https://github.com/wany614/evidence-lens/releases/download/unit/unit.bin",
          "size":len(data),"digest":"sha256:"+hashlib.sha256(data).hexdigest(),"id":123}
    pin={"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
    assert public.download(Reader(data),spec,pin,tmp_path)["sha256"]==pin["sha256"]
    bad=tmp_path/"bad";bad.mkdir()
    with pytest.raises(ValueError,match="independent file pin"):
        public.download(Reader(data),spec,{**pin,"sha256":"d"*64},bad)
    assert (bad/"unit.bin.partial").read_bytes()==data and not (bad/"unit.bin").exists()
    with pytest.raises(ValueError,match="production B"):
        public.same_candidate({"schema":"synthetic_fixture"},rel.TAG,"f"*40)


def test_native_512_whole_grid_threshold_crossing(tmp_path):
    expected=tmp_path/"expected";actual=tmp_path/"outputs";actual.mkdir()
    risk=np.zeros((512,512),np.float32);risk[0,0]=np.nan;risk[0,1]=.5
    valid=np.isfinite(risk);state=np.where(valid,risk>=.5,255).astype(np.uint8)
    tau={m:{"tau":.5} for m in native.METHODS}
    for m in native.METHODS:
        (expected/m).mkdir(parents=True)
        for v in native.VIEWS:np.savez_compressed(expected/m/(v+".npz"),risk=risk,valid=valid,state=state)
    pred={}
    for k in native.KEYS.values():pred.update({k:risk,k+"_valid":valid,k+"_state":state})
    for v in native.VIEWS:np.savez_compressed(actual/(v+".npz"),**pred)
    rv=native.formal_compare(expected,actual,tau)
    assert sum(c["pixels"] for d in rv.values() for c in d.values())==4718592
    assert all(c["passed"] for d in rv.values() for c in d.values())
    k=native.KEYS["L2-vision-ft"];pred[k]=risk.copy();pred[k][0,1]=np.nextafter(np.float32(.5),np.float32(0))
    pred[k+"_state"]=state.copy();pred[k+"_state"][0,1]=0
    np.savez_compressed(actual/"v06.npz",**pred)
    rv=native.formal_compare(expected,actual,tau)
    assert not rv["v06"]["L2-vision-ft"]["passed"] and rv["v06"]["L2-vision-ft"]["threshold_state_difference"]==1
    assert rv["v06"]["L2-vision-ft"]["max_abs"]<.0002


def test_unlisted_truth_is_refused_before_model_io(tmp_path):
    root=tmp_path/"case";root.mkdir()
    names={"input/"+n for n in native.DATA_NAMES}|{"expected/"+m+"/"+v+".npz" for m in native.METHODS for v in native.VIEWS}
    for n in names:rel.write(root,n,b"unit file")
    doc={"schema":"el-vision-native-release-case-1","condition":"B0","split":"development","shape":[512,512],
         "views":list(native.VIEWS),"methods":list(native.METHODS),"models":{k:{"run_id":"20260101T000000000001Z-0123456789abcdef",
         "manifest_sha256":"b"*64,"model_sha256":"a"*64} for k in ("L1",*rel.BRANCHES)},"files":{n:rel.sig(root/n) for n in names}}
    rel.write(root,"case.json",rel.entries(doc));digest=rel.sha(root/"case.json")
    assert native.fingerprint_case(root,digest)==doc
    (root/"input/ground_truth.npz").write_bytes(b"never export")
    with pytest.raises(ValueError,match="undeclared assets"):
        native.fingerprint_case(root,digest)


def test_private_deploy_read_barrier_applies_at_import(tmp_path):
    work=tmp_path/"work";runner=work/"runner";runner.mkdir(parents=True)
    for n in ("models","output","tmp"):(work/n).mkdir()
    for n in native.DATA_NAMES:rel.write(work/"allowed-input",n,b"authored fixture only")
    leak=tmp_path/"truth.npy";np.save(leak,np.ones(4))
    (runner/"smoke.py").write_text("open("+repr(str(leak))+",'rb').read()\n")
    p=subprocess.run([sys.executable,"-I","-B",str(HERE/"scripts/vision_release_native.py"),"audited-scoring","--work",str(work)],
                     input="{}",text=True,capture_output=True,cwd=work)
    assert p.returncode and "outside isolated deployment inputs: truth.npy" in p.stderr


def test_install_uses_new_offline_venv(tmp_path,monkeypatch):
    for name in ("HTTP_PROXY","https_proxy","ALL_PROXY","REQUESTS_CA_BUNDLE","SSL_CERT_FILE","PIP_INDEX_URL","PIP_CONFIG_FILE","NETRC"):
        monkeypatch.setenv(name,"invalid-inherited-value")
    wh=tmp_path/"wheels";wh.mkdir();requirements=tmp_path/"requirements.txt";requirements.write_text("vision-grade-fixture==0.0.1\n")
    di="vision_grade_fixture-0.0.1.dist-info/"
    with zipfile.ZipFile(wh/"vision_grade_fixture-0.0.1-py3-none-any.whl","w") as z:
        values={"vision_grade_fixture.py":b"VALUE=42\n",di+"METADATA":b"Metadata-Version: 2.1\nName: vision-grade-fixture\nVersion: 0.0.1\n",
                di+"WHEEL":b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
        values[di+"RECORD"]=("".join(n+",,\n" for n in [*values,di+"RECORD"])).encode()
        for n,v in values.items():z.writestr(n,v)
    exe=public.install(tmp_path,requirements,wh)
    out=subprocess.check_output([str(exe),"-I","-c","import vision_grade_fixture as f;print(f.VALUE)"],text=True)
    assert out.strip()=="42" and exe.is_relative_to(tmp_path/"clean-cpu")
    receipt=rel.read(tmp_path/"installation.json");transaction=receipt["transactions"][0];wheel=next(wh.glob('*.whl'))
    assert receipt['source_mode']=='offline_local_wheelhouse' and receipt['installed_distributions']['vision-grade-fixture']=='0.0.1'
    assert transaction['wheels']==[{'distribution':'vision-grade-fixture','version':'0.0.1','requested':True,'wheel':wheel.name,'sha256':rel.sha(wheel),'origin':'local wheelhouse'}]
    assert transaction['pip_report_sha256']==rel.sha(tmp_path/'pip-report-0.json')


def test_installer_environment_and_scope_stay_distinct(tmp_path,monkeypatch):
    for name in ('HTTP_PROXY','https_proxy','ALL_PROXY','NO_PROXY','REQUESTS_CA_BUNDLE','CURL_CA_BUNDLE','SSL_CERT_FILE','SSL_CERT_DIR','PIP_CERT','PIP_INDEX_URL','GITHUB_TOKEN'):
        monkeypatch.setenv(name,'never-inherit')
    monkeypatch.setenv('HOME',str(tmp_path/'existing-profile'))
    env=public.installer_environment(tmp_path)
    assert 'never-inherit' not in env.values() and env['HOME']==env['USERPROFILE'] and not list(Path(env['HOME']).iterdir())
    assert env['NETRC']==env['PIP_CONFIG_FILE']==public.os.devnull and 'HTTPS_PROXY' not in env
    assert env['NO_PROXY']==env['no_proxy']=='*'
    assert public.verification_scope(None)[0]=='verified_builtin_only'
    with pytest.raises(ValueError,match='requires the public 512'):public.verification_scope(None,True)
    with pytest.raises(ValueError,match='incomplete'):public.verification_scope({'method_views':42,'coordinates':42*512**2})
    assert public.verification_scope({'independent_evaluation_sha256':'4'*64,'method_views':42,'coordinates':42*512**2},True)[0]=='verified_full'


def test_online_wheel_record_requires_clean_origin_and_download_digest(tmp_path):
    p=tmp_path/'report.json';record={'metadata':{'name':'example','version':'1.0'},'requested':True,
         'download_info':{'url':'https://files.pythonhosted.org/sample/example-1.0-py3-none-any.whl','archive_info':{'hashes':{'sha256':'f'*64}}}}
    def save():p.write_text(json.dumps({'version':'1','pip_version':'26.0','install':[record]}))
    save();r=public.installed_wheels(p,'https://pypi.org/simple',None)
    assert r['wheels'][0]['origin']=='files.pythonhosted.org' and r['pip_report_sha256']==rel.sha(p)
    record['download_info']['url']='https://unregistered.invalid/example-1.0-py3-none-any.whl';save()
    with pytest.raises(ValueError,match='unexpected download host'):public.installed_wheels(p,'https://pypi.org/simple',None)


def test_external_body_uses_actual_cohorts_failures_and_frozen_default():
    rows={"new_confirmation":{"families":["s1","s2"],"planned_family_conditions":6,"planned_views":36,
                              "render_status":{"completed":35,"render_failed":1},"scientific_view_dispositions":{"evaluated":35,"terminal_upstream_failure":1}},
          "historical_regression":{"families":["old"],"planned_family_conditions":3,"planned_views":18,
                               "render_status":{"completed":18},"scientific_view_dispositions":{"evaluated":18}}}
    for cohort in rows.values():
        n=len(cohort["families"])
        rate={"value":.5,"defined_families":n,"total_families":n}
        cohort["family_equal"]=[{"condition":"B0","cohort":"all","methods":{
            m:{key:rate for key in ("seven_common_ap","low_coverage","low_risk_error")}
            for m in ("L1-full",*rel.BRANCHES)}}]
    outcome={"default_method":"L1-full","code_commit":"e"*40,"models":{"L1":{"run_id":"unit-source","model_sha256":"d"*64}}}
    costs={"observed_runs":[{"run_id":"explicit-failed-run","run_kind":"render","status":"failed"}]}
    notes=rel.release_notes(outcome,{"cohorts":rows},costs,{"unit.zip":{"bytes":123,"sha256":"d"*64}})
    for text in ("**L1-full**","New confirmation | 2 | 6 | 36","render_failed 1","terminal_upstream_failure 1","explicit-failed-run","Historical regression | 1 | 3 | 18","unit.zip | 123"):
        assert text in notes


def test_separate_public_plane_context_and_full_map_receipt(tmp_path):
    import vision_public_smoke as smoke
    from PIL import Image
    case=tmp_path/"case";case.mkdir();(case/"predictions").mkdir()
    Image.new("RGB",(512,512)).save(case/"input.png")
    layout={v:{"width":512,"height":512} for v in ("source",*smoke.VIEWS)}
    smoke.put(case/"cameras.json",{"views":layout})
    for view in layout:(case/"predictions"/(view+".npz")).write_bytes(view.encode())
    producer={"adapter":"evidence-lens-textured-plane-1","input_sha256":smoke.sha(case/"input.png"),
              "camera_sha256":smoke.sha(case/"cameras.json"),"failed_views":{},
              "views":{v:smoke.signature(case/"predictions"/(v+".npz")) for v in layout}}
    smoke.put(case/"predictions/prediction.json",producer)
    reg=dict(plane_z_m=2.,resolution_stride=1,adapter=producer["adapter"],condition_id="public-plane",scene_family_id="scene_037_bottle_carrier")
    doc=dict(schema="el-vision-public-adapter-case-1",family=reg["scene_family_id"],condition="public-plane",generator="authored-two-metre-plane",
             width=512,height=512,files={n:smoke.signature(case/n) for n in smoke.CONTROL_INPUTS})
    smoke.put(case/"case.json",doc);digest=smoke.sha(case/"case.json")
    assert smoke.control_inputs(case,digest,tmp_path/"allowed",reg)==doc
    zipped=tmp_path/"external-control.zip"
    with zipfile.ZipFile(zipped,"w") as z:
        for n in sorted(smoke.CONTROL_INPUTS|{"case.json"}):z.write(case/n,"study/control/"+n)
        z.writestr("public-reference/separate-depth.npz",b"separate evaluation")
    assert public.extract_control(zipped,digest,tmp_path/"checked-attachment")==doc
    assert {p.relative_to(tmp_path/"checked-attachment").as_posix() for p in (tmp_path/"checked-attachment").rglob("*") if p.is_file()}==smoke.CONTROL_INPUTS|{"case.json"}
    with pytest.raises(ValueError,match="independent pin"):public.extract_control(zipped,"b"*64,tmp_path/"bad-extraction")
    for key,bad in (("condition","B0"),("generator","SHARP"),("width",128)):
        smoke.put(case/"case.json",{**doc,key:bad})
        with pytest.raises(ValueError,match="registered native"):
            smoke.control_inputs(case,smoke.sha(case/"case.json"),tmp_path/"unused",reg)
    smoke.put(case/"case.json",doc)
    (case/"truth.npz").write_bytes(b"separate reference")
    with pytest.raises(ValueError,match="sibling directory"):smoke.control_inputs(case,digest,tmp_path/"unused",reg)
    (case/"truth.npz").unlink();(case/"predictions/v01.npz").write_bytes(b"changed")
    with pytest.raises(ValueError,match="input differs"):smoke.control_inputs(case,digest,tmp_path/"change",reg)
    # A real missing score remains unknown while every valid coordinate uses the signed cutoff.
    scored=tmp_path/"scores";scored.mkdir();out=tmp_path/"result";out.mkdir()
    x=np.full((512,512),.5,np.float32);x[0,0]=np.nan;valid=np.isfinite(x)
    arr={"l0_reason":np.zeros(x.shape,np.uint8)}
    for k in smoke.METHODS.values():arr.update({k:x,k+"_valid":valid,k+"_state":np.where(valid,1,255).astype(np.uint8)})
    for view in smoke.VIEWS:np.savez_compressed(scored/(view+".npz"),**arr)
    result=smoke.control_result(scored,out,{m:{"tau":.5} for m in smoke.METHODS})
    assert len(result["map_files"])==36 and len(result["score_files"])==6
    assert sum(len(r) for r in result["views"].values())==42 and result["views"]["v06"]["L2-vision-ft"]["high"]==512**2-1


def test_native_proof_sources_and_real_full_pixel_anchor(tmp_path,monkeypatch):
    repo=Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip())
    monkeypatch.syspath_prepend(str(repo/"scripts"))
    class Input:
        def __init__(self,n,kind,exp="unit",family="fixture-dev"):
            self.pin={"run_id":f"20260101T0000000000{n:02d}Z-0123456789abcdef","manifest_sha256":f"{n:064x}"}
            class Run:pass
            self.run=Run();self.run.run={"run_kind":kind,"experiment_id":exp,"scene_family_id":family,"condition_id":"B0","split":"development"}
            self.root=tmp_path/self.pin["run_id"];self.root.mkdir();self.ps={}
        def parents(self):return self.ps
        def file(self,n):return self.root/n
        def json(self,n):return rel.read(self.file(n))
    orig=Input(1,"archive_verify","stage2-release-native");ref=Input(2,"scene_render");ren=Input(3,"render")
    baseline=Input(4,"model_score");f=Input(5,"model_score");ft=Input(6,"model_score");joint=Input(7,"report")
    lookup={x.pin["run_id"]:x for x in (orig,ref,ren,baseline,f,ft,joint)}
    get=lambda spec:lookup[spec["run_id"]]
    ms={name:dict(Input(i,"threshold_select" if i==8 else "model_fit").pin,model_sha256=f"{i:064x}") for i,name in enumerate(("L1",*rel.BRANCHES),8)}
    method={"catalog_sha256":"b"*64,"development_dataset_sha256":"d"*64,"cpu_costs":{b:{"pin":r.pin} for b,r in zip(rel.BRANCHES,(f,ft))}}
    roster=rel.write(joint.root,"roster.json",rel.entries({"stages":{"baseline_scores":baseline.pin}}))
    method["input_roster_sha256"]=rel.sha(roster)
    rel.write(joint.root,"inputs.json",rel.entries({"inputs":[{"role":"method_roster","path":"roster.json"}]}))
    meth=rel.write(joint.root,"payload/method-freeze.json",rel.entries(method))
    dest=tmp_path/"public"
    rel.write(dest,"protocol/stage2-families-v1.json",rel.entries({"families":[{"scene_family_id":"fixture-dev","split":"development"}]}))
    models={n:{k:s[k] for k in ("run_id","manifest_sha256","model_sha256")} for n,s in ms.items()}
    pins={"L1-full":baseline.pin,rel.BRANCHES[0]:f.pin,rel.BRANCHES[1]:ft.pin}
    shared={"family":"fixture-dev","trial_id":"fixture-dev-B0","condition":"B0","split":"development","models":models,
            "catalog_sha256":"b"*64,"method":joint.pin,"method_sha256":rel.sha(meth),"references":pins,
            "reference":ref.pin,"render":ren.pin,"selection_sha256":"0"*64}
    inputs={};spec={}
    for n in native.DATA_NAMES:
        src=ren if n.startswith("predictions/") else ref
        p=rel.write(src.root,"payload/"+n.removeprefix("predictions/"),n.encode())
        inputs[n]=rel.sig(p);spec["input/"+n]=inputs[n]
    bviews=[];views={};grid=np.zeros((512,512),np.float32);gsha=hashlib.sha256(grid.tobytes()).hexdigest()
    for v in native.VIEWS:
        vb={}
        for m,run in (("L1-full",baseline),(rel.BRANCHES[0],f),(rel.BRANCHES[1],ft)):
            item=("fixture-dev/B0/" if m=="L1-full" else "cold/")+v+".npz"
            p=run.root/"payload"/item;p.parent.mkdir(exist_ok=True,parents=True)
            np.savez_compressed(p,**{("l1_full" if m=="L1-full" else "risk"):grid})
            if m=="L1-full":bviews.append({"view":v,"family":"fixture-dev","condition":"B0","score":dict(path=item,**rel.sig(p))})
            else:spec["expected/"+m+"/"+v+".npz"]=rel.sig(p)
            vb[m]={"pixels":512**2,"finite_expected":512**2,"finite_actual":512**2,"finite_common":512**2,"passed":True,"support_difference":0,
                   "threshold_state_difference":0,"absolute_tolerance":.000001 if m=="L1-full" else .0002,"max_abs":0.,"mean_abs":0.,
                   "expected_risk_sha256":gsha,"actual_risk_sha256":gsha}
        views[v]=vb
    sol={"method":"original-five","model_sha256":models["L1"]["model_sha256"],"dataset_sha256":method["development_dataset_sha256"],"views":bviews}
    rel.write(baseline.root,"payload/scores.json",rel.entries(sol))
    case=rel.write(orig.root,"case.json",rel.entries(dict(schema="el-vision-native-release-case-1",files=spec,**shared)))
    rel.write(orig.root,"inputs.json",rel.entries({"inputs":[{"role":"case","path":"case.json"}]}))
    source={"scripts/vision_release_native.py":{"sha256":"a"*64},"scripts/vision_public_smoke.py":{"sha256":"e"*64}}
    proof=dict(shared,schema="el-vision-formal-native-equivalence-1",passed=True,run_id=orig.pin["run_id"],case_sha256=rel.sha(case),
               grid=[512,512],runtime={"system":"Darwin","device":"cpu"},grader_sha256="a"*64,smoke_source_sha256="e"*64,
               methods=list(native.METHODS),read_roles=inputs,views=views)
    orig.ps={x["run_id"]:x["manifest_sha256"] for x in (joint.pin,ref.pin,ren.pin,*models.values(),*pins.values())}
    pp=rel.write(orig.root,"payload/native-validation.json",rel.entries(proof))
    checked=rel.checked_native(dest,orig,models,joint,method,source,get)
    assert checked["checked_coordinates"]==4718592
    proof["views"]["v06"]["L1-full"]["expected_risk_sha256"]="2"*64;pp.write_bytes(rel.entries(proof))
    with pytest.raises(ValueError,match="original score values"):
        rel.checked_native(dest,orig,models,joint,method,source,get)
