"""Build the v0.2.0-rc.1 CPU model release from authenticated research records and one authorized example."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE=Path(__file__).resolve()
TAG="v0.2.0-rc.1"
BASE="b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
BASE_URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
DINO_COMMIT="7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_LICENSE="600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7"
RC3_ZIP="2e770bf450b9b8131d98a741641b990a00e01a2176088f34c1b6b8f099f96fda"
RC3_MANIFEST="e63c4155a38f690e8ddb39683a213c08209fd46e7122352c4f6f3fbf1277d621"
BRANCHES=("L2-frozen","L2-vision-ft")
VIEWS=tuple(f"v{i:02d}" for i in range(1,7))
BASELINE=("constant","1-alpha","L0","L1-full","L1-geometry")
REPORTS=("study-summary.json","family-metrics.csv","paired-changes.csv","execution-costs.json","selected-method.json")
RID=re.compile(r"\d{8}T\d{12}Z-[a-f0-9]{16}")
HASH=re.compile(r"[a-f0-9]{64}")
EXAMPLE={"input.png","cameras.json","provenance.json","v01-prediction.png","predictions/prediction.json","predictions/depth-probe.json",
         "predictions/source.npz",*("predictions/"+v+".npz" for v in VIEWS)}


def require(ok,text):
    if not ok:raise ValueError(text)


def sha(p):
    with Path(p).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def fingerprint(raw):return hashlib.sha256(raw).hexdigest()


def read(p):return json.loads(Path(p).read_bytes())


def entries(data):return (json.dumps(data,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n").encode()


def sig(p):return {"sha256":sha(p),"bytes":p.stat().st_size}


def pin(value):
    rid,sep,s=value.partition("=")
    require(sep and RID.fullmatch(rid) and HASH.fullmatch(s),"expected run_id=manifest_sha256")
    return {"run_id":rid,"manifest_sha256":s}


def safe_name(n):
    require(isinstance(n,str) and all(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*",s) for s in n.split("/")),
            "nonportable public asset name")
    return n


def base_example(archive,out):
    require(sha(archive)==RC3_ZIP,"the example must come from the approved public rc.3 ZIP")
    with zipfile.ZipFile(archive) as z:
        raw=z.read("release.json");idx=json.loads(raw)
        require(fingerprint(raw)==RC3_MANIFEST and idx["version"]=="v0.1.1-rc.3" and idx["kind"]=="heldout_study"
                and idx["publication"]["repository"]=="wany614/evidence-lens" and
                idx["license_scope"]["example"]["license_expression"]=="CC-BY-4.0","unexpected example publication origin")
        require({n.removeprefix("example/") for n in z.namelist() if n.startswith("example/")}==EXAMPLE,"approved seven-view example differs")
        for name in sorted(EXAMPLE):
            n="example/"+name
            require(n in idx["license_scope"]["example"]["members"],"outside the authorized example class")
            raw=z.read(n)
            require(len(raw)==idx["files"][n]["bytes"] and fingerprint(raw)==idx["files"][n]["sha256"],"example hash differs")
            p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)


def grade_example(a):
    from vision_public_smoke import compute
    require(isinstance(a.run_id or os.environ.get("EL_RUN_ID"),str) and RID.fullmatch(a.run_id or os.environ["EL_RUN_ID"]),
            "record the actual CPU inference with the original recorder")
    if a.run_id and os.environ.get("EL_RUN_ID"): require(a.run_id==os.environ["EL_RUN_ID"],"example run identity differs")
    rid=a.run_id or os.environ["EL_RUN_ID"]
    out=a.output;out.mkdir(parents=True,exist_ok=True);require(not any(out.iterdir()),"use an empty example result directory")
    packages={"L1":a.l1,BRANCHES[0]:a.frozen,BRANCHES[1]:a.ft}
    hashes={"L1":a.l1_sha256,BRANCHES[0]:a.frozen_sha256,BRANCHES[1]:a.ft_sha256}
    with tempfile.TemporaryDirectory() as t:
        example=Path(t)/"example";base_example(a.rc3_archive,example)
        runtime=compute(example,packages,hashes,out/"example-scores")
    doc={"schema":"el-vision-deployment-example-1","run_id":rid,"kind":"authored_textured_plane","source_tag":"v0.1.1-rc.3",
         "source_zip_sha256":RC3_ZIP,"source_release_sha256":RC3_MANIFEST,"models":hashes,"runtime":runtime,"grid":[128,128],
         "methods":[*BASELINE,*BRANCHES],"views":{v:sig(out/"example-scores"/(v+".npz")) for v in VIEWS}}
    (out/"deployment-example.json").write_bytes(entries(doc));return doc


class Original:
    """The store authenticates each consumed member; the private build receipt names those exact sources."""
    def __init__(self,store,shelf,p):
        from evidence_lens.formal_inputs import PinnedRun
        root=next((x for x in (store,shelf) if x and (x/"runs"/p["run_id"]).is_dir()),None)
        require(root is not None,"missing original run "+p["run_id"])
        self.pin={k:p[k] for k in ("run_id","manifest_sha256")}
        self.run=PinnedRun(root,self.pin)
        require(self.run.run["execution_status"]=="succeeded","release source did not succeed")
        self.files={"run.json":{k:self.run.files["run.json"][k] for k in ("bytes","sha256")}}
        self.cache={"run.json":self.run.root/"run.json"}
    def file(self,n):
        safe_name(n)
        if n not in self.cache:
            p=self.run.open(n);self.cache[n]=p;self.files[n]={k:self.run.files[n][k] for k in ("bytes","sha256")}
        return self.cache[n]
    def json(self,n):return read(self.file(n))
    def parents(self):
        return {r["run_id"]:r["manifest_sha256"] for r in self.json("inputs.json")["parents"]}


def model_proof(branch,orig,expected):
    require(orig.run.run["run_kind"]=="model_fit" and orig.run.run["experiment_id"]=="stage2-vision-train",
            "a genuine formal seed-A neural model is required")
    training=orig.json("payload/training.json");idx=orig.json("payload/package/model.json")
    setup=orig.json("payload/setup.json")
    grad=orig.json("payload/gradient-witnesses.json");changes=orig.json("payload/parameter-changes.json")
    entry=orig.json("payload/base-verification.json");dev=orig.json("payload/development.json")
    require(training["schema"]=="el-vision-training-1" and training["phase"]=="train" and training["status"]=="completed"
            and training["run_id"]==idx["model_id"]==orig.pin["run_id"] and training["branch"]==idx["branch"]==branch
            and training["seed"]==idx["seed"]==1729 and training["best_epoch"]>=1 and training["optimizer_steps"]>0,
            "a genuine formal seed-A neural model is required")
    require(idx["schema"]=="el-vision-model-1" and idx["complete_backbone_included"] is True and idx["dtype"]=="float32"
            and idx["base_checkpoint_sha256"]==training["base_checkpoint_sha256"]==BASE and entry["base"]["revision"]==DINO_COMMIT
            and entry["base"]["license_sha256"]==DINO_LICENSE and entry["base"]["checkpoint_sha256"]==BASE
            and entry["strict_equal_loaded_tensors"] is True and entry["parameters"]==22056576 and entry["tensors"]==175
            and training["frozen_identical"] is True and changes["frozen_identical"] is True and
            changes["frozen_before"]==changes["frozen_after"] and len(changes["frozen_before"])>=140,
            "base checkpoint, FP32 export or frozen-tensor evidence differs")
    require(sha(orig.file("payload/package/model.json"))==training["package"]["sha256"]==expected["model_sha256"] and
            training["threshold"]==idx["threshold"]==dev["threshold"]["tau"],"chosen operating point or model identity differs")
    groups=setup["optimizer"]["groups"];names={n for g in groups for n in g["names"]}
    require(names==set(changes["trainable"]) and not names & changes["frozen_before"].keys() and
            set(setup["initial_tensors"])==names|changes["frozen_before"].keys(),"parameter evidence inventory differs")
    allowed=("head.","projection.")+(("backbone.blocks.10.","backbone.blocks.11.","backbone.norm.") if branch==BRANCHES[1] else ())
    require(all(n.startswith(allowed) for n in names) and all(any(n.startswith(k) and changes["trainable"][n]["changed"]>0 for n in names)
            for k in allowed),"actual trained groups differ from the registered branch")
    witness=["projection.weight","head.2.weight"]+(["backbone.blocks.10.attn.qkv.weight","backbone.blocks.11.attn.qkv.weight","backbone.norm.weight"]
                                                    if branch==BRANCHES[1] else [])
    require(len(grad)==training["task_gradient_minibatches"]==8 and all(set(r["gradients_of_unregularized_view_mean_bce"])==set(witness) for r in grad) and all(sum(r["gradients_of_unregularized_view_mean_bce"][n]["l2_norm"]>0 and
            r["gradients_of_unregularized_view_mean_bce"][n]["nonzero_coordinates"]>0 for r in grad)>=3 for n in witness),
            "missing multiple data-only BCE gradient measurements")
    from safetensors import safe_open
    import numpy as np
    with safe_open(str(orig.file("payload/package/weights.safetensors")),framework="numpy") as tensors:
        require(set(tensors.keys())==names|changes["frozen_before"].keys(),"full tensor export and change ledger disagree")
        count=0
        for n in tensors.keys():
            x=tensors.get_tensor(n)
            expected_sha=changes["trainable"][n]["after_sha256"] if n in names else changes["frozen_after"][n]
            require(x.dtype==np.float32 and np.all(np.isfinite(x)) and hashlib.sha256(x.tobytes()).hexdigest()==expected_sha,
                    "exported FP32 tensor differs from recorded numerical evidence: "+n)
            count+=x.size
        require(count==idx["total_parameters"] and count<25_000_000,"architecture size or full parameter count differs")
    from evidence_lens_vision.contract import json_bytes
    table={**changes["frozen_after"],**{n:v["after_sha256"] for n,v in changes["trainable"].items()}}
    require(training["cpu_export"]["all_tensor_sha256_equal"] is True and len(table)==training["cpu_export"]["tensor_count"]==181 and
            training["cpu_export"]["tensor_table_sha256"]==fingerprint(json_bytes(table)), "independent CPU tensor reload differs")
    flags=setup["precision"]["backend_readback"]
    require(set(setup["gradient_witness_names"])==set(witness) and flags["deterministic_algorithms"] is True and
            all(flags[k] is False for k in ("cuda_matmul_allow_tf32","cudnn_allow_tf32","cudnn_benchmark","deterministic_warn_only"))
            and flags["CUBLAS_WORKSPACE_CONFIG"] in (":16:8",":4096:8") and setup["device"]=="cuda",
            "registered GPU precision or witness readback differs")
    return training,idx


def approved_models(method):
    require(method["schema"]=="el-vision-method-freeze-1" and method["methods"]==[*BASELINE,*BRANCHES] and
            len(set(method["test_family_ids"]))==len(method["test_family_ids"])==8 and method["display_rule"]=="v01"
            and method["seeds"]==[1729,2718,3141] and method["primary_seed"]==1729 and HASH.fullmatch(method["catalog_sha256"])
            and set(method["thresholds"])==set(method["methods"]) and len(method["training_receipts"])==8,
            "release needs the completed registered eight-candidate/two-branch joint method freeze")
    models=method["models"]
    require(set(models)=={"L1-full","L1-geometry",*BRANCHES} and all(
        t["domain"]=="development-B0" and type(t["tau"]) in (int,float) and math.isfinite(t["tau"]) and 0<=t["tau"]<=1.000001
        for t in method["thresholds"].values()) and method["thresholds"]["L0"]["tau"]==.5,
        "incomplete model roster or invalid development operating points")
    l1=models["L1-full"];geometry=models["L1-geometry"]
    require(all(l1[k]==geometry[k] for k in ("run_id","manifest_sha256","model_sha256")),"the two logistic models are from different frozen packages")
    for name in (*BRANCHES,"L1-full","L1-geometry"):
        m=models[name];require(RID.fullmatch(m["run_id"]) and HASH.fullmatch(m["manifest_sha256"]) and HASH.fullmatch(m["model_sha256"])
              and any(Path(r["path"]).name=="model.json" and r["sha256"]==m["model_sha256"] for r in m["files"]),
              "selected model lacks a matching sealed model index: "+name)
    return {"L1":l1,**{b:models[b] for b in BRANCHES}}


def git_blob(repo,commit,path):
    safe_name(path)
    require(re.fullmatch(r"[a-f0-9]{40}",commit),"use the full public candidate code commit")
    raw=subprocess.check_output(["git","-C",str(repo),"ls-tree","-z",commit,"--",path])
    require(raw.endswith(b"\0") and raw[:-1].partition(b"\t")[2]==path.encode() and raw.split()[0] in (b"100644",b"100755"),
            "expected a regular committed publication file: "+path)
    return subprocess.check_output(["git","-C",str(repo),"cat-file","blob",raw.partition(b"\t")[0].split()[2].decode()])


def public_json(raw,name):
    from release_model import safe_text
    safe_text(name,raw)
    return json.loads(raw)


def archive_dir(folder,output):
    members=sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    with zipfile.ZipFile(output,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name in members:
            zi=zipfile.ZipInfo(name,(1980,1,1,0,0,0));zi.create_system=3;zi.external_attr=0o100644<<16
            zi.compress_type=zipfile.ZIP_DEFLATED;zi._compresslevel=6
            with z.open(zi,"w") as dst,(folder/name).open("rb") as src:shutil.copyfileobj(src,dst,2**20)
    with zipfile.ZipFile(output) as z:
        require(z.namelist()==members,"archive membership changed")
        for name in members:
            with z.open(name) as f: require(hashlib.file_digest(f,"sha256").hexdigest()==sha(folder/name),"archive member differs: "+name)
    return members

RUNTIME={f"runtime/{pkg}/{name}.py" for pkg,names in {
    "evidence_lens":("__init__","features","l1","risk_model","scoring_io"),
    "evidence_lens_vision":("__init__","contract","model","deploy","native")}.items() for name in names}
NEURAL=RUNTIME|{"weights.safetensors","architecture.json","preprocessing.json","base.json","DINOv2-LICENSE",
                "LICENSE","NOTICE","score-native.py","infer.py","requirements.txt"}
L1={"model.json","feature_schema.json","L1-full.json","L1-geometry.json","thresholds.json"}
CONDITIONS={"B0","R1","Q1"}
DOCS=("README.md","MODEL_CARD.md","REBUILD.md","NOTICE-B.txt","THIRD_PARTY_NOTICES.md","requirements.txt")


def write(root,name,body):
    path=root/safe_name(name)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as out:out.write(body)
    return path


def copy(root,name,source):
    path=root/safe_name(name)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as out,source.open("rb") as inp:shutil.copyfileobj(inp,out,2**20)
    return path


def same_parent(parents,spec):
    require(parents.get(spec["run_id"])==spec["manifest_sha256"],"missing or mismatched direct sealed parent "+spec["run_id"])


def report_tables(rows,method):
    methods=set(method["methods"])
    keys={(r["cohort"],r["scene_family_id"],r["condition_id"],r["method"]) for r in rows}
    require(len(keys)==len(rows)==252 and all(r["status"] for r in rows),"incomplete or duplicate public cohort results")
    old={k[1] for k in keys if k[0]=="historical_regression"}
    new=set(method["test_family_ids"])
    require(len(old)==4 and not old & new,"historical and new confirmations overlap")
    expected={(cohort,f,c,m) for cohort,fams in (("new_confirmation",new),("historical_regression",old))
              for f in fams for c in CONDITIONS for m in methods}
    require(keys==expected,"required conditions, methods or families are missing from the public table")
    return {"new_confirmation":{"families":sorted(new),"family_condition_method_rows":168},
            "historical_regression":{"families":sorted(old),"family_condition_method_rows":84}}


def copy_report(root,method,freeze,report):
    from release_model import safe_text
    require(report.run.run["run_kind"]=="report" and report.run.run["experiment_id"]=="stage2-vision-confirmation-report",
            "the complete B confirmation report is required")
    same_parent(report.parents(),freeze.pin)
    contents={}
    for name in (*REPORTS,"manifest.json"):
        data=report.file("payload/public-numerical/"+name).read_bytes()
        safe_text(name,data)
        contents[name]=data
        write(root,"study/"+name,data)
    method_sha=sha(freeze.file("payload/method-freeze.json"))
    summary=json.loads(contents["study-summary.json"]);selection=json.loads(contents["selected-method.json"])
    require(summary["schema"]=="el-vision-public-study-1" and selection["schema"]=="el-vision-public-method-1" and
            summary["method_sha256"]==selection["method_sha256"]==method_sha and summary["catalog_sha256"]==method["catalog_sha256"]
            and all(summary["method_freeze"][k]==v for k,v in freeze.pin.items()) and summary["method_freeze"]["sha256"]==method_sha
            and summary["run_id"]==report.pin["run_id"] and summary["study_sha256"]==method["study_sha256"]
            and selection["thresholds"]==method["thresholds"] and all(selection["models"][k]=={n:s[n] for n in ("run_id","manifest_sha256","model_sha256")} for k,s in method["models"].items())
            and summary["methods"]==method["methods"] and set(summary["cohorts"])=={"new_confirmation","historical_regression"}
            and selection["default_method"]==method["deployment_rule"]["default"] and selection["deployment_rule"]==method["deployment_rule"],
            "public outcome or deployment decision differs from the sealed method")
    rows=list(csv.DictReader(io.StringIO(contents["family-metrics.csv"].decode())))
    scope=report_tables(rows,method)
    for r in rows:
        V,E,H,L,U,Y,HY,LY,UY,S5,S7=(int(r[n]) for n in ("V","E","H","L","U","Y","HY","LY","UY","S5","S7"))
        require(0<=S7<=S5<=E<=V and min(H,L,U,Y,HY,LY,UY)>=0 and E==H+L+U and Y==HY+LY+UY and
                HY<=H and LY<=L and UY<=U and int(r["views_render_failed"])+int(r["views_rendered"])==int(r["views_planned"])==6,
                "public RGB-error counts or scheduled views disagree")
    table_manifest=json.loads(contents["manifest.json"])
    require(table_manifest["schema"]=="el-vision-public-numerical-1" and table_manifest["method_sha256"]==method_sha and
            table_manifest["report_run_id"]==report.pin["run_id"] and set(table_manifest["files"])==set(REPORTS),
            "numerical manifest must belong to this report and seven-method freeze")
    for cohort,v in scope.items():
        c=summary["cohorts"][cohort]
        require(sorted(c["families"])==v["families"] and set(c["conditions"])==CONDITIONS and
                c["planned_family_conditions"]==len(v["families"])*3 and c["planned_views"]==len(v["families"])*18,
                "summary, planned views and full family table disagree")
        same_parent(report.parents(),c["evaluation"])
    for name in REPORTS:
        v=table_manifest["files"][name]
        require(v=={"sha256":fingerprint(contents[name]),"bytes":len(contents[name])},"numerical export hash or size differs for "+name)
    return scope


def selected_training(get,freeze,method):
    from evidence_lens_vision.selection import choose
    parents=freeze.parents();rows=[]
    for recorded in method["training_receipts"]:
        same_parent(parents,recorded)
        fit=get(recorded);p=fit.file("payload/training.json");r=read(p)
        require(fit.run.run["run_kind"]=="model_fit" and fit.run.run["experiment_id"]=="stage2-vision-train" and
                r["run_id"]==recorded["run_id"] and r["branch"]==recorded["method"] and r["seed"]==recorded["seed"]
                and r["best_epoch"]==recorded["best_epoch"] and r["package"]["sha256"]==recorded["model_sha256"]
                and sha(fit.file("payload/package/model.json"))==recorded["model_sha256"] and sha(p)==recorded["training_json_sha256"],
                "scientific fit does not match a registered candidate")
        require(r["training_dataset_sha256"]==method["train_dataset_sha256"] and r["development_dataset_sha256"]==method["development_dataset_sha256"],
                "candidate used another learning split")
        rows.append(dict(r,manifest_sha256=recorded["manifest_sha256"],model_sha256=recorded["model_sha256"]))
    selection=choose(rows,"primary")
    require(all(selection["chosen"][b]["model_sha256"]==method["models"][b]["model_sha256"] for b in BRANCHES),
            "independently recomputed primary choice differs")
    return selection


def copy_models(root,models,method,get):
    from evidence_lens.l1 import load_package
    for name,spec in method["models"].items():
        src=get(spec)
        for file in spec["files"]:
            require(safe_name(file["path"]).startswith("payload/") and
                    sig(src.file(file["path"]))=={k:file[k] for k in ("sha256","bytes")},
                    "frozen model member differs for "+name)
    l1=get(models["L1"])
    require(l1.run.run["run_kind"]=="threshold_select","a development-frozen deployable L1 is required")
    for name in sorted(L1):copy(root,"models/L1/"+name,l1.file("payload/model/"+name))
    package=load_package(root/"models/L1",expected_index_sha256=models["L1"]["model_sha256"])
    index=package["index"];fit_id=index["fit_run_id"]
    require(index["model_id"]==models["L1"]["run_id"] and fit_id in l1.parents() and
            all(v==method["thresholds"][k]["tau"] for k,v in package["thresholds"].items()),"L1 producer or operating thresholds differ")
    fit=get(dict(run_id=fit_id,manifest_sha256=l1.parents()[fit_id]))
    require(fit.run.run["run_kind"]=="model_fit","L1 threshold parent is not the original fit")
    fit_receipt=fit.json("payload/fit.json");initial=fit.json("payload/model/model.json")
    require(fit_receipt["schema"]=="el-vision-lr-fit-1" and fit_receipt["status"]=="fitted" and
            fit_receipt["run_id"]==initial["model_id"]==initial["fit_run_id"]==fit_id and
            fit_receipt["training_dataset_sha256"]==method["train_dataset_sha256"] and len(set(fit_receipt["families"]))==28 and
            fit_receipt["model_sha256"]==sha(fit.file("payload/model/model.json")) and initial["status"]=="fit" and
            set(initial["files"])==L1-{"model.json","thresholds.json"},"the original B logistic fit or training roster differs")
    for name,d in initial["files"].items():
        require(index["files"][name]==d==sha(fit.file("payload/model/"+name)),"development changed original fitted parameters "+name)
    cv=fit.file("payload/cv_results.json")
    require(sha(cv)==fit_receipt["cv_sha256"],"L1 cross-validation bytes differ")
    cv_raw=cv.read_bytes();public_json(cv_raw,"cv_results.json");write(root,"evidence/L1/cv_results.json",cv_raw)
    write(root,"evidence/L1/identity.json",entries({"fit":fit.pin,"rows":fit_receipt["rows"],"positive_rows":fit_receipt["positive_rows"],
        "training_dataset_sha256":fit_receipt["training_dataset_sha256"],"fit_receipt_sha256":sha(fit.file("payload/fit.json")),"threshold_select":l1.pin,
        "training_manifest_sha256":index["training_manifest_sha256"],"feature_schema_digest":index["feature_schema_digest"],
        "thresholds":package["thresholds"],"coefficients":{k:{"C":v["C"],"columns":v["columns"]} for k,v in package["models"].items()}}))
    published={"L1":{"license":"Apache-2.0","trained_by":"wany614","files":{n:sig(root/"models/L1"/n) for n in sorted(L1)}}}
    for branch in BRANCHES:
        spec=models[branch];orig=get(spec)
        training,idx=model_proof(branch,orig,spec)
        require(set(idx["files"])==NEURAL and idx["threshold"]==method["thresholds"][branch]["tau"],
                "unexpected deployed neural package scope or operating point")
        copy(root,"models/"+branch+"/model.json",orig.file("payload/package/model.json"))
        for n,record in sorted(idx["files"].items()):
            src=orig.file("payload/package/"+n)
            require(record==dict(path=n,**orig.files["payload/package/"+n]),"neural package file differs: "+n)
            copy(root,"models/"+branch+"/"+n,src)
        require(sha(root/"models"/branch/"DINOv2-LICENSE")==DINO_LICENSE,"upstream license bytes differ")
        for n in ("setup.json","base-verification.json","curves.json","gradient-witnesses.json","parameter-changes.json"):
            p=orig.file("payload/"+n)
            require(training["files"][n]["sha256"]==sha(p),"scientific proof differs from training receipt "+n)
            data=p.read_bytes();public_json(data,n);write(root,"evidence/"+branch+"/"+n,data)
        public={k:v for k,v in training.items() if k!="files"}
        public["origin"]=orig.pin
        data=entries(public);public_json(data,"training.json");write(root,"evidence/"+branch+"/training.json",data)
        published[branch]={"license":"Apache-2.0","upstream":"Meta DINOv2 ViT-S/14","upstream_base_sha256":BASE,
                           "modifications":"wany614 trained the projection and risk head" + (" and pretrained blocks 10, 11 and final LayerNorm." if branch==BRANCHES[1] else "."),
                           "weights":sig(root/"models"/branch/"weights.safetensors"),"model":sig(root/"models"/branch/"model.json")}
    return published


def write_example(root,grade,models,approved):
    require(grade.run.run["run_kind"]=="model_score","the deployment example requires a recorded model score")
    parents=grade.parents()
    for spec in models.values():same_parent(parents,spec)
    base_example(approved,root/"example")
    raw=grade.file("payload/deployment-example.json").read_bytes();doc=public_json(raw,"deployment-example.json")
    require(doc["schema"]=="el-vision-deployment-example-1" and doc["run_id"]==grade.pin["run_id"]
            and doc["models"]=={b:m["model_sha256"] for b,m in models.items()} and doc["source_zip_sha256"]==RC3_ZIP
            and doc["source_release_sha256"]==RC3_MANIFEST and doc["kind"]=="authored_textured_plane"
            and doc["grid"]==[128,128] and doc["methods"]==[*BASELINE,*BRANCHES] and set(doc["views"])==set(VIEWS)
            and doc["runtime"]["device"]=="cpu","mismatched evidence for the authored example")
    write(root,"evidence/example-grade.json",raw)
    for view in VIEWS:
        path=grade.file("payload/example-scores/"+view+".npz");require(sig(path)==doc["views"][view],"example score member differs")
        copy(root,"expected/"+view+".npz",path)
    return {"source_zip_sha256":RC3_ZIP,"source_release_sha256":RC3_MANIFEST,"grade_run":grade.pin,"grade_sha256":fingerprint(raw)}


def verify_native(root, destination, models, thresholds):
    # The clean interpreter selects the signed model runtime even if build-time Evidence Lens is already imported.
    entry="""import json,sys
from pathlib import Path
root=Path(sys.argv[1]);work=Path(sys.argv[2]);spec=json.load(sys.stdin)
sys.path.insert(0,str(root))
from smoke import compute, compare
models={n:root/'models'/n for n in spec['models']}
runtime=compute(root/'example',models,spec['models'],work/'scores')
r=compare(root/'expected',work/'scores',work,spec['thresholds'])
print(json.dumps(dict(runtime=runtime,**r),sort_keys=True,allow_nan=False))
"""
    with tempfile.TemporaryDirectory(prefix="evidence-lens-native-") as scratch:
        proc=subprocess.run([sys.executable,"-I","-B","-c",entry,str(root),scratch],
                            input=json.dumps({"models":{n:v["model_sha256"] for n,v in models.items()},"thresholds":thresholds}),
                            text=True,capture_output=True)
        if proc.returncode:
            raise ValueError("native public example failed: "+proc.stderr[-2400:])
        result=json.loads(proc.stdout)
        require(not result["failed"] and len(result["views"])==6 and all(len(v)==7 for v in result["views"].values())
                and len(result["map_files"])==36,"native reproduction differs: "+", ".join(result["failed"]))
        write(destination,"build-smoke.json",entries(dict(schema="el-vision-build-smoke-1",**result)))


def checked_native(out,original,models,freeze,method,sources,get):
    import numpy as np
    from vision_release_native import DATA_NAMES, METHODS
    p=original.file("payload/native-validation.json")
    body=p.read_bytes();proof=public_json(body,"native-validation.json")
    chosen=sorted(r["scene_family_id"] for r in read(out/"protocol/stage2-families-v1.json")["families"] if r["split"]=="development")[0]
    want={k:{j:v[j] for j in ("model_sha256","manifest_sha256","run_id")} for k,v in models.items()}
    require(original.run.run["run_kind"]=="archive_verify" and original.run.run["experiment_id"]=="stage2-release-native"
            and proof["schema"]=="el-vision-formal-native-equivalence-1" and proof["passed"] is True
            and proof["run_id"]==original.pin["run_id"] and proof["method"]==freeze.pin
            and proof["method_sha256"]==sha(freeze.file("payload/method-freeze.json")) and proof["models"]==want
            and proof["catalog_sha256"]==method["catalog_sha256"] and proof["family"]==chosen
            and proof["split"]=="development" and proof["condition"]=="B0" and proof["grid"]==[512,512]
            and proof["runtime"]["system"]=="Darwin" and proof["runtime"]["device"]=="cpu"
            and proof["grader_sha256"]==sources["scripts/vision_release_native.py"]["sha256"]
            and proof["smoke_source_sha256"]==sources["scripts/vision_public_smoke.py"]["sha256"]
            and set(proof["read_roles"])==DATA_NAMES and set(proof["views"])==set(VIEWS)
            and proof["methods"]==list(METHODS),"a complete independent Mac formal-native release grade is required")
    for v,methods in proof["views"].items():
        require(set(methods)==set(METHODS),"missing formal native method for "+v)
        for m,r in methods.items():
            tol=.000001 if m=="L1-full" else .0002
            require(r["passed"] is True and r["pixels"]==512**2 and 0<r["finite_actual"]==r["finite_expected"]==r["finite_common"]<=512**2
                    and r["support_difference"]==r["threshold_state_difference"]==0 and r["absolute_tolerance"]==tol
                    and 0<=r["mean_abs"]<=r["max_abs"]<=tol,"formal native comparison failed for "+m+"/"+v)
    inputs={r["role"]:r for r in original.json("inputs.json")["inputs"]}
    casep=original.file(inputs["case"]["path"]);case=read(casep)
    require(sha(casep)==proof["case_sha256"] and case["schema"]=="el-vision-native-release-case-1" and case["models"]==want and
            all(case[k]==proof[k] for k in ("family","trial_id","condition","split","catalog_sha256","method","method_sha256","references","reference","render","selection_sha256")),
            "formal case fingerprint or original identity differs from its seal")
    require(all(case["files"]["input/"+n]==spec for n,spec in proof["read_roles"].items()),"recorded inference read a different input packet")
    ref,render=get(case["reference"]),get(case["render"])
    for n,entry in proof["read_roles"].items():
        run=render if n.startswith("predictions/") else ref
        member="payload/"+n.removeprefix("predictions/")
        require(sig(run.file(member))==entry and run.run.run["scene_family_id"]==chosen and run.run.run["split"]=="development"
                and run.run.run["condition_id"]=="B0","original formal family input differs for "+n)
    original_roster={r["role"]:r for r in freeze.json("inputs.json")["inputs"]}["method_roster"]
    rosterp=freeze.file(original_roster["path"]);require(sha(rosterp)==method["input_roster_sha256"],"method roster differs")
    baseline_pin=read(rosterp)["stages"]["baseline_scores"]
    baseline=get(baseline_pin)
    require(proof["references"]["L1-full"]==baseline.pin,"independent logistic output is not from its frozen baseline score parent")
    scores=baseline.json("payload/scores.json")
    require(scores["method"]=="original-five" and scores["model_sha256"]==want["L1"]["model_sha256"]
            and scores["dataset_sha256"]==method["development_dataset_sha256"],"incorrect original formal logistic map source")
    bv={r["view"]:r["score"] for r in scores["views"] if r["family"]==chosen and r["condition"]=="B0"}
    require(set(bv)==set(VIEWS),"incomplete canonical logistic score roster")
    for b in BRANCHES:
        require(proof["references"][b]==method["cpu_costs"][b]["pin"],"independent neural output is not its frozen native cost parent")
    for v,checks in proof["views"].items():
        for m,r in checks.items():
            if m=="L1-full":
                path=baseline.file("payload/"+safe_name(bv[v]["path"]));key="l1_full"
                require(sig(path)=={k:bv[v][k] for k in ("bytes","sha256")},"original formal logistic NPZ changed")
            else:
                path=get(proof["references"][m]).file("payload/cold/"+v+".npz");key="risk"
                require(sig(path)==case["files"]["expected/"+m+"/"+v+".npz"],"original formal visual NPZ changed")
            with np.load(path,allow_pickle=False) as z:grid=z[key]
            require(grid.dtype==np.float32 and grid.shape==(512,512) and HASH.fullmatch(r["actual_risk_sha256"]) and
                    fingerprint(np.ascontiguousarray(grid,dtype="<f4").tobytes())==r["expected_risk_sha256"],
                    "formal case used different original score values for "+m+"/"+v)
    parents=original.parents()
    for s in (freeze.pin,proof["reference"],proof["render"],*models.values(),*proof["references"].values()):same_parent(parents,s)
    write(out,"evidence/formal-native-validation.json",body)
    return dict(original.pin,sha256=fingerprint(body),family=chosen,views=6,methods=3,checked_coordinates=512**2*18)


def release_notes(doc,summary,costs,assets):
    lines=["# Evidence Lens "+TAG,"","Native-view risk scoring with the original logistic model and two DINOv2 ViT-S/14 variants. "
           "The development-frozen default is **"+doc["default_method"]+"**.","",
           "| Cohort | Families | Planned family / condition pairs | Scheduled views | Original render dispositions |",
           "| --- | ---: | ---: | ---: | --- |"]
    for key,label in (("new_confirmation","New confirmation"),("historical_regression","Historical regression")):
        c=summary["cohorts"][key]
        disp=", ".join(n+" "+str(v) for n,v in sorted(c["render_status"].items()))
        dispositions=", ".join(n+" "+str(v) for n,v in sorted(c["scientific_view_dispositions"].items()))
        disp+=". Scientific dispositions " + dispositions
        lines.append(f"| {label} | {len(c['families'])} | {c['planned_family_conditions']} | {c['planned_views']} | {disp} |")
    systems=[]
    for r in costs["observed_runs"]:
        if r["status"]!="succeeded":systems.append(r["run_id"]+" ("+r["run_kind"]+", "+r["status"]+")")
    lines+=["","The public numerical attachment includes all family rows, denominator counts, paired changes and recorded costs. "
           "The quantitative study uses the separately identified new and historical cohorts. "
           "The CPU ZIP contains the project-authored plane input and predictions.","",
           ("The published inventory records "+str(len(systems))+" unsuccessful runs. "+", ".join(sorted(systems))+"." if systems else "Every run in the observed inventory completed successfully."),"",
           "The authorized example covers six native views and seven methods. An independent full-image CPU check covers all "
           "three fitted model packages and all six 512 × 512 views of the fixed development B0 case. "
           "The numerical proof and model identity are inside the CPU archive.","","| Model | Original run | SHA-256 of model index |",
           "| --- | --- | --- |"]
    for k,v in doc["models"].items():lines.append(f"| {k} | `{v['run_id']}` | `{v['model_sha256']}` |")
    lines+=["","Source commit `"+doc["code_commit"]+"`. Full-precision safetensors contain the complete visual backbones. "
            "The original Meta base is identified by `"+BASE+"`.","",
            "Use Python 3.11 or 3.12. After extracting the CPU ZIP, run `python3 -I -B smoke.py --output ../vision-result` "
            "on macOS/Linux or `py -3.12 -I -B smoke.py --output ../vision-result` on Windows. The `--case` option accepts the "
            "independently supplied 512-pixel `public-plane` deployment control for the registered bottle-carrier cameras. "
            "Its attachment and case manifest have separate checksums.","","| Asset | Bytes | SHA-256 |",
            "| --- | ---: | --- |"]
    for k,r in sorted(assets.items()):lines.append(f"| {k} | {r['bytes']} | `{r['sha256']}` |")
    lines+=["","Project source and model additions are Apache-2.0. Original authorized documentation, plane media and quantitative "
            "records are CC BY 4.0. Credit wany614. Meta's upstream license and base distribution are identified in the bundle."]
    return "\n".join(lines)+"\n"


def release_sources(repo,commit,out):
    protocols=("protocol/stage2-training-v1.json","protocol/stage2-families-v1.json","protocol/stage2-inference-parity-v1.json",
               "protocol/families-v1.json","configs/vision-base-v1.json","configs/vision-public-control-v1.json")
    paths=("scripts/vision_release.py","scripts/vision_public_smoke.py","scripts/vision_release_native.py","scripts/verify_vision_public.py","scripts/vision_control_attachment.py",
           *protocols,
           "LICENSE","NOTICE","licenses/CC-BY-4.0.txt",*("docs/releases/"+TAG+"/"+n for n in DOCS))
    data={p:git_blob(repo,commit,p) for p in paths}
    for n in DOCS:
        write(out,n,data["docs/releases/"+TAG+"/"+n])
    for n in ("LICENSE","NOTICE","licenses/CC-BY-4.0.txt"):
        write(out,n,data[n])
    write(out,"smoke.py",data["scripts/vision_public_smoke.py"])
    for n in protocols:
        write(out,n,data[n])
    return {n:{"bytes":len(v),"sha256":fingerprint(v)} for n,v in sorted(data.items())}


def build(a):
    from evidence_lens.scoring_io import new_output
    from vision_public_smoke import check_assets
    require(a.version==TAG and not a.output.exists(),"choose the registered B version and a new release directory")
    runs={}
    def get(p):
        rid=p["run_id"]
        if rid not in runs:runs[rid]=Original(a.store,a.shelf,p)
        require(runs[rid].pin["manifest_sha256"]==p["manifest_sha256"],"conflicting original manifest "+rid)
        return runs[rid]
    freeze=get(pin(a.method));method=freeze.json("payload/method-freeze.json")
    require(freeze.run.run["run_kind"]=="report" and freeze.run.run["experiment_id"]=="stage2-method-freeze"
            and method["run_id"]==freeze.pin["run_id"],"a completed canonical method freeze is required")
    models=approved_models(method)
    method_sha=sha(freeze.file("payload/method-freeze.json"))
    marker_path=a.store/"protocol/stage2"/method["catalog_sha256"]/"test-opening.json"
    require(marker_path.is_file() and not marker_path.is_symlink(),"no canonical test opening for this cohort")
    marker=read(marker_path)
    require(marker["schema"]=="el-stage2-test-opening-1" and marker["method_sha256"]==method_sha
            and marker["freeze_run_id"]==freeze.pin["run_id"] and marker["freeze_manifest_sha256"]==freeze.pin["manifest_sha256"]
            and marker["method_member"]=="payload/method-freeze.json" and marker["catalog_sha256"]==method["catalog_sha256"]
            and sorted(marker["test_family_ids"])==sorted(method["test_family_ids"]),"test opening and selected method differ")
    original_source={"repository":"https://github.com/facebookresearch/dinov2","revision":DINO_COMMIT,
                     "upstream_model_card":"https://github.com/facebookresearch/dinov2/blob/"+DINO_COMMIT+"/MODEL_CARD.md",
                     "original_distribution":BASE_URL,"sha256":BASE,"bytes":88283115,"architecture":"dinov2_vits14","registers":0,
                     "license":"Apache-2.0","license_sha256":DINO_LICENSE}
    report=get(pin(a.report));grade=get(pin(a.example_grade));native_grade=get(pin(a.native_grade))
    with new_output(a.output) as dest,tempfile.TemporaryDirectory(prefix="evidence-lens-public-") as tmp:
        out=Path(tmp);sources=release_sources(a.repository,a.commit,out)
        require(sha(out/"protocol/stage2-families-v1.json")==method["catalog_sha256"] and
                sha(out/"protocol/families-v1.json")==method["original_catalog_sha256"] and
                sha(out/"protocol/stage2-inference-parity-v1.json")==method["inference_parity_protocol_sha256"] and
                sha(out/"protocol/stage2-training-v1.json")==method["study_sha256"],"public code commit carries a different study or split")
        scope=copy_report(out,method,freeze,report)
        selection=selected_training(get,freeze,method)
        selected_fit=get(models[BRANCHES[0]])
        audit=selected_fit.json("payload/transport-audit.json")
        worker=selected_fit.file("inputs/worker/worker.zip")
        require(audit["worker_sha256"]==sha(worker)=="19998e7fd69a6ba305633b1c62d5da0d9f09fe1ef5e3cb7b6177f7a9f54cc8b9",
                "the formal, registered scientific worker does not match the source ZIP")
        with zipfile.ZipFile(worker) as z:
            inventory=json.loads(z.read("worker-source.json"))
            require(fingerprint(z.read("worker-source.json"))==selection["source_sha256"]["worker-source.json"] and
                    set(inventory["files"])|{"worker-source.json"}==set(z.namelist()) and len(z.namelist())==29,
                    "frozen scientific source inventory differs")
            for n,v in inventory["files"].items():
                body=z.read(safe_name(n))
                require(len(body)==v["bytes"] and fingerprint(body)==v["sha256"],"frozen source hash mismatch "+n)
        copy(out,"evidence/formal-training-source.zip",worker)
        rights=copy_models(out,models,method,get)
        native=checked_native(out,native_grade,models,freeze,method,sources,get)
        example=write_example(out,grade,models,a.rc3_archive)
        verify_native(out,dest,models,method["thresholds"])
        write(out,"evidence/selected-training.json",entries(selection))
        write(out,"evidence/method-freeze.json",freeze.file("payload/method-freeze.json").read_bytes())
        write(out,"licenses/DINOv2-LICENSE.txt",(out/"models"/BRANCHES[0]/"DINOv2-LICENSE").read_bytes())
        write(out,"licenses/ASSET-SCOPE.json",entries({"schema":"el-vision-release-rights-1","version":TAG,
              "credit":"wany614","software_and_original_model_parameters":"Apache-2.0","authored_example_and_original_records":"CC-BY-4.0",
              "upstream_base":original_source,"models":rights,"example_origin":example,
              "formal_publication":"Original quantitative tables and project-authored model documentation."}))
        write(out,"BASE-SOURCE.json",entries(original_source))
        weights={k:sig(out/"models"/"L1"/(k+".json")) for k in ("L1-full","L1-geometry")}
        weights.update({b:sig(out/"models"/b/"weights.safetensors") for b in BRANCHES})
        provenance={n:{k:s[k] for k in ("run_id","manifest_sha256","model_sha256")} for n,s in models.items()}
        for desc in ("thresholds","deployment_rule"):
            public_json(entries(method[desc]),desc+".json")
        doc={"schema":"el-vision-public-release-1","version":TAG,"repository":"wany614/evidence-lens","attribution":"wany614",
             "code_commit":a.commit,"scientific_status":"confirmed_heldout_report_recorded","models":provenance,"weights":weights,
             "methods":method["methods"],"thresholds":method["thresholds"],"default_method":method["deployment_rule"]["default"],
             "method_freeze":dict(freeze.pin,sha256=method_sha),"report":report.pin,"example":example,"native_validation":native,"study":scope,
             "study_sha256":method["study_sha256"],"catalog_sha256":method["catalog_sha256"],"original_catalog_sha256":method["original_catalog_sha256"],
             "inference_parity_protocol_sha256":method["inference_parity_protocol_sha256"],"opening_sha256":sha(marker_path),
             "upstream":original_source,"files":{p.relative_to(out).as_posix():sig(p) for p in sorted(out.rglob("*")) if p.is_file()}}
        raw=entries(doc);public_json(raw,"release.json");write(out,"release.json",raw)
        signatures={**{n:v["sha256"] for n,v in doc["files"].items()},"release.json":fingerprint(raw)}
        write(out,"SHA256SUMS","".join(signatures[n]+"  "+n+"\n" for n in sorted(signatures)).encode())
        check_assets(out,fingerprint(raw))
        name="evidence-lens-"+TAG+"-cpu.zip"
        archive_dir(out,dest/name)
        numbers="evidence-lens-"+TAG+"-numerical.zip"
        archive_dir(out/"study",dest/numbers)
        assets={name:sig(dest/name),numbers:sig(dest/numbers)}
        if a.base_asset:
            require(sig(a.base_asset)=={"sha256":BASE,"bytes":88283115},"optional raw upstream checkpoint differs")
            n="dinov2_vits14_pretrain.pth";path=copy(dest,n,a.base_asset);assets[n]=sig(path)
        require(all(v["bytes"]<2*1024**3 for v in assets.values()),"release exceeds the single-asset size budget")
        body=release_notes(doc,read(out/"study/study-summary.json"),read(out/"study/execution-costs.json"),assets)
        write(dest,"RELEASE_NOTES.md",body.encode())
        write(dest,"SHA256SUMS","".join(assets[n]["sha256"]+"  "+n+"\n" for n in sorted(assets)).encode())
        receipt={"schema":"el-vision-public-candidate-1","version":TAG,"code_commit":a.commit,"assets":assets,
                 "release_sha256":fingerprint(raw),"members":len(signatures)+1,"models":provenance,"weights":weights,"method":freeze.pin,
                 "report":report.pin,"example_grade":grade.pin,"native_grade":native_grade.pin,"base":original_source,
                 "supporting_assets":{n:sig(dest/n) for n in ("SHA256SUMS","RELEASE_NOTES.md")}}
        write(dest,"candidate.json",entries(receipt))
        source_list=[]
        for rid,original in sorted(runs.items()):
            parents=original.parents()
            source_list.append(dict(original.pin,run_kind=original.run.run["run_kind"],experiment_id=original.run.run["experiment_id"],
                                    parents=parents,verified_members=original.files))
        write(dest,"private-sources.json",entries({"schema":"el-vision-private-release-sources-1","candidate":receipt,
             "committed_sources":sources,"authorized_former_example_zip_sha256":RC3_ZIP,"opening_sha256":sha(marker_path),
             "original_runs":source_list}))
        if a.rebuild_of:
            for n in ("candidate.json","private-sources.json","SHA256SUMS","RELEASE_NOTES.md",*assets):
                require(sig(dest/n)==sig(a.rebuild_of/n),"independent release rebuild differs: "+n)
    return receipt


def isolated_build(a):
    require(git_blob(a.repository,a.commit,"scripts/vision_release.py")==HERE.read_bytes(),
            "run the builder from the specified public source commit")
    with tempfile.TemporaryDirectory(prefix="evidence-lens-source-") as work:
        root=Path(work);archive=root/"source.zip"
        subprocess.run(["git","-C",str(a.repository),"archive","--format=zip","--output",str(archive),a.commit,"--",
                        "src/evidence_lens","src/evidence_lens_vision","scripts"],check=True)
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.endswith(".py"):continue
                name=safe_name(info.filename)
                require(not (info.external_attr>>16 & 0o170000)==0o120000,"unexpected Git source symlink")
                write(root,name,z.read(info))
        args={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()}
        bootstrap="""import json,sys
from pathlib import Path
root=Path(sys.argv[1])
sys.path[:0]=[str(root/'scripts'),str(root/'src')]
import vision_release as r
a=r.argparse.Namespace(**json.load(sys.stdin))
for k in ('repository','store','shelf','output','rc3_archive','base_asset','rebuild_of'):
    if getattr(a,k):setattr(a,k,Path(getattr(a,k)))
print(json.dumps(r.build(a),sort_keys=True))
"""
        proc=subprocess.run([sys.executable,"-I","-B","-c",bootstrap,str(root)],input=json.dumps(args),text=True,check=True,capture_output=False)
        return proc.returncode


def main():
    if sys.version_info<(3,11):raise SystemExit("Evidence Lens releases require Python 3.11 or newer.")
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    g=sub.add_parser("grade-example")
    for k in ("rc3-archive","l1","frozen","ft","output"):g.add_argument("--"+k,type=Path,required=True)
    for k in ("l1-sha256","frozen-sha256","ft-sha256"):g.add_argument("--"+k,required=True)
    g.add_argument("--run-id")
    b=sub.add_parser("build")
    for k in ("repository","store","rc3-archive","output"):b.add_argument("--"+k,type=Path,required=True)
    for k in ("shelf","base-asset","rebuild-of"):b.add_argument("--"+k,type=Path)
    for k in ("version","commit","method","report","example-grade","native-grade"):b.add_argument("--"+k,required=True)
    a=p.parse_args()
    if a.action=="build":return isolated_build(a)
    sys.path.insert(0,str(HERE.parent))
    doc=grade_example(a)
    print(json.dumps({"run_id":doc["run_id"],"views":len(doc["views"]),"sha256":sha(a.output/"deployment-example.json")},sort_keys=True))


if __name__=="__main__":
    main()
