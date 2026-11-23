"""Synthetic dispatches check the dataset evidence and the existing archive return protocol."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import vision_gpu_job as job


def synthetic_dataset(dir):
    dir.mkdir()
    views=[]
    for i in range(1,7):
        v=f"v{i:02d}";folder=dir/"f"/"B0"/v
        folder.mkdir(parents=True)
        r={"family":"f","condition":"B0","view":v,"status":"completed","reason":None,"full":None}
        for key,ext in (("rgb","npy"),("samples","npz")):
            p=folder/(key+"."+ext);p.write_bytes(("synthetic-"+v+key).encode())
            r[key]={"path":p.relative_to(dir).as_posix(),**job.sig(p)}
        views.append(r)
    doc={"schema":"el-vision-dataset-1","split":"train","views":views}
    (dir/"dataset.json").write_bytes(job.encoded(doc))
    return {"files":job.data_inventory(doc,"train"),"views":6,"manifest":job.sig(dir/"dataset.json")}


def test_external_data_inventory_and_sealed_child_failure(tmp_path,monkeypatch):
    d=tmp_path/"data";expected=synthetic_dataset(d)
    assert job.audit_dataset(d,d/"dataset.json",expected)["files"]==13
    bad=d/"f/B0/v06/samples.npz";original=bad.read_bytes()
    bad.write_bytes(b"replaced")
    with pytest.raises(ValueError,match="member differs"):
        job.audit_dataset(d,d/"dataset.json",expected)
    bad.write_bytes(original)
    (d/"extra").write_text("extra")
    with pytest.raises(ValueError,match="missing or extra"):
        job.audit_dataset(d,d/"dataset.json",expected)
    (d/"extra").unlink()
    fake=tmp_path/"source.zip"
    with zipfile.ZipFile(fake,"w") as z:
        z.writestr("scripts/vision_worker.py","from pathlib import Path\nimport sys\n"
                   "a=Path(sys.argv[sys.argv.index('--output')+1]);a.mkdir()\n"
                   "(a/'partial.json').write_text('partial synthetic unit')\nraise SystemExit(7)\n")
    cfg=tmp_path/"config.json";cfg.write_bytes(job.encoded({"phase":"pilot"}))
    info={"phase":"pilot","worker_sha256":job.sha(fake),"configuration_sha256":job.sha(cfg),"datasets":{"train":expected},"parents":{}}
    plan=tmp_path/"job.json";plan.write_bytes(job.encoded(info))
    for k,v in {"EL_RUN_ID":"unit","EL_VISION_TRAIN_ROOT":str(d),"EL_VISION_DINO_ROOT":str(tmp_path)}.items():monkeypatch.setenv(k,v)
    args=SimpleNamespace(job=plan,config=cfg,worker_zip=fake,train=d/"dataset.json",base_pin=cfg,base=cfg,run_id="unit",
                         preprocessing=cfg,output=tmp_path/"out",development=None,lr_selection=None)
    assert job.scientific(args)==7
    assert (args.output/"partial.json").read_text()=="partial synthetic unit"
    assert job.load(args.output/"transport-audit.json")["datasets"]["train"]["dataset_sha256"]==job.sha(d/"dataset.json")


def test_original_archive_failed_return_is_verified_and_idempotent(tmp_path):
    pytest.importorskip("pyarrow")
    from evidence_lens import recorder,archive
    src=tmp_path/"source";(src/"scripts").mkdir(parents=True)
    (src/"scripts/stub.py").write_text("raise SystemExit(3)\n")
    handoff=tmp_path/"job";handoff.mkdir()
    info={"schema":"el-vision-gpu-job-1","phase":"pilot","trial_id":"synthetic-return","experiment_id":"stage2-vision-pilot",
          "files":{},"parents":{},"datasets":{}}
    (handoff/"job.json").write_bytes(job.encoded(info));dh=job.sha(handoff/"job.json")
    cfg=tmp_path/"config.json";cfg.write_bytes(job.encoded({"formal_handoff_sha256":dh}))
    # The fixture uses the production recorder/archiver, with its own tiny independent store.
    store=tmp_path/"producer";store.mkdir()
    policy=tmp_path/"storage.json";policy.write_bytes(job.encoded({"local_free_floor_bytes":0,"local_active_batch_target_bytes":1<<30}))
    args=SimpleNamespace(root=store,source_root=src,kind="archive_verify",config=cfg,storage_policy=policy,
                   experiment="unit-upstream",trial="unit-parent",interval=None,parent=[],retry_of=None,input=[],
                   shared_input=[],input_parent=[],command=["--","{python}","-c","pass"])
    parent=recorder.record(args)
    pzip=tmp_path/(parent["run_id"]+".zip")
    archive.pack(Path(parent["path"]),pzip,store)
    assert archive.receive(pzip,tmp_path/"canonical",job.sha(pzip),policy)["status"]=="accepted"
    args.kind="model_fit";args.experiment=info["experiment_id"];args.trial=info["trial_id"]
    args.parent=[parent["run_id"]+"="+parent["manifest_sha256"]]
    args.command=["--","{python}","{source}/scripts/stub.py"]
    result=recorder.record(args)
    assert result["execution_status"]=="failed"
    bundle=Path(result["path"]);zip_path=tmp_path/(result["run_id"]+".zip");archive.pack(bundle,zip_path,store)
    receipt=tmp_path/"receipt.json";receipt.write_bytes(job.encoded({"schema":"el-worker-return-1","run":result["run_id"],
                  "handoff_sha256":dh,"status":"failed","exit_code":result["exit_code"],"trial_id":info["trial_id"],"run_kind":"model_fit",
                  "manifest_sha256":result["manifest_sha256"],"package":zip_path.name,**job.sig(zip_path)}))
    a=SimpleNamespace(job=handoff,job_sha256=dh,receipt=receipt,archive=zip_path,store=tmp_path/"canonical")
    assert job.receive_mac(a)["status"]=="accepted"
    assert job.receive_mac(a)["status"]=="identical"
    doc=job.load(receipt);doc["handoff_sha256"]="a"*64;receipt.write_bytes(job.encoded(doc))
    with pytest.raises(ValueError,match="return receipt"):
        job.receive_mac(a)
