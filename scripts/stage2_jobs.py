"""Build a single-image input-only Windows batch from the registered stage-two references."""
import argparse
import json
import sys
from pathlib import Path

from evidence_lens._bundle import encoded, load, sha
from evidence_lens._verify import require
from formal_lineage import VerifiedRuns, reference_identity, run_pin
from make_handoff import publish, sha_bytes, source_files, zip_bytes
from run_windows_batch import deliver


def dispatch_family(project, store, catalog, catalog_hash, row, directory, members):
    family = row["scene_family_id"]
    ref_dir = store / "formal-references" / catalog_hash
    choice = load(ref_dir / (family + ".json"))
    trial = row["trials"]["B0"]
    require(choice["scene_family_id"] == family and choice["trial_id"] == trial and
            choice["dataset_hash"] == catalog_hash and choice["split"] == row["split"],
            "Reference selection does not match registered split and family")
    verified = VerifiedRuns(store)
    meta, reference, audit = reference_identity(verified, choice["reference_run_id"], choice["readback_run_id"])
    require(reference["manifest_sha256"] == choice["reference_manifest_sha256"] and
            audit["manifest_sha256"] == choice["readback_manifest_sha256"], "Reference pins differ from selection")
    identity = {key:reference["run"][key] for key in ("dataset_id", "dataset_hash", "split", "scene_family_id",
                "scene_id", "scene_version", "trial_id", "condition_id")}
    require(identity["dataset_id"] == catalog["dataset_id"] and identity["dataset_hash"] == catalog_hash and
            identity["scene_family_id"] == family and identity["scene_id"] == row["scene_id"] and
            identity["condition_id"] == "B0" and identity["split"] == row["split"] and identity["trial_id"] == trial,
            "Selected reference belongs to another experiment")
    require(meta["formal"]["recipe_sha256"] == row["recipe_sha256"] and
            meta["camera_sha256"] == row["camera_sha256"], "Recipe or camera changed")
    identity.update(scene_hash=meta["blend_sha256"], camera_set_hash=meta["camera_sha256"])
    payload = store / "runs" / choice["reference_run_id"] / "payload"
    files = {n:(payload/n).read_bytes() for n in ("input.png","cameras.json")}
    require(sha_bytes(files["input.png"]) == meta["input_sha256"] and
            sha_bytes(files["cameras.json"]) == meta["camera_sha256"], "Source photograph or camera changed")
    files.update({"code/"+n:v for n,v in members.items()})
    job={"schema":"el-worker-handoff-2","experiment_id":"stage2-single-image",
         "trial_id":trial,"condition":"B0","protocol":"1.0","identity":identity,
         "reference":run_pin(reference),"readback":run_pin(audit),
         "worker_source_sha256":sha_bytes(zip_bytes(members)),
         "files":{n:sha_bytes(v) for n,v in files.items()}}
    files["handoff.json"]=encoded(job)
    packed=zip_bytes(files)
    path=directory/(family+".zip")
    publish(path,packed)
    binding={"schema":"el-formal-dispatch-1","identity":identity,"reference":job["reference"],
             "readback":job["readback"],"handoff_sha256":sha_bytes(files["handoff.json"]),
             "container_sha256":sha_bytes(packed),"worker_source_sha256":job["worker_source_sha256"]}
    publish(store/"formal"/catalog_hash/"dispatch"/(trial+".json"),encoded(binding))
    return {"name":path.name,"trial_id":trial,"bytes":len(packed),"sha256":sha_bytes(packed),
            "handoff_sha256":sha_bytes(files["handoff.json"])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("project","store","catalog","exchange"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--catalog-sha256",required=True)
    p.add_argument("--batch",required=True)
    p.add_argument("--families",nargs="+",required=True)
    a=p.parse_args()
    project,store=a.project.resolve(),a.store.resolve()
    require(sha(a.catalog)==a.catalog_sha256,"Registry checksum changed")
    catalog=load(a.catalog)
    require(catalog["schema"]=="el-vision-family-roster-1" and len(catalog["families"])==24,"Unexpected registry")
    rows={r["scene_family_id"]:r for r in catalog["families"]}
    require(len(a.families)==len(set(a.families)) and set(a.families)<=rows.keys(),"Unknown or duplicate request")
    require(a.batch.replace("-","").isalnum() and not a.batch.startswith("-"), "Invalid batch name")
    if any(rows[f]["split"]=="test" for f in a.families):
        from stage2_opening import verify_opening
        verify_opening(store, catalog, a.catalog_sha256)
    local=store/"stage2"/a.catalog_sha256/"dispatches"/a.batch
    local.mkdir(exist_ok=True,parents=True)
    members=source_files(formal=True)
    jobs=[dispatch_family(project,store,catalog,a.catalog_sha256,rows[f],local,members) for f in a.families]
    runner=project/"scripts/run_windows_batch.py"
    batch={"schema":"el-windows-b0-batch-1","batch_id":a.batch,"runner_sha256":sha(runner),
           "family_catalog_sha256":a.catalog_sha256,"jobs":jobs}
    file=local/"batch.json";publish(file,encoded(batch))
    for dest in ("Jobs","Returns"):
        (a.exchange/dest/a.batch).mkdir(exist_ok=True,parents=True)
    remote=a.exchange/"Jobs"/a.batch
    for job in jobs:deliver(local/job["name"],remote/job["name"])
    deliver(runner,remote/runner.name)
    deliver(file,remote/file.name)
    print(json.dumps({"batch":a.batch,"sha256":sha(file),"runner_sha256":sha(runner),"families":a.families}))


if __name__=="__main__":
    main()
