"""Authenticate one explicitly dispatched DTU Windows batch and receive its returned scientific originals."""

import argparse
import hashlib
import io
import json
import re
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evidence_lens import stage3_gaussians
from evidence_lens._bundle import HASH, RUN_ID, encoded, load, sha
from evidence_lens._index import writer_gate
from evidence_lens._verify import require, verify_bundle
from evidence_lens.archive import receive
from dtu_reconstruct_windows import OPT_SHA, SETS, TARGETS
from make_handoff import publish, zip_bytes
from stage3_intake import authenticate_reconstruction, authenticate_render, retry_predecessor_pending
from stage3_reference import located


HASH_SHORT = re.compile(r"[0-9a-f]{16}")


def signature(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def members(z, names):
    require(len(z.infolist()) == len(names) and set(z.namelist()) == set(names), "Unexpected handoff ZIP inventory")


def dispatch(folder, batch_sha, pack_sha):
    batch = load(folder / "batch.json")
    require(sha(folder / "batch.json") == batch_sha and batch["schema"] == "el-dtu-worker-batch-1"
            and batch["object_id"] in ("set1", "set6"), "Wrong Mac batch")
    diagnostic = batch["batch"] == "d1-dtu-set1-1024-diagnostic"
    require(not diagnostic or batch["object_id"]=="set1", "1024 is restricted to the registered development object")
    suffix = "1024-diagnostic" if diagnostic else "256"
    name = "d1-dtu-" + batch["object_id"] + "-" + suffix
    require(batch["batch"] == name and len(batch["jobs"]) == 2 and
            {r["condition"] for r in batch["jobs"]} == set(SETS), "Incomplete registered pair")
    pack = folder / (name + ".zip")
    require(sha(pack) == pack_sha, "Wrong Mac dispatch container")
    jobs = {}
    with zipfile.ZipFile(pack) as z:
        source = batch["worker_files"]
        members(z, {"batch.json", *("worker/" + k for k in source), *(j["path"] for j in batch["jobs"])})
        require(z.read("batch.json") == (folder / "batch.json").read_bytes(), "Batch copy changed")
        worker = {n: z.read("worker/" + n) for n in source}
        require(all(signature(v) == source[n] for n,v in worker.items()) and
                signature(zip_bytes(worker))["sha256"] == batch["worker_sha256"],
                "Windows worker code differs from dispatch")
        require(sha(Path(stage3_gaussians.__file__)) == source["src/evidence_lens/stage3_gaussians.py"]["sha256"],
                "Local reconstruction checker differs from dispatched implementation")
        for row in batch["jobs"]:
            raw = z.read(row["path"])
            require(signature(raw) == {k: row[k] for k in ("bytes", "sha256")}, "Changed job container")
            with zipfile.ZipFile(io.BytesIO(raw)) as job_zip:
                body = job_zip.read("job.json")
                require(signature(body)["sha256"] == row["job_sha256"], "Changed job manifest")
                job = json.loads(body)
                ids = SETS[row["condition"]]
                names = {"optimization.json", "source_cameras.json", "render_cameras.json", *("sources/"+s+".png" for s in ids)}
                members(job_zip, names | {"job.json"})
                require(job["schema"] == "el-dtu-job-1" and job["object_id"] == batch["object_id"]
                        and job["condition"] == row["condition"] and job["generator_source_ids"] == ids
                        and job["target_ids"] == TARGETS and job["optimization_steps"] == (1024 if diagnostic else 256)
                        and job["trial_id"] == row["trial_id"] == f"d1-dtu-{job['object_id']}-{job['condition']}-{suffix}"
                        and job["experiment_id"] == ("d1-dtu-debug-only-1024" if diagnostic else "d1-dtu-source-only-256")
                        and bool(job.get("diagnostic_decision"))==diagnostic and job["worker_sha256"]==batch["worker_sha256"]
                        and job["camera_plan_sha256"] == batch["camera_plan_sha256"]
                        and set(job["files"]) == names and job["files"]["optimization.json"]["sha256"] == OPT_SHA
                        and all(signature(job_zip.read(k)) == job["files"][k] for k in names),
                        "Unexpected job identity, source subset or camera/optimizer bytes")
                jobs[row["job_sha256"]] = job
    return batch, jobs


def original(store, pin, name):
    path = located(store, pin)
    require(sha(path/"manifest.json") == pin["manifest_sha256"] and load(path/"completion.json")["manifest_sha256"] == pin["manifest_sha256"],
            "Original receipt differs")
    expected = next((x for x in load(path/"manifest.json")["files"] if x["path"] == name), None)
    member = path/name
    require(expected and member.is_file() and not member.is_symlink() and member.stat().st_size == expected["bytes"]
            and sha(member) == expected["sha256"], "Original member differs: " + name)
    return member


def check_sources(store, j):
    source, rig, budget = j["reference"], j["readback"], j["budget"]
    sc = "source_cameras.json"
    rc = "render_cameras.json"
    plan = original(store, rig, "payload/camera-plan.json")
    require(sha(plan) == j["camera_plan_sha256"] and
            sha(original(store, budget, "payload/optimizer.json")) == j["files"]["optimization.json"]["sha256"],
            "Camera plan or budget original differs")
    for file in (sc, rc):
        require(sha(original(store, rig, f"payload/condition/{j['condition']}/{file}")) == j["files"][file]["sha256"],
                "Registered camera file differs")
    meta_path = original(store, source, "payload/source-images.json")
    meta = load(meta_path)
    require(j["identity"]["scene_hash"] == sha(meta_path) and meta["object_id"] == j["object_id"]
            and meta["camera_plan_sha256"] == j["camera_plan_sha256"] and meta["opening_sha256"] == j.get("opening_sha256"),
            "Wrong source conversion")
    if j.get("diagnostic_decision"):
        ds=original(store,j["diagnostic_decision"],"payload/diagnostic-decision.json")
        decision=load(ds)
        require(sha(ds)==j["diagnostic_decision_sha256"] and decision["run_id"]==j["diagnostic_decision"]["run_id"]
                and decision["schema"]=="el-dtu-development-diagnostic-1" and decision["object_id"]==j["object_id"]=="set1"
                and decision["conditions"]==["M2","M4"] and decision["updates"]==j["optimization_steps"]==1024
                and decision["external_updates"]==256 and decision["maximum_diagnostic_jobs"]==2,
                "Unauthorized diagnostic budget")
        base=load(store/"dtu/intake/d1-dtu-set1-256"/j["condition"]/"selected.json")
        require(base["source_conversion"]==source and base["registration"]==rig and base["budget"]==budget and
                base["job_sha256"]==j["baseline_handoff_sha256"], "Diagnostic detached from main trial")
    if j["object_id"] == "set6":
        op = original(store, j["opening"], "payload/opening.json")
        require(sha(op) == j["opening_sha256"] == sha(store/"dtu-authority/dtu-crop256-v1/opening.json")
                and load(op)["schema"] == "el-dtu-opening-1" and load(op)["object_ids"] == ["set6"],
                "Wrong held-out opening")
    else:
        require(j.get("opening") is None and j.get("opening_sha256") is None, "Unexpected development opening")
    for sid in j["generator_source_ids"]:
        f = "sources/"+sid+".png"
        require(sha(original(store, source, "payload/"+f)) == j["files"][f]["sha256"]
                == meta["views"][sid]["rgb"]["sha256"], "Source bytes differ")


def verify_stage(path, receipt, j, batch, jobsha, producer):
    checked = verify_bundle(path)
    r = checked["run"]
    kind = receipt["run_kind"]
    require((r["run_id"], checked["manifest_sha256"], r["run_kind"], r["execution_status"], r["exit_code"]) ==
            (receipt["run"], receipt["manifest_sha256"], kind, receipt["status"], receipt["exit_code"])
            and r["trial_id"] == j["trial_id"] and r["experiment_id"] == j["experiment_id"]
            and all(r[k] == v for k,v in j["identity"].items()), "Sealed return has another identity")
    parents = [j["readback"], j["reference"], j["budget"]]
    if j.get("diagnostic_decision"):
        parents.append(j["diagnostic_decision"])
    if j.get("opening"):
        parents.append(j["opening"])
    if kind == "render":
        parents.append(producer)
    cfg = load(path/"config.resolved.yaml")
    require(checked["inputs"]["parents"] == parents and cfg["formal_handoff_sha256"] == jobsha
            and cfg["worker_sha256"] == batch["worker_sha256"] and cfg["camera_plan_sha256"] == j["camera_plan_sha256"]
            and cfg["source_conversion"] == j["reference"] and cfg["camera_registration"] == j["readback"]
            and cfg["production_budget"] == j["budget"] and cfg["opening_sha256"] == j["opening_sha256"]
            and cfg.get("diagnostic_decision")==j.get("diagnostic_decision")
            and cfg.get("diagnostic_decision_sha256")==j.get("diagnostic_decision_sha256")
            and cfg["optimizer_sha256"] == OPT_SHA and cfg["optimization_steps"] == j["optimization_steps"],
            "Returned stage has another source, parent or registered budget")
    if r["execution_status"] != "succeeded":
        return r
    source_inventory = {v["path"]: {k:v[k] for k in ("bytes","sha256")} for v in load(path/"code_snapshot.json")["files"]}
    require(source_inventory == batch["worker_files"], "Frozen code snapshot differs from the dispatched worker")
    roles = {x["role"]:x for x in checked["inputs"]["inputs"]}
    def match(role, owner, digest):
        v=roles[role]
        require(v["parent_run_id"] == owner["run_id"] and len(v["files"]) == 1 and v["files"][0]["sha256"] == digest,
                "Unexpected recorded input: "+role)
    require(set(roles) == ({"optimizer","cameras",*j["generator_source_ids"]} if kind=="generation" else
                          {"optimizer","cameras","checkpoint","reconstruction"}), "Unexpected worker input scope")
    match("optimizer",j["budget"],OPT_SHA)
    camera = "source_cameras.json" if kind=="generation" else "render_cameras.json"
    match("cameras",j["readback"],j["files"][camera]["sha256"])
    if kind=="generation":
        for s in j["generator_source_ids"]:
            match(s,j["reference"],j["files"]["sources/"+s+".png"]["sha256"])
    else:
        p=load(path/"payload/prediction.json")
        match("checkpoint",producer,p["checkpoint_sha256"])
        match("reconstruction",producer,p["reconstruction_sha256"])
    return r


def camera_probe(payload, camera_path, j):
    evidence=load(payload/"dtu-camera-probe.json")
    cameras=load(camera_path)["views"]
    require(evidence["schema"]=="el-dtu-native-gpu-probe-1" and evidence["passed"] is True
            and evidence["render_camera_sha256"]==j["files"]["render_cameras.json"]["sha256"]
            and evidence["prediction_sha256"]==sha(payload/"prediction.json")
            and evidence["original_C_probe_sha256"]==sha(payload/"depth-probe.json") and set(evidence["cameras"])==set(cameras),
            "Incorrect DTU measured camera certificate")
    for key,c in cameras.items():
        v=evidence["cameras"][key]; K=np.asarray(c["K"])
        require(v["passed"] is True and v["measured_empty_alpha"]==0 and np.array_equal(K,v["integer_K"])
                and np.array_equal(c["c2w"],v["c2w"])
                and np.max(np.abs(np.array([[1,0,.5],[0,1,.5],[0,0,1]])@K-v["measured_gsplat_edge_K"]))<1e-4
                and len(v["cases"])==2, "Camera calibration readback mismatch: "+key)
        for q in v["cases"]:
            require(abs(q["actual_ED_m"]-q["expected_ED_m"])<=1e-4 and abs(q["actual_D_m"]-q["expected_D_m"])<=1e-4 and
                    abs(q["actual_alpha"]-q["expected_alpha"])<=5e-4 and q["actual_alpha"]-q["neighbour_max_alpha"]>.004
                    and np.max(np.abs(np.asarray(q["actual_linear_rgb"])-q["expected_linear_rgb"]))<.0005,
                    "Camera expectation, opacity or colour mismatch: "+key)


def close_transport(root, returns, batch, batch_sha, pack_sha, jobs, stages, label):
    """Bind every journaled sealed return and the independent Windows completion lists."""
    base = batch["batch"]
    summaries = sorted(p for p in returns.glob(base+"-*.json")
                       if p.name == base+"-worker-summary.json" or HASH_SHORT.fullmatch(p.name[len(base)+1:-5]))
    if not summaries:
        return
    from_manifest = {}
    source_files = {}
    seen_summaries = set()
    for file in sorted(summaries, key=lambda p: (p.name != base+"-worker-summary.json", p.name)):
        digest = sha(file)
        if digest in seen_summaries:
            continue
        seen_summaries.add(digest)
        s = load(file)
        require(s["schema"]=="el-dtu-worker-completion-1" and s["batch"]==base and s["batch_sha256"]==batch_sha,
                "Wrong worker completion manifest")
        for row in s["jobs"]:
            h=row["job_sha256"]
            require(h in jobs and row["trial_id"]==jobs[h]["trial_id"], "Unregistered summary job")
            for rec in row["runs"]:
                rid=rec["run"]
                require(rec["handoff_sha256"]==h and rec["trial_id"]==row["trial_id"] and RUN_ID.fullmatch(rid)
                        and from_manifest.setdefault(rid,rec)==rec, "Conflicting return summary")
        source_files[file.name]=signature(file.read_bytes())
    wanted={v["run_id"] for x in stages.values() for v in x["attempts"]}
    if not wanted <= set(from_manifest):
        return
    require(set(from_manifest)==wanted, "Worker summary names a return that has not been verified")
    for rid, rec in from_manifest.items():
        file=returns/(rid+".receipt.json")
        require(file.is_file() and load(file)==rec and rec["package"]==rid+".zip", "Worker summary and file receipt differ")
        source_files[file.name]=signature(file.read_bytes())
        source_files[rec["package"]]={"sha256":rec["sha256"],"bytes":rec["bytes"]}
    data={"schema":"el-dtu-batch-transport-1","batch":base,"source":label,"batch_sha256":batch_sha,
          "dispatch_container_sha256":pack_sha,"files":source_files,
          "selections":{c:sha(root/c/"selected.json") for c in SETS},
          "producer_return_bytes":sum(r["bytes"] for r in from_manifest.values()),
          "observed_provider_wire_bytes":None,
          "transfer_scope":"Verified FileProvider/local file bytes. Network wire bytes depend on cache residence and were not separately instrumented."}
    publish(root/"transport-manifest.json",encoded(data))


def receive_batch(store, folder, batch_sha, pack_sha, returns, label, limits):
    batch,jobs=dispatch(folder,batch_sha,pack_sha)
    root=store/"dtu/intake"/batch["batch"]
    receipts={h:{"generation":[],"render":[]} for h in jobs}
    for file in sorted(returns.glob("*.receipt.json")):
        receipt=load(file);h=receipt["handoff_sha256"];rid=receipt["run"]
        require(h in jobs and receipt["schema"]=="el-worker-return-1" and RUN_ID.fullmatch(rid)
                and file.name==rid+".receipt.json" and receipt["package"]==rid+".zip"
                and receipt["trial_id"]==jobs[h]["trial_id"] and receipt["run_kind"] in ("generation","render")
                and HASH.fullmatch(receipt["manifest_sha256"]) and HASH.fullmatch(receipt["sha256"]),
                "Wrong return identity in the selected batch folder")
        receipts[h][receipt["run_kind"]].append((file,receipt))
    result={}
    for h,j in jobs.items():
        c=j["condition"];out=root/c;out.mkdir(parents=True,exist_ok=True)
        with writer_gate(out):
            check_sources(store,j)
            selected={};attempts=[];science={};pending="waiting_generation"
            for kind in ("generation","render"):
                good=[]
                for file,receipt in sorted(receipts[h][kind],key=lambda item:item[1]["run"]):
                    rid=receipt["run"];archive=returns/receipt["package"]
                    if not archive.is_file() or archive.stat().st_size!=receipt["bytes"]:
                        pending="waiting_complete_"+kind;continue
                    require(not archive.is_symlink() and sha(archive)==receipt["sha256"],"Return container digest differs")
                    if retry_predecessor_pending(archive,store):
                        pending="waiting_retry_parent";continue
                    pin={"run_id":rid,"manifest_sha256":receipt["manifest_sha256"]}
                    marker=out/"transfers"/(rid+".json")
                    if marker.exists():
                        transfer=load(marker)
                        require(transfer["worker_receipt_sha256"]==sha(file) and transfer["container_sha256"]==receipt["sha256"]
                                and transfer["manifest_sha256"]==pin["manifest_sha256"],"Changed previous return")
                    else:
                        x=receive(archive,store,receipt["sha256"],limits,shelf=store.parent/"cold-shelf")
                        if x["status"]=="deferred_capacity":
                            pending="waiting_capacity";continue
                        require(x["status"] in ("accepted","identical") and (x["run_id"],x["manifest_sha256"])==(rid,pin["manifest_sha256"]),
                                "Archive receive failed: "+str(x.get("reason")))
                        transfer={"schema":"el-dtu-return-intake-1",**pin,"source":label,"worker_receipt_sha256":sha(file),
                                  "container_sha256":receipt["sha256"],"container_bytes":receipt["bytes"],
                                  "batch_sha256":batch_sha,"job_sha256":h,"archive_receive":x}
                        publish(marker,encoded(transfer))
                    path=located(store,pin)
                    meta=verify_stage(path,receipt,j,batch,h,selected.get("generation"))
                    attempts.append({**pin,"stage":kind,"status":meta["execution_status"],"retry_of":meta["retry_of_run_id"],
                                     "return_sha256":receipt["sha256"],"return_bytes":receipt["bytes"]})
                    if meta["execution_status"]=="succeeded":
                        good.append(pin)
                require(len(good)<=1,"Multiple successful attempts for one formal stage")
                if not good:
                    if any(x["stage"]==kind for x in attempts):pending="recorded_failure_"+kind
                    break
                selected[kind]=good[0]
                if kind=="generation":
                    science[kind]=authenticate_reconstruction(located(store,good[0]),j,stage3_gaussians)
                    if j.get("diagnostic_decision"):
                        base=load(store/"dtu/intake/d1-dtu-set1-256"/c/"selected.json")
                        prior=base["numerical_readback"]["generation"]
                        require(prior["gaussian_count"]==science[kind]["gaussian_count"]==4096 and
                                prior["initial_parameter_sha256"]==science[kind]["initial_parameter_sha256"] and
                                prior["initial_checkpoint_sha256"]==science[kind]["initial_checkpoint_sha256"],
                                "Diagnostic changed the original Gaussian initialization")
                    pending="waiting_render"
                else:
                    gen=selected["generation"];render=located(store,good[0]);payload=render/"payload"
                    science[kind]=authenticate_render(render,j,gen,located(store,gen),science["generation"])
                    camera=original(store,j["readback"],"payload/condition/"+c+"/render_cameras.json")
                    camera_probe(payload,camera,j)
                    record={"schema":"el-dtu-return-selection-1","object_id":j["object_id"],"condition":c,"trial_id":j["trial_id"],
                            "source_conversion":j["reference"],"registration":j["readback"],"budget":j["budget"],"opening":j["opening"],
                            "opening_sha256":j["opening_sha256"],"batch_sha256":batch_sha,"container_sha256":pack_sha,"job_sha256":h,
                            "worker_sha256":batch["worker_sha256"],**selected,"attempts":attempts,
                            "checkpoint_sha256":science["generation"]["checkpoint_sha256"],"prediction_sha256":sha(payload/"prediction.json"),
                            "probe_sha256":sha(payload/"dtu-camera-probe.json"),"views":science["render"]["views"],
                            "numerical_readback":science}
                    if j.get("diagnostic_decision"):
                        record.update(diagnostic_decision=j["diagnostic_decision"],diagnostic_decision_sha256=j["diagnostic_decision_sha256"],
                                      baseline_handoff_sha256=j["baseline_handoff_sha256"],arm="diagnostic-1024")
                    publish(out/"selected.json",encoded(record))
                    pending="verified"
            result[c]={"state":pending,**selected,"attempts":attempts}
    if all(v["state"]=="verified" for v in result.values()):
        close_transport(root,returns,batch,batch_sha,pack_sha,jobs,result,label)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("store","dispatch","returns","limits"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("batch-sha256","container-sha256","source-label"):
        p.add_argument("--"+name,required=True)
    a=p.parse_args()
    print(json.dumps(receive_batch(a.store.resolve(),a.dispatch,a.batch_sha256,a.container_sha256,a.returns,
                                   a.source_label,a.limits),sort_keys=True))
