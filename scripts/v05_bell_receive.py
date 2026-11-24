"""Verify and import one original Windows bell render and issue a Mac readback receipt."""

import argparse
import json
import os
from pathlib import Path
import zipfile

from dual_records import bind, resolve, selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require, verify_bundle
from t0_reconstruct_intake import receive_verified_zip
from v05_bell_common import CHECKPOINT, OLD, SIZES, SOURCES, TARGETS


def history_roots(store, parents):
    remaining = list(parents)
    seen, old = set(), {}
    while remaining:
        ref = remaining.pop()
        rid = ref["run_id"]
        if rid in seen:
            continue
        seen.add(rid)
        directory = store/"runs"/rid
        if directory.is_dir():
            require(sha(directory/"manifest.json") == ref["manifest_sha256"],"changed existing parent")
            records = load(directory/"inputs.json")
            remaining.extend(records["parents"])
            if records.get("retry_of"):
                remaining.append(records["retry_of"])
        else:
            verified = verify_bundle((store/"history/runs"/rid).resolve())
            require(verified["manifest_sha256"] == ref["manifest_sha256"],"wrong historical boundary")
            old[rid] = verified
    return old


def intake(s, return_receipt):
    signed=load(return_receipt)
    rid,manifest,zip_sha=signed["run_id"],signed["manifest_sha256"],signed["sha256"]
    require(signed["package"]==rid+".zip" and signed["status"]=="succeeded" and signed["worker_exit_code"]==0,
            "native return disposition differs")
    archive=return_receipt.parent/signed["package"]
    frame_proof=return_receipt.parent/"frame-verification.json"
    prelaunch=return_receipt.parent/"controller-prelaunch.json"
    require(archive.stat().st_size==signed["bytes"] and sha(frame_proof)==signed["frame_verification_sha256"]
            and sha(prelaunch)==signed["prelaunch_record_sha256"] and sha(return_receipt.parent/"manifest.json")==manifest,
            "native frame, manifest or prelaunch side receipt differs")
    ticket = selected(s,"a-bell-packet-six")
    dispatch=load(resolve((s,s/"history"),ticket))
    require(sha(archive)==zip_sha,"returned native archive container differs")
    with zipfile.ZipFile(archive) as z:
        info=json.loads(z.read("run.json")); inp=json.loads(z.read("inputs.json")); cp=json.loads(z.read("completion.json"))
        require(info["run_id"]==rid and info["execution_status"]=="succeeded" and info["run_kind"]=="render"
                and info["experiment_id"]=="dual-track-a" and info["trial_id"]=="a-bell-gpu" and cp["manifest_sha256"]==manifest,
                "incorrect Windows trial or termination")
        all_inputs={row["sha256"] for i in inp["inputs"] for row in i["files"]}
        local_packet=resolve((s,s/"history"),bind(s,ticket["run_id"],"payload/bell-gpu.zip"))
        import hashlib
        with zipfile.ZipFile(local_packet) as packed:
            expected={hashlib.sha256(packed.read(n)).hexdigest() for n in packed.namelist()}
        linked=any(p["run_id"]==ticket["run_id"] and p["manifest_sha256"]==ticket["manifest_sha256"] for p in inp["parents"])
        require(dispatch["archive_sha256"] in all_inputs or expected<=all_inputs,
                "neither the issued archive nor its complete verified files occurred in the Windows input record")
        require(linked or dispatch["archive_sha256"] in all_inputs,"no link to immutable Mac packet")
    anchors=history_roots(s,inp["parents"])
    result=receive_verified_zip(archive,s,zip_sha,Path(__file__).resolve().parents[1]/"configs/storage.json",anchors,ticket)
    require(result["status"] in ("accepted","identical") and result["run_id"]==rid and result["manifest_sha256"]==manifest,
            "native cross-device recorder verification failed: "+str(result))
    pins={"ticket":ticket,"native":bind(s,rid,"payload/predictions.json"),"probe":bind(s,rid,"payload/depth-probe.json")}
    cmd=["{python}","-B","{source}/scripts/v05_bell_receive.py","worker","--config","{config}","--out","{out}",
         "--native","{input:native}","--probe","{input:probe}","--ticket","{input:ticket}"]
    require(signed["predictions_sha256"]==pins["native"]["sha256"] and signed["depth_probe_sha256"]==pins["probe"]["sha256"]
            and signed["input_zip_sha256"]==dispatch["archive_sha256"],"return receipt disagrees with original artifacts")
    return stage(s,"a-bell-gpu-readback","transfer_verify",pins,{"archive_sha256":zip_sha,"archive_bytes":archive.stat().st_size,
                 "external_sources":list(anchors),"native_record_declares_mac_parent":linked,"return_receipt_sha256":sha(return_receipt),
                 "frame_verification_sha256":sha(frame_proof),"prelaunch_record_sha256":sha(prelaunch)},cmd,"payload/readback.json",24000,
                 {"signed_return":return_receipt,"frame_proof":frame_proof,"prelaunch":prelaunch})


def worker(a):
    cfg=load(a.config);p=load(a.native);ticket=load(a.ticket);probe=load(a.probe)
    require(p["schema"]=="el-v05-bell-render-1" and p["checkpoint_sha256"]==CHECKPOINT==ticket["checkpoint_sha256"]
            and p["registered_packet_run"]==ticket["run_id"] and p["run_id"]==cfg["sources"]["native"]["run_id"]
            and p["source_photo_reads"]==p["target_reference_reads"]==p["new_optimization_steps"]==0
            and p["gaussians"]==4096 and probe["passed"] and sha(a.probe)==p["depth_probe_sha256"],"inconsistent original renderer evidence")
    require(set(p["frames"])=={*SIZES,OLD} and all(set(v)==set(SOURCES+TARGETS) for v in p["frames"].values()),"missing a registered grid or camera")
    save(a.out/"readback.json",{"schema":"el-v05-bell-render-readback-1","run_id":os.environ["EL_RUN_ID"],"native":cfg["sources"]["native"],
         "packet":cfg["sources"]["ticket"],"container_sha256":cfg["archive_sha256"],"container_bytes":cfg["archive_bytes"],"native_frames":60,
         "registered_fit":CHECKPOINT,"depth_probe":cfg["sources"]["probe"],"gaussians":p["gaussians"],"runtime":p["runtime"],"target_reference_reads":0})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("receive","worker"))
    for k in ("store","receipt","config","out","native","probe","ticket"):p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.action=="worker":worker(a)
    else:print(json.dumps(intake(a.store.resolve(),a.receipt)))
