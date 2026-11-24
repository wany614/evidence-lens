"""Verify Windows' six-confirmation storage receipt against the immutable queue and all original native deliveries."""

import argparse
import json
import os
from pathlib import Path

from dual_records import selected, resolve, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require


def register(store, disk, ledger):
    pins={"queue":selected(store,"b-confirmation-fit-queue"),"model":selected(store,"b-model-freeze"),
          "science":selected(store,"b-upstream-closeout")}
    roots=store,store/"history"
    queue=load(resolve(roots,pins["queue"]));science=load(resolve(roots,pins["science"]))
    for job in queue["fits"]:
        pins[job["tag"]]=science["cases"][job["tag"]]["proof"]
    cmd=["{python}","-B","{source}/scripts/many_upstream_disk.py","worker","--out","{out}","--config","{config}"]
    for role in [*pins,"disk","return_ledger"]:cmd+=["--input",role,"{input:"+role+"}"]
    return stage(store,"b-windows-confirmation-disk","archive_verify",pins,
                 {"scope":"exact six confirmation projects on the native D drive","measurement":"final allocated file bytes plus sampled temporary logical lengths"},
                 cmd,"payload/storage.json",18000,extra_files={"disk":disk,"return_ledger":ledger})


def verify(cfg, paths, out):
    disk,ledger,queue,science=(load(paths[key]) for key in ("disk","return_ledger","queue","science"))
    require(disk["schema"]=="el-windows-dual-track-b-confirmation-storage-1" and disk["units"]=="bytes" and disk["volume"]=="D:"
            and ledger["schema"]=="el-windows-dual-track-b-confirmation-queue-return-1" and disk["return_ledger_sha256"]==sha(paths["return_ledger"])
            and disk["queue_run"]==ledger["queue_run"]==queue["run_id"] and disk["queue_sha256"]==ledger["queue_sha256"]==sha(paths["queue"])
            and ledger["queue_manifest_sha256"]==cfg["sources"]["queue"]["manifest_sha256"]
            and disk["model_freeze_sha256"]==cfg["sources"]["model"]["sha256"] and ledger["model_freeze"]==cfg["sources"]["model"],"Windows queue, freeze or disk ledger changed")
    before,after=disk["before_snapshot"],disk["after_snapshot"]
    require(len(queue["fits"])==len(ledger["jobs"])==ledger["confirmation_runs"]==6 and not before["cases_present"] and not before["errors"]
            and not after["errors"] and not before["reparse_points_skipped"] and not after["reparse_points_skipped"],"incomplete native storage survey")
    for ticket,case in zip(queue["fits"],ledger["jobs"]):
        tag=ticket["tag"];native=load(paths[tag])["native"]
        require(case["trial"]==ticket["trial"] and native["run_id"]==case["run_id"]==after["six_trial_runs"][ticket["trial"]]
                and native["manifest_sha256"]==case["manifest_sha256"] and case["worker_status"]=="succeeded"
                and case["package"]["bytes"]==native["bytes"] and case["package"]["sha256"]==native["container_sha256"]
                and case["input_zip_sha256"]==ticket["packet"]["sha256"] and case["input_dispatch_sha256"]==ticket["pin"]["sha256"]
                and case["H_reference_reads"]==0,"return ledger differs from an original accepted native fit")
    combined=disk["combined_scoped_delta"];groups=disk["group_delta"];classes=disk["disjoint_storage_classes_delta"]
    require(set(groups)==set(after["groups"])==set(before["groups"]),"storage category drift")
    for key,value in combined.items():
        for name,delta in groups.items():
            require(after["groups"][name][key]-before["groups"][name][key]==delta[key],"inconsistent native endpoint subtraction")
        require(sum(v[key] for v in groups.values())==sum(v[key] for v in classes.values())==value,"native storage partitions do not reconcile")
    samples=disk["sampling"]
    require(set(samples["per_case"])==set(after["six_trial_runs"]) and samples["samples"]>0
            and max(v["combined_peak_logical_bytes"] for v in samples["per_case"].values())==samples["max_combined_scratch_logical_bytes"]
            and abs(ledger["totals"]["B_worker_wall_s_successes_plus_previous_nine"]-science["charged_gpu_seconds"])<1e-7,
            "sampled temporary footprint or native resource accounting differs")
    save(out/"storage.json",{"schema":"el-many-confirmation-storage-readback-1","run_id":os.environ["EL_RUN_ID"],
         "queue":cfg["sources"]["queue"],"model_freeze":cfg["sources"]["model"],"upstream":cfg["sources"]["science"],
         "native_disk_receipt_sha256":sha(paths["disk"]),"native_six_returns_sha256":sha(paths["return_ledger"]),
         "verified_native_runs":after["six_trial_runs"],"totals":combined,"disjoint_classes":classes,
         "allocation_method":disk["allocation_method"],"scope":disk["scope"],"sampling":{"count":samples["samples"],
         "max_gap_seconds":samples["max_actual_gap_seconds"],"max_combined_scratch_logical_bytes":samples["max_combined_scratch_logical_bytes"],
         "per_case_peak_logical_bytes":{tag:data["combined_peak_logical_bytes"] for tag,data in samples["per_case"].items()},
         "interpretation":samples["basis"]},"source":"Windows native local filesystem counters; Mac independently verifies six original run/container identities and both endpoint/delta arithmetic reconciliations."})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("record","worker"))
    for name in ("store","disk","ledger","config","out"):p.add_argument("--"+name,type=Path)
    p.add_argument("--input",action="append",nargs=2,default=[]);a=p.parse_args()
    if a.action=="record":print(json.dumps(register(a.store.resolve(),a.disk,a.ledger)))
    else:verify(load(a.config),{k:Path(v) for k,v in a.input},a.out)
