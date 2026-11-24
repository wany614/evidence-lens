"""Account for unique study runs, actual resource samples, direct timers and retained bytes."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

from evidence_lens._bundle import load, save
from evidence_lens._verify import require
from dual_records import selected,stage


def read_meter(folder):
    line_count=0
    measured_cpu=0.
    covered=0.
    maximum_gpu_process=None
    maximum_device=None
    preceding=0.
    meter=folder/"telemetry.csv"
    if meter.is_file():
        with meter.open(encoding="utf-8",newline="") as f:
            for r in csv.DictReader(f):
                line_count+=1
                t=float(r["elapsed_seconds"])
                dt=max(0.,t-preceding)
                if r.get("tree_cpu_percent"):
                    measured_cpu+=dt*float(r["tree_cpu_percent"])/100
                    covered+=dt
                preceding=t
                g=r.get("gpu_process_memory_bytes")
                if g:
                    try: maximum_gpu_process=max(int(g),maximum_gpu_process or 0)
                    except ValueError: pass
                if r.get("gpu_devices"):
                    try:
                        devs=json.loads(r["gpu_devices"])
                        for dev in devs:
                            if not isinstance(dev,dict):continue
                            b=dev.get("memory.used",dev.get("memory_used_mib",dev.get("used_mib")))
                            if b is not None:maximum_device=max(int(b)*1048576,maximum_device or 0)
                    except (ValueError,TypeError): pass
    return dict(samples=line_count,sampled_tree_cpu_seconds=measured_cpu,cpu_sample_interval_seconds=covered,
                peak_monitored_gpu_process_bytes=maximum_gpu_process,peak_reported_device_bytes=maximum_device)


def native_cpu(folder,label):
    """Count only direct timers for this worker. Some summaries also embed their parents' timers."""
    if label.startswith("b-fit-") and (folder/"asset.json").is_file():
        return load(folder/"asset.json").get("runtime",{}).get("process_cpu_seconds")
    if label.startswith("b-bank-performance"):
        files=sorted(folder.glob("*-threads.json"))
        return sum(load(p)["process_cpu_seconds"] for p in files) if files else None
    for name,field in (("sfm.json","source_cpu_seconds"),("sample.json","cpu_core_seconds"),
                       ("scores.json","cpu_seconds"),("evaluation.json","cpu_core_seconds"),("fits.json","cpu_core_seconds")):
        if (folder/name).is_file():return load(folder/name).get(field)
    for name in ("source.json","truth.json","visibility.json"):
        if not (folder/name).is_file():continue
        d=load(folder/name).get("resource",{})
        duration=d.get("worker_cpu_seconds",d.get("cpu_seconds"))
        if duration is not None:return duration+d.get("child_cpu_seconds",0)
    return None


def account(s):
    rows=[]; seen={}; unique={}; bucket_bytes=0
    inodes=set(); allocated=0
    for bucket in ("runs","quarantine"):
        for folder in sorted((s/bucket).iterdir()):
            end=folder/"completion.json"
            if not end.is_file():continue
            r=load(folder/"run.json")
            label=r.get("trial_id") or ""
            if not (r.get("experiment_id")=="dual-track-b" or label.startswith("b-")):continue
            complete=load(end)
            files=load(folder/"manifest.json")["files"]
            payload=[f for f in files if f["path"].startswith("payload/")]
            for f in payload:
                size=unique.setdefault(f["sha256"],f["bytes"])
                require(size==f["bytes"],"same hash has a different size")
            local_bytes=sum(f["bytes"] for f in files)
            bucket_bytes+=local_bytes
            for path in folder.rglob("*"):
                if path.is_symlink() or not path.is_file():continue
                st=path.stat();key=(st.st_dev,st.st_ino)
                if key not in inodes:
                    allocated+=st.st_blocks*512
                    inodes.add(key)
            logical=sum(v["bytes"] for v in payload)
            time=float(r.get("elapsed_seconds_monotonic") or 0)
            fit=(r["run_kind"]=="model_fit" and label.startswith("b-fit-"))
            native=native_cpu(folder/"payload",label)
            meter=read_meter(folder)
            logical_cores=load(folder/"environment.json").get("cpu_logical")
            accounted=max(native or 0.,meter["sampled_tree_cpu_seconds"])
            unobserved=max(0.,time-meter["cpu_sample_interval_seconds"])
            row=dict(run_id=folder.name,trial=label,kind=r["run_kind"],bucket=bucket,
                 status=complete["execution_status"],manifest_sha256=complete["manifest_sha256"],
                 worker_seconds=time,peak_tree_rss_bytes=r.get("sampled_peak_tree_rss_bytes"),
                 exit_code=r.get("exit_code"),native_cpu_core_seconds=native,logical_cores=logical_cores,
                 accounted_cpu_core_seconds=accounted,uncovered_wall_seconds=unobserved,
                 extra_core_seconds_if_uncovered_saturated=unobserved*logical_cores if logical_cores else None,
                 logical_manifest_bytes=local_bytes,payload_bytes=logical,upstream_fit=fit,**meter)
            rows.append(row)
            if folder.name not in seen:seen[folder.name]=row
    fits=[v for v in seen.values() if v["upstream_fit"]]
    return dict(schema="el-many-resource-account-1",scope="native and received B runs through this snapshot; each run charged once",
        fit_attempts=len(fits),gpu_fit_worker_seconds=sum(x["worker_seconds"] for x in fits),
        fit_accounting="Full native fit worker wall includes the CUDA path, startup and native validation. Standalone auxiliary GPU jobs must be added separately.",
        cpu_sampled_tree_core_seconds=sum(x["sampled_tree_cpu_seconds"] for x in seen.values()),
        accounted_cpu_core_seconds=sum(x["accounted_cpu_core_seconds"] for x in seen.values()),
        measured_native_core_seconds=sum(x["native_cpu_core_seconds"] or 0 for x in seen.values()),
        max_allowance_uncovered_seconds=sum(x["extra_core_seconds_if_uncovered_saturated"] or 0 for x in seen.values()),
        cpu_scope="Per-run accounting takes the larger of integrated sampled tree CPU and any available direct worker timer. The gap allowance charges all logical cores for intervals without a tree delta. Short-lived children may escape the process sampler.",
        cpu_covered_wall_seconds=sum(x["cpu_sample_interval_seconds"] for x in seen.values()),
        unique_payload_bytes=sum(unique.values()),unique_payload_member_count=len(unique),
        all_manifest_logical_bytes=bucket_bytes,allocated_B_archive_bytes=allocated,
        allocated_scope="Mac terminal B recorder directories, allocated blocks, hardlinks counted once. Transfer and working copies are reported separately.",
        free_bytes=shutil.disk_usage(s).free,runs=rows)


def main(a):
    if a.record:
        pins={k:selected(a.store,v) for k,v in {"study":"b-study-registration","results":"b-scientific-results"}.items()}
        print(stage(a.store,"b-resources-and-failures","report",pins,{"scope":"through end of full scientific evaluation"},
            ["{python}","scripts/many_resources.py","--store",str(a.store),"--out","{out}"],
            "payload/resources.json",360000))
        return
    result=account(a.store)
    if a.out:
        result["run_id"]=os.getenv("EL_RUN_ID")
        result["allocated_store_bytes_including_prior_A"]=int(subprocess.check_output(
            ["du","-sk",str(a.store)],text=True).split()[0])*1024
        save(a.out/"resources.json",result)
    else:
        print(json.dumps({k:v for k,v in result.items() if k!="runs"},sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store",type=Path,required=True)
    p.add_argument("--out",type=Path)
    p.add_argument("--record",action="store_true")
    main(p.parse_args())
