"""Seal measured multi-source results and export the portable public release inventory."""

import argparse
import csv
import os
from pathlib import Path
import re
import subprocess

from dual_records import bind,resolve,selected,stage
from evidence_lens._bundle import load,save,sha
from evidence_lens._verify import require
from many_model_release import ARCHIVE
from many_runtime import WHEEL


TAG="many-source-v0.1.0-dev.0"
MANIFEST="B_MANIFEST.json"
PHASE_A="sha256:6e8e837afa0818cdf8c100b494672b46c10bdbd22e522f388983543f0a5b0f23"
STAGES={"baseline":"a-baseline","roles":"b-data-registration","study":"b-study-registration","freeze":"b-model-freeze",
        "opening":"b-confirmation-h-opening","upstream":"b-upstream-closeout","results":"b-scientific-results",
        "independent":"b-independent-arithmetic-check","model":"b-model-package","deployment":"b-bank-deployment-tables",
        "resources":"b-resources-and-failures","examples":"b-upstream-adapter-examples",
        "external":"b-external-import-acceptance","source_audit":"b-data-preopening-audit","runtime":"b-score-runtime",
        "tests":"b-frozen-science-regressions"}
ASSETS={
    "wheel":("b-score-core-build",WHEEL,"Apache-2.0 original code. Dependency and model notices are separate."),
    "trained":("b-model-package",ARCHIVE,"Apache-2.0 original parameters and preprocessing; CC BY 4.0 original writing and grantable measurements."),
    "numerical":("b-scientific-results","many-source-results.zip","CC BY 4.0, to the extent rights apply to the original measurements and writing."),
    "bank":("b-bank-public-example","evidence-lens-b04-r16-example.zip","Apache-2.0 original parameters; CC BY 4.0 authored RGB, depth, opacity and cameras."),
    "reports":("b-bank-deployment-tables","evidence-lens-b04-r16-reports.zip","CC BY 4.0 original panels and grantable measurements."),
    "native":("b-upstream-adapter-examples","many-authored-native-b04-r16.zip","Apache-2.0 original Gaussian parameters; CC BY 4.0 authored observations and renders."),
    "negative":("b-upstream-adapter-examples","many-incomplete-native-example.zip","CC BY 4.0 original missing-field specimen and description.")}


def dispatch(a):
    s=a.store
    pins={k:selected(s,v) for k,v in STAGES.items()}
    if (s/"work/controller/b-bank-linux-mac-comparison/selected.json").is_file():
        pins["linux"]=selected(s,"b-bank-linux-mac-comparison")
    details={"phase":"B","tag":TAG}
    if a.operation=="inventory":
        pins["study_closeout"]=selected(s,"b-scientific-closeout")
        require(a.commit and re.fullmatch(r"[0-9a-f]{40}",a.commit)
                and subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()==a.commit,"different release commit")
        details["code_commit"]=a.commit
        for k,(label,name,_) in ASSETS.items():pins[k]=bind(s,selected(s,label)["run_id"],"payload/"+name)
    cmd=["{python}","scripts/many_seal.py",a.operation+"-worker","--config","{config}","--out","{out}"]
    for k in pins:cmd+=["--file",k,"{input:"+k+"}"]
    name,receipt=("b-research-release",MANIFEST) if a.operation=="inventory" else ("b-scientific-closeout","status.json")
    return stage(s,name,"archive_verify",pins,details,cmd,"payload/"+receipt,220000)


def close(a,cfg,paths):
    result,resources,up,cpu,model,freeze,opening=(load(paths[k]) for k in
        ("results","resources","upstream","deployment","model","freeze","opening"))
    totals=resources["runs"]
    fails=[r for r in totals if r["status"]!="succeeded"]
    limits=up["policy"]
    checks={
        "all_planned_upstream_returns":up["issued_fit_count"]==15 and up["succeeded"]==15,
        "all_planned_confirmation_measurements":all(r["views"]==r["complete_p2_views"] for r in result["scenes"] if r["role"]=="confirmation"),
        "signed_new_head":freeze["final_model"]["sha256"]==model["weights_sha256"],
        "prepared_banks":up["total_source_renders"]==1053 and up["total_target_predictions"]==102,
        "closed_reference_path":up["independent_confirmation_photos_accessed_by_upstream"]==0 and
            set(opening["terminals"])=={"living_room","bridge","b05-perforated-gallery","b06-corner-alcoves"},
        "gpu_time":resources["gpu_fit_worker_seconds"]<=limits["max_gpu_seconds"],
        "fit_limit":resources["fit_attempts"]<=limits["max_fits_including_failures"],
        "cpu_time":resources["accounted_cpu_core_seconds"]+resources["max_allowance_uncovered_seconds"]<=limits["max_cpu_core_seconds"],
        "archive":resources["allocated_B_archive_bytes"]<=limits["new_archive_bytes"],
        "scoring_rss":cpu["all_tested_native_rss_within_2gib"],
        "source_cache":cpu["all_tested_lru_peaks_within_512mib"]}
    status=result["comparison_status"] if all(checks.values()) else "blocked_or_incomplete"
    receipt=dict(schema="el-many-study-closeout-1",run_id=os.environ["EL_RUN_ID"],status=status,
        numerical_endpoint=result["comparison_status"],checks=checks,A_BASELINE_ID=PHASE_A,
        science_code_commit=freeze["science_code_commit"],model_id=model["model_id"],model_sha256=model["weights_sha256"],
        strong_rule=freeze["strong_simple_rule"],opening_run=opening["run_id"],run_count=len(totals),
        failures=[{"id":r["run_id"],"stage":r["trial"],"status":r["status"],"exit_code":r.get("exit_code")} for r in fails],
        resources={k:resources[k] for k in ("gpu_fit_worker_seconds","fit_attempts","accounted_cpu_core_seconds",
            "max_allowance_uncovered_seconds","all_manifest_logical_bytes","unique_payload_bytes","allocated_B_archive_bytes")},
        limits=limits,external=[r for r in result["scenes"] if r["scene"] in ("living_room","bridge")],
        parents=cfg["sources"])
    save(a.out/"status.json",receipt)
    fields=["run_id","trial","kind","bucket","status","exit_code","manifest_sha256","worker_seconds","native_cpu_core_seconds",
            "sampled_tree_cpu_seconds","uncovered_wall_seconds","peak_tree_rss_bytes","peak_reported_device_bytes",
            "logical_manifest_bytes","payload_bytes"]
    with (a.out/"runs.csv").open("x",encoding="utf-8",newline="") as dst:
        w=csv.DictWriter(dst,fieldnames=fields,extrasaction="ignore",lineterminator="\n")
        w.writeheader();w.writerows(totals)
    rs=receipt["resources"]
    lines=["# Resources and unsuccessful operations","",
        f"The native archive contains {len(totals)} terminal study records through the measured science and deployment cut. "
        f"{len(fails)} ended unsuccessfully. All original failures retain their native run IDs.","",
        "| Resource | Observed | Registered limit |","|---|---:|---:|",
        f"| Upstream Gaussian fits | {resources['fit_attempts']} | {limits['max_fits_including_failures']} |",
        f"| GPU, charged full native recorder wall | {rs['gpu_fit_worker_seconds']:.2f} s | {limits['max_gpu_seconds']:,} s |",
        f"| CPU, accounted process-tree core seconds | {rs['accounted_cpu_core_seconds']:.2f} s | {limits['max_cpu_core_seconds']:,} s |",
        f"| Additional allowance if every logical core was active during unsampled intervals | {rs['max_allowance_uncovered_seconds']:.2f} s | included in the budget check |",
        f"| Mac terminal archive, allocated blocks | {rs['allocated_B_archive_bytes']:,} bytes | {limits['new_archive_bytes']:,} bytes |",
        f"| Unique payload content | {rs['unique_payload_bytes']:,} bytes | measured separately |",
        f"| Manifest logical bytes, including repeated source snapshots and input evidence | {rs['all_manifest_logical_bytes']:,} bytes | measured separately |","",
        "Per-run CPU accounting takes the larger of a direct process timer and integrated tree samples, where available. "
        "Short-lived children may escape the sampler. The GPU charge includes native fitting, reload, rendering, parity checks and process overhead. "
        "Machine-wide NVIDIA measurements and per-process PyTorch allocation measure different resources. The Mac block count includes hardlinked files once; "
        "Windows working files and cloud transport replicas have separate receipts.","","## Native failure history",""]
    lines += [f"- `{r['run_id']}` · `{r['trial']}` · {r['status']}, exit {r.get('exit_code')}." for r in fails] or ["No unsuccessful native B record in this cut."]
    lines+=["","[Per-run measurements](runs.csv) · [Scorer hot, cold and full-bank measurements](deployment/README.md)"]
    (a.out/"resources-and-failures.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def inventory(a,cfg,paths):
    status=load(paths["study_closeout"])
    require(status["status"] in ("completed_supported","completed_negative","inconclusive_due_to_upstream"),
            "this experimental release requires a completed science closure")
    assets={}
    for key,(_,name,rights) in ASSETS.items():
        f=paths[key];pin=cfg["sources"][key]
        require(f.name==name and sha(f)==pin["sha256"],"different proposed public asset")
        assets[name]=dict(role=key,license=rights,bytes=f.stat().st_size,sha256=pin["sha256"],parent=pin)
    output=dict(schema="el-many-source-release-1",tag=TAG,run_id=os.environ["EL_RUN_ID"],status=status["status"],
        code_commit=cfg["code_commit"],science_code_commit=status["science_code_commit"],A_BASELINE_ID=PHASE_A,
        package_version="0.1.0.dev0",model_id=status["model_id"],model_sha256=status["model_sha256"],
        parents={k:v for k,v in cfg["sources"].items() if k not in ASSETS},assets=assets)
    save(a.out/MANIFEST,output)
    sums={n:d["sha256"] for n,d in assets.items()}
    sums[MANIFEST]=sha(a.out/MANIFEST)
    (a.out/"SHA256SUMS").write_text("".join(v+"  "+k+"\n" for k,v in sorted(sums.items())),encoding="ascii")
    result=load(paths["results"])
    rows=[r for r in result["scenes"] if r["scene"] in ("living_room","bridge")]
    notes=["# Multi-source CPU evidence audit","","A separately installable offline source-bank prototype and its genuinely fitted NumPy risk-ranking model. "
        "Install the two wheels for the released 378 × 252 CPU profile. The authored input and report archives reproduce a limited query and an explicit full audit.","",
        f"Study disposition: **{status['status']}**. At K16 the frozen simple comparator was **{status['strong_rule']}**. "
        "The table gives errors among the lowest-ranked fifth of each target's common domain. A scene contributes ten target views.","",
        "| Scene | New errors | Simple errors | Paired accepted pixels |","|---|---:|---:|---:|"]
    notes += [f"| {r['scene']} | {r['new_errors']:,} | {r['rule_errors']:,} | {r['quota']:,} |" for r in rows]
    notes+=["",f"Tagged code `{cfg['code_commit']}`. Model `{status['model_sha256']}`. "
            "The prerelease preserves the original v0.5.0 CPU toolkit and v0.3.0-rc.1 neural weights as separate dependencies. "
            "[Study, model card and practical installation](https://github.com/wany614/evidence-lens/tree/"+TAG+"/docs/research/many-source-v1) · "
            "[Base wheel](https://github.com/wany614/evidence-lens/releases/tag/v0.5.0)"]
    (a.out/"release-notes.md").write_text("\n".join(notes)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation",choices=["close","inventory","close-worker","inventory-worker"])
    for n in ("store","out","config"):p.add_argument("--"+n,type=Path)
    p.add_argument("--commit");p.add_argument("--file",nargs=2,action="append")
    a=p.parse_args()
    if a.store:print(dispatch(a))
    else:
        cfg=load(a.config);paths={k:Path(v) for k,v in a.file}
        (close if a.operation=="close-worker" else inventory)(a,cfg,paths)
