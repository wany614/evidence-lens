"""Derive the two preregistered set1 1,024-step diagnostic jobs from the actual original 256-step ZIP."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

from evidence_lens._bundle import encoded,load,sha
from evidence_lens._verify import require
from dtu_handoff import payload,pin
from make_handoff import publish,sha_bytes,zip_bytes
from stage3_jobs import signature

MAIN="784a12587fa698e764abb84fb5cbbf14a6b7699ad9cb1d5850bd0de6a95c742e"
RUNNER="scripts/dtu_reconstruct_windows.py"
MAIN_RUNNER="82d13f0a3c67f983a7b11866bb5aa0ed9b4e386aeec638440903851f01b06669"
DECISION="e5ff509f7079b42c0d742c774e5b469dfa7e73258eca24e66391419f3395a927"
BATCH="d1-dtu-set1-1024-diagnostic"


def source_variant(raw):
    require(sha_bytes(raw)==MAIN_RUNNER,"Diagnostic must derive from the exact original Windows source")
    text=raw.decode("utf-8")
    edits=[
        ('j["experiment_id"]=="d1-dtu-source-only-256"','j["experiment_id"]=="d1-dtu-debug-only-1024"'),
        ('j["optimization_steps"]==256','j["optimization_steps"]==1024'),
        ('j["object_id"] in ("set1","set6")','j["object_id"]=="set1" and j["opening"] is None and j["diagnostic_decision_sha256"]=="'+DECISION+'"'),
        ('parents=[reg,conv,budget]','parents=[reg,conv,budget,j["diagnostic_decision"]]'),
        ('"optimization_steps":256,"worker_sha256":j["worker_sha256"]',
         '"optimization_steps":1024,"diagnostic_decision":j["diagnostic_decision"],"diagnostic_decision_sha256":j["diagnostic_decision_sha256"],"worker_sha256":j["worker_sha256"]'),
    ]
    for before,after in edits:
        require(text.count(before)==1,"Unexpected frozen source form")
        text=text.replace(before,after)
    result=text.encode()
    compile(result,RUNNER,"exec")
    return result


def contents(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        require(len(z.namelist())==len(set(z.namelist())),"Duplicate ZIP members")
        return {n:z.read(n) for n in z.namelist()}


def build(a):
    sel=payload(a.store.resolve(),a.decision)/"diagnostic-decision.json"
    decision=load(sel)
    require(sha(sel)==DECISION and decision["schema"]=="el-dtu-development-diagnostic-1" and decision["run_id"]==a.decision["run_id"]
            and (decision["object_id"],decision["updates"],decision["external_updates"],decision["maximum_diagnostic_jobs"])==("set1",1024,256,2)
            and decision["conditions"]==["M2","M4"],"Not the sealed set1 decision")
    raw=a.original.read_bytes();require(sha_bytes(raw)==MAIN,"Expected the actual accepted set1 main batch")
    old=contents(raw);manifest=json.loads(old["batch.json"])
    worker={k[len("worker/"):]:v for k,v in old.items() if k.startswith("worker/")}
    baseline_worker_sha=sha_bytes(zip_bytes(worker));require(baseline_worker_sha==manifest["worker_sha256"],"Original worker differs")
    worker[RUNNER]=source_variant(worker[RUNNER])
    wsha=sha_bytes(zip_bytes(worker)); parts={};rows=[];baseline_i={}
    for entry in manifest["jobs"]:
        original=old[entry["path"]];require(sha_bytes(original)==entry["sha256"],"Original condition ZIP changed")
        files=contents(original);oldjob=files.pop("job.json");j=json.loads(oldjob);c=j["condition"]
        require(j["object_id"]=="set1" and j["optimization_steps"]==256 and j["opening"] is None and sha_bytes(oldjob)==entry["job_sha256"],
                "Wrong starting study arm or object")
        names={n:signature(b) for n,b in files.items()}
        require(names==j["files"],"A source, optimizer or calibration differs from the original Windows job")
        tid="d1-dtu-set1-"+c+"-1024-diagnostic"
        j.update(trial_id=tid,experiment_id="d1-dtu-debug-only-1024",optimization_steps=1024,worker_sha256=wsha,
                 diagnostic_decision=a.decision,diagnostic_decision_sha256=DECISION,baseline_handoff_sha256=entry["job_sha256"])
        new=encoded(j);data=zip_bytes({**files,"job.json":new});path="jobs/"+tid+".zip";parts[path]=data
        rows.append({"trial_id":tid,"condition":c,"path":path,"job_sha256":sha_bytes(new),**signature(data)})
        baseline_i[c]={"job_sha256":entry["job_sha256"],"byte_identical_payloads":names}
    require([v["condition"] for v in rows]==["M2","M4"],"Expected the complete pair")
    meta={**manifest,"batch":BATCH,"jobs":rows,"worker_files":{k:signature(v) for k,v in worker.items()},"worker_sha256":wsha,
          "diagnostic_decision":a.decision,"diagnostic_decision_sha256":DECISION,"baseline_container_sha256":MAIN,
          "baseline_worker_sha256":baseline_worker_sha,"unchanged_numeric_input_members":baseline_i}
    rawmeta=encoded(meta);z=zip_bytes({"batch.json":rawmeta,**parts,**{"worker/"+k:v for k,v in worker.items()}})
    first=load(a.original.parent/"delivery.json")
    receipt={**first,"batch":BATCH,"batch_sha256":sha_bytes(rawmeta),"container_sha256":sha_bytes(z),"container_bytes":len(z),
             "jobs":rows,"worker_sha256":wsha,"diagnostic_decision":a.decision,"diagnostic_decision_sha256":DECISION}
    publish(a.out/"batch.json",rawmeta);publish(a.out/(BATCH+".zip"),z);publish(a.out/"delivery.json",encoded(receipt))
    return receipt


if __name__=="__main__":
    q=argparse.ArgumentParser(description=__doc__);q.add_argument("--original",type=Path,required=True);q.add_argument("--out",type=Path,required=True)
    q.add_argument("--decision",type=pin,required=True);q.add_argument("--store",type=Path,default=Path(".research"))
    print(json.dumps(build(q.parse_args()),sort_keys=True))
