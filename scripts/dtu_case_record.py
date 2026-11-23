"""Seal the published 057 adapter regression or the authorized DTU case after a complete return."""
import argparse
import hashlib
import io
import json
import shutil
import tempfile
from pathlib import Path
import zipfile

from evidence_lens._bundle import encoded,load,sha
from evidence_lens._verify import require,verify_bundle
from dtu_convert import opening
from dtu_handoff import pin,payload,snapshot,RIG,recorded_opening
from stage3_reference import perform,located

OLD={"run_id":"20261123T162553037969Z-c6c429e1d8704905","manifest_sha256":"746df66da8aee2cfdcf2de1981e731f9e488cd128a76458b4b4ea7e2eaa218fa"}
CODE=["scripts/dtu_case.py",*("src/evidence_lens/"+n+".py" for n in
    ("__init__","_bundle","dtu_camera","stage3_gaussians","multisource_io","multisource","features","scoring_io"))]


def work(a):
    project,store=a.project.resolve(),a.store.resolve()
    scratch=store.parent/"dtu-case-scratch";scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch,prefix="case-") as tmp:
        base=Path(tmp); src=base/"src-code";snapshot(project,src,CODE)
        command=["{python}","{source}/scripts/dtu_case.py"]
        if a.action=="published057":
            data=payload(store,OLD)
            cfg={"experiment_id":"d1-dtu-public-regression","trial_id":"published057-external-case-roundtrip","scene_family_id":"scene_057_offset_bell_cage",
                 "recording":{"expected_output_bytes":24000000}}
            result=perform(store,project,src,cfg,"archive_verify",{"original":(data/"CASE","public")},{"public":OLD},
                        command+["published057","--original","{input:original}","--out","{out}"])
        else:
            op=opening(a.object_id,a.opening,a.opening_sha256)
            delivery=load(a.batch/"delivery.json")
            require(delivery["container_sha256"]==sha(a.batch/(delivery["batch"]+".zip")) and
                    delivery["batch_sha256"]==sha(a.batch/"batch.json") and delivery["opening_sha256"]==a.opening_sha256,
                    "Changed original source-only handoff or opening")
            if op: original_marker=recorded_opening(store,delivery["opening"],a.opening,a.opening_sha256,"sources")
            chosen=load(a.selection)
            require(chosen["schema"]=="el-dtu-return-selection-1" and chosen["object_id"]==a.object_id and
                    chosen["condition"]==a.condition and chosen["source_conversion"]==delivery["source_conversion"],
                    "Wrong verified Windows input selection")
            pins={"rig":RIG,"sources":chosen["source_conversion"],"gen":chosen["generation"],"render":chosen["render"]}
            if op: pins["opening"]=delivery["opening"]
            roots={k:payload(store,pin) for k,pin in pins.items()}
            tid="d1-dtu-"+a.object_id+"-"+a.condition+"-256";require(chosen["trial_id"]==tid,"Selected trial differs")
            packet=base/"selected-ticket";packet.mkdir()
            entry=next(x for x in delivery["jobs"] if x["trial_id"]==tid)
            with zipfile.ZipFile(a.batch/(delivery["batch"]+".zip")) as z:
                nested=z.read(entry["path"])
            require(hashlib.sha256(nested).hexdigest()==entry["sha256"],"Changed registered job ZIP")
            with zipfile.ZipFile(io.BytesIO(nested)) as z:raw=z.read("job.json")
            require(hashlib.sha256(raw).hexdigest()==entry["job_sha256"],"Registered job identity differs")
            (packet/"job.json").write_bytes(raw)
            for k in ("gen","render"):
                fl=roots[k].parent
                for old,suffix in (("run.json","run"),("inputs.json","inputs"),("config.resolved.yaml","config")):
                    shutil.copyfile(fl/old,packet/(k+"-"+suffix+".json"))
            shutil.copyfile(roots["gen"]/"reconstruction.json",packet/"reconstruction.json")
            (packet/"pins.json").write_bytes(encoded({**pins,"job_sha256":entry["job_sha256"]}))
            cfg={"experiment_id":"d1-dtu-case-readback","trial_id":tid+"-case","scene_family_id":a.object_id,"condition_id":a.condition,
                 "selection_sha256":sha(a.selection),"selection":chosen,"registered_job_sha256":entry["job_sha256"],
                 "opening_sha256":a.opening_sha256,"prior_nonstarted_engineering_attempt":a.prior_preflight,"recording":{"expected_output_bytes":18000000}}
            inputs={k:(roots[k],k) for k in ("rig","sources","gen","render")};inputs["packet"]=(packet,None)
            inputs["selection"]=(a.selection,None)
            if op: inputs["opening"]=(original_marker,"opening")
            if a.prior_preflight:
                failed=located(store,a.prior_preflight); failure=verify_bundle(failed)
                from evidence_lens.recorder import worker_started
                require(failure["run"]["execution_status"]=="failed" and failure["run"]["trial_id"]==tid+"-case" and not worker_started(failed,failure["run"]),"Only a genuine nonstarted readback preflight can be recovered here")
                shutil.copyfile(failed/"run.json",packet/"prior-engineering-preflight.json")
            cmd=command+["build","--out","{out}"]
            for n in ("packet","rig","sources","gen","render"):cmd+=["--"+n,"{input:"+n+"}"]
            result=perform(store,project,src,cfg,"archive_verify",inputs,pins,cmd)
        folder=located(store,result)/"payload"
        return {"pin":result,"regression_sha256":sha(folder/"regression.json")} if a.action=="published057" else {
                "pin":result,"case_sha256":sha(folder/"case/case.json"),"case_provenance_sha256":sha(folder/"case-provenance.json"),
                "reconstruction_readback_sha256":sha(folder/"reconstruction-readback.json")}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    for n in ("published057","build"):
        q=sub.add_parser(n);q.add_argument("--project",type=Path,default=Path("."));q.add_argument("--store",type=Path,default=Path(".research"))
        if n=="build":
            for arg in ("batch","selection"):q.add_argument("--"+arg,type=Path,required=True)
            q.add_argument("--object-id",choices=("set1","set6"),required=True);q.add_argument("--condition",choices=("M2","M4"),required=True)
            q.add_argument("--opening",type=Path);q.add_argument("--opening-sha256");q.add_argument("--prior-preflight",type=pin)
    print(json.dumps(work(p.parse_args()),sort_keys=True))
