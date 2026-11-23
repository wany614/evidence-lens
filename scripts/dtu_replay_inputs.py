"""Independently retrieve the DTU metadata and reproduce the registered camera layout."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from evidence_lens._bundle import sha
from evidence_lens._verify import require, verify_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, required=True)
    a = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    store = a.store.expanduser().resolve()
    ledger = store / "dtu-replay"
    ledger.mkdir(exist_ok=True, parents=True)
    def record(stage, cfg, argv, parents=(), bindings=None):
        selected = ledger / (stage+".json")
        if selected.exists():
            ref = json.loads(selected.read_bytes())
            checked = verify_bundle(store/"runs"/ref["run_id"])
            require(checked["run"]["execution_status"] == "succeeded" and
                    checked["manifest_sha256"] == ref["manifest_sha256"], "replay receipt does not verify")
            return ref
        cmd = [sys.executable, "-m", "evidence_lens.recorder", "--root",str(store),"--source-root",str(project),
               "--config",str(cfg),"--kind","archive_verify","--experiment","d1-public-input-replay","--trial",stage]
        for ref in parents:
            cmd.extend(["--parent",ref["run_id"]+"="+ref["manifest_sha256"]])
        for role,(path,parent) in (bindings or {}).items():
            cmd.extend(["--input",role+"="+str(path)])
            if parent:
                cmd.extend(["--input-parent",role+"="+parent["run_id"]])
        result = subprocess.run(cmd+["--",*argv],text=True,stdout=subprocess.PIPE,check=False)
        sys.stderr.write(result.stdout)
        if result.returncode:
            raise RuntimeError("recorder failed; inspect the sealed run before another attempt")
        report = json.loads(result.stdout.strip().splitlines()[-1])
        ref = {k:report[k] for k in ("run_id","manifest_sha256")}
        verify_bundle(store/"runs"/ref["run_id"])
        with selected.open("x") as f:
            json.dump(ref,f,indent=2)
        return ref

    def payload(ref):
        return store / "runs" / ref["run_id"] / "payload"
    templates = project / "configs/dtu-v1"
    sample = ["{python}", "{source}/scripts/dtu_sample.py","--request","{config}","--out","{out}"]
    catalog = record("catalog", templates/"catalog.json", sample)
    url = json.loads((payload(catalog)/"links.json").read_bytes())["SampleSet"]
    expected = json.loads((templates/"index.json").read_bytes())["url"]
    require(url.replace("http://","https://",1) == expected, "official SampleSet link changed")
    index = record("index", templates/"index.json", sample, [catalog])
    sealed = json.loads((payload(index)/"archive.json").read_bytes())
    require(sealed == json.loads((templates/"calibration.json").read_bytes())["archive"], "published ZIP directory changed")
    cal = record("calibration", templates/"calibration.json", sample, [index])
    docs = record("dataset-documentation", templates/"dataset-documentation.json", sample, [index])
    root = payload(cal)/"members/SampleSet"
    bindings = {"readme":(root/"ReadMe.txt",cal),"matrices":(root/"MVS Data/Calibration/cal18",cal),
                "evaluator":(payload(docs)/"members/SampleSet/Matlab evaluation code/BaseEval2Obj_web.m",docs),
                "directory":(payload(index)/"directory.json",index),
                "baseline":(templates/"stage-c-camera-layout.json",None)}
    cmd = ["{python}","{source}/scripts/dtu_camera.py","--out","{out}"]
    for key in bindings:
        cmd.extend(["--"+key,"{input:"+key+"}"])
    rig = record("camera-layout", project/"protocol/dtu-external-v1.json", cmd, [cal,docs,index],bindings)
    study = json.loads((project/"protocol/dtu-external-v1.json").read_bytes())
    require(sha(payload(rig)/"view-set.json") == study["camera"]["views_sha256"],
            "camera choice or recorded numerical convention differs; inspect the new layout")
    print(json.dumps({"camera":rig,"camera_directory":str(payload(rig)),
                      "camera_plan_sha256":sha(payload(rig)/"camera-plan.json"),"expected_plan_sha256":study["camera"]["plan_sha256"]}))


if __name__ == "__main__":
    main()
