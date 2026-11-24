"""Package the authored external native fit and an independently rejected missing-depth example."""

import argparse
import io
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from dual_records import bind, selected, resolve, stage
from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens_bundle.common import require


def control(s):
    bank=selected(s,"b-bank-b04-layered-louvers-r16")
    proof=bind(s,bank["run_id"],"payload/upstream-evidence.json")
    summary=load(resolve((s,s/"history"),proof));reg=summary["native_registry"]
    pins={"bank":bank,"proof":proof,**{k:reg[k] for k in ("data","asset","checkpoint","parity","probe","training")}}
    asset=load(resolve((s,s/"history"),reg["asset"]))
    for i,k in enumerate(asset["R"]+asset["H"]):
        pins[f"frame-{i}"]=bind(s,reg["asset"]["run_id"],"payload/"+asset["frames"][k]["path"])
    cmd=["{python}","-B","{source}/scripts/many_upstream_examples.py","worker","--out","{out}","--config","{config}"]
    for n in pins:cmd+=["--input",n,"{input:"+n+"}"]
    return stage(s,"b-upstream-adapter-examples","dataset_build",pins,{"scene_id":"b04-layered-louvers","scope":"authored optical observations, native model and project source licensing",
                 "negative":"one real R self-render stripped of the expected camera-Z array, no fitting"},cmd,"payload/examples.json",105_000_000)


def worker(cfg,files,out):
    asset=load(files["asset"]);bank=load(files["bank"]);proof=load(files["proof"])
    require(bank["scene_id"]==asset["scene_id"]==cfg["scene_id"] and proof["bank_sha256"]==sha(files["bank"])
            and bank["reconstruction_provenance"]=="verified","wrong original positive example")
    positive=out/"many-authored-native-b04-r16.zip"
    terms=(("Authored layered-louvers scene, source photography, calibrated camera data and native Gaussian renders by wany614, "
            "Evidence Lens; CC BY 4.0 for grantable project data, https://creativecommons.org/licenses/by/4.0/. "
            "gsplat is an external Apache-2.0 project, nerfstudio-project/gsplat commit "+asset["official_commit"]+".\n").encode())
    lineage=encoded(proof["native_registry"]);lineage_sha=hashlib.sha256(lineage).hexdigest()
    with zipfile.ZipFile(positive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        z.writestr("NOTICE.txt",terms);z.write(files["data"],"sources.zip");z.write(files["asset"],"native/asset.json")
        z.writestr("native/lineage.json",lineage)
        for key,role in (("checkpoint","checkpoint"),("parity","native_parity"),("probe","depth_probe"),("training","training")):
            require(sha(files[key])==asset[role]["sha256"],"wrong example checkpoint or evidence")
            z.write(files[key],"native/"+asset[role]["path"])
        for i,k in enumerate(asset["R"]+asset["H"]):
            path=files[f"frame-{i}"];spec=asset["frames"][k];require(sha(path)==spec["sha256"],"example native frame mismatch")
            z.write(path,"native/"+spec["path"])
        z.writestr("identity.json",encoded({"schema":"el-many-native-authored-example-1","trusted_asset_sha256":sha(files["asset"]),
                   "trusted_source_sha256":sha(files["data"]),"trusted_lineage_sha256":lineage_sha,"verified_bank_sha256":sha(files["bank"]),"actual_R":asset["R"],"target_poses_only":asset["H"]}))
    bad=out/"many-incomplete-native-example.zip"
    with np.load(files["frame-0"],allow_pickle=False) as f:
        b=io.BytesIO();np.savez_compressed(b,rgb=f["rgb"],alpha=f["alpha"])
    with zipfile.ZipFile(bad,"x",compression=zipfile.ZIP_STORED) as z:
        z.writestr("NOTICE.txt",terms);z.writestr("missing-camera-z.npz",b.getvalue())
        z.writestr("expected.json",encoded({"schema":"el-many-native-negative-example-1","original_frame":asset["frames"][asset["R"][0]],
                     "removed":["z"],"status":"incomplete","actual_bank_rejection":proof["partial_input_rejection"]}))
    save(out/"examples.json",{"schema":"el-many-external-adapter-examples-1","run_id":os.environ["EL_RUN_ID"],
         "complete":{"path":positive.name,"sha256":sha(positive),"bytes":positive.stat().st_size,"trust_native":sha(files["asset"]),
                     "trust_source":sha(files["data"]),"trust_lineage":lineage_sha,"expected_bank":sha(files["bank"]),"native_checkpoint":asset["checkpoint"],"source_count":len(asset["R"])},
         "incomplete":{"path":bad.name,"sha256":sha(bad),"bytes":bad.stat().st_size,"removed":"expected camera-Z",
                       "diagnostic":proof["partial_input_rejection"]},"confirmed_external_photographs":0,"references_in_bundles":0,"bank":cfg["sources"]["bank"]})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("record","worker"))
    for n in ("config","store","out"):p.add_argument("--"+n,type=Path)
    p.add_argument("--input",action="append",nargs=2,default=[]);a=p.parse_args()
    if a.action=="worker":worker(load(a.config),{k:Path(v) for k,v in a.input},a.out)
    else:print(json.dumps(control(a.store.resolve())))
