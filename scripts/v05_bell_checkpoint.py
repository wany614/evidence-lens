"""Record an exposed below-cap checkpoint and malformed negative control on the new upstream importer."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from evidence_lens._bundle import load, save, sha
from evidence_lens_asset.gaussian import RECOVERY, load as gaussians
from evidence_lens_asset.permissions import source_grant
from evidence_lens_bundle.common import BundleError, encoded, require
from hashlib import sha256

NATIVE = {"run_id":"20261124T042032489915Z-3a4f50090b344cab","manifest_sha256":"5af8160bc13a96753badd2d4e27aec85fd6e70fddfd735172ad3103a79520e66",
          "member":"payload/checkpoints/step-5000.pt","sha256":"4d9071f05f99ec45c6649c20608706c7c64f5cbee893d40ef809f1ae94cf494f"}


def worker(cfg, checkpoint, out):
    import torch

    require(cfg["sources"]["checkpoint"] == NATIVE and sha(checkpoint) == NATIVE["sha256"], "wrong exposed engineering control")
    params = gaussians(checkpoint,RECOVERY,1)
    ck = torch.load(checkpoint,map_location="cpu",weights_only=True)
    cap = ck["recipe"]["topology"]["cap"]
    n = len(params["means"])
    require(n == 16251 and n < cap == 20000, "wrong registered regression object or native count")
    with tempfile.TemporaryDirectory() as td:
        malformed = Path(td)/"wrong-shape.pt"
        blob = {k:ck[k] for k in ("schema","step","recipe")}
        blob["params"] = {**ck["params"],"opacities":ck["params"]["opacities"][:-1].clone()}
        torch.save(blob,malformed)
        try:
            gaussians(malformed,RECOVERY,1)
        except BundleError as exc:
            bad = str(exc)
        else:
            raise ValueError("mismatched parameter count accepted")
    registration={"schema":"el-upstream-layout-authority-1","fixed_layouts":{"M2":["24","22"],"M4":["24","23","17","22"]},
                  "optional_pool":["7","10","13","17","20","23","31","40"]}
    key=sha256(encoded(registration)).hexdigest()
    failure={"schema":"el-upstream-optional-choice-1","registration_sha256":key,"status":"failed","reason":"registered fixture","selected_ids":None}
    taken={**failure,"status":"selected","reason":None,"selected_ids":["17","23"]}
    fixed=source_grant(registration,"fixed-M4",failure)
    stopped=source_grant(registration,"optional-M4",failure)
    selected=source_grant(registration,"optional-M4",taken)
    require(fixed==["24","23","17","22"] and stopped==["24","22"] and selected==["24","17","23","22"],"branch-local grant or source order mismatch")
    save(out/"checkpoint.json",{"schema":"el-v05-gaussian-contract-1","run_id":os.environ["EL_RUN_ID"],"historical_input":NATIVE,
         "scope":"exposed object engineering regression; only existing checkpoint tensors were opened", "profile":RECOVERY,
         "actual_gaussians":n,"registered_cap":cap,"tensor_shapes":{k:list(v.shape) for k,v in params.items()},
         "malformed_opacities_rows":n-1,"malformed_rejection":bad,"optional_failure_fixed_M4":fixed,
         "optional_failure_optional_branch":stopped,"successful_optional_branch":selected,"photographs_opened":0,"historical_T1_reruns":0})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("record","worker"))
    for k in ("store","config","out","checkpoint","torch-python"):p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.action=="worker":worker(load(a.config),a.checkpoint,a.out)
    else:
        from dual_records import selected,stage
        s=a.store.resolve();pins={"entry":selected(s,"a-grid-registration"),"checkpoint":NATIVE}
        cmd=[str(a.torch_python.absolute()),"-B","{source}/scripts/v05_bell_checkpoint.py","worker","--config","{config}","--out","{out}","--checkpoint","{input:checkpoint}"]
        print(json.dumps(stage(s,"a-upstream-gaussian-contract","archive_verify",pins,{"variant":"original and opacity-row-minus-one", "role":"exposed regression"},cmd,"payload/checkpoint.json",8000)))
