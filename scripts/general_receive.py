"""Receive a returned engineering run and bind its native archive to the exact dispatched bytes."""
import argparse
import json
from pathlib import Path
import os
import time
from evidence_lens.archive import receive
from evidence_lens._bundle import load, save, sha, utc
from evidence_lens._verify import require
from general_records import bind, resolve, stage


def worker(a):
    cfg=load(a.config)
    rr=load(a.receipt)
    require(sha(a.receipt)==cfg["extra_sha256"]["return_receipt"] and rr["package"]==a.archive.name
            and rr["bytes"]==a.archive.stat().st_size and rr[cfg.get("identity_key","input_zip_sha256")]==cfg["sources"]["ticket"]["sha256"]
            and rr["producer_run_id"]==cfg["sources"]["ticket"]["run_id"]
            and rr["status"]=="succeeded" and rr["worker_exit_code"]==0,
            "return differs from the named engineering ticket")
    start=utc(); before=time.monotonic()
    accepted=receive(a.archive,a.store,rr["sha256"])
    require(accepted["status"] in ("accepted","identical") and accepted["run_id"]==rr["run"]
            and accepted["manifest_sha256"]==rr["manifest_sha256"],"returned archive failed native verification")
    pin=bind(a.store,rr["run"],cfg["result_member"])
    require(pin["manifest_sha256"]==rr["manifest_sha256"],"received member belongs to a different run")
    save(a.out/"readback.json",{"schema":"el-general-native-engineering-readback-1","run_id":os.environ["EL_RUN_ID"],
         "native":pin,"submitted":cfg["sources"]["ticket"],"return_container_sha256":rr["sha256"],
         "container_bytes":rr["bytes"],"started_utc":start,"finished_utc":utc(),
         "local_intake_seconds":time.monotonic()-before,"provider_network_seconds":None,
         "native_transfer_status":accepted["status"],"transfer_receipt_id":accepted["receipt_id"]})


def main(a):
    if a.action=="worker":
        return worker(a)
    r=a.store
    t=bind(r,a.ticket_run,a.ticket_member)
    return stage(r,r/"history",a.label,"archive_verify",{"ticket":t},{"result_member":a.result_member,"identity_key":a.identity_key},
                 ["{python}","{source}/scripts/general_receive.py","worker","--config","{config}",
                  "--store",str(r),"--archive",str(a.archive),"--receipt",str(a.receipt),"--out","{out}"],
                 "payload/readback.json",5000,{"return_receipt":a.receipt})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("record","worker"))
    for n in ("store","archive","receipt","config","out"):
        p.add_argument("--"+n,type=Path)
    p.add_argument("--identity-key",choices=("input_zip_sha256","input_release_manifest_sha256"),default="input_zip_sha256")
    for n in ("label","ticket-run","ticket-member","result-member"):
        p.add_argument("--"+n)
    print(json.dumps(main(p.parse_args()),sort_keys=True))
