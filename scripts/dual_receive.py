"""Receive an intact, content-pinned native run and preserve its real disposition."""

import argparse
import json
import os
from pathlib import Path
import time

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require, verify_bundle
from evidence_lens.archive import receive
from dual_records import selected, stage


def worker(a):
    receipt = load(a.receipt)
    expected = receipt.get("run_id", receipt.get("run"))
    status = receipt["status"]
    require(expected and status in {"succeeded", "failed", "interrupted", "cancelled"}
            and (status == "succeeded") == (receipt.get("worker_exit_code") == 0),
            "inconsistent native status")
    require(receipt["package"] == a.archive.name and receipt["bytes"] == a.archive.stat().st_size,
            "return container differs from receipt")
    start = time.monotonic()
    imported = receive(a.archive, a.store, receipt["sha256"])
    require(imported["status"] in {"accepted", "identical"}
            and imported["manifest_sha256"] == receipt["manifest_sha256"] and imported["run_id"] == expected,
            f"native archive intake: {imported.get('status')} {imported.get('reason','')}")
    root = a.store / "runs" / expected
    native = verify_bundle(root)
    require(native["run"]["execution_status"] == status, "transport and native dispositions differ")
    if a.ticket:
        parameters = load(a.config)
        ticket = parameters["sources"]["ticket"]
        require(ticket["run_id"] in native["run"]["parent_run_ids"], "missing native ticket parent")
        inputs = load(root / "inputs.json")["inputs"]
        require(any(file["sha256"] == ticket["sha256"] for item in inputs for file in item["files"]),
                "native stage did not bind the ticket bytes")
    save(a.out / "intake.json", {"schema":"el-dual-native-intake-1","run_id":os.environ["EL_RUN_ID"],
         "native_run":expected,"native_manifest_sha256":native["manifest_sha256"],"native_status":status,
         "native_trial":native["run"]["trial_id"],"container_sha256":receipt["sha256"],
         "container_bytes":receipt["bytes"],"verified_files":len(native["manifest"]["files"]),
         "seconds":time.monotonic()-start,"transfer_receipt_id":imported["receipt_id"]})


def main(a):
    if a.action == "worker":
        return worker(a)
    receipt = load(a.receipt)
    archive = a.receipt.parent / receipt["package"]
    parents = {"phase": selected(a.store, a.parent)}
    if a.ticket:
        parents["ticket"] = selected(a.store, a.ticket)
    cmd = ["{python}", "scripts/dual_receive.py", "worker", "--store",str(a.store),"--archive",str(archive),
           "--receipt","{input:return_receipt}","--out","{out}","--config","{config}"]
    if a.ticket:
        cmd += ["--ticket", a.ticket]
    return stage(a.store, a.label, "transfer_verify", parents,
                 {"expected_container":receipt["sha256"],"expected_disposition":receipt["status"]},
                 cmd, "payload/intake.json", 12000, {"return_receipt":a.receipt})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("record","worker"))
    for key in ["store","archive","out","receipt","config"]:
        parser.add_argument("--"+key,type=Path)
    for key in ["parent","ticket","label"]:
        parser.add_argument("--"+key)
    print(json.dumps(main(parser.parse_args())))
