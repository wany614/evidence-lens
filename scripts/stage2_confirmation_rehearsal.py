"""Exercise real recorder quota, immutable retry and remote-return hydration in private synthetic stores."""

import argparse
import contextlib
import io
import json
import os
import shutil
from pathlib import Path

from evidence_lens import _capacity, recorder
from evidence_lens._bundle import encoded, load, sha
from evidence_lens._verify import require, verify_bundle
from evidence_lens.archive import receive
from make_handoff import publish
import run_stage2_confirmation as queue
from worker_jobs import dispatch


def exercise(root):
    require(not root.exists(), "Rehearsal requires a new exclusive directory")
    root.mkdir(parents=True)
    source = root / "source"
    win = root / "producer"
    mac = root / "receiver"
    outgoing = root / "outgoing"
    landing = root / "landing"
    cold = root / "retained"
    for p in (source / "scripts", source / "configs", win, mac, outgoing, landing, cold):
        p.mkdir(parents=True, exist_ok=True)
    (source / "scripts" / "synthetic.py").write_text("# Explicitly synthetic transport exercise.\n")
    limits = {"local_active_batch_target_bytes": 4 * 1024 * 1024, "local_free_floor_bytes": 0}
    storage = source / "configs/storage.json"
    publish(storage, encoded(limits))
    digest = "7" * 64
    trial = "synthetic_confirmation_recovery-B0"
    exp = "stage2-confirmation-fixture"
    cfg = root / "config.json"
    publish(
        cfg,
        encoded(
            {
                "formal_handoff_sha256": digest,
                "dataset_hash": "1" * 64,
                "trial_id": trial,
                "recording": {"expected_output_bytes": 128, "telemetry_interval_seconds": 0.1},
            }
        ),
    )
    job = {
        "trial_id": trial,
        "identity": {"dataset_hash": "1" * 64},
        "experiment_id": exp,
        "reference": None,
        "readback": None,
    }
    descriptor = {"trial_id": trial, "handoff_sha256": digest}
    calls = []

    def launch(kind):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = recorder.main(
                [
                    "--root",
                    str(win),
                    "--source-root",
                    str(source),
                    "--config",
                    str(cfg),
                    "--storage-policy",
                    str(storage),
                    "--kind",
                    kind,
                    "--experiment",
                    exp,
                    "--trial",
                    trial,
                    "--",
                    "{python}",
                    "-c",
                    "import pathlib,sys;pathlib.Path(sys.argv[1],'synthetic.txt').write_bytes(b'fixture transport only')",
                    "{out}",
                ]
            )
        result = json.loads(buf.getvalue())
        require(rc == 0, "Fixture generator must finish before its transfer")
        calls.append(result)
        return result

    filler = outgoing / "prior-packaging-fixture.bin"
    with filler.open("xb") as f:
        f.write(b"X" * (4 * 1024 * 1024))

    def attempt():
        return dispatch(
            win, outgoing, job, digest, "generation", None, None, None, lambda: launch("generation")
        )

    try:
        attempt()
    except ValueError as e:
        require(str(e) == queue.QUOTA, "Unexpected scientific or transport failure")
        reason = str(e)
    else:
        raise ValueError("Real recorder/packaging quota was not exercised")
    require(
        len(calls) == 1 and len(list((win / "runs").iterdir())) == 1,
        "Expected one genuine seal before transfer hold",
    )
    rid = calls[0]["run_id"]
    first = verify_bundle(win / "runs" / rid)["manifest_sha256"]
    capacity = _capacity.check(win, 65536, limits)
    outgoing_bytes = _capacity.bytes_on_tree(outgoing)
    require(
        not list(outgoing.glob("*.receipt.json")) and not list(outgoing.glob("*.zip")),
        "Unaccepted partial outage return",
    )
    os.rename(filler, cold / filler.name)
    original = attempt()
    same = attempt()
    require(
        same == original and len(calls) == 1 and first == original["manifest_sha256"],
        "Restart changed a scientific run or repeated execution",
    )
    mr = outgoing / (rid + ".receipt.json")
    mp = outgoing / (rid + ".zip")
    shutil.copyfile(mr, landing / mr.name)
    partial = queue.return_state(landing, [descriptor])[trial]
    require(partial["status"] == "in_transit", "Deferred sync classified as science")
    shutil.copyfile(mp, landing / mp.name)
    arrived = queue.return_state(landing, [descriptor])[trial]
    require(
        arrived["status"] == "waiting_for_return" and len(arrived["attempts"]) == 1,
        "Producer arrival completed an absent render",
    )
    r1 = receive(landing / mp.name, mac, original["sha256"], storage)
    r2 = receive(landing / mp.name, mac, original["sha256"], storage)
    require(
        (r1["status"], r2["status"]) == ("accepted", "identical")
        and verify_bundle(mac / "runs" / rid)["manifest_sha256"] == first,
        "Independent receiver or idempotent transfer failed",
    )
    return {
        "schema": "el-stage2-confirmation-rehearsal-1",
        "evidence_kind": "synthetic_engineering_fixture",
        "scientific_test_families_accessed": 0,
        "producer_run_id": rid,
        "manifest_sha256": first,
        "windows_stage_launches": len(calls),
        "preserved_original_run": True,
        "active_target_bytes": limits["local_active_batch_target_bytes"],
        "outgoing_bytes_at_deferral": outgoing_bytes,
        "producer_store_bytes_at_deferral": capacity["active_bytes"],
        "reason": reason,
        "resume_on_same_seal": True,
        "receipt_before_zip": partial["status"],
        "producer_without_render": arrived["status"],
        "container_sha256": original["sha256"],
        "container_bytes": original["bytes"],
        "receiver_dispositions": [r1["status"], r2["status"]],
        "receiver_transfer_receipts": [r1["receipt_id"], r2["receipt_id"]],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture-root", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    a = p.parse_args()
    result = exercise(a.fixture_root.resolve())
    publish(a.result, encoded(result))
    print(json.dumps(result))
