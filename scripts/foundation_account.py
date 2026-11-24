"""Count terminal records through the public C download, preserving earlier accounting cuts."""

import argparse
from collections import Counter
import os
from pathlib import Path

from dual_records import selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from many_resources import account as many_account


def collect(store, cutoff, out):
    end = load(store / "runs" / load(cutoff)["run_id"] / "run.json")["ended_at_utc"]
    rows = []
    for bucket in ("runs", "quarantine"):
        for folder in sorted((store / bucket).iterdir()):
            if not (folder / "completion.json").is_file():
                continue
            record = load(folder / "run.json")
            label = record["trial_id"] or ""
            experiment = record["experiment_id"] or ""
            phase = label[:1] if label.startswith(("a-", "b-", "c-")) else None
            if experiment == "dual-track-v1-A-native-start":
                phase = "a"
            if phase is None or record["ended_at_utc"] > end:
                continue
            completion = load(folder / "completion.json")
            manifest = sha(folder / "manifest.json")
            require(manifest == completion["manifest_sha256"] and folder.name == record["run_id"],
                    "terminal identity differs")
            rows.append(dict(phase=phase.upper(), run_id=record["run_id"], stage=label,
                status=record["execution_status"], receiver=bucket, manifest_sha256=manifest,
                seconds=record.get("elapsed_seconds_monotonic"), exit_code=record.get("exit_code")))
    require(len({r["run_id"] for r in rows}) == len(rows), "duplicate terminal run")
    counts = {phase: dict(Counter(r["status"] for r in rows if r["phase"] == phase)) for phase in ("A", "B", "C")}
    full_b = many_account(store)
    science_b = load(store / "runs" / selected(store, "b-scientific-closeout")["run_id"] / "payload/status.json")
    require(len(full_b["runs"]) == sum(counts["B"].values()), "B changed after the public cutoff")
    save(out / "run-ledger.json", dict(schema="el-dual-track-outcome-ledger-1", run_id=os.environ["EL_RUN_ID"],
        cutoff_utc=end, counts=counts, native_records=len(rows), rows=sorted(rows, key=lambda r: r["run_id"]),
        scope="Terminal A/B/C native manifests on the Mac receiver through the C anonymous download. "
              "The ledger recording itself and subsequent proof publication occur after this cut.",
        additional_native_intake= science_b["independently_reconciled_intake"],
        B_scientific_cut={"terminal":science_b["run_count"], "science_failures":len(science_b["failures"]),
                          **science_b["resources"]},
        B_complete_cut={k:full_b[k] for k in ("gpu_fit_worker_seconds", "fit_attempts", "accounted_cpu_core_seconds",
            "max_allowance_uncovered_seconds", "all_manifest_logical_bytes", "allocated_B_archive_bytes",
            "unique_payload_bytes")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--cutoff", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.out:
        collect(a.store, a.cutoff, a.out)
    else:
        labels = {"A": "a-baseline", "B": "b-research-release", "science": "b-scientific-closeout",
                  "C": "c-foundation-release", "downloads": "c-anonymous-readback"}
        pins = {k: selected(a.store, label) for k, label in labels.items()}
        print(stage(a.store, "c-final-outcome-ledger", "archive_verify", pins, {"cut": "C anonymous public retrieval"},
              ["{python}", "scripts/foundation_account.py", "--store", str(a.store),
               "--cutoff", "{input:downloads}", "--out", "{out}"], "payload/run-ledger.json", 160000))
