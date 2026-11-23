"""Export the exact shared train and development inputs required by all four recorded C fits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from evidence_lens._bundle import local_path, save, sha
from evidence_lens._verify import parent_directory, require
from evidence_lens.formal_inputs import PinnedRun
from evidence_lens.scoring_io import new_output


def write_dataset(output, run, table, split):
    manifest = run.open("payload/dataset.json")
    require(sha(manifest) == table["manifest"]["sha256"] and table["split"] == split,
            "Job refers to a different recorded dataset")
    members = {"dataset.json": table["manifest"], **table["files"]}
    target = output / (split + "-dataset.zip")
    with zipfile.ZipFile(target, "x", allowZip64=True) as stream:
        for name, entry in sorted(members.items()):
            local_path(name)
            member = "payload/" + name
            source = run.open(member)
            require(source.stat().st_size == entry["bytes"] and run.files[member]["sha256"] == entry["sha256"],
                    "Original data member differs from the fit contract")
            z = zipfile.ZipInfo(name)
            z.compress_type = zipfile.ZIP_STORED if name.endswith(".npz") else zipfile.ZIP_DEFLATED
            z.create_system = 3
            z.external_attr = 0o100444 << 16
            with stream.open(z, "w", force_zip64=True) as dest, source.open("rb") as incoming:
                shutil.copyfileobj(incoming, dest, length=2**20)
    with zipfile.ZipFile(target) as archived:
        require(len(archived.infolist()) == len(members) and set(archived.namelist()) == set(members),
                "Dataset transport roster differs")
        for name, row in members.items():
            with archived.open(name) as source:
                require(archived.getinfo(name).file_size == row["bytes"]
                        and hashlib.file_digest(source, "sha256").hexdigest() == row["sha256"],
                        "Exported training input failed its independent readback")
    return {"name": target.name, "source": run.pin, "dataset_sha256": table["manifest"]["sha256"],
            "files": len(members), "uncompressed_bytes": sum(i["bytes"] for i in members.values()),
            "sha256": sha(target), "bytes": target.stat().st_size}


def main():
    from multi_risk_job import check_job
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--job-sha256", required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--shelf", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    store = a.store.resolve()
    shelf = (a.shelf or store.parent/"cold-shelf").resolve()
    job = check_job(a.job.resolve(), a.job_sha256)
    require(job["experiment_id"] == "stage3-risk-neural-fit" and set(job["datasets"]) == {"train", "development"},
            "Expected a registered C neural training job")
    with new_output(a.output) as out:
        rows = {}
        for split in ("train", "development"):
            pin = job["parents"][split]
            path = parent_directory(store, pin, shelf)
            original = PinnedRun(path.parent.parent, pin)
            require(original.run["run_kind"] == "dataset_build" and original.run["execution_status"] == "succeeded",
                    "Shared input must come from a successful dataset run")
            rows[split] = write_dataset(out, original, job["datasets"][split], split)
        receipt = {"schema": "el-stage3-training-transport-1", "job_sha256": a.job_sha256,
                   "worker_sha256": job["worker_sha256"], "datasets": rows}
        save(out/"transport.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
