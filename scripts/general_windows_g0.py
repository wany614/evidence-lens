"""Project a native Windows inventory and its independent index check into a portable G0 receipt."""
import argparse
import os
from pathlib import Path
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require


def project(native, index, out):
    doc=load(native); supplement=load(index)
    require(doc["schema"]=="evidence-lens-g0-windows-private-receipt-1"
            and not supplement["receipt_manifest_mismatches"] and not supplement["errors"]
            and len(doc["raw_seals_retained_on_research_disk"])==2, "inconsistent Windows index readback")
    seals=[{"run_id":r["run"],"manifest_sha256":r["manifest_sha256"],"container_sha256":r["archive_sha256"],
            "bytes":r["archive_bytes"]} for r in doc["raw_seals_retained_on_research_disk"]]
    require(seals[0]["run_id"]==doc["g0_run_id"] and seals[1]["run_id"]==supplement["run_id"],
            "Windows scope and native receipts differ")
    info=doc["cpu_os"]; worker=doc["worker"]; cov=doc["coverage"]
    output={"schema":"el-general-windows-inventory-1","mac_run_id":os.environ["EL_RUN_ID"],"raw_seals":seals,
            "native_receipt_sha256":sha(native),"index_receipt_sha256":sha(index),
            "platform":{"system":info["windows"],"version":info["version"],"cpu":info["cpu"],"ram_bytes":info["ram_bytes"],
                        "gpu":doc["gpu"],"python":worker["python"],"torch":worker["torch"],"torch_cuda_runtime":worker["torch_cuda_runtime"]},
            "disk":doc["storage"],"census_utc":doc["pre_science_census_end_utc"],
            "independent_supplement_utc":doc["historical_index_supplement_end_utc"],
            "historical_indices":cov["independent_historical_sqlite_indices"],"indexed_archive_rows":cov["historical_archive_rows"],
            "matching_original_manifest_pairs":cov["completion_manifest_pairs_matched"],
            "scope_note":cov["error_note"],"historical_manifest_mismatches":len(supplement["receipt_manifest_mismatches"])}
    save(out/"inventory.json",output)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("native","index","out"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args();project(a.native,a.index,a.out)
