"""Authorize the registered eight new test families with a sealed, complete method freeze."""
import argparse
import math
from pathlib import Path

from evidence_lens._bundle import HASH, RUN_ID, encoded, load, local_path, sha
from evidence_lens._verify import require, verify_bundle
from make_handoff import publish

METHODS = ("constant", "1-alpha", "L0", "L1-full", "L1-geometry", "L2-frozen", "L2-vision-ft")
WEIGHTED = METHODS[3:]
MEMBER = "payload/method-freeze.json"


def holdouts(catalog):
    families = [row["scene_family_id"] for row in catalog["families"] if row["split"] == "test"]
    require(len(set(families)) == len(families) == 8, "Expected eight registered confirmation families")
    return sorted(families)


def sealed(store, rid, expected):
    require(isinstance(rid, str) and RUN_ID.fullmatch(rid) and isinstance(expected, str) and HASH.fullmatch(expected),
            "Malformed run or manifest identifier")
    directory = Path(store) / "runs" / rid
    require(sha(directory / "manifest.json") == expected, "Freeze references a different manifest")
    return directory, verify_bundle(directory)


def verify_methods(store, directory, checked, catalog, digest):
    run = checked["run"]
    require(run["execution_status"] == "succeeded" and run["run_kind"] == "report"
            and run["experiment_id"] == "stage2-method-freeze", "Method freeze did not complete")
    method = load(directory / MEMBER)
    require(method["schema"] == "el-vision-method-freeze-1" and method["catalog_sha256"] == digest
            and sorted(method["test_family_ids"]) == holdouts(catalog), "Wrong confirmation population")
    require(set(method["methods"]) == set(METHODS) and len(method["methods"]) == 7 and
            set(method["thresholds"]) == set(METHODS), "Missing comparison or development threshold")
    for name in METHODS:
        threshold = method["thresholds"][name]
        require(isinstance(threshold, dict) and type(threshold.get("tau")) in (int, float)
                and math.isfinite(threshold["tau"]) and threshold["domain"] == "development-B0",
                "Missing finite development B0 threshold for " + name)
    require(set(method["models"]) == set(WEIGHTED) and method["display_rule"] == "v01"
            and method["seeds"] == [1729,2718,3141] and method["primary_seed"] == 1729, "Model or display rules differ")
    parent_pins = {row["run_id"]:row["manifest_sha256"] for row in checked["inputs"]["parents"]}
    for name in WEIGHTED:
        spec = method["models"][name]
        require(spec["run_id"] in parent_pins and parent_pins[spec["run_id"]] == spec["manifest_sha256"]
                and spec["files"], "Selected model is not a direct pinned parent")
        bundle=Path(store)/"runs"/spec["run_id"]
        manifest=load(bundle/"manifest.json")
        require(sha(bundle/"manifest.json") == spec["manifest_sha256"], "Selected model manifest changed")
        members={m["path"]:m["sha256"] for m in manifest["files"]}
        for f in spec["files"]:
            key=local_path(f["path"])
            require(key.startswith("payload/") and members.get(key)==f["sha256"] and HASH.fullmatch(f["sha256"]),
                    "Weight identity is absent from the selected model")
    require(method["deployment_rule"] and method["training_receipts"], "Training or deployment selection missing")
    records=method["training_receipts"]
    require(len(records)<=8 and len({r["run_id"] for r in records})==len(records), "Training budget or receipt duplicates")
    for mode in ("L2-frozen","L2-vision-ft"):
        require({r["seed"] for r in records if r["method"]==mode} == set(method["seeds"]),
                "Missing matched seed records for "+mode)
    require(all(parent_pins.get(r["run_id"]) == r["manifest_sha256"] for r in records),
            "Candidate training attempts must be pinned parents")
    return method


def verify_opening(store, catalog, catalog_sha256):
    store=Path(store)
    path=store/"protocol/stage2"/catalog_sha256/"test-opening.json"
    require(path.is_file(), "The registered stage-two test cohort is still sealed")
    marker=load(path)
    require(marker["schema"]=="el-stage2-test-opening-1" and marker["catalog_sha256"]==catalog_sha256
            and sorted(marker["test_family_ids"])==holdouts(catalog) and marker["method_member"]==MEMBER,
            "Opening does not match the registered cohort")
    directory,checked=sealed(store,marker["freeze_run_id"],marker["freeze_manifest_sha256"])
    require(sha(directory/MEMBER)==marker["method_sha256"],"Sealed method payload changed")
    verify_methods(store,directory,checked,catalog,catalog_sha256)
    return marker


def open_cohort(store,catalog_path,expected_catalog,rid,manifest):
    require(sha(catalog_path)==expected_catalog,"Registry digest mismatch")
    catalog=load(catalog_path)
    directory,checked=sealed(store,rid,manifest)
    verify_methods(store,directory,checked,catalog,expected_catalog)
    marker={"schema":"el-stage2-test-opening-1","catalog_sha256":expected_catalog,"test_family_ids":holdouts(catalog),
            "freeze_run_id":rid,"freeze_manifest_sha256":manifest,"method_member":MEMBER,"method_sha256":sha(directory/MEMBER)}
    path=store/"protocol/stage2"/expected_catalog/"test-opening.json"
    publish(path,encoded(marker))
    return verify_opening(store,catalog,expected_catalog)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store",type=Path,required=True)
    p.add_argument("--catalog",type=Path,required=True)
    p.add_argument("--catalog-sha256",required=True)
    p.add_argument("--freeze-run",required=True)
    p.add_argument("--freeze-manifest",required=True)
    a=p.parse_args()
    print(open_cohort(a.store,a.catalog,a.catalog_sha256,a.freeze_run,a.freeze_manifest))
