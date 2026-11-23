"""Authorize the registered eight new test families with a sealed, complete method freeze."""
import argparse
from pathlib import Path

from evidence_lens._bundle import HASH, RUN_ID, encoded, load, sha
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
    from evidence_lens.formal_inputs import PinnedRun
    from evidence_lens.stage2_authority import verify_frozen_authority

    root=Path(store).resolve()
    rid=checked["run"]["run_id"]
    require(directory.resolve()==root/"runs"/rid,"Incorrect frozen run directory")
    freeze=PinnedRun(root,{"run_id":rid,"manifest_sha256":checked["manifest_sha256"]})
    return verify_frozen_authority((root,root.parent/"cold-shelf"),freeze,digest,holdouts(catalog))


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
