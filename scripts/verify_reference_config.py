"""Read-only verification of both recorder settings used by a selected or resumed B0 reference."""

import argparse
import hashlib
import json
from pathlib import Path

from evidence_lens._bundle import encoded, load
from evidence_lens.blender_reference.formal_catalog import resolved, validate, value_hash
from evidence_lens.formal_inputs import PinnedRun, require
from evidence_lens.scoring_io import file_hash

ROOT = Path(__file__).resolve().parents[1]
# Both legacy inputs already had independently sealed run IDs and manifest pins before P3.
LEGACY_CONFIG = {"reference": "5b72e3e10dbd552e9aad5a432659980f6ed61bc4c542d3f5d7254ee98fafbad0",
                 "readback": "9533cf57ec46628ebb4640498f8afb8f48a994bbee1778885674852fc908b0c2"}


def verify(store, selection, checkout=ROOT):
    from evidence_lens._capacity import policy
    from evidence_lens.recorder import DEFAULTS

    catalog, designs, base, lock = validate(checkout)
    row = next(r for r in catalog["families"] if r["scene_family_id"] == selection["scene_family_id"])
    require(selection["schema"] == "el-formal-reference-selection-1" and selection["dataset_hash"] == lock["catalog_sha256"]
            and selection["trial_id"] == row["trial_id"], "selection does not match its frozen family")
    expected = None
    if row["ordinal"] != 1:
        expected = resolved(row, catalog, designs, base, lock)
        expected["recording"] = DEFAULTS | expected["recording"] | {"storage_limits": policy(None, checkout)}
    hashes = {}
    for role, kind in (("reference", "scene_render"), ("readback", "archive_verify")):
        pin = {"run_id": selection[role+"_run_id"], "manifest_sha256": selection[role+"_manifest_sha256"]}
        checked = PinnedRun(store, pin)
        require(checked.run["run_kind"] == kind and checked.run["execution_status"] == "succeeded"
                and checked.run["trial_id"] == row["trial_id"], f"{role} differs from the family")
        path = checked.open("config.resolved.yaml")
        actual = file_hash(path)
        desired = hashlib.sha256(encoded(expected)).hexdigest() if expected is not None else LEGACY_CONFIG[role]
        require(actual == desired == checked.run["config_sha256"], f"{role} did not use the frozen resolved config")
        if expected is None:
            reuse = row["reuse"]
            require(pin["run_id"] == reuse[role+"_run_id"] and pin["manifest_sha256"] == reuse[role+"_manifest_sha256"]
                    and value_hash(load(path)["reference"]) == row["recipe_sha256"], "tabletop legacy recipe or run mismatch")
        if role == "reference":
            meta = load(checked.open("payload/reference.json"))
            require(meta["camera_sha256"] == row["camera_sha256"] == file_hash(checked.open("payload/cameras.json")),
                    "native B0 camera does not match frozen catalog")
            if expected is not None:
                require(meta["formal"]["recipe_sha256"] == row["recipe_sha256"], "formal recipe digest mismatch")
        else:
            parent = {"run_id": selection["reference_run_id"], "manifest_sha256": selection["reference_manifest_sha256"]}
            require(load(checked.open("inputs.json"))["parents"] == [parent], "readback parent differs")
        hashes[role] = actual
    return {"schema": "el-reference-config-check-1", "passed": True, "scene_family_id": row["scene_family_id"],
            "trial_id": row["trial_id"], "config_sha256": hashes, "camera_sha256": row["camera_sha256"],
            "recipe_sha256": row["recipe_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.store, load(args.selection)), sort_keys=True))
