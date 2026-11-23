"""Pin the selection files of explicitly registered learning or opened confirmation trials."""

import argparse
from pathlib import Path

from evidence_lens.formal_inputs import require
from evidence_lens.scoring_io import file_hash, new_output, write_json
from vision_sources import signed, test_authority


def registry(store, shelf, catalog_pin, split, output, families=None):
    catalog = signed(*catalog_pin)
    require(catalog["schema"] == "el-vision-family-roster-1", "a Stage B family catalog is required")
    rows = [r for r in catalog["families"] if r["split"] == split]
    if split == "test":
        require(not families, "confirmation opens for the complete registered cohort")
        test_authority([store.resolve(), shelf.resolve()], catalog_pin)
    if families:
        require(
            len(set(families)) == len(families) and set(families) <= {r["scene_family_id"] for r in rows},
            "requested families differ from the registered split",
        )
        rows = [r for r in rows if r["scene_family_id"] in families]
    trials = {}
    for row in rows:
        for condition, trial in row["trials"].items():
            require(trial not in trials and (split == "test" or condition == "B0"), "wrong or repeated trial")
            path = store / "trials" / trial / "trial_selection.json"
            digest = file_hash(path)
            selection = signed(path, digest)
            require(
                selection["trial_id"] == trial
                and selection["scene_family_id"] == row["scene_family_id"]
                and selection["condition_id"] == condition
                and selection["selection_status"] in ("selected", "failed"),
                "selection belongs to another trial or is not terminal",
            )
            trials[trial] = {"selection_sha256": digest, "status": selection["selection_status"]}
    require(trials, "empty split")
    with new_output(output) as out:
        result = {
            "schema": "el-vision-selected-trials-1",
            "catalog_sha256": catalog_pin[1],
            "split": split,
            "trials": trials,
        }
        write_json(out / "selections.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("store", "shelf", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--catalog", nargs=2, required=True, metavar=("CATALOG", "SHA256"))
    p.add_argument("--split", choices=("train", "development", "test"), required=True)
    p.add_argument("--family", action="append")
    a = p.parse_args()
    registry(a.store, a.shelf, a.catalog, a.split, a.output, a.family)
    print(file_hash(a.output / "selections.json"))
