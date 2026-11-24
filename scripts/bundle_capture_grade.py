"""Read two sealed score products and map real official capture IDs to their original Case roles."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from bundle_grade import compare, read
from bundle_original import check, put, sha


def grade(a):
    exposure = json.loads(a.provenance.read_bytes())
    old = json.loads((a.old / "scores.json").read_bytes())
    check(
        exposure["schema"] == "el-dtu-case-provenance-1"
        and exposure["object_id"] == "set1"
        and old["case_sha256"] == exposure["case_sha256"],
        "not the authorized old case",
    )
    result = []
    for method in ("MS-L1", "B-L1-transfer"):
        base = a.new / method
        new = json.loads((base / "scores.json").read_bytes())
        check(
            new["checkpoint_sha256"] == old["checkpoint_sha256"]
            and all(
                new["model"][k] == old["models"][method][k]
                for k in ("model_id", "model_index_sha256", "weights_sha256", "threshold")
            ),
            "model/reconstruction changed",
        )
        new_by_id = {v["id"]: v for v in new["views"]}
        for k, i in exposure["target_view_ids"].items():
            q = new_by_id[str(i)]
            original = old["views"][k]["methods"][method]
            check(
                q["render"]["file"]["sha256"] == original["target_prediction_sha256"],
                "not the same native predicted pixels",
            )
            with (
                np.load(read(base, q["arrays"]), allow_pickle=False) as n,
                np.load(read(a.old, original["file"]), allow_pickle=False) as o,
            ):
                c = compare(n, o)
                full = np.array_equal(n["risk"].view(np.uint32), o["risk"].view(np.uint32))
            result.append(
                {
                    "method": method,
                    "official_camera_id": i,
                    "archived_case_role": k,
                    "all_float32_bits_identical": full,
                    **c,
                }
            )
    passed = len(result) == 12 and all(
        x["all_float32_bits_identical"] and x["finite_disagreements"] == x["state_disagreements"] == 0
        for x in result
    )
    put(
        a.out / "external-id-parity.json",
        {
            "schema": "el-bundle-native-id-parity-1",
            "run_id": os.getenv("EL_RUN_ID"),
            "passed": passed,
            "original_case_sha256": old["case_sha256"],
            "old_scores_sha256": sha((a.old / "scores.json").read_bytes()),
            "source_order": [24, 23, 17, 22],
            "primary_source_id": 24,
            "common_endpoints": [24, 22],
            "target_order": [16, 18, 25, 37, 35, 36],
            "original_target_reference_reads": 0,
            "rows": result,
        },
    )
    check(passed, "actual captured-ID conversion changes native results")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("old", "new", "provenance", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    grade(p.parse_args())
