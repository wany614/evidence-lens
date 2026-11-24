"""Independent comparison with live archived numerics and pre-existing public expected risk arrays."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile

import numpy as np

from bundle_build import run
from bundle_grade import compare, read
from bundle_original import EXAMPLE, METHODS, check, put, sha
from bundle_score_record import released_models


def grade(a):
    rows = []
    with tempfile.TemporaryDirectory(dir=a.out.parent) as tmp:
        temp = Path(tmp)
        cpu = temp / "cpu"
        legacy = temp / "outputs"
        released_models(a.cpu, cpu)
        run(
            [
                a.vision_python,
                "-I",
                "-B",
                Path(__file__).with_name("bundle_legacy_replay.py"),
                "--cpu",
                cpu,
                "--cases",
                a.cases,
                "--out",
                legacy,
            ]
        )
        with a.example.open("rb") as f:
            check(hashlib.file_digest(f, "sha256").hexdigest() == EXAMPLE, "wrong archived example")
        with ZipFile(a.example) as ex:
            files = json.loads(ex.read("example.json"))["files"]

            def bound(path):
                raw = ex.read(path)
                check(
                    len(raw) == files[path]["bytes"] and sha(raw) == files[path]["sha256"],
                    "invalid original example member",
                )
                return raw

            for c in ("M2", "M4"):
                for method in METHODS:
                    root = a.scores / c / method
                    score = json.loads((root / "scores.json").read_bytes())
                    scope = "one" if method == "B-L1-transfer" else "all"
                    orig_index = "EXPECTED/" + method + "/native-scores.json"
                    originals = json.loads(bound(orig_index))
                    check(
                        originals["model_index_sha256"] == score["model"]["model_index_sha256"]
                        and originals["threshold"] == score["model"]["threshold"],
                        "score model/threshold differs",
                    )
                    views = {q["view"]: q for q in originals["views"] if q["condition"] == c}
                    for view in score["views"]:
                        t = view["id"]
                        spec = views[t]["modes"][scope]["file"]
                        archived = bound("EXPECTED/" + method + "/" + spec["path"])
                        check(sha(archived) == spec["sha256"], "expected map differs")
                        with (
                            np.load(read(root, view["arrays"]), allow_pickle=False) as new,
                            np.load(legacy / c / method / (t + ".npz"), allow_pickle=False) as old,
                            np.load(io.BytesIO(archived), allow_pickle=False) as gold,
                        ):
                            direct = compare(new, old)
                            prior = compare(new, gold)
                            bits = np.array_equal(new["risk"].view(np.uint32), old["risk"].view(np.uint32))
                        rows.append(
                            {
                                "condition": c,
                                "target_id": t,
                                "method": method,
                                "scope": scope,
                                "new_array_sha256": view["arrays"]["sha256"],
                                "original_published_sha256": spec["sha256"],
                                "fresh_legacy_all_float32_bits_equal": bits,
                                "fresh_legacy": direct,
                                "published_expected": prior,
                            }
                        )
        ok = all(
            q["fresh_legacy_all_float32_bits_equal"]
            and q["fresh_legacy"]["state_disagreements"] == q["fresh_legacy"]["finite_disagreements"] == 0
            for q in rows
        )
        public_ok = all(
            q["published_expected"]["state_disagreements"]
            == q["published_expected"]["finite_disagreements"]
            == 0
            and q["published_expected"]["max_absolute_difference"] <= 1e-4
            for q in rows
        )
        receipt = {
            "schema": "el-bundle-published-original-regression-1",
            "run_id": os.getenv("EL_RUN_ID"),
            "conditions": 2,
            "methods": list(METHODS),
            "observations": len(rows),
            "canvas_method_pixels": sum(q["fresh_legacy"]["canvas"] for q in rows),
            "fresh_legacy_strict_passed": ok,
            "published_expected_state_and_tolerance_passed": public_ok,
            "published_tolerance": 1e-4,
            "original_runtime": json.loads((legacy / "runtime.json").read_bytes()),
            "rows": rows,
        }
        put(a.out / "compatibility.json", receipt)
        check(
            len(rows) == 48 and ok and public_ok,
            "fresh legacy parity or comparison with published scores differs",
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("cpu", "cases", "scores", "example", "vision-python", "out"):
        p.add_argument("--" + n, required=True, type=Path)
    grade(p.parse_args())
