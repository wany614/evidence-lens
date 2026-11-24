"""Independently replay the pinned original helical-bead distribution on a CPU."""

import argparse
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

from bundle_build import environment, run
from bundle_original import check, put, sha


def checked_zip(archive, receipt, destination):
    check(archive.stat().st_size == receipt["bytes"], "portable archive size differs")
    check(sha(archive.read_bytes()) == receipt["sha256"], "portable archive SHA-256 differs")
    with ZipFile(archive) as z:
        names = z.namelist()
        check(len(set(names)) == len(names) == receipt["members"], "ambiguous ZIP inventory")
        index = z.read("manifest.json")
        check(sha(index) == receipt["manifest_sha256"], "portable inventory SHA-256 differs")
        expected = json.loads(index)
        check(set(expected) == set(names) - {"manifest.json"}, "incomplete portable inventory")
        for name, entry in expected.items():
            p = PurePosixPath(name)
            check(
                p.as_posix() == name
                and not p.is_absolute()
                and ".." not in p.parts
                and "\\" not in name
                and all(":" not in part for part in p.parts),
                "non-portable member path",
            )
            raw = z.read(name)
            check(len(raw) == entry["bytes"] and sha(raw) == entry["sha256"], "member differs: " + name)
            to = destination.joinpath(*p.parts)
            to.parent.mkdir(parents=True, exist_ok=True)
            to.write_bytes(raw)
    return expected


def compare(original, replay, arrays, gold):
    check(original["target_order"] == replay["target_order"] == ["70", "109", "13"], "target IDs differ")
    check(original["source_order"] == replay["source_order"] == ["41", "5"], "source IDs differ")
    check(
        original["model"] == replay["model"] and original["counts"] == replay["counts"],
        "score identity differs",
    )
    rows = []
    for origin, actual in zip(original["views"], replay["views"], strict=True):
        target = actual["id"]
        check(origin["id"] == target and actual["counts"] == origin["counts"], "target counts differ")
        known = gold / origin["arrays"]["path"]
        distributed = arrays / "zip" / "expected" / (target + ".npz")
        generated = arrays / "scores" / actual["arrays"]["path"]
        check(
            sha(known.read_bytes()) == origin["arrays"]["sha256"] == sha(distributed.read_bytes()),
            "distributed target differs from the independent original",
        )
        check(sha(generated.read_bytes()) == actual["arrays"]["sha256"], "new output checksum differs")
        with np.load(known, allow_pickle=False) as before, np.load(generated, allow_pickle=False) as after:
            check(
                set(before.files) == set(after.files) == {"risk", "valid", "state", "unknown_reason"},
                "unexpected score arrays",
            )
            diffs = {}
            for k in before.files:
                x, y = before[k], after[k]
                check(x.shape == y.shape == (256, 256) and x.dtype == y.dtype, "array layout differs")
                if k == "risk":
                    check(x.dtype == np.float32, "risk must be float32")
                    x, y = x.view(np.uint32), y.view(np.uint32)
                diffs[k] = int(np.count_nonzero(x != y))
            check(not any(diffs.values()), "clean replay changed numerical results")
            rows.append(
                {
                    "target": target,
                    "pixels": before["state"].size,
                    "differences": diffs,
                    "original_sha256": origin["arrays"]["sha256"],
                    "replayed_sha256": actual["arrays"]["sha256"],
                }
            )
    return rows


def verify(a):
    meta = json.loads(a.receipt.read_bytes())
    check(sha(a.receipt.read_bytes()) == a.receipt_sha256, "independent distribution receipt differs")
    original = json.loads((a.original_scores / "scores.json").read_bytes())
    with tempfile.TemporaryDirectory(dir=a.out.parent) as name:
        tmp = Path(name)
        extracted = tmp / "zip"
        members = checked_zip(a.archive, meta, extracted)
        wheel = extracted / "evidence_lens-0.4.0b1-py3-none-any.whl"
        check(sha(wheel.read_bytes()) == a.wheel_sha256, "wheel differs from original build")
        cli, installed = environment(tmp, wheel)
        prepared = extracted / "prepared"
        bundle_sha = sha((prepared / "bundle.json").read_bytes())
        check(bundle_sha == original["bundle_sha256"], "wrong prepared example")
        run([cli, "validate", "--bundle", prepared, "--bundle-sha256", bundle_sha])
        scores = tmp / "scores"
        run([cli, "score", "--bundle", prepared, "--bundle-sha256", bundle_sha, "--out", scores])
        rerun = json.loads((scores / "scores.json").read_bytes())
        rows = compare(original, rerun, tmp, a.original_scores)
        html = tmp / "replayed.html"
        run([cli, "report", "--scores", scores, "--out", html])
        for public, field in [
            ("prepared-report.html", "example_html_sha256"),
            ("outside-scope-report.html", "outside_html_sha256"),
        ]:
            check(sha((extracted / public).read_bytes()) == meta[field], "distributed HTML differs")
    put(
        a.out / "portable-readback.json",
        {
            "schema": "el-b2-portable-independent-readback-1",
            "run_id": os.environ.get("EL_RUN_ID"),
            "archive_sha256": meta["sha256"],
            "receipt_sha256": a.receipt_sha256,
            "verified_zip_members": len(members) + 1,
            "wheel_sha256": a.wheel_sha256,
            "installed": installed,
            "bundle_sha256": bundle_sha,
            "original_scores_sha256": sha((a.original_scores / "scores.json").read_bytes()),
            "target_rows": rows,
            "total_pixels": sum(x["pixels"] for x in rows),
            "fresh_html_generated": True,
            "target_reference_images_opened": 0,
            "passed": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("archive", "receipt", "original-scores", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("receipt-sha256", "wheel-sha256"):
        parser.add_argument("--" + flag, required=True)
    verify(parser.parse_args())
