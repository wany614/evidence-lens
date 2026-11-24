"""Compare published original score arrays with a new CPU score directory. Requires NumPy."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path, spec):
    rel = Path(spec["path"])
    if rel.is_absolute() or ".." in rel.parts or not (path / rel).resolve().is_relative_to(path.resolve()):
        raise ValueError("unsafe score member")
    src = path / rel
    data = src.read_bytes()
    if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError("unexpected member bytes: " + str(rel))
    return src


def compare(first, second):
    a, b = first["risk"], second["risk"]
    for k in ("risk", "valid", "state"):
        if first[k].shape != (256, 256) or second[k].shape != first[k].shape:
            raise ValueError("score shape differs")
    af = np.isfinite(a)
    bf = np.isfinite(b)
    common = af & bf
    if (
        a.dtype != np.float32
        or b.dtype != np.float32
        or not np.array_equal(af, first["valid"])
        or not np.array_equal(bf, second["valid"])
    ):
        raise ValueError("malformed numerical risk")
    errors = np.abs(a[common].astype(np.float64) - b[common])
    return {
        "canvas": int(af.size),
        "finite_together": int(common.sum()),
        "finite_disagreements": int(np.count_nonzero(af != bf)),
        "state_disagreements": int(np.count_nonzero(first["state"] != second["state"])),
        "finite_float32_bits_differ": int(
            np.count_nonzero(a[common].view(np.uint32) != b[common].view(np.uint32))
        ),
        "max_absolute_difference": float(errors.max()) if errors.size else None,
    }


def grade(scores, expected, tolerance=0.0001):
    new = json.loads((scores / "scores.json").read_bytes())
    gold = json.loads((expected / "expected.json").read_bytes())
    if (
        new["bundle_sha256"],
        new["condition"],
        new["method"],
        new["model"]["model_index_sha256"],
        new["model"]["threshold"],
    ) != (
        gold["bundle_sha256"],
        gold["condition"],
        gold["method"],
        gold["model_index_sha256"],
        gold["threshold"],
    ):
        raise ValueError("bundle/condition/method/threshold differs")
    if new["target_order"] != list(gold["view_order"]):
        raise ValueError("target roster differs")
    rows = {}
    for n, v in zip(new["target_order"], new["views"], strict=True):
        if v["id"] != n:
            raise ValueError("target order differs")
        with (
            np.load(read(scores, v["arrays"]), allow_pickle=False) as a,
            np.load(read(expected, gold["views"][n]), allow_pickle=False) as b,
        ):
            rows[n] = compare(a, b)
    good = all(
        r["finite_disagreements"] == r["state_disagreements"] == 0
        and r["max_absolute_difference"] is not None
        and r["max_absolute_difference"] <= tolerance
        for r in rows.values()
    )
    return {
        "schema": "el-cross-platform-bundle-grade-1",
        "passed": good,
        "absolute_tolerance": tolerance,
        "all_finite_float32_bitwise_equal": all(q["finite_float32_bits_differ"] == 0 for q in rows.values()),
        "views": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("scores", "expected", "out"):
        p.add_argument("--" + n, required=True, type=Path)
    a = p.parse_args()
    r = grade(a.scores, a.expected)
    with a.out.open("x", encoding="utf-8") as f:
        json.dump(r, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(r))
    raise SystemExit(0 if r["passed"] else 1)
