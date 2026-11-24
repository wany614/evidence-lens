"""Compare installed platforms from original per-view NPY bytes without target photographs or labels."""

import argparse
from contextlib import ExitStack
import hashlib
from io import BytesIO
import os
from pathlib import Path
import zipfile

import numpy as np

from v05_platform_replay import CASES, MODELS, VIEWS, compact_arrays, digest, load, require, save

CHANNELS = {"risk": np.float32, "valid": np.bool_, "state": np.uint8, "unknown_reason": np.uint8}
GRIDS = {**CASES, "historical-original": (256, 256)}


def mac_pack(a):
    index = load(a.parent_manifest)
    require(digest(a.parent_manifest) == a.manifest_sha256, "changed Mac parent manifest")
    members = {row["path"]: row["sha256"] for row in index["files"]}
    a.out.mkdir(exist_ok=True, parents=True)
    raw = compact_arrays(a.installed, "vision", a.out)
    parent = a.parent_manifest.parent.resolve()
    installation = a.installed / "installation.json"
    require(
        digest(installation) == members[installation.resolve().relative_to(parent).as_posix()],
        "changed installation receipt",
    )
    for key, record in raw["cells"].items():
        tag, method = key.split("/")
        root = (
            a.installed / "legacy-vision-0.5.0"
            if tag == "historical-original"
            else a.installed / "profiles-vision" / tag
        )
        scores = root / method / "scores" / "scores.json"
        require(
            members[scores.resolve().relative_to(parent).as_posix()] == record["scores_sha256"],
            "changed Mac scores index",
        )
        for v in load(scores)["views"]:
            path = (scores.parent / v["arrays"]["path"]).resolve()
            require(
                members[path.relative_to(parent).as_posix()] == record["views"][v["id"]]["sha256"],
                "Mac array absent from sealed parent",
            )
    save(
        a.out / "mac-arrays.json",
        {
            "schema": "el-v05-mac-array-projection-1",
            "run_id": os.environ["EL_RUN_ID"],
            "parent_manifest_sha256": a.manifest_sha256,
            "native_installation_sha256": digest(installation),
            "array_index_sha256": digest(a.out / "array-index.json"),
            "score_arrays_sha256": raw["archive_sha256"],
            "target_arrays": raw["target_arrays"],
        },
    )


def archive_data(stack, index, archive):
    raw = load(index)
    require(
        raw["schema"] == "el-v05-platform-arrays-1" and digest(archive) == raw["archive_sha256"],
        "changed score archive",
    )
    expected = {f"{tag}/{m}" for tag in GRIDS for m in raw["methods"]}
    require(
        set(raw["cells"]) == expected and raw["target_arrays"] == len(expected) * 6, "incomplete score matrix"
    )
    z = stack.enter_context(zipfile.ZipFile(archive))
    members = {f"{key}/{v}.npz" for key in expected for v in VIEWS}
    require(
        set(z.namelist()) == members and len(z.namelist()) == len(members), "unlisted or missing view array"
    )
    return raw, z


def read_array(data, tag, method, view):
    metadata, archive = data
    cell = metadata["cells"][f"{tag}/{method}"]
    require(
        tuple(cell["analysis_wh"]) == GRIDS[tag] and set(cell["views"]) == set(VIEWS), "wrong grid or targets"
    )
    ref = cell["views"][view]
    require(ref["path"] == f"{tag}/{method}/{view}.npz", "wrong view identity")
    buf = archive.read(ref["path"])
    require(
        len(buf) == ref["bytes"] and hashlib.sha256(buf).hexdigest() == ref["sha256"], "changed original NPZ"
    )
    with np.load(BytesIO(buf), allow_pickle=False) as n:
        require(set(n.files) == set(CHANNELS), "unexpected score channels")
        arrays = {k: n[k] for k in CHANNELS}
    shape = GRIDS[tag][::-1]
    require(
        all(t.dtype == CHANNELS[k] and t.shape == shape for k, t in arrays.items()),
        "wrong channel dtype or extent",
    )
    risk, valid, state, reason = (arrays[k] for k in CHANNELS)
    threshold = cell["model"]["threshold"]
    require(
        np.array_equal(np.isfinite(risk), valid)
        and np.isnan(risk[~valid]).all()
        and np.all((risk[valid] >= 0) & (risk[valid] <= 1)),
        "inconsistent risk or validity",
    )
    require(
        np.array_equal(valid, reason == 0)
        and np.all(state[~valid] == 255)
        and np.array_equal(state[valid], risk[valid].astype(np.float64) >= threshold),
        "inconsistent states, reasons or threshold",
    )
    return arrays


def quotas(left, right, valid_l, valid_r):
    def order(risk, valid):
        ids = np.flatnonzero(valid)
        return ids[np.lexsort((ids, risk.ravel()[ids]))]

    x, y = order(left["risk"], valid_l), order(right["risk"], valid_r)
    result = []
    for percent in (10, 20):
        nl, nr = percent * len(x) // 100, percent * len(y) // 100
        common = int(np.intersect1d(x[:nl], y[:nr], assume_unique=True).size)
        result.append(
            {
                "percent": percent,
                "mac_eligible": len(x),
                "host_eligible": len(y),
                "mac_selected": nl,
                "host_selected": nr,
                "removed": nl - common,
                "added": nr - common,
            }
        )
    return result


def compare_view(left, right, tau):
    v, w = left["valid"], right["valid"]
    paired = v & w
    x, y = left["risk"][paired], right["risk"][paired]
    delta = y.astype(np.float64) - x.astype(np.float64)
    crossings = int(np.count_nonzero((x.astype(np.float64) < tau) != (y.astype(np.float64) < tau)))
    diffs = {"risk": int(np.count_nonzero(left["risk"].view(np.uint32) != right["risk"].view(np.uint32)))}
    diffs.update(
        {k: int(np.count_nonzero(left[k] != right[k])) for k in ("valid", "state", "unknown_reason")}
    )
    return {
        "canvas": v.size,
        "mac_finite": int(v.sum()),
        "host_finite": int(w.sum()),
        "common_finite": len(delta),
        "changed_pixels": diffs,
        "bit_exact": not any(diffs.values()),
        "threshold": tau,
        "threshold_crossings": crossings,
        "max_abs_common": float(np.abs(delta).max()) if delta.size else None,
        "min_signed_host_minus_mac": float(delta.min()) if delta.size else None,
        "max_signed_host_minus_mac": float(delta.max()) if delta.size else None,
        "sum_signed_host_minus_mac": float(delta.sum()),
        "sum_abs": float(np.abs(delta).sum()),
        "quota_on_method_finite": quotas(left, right, v, w),
    }


def summary(rows):
    maxes = [r["max_abs_common"] for r in rows if r["common_finite"]]
    lows = [r["min_signed_host_minus_mac"] for r in rows if r["common_finite"]]
    highs = [r["max_signed_host_minus_mac"] for r in rows if r["common_finite"]]
    common = sum(r["common_finite"] for r in rows)
    q = [e for r in rows for e in r["quota_on_method_finite"]]
    return {
        "views": len(rows),
        "bit_exact_views": sum(r["bit_exact"] for r in rows),
        **{
            k: sum(r[k] for r in rows)
            for k in ("canvas", "mac_finite", "host_finite", "common_finite", "threshold_crossings")
        },
        "changed_pixels": {k: sum(r["changed_pixels"][k] for r in rows) for k in CHANNELS},
        "max_abs_common": max(maxes, default=None),
        "min_signed_host_minus_mac": min(lows, default=None),
        "max_signed_host_minus_mac": max(highs, default=None),
        "mean_signed_host_minus_mac": sum(r["sum_signed_host_minus_mac"] for r in rows) / common
        if common
        else None,
        "mean_abs_common": sum(r["sum_abs"] for r in rows) / common if common else None,
        "method_finite_quota_tests": len(q),
        "changed_method_finite_quota_tests": sum(bool(e["added"] or e["removed"]) for e in q),
        "quota_members_added": sum(e["added"] for e in q),
        "quota_members_removed": sum(e["removed"] for e in q),
    }


def compare(a):
    a.out.mkdir(exist_ok=True, parents=True)
    with ExitStack() as stack:
        mac = archive_data(stack, a.mac_index, a.mac_arrays)
        host = archive_data(stack, a.host_index, a.host_arrays)
        methods = host[0]["methods"]
        require(set(methods) <= set(mac[0]["methods"]), "missing common methods")
        require(
            set(methods) == (set(MODELS) if a.host == "Windows" else {"MS-L1"}),
            "unexpected host method roster",
        )
        details = {}
        for tag in GRIDS:
            for m in methods:
                key = tag + "/" + m
                l, r = mac[0]["cells"][key], host[0]["cells"][key]
                require(
                    l["model"] == r["model"] and l["bundle_sha256"] == r["bundle_sha256"],
                    "crossed data or model identity",
                )
                details[key] = {
                    "grid": tag,
                    "analysis_wh": l["analysis_wh"],
                    "method": m,
                    "model": l["model"],
                    "bundle_sha256": l["bundle_sha256"],
                    "views": {},
                }
            for view in VIEWS:
                left = {m: read_array(mac, tag, m, view) for m in methods}
                right = {m: read_array(host, tag, m, view) for m in methods}
                common_l = np.logical_and.reduce([v["valid"] for v in left.values()])
                common_r = np.logical_and.reduce([v["valid"] for v in right.values()])
                for m in methods:
                    cell = details[tag + "/" + m]
                    row = compare_view(left[m], right[m], cell["model"]["threshold"])
                    if len(methods) == 4:
                        row["quota_four_method_common_finite"] = quotas(left[m], right[m], common_l, common_r)
                    cell["views"][view] = row
    totals = {key: summary(list(cell["views"].values())) for key, cell in details.items()}
    save(
        a.out / "views.json",
        {
            "schema": "el-v05-cross-platform-views-1",
            "run_id": os.environ["EL_RUN_ID"],
            "host": a.host,
            "threshold": "applied to FP32 scores promoted to float64",
            "quota": "per-view floor(q*eligible), ascending original FP32 score then flattened row-major index; no target labels",
            "mac_native_array_sha256": digest(a.mac_arrays),
            "host_native_array_sha256": digest(a.host_arrays),
            "cells": details,
        },
    )
    save(
        a.out / "differences.json",
        {
            "schema": "el-v05-cross-platform-numerics-1",
            "run_id": os.environ["EL_RUN_ID"],
            "host": a.host,
            "mac_index_sha256": digest(a.mac_index),
            "host_index_sha256": digest(a.host_index),
            "views_sha256": digest(a.out / "views.json"),
            "cells": totals,
            "method_totals": {
                m: summary(
                    [r for cell in details.values() if cell["method"] == m for r in cell["views"].values()]
                )
                for m in methods
            },
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("mac-pack", "compare"))
    for k in ("out", "mac-index", "mac-arrays", "host-index", "host-arrays", "installed", "parent-manifest"):
        p.add_argument("--" + k, type=Path, required=k == "out")
    p.add_argument("--manifest-sha256")
    p.add_argument("--host", choices=("Windows", "Linux"))
    args = p.parse_args()
    (mac_pack if args.action == "mac-pack" else compare)(args)
