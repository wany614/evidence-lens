"""Independently recompute every public fixture colour, opacity, depth and calibration comparison on CPU."""

import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
from evidence_lens_bundle.common import require, sha, write


def work(a):
    p = json.loads(a.protocol.read_bytes())
    receipt = json.loads((a.windows / "parity.json").read_bytes())
    spec = p["comparisons"]
    expected = spec["absolute_tolerance"]
    rows = []
    require(
        receipt["passed"]
        and receipt["protocol_sha256"] == sha(a.protocol.read_bytes())
        and receipt["fixture_sha256"] == sha((a.fixture / "fixture.json").read_bytes()),
        "wrong protocol or fixture for CUDA results",
    )
    keyed = {(r["asset"], r["colmap"], r["view_id"]): r for r in receipt["rows"]}
    require(
        len(keyed)
        == len(receipt["rows"])
        == len(spec["assets"]) * len(spec["camera_encodings"]) * len(spec["views"]),
        "unexpected or duplicate CUDA comparison",
    )
    original = json.loads((a.fixture / "native-cameras.json").read_bytes())
    for asset in spec["assets"]:
        for encoding in spec["camera_encodings"]:
            bundle = a.windows / "bundles" / asset / encoding
            meta = json.loads((bundle / "bundle.json").read_bytes())
            producer = json.loads((bundle / "producer.json").read_bytes())
            cameras = json.loads((bundle / "cameras.json").read_bytes())
            require(
                meta["source_order"] + meta["target_order"] == spec["views"]
                and meta["primary_source_id"] == p["scene"]["primary"]
                and meta["endpoint_source_ids"] == p["scene"]["endpoints"],
                "real IDs differ",
            )
            require(
                producer["depth_verified"]
                and sha((bundle / "depth-probe.json").read_bytes()) == producer["depth_probe_sha256"],
                "depth probe differs",
            )
            for view in spec["views"]:
                known = keyed[asset, encoding, view]
                src = a.windows / "native" / asset / (view + ".npz")
                dst = bundle / producer["frames"][view]["file"]["path"]
                require(
                    sha(src.read_bytes()) == known["native_sha256"]
                    and sha(dst.read_bytes()) == known["import_sha256"]
                    and known["bundle_sha256"] == sha((bundle / "bundle.json").read_bytes()),
                    "numerical file identity differs",
                )
                geom = max(
                    float(
                        np.max(np.abs(np.asarray(original[view][k]) - np.asarray(cameras["views"][view][k])))
                    )
                    for k in ("K", "c2w")
                )
                require(
                    geom == known["camera_max_abs"] and geom <= expected["camera_K_and_pose"],
                    "metric camera disagrees",
                )
                s = {"asset": asset, "encoding": encoding, "view": view, "camera_max_abs": geom}
                with np.load(src, allow_pickle=False) as left, np.load(dst, allow_pickle=False) as right:
                    for channel, key in (("rgb", "rgb"), ("alpha", "alpha"), ("z", "expected_camera_Z_m")):
                        x = left[channel]
                        y = right[channel]
                        ok = np.isfinite(x)
                        good = np.isfinite(y)
                        require(np.array_equal(ok, good), "non-finite support changed")
                        if channel == "z":
                            require(
                                np.array_equal(ok, left["alpha"] > 0)
                                and np.array_equal(good, right["alpha"] > 0),
                                "depth/opacity support mismatch",
                            )
                        delta = np.abs(x[ok].astype(np.float64) - y[ok])
                        maximum = float(np.max(delta))
                        mean = float(np.mean(delta))
                        g = known["channels"][channel]
                        require(
                            g["passed"]
                            and g["n"] == delta.size
                            and g["total"] == x.size
                            and g["support_difference"] == 0
                            and g["max_abs"] == maximum
                            and abs(g["mean_abs"] - mean) <= 1e-18
                            and maximum <= expected[key],
                            "independent numerical disagreement: " + channel,
                        )
                        s.update(
                            {
                                channel + "_finite": delta.size,
                                channel + "_total": x.size,
                                channel + "_max_abs": maximum,
                                channel + "_mean_abs": mean,
                            }
                        )
                rows.append(s)
    with (a.out / "render-parity.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write(
        a.out / "readback.json",
        {
            "schema": "el-b2-independent-asset-grade-1",
            "run_id": os.environ["EL_RUN_ID"],
            "original_windows_run": receipt["run_id"],
            "windows_parity_sha256": sha((a.windows / "parity.json").read_bytes()),
            "protocol_sha256": receipt["protocol_sha256"],
            "fixture_sha256": receipt["fixture_sha256"],
            "rows": len(rows),
            "channel_comparisons": 3 * len(rows),
            "camera_max_abs": max(r["camera_max_abs"] for r in rows),
            "channels": {
                c: {
                    "finite_compared": sum(r[c + "_finite"] for r in rows),
                    "total_components": sum(r[c + "_total"] for r in rows),
                    "max_abs": max(r[c + "_max_abs"] for r in rows),
                    "max_view_mean_abs": max(r[c + "_mean_abs"] for r in rows),
                    "support_differences": 0,
                }
                for c in ("rgb", "alpha", "z")
            },
            "csv_sha256": sha((a.out / "render-parity.csv").read_bytes()),
            "device": receipt["device"],
            "gsplat": receipt["gsplat"],
            "torch": receipt["torch"],
            "cuda": receipt["cuda"],
            "source_images_external": False,
            "passed": True,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("fixture", "protocol", "windows", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    work(p.parse_args())
