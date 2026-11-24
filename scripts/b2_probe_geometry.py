"""Build virtual and candidate cameras from the ten allowed physical camera poses."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require


def rank(seed, i):
    return hashlib.sha256(f"q|{seed}|{i}".encode()).digest()


def rotation(matrix):
    u, _, vt = np.linalg.svd(matrix)
    adjust = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    return u @ adjust @ vt


def generate(rig, src, protocol):
    s0, pool = protocol["physical_endpoint_ids"], protocol["candidate_ids"]
    require(
        rig["endpoint_ids"] == s0
        and rig["candidate_ids"] == pool
        and set(rig["cameras"]) == {str(i) for i in [*s0, *pool]}
        and set(src["views"]) == {"s00", "s03"},
        "query builder accepts only endpoints and permitted candidates",
    )
    cameras = rig["cameras"]
    for key, i in zip(("s00", "s03"), s0, strict=True):
        for n in ("K", "c2w", "clip_m"):
            require(
                np.allclose(
                    src["views"][key][n], cameras[str(i)][n], atol=1e-10, rtol=0
                ),
                "endpoint frame changed",
            )
    doc = {
        **src,
        "views": {**src["views"], **{f"c{i:03}": cameras[str(i)] for i in pool}},
    }
    q = protocol["virtual_query"]
    fraction = q["candidate_fraction"]
    origins = [cameras[str(i)] for i in s0]
    track = []
    for n, i in enumerate(
        sorted(pool, key=lambda i: (rank(q["seed"], i), i))[: q["count"]]
    ):
        end = cameras[str(i)]
        k, pose = [
            sum(
                np.array(x[field]) * w
                for x, w in zip(
                    [*origins, end],
                    [(1 - fraction) / 2, (1 - fraction) / 2, fraction],
                    strict=True,
                )
            )
            for field in ("K", "c2w")
        ]
        pose[:3, :3] = rotation(pose[:3, :3])
        require(
            all(
                (c["height"], c["width"], c["clip_m"])
                == (end["height"], end["width"], end["clip_m"])
                for c in origins
            ),
            "camera dimensions or clips differ",
        )
        name = f"q{n:02}"
        doc["views"][name] = {**end, "c2w": pose.tolist(), "K": k.tolist()}
        track.append({"query": name, "candidate_anchor_id": i})
    return doc, track


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("rig", "source", "protocol", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    cams, trace = generate(load(a.rig), load(a.source), load(a.protocol))
    save(a.out / "cameras.json", cams)
    save(
        a.out / "geometry.json",
        {
            "schema": "el-b2-query-geometry-1",
            "cameras_sha256": sha(a.out / "cameras.json"),
            "protocol_sha256": sha(a.protocol),
            "rig_sha256": sha(a.rig),
            "source_sha256": sha(a.source),
            "trace": trace,
        },
    )
