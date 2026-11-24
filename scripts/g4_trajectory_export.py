"""Check rendered trajectories against their original request and emit a portable deployment-only archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np

from evidence_lens_bundle import common
from evidence_lens_bundle.bundle import Bundle
from evidence_lens_bundle.common import encoded, json_bytes, member, parts, require, sha, write

README = """# Original static camera trajectories

These are CPU-ready RGB, camera-Z and opacity predictions of the original helical beads and offset bell cage.
Each main bundle contains 120 distinct calibrated views on a smooth closed path.
The two shorter bead bundles use a genuine second Gaussian snapshot and a different source observation.
Credit wany614 / Evidence Lens under CC BY 4.0. Source and renderer tools use Apache-2.0.

After installing the Evidence Lens engineering package, run:
    python -m evidence_lens_bundle.replay --bundle bell-M4 --hz 10 --seconds 60 --out results
    python -m evidence_lens_bundle.replay --bundle beads-M4 --switch 20=beads-map2-M4 --switch 40=beads-map2-source2-M4 --hz 10 --seconds 6 --out revisions

Open results/replay.html or revisions/replay.html offline. Existing arrays are CPU-scored.
New Gaussian views require the separate CUDA renderer. These are static engineering stress inputs.
"""


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(a):
    require(os.environ.get("EL_RUN_ID"), "record the upstream input verification and archive")
    if a.assets:
        # The source recorder snapshots .py files; use independently pinned profile resources.
        common.ASSETS = a.assets
    doc = json_bytes((a.trajectories / "trajectories.json").read_bytes())
    index = {s["sequence_id"]: s for s in doc["sequences"]}
    require(
        list(index) == [s["sequence_id"] for s in doc["sequences"]] and len(index) == 6,
        "wrong sequence count or duplicate",
    )
    with (a.trajectories / "upstream-frames.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    measured = {(r["sequence_id"], r["view_id"]): r for r in rows}
    require(len(measured) == len(rows) == 548, "wrong or duplicate upstream render cost rows")
    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / "evidence-lens-static-trajectories.zip"
    reports = []
    with ZipFile(a.ticket) as request, ZipFile(dest, "x", compression=ZIP_STORED) as z:
        raw = request.read("ticket.json")
        ticket = json_bytes(raw)
        require(sha(raw) == doc["ticket_sha256"], "worker references a different render request")
        require(set(index) == {s["id"] for s in ticket["sequences"]}, "unexpected output sequence identities")
        for spec in ticket["sequences"]:
            name = spec["id"]
            seq = index[name]
            root = a.trajectories / name
            b = Bundle(root, seq["bundle_sha256"])
            b.validate()
            cameras = json_bytes(member(root, b.meta["camera"]))
            require(
                b.meta["camera"]["sha256"] == spec["camera"]["sha256"]
                and b.producer["checkpoint_sha256"] == seq["map_revision"] == spec["asset"]["sha256"]
                and tuple(spec["source_order"]) == b.order
                and tuple(spec["target_order"]) == b.targets
                and seq["source_photos"] == {s: spec["photos"][s]["sha256"] for s in b.order},
                "prepared evidence differs from original planned cameras, map or authorized sources",
            )
            source_hash = sha(
                encoded([[s, b.meta["sources"][s]["sha256"], cameras["views"][s]] for s in b.order])
            )
            require(
                source_hash == seq["source_revision"]
                and all(b.meta["sources"][s]["sha256"] == spec["photos"][s]["sha256"] for s in b.order),
                "source version or photograph changed in transit",
            )
            require(
                b.producer["renderer"]["run_id"] == doc["run_id"]
                and b.producer["renderer"]["receipt_sha256"] == seq["renderer_revision"]
                and seq["renderer_revision"] == hash_file(root / "renderer.json")
                and b.producer["depth_probe_sha256"] == hash_file(root / "depth-probe.json")
                and b.producer["generator"]["receipt_sha256"] == hash_file(root / "map-origin.json")
                and json_bytes((root / "map-origin.json").read_bytes()) == spec["map_origin"],
                "upstream renderer, original map receipt, or depth probe identity differs",
            )
            poses = np.stack([b.cameras[v].c2w for v in b.targets])
            pair = np.concatenate((poses, poses[:1])) if len(poses) == 120 else poses
            step = np.linalg.norm(np.diff(pair[:, :3, 3], axis=0), axis=1)
            cos = (np.einsum("nij,nij->n", pair[:-1, :3, :3], pair[1:, :3, :3]) - 1) / 2
            angle = np.arccos(np.clip(cos, -1, 1))
            require(
                len(set(x.tobytes() for x in poses)) == len(poses)
                and len(poses) == seq["distinct_target_poses"]
                and np.all(step > 0)
                and step.max() < 0.06
                and angle.max() < 0.05,
                "pose grid is discontinuous or duplicate",
            )
            totals = {"target_render": 0.0, "source_self_render": 0.0, "encode_write_hash_seconds": 0.0}
            files = {
                "bundle.json",
                "cameras.json",
                "producer.json",
                "renderer.json",
                "map-origin.json",
                "depth-probe.json",
            }
            for s in b.order:
                files.add(b.meta["sources"][s]["path"])
            for v in (*b.order, *b.targets):
                frame = b.producer["frames"][v]["file"]
                record = measured[(name, v)]
                role = "source_self_render" if v in b.order else "target_render"
                require(
                    record["role"] == role
                    and record["map_sha256"] == seq["map_revision"]
                    and record["camera_sha256"] == sha(encoded(cameras["views"][v]))
                    and record["frame_sha256"] == frame["sha256"]
                    and int(record["output_bytes"]) == frame["bytes"],
                    "raw upstream rows or input hashes changed",
                )
                for key in ("render_sync_transfer_to_host_seconds", "encode_write_hash_seconds"):
                    seconds = float(record[key])
                    require(math.isfinite(seconds) and seconds >= 0, "upstream cost is invalid")
                    totals[role if key.startswith("render_") else key] += seconds
                files.add(frame["path"])
            for f in sorted(files):
                parts(f)
                z.write(root / f, name + "/" + f)
            reports.append(
                {
                    "sequence_id": name,
                    "bundle_sha256": b.sha256,
                    "n_target": len(b.targets),
                    "n_sources": len(b.order),
                    "map_sha256": seq["map_revision"],
                    "source_sha256": source_hash,
                    "max_translation_step_m": float(step.max()),
                    "max_rotation_step_rad": float(angle.max()),
                    **totals,
                    "map_origin": spec["map_origin"],
                }
            )
        for f in ("trajectories.json", "upstream-frames.csv", "ORIGINAL-EXAMPLE-NOTICE.txt"):
            z.write(a.trajectories / f, f)
        for f in ("LICENSE", "NOTICE", "CC-BY-4.0.txt"):
            z.writestr(f, request.read(f))
        z.writestr("README.md", README.encode())
    write(
        a.out / "trajectory-verification.json",
        {
            "schema": "el-original-static-export-1",
            "run_id": os.environ["EL_RUN_ID"],
            "renderer_run_id": doc["run_id"],
            "renderer_ticket_sha256": doc["ticket_sha256"],
            "sequences": reports,
            "raw_cost_rows": len(rows),
            "archive": {"file": dest.name, "sha256": hash_file(dest), "bytes": dest.stat().st_size},
            "release_scope": "Only original authorized deployment evidence, its measured per-frame costs and required notices.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("out", "ticket", "trajectories", "assets"):
        p.add_argument("--" + k, type=Path, required=k != "assets")
    export(p.parse_args())
