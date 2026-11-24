"""Render the registered original static trajectory ticket with the released CUDA asset renderer."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
from evidence_lens_asset.gaussian import load
from evidence_lens_asset.render import depth_probe, device, layer
from evidence_lens_bundle.bundle import Bundle
from evidence_lens_bundle.common import (
    PROFILE, RELEASE_INDEX_SHA256, SEMANTICS, encoded, member, profile, require, row, sha, write,
)


def file_record(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return row(name, data)


def render_sequence(root, ticket, spec, output, torch, renderer, probe, costs):
    start = time.perf_counter()
    asset = spec["asset"]
    values = load(root / asset["path"], spec["gaussian_profile"], spec["meters_per_unit"])
    order, targets = spec["source_order"], spec["target_order"]
    dest = output / spec["id"]
    dest.mkdir()
    (dest / "renders").mkdir()
    cams = json.loads(member(root, spec["camera"]))
    cam_record = file_record(dest, "cameras.json", encoded(cams))
    probe_record = file_record(dest, "depth-probe.json", encoded(probe))
    render_record = file_record(dest, "renderer.json", encoded(renderer))
    gen = file_record(dest, "map-origin.json", encoded(spec["map_origin"]))
    photos = {
        s: file_record(dest, f"sources/{i:03d}.png", member(root, spec["photos"][s]))
        for i, s in enumerate(order)
    }
    frames = {}
    for i, view in enumerate(order + targets):
        camera = cams["views"][view]
        torch.cuda.synchronize()
        before = time.perf_counter()
        arrays = layer(torch, values, camera)
        torch.cuda.synchronize()
        rendered = time.perf_counter()
        name = f"renders/{i:03d}.npz"
        np.savez_compressed(dest / name, **arrays)
        record = row(name, (dest / name).read_bytes())
        written = time.perf_counter()
        frames[view] = {"status": "ready", "file": record}
        costs.append({
            "sequence_id": spec["id"], "role": "source_self_render" if view in order else "target_render",
            "view_id": view, "camera_sha256": sha(encoded(camera)), "map_sha256": asset["sha256"],
            "frame_sha256": record["sha256"], "output_bytes": record["bytes"],
            "render_sync_transfer_to_host_seconds": rendered - before,
            "encode_write_hash_seconds": written - rendered,
        })
    producer = {
        "schema": "el-render-evidence-1",
        "generator": {"run_id": spec["map_origin"]["run_id"], "receipt_sha256": gen["sha256"]},
        "renderer": {"run_id": os.environ["EL_RUN_ID"], "receipt_sha256": render_record["sha256"]},
        "checkpoint_sha256": asset["sha256"], "source_rgb_sha256": {x: photos[x]["sha256"] for x in order},
        "source_order": order, "target_order": targets, "camera_sha256": cam_record["sha256"],
        "semantics": SEMANTICS, "depth_verified": True, "depth_probe_sha256": probe_record["sha256"],
        "frames": frames,
    }
    p = file_record(dest, "producer.json", encoded(producer))
    write(dest / "bundle.json", {
        "schema": "el-evidence-bundle-1", "condition": f"M{len(order)}",
        "source_order": order, "primary_source_id": order[0], "endpoint_source_ids": [order[0], order[-1]],
        "target_order": targets, "input_domain": spec["scene"] + ", authored static engineering trajectory",
        "creation": {"run_id": os.environ["EL_RUN_ID"], "implementation": "registered gsplat asset trajectory",
                     "code_sha256": sha(Path(__file__).read_bytes()), "config_sha256": sha(encoded(spec)),
                     "original_case_sha256": None},
        "camera": cam_record, "geometry": {
            "prepared_camera_and_z_unit": "m", "meters_per_original_unit": spec["meters_per_unit"],
            "metric_evidence": spec["metric_evidence"], "crop": "none", "resize": "native 256 squared",
            "undistortion": "native undistorted pinhole", "scoring_resampling": "none"},
        "sources": photos, "producer": p,
        "risk_profile": {"id": PROFILE, "sha256": sha(encoded(profile())), "release_index_sha256": RELEASE_INDEX_SHA256},
    })
    checked = Bundle(dest)
    checked.validate()
    source_revision = sha(encoded([[s, spec["photos"][s]["sha256"], cams["views"][s]] for s in order]))
    return {
        "sequence_id": spec["id"], "scene": spec["scene"], "bundle_sha256": checked.sha256,
        "condition": checked.condition, "map_revision": asset["sha256"], "source_revision": source_revision,
        "renderer_revision": render_record["sha256"], "distinct_target_poses": len({
            sha(encoded(cams["views"][v]["c2w"])) for v in targets}),
        "source_order": order, "source_photos": {s: photos[s]["sha256"] for s in order},
        "map_origin": spec["map_origin"], "wall_seconds_including_validation": time.perf_counter() - start,
    }


def execute(root, out):
    total = time.perf_counter()
    require(os.environ.get("EL_RUN_ID"), "run the trajectory worker under the recorder")
    require(not out.exists() or not any(out.iterdir()), "expected an empty recorder payload")
    out.mkdir(parents=True, exist_ok=True)
    ticket = json.loads((root / "ticket.json").read_bytes())
    require(ticket["schema"] == "el-static-trajectory-ticket-1", "unrecognized trajectory ticket")
    for record in ticket["files"]:
        member(root, record)
    torch = device()
    torch.cuda.reset_peak_memory_stats()
    renderer = {
        "schema": "el-static-trajectory-renderer-1", "run_id": os.environ["EL_RUN_ID"],
        "engine": "gsplat", "version": "1.5.3", "configuration": "RGB+ED, classic, eps2d=.3, near=.01/far=100m",
        "implementation_sha256": sha((root / "runtime/evidence_lens_asset/render.py").read_bytes()),
        "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "python": platform.python_version(), "numpy": np.__version__,
    }
    rows, sequences = [], []
    with torch.no_grad():
        probe = depth_probe(torch)
        for spec in ticket["sequences"]:
            sequences.append(render_sequence(root, ticket, spec, out, torch, renderer, probe, rows))
    with (out / "upstream-frames.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    notice = root / "ORIGINAL-EXAMPLE-NOTICE.txt"
    shutil.copyfile(notice, out / notice.name)
    write(out / "trajectories.json", {
        "schema": "el-static-trajectories-1", "run_id": os.environ["EL_RUN_ID"], "sequences": sequences,
        "ticket_sha256": sha((root / "ticket.json").read_bytes()), "renderer": renderer,
        "total_seconds": time.perf_counter() - total, "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
        "reserved_peak_bytes": torch.cuda.max_memory_reserved(), "risk_training_updates": 0,
        "cross_device_transfer": "Recorded by the receiving controller, outside this worker.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticket", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    execute(args.ticket, args.out)
