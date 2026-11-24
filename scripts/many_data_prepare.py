"""Materialize only the permitted photographs; keep authored optical truth in its evaluator role."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import tempfile
import time
import zipfile

import numpy as np
from PIL import Image

from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens._verify import require
from evidence_lens_bundle.grids import build_grids, resample_photo
from many_data_eth import extract_role
from many_data_original import Renderer, cameras

PROFILE = "analysis-378x252-v1"
SIZE = (378, 252)


def member(path, name):
    return {"path": name, "sha256": sha(path), "bytes": path.stat().st_size}


def photo(path, grid, output):
    start = time.perf_counter()
    with Image.open(path) as image:
        image.load()
        require(
            image.mode == "RGB" and list(image.size) == grid["original_size"],
            "camera/decoded-unaltered-native-RGB mismatch",
        )
        elapsed = time.perf_counter() - start
        resized = resample_photo(image, grid, SIZE)
        resized.save(output, format="PNG", compress_level=3)
        return elapsed


def subset_document(a, meta, views, src_ids, target_ids, original, analysis, permission):
    names = src_ids + target_ids
    grids = build_grids(PROFILE, {i: views[i] for i in names})
    per_camera = {}
    for i in names:
        g = grids["views"][i]
        v = views[i]
        per_camera[i] = {
            "width": SIZE[0],
            "height": SIZE[1],
            "K": g["analysis_K"],
            "c2w": v["c2w"],
            "original_name": v["original_name"],
            "original_size": g["original_size"],
            "original_K": g["original_K"],
            "distortion": g["distortion"],
            "original_to_analysis": g["original_to_analysis"],
        }
        for name in ("official_image_id", "official_camera_id", "official_corner_principal_point"):
            if name in v:
                per_camera[i][name] = v[name]
    frame = {
        "world_to_deployment": np.eye(4).tolist(),
        "unit": meta["unit"],
        "metric_anchor_ids": [],
        "basis": "right_down_forward_camera_in_original_world",
        "normalization": "fixed identity for all source subsets and held-out targets",
        "unit_evidence": "designed metres"
        if meta["unit"] == "m"
        else "official calibration frame; per-scene metre scale not asserted",
    }
    return {
        "schema": "el-many-source-data-1",
        "run_id": os.environ["EL_RUN_ID"],
        "A_BASELINE_ID": meta["baseline"],
        "scene_id": a.scene,
        "split": meta["split"],
        "configuration": f"R{len(src_ids)}",
        "R": src_ids,
        "A": src_ids,
        "H": target_ids,
        "views": per_camera,
        "grids": grids,
        "unit": meta["unit"],
        "frame_receipt": frame,
        "permission": permission,
        "source_rgb": {i: analysis[i] for i in src_ids},
        "original_rgb": {i: original[i] for i in src_ids},
        "original_rgb_location": "R-only original supplement, physical member identifiers and SHA-256; analysis PNG members are in this package.",
        "photometry": "8-bit display RGB as decoded without EXIF rotation; A integer centre crop and Pillow LANCZOS",
        "target_pixels_in_package": 0,
        "point_tracks_in_package": 0,
        "geometry_prior_in_package": 0,
    }


def validate_opening(a, meta, H, protocol):
    gate = load(a.opening)
    require(
        gate.get("schema") == "el-many-h-opening-1" and gate["allowed_targets"].get(a.scene) == H,
        "wrong H-role grant or ordered targets",
    )
    if meta["split"] == "confirmation":
        confirmation_terminals(gate, protocol)
    else:
        require(
            gate["phase"] == "train-development"
            and not any(
                r["id"] in gate["allowed_targets"]
                for r in protocol["originals"] + protocol["external"]["roster"]
                if r["split"] == "confirmation"
            ),
            "unexpected training/development access",
        )
    source = load(a.source_index)
    require(
        source["scene_id"] == a.scene and source["H"] == H and source["H_photo_decodes"] == 0,
        "wrong prior deployment permission",
    )


def confirmation_terminals(gate, protocol):
    """Validate the common opening across every target and every registered reconstruction."""
    scenes = [
        r for r in protocol["originals"] + protocol["external"]["roster"] if r["split"] == "confirmation"
    ]
    targets = {r["id"]: [f"h{i:02d}" for i in range(r.get("H", 6))] for r in scenes}
    require(
        gate.get("schema") == "el-many-h-opening-1"
        and gate.get("phase") == "confirmation"
        and gate.get("allowed_targets") == targets,
        "incomplete common confirmation target roster",
    )
    terminals = gate.get("terminals", {})
    canonical = json.dumps(terminals, sort_keys=True, separators=(",", ":")).encode()
    require(
        set(terminals) == set(targets)
        and hashlib.sha256(canonical).hexdigest() == gate.get("prediction_freeze_sha256"),
        "different frozen prediction/terminal matrix",
    )
    pins = [gate["model_freeze"]]
    for scene in scenes:
        sid = scene["id"]
        counts = ["16", "128"] if "seed" in scene else [str(scene["R"])]
        require(set(terminals[sid]) == set(counts), "missing confirmation reconstruction")
        for n in counts:
            item = terminals[sid][n]
            require(item.get("status") in {"succeeded", "failed"}, "missing terminal disposition")
            if item["status"] == "succeeded":
                outcomes = item["target_outcomes"]
                require(
                    set(outcomes) == set(targets[sid])
                    and set(outcomes.values()) <= {"ready", "missing_target_prediction"}
                    and item["receipt"]["run_id"] == item["scores_run"],
                    "different prediction view denominator",
                )
            else:
                require(item.get("native_failure"), "missing original failure evidence")
            pins.append(item["receipt"])
    return pins


def prepare(a, protocol):
    started, cpu = time.perf_counter(), time.process_time()
    value = next(r for r in protocol["originals"] + protocol["external"]["roster"] if r["id"] == a.scene)
    is_original = "seed" in value
    source_mode = a.action == "source-worker"
    role = "R" if source_mode else "H"
    metadata = load(a.roster) if not is_original else None
    views = cameras() if is_original else metadata["views"]
    R = sorted(i for i in views if i.startswith("r"))
    H = sorted(i for i in views if i.startswith("h"))
    ids = R if source_mode else H
    meta = {
        "split": value["split"],
        "unit": "m" if is_original else "scene_units",
        "baseline": protocol["A_BASELINE_ID"],
    }
    permission = {
        "scene_id": a.scene,
        "split": meta["split"],
        "R": R,
        "A": R,
        "H": H,
        "source_dataset": "original authored six-family study"
        if is_original
        else "ETH3D high-resolution multi-view, official undistorted DSLR JPEG",
        "license": "CC BY 4.0, wany614 (authored geometry, images and data)"
        if is_original
        else protocol["external"]["license"],
        "calibration": "registered authored pinhole cameras"
        if is_original
        else "official calibration may use the full acquisition",
        "rgb_access": "deployment only reads actual R photographs. H has camera metadata only.",
        "official_geometry_features_used": False,
    }
    grids = build_grids(PROFILE, views)
    if not source_mode:
        validate_opening(a, meta, H, protocol)
    with tempfile.TemporaryDirectory(dir=a.out.parent) as folder:
        tmp = Path(folder)
        raw = tmp / "original"
        raw.mkdir()
        analysis_dir = tmp / "analysis"
        analysis_dir.mkdir()
        render = Renderer(a.renderer, tmp, a.scene, value["seed"]) if is_original else None
        first_pixels_start = datetime.now(timezone.utc).isoformat()
        authored_render_starts = {}
        if is_original:
            originals = {}
            for i in ids:
                view = views[i]
                dest = raw / (i + ".png")
                authored_render_starts[i] = datetime.now(timezone.utc).isoformat()
                pixels = render.render(view["K"], view["c2w"], view["width"], view["height"], color=True)
                Image.fromarray(pixels).save(dest, compress_level=3)
                originals[i] = dest
        else:
            originals = extract_role(a.archive, metadata, ids, role, raw)
        first_pixels_end = datetime.now(timezone.utc).isoformat()
        originals_manifest = {}
        analysis = {}
        seconds = 0
        accesses = {}
        depth_checks = {}
        archive = a.out / ("R-originals.zip" if source_mode else "truth.zip")
        paths = {}
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as z:
            for i in ids:
                src = originals[i]
                raw_member = "original/" + i + src.suffix.lower()
                individual_read = datetime.now(timezone.utc).isoformat()
                originals_manifest[i] = member(src, raw_member)
                originals_manifest[i]["original_name"] = views[i]["original_name"]
                z.write(src, raw_member)
                ap = analysis_dir / (i + ".png")
                before = datetime.now(timezone.utc).isoformat()
                elapsed = photo(src, grids["views"][i], ap)
                seconds += elapsed
                accesses[i] = {
                    "role": role,
                    "original_bytes": src.stat().st_size,
                    "decode_start_utc": before,
                    "decode_seconds": elapsed,
                    "first_individual_raw_file_open_utc": individual_read,
                    "authored_render_start_utc": authored_render_starts.get(i),
                }
                analysis[i] = member(ap, "analysis/" + ap.name)
                if not source_mode:
                    z.write(ap, analysis[i]["path"])
            if not source_mode and is_original:
                for i in R + H:
                    cam = views[i]
                    optical_z = render.render(grids["views"][i]["analysis_K"], cam["c2w"], *SIZE, color=False)
                    require(
                        np.isfinite(optical_z).any() and np.all(optical_z[np.isfinite(optical_z)] > 0),
                        "invalid original surface intersections",
                    )
                    depth_checks[i] = render.check_depth(
                        grids["views"][i]["analysis_K"], cam["c2w"], optical_z
                    )
                    zp = tmp / (i + ".npz")
                    np.savez_compressed(zp, z=optical_z)
                    paths[i] = member(zp, "surface-z/" + zp.name)
                    z.write(zp, paths[i]["path"])
        common = {
            "scene_id": a.scene,
            "split": meta["split"],
            "run_id": os.environ["EL_RUN_ID"],
            "R": R,
            "H": H,
            "permission": permission,
            "original_count": len(ids),
            "image_decode_calls": len(ids),
            "decoded_original_bytes": sum(spec["bytes"] for spec in originals_manifest.values()),
            "image_decode_seconds": seconds,
            "first_access_role": role,
            "H_photo_decodes": 0 if source_mode else len(H),
            "photo_decode_events": accesses,
            "original_pixel_materialization": {
                "started_at_utc": first_pixels_start,
                "ended_at_utc": first_pixels_end,
                "operation": "individually timestamped authored image renders"
                if is_original
                else "one role-restricted archive extraction; timestamps bracket the entire extraction",
            },
            "file_bytes_read_for_verified_opaque_transport": 0 if is_original else a.archive.stat().st_size,
        }
        if source_mode:
            packs = {}
            options = [R[::8], R] if is_original else [R]
            for src_ids in options:
                name = f"R{len(src_ids)}"
                p = a.out / (name + ".zip")
                doc = subset_document(
                    a,
                    meta,
                    views,
                    src_ids,
                    H,
                    originals_manifest,
                    analysis,
                    {**permission, "R": src_ids, "A": src_ids},
                )
                blob = encoded(doc)
                with zipfile.ZipFile(p, "x", compression=zipfile.ZIP_STORED) as pack:
                    pack.writestr("scene.json", blob)
                    for i in src_ids:
                        pack.write(analysis_dir / (i + ".png"), analysis[i]["path"])
                packs[name] = member(p, p.name)
            result = {
                **common,
                "schema": "el-many-source-delivery-1",
                "packages": packs,
                "original_supplement": member(archive, archive.name),
                "originals": originals_manifest,
                "grids_profile": PROFILE,
                "H_data_members_sent": 0,
                "source_truth_depth_members_sent": 0,
            }
            output = "source.json"
        else:
            result = {
                **common,
                "schema": "el-many-evaluator-truth-1",
                "archive": member(archive, archive.name),
                "targets": {
                    i: {
                        "original_rgb": originals_manifest[i],
                        "analysis_rgb": analysis[i],
                        "target_z": paths.get(i),
                    }
                    for i in H
                },
                "z": paths,
                "geometry_truth_role": "opaque first-hit optical camera-Z, NaN for background"
                if is_original
                else "unavailable",
                "independent_world_ray_checks": depth_checks,
                "camera_frame": "same calibrated integer-centre K/c2w as source scene; depth on untouched 378x252 grid",
                "inputs": load(a.config)["sources"],
                "opening_sha256": sha(a.opening),
            }
            output = "truth.json"
        result["resource"] = {
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu,
            "max_process_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "child_cpu_seconds": sum(resource.getrusage(resource.RUSAGE_CHILDREN)[:2]),
            "worker_threads": 1,
            "payload_bytes_before_receipt": sum(p.stat().st_size for p in a.out.iterdir() if p.is_file()),
        }
        save(a.out / output, result)
