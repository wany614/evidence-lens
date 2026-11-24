"""Inspect external calibrated images and a Gaussian asset on CPU, with no target RGB reads."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import numpy as np
from PIL import Image
from evidence_lens.multisource_io import cameras as check_cameras
from evidence_lens_bundle.common import (
    Unsupported,
    exact_keys,
    json_bytes,
    parts,
    read_file,
    require,
    sha,
    text,
)

from . import colmap, gaussian


def checksum(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def metrology(document):
    evidence = document.get("metric")
    require(
        isinstance(evidence, dict), "missing metric scale and its source", Unsupported
    )
    scale = evidence.get("meters_per_unit")
    require(
        type(scale) in (int, float) and np.isfinite(scale) and scale > 0,
        "positive finite meters_per_unit required",
        Unsupported,
    )
    text(evidence.get("source"), "metric evidence description")
    require(
        evidence.get("kind")
        in ("explicit_metric_declaration", "surveyed_source_baseline"),
        "unsupported scale evidence",
        Unsupported,
    )
    return float(scale), evidence


def source_png(root, name, camera):
    parts(name)
    original = read_file(root, name)
    with Image.open(io.BytesIO(original)) as im:
        require(
            im.format in ("PNG", "JPEG")
            and im.mode == "RGB"
            and im.size == (camera["width"], camera["height"]),
            "source must be a native-size RGB8 PNG/JPEG",
            Unsupported,
        )
        require(
            im.getexif().get(274, 1) == 1 and not im.info.get("icc_profile"),
            "source requires evidenced orientation or colour conversion",
            Unsupported,
        )
        pixels = np.asarray(im, dtype=np.uint8)
    f = io.BytesIO()
    Image.fromarray(pixels).save(f, format="PNG")
    return f.getvalue(), {"name": name, "sha256": sha(original), "bytes": len(original)}


def inspect(
    model_dir, images_dir, gaussian_path, plan_path, encoding, gaussian_profile
):
    plan = json_bytes(Path(plan_path).read_bytes())
    version = plan.get("schema")
    require(
        version in ("el-asset-view-plan-1", "el-asset-view-plan-2"),
        "unsupported asset view plan",
        Unsupported,
    )
    text(plan.get("input_domain"), "input domain")
    if version == "el-asset-view-plan-2" and "source_authorization" in plan:
        from evidence_lens_bundle.common import encoded
        from .permissions import source_grant

        grant = plan["source_authorization"]
        contract = grant["registration"]
        require(
            sha(encoded(contract)) == grant["registration_sha256"],
            "source registration hash differs",
        )
        choice = grant.get("choice") if grant["branch"] == "optional-M4" else None
        if choice is not None:
            require(
                sha(encoded(choice)) == grant.get("choice_sha256"),
                "optional choice or failure hash differs",
            )
        allowed = source_grant(contract, grant["branch"], choice)
        require(
            allowed == [v["id"] for v in plan["sources"]],
            "input image order exceeds its branch-specific grant",
        )
    scale, evidence = metrology(plan)
    cams, imgs = colmap.load(model_dir, encoding)
    recovery = {}
    arr = gaussian.load(gaussian_path, gaussian_profile, scale, metadata=recovery)
    if gaussian_profile == gaussian.RECOVERY:
        clip = plan.get("recovery_clip_m")
        require(
            isinstance(clip, list)
            and len(clip) == 2
            and all(type(v) in (float, int) and np.isfinite(v) for v in clip)
            and 0 < clip[0] < clip[1],
            "recovery render requires the original camera clipping planes in recovery_clip_m",
            Unsupported,
        )
        recovery["clip_m"] = clip
    require(
        isinstance(plan.get("sources"), list)
        and len(plan["sources"]) in (2, 4)
        and isinstance(plan.get("targets"), list)
        and plan["targets"],
        "original profile requires two/four source images and at least one target",
        Unsupported,
    )
    photos = {}
    photo_proofs = {}
    derived = {}
    for role in ("sources", "targets"):
        for entry in plan[role]:
            require(isinstance(entry, dict), "invalid camera selection")
            name = text(entry.get("id"), "view ID")
            require(name not in derived, "duplicate source or target ID " + name)
            if "image_id" in entry:
                exact_keys(entry, ("id", "image_id"), "selected COLMAP image")
                i = entry["image_id"]
                require(
                    type(i) is int and str(i) == name and i in imgs,
                    "preserve the declared COLMAP image ID " + name,
                )
                camera_id, q, t, filename = imgs[i]
                cam = dict(cams[camera_id], c2w=colmap.pose(q, t, scale))
                if role == "sources":
                    photos[name], photo_proofs[name] = source_png(
                        images_dir, filename, cam
                    )
            else:
                require(
                    role == "targets",
                    "source must reference a calibrated original image",
                )
                exact_keys(entry, ("id", "camera_m_integer_center"), "novel target")
                cam = entry["camera_m_integer_center"]
                require(
                    isinstance(cam, dict)
                    and cam.get("projection") == "pinhole"
                    and cam.get("distortion") == "none",
                    "novel camera must declare an undistorted metric pinhole",
                    Unsupported,
                )
            if version == "el-asset-view-plan-1":
                require(
                    cam["width"] == cam["height"] == 256,
                    "legacy plan requires native 256 x 256 cameras",
                    Unsupported,
                )
            derived[name] = cam
    source_order = list(photos)
    targets = [i["id"] for i in plan["targets"]]
    require(
        plan.get("primary_source_id") in photos
        and isinstance(plan.get("endpoint_source_ids"), list)
        and len(plan["endpoint_source_ids"]) == 2
        and len(set(plan["endpoint_source_ids"])) == 2
        and set(plan["endpoint_source_ids"]) <= set(photos),
        "invalid primary or endpoint source selection",
    )
    if evidence["kind"] == "surveyed_source_baseline":
        a, b = evidence.get("source_ids", [None, None])
        require(
            a in photos and b in photos and a != b,
            "survey must name two different source cameras",
            Unsupported,
        )
        distance = evidence.get("distance_m")
        tolerance = evidence.get("tolerance_m")
        require(
            type(distance) in (int, float)
            and type(tolerance) in (int, float)
            and np.isfinite([distance, tolerance]).all()
            and distance > 0
            and 0 <= tolerance < distance,
            "invalid metric baseline",
            Unsupported,
        )
        measured = np.linalg.norm(
            np.asarray(derived[a]["c2w"])[:3, 3] - np.asarray(derived[b]["c2w"])[:3, 3]
        )
        require(
            abs(measured - distance) <= tolerance,
            "metric scale disagrees with declared source baseline",
            Unsupported,
        )
    doc = {
        "schema": "el-camera-1",
        "unit": "m",
        "pixel_center": "integer",
        "image_origin": "top_left",
        "axis_convention": "OpenCV x right, y down, z forward",
        "views": derived,
    }
    check_cameras(doc, [*photos, *targets])
    grids = None
    if version == "el-asset-view-plan-2":
        from evidence_lens_bundle.grids import build_grids, resample_photo

        grids = build_grids(
            plan.get("analysis_profile"), derived, display_size=plan.get("display_size")
        )
        if "grids" in plan:
            require(
                plan["grids"] == grids,
                "view-plan grid differs from the calibrated originals",
            )
        size = grids["analysis_size"]
        for name, view_grid in grids["views"].items():
            old = derived[name]
            # A novel COLMAP view can carry legacy redundant calibrations. Recompute them on its new grid.
            current = {
                **old,
                "K": view_grid["analysis_K"],
                "width": size[0],
                "height": size[1],
            }
            if "K_edge" in current:
                current["K_edge"] = (
                    np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
                    @ np.asarray(current["K"])
                ).tolist()
            derived[name] = current
            if name in photos:
                with Image.open(io.BytesIO(photos[name])) as image:
                    sampled = resample_photo(image, view_grid, size)
                out = io.BytesIO()
                Image.fromarray(np.asarray(sampled, dtype=np.uint8)).save(
                    out, format="PNG"
                )
                photos[name] = out.getvalue()
                photo_proofs[name]["analysis_png_sha256"] = sha(photos[name])
        check_cameras(doc, [*photos, *targets])
    suffix = "txt" if encoding == "text" else "bin"
    stats = {
        "schema": "el-asset-inspection-1",
        "colmap_version": "3.10",
        "colmap_encoding": encoding,
        "input_domain": plan["input_domain"],
        "source_order": source_order,
        "target_order": targets,
        "primary_source_id": plan["primary_source_id"],
        "endpoint_source_ids": plan["endpoint_source_ids"],
        "condition": "M" + str(len(photos)),
        "metric": evidence,
        "images": photo_proofs,
        "colmap_files": {
            n: checksum(Path(model_dir) / f"{n}.{suffix}")
            for n in ("images", "cameras")
        },
        "plan_sha256": checksum(plan_path),
        "asset_sha256": checksum(gaussian_path),
        "producer_role": "Imported reconstruction asset; upstream training run unspecified",
        "gaussian_profile": gaussian_profile,
        "gaussians": len(arr["means"]),
        "sh_degree": (3 if arr["sh"].shape[1] == 16 else 0) if "sh" in arr else None,
        "renderer": {
            "implementation": "gsplat",
            "version": "1.5.3",
            "device": "cuda",
            "required_for": "rendering the reconstruction asset",
        },
        "scorer": {"device": "cpu", "default": "MS-L1"},
    }
    if grids is not None:
        stats.update(
            schema="el-asset-inspection-2",
            grids=grids,
            source_resampling="calibrated integer crop and display-sRGB8 Pillow LANCZOS v1",
        )
    if recovery:
        stats["recovery_render"] = recovery
    return stats, doc, photos, arr
