"""Versioned deployment-only identities for native arrays and sequence metadata."""

from collections import OrderedDict
import hashlib
import math

import numpy as np

from evidence_lens.multisource_io import cameras
from evidence_lens.scoring_io import Camera
from .bundle import ids
from .common import SEMANTICS, Unsupported, digest, encoded, exact_keys, json_bytes, require, sha, text

CAMERA = {
    "schema": "el-camera-1",
    "unit": "m",
    "pixel_center": "integer",
    "image_origin": "top_left",
    "axis_convention": "OpenCV x right, y down, z forward",
    "projection": "pinhole",
    "distortion": "none",
}
VERSIONS = ("map_revision", "source_revision", "renderer_revision", "checkpoint_sha256")
TASK = "display_srgb_mean_absolute_photometric_error_gt_0.05"
CAPABILITIES = {
    "prepared_cpu_risk": {"accepted": True, "used": True, "validated": "native 256x256 M2/M4"},
    "pose_covariance": {"accepted": True, "used": False, "validated": False},
    "dynamic_mask": {"accepted": True, "used": False, "validated": False},
    "source_selection_policy": {"accepted": True, "used": False, "validated": False},
    "gpu_risk": {"accepted": False, "used": False, "validated": False},
    "upstream_cuda_asset_render": {
        "accepted": True,
        "used": False,
        "validated": "separate gsplat asset command",
    },
}


def array_sha(a):
    """Digest dtype, shape, and native values in C order, including all NaN bits."""
    require(isinstance(a, np.ndarray) and not a.dtype.hasobject, "numeric ndarray required")
    h = hashlib.sha256(encoded({"dtype": a.dtype.str, "shape": list(a.shape)}))
    h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return h.hexdigest()


def camera_identity(cam, analysis_size=(256, 256)):
    require(
        isinstance(cam, Camera) and (cam.width, cam.height) == tuple(analysis_size),
        f"expected native {analysis_size[0]}x{analysis_size[1]} Camera",
        Unsupported,
    )
    require(
        all(isinstance(a, np.ndarray) and a.dtype == np.float64 for a in (cam.K, cam.c2w)),
        "K and c2w must be float64",
    )
    cameras(
        {**CAMERA, "views": {"view": {"height": cam.height, "width": cam.width, "K": cam.K, "c2w": cam.c2w}}},
        ["view"],
    )
    return {
        "width": cam.width,
        "height": cam.height,
        "K_sha256": array_sha(cam.K),
        "c2w_sha256": array_sha(cam.c2w),
    }


def render_identity(frame):
    if frame is None:
        return None
    exact_keys(frame, ("rgb", "z", "alpha"), "deployment RGB/Z/alpha")
    return {k: array_sha(frame[k]) for k in ("rgb", "z", "alpha")}


def source_identity(image, frame, camera, analysis_size=(256, 256)):
    return {
        "rgb_sha256": array_sha(image),
        "self_render_arrays": render_identity(frame),
        "camera": camera_identity(camera, analysis_size),
    }


def _verified_digest(bundle, source_id=None, target_id=None):
    """Cache only digests of authenticated file decodes, outside the scoring loop."""
    key = ("source", source_id) if source_id is not None else ("target", target_id)
    memo = getattr(bundle, "_deployment_array_digests", None)
    if memo is None:
        memo = bundle._deployment_array_digests = OrderedDict()
    if key not in memo:
        if source_id is not None:
            memo[key] = sha(
                encoded(source_identity(*bundle.source(source_id), analysis_size=bundle.analysis_size))
            )
        else:
            memo[key] = sha(encoded(render_identity(bundle.frame(target_id))))
        # At most 256 SHA-256 strings plus view IDs per Bundle. No prediction array is held.
        if len(memo) > 256:
            memo.popitem(last=False)
    memo.move_to_end(key)
    return memo[key]


def binding_from_bundle(bundle, *, map_revision=None, source_revision=None):
    """Adapt the verified bundle manifest without opening target/reference photographs."""
    p, b = bundle.producer, bundle.meta
    source_files = {
        s: {
            "rgb_file_sha256": b["sources"][s]["sha256"],
            "self_render": p["frames"][s],
            "decoded_source_sha256": _verified_digest(bundle, source_id=s),
        }
        for s in bundle.order
    }
    # Hash only source-side producer data. Target-list changes alone do not invalidate source evidence.
    source_version = sha(
        encoded(
            {
                "files": source_files,
                "ordered_ids": list(bundle.order),
                "cameras": {
                    s: camera_identity(bundle.cameras[s], bundle.analysis_size) for s in bundle.order
                },
                **(
                    {
                        "source_grids": {s: bundle.grids["views"][s] for s in bundle.order},
                        "analysis_profile": bundle.analysis_profile,
                    }
                    if bundle.versioned_grid
                    else {}
                ),
                "primary": bundle.primary,
                "endpoints": list(bundle.endpoints),
            }
        )
    )
    return {
        "schema": "el-resident-binding-2" if bundle.versioned_grid else "el-resident-binding-1",
        **(
            {"grids": json_bytes(encoded(bundle.grids)), "grids_sha256": bundle.producer["grids_sha256"]}
            if bundle.versioned_grid
            else {}
        ),
        "map_revision": map_revision or p["generator"]["run_id"],
        "source_revision": source_revision or source_version,
        "checkpoint_sha256": p["checkpoint_sha256"],
        "generator_receipt_sha256": p["generator"]["receipt_sha256"],
        "renderer_revision": p["renderer"]["run_id"],
        "renderer_receipt_sha256": p["renderer"]["receipt_sha256"],
        "bundle_config_sha256": b["creation"]["config_sha256"],
        "renderer_config_sha256": None,
        "source_files": source_files,
        "source_order": list(bundle.order),
        "condition": bundle.condition,
        "primary_source_id": bundle.primary,
        "endpoint_source_ids": list(bundle.endpoints),
        "geometry": dict(b["geometry"]),
        "camera_convention": dict(CAMERA),
        "semantics": dict(p["semantics"]),
        "depth_verified": p["depth_verified"],
        "depth_probe_sha256": p["depth_probe_sha256"],
        "input_domain": b["input_domain"],
        "source_selection_policy": "authorized prepared-bundle source order",
        "task_label_schema": TASK,
    }


def check_grid_cameras(grids, views):
    """Recheck each observed camera against that view's raw-to-analysis contract."""
    from .grids import validate_grids

    require(
        isinstance(grids, dict)
        and isinstance(grids.get("views"), dict)
        and set(views) <= grids["views"].keys(),
        "missing original-to-analysis grid for an observed view",
    )
    require(all(isinstance(cam, Camera) for cam in views.values()), "expected analysis cameras")
    selected = {**grids, "views": {name: grids["views"][name] for name in views}}
    camera_doc = {
        "views": {name: {"width": cam.width, "height": cam.height, "K": cam.K} for name, cam in views.items()}
    }
    return validate_grids(selected, camera_doc, tuple(views))


def check_binding(b, source_ids):
    required = {
        "schema",
        *VERSIONS,
        "generator_receipt_sha256",
        "renderer_receipt_sha256",
        "renderer_config_sha256",
        "bundle_config_sha256",
        "source_files",
        "source_order",
        "condition",
        "primary_source_id",
        "endpoint_source_ids",
        "geometry",
        "camera_convention",
        "semantics",
        "depth_verified",
        "depth_probe_sha256",
        "input_domain",
        "source_selection_policy",
        "task_label_schema",
    }
    versioned = isinstance(b, dict) and b.get("schema") == "el-resident-binding-2"
    if versioned:
        required.update(("grids", "grids_sha256"))
    exact_keys(b, required, "version binding")
    require(
        b["schema"] in ("el-resident-binding-1", "el-resident-binding-2")
        and b["semantics"] == SEMANTICS
        and b["camera_convention"] == CAMERA
        and b["task_label_schema"] == TASK,
        "unsupported camera, channel or risk task semantics",
        Unsupported,
    )
    geo = b["geometry"]
    exact_keys(
        geo,
        (
            "prepared_camera_and_z_unit",
            "meters_per_original_unit",
            "metric_evidence",
            "crop",
            "resize",
            "undistortion",
            "scoring_resampling",
        ),
        "geometry",
    )
    scale = geo["meters_per_original_unit"]
    require(
        geo["prepared_camera_and_z_unit"] == "m"
        and type(scale) in (float, int)
        and math.isfinite(scale)
        and scale > 0
        and geo["scoring_resampling"] == "none",
        "native grid and positive evidenced metric scale required",
        Unsupported,
    )
    for k in ("metric_evidence", "crop", "resize", "undistortion"):
        text(geo[k], k)
    order = ids(b["source_order"], "source ID")
    require(order == tuple(source_ids), "input insertion order differs from authorized source order")
    if versioned:
        from .grids import validate_grid_record

        require(sha(encoded(b["grids"])) == b["grids_sha256"], "source analysis grid digest differs")
        require(
            isinstance(b["grids"], dict) and isinstance(b["grids"].get("views"), dict),
            "missing source analysis grids",
        )
        validate_grid_record(b["grids"], tuple(b["grids"]["views"]))
        check_grid_cameras(b["grids"], {name: source_ids[name][2] for name in order})
    require(
        b["condition"] in ("M2", "M4") and len(order) == int(b["condition"][1]),
        "only matching M2/M4 source sets are supported",
        Unsupported,
    )
    endpoints = ids(b["endpoint_source_ids"], "common endpoint")
    require(
        len(endpoints) == 2 and set(endpoints) <= set(order) and b["primary_source_id"] in order,
        "source endpoint or primary ID is not authorized",
    )
    for k in (*VERSIONS[:3], "input_domain", "source_selection_policy"):
        text(b[k], k)
    for k in ("checkpoint_sha256", "generator_receipt_sha256", "renderer_receipt_sha256"):
        digest(b[k], k)
    for k in ("renderer_config_sha256", "bundle_config_sha256"):
        if b[k] is not None:
            digest(b[k], k)
    require(type(b["depth_verified"]) is bool, "boolean depth validation required")
    if b["depth_verified"] or b["depth_probe_sha256"] is not None:
        digest(b["depth_probe_sha256"], "depth probe")
    if b["source_files"] is not None:
        from .bundle import disposition

        exact_keys(b["source_files"], order, "source file identities")
        for item in b["source_files"].values():
            exact_keys(
                item, ("rgb_file_sha256", "self_render", "decoded_source_sha256"), "source disposition"
            )
            digest(item["rgb_file_sha256"], "source photo")
            digest(item["decoded_source_sha256"], "decoded authorized source")
            disposition(item["self_render"])
            if item["self_render"]["status"] == "ready":
                f = item["self_render"]["file"]
                exact_keys(f, ("path", "sha256", "bytes"), "source render file")
                digest(f["sha256"], "source self-render")
                require(type(f["bytes"]) is int and f["bytes"] > 0, "invalid source render length")
                text(f["path"], "source render identifier")


def frame_meta_from_bundle(
    bundle, view, *, sequence_id=None, frame_id=None, map_revision=None, source_revision=None, binding=None
):
    """Read target identity from a prepared manifest. May be computed once at input admission."""
    require(view in bundle.targets, "target camera must be a declared deployment target")
    b = (
        binding_from_bundle(bundle, map_revision=map_revision, source_revision=source_revision)
        if binding is None
        else binding
    )
    require(
        b["schema"] == ("el-resident-binding-2" if bundle.versioned_grid else "el-resident-binding-1")
        and (
            not bundle.versioned_grid
            or b.get("grids_sha256") == bundle.producer["grids_sha256"]
            and b.get("grids") == bundle.grids
        )
        and b["checkpoint_sha256"] == bundle.producer["checkpoint_sha256"]
        and b["renderer_revision"] == bundle.producer["renderer"]["run_id"]
        and b["renderer_receipt_sha256"] == bundle.producer["renderer"]["receipt_sha256"]
        and b["source_order"] == list(bundle.order)
        and b["primary_source_id"] == bundle.primary
        and b["endpoint_source_ids"] == list(bundle.endpoints)
        and b["source_files"] is not None
        and all(
            b["source_files"][s]["rgb_file_sha256"] == bundle.meta["sources"][s]["sha256"]
            and b["source_files"][s]["self_render"] == bundle.producer["frames"][s]
            for s in bundle.order
        ),
        "frame binding differs from verified producer",
    )
    item = bundle.producer["frames"][view]
    return {
        **{k: b[k] for k in VERSIONS},
        "source_binding_sha256": sha(encoded(b)),
        "sequence_id": sequence_id or bundle.sha256,
        "frame_id": view if frame_id is None else frame_id,
        "target_view_id": view,
        "bundle_sha256": bundle.sha256,
        "target_camera": camera_identity(bundle.cameras[view], bundle.analysis_size),
        **(
            {"target_grid": json_bytes(encoded(bundle.grids["views"][view]))} if bundle.versioned_grid else {}
        ),
        "render_status": item["status"],
        "render_failure": item.get("reason"),
        "target_file_sha256": item.get("file", {}).get("sha256"),
        "target_array_sha256": _verified_digest(bundle, target_id=view),
    }


def check_frame_meta(meta, binding_versions, cam, frame, *, versioned_grid=False):
    required = {*VERSIONS, "frame_id", "sequence_id"}
    optional = {
        "bundle_sha256",
        "target_view_id",
        "target_camera",
        "target_file_sha256",
        "target_array_sha256",
        "source_binding_sha256",
        "render_status",
        "render_failure",
        "clock_domain",
        "capture_time",
        "pose_covariance",
        "dynamic_mask",
    }
    if versioned_grid:
        required.update(("target_grid", "target_view_id", "source_binding_sha256"))
    require(
        isinstance(meta, dict) and required <= meta.keys() <= required | optional,
        "unexpected or missing deployment frame metadata",
    )
    if versioned_grid:
        digest(meta["source_binding_sha256"], "source binding")
        text(meta["target_view_id"], "target view")
        require(isinstance(meta["target_grid"], dict), "missing original-to-analysis target grid")
    require(
        tuple(meta[k] for k in VERSIONS) == binding_versions, "frame version differs from source snapshot"
    )
    text(meta["sequence_id"], "sequence ID")
    f = meta["frame_id"]
    require(
        (type(f) is int and f >= 0) or (isinstance(f, str) and f.strip() and f.isprintable()),
        "invalid frame ID",
    )
    for k in ("bundle_sha256", "target_file_sha256", "target_array_sha256", "source_binding_sha256"):
        if meta.get(k) is not None:
            digest(meta[k], k)
    require(
        meta.get("target_file_sha256") is None or meta.get("target_array_sha256") is not None,
        "a file-backed prediction requires its verified decoded-array identity",
    )
    require(
        meta.get("bundle_sha256") is None or meta.get("source_binding_sha256") is not None,
        "a file-backed frame requires its verified source binding",
    )
    if "target_camera" in meta:
        require(meta["target_camera"] == cam, "target camera differs from declared identity")
    if "render_status" in meta:
        require(
            meta["render_status"] == ("render_failed" if frame is None else "ready"),
            "target render disposition differs from supplied arrays",
        )
    if meta.get("target_view_id") is not None:
        text(meta["target_view_id"], "target view")
    if meta.get("render_failure") is not None:
        require(frame is None, "successful frame has a render failure")
        text(meta["render_failure"], "render failure")
    if meta.get("capture_time") is not None:
        require(
            type(meta["capture_time"]) in (float, int) and math.isfinite(meta["capture_time"]),
            "invalid capture time",
        )
        text(meta.get("clock_domain"), "capture clock domain")
    if meta.get("clock_domain") is not None:
        text(meta["clock_domain"], "clock domain")
    if meta.get("pose_covariance") is not None:
        pc = meta["pose_covariance"]
        exact_keys(pc, ("matrix", "coordinates", "units"), "pose covariance")
        vals = np.asarray(pc["matrix"], dtype=np.float64)
        require(vals.shape == (6, 6) and np.isfinite(vals).all(), "expected finite 6x6 pose covariance")
        for k in ("coordinates", "units"):
            text(pc[k], k)
    if meta.get("dynamic_mask") is not None:
        mask = meta["dynamic_mask"]
        exact_keys(mask, ("reference", "source", "time", "clock_domain"), "dynamic mask metadata")
        for k in ("source", "reference"):
            text(mask[k], k)
        if mask["time"] is not None:
            require(type(mask["time"]) in (int, float) and math.isfinite(mask["time"]), "invalid mask time")
            text(mask["clock_domain"], "mask clock domain")
