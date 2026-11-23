"""Private evaluation-side reference reads and actual-error labels."""

import io
import json
from pathlib import Path

import numpy as np

from .scoring_io import TARGETS, Camera, ReadAudit, alpha_valid, check_rgb, depth_valid, rgb_valid


VISIBILITY = {0: "background_or_invalid", 1: "observed", 2: "disoccluded", 3: "outside_input_fov"}
RGB_THRESHOLDS = (0.025, 0.05, 0.10)
LABEL_VERSION = "el-labels-1"


class ReferenceBundle:
    def __init__(self, directory: Path, audit: ReadAudit):
        self.directory = directory
        self.audit = audit
        self.meta = json.loads(audit.read("reference_metadata", directory / "reference.json"))
        for key, expected in (("schema", "el-reference-1"), ("camera_sha256", audit.sha("cameras")),
                              ("input_sha256", audit.sha("input_png"))):
            if self.meta.get(key) != expected:
                raise ValueError(f"reference.json: incompatible {key}")
        validation = json.loads(audit.read("reference_validation", directory / "validation" / "results.json"))
        if (self.meta.get("validation_passed") is not True or validation.get("passed") is not True
                or self.meta.get("validation_sha256") != audit.sha("reference_validation")):
            raise ValueError("reference validation failed or its digest does not match reference.json")

    def read(self, view: str, camera: Camera):
        if view not in TARGETS:
            raise ValueError(f"invalid target id: {view}")
        role = f"reference/{view}"
        data = self.audit.read(role, self.directory / "reference" / f"{view}.npz")
        if self.audit.sha(role) != self.meta["views"][view]["reference_sha256"]:
            raise ValueError(f"{role}: reference hash mismatch")
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            result = {k: archive[k] for k in ("rgb", "z", "position", "object_id", "visibility", "edge")}
        check_rgb(result["rgb"], camera, role)
        for k in ("object_id", "visibility", "edge", "z"):
            if result[k].shape != camera.shape:
                raise ValueError(f"{role}/{k}: dimensions disagree with camera")
        pos, z = result["position"], result["z"]
        if z.dtype != np.float32 or pos.dtype != np.float32 or pos.shape != (*camera.shape, 3):
            raise ValueError(f"{role}: expected float32 camera-Z and world Position")
        if result["object_id"].dtype.kind not in "iu" or result["visibility"].dtype != np.uint8:
            raise ValueError(f"{role}: invalid object_id or visibility dtype")
        if np.any(result["visibility"] > max(VISIBILITY)):
            raise ValueError(f"{role}: unknown visibility code")
        edge = result["edge"]
        if edge.dtype not in (np.bool_, np.uint8) or np.any((edge != 0) & (edge != 1)):
            raise ValueError(f"{role}: edge mask must be boolean or binary uint8")
        result["edge"] = edge.astype(bool, copy=False)
        surface = depth_valid(z)
        if not np.array_equal(surface, np.isfinite(pos).all(axis=-1)):
            raise ValueError(f"{role}: Position availability disagrees with positive camera-Z")
        if np.any((result["visibility"] != 0) & ~surface):
            raise ValueError(f"{role}: non-surface pixel has a geometric visibility class")
        if surface.any():
            world_to_camera = np.linalg.inv(camera.c2w)
            points = pos[surface] @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
            if not np.allclose(points[:, 2], z[surface], atol=1e-4, rtol=1e-4):
                raise ValueError(f"{role}: camera-Z disagrees with world Position and camera")
            yy, xx = np.nonzero(surface)
            projections = points @ camera.K.T
            uv = projections[:, :2] / projections[:, 2, None]
            if np.max(np.hypot(uv[:, 0] - xx, uv[:, 1] - yy)) > 1.0 + 1e-5:
                raise ValueError(f"{role}: Position-to-pixel projection exceeds 1 pixel")
        return result


def label_view(ref, pred, verified):
    v = rgb_valid(ref["rgb"])
    shape = v.shape
    surface = depth_valid(ref["z"]) & np.isfinite(ref["position"]).all(axis=-1)
    pred_rgb_ok = rgb_valid(pred["rgb"]) if pred is not None else np.zeros(shape, bool)
    z_ok = depth_valid(pred["z"]) & verified if pred is not None else np.zeros(shape, bool)
    a_ok = alpha_valid(pred["alpha"]) if pred is not None else np.zeros(shape, bool)
    low_alpha = a_ok & (pred["alpha"] < 0.5) if pred is not None else np.zeros(shape, bool)
    evaluable = v & pred_rgb_ok
    z_eval = surface & z_ok & a_ok & ~low_alpha
    rgb_error = np.full(shape, np.nan, np.float32)
    signed_z = np.full_like(rgb_error, np.nan)
    signed_rel_z = np.full_like(rgb_error, np.nan)
    if pred is not None:
        rgb_error[evaluable] = np.abs(pred["rgb"][evaluable] - ref["rgb"][evaluable]).mean(axis=-1)
        signed_z[z_eval] = pred["z"][z_eval] - ref["z"][z_eval]
        signed_rel_z[z_eval] = signed_z[z_eval] / ref["z"][z_eval]
    y = np.full(shape, 255, np.uint8)
    y[evaluable] = rgb_error[evaluable] > np.float32(0.05)
    z_high = np.full(shape, 255, np.uint8)
    z_high[z_eval] = np.abs(signed_z[z_eval]) > np.maximum(np.float32(0.02), np.float32(0.02) * ref["z"][z_eval])
    return {
        "reference_valid": v, "reference_surface": surface, "rgb_evaluable": evaluable,
        "prediction_rgb_valid": pred_rgb_ok, "invalid_rgb": ~pred_rgb_ok, "rgb_error": rgb_error,
        "rgb_high_error": y, "invalid_depth": ~z_ok, "depth_evaluable": z_eval,
        "depth_error": np.abs(signed_z), "signed_depth_error": signed_z,
        "relative_depth_error": np.abs(signed_rel_z), "signed_relative_depth_error": signed_rel_z,
        "depth_high_error": z_high, "alpha_valid": a_ok, "low_alpha": low_alpha,
        "low_support": surface & low_alpha, "visibility": ref["visibility"], "edge": ref["edge"],
    }
