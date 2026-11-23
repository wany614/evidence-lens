"""File contracts shared by CPU scoring and evaluation. No reference lookups here."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


TARGETS = tuple(f"v{i:02d}" for i in range(1, 7))
VIEWS = ("source", *TARGETS)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class ReadAudit:
    """Hash exactly the bytes decoded, with stable role names instead of machine paths."""

    def __init__(self):
        self.files = {}

    def read(self, role: str, path: Path) -> bytes:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{role} must be a regular file: {path}")
        data = path.read_bytes()
        record = {"sha256": digest(data), "bytes": len(data)}
        if role in self.files and self.files[role] != record:
            raise ValueError(f"{role} changed during the run")
        self.files[role] = record
        return data

    def sha(self, role: str) -> str:
        return self.files[role]["sha256"]


@contextmanager
def new_output(directory: Path):
    """Commit a complete new derived product; existing results are immutable."""
    directory = Path(directory)
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"output already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        yield temporary
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    K: np.ndarray
    c2w: np.ndarray

    @property
    def shape(self):
        return self.height, self.width


def load_cameras(path: Path, audit: ReadAudit) -> dict[str, Camera]:
    bundle = json.loads(audit.read("cameras", path))
    if (bundle.get("schema"), bundle.get("unit"), bundle.get("pixel_center")) != (
        "el-camera-1", "m", "integer"
    ):
        raise ValueError("expected el-camera-1, metres, integer pixel centres")
    if set(bundle["views"]) != set(VIEWS):
        raise ValueError("camera bundle must contain source and v01 through v06")
    cameras = {}
    for view in VIEWS:
        spec = bundle["views"][view]
        width, height = spec["width"], spec["height"]
        if any(type(n) is not int or n <= 0 for n in (width, height)):
            raise ValueError(f"{view}: invalid camera dimensions")
        K = np.asarray(spec["K"], dtype=np.float64)
        pose = np.asarray(spec["c2w"], dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all() or not np.allclose(K[2], [0, 0, 1]):
            raise ValueError(f"{view}: invalid pinhole intrinsics")
        if min(K[0, 0], K[1, 1]) <= 0 or not np.allclose([K[0, 1], K[1, 0]], 0):
            raise ValueError(f"{view}: positive focal lengths and zero skew required")
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError(f"{view}: invalid homogeneous c2w")
        R = pose[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
            raise ValueError(f"{view}: c2w must be a proper rigid transform")
        cameras[view] = Camera(width, height, K, pose)
    return cameras


def load_source_png(path: Path, camera: Camera, audit: ReadAudit) -> np.ndarray:
    data = audit.read("input_png", path)
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) < 26 or data[24:26] not in (b"\x08\x02", b"\x08\x06"):
        raise ValueError("source must be an 8-bit RGB or opaque RGBA PNG")
    with Image.open(io.BytesIO(data)) as image:
        if image.size != (camera.width, camera.height):
            raise ValueError("input PNG dimensions disagree with the source camera")
        pixels = np.asarray(image)
    if pixels.shape[-1] == 4 and np.any(pixels[..., 3] != 255):
        raise ValueError("source PNG has non-opaque pixels")
    return pixels[..., :3].astype(np.float32) / np.float32(255)


def check_rgb(array: np.ndarray, camera: Camera, role: str):
    if array.shape != (*camera.shape, 3) or array.dtype != np.float32:
        raise ValueError(f"{role}: expected float32 RGB of shape {(*camera.shape, 3)}")
    if np.any(np.isfinite(array) & ((array < 0) | (array > 1))):
        raise ValueError(f"{role}: finite display-sRGB channel outside [0,1]")


def rgb_valid(array):
    return np.isfinite(array).all(axis=-1)


def alpha_valid(array):
    return np.isfinite(array) & (array >= 0) & (array <= 1)


def depth_valid(array):
    return np.isfinite(array) & (array > 0)


class PredictionBundle:
    def __init__(self, directory: Path, audit: ReadAudit):
        self.directory = directory
        self.audit = audit
        self.meta = json.loads(audit.read("prediction_metadata", directory / "prediction.json"))
        expected = {
            "schema": "el-prediction-1",
            "rgb_encoding": "display_srgb_0_1",
            "depth_semantics": "expected_camera_z_m",
            "alpha_semantics": "accumulated_opacity",
            "input_sha256": audit.sha("input_png"),
            "camera_sha256": audit.sha("cameras"),
        }
        for key, value in expected.items():
            if self.meta.get(key) != value:
                raise ValueError(f"prediction.json: incompatible {key}")
        if not isinstance(self.meta.get("producer_run_id"), str) or not self.meta["producer_run_id"].strip():
            raise ValueError("prediction.json requires producer_run_id")
        validation = self.meta.get("depth_validation")
        checks = ("off_axis_plane_passed", "alpha_normalization_passed")
        if not isinstance(validation, dict) or any(type(validation.get(key)) is not bool for key in checks):
            raise ValueError(f"prediction.json requires boolean depth checks {checks}")
        self.depth_verified = all(validation[key] for key in checks)
        self.failed = self.meta.get("failed_views", {})
        if not isinstance(self.failed, dict) or not set(self.failed).issubset(VIEWS) or any(
            not isinstance(reason, str) or not reason.strip() for reason in self.failed.values()
        ):
            raise ValueError("invalid failed_views mapping in prediction.json")

    def read(self, view: str, camera: Camera) -> dict[str, np.ndarray] | None:
        if view not in VIEWS:
            raise ValueError(f"invalid camera id: {view}")
        path = self.directory / f"{view}.npz"
        if view in self.failed:
            if path.exists() or path.is_symlink():
                raise ValueError(f"{view}: declared failed but prediction exists")
            return None
        role = f"prediction/{view}"
        data = self.audit.read(role, path)
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if len(archive.files) != 3 or set(archive.files) != {"rgb", "z", "alpha"}:
                raise ValueError(f"{role}: expected exactly rgb, z, alpha")
            result = {key: archive[key] for key in ("rgb", "z", "alpha")}
        check_rgb(result["rgb"], camera, role)
        for key in ("z", "alpha"):
            if result[key].shape != camera.shape or result[key].dtype != np.float32:
                raise ValueError(f"{role}/{key}: expected float32 of shape {camera.shape}")
        if np.any((result["alpha"] == 0) & depth_valid(result["z"])):
            raise ValueError(f"{role}: zero-alpha pixels cannot have a valid expected depth")
        return result
