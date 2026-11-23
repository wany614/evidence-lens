"""Verified deployment file reader for a single fixed reconstruction. No label or reference paths."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .multisource import CONDITIONS, SOURCES, TARGETS, extract, source_ids, validate_render
from .scoring_io import Camera, digest

HASH = re.compile(r"[a-f0-9]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_member(root, entry, expected_path):
    """Fixed role paths and hashes are checked before any decode."""
    require(entry.get("path") == expected_path, "unexpected asset role or path: " + expected_path)
    path = Path(root) / expected_path
    require(
        path.resolve().is_relative_to(Path(root).resolve()) and not path.is_symlink() and path.is_file(),
        "invalid deployment file: " + expected_path,
    )
    data = path.read_bytes()
    require(
        type(entry.get("bytes")) is int
        and len(data) == entry["bytes"]
        and digest(data) == entry.get("sha256"),
        "file identity differs: " + expected_path,
    )
    return data


def cameras(bundle, expected):
    require(
        (
            bundle.get("schema"),
            bundle.get("unit"),
            bundle.get("pixel_center"),
            bundle.get("image_origin", "top_left"),
            bundle.get("axis_convention", "OpenCV x right, y down, z forward"),
        )
        == ("el-camera-1", "m", "integer", "top_left", "OpenCV x right, y down, z forward")
        and set(bundle.get("views", {})) == set(expected),
        "incompatible calibrated camera set",
    )
    result = {}
    for name in expected:
        a = bundle["views"][name]
        w, h = a["width"], a["height"]
        require(all(type(n) is int and n > 0 for n in (w, h)), "invalid camera dimensions")
        k, p = np.asarray(a["K"], np.float64), np.asarray(a["c2w"], np.float64)
        require(
            k.shape == (3, 3)
            and np.isfinite(k).all()
            and min(k[0, 0], k[1, 1]) > 0
            and np.allclose(k[2], [0, 0, 1])
            and np.allclose([k[0, 1], k[1, 0]], 0),
            "invalid camera intrinsics",
        )
        require(
            p.shape == (4, 4)
            and np.isfinite(p).all()
            and np.allclose(p[3], [0, 0, 0, 1])
            and np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(p[:3, :3]), 1, atol=1e-5),
            "invalid camera pose",
        )
        if "w2c" in a:
            require(np.allclose(a["w2c"], np.linalg.inv(p), atol=1e-5), "w2c conflicts with calibrated pose")
        if "K_edge" in a:
            require(
                np.allclose(a["K_edge"], np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]) @ k, atol=1e-5),
                "pixel-edge calibration conflicts with integer pixel centres",
            )
        result[name] = Camera(w, h, k, p)
    return result


def decode_prediction(data, camera):
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        require(len(z.files) == 3 and set(z.files) == {"rgb", "z", "alpha"}, "unexpected predicted array")
        frame = {k: z[k] for k in z.files}
    validate_render(frame, camera)
    return frame


def decode_png(data, camera):
    require(data[:8] == b"\x89PNG\r\n\x1a\n" and data[24:26] == b"\x08\x02", "expected RGB8 source PNG")
    with Image.open(io.BytesIO(data)) as im:
        require(
            im.size == (camera.width, camera.height) and im.mode == "RGB",
            "source image/camera shape mismatch",
        )
        return np.asarray(im, dtype=np.float32) / np.float32(255)


class Case:
    def __init__(self, directory, expected_case_sha256=None):
        self.root = Path(directory)
        case_path = self.root / "case.json"
        require(case_path.is_file() and not case_path.is_symlink(), "missing case.json")
        b = case_path.read_bytes()
        self.sha256 = digest(b)
        require(expected_case_sha256 is None or expected_case_sha256 == self.sha256, "case identity mismatch")
        self.meta = m = json.loads(b)
        require(
            m.get("schema") == "el-stage3-deploy-case-1" and m.get("condition") in CONDITIONS,
            "unknown stage-three deployment case",
        )
        self.condition = m["condition"]
        self.generator = CONDITIONS[self.condition]
        require(
            m.get("generator_source_ids") == list(self.generator)
            and m.get("target_ids") == list(TARGETS)
            and set(m["source_images"]) == set(self.generator),
            "source authorization mismatch",
        )
        self.pred = json.loads(read_member(self.root, m["prediction"], "prediction.json"))
        p = self.pred
        require(
            p.get("schema") == "el-stage3-prediction-1"
            and type(p.get("depth_verified")) is bool
            and p.get("generator_source_ids") == list(self.generator)
            and p.get("target_ids") == list(TARGETS)
            and p.get("reconstruction_run_id") == m["reconstruction_run_id"]
            and p.get("checkpoint_sha256") == m.get("checkpoint_sha256")
            and HASH.fullmatch(str(p.get("checkpoint_sha256", "")))
            and p.get("source_rgb_sha256") == {s: m["source_images"][s]["sha256"] for s in self.generator},
            "prediction/source/reconstruction identity or numerical semantics changed",
        )
        for key, expected in (
            ("rgb_encoding", "display_srgb_0_1"),
            ("depth_semantics", "expected_camera_z_m"),
            ("alpha_semantics", "accumulated_opacity"),
        ):
            require(p.get(key, expected) == expected, "unsupported predicted channel semantics")
        src = json.loads(read_member(self.root, m["source_cameras"], "source_cameras.json"))
        ren = json.loads(read_member(self.root, m["render_cameras"], "render_cameras.json"))
        require(
            m["source_cameras"]["sha256"] == p.get("source_camera_sha256")
            and m["render_cameras"]["sha256"] == p.get("render_camera_sha256"),
            "predicted calibration identity differs",
        )
        cameras(src, self.generator)
        self.cameras = cameras(ren, (*self.generator, *TARGETS))
        require(
            all(src["views"][s] == ren["views"][s] for s in self.generator),
            "source and render calibration differ",
        )
        self.failed = p.get("failed_views", {})
        expected_views = set(self.generator) | set(TARGETS)
        require(
            isinstance(self.failed, dict)
            and set(self.failed) <= expected_views
            and all(isinstance(x, str) and x for x in self.failed.values())
            and set(p["views"]) == expected_views - set(self.failed)
            and set(m["renders"]) == set(p["views"]),
            "incomplete predicted view dispositions",
        )
        for view, e in m["renders"].items():
            require(
                all(e[k] == p["views"][view].get(k) for k in ("bytes", "sha256"))
                and e["path"] == f"predictions/{view}.npz",
                "render identity differs for " + view,
            )

    def frame(self, view):
        require(view in (*self.generator, *TARGETS), "unavailable render view")
        if view in self.failed:
            return None
        return decode_prediction(
            read_member(self.root, self.meta["renders"][view], f"predictions/{view}.npz"), self.cameras[view]
        )

    def sources(self, scope):
        result = {}
        for s in source_ids(self.condition, scope):
            b = read_member(self.root, self.meta["source_images"][s], f"sources/{s}.png")
            result[s] = (decode_png(b, self.cameras[s]), self.frame(s), self.cameras[s])
        return result

    def features(self, view, scope):
        require(view in TARGETS, "only preregistered new target views may be scored")
        t = self.frame(view)
        if t is None:
            raise ValueError(f"target render failed for {view}: {self.failed[view]}")
        return t["rgb"], extract(
            t, self.cameras[view], self.sources(scope), depth_verified=self.pred["depth_verified"]
        )


def load_native(case_dir, view, scope, *, expected_case_sha256=None):
    return Case(case_dir, expected_case_sha256).features(view, scope)
