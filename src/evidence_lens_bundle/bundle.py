"""Checked M2/M4 deployment grid. Camera IDs are opaque; they are never filenames."""

from __future__ import annotations

import math
from pathlib import Path

from evidence_lens.multisource_io import cameras, decode_png, decode_prediction

from .common import (
    PROFILE,
    RELEASE_INDEX_SHA256,
    SEMANTICS,
    Unsupported,
    digest,
    encoded,
    exact_keys,
    json_bytes,
    member,
    profile,
    read_file,
    require,
    sha,
    text,
    view_objects,
)


def ids(names, role):
    require(isinstance(names, list) and names, "missing " + role)
    for name in names:
        text(name, role)
    require(len(set(names)) == len(names), "duplicate " + role)
    return tuple(names)


def disposition(item):
    require(isinstance(item, dict), "invalid render disposition")
    if item.get("status") == "render_failed":
        exact_keys(item, ("status", "reason"), "failed render")
        text(item["reason"], "render failure reason")
    else:
        exact_keys(item, ("status", "file"), "ready render")
        require(item["status"] == "ready", "unsupported render status")


class Bundle:
    def __init__(self, root, expected_sha256=None):
        self.root = Path(root)
        raw = read_file(self.root, "bundle.json")
        self.sha256 = sha(raw)
        if expected_sha256 is not None:
            require(expected_sha256 == self.sha256, "bundle SHA-256 differs")
        b = self.meta = json_bytes(raw)
        exact_keys(
            b,
            (
                "schema",
                "creation",
                "input_domain",
                "condition",
                "source_order",
                "primary_source_id",
                "endpoint_source_ids",
                "target_order",
                "camera",
                "geometry",
                "sources",
                "producer",
                "risk_profile",
            ),
            "bundle (deployment evidence only)",
        )
        require(b["schema"] == "el-evidence-bundle-1", "unsupported bundle schema", Unsupported)
        self.condition = b["condition"]
        self.order = ids(b["source_order"], "source ID")
        self.targets = ids(b["target_order"], "target ID")
        self.endpoints = ids(b["endpoint_source_ids"], "endpoint ID")
        self.primary = b["primary_source_id"]
        require(
            self.condition in ("M2", "M4") and len(self.order) == {"M2": 2, "M4": 4}[self.condition],
            "released model domain requires 2 or 4 sources with matching M2/M4 condition",
            Unsupported,
        )
        require(
            len(self.endpoints) == 2
            and set(self.endpoints) <= set(self.order)
            and self.primary in self.order,
            "unauthorized endpoints or primary source",
        )
        require(set(self.order).isdisjoint(self.targets), "target overlaps authorized photos")
        exact_keys(b["sources"], self.order, "authorized source photo IDs")
        text(b["input_domain"], "actual input domain")
        creation = b["creation"]
        fields = {"run_id", "implementation", "code_sha256", "config_sha256", "original_case_sha256"}
        if "legacy_proof_sha256" in creation:
            digest(creation["legacy_proof_sha256"], "legacy proof")
            fields.add("legacy_proof_sha256")
        exact_keys(creation, fields, "creation")
        for k in ("implementation", "run_id"):
            text(creation[k], k)
        for k in ("code_sha256", "config_sha256"):
            digest(creation[k], k)
        if creation["original_case_sha256"] is not None:
            digest(creation["original_case_sha256"], "original legacy case")
        p = profile()
        require(
            b["risk_profile"]
            == {"id": PROFILE, "sha256": sha(encoded(p)), "release_index_sha256": RELEASE_INDEX_SHA256},
            "unsupported or changed frozen model/threshold profile",
            Unsupported,
        )
        g = b["geometry"]
        exact_keys(
            g,
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
        scale = g["meters_per_original_unit"]
        require(
            g["prepared_camera_and_z_unit"] == "m"
            and type(scale) in (int, float)
            and math.isfinite(scale)
            and scale > 0,
            "positive evidenced metric scale required",
            Unsupported,
        )
        for k in ("metric_evidence", "crop", "resize", "undistortion"):
            text(g[k], k)
        require(g["scoring_resampling"] == "none", "score resizing is unsupported", Unsupported)
        cam = json_bytes(member(self.root, b["camera"], expected="cameras.json"))
        require(
            (cam.get("axis_convention"), cam.get("image_origin"), cam.get("pixel_center"), cam.get("unit"))
            == ("OpenCV x right, y down, z forward", "top_left", "integer", "m"),
            "explicit OpenCV integer-centre, top-left, metric cameras required",
            Unsupported,
        )
        lens = tuple(self.order) + self.targets
        view_objects(cam, "camera")
        self.cameras = cameras(cam, lens)
        for name in lens:
            v = cam["views"][name]
            require(
                v.get("projection") == "pinhole" and v.get("distortion") == "none",
                "prepared camera must be an undistorted pinhole: " + name,
                Unsupported,
            )
            require(
                self.cameras[name].shape == (256, 256),
                "released model domain requires native 256 x 256: " + name,
                Unsupported,
            )
        producer = self.producer = json_bytes(member(self.root, b["producer"], expected="producer.json"))
        exact_keys(
            producer,
            (
                "schema",
                "generator",
                "renderer",
                "checkpoint_sha256",
                "camera_sha256",
                "source_rgb_sha256",
                "source_order",
                "target_order",
                "semantics",
                "depth_verified",
                "depth_probe_sha256",
                "frames",
            ),
            "producer",
        )
        require(producer["schema"] == "el-render-evidence-1", "unsupported producer schema", Unsupported)
        for role in ("generator", "renderer"):
            ident = producer[role]
            exact_keys(ident, ("run_id", "receipt_sha256"), role)
            text(ident["run_id"], role + " run")
            digest(ident["receipt_sha256"], role + " receipt")
        digest(producer["checkpoint_sha256"], "reconstruction checkpoint")
        require(type(producer["depth_verified"]) is bool, "expected boolean depth verification")
        if producer["depth_verified"] or producer["depth_probe_sha256"] is not None:
            digest(producer["depth_probe_sha256"], "camera-Z probe")
        require(
            producer["semantics"] == SEMANTICS, "unsupported RGB, camera-Z or alpha semantics", Unsupported
        )
        require(
            producer["camera_sha256"] == b["camera"]["sha256"]
            and producer["source_order"] == list(self.order)
            and producer["target_order"] == list(self.targets),
            "producer camera or authorized view order differs",
        )
        require(
            producer["source_rgb_sha256"] == {s: b["sources"][s]["sha256"] for s in self.order},
            "source photo hashes differ from generator inputs",
        )
        exact_keys(producer["frames"], lens, "render dispositions")
        paths = []
        for s in self.order:
            paths.append(b["sources"][s]["path"])
        for k in lens:
            spec = producer["frames"][k]
            disposition(spec)
            if spec["status"] == "ready":
                paths.append(spec["file"]["path"])
        require(len(paths) == len(set(paths)), "one asset path assigned multiple roles")

    def photo_bytes(self, source):
        return member(self.root, self.meta["sources"][source], prefix="sources")

    def frame(self, view):
        item = self.producer["frames"][view]
        if item["status"] == "render_failed":
            return None
        return decode_prediction(member(self.root, item["file"], prefix="renders"), self.cameras[view])

    def source(self, source):
        return (
            decode_png(self.photo_bytes(source), self.cameras[source]),
            self.frame(source),
            self.cameras[source],
        )

    def validate(self):
        for s in self.order:
            self.source(s)
        for t in self.targets:
            self.frame(t)
        return {
            "schema": "el-bundle-validation-1",
            "bundle_sha256": self.sha256,
            "condition": self.condition,
            "source_order": list(self.order),
            "target_order": list(self.targets),
            "primary_source_id": self.primary,
            "endpoint_source_ids": list(self.endpoints),
            "render_failed": {
                v: x["reason"] for v, x in self.producer["frames"].items() if x["status"] == "render_failed"
            },
            "depth_verified": self.producer["depth_verified"],
            "status": "valid",
        }
