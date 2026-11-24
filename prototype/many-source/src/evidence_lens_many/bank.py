"""Deployment-only, content-addressed camera and source inventory."""

from collections import OrderedDict
import io
from pathlib import Path
from time import perf_counter
from types import MappingProxyType

import numpy as np
from evidence_lens.multisource_io import decode_png, decode_prediction
from evidence_lens.scoring_io import Camera
from evidence_lens_bundle.common import (
    digest,
    encoded,
    exact_keys,
    json_bytes,
    parts,
    read_file,
    require,
    sha,
    text,
)
from evidence_lens_bundle.grids import validate_grids

from .schema import BASELINE_ID, PROFILE, SIZE

MAX_CACHE = 512 << 20


def build_manifest(
    *,
    scene_id,
    checkpoint_sha256,
    renderer_id,
    reconstruction_provenance,
    reconstruction_ids,
    accessible_ids,
    target_ids,
    metric_anchor_ids,
    frame_receipt,
    cameras,
    grids,
    sources,
    targets,
    depth_verified,
    depth_probe_sha256,
    permission,
    provenance=None,
):
    return {
        "schema": "el-many-source-bank-1",
        "baseline_id": BASELINE_ID,
        "scene_id": scene_id,
        "checkpoint_sha256": checkpoint_sha256,
        "renderer_id": renderer_id,
        "reconstruction_provenance": reconstruction_provenance,
        "reconstruction_ids": list(reconstruction_ids),
        "accessible_ids": list(accessible_ids),
        "target_ids": list(target_ids),
        "metric_anchor_ids": list(metric_anchor_ids),
        "frame_receipt": frame_receipt,
        "cameras": cameras,
        "grids": grids,
        "sources": sources,
        "targets": targets,
        "depth_verified": depth_verified,
        "depth_probe_sha256": depth_probe_sha256,
        "permission": permission,
        **({"provenance": provenance} if provenance is not None else {}),
    }


def _ids(values, role):
    require(
        isinstance(values, list)
        and all(isinstance(v, str) and v and v.isprintable() for v in values)
        and len(values) == len(set(values)),
        "invalid or repeated " + role,
    )
    return tuple(values)


def _spec(value):
    if value is None:
        return
    exact_keys(value, ("path", "sha256", "bytes"), "file identity")
    parts(value["path"])
    digest(value["sha256"], "file")
    require(type(value["bytes"]) is int and value["bytes"] > 0, "nonpositive file size")


def _camera(row):
    k = np.asarray(row.get("K"), np.float64)
    pose = np.asarray(row.get("c2w"), np.float64)
    require(
        (row.get("width"), row.get("height")) == SIZE
        and k.shape == (3, 3)
        and np.isfinite(k).all()
        and np.array_equal(k[2], [0, 0, 1])
        and k[0, 0] > 0
        and k[1, 1] > 0
        and k[0, 1] == k[1, 0] == 0,
        "expected fixed 378x252 integer-centre pinhole K",
    )
    require(
        pose.shape == (4, 4)
        and np.isfinite(pose).all()
        and np.allclose(pose[3], [0, 0, 0, 1])
        and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), rtol=0, atol=1e-5)
        and abs(np.linalg.det(pose[:3, :3]) - 1) < 1e-5,
        "expected OpenCV proper rigid c2w",
    )
    return Camera(*SIZE, _fixed(k), _fixed(pose))


def _fixed(a):
    return np.frombuffer(a.tobytes(order="C"), dtype=a.dtype).reshape(a.shape)


def _stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class Bank:
    def __init__(self, root, expected_sha256=None, *, cache_bytes=128 << 20):
        self.root = Path(root)
        raw = read_file(self.root, "bank.json")
        self.sha256 = sha(raw)
        require(expected_sha256 is None or expected_sha256 == self.sha256, "bank manifest SHA-256 differs")
        self.meta = m = json_bytes(raw)
        keys = "schema baseline_id scene_id checkpoint_sha256 renderer_id reconstruction_provenance reconstruction_ids accessible_ids target_ids metric_anchor_ids frame_receipt cameras grids sources targets depth_verified depth_probe_sha256 permission".split()
        exact_keys(m, [*keys, *(("provenance",) if "provenance" in m else ())], "source bank")
        require(
            m["schema"] == "el-many-source-bank-1" and m["baseline_id"] == BASELINE_ID,
            "unsupported bank or A baseline",
        )
        text(m["scene_id"], "scene")
        text(m["renderer_id"], "renderer")
        digest(m["checkpoint_sha256"], "checkpoint")
        self.reconstruction_ids = _ids(m["reconstruction_ids"], "reconstruction IDs")
        self.accessible_ids = _ids(m["accessible_ids"], "accessible IDs")
        self.target_ids = _ids(m["target_ids"], "holdout camera IDs")
        anchors = _ids(m["metric_anchor_ids"], "frame anchors")
        require(
            self.accessible_ids
            and self.target_ids
            and set(self.target_ids).isdisjoint((*self.accessible_ids, *self.reconstruction_ids))
            and set(anchors) <= set(self.reconstruction_ids) | set(self.accessible_ids),
            "R, A, H or frame anchors overlap incorrectly",
        )
        require(
            m["reconstruction_provenance"] in ("verified", "unverified"), "unknown reconstruction provenance"
        )
        if m["reconstruction_provenance"] == "verified":
            require(bool(self.reconstruction_ids), "verified map requires actual reconstruction identities")
        frame = m["frame_receipt"]
        tx = np.asarray(frame.get("matrix"), dtype=np.float64)
        require(
            frame.get("unit") in ("m", "scene_units")
            and tx.shape == (4, 4)
            and np.isfinite(tx).all()
            and np.allclose(tx[3], [0, 0, 0, 1])
            and np.linalg.det(tx[:3, :3]) > 0,
            "missing fixed coordinate frame, unit or proper transform",
        )
        gram = tx[:3, :3].T @ tx[:3, :3]
        require(
            np.allclose(gram, np.eye(3) * np.trace(gram) / 3, rtol=1e-6, atol=1e-9),
            "frame must be a positive similarity",
        )
        granted = _ids(m["permission"].get("source_ids"), "source permissions")
        require(set(granted) <= set(self.accessible_ids), "source grant exceeds A")
        self.granted = frozenset(granted)
        text(m["permission"].get("terms"), "source terms")
        require(type(m["depth_verified"]) is bool, "depth verification status required")
        if m["depth_verified"] or m["depth_probe_sha256"] is not None:
            digest(m["depth_probe_sha256"], "expected-depth probe")
        views = (*self.accessible_ids, *self.target_ids)
        exact_keys(m["cameras"], views, "deployment cameras")
        require(
            validate_grids(m["grids"], {"views": m["cameras"]}, views) == PROFILE,
            "unsupported analysis profile",
        )
        self.cameras = {name: _camera(row) for name, row in m["cameras"].items()}
        exact_keys(m["sources"], self.accessible_ids, "source role")
        exact_keys(m["targets"], self.target_ids, "target prediction role")
        self.aliases = {}
        parents = {name: name for name in self.accessible_ids}
        fingerprints = {}

        def root(name):
            while parents[name] != name:
                parents[name] = parents[parents[name]]
                name = parents[name]
            return name

        for name in self.accessible_ids:
            row = m["sources"][name]
            exact_keys(row, ("photo", "render", "original_sha256"), "source photo and self-render")
            if row["original_sha256"] is not None or row["photo"] is not None:
                digest(row["original_sha256"], "registered original")
            _spec(row["photo"])
            _spec(row["render"])
            for role, fingerprint in (
                ("original", row["original_sha256"]),
                ("analysis", row["photo"]["sha256"] if row["photo"] else None),
            ):
                if fingerprint is not None:
                    previous = fingerprints.setdefault((role, fingerprint), name)
                    parents[root(name)] = root(previous)
        unique = {}
        for name in self.accessible_ids:
            unique.setdefault(root(name), []).append(name)
        primary = []
        for group in unique.values():
            origin = min(
                group,
                key=lambda n: (
                    n not in self.granted,
                    m["sources"][n]["photo"] is None,
                    m["sources"][n]["render"] is None,
                    self.accessible_ids.index(n),
                ),
            )
            primary.append(origin)
            self.aliases.update({name: origin for name in group if name != origin})
        primary = set(primary)
        self.ids = tuple(n for n in self.accessible_ids if n in primary)
        self.source_index = {name: i for i, name in enumerate(self.ids)}
        for row in m["targets"].values():
            exact_keys(row, ("render",), "target prediction")
            _spec(row["render"])
        photos = {r["photo"]["path"] for r in m["sources"].values() if r["photo"]}
        self_renders = {r["render"]["path"] for r in m["sources"].values() if r["render"]}
        targets = [r["render"]["path"] for r in m["targets"].values() if r["render"]]
        require(
            photos.isdisjoint(self_renders)
            and set(targets).isdisjoint(photos | self_renders)
            and len(targets) == len(set(targets)),
            "a file is assigned incompatible source and target roles",
        )
        require(
            type(cache_bytes) is int and 0 <= cache_bytes <= MAX_CACHE,
            "source LRU capacity must be between zero and 512 MiB",
        )
        self.capacity = cache_bytes
        self._cache = OrderedDict()
        self._bytes = 0
        self.counters = dict(
            cache_hits=0,
            cache_misses=0,
            cache_evictions=0,
            cache_peak_bytes=0,
            source_photos_decoded=0,
            source_renders_decoded=0,
            target_rgb_decoded=0,
            target_geometry_decoded=0,
            source_bytes_read=0,
            target_bytes_read=0,
            source_decode_seconds=0.0,
            target_decode_seconds=0.0,
        )

    def _path(self, spec):
        if spec is None:
            return None
        current = self.root
        for part in parts(spec["path"]):
            current = current / part
            require(not current.is_symlink(), "symlink in deployment member")
        return current if current.is_file() else None

    def available(self, source):
        row = self.meta["sources"][source]
        return (
            source in self.granted
            and self._path(row["photo"]) is not None
            and self._path(row["render"]) is not None
        )

    def _data(self, spec, role):
        raw = read_file(self.root, spec["path"])
        self.counters[role + "_bytes_read"] += len(raw)
        require(
            len(raw) == spec["bytes"] and sha(raw) == spec["sha256"],
            "changed " + role + " file " + spec["path"],
        )
        return raw

    def source(self, source):
        require(source in self.source_index, "unregistered or duplicate query source")
        if not self.available(source):
            return None
        row = self.meta["sources"][source]
        camera = self.cameras[source]
        stamps = tuple(_stamp(self._path(row[k])) for k in ("photo", "render"))
        key = (self.sha256, source, row["photo"]["sha256"], row["render"]["sha256"], stamps)
        tick = perf_counter()
        if key in self._cache:
            self.counters["cache_hits"] += 1
            self._cache.move_to_end(key)
            return self._cache[key][0]
        self.counters["cache_misses"] += 1
        photo = _fixed(decode_png(self._data(row["photo"], "source"), camera))
        self.counters["source_photos_decoded"] += 1
        frame = MappingProxyType(
            {k: _fixed(a) for k, a in decode_prediction(self._data(row["render"], "source"), camera).items()}
        )
        self.counters["source_renders_decoded"] += 1
        self.counters["source_decode_seconds"] += perf_counter() - tick
        require(
            stamps == tuple(_stamp(self._path(row[k])) for k in ("photo", "render")),
            "source changed during decode",
        )
        value = (photo, frame, camera)
        size = (
            photo.nbytes
            + sum(a.nbytes for a in frame.values())
            + len(encoded({"bank": self.sha256, "source": source, "file": row}))
        )
        if size <= self.capacity:
            while self._bytes + size > self.capacity:
                _, (_, freed) = self._cache.popitem(last=False)
                self._bytes -= freed
                self.counters["cache_evictions"] += 1
            self._cache[key] = value, size
            self._bytes += size
            self.counters["cache_peak_bytes"] = max(self.counters["cache_peak_bytes"], self._bytes)
        return value

    def _target(self, name, geometry):
        require(name in self.target_ids, "unregistered target camera")
        spec = self.meta["targets"][name]["render"]
        if self._path(spec) is None:
            return None
        tick = perf_counter()
        raw = self._data(spec, "target")
        if geometry:
            with np.load(io.BytesIO(raw), allow_pickle=False) as arr:
                require(
                    set(arr.files) == {"rgb", "alpha", "z"} and len(arr.files) == 3,
                    "wrong target prediction roles",
                )
                frame = {k: arr[k] for k in ("z", "alpha")}
            require(
                all(a.shape == SIZE[::-1] and a.dtype == np.float32 for a in frame.values()),
                "wrong target geometry grid",
            )
        else:
            frame = decode_prediction(raw, self.cameras[name])
            self.counters["target_rgb_decoded"] += 1
        self.counters["target_geometry_decoded"] += 1
        self.counters["target_decode_seconds"] += perf_counter() - tick
        return frame

    def target(self, name):
        return self._target(name, False)

    def target_geometry(self, name):
        return self._target(name, True)

    def inspect(self):
        from .select import rig_geometry

        names = tuple(sorted(n for n in self.ids if n in self.granted))
        pairs = []
        if names:
            close = rig_geometry([self.cameras[n] for n in names])[-1]
            pairs = [[names[i], names[j]] for i, j in zip(*np.nonzero(np.triu(close, 1)))]
        return {
            "bank_sha256": self.sha256,
            "baseline_id": BASELINE_ID,
            "scene_id": self.meta["scene_id"],
            "reconstruction_ids": self.reconstruction_ids,
            "registered_accessible_ids": self.accessible_ids,
            "unique_accessible_ids": self.ids,
            "duplicate_aliases": self.aliases,
            "near_duplicate_pairs": pairs,
            "target_camera_ids": self.target_ids,
            "unavailable_source_ids": [n for n in self.ids if not self.available(n)],
            "missing_target_prediction_ids": [
                n for n in self.target_ids if self._path(self.meta["targets"][n]["render"]) is None
            ],
            "depth_verified": self.meta["depth_verified"],
            "reconstruction_provenance": self.meta["reconstruction_provenance"],
            "frame_receipt": self.meta["frame_receipt"],
            "metric_anchor_ids": self.meta["metric_anchor_ids"],
            "permission": self.meta["permission"],
        }
