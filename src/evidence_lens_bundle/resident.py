"""Serial, array-in/array-out inference with the frozen CPU model and original cutoff."""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, replace
import threading
from time import perf_counter
from types import MappingProxyType

import numpy as np

from evidence_lens.multisource import camera_rays, source_self_mae, validate_render
from evidence_lens.scoring_io import Camera, check_rgb, rgb_valid
from .bindings import (
    CAPABILITIES,
    VERSIONS,
    array_sha,
    binding_from_bundle,
    camera_identity,
    check_binding,
    check_frame_meta,
    frame_meta_from_bundle,
    render_identity,
    source_identity,
)
from .common import PROFILE, RELEASE_INDEX_SHA256, Unsupported, encoded, json_bytes, require, sha
from .scoring import calculate_frame, models

__all__ = ["CAPABILITIES", "load_scorer", "binding_from_bundle", "frame_meta_from_bundle", "PreparedSources"]


def _immutable(array):
    # A bytes-backed ndarray cannot be made writable via setflags or an exposed base.
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _snapshot_source(item):
    image, frame, cam = item
    photo = _immutable(image)
    check_rgb(photo, cam, "authorized source")
    require(rgb_valid(photo).all(), "original source RGB must be finite")
    copied = None
    if frame is not None:
        copied = MappingProxyType({k: _immutable(a) for k, a in frame.items()})
        validate_render(copied, cam)
    camera = Camera(cam.width, cam.height, _immutable(cam.K), _immutable(cam.c2w))
    return photo, copied, camera


class _Cache:
    """One LRU for owned snapshots and rays. Payload sizes include array storage and identity bytes."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = self.bytes = self.peak = 0

    def get(self, key):
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key][0]
        self.misses += 1
        return None

    def put(self, key, value, size):
        if size > self.capacity:
            return
        while self.bytes + size > self.capacity:
            _, (_, freed) = self.entries.popitem(last=False)
            self.evictions += 1
            self.bytes -= freed
        self.entries[key] = value, size
        self.bytes += size
        self.peak = max(self.peak, self.bytes)

    def info(self):
        return {
            "capacity_bytes": self.capacity,
            "bytes": self.bytes,
            "peak_bytes": self.peak,
            "entries": len(self.entries),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


@dataclass(frozen=True, eq=False, slots=True)
class PreparedSources:
    """An immutable version snapshot owned by its originating scorer. Release references after use."""

    key: str
    _owner: object
    _identity: bytes
    _versions: tuple
    binding_sha256: str
    _sources: object
    _self_mae: object
    _primary: str
    _verified: bool
    prepare_seconds: float = 0.0
    cache_hit: bool = False

    @property
    def identity(self):
        return json_bytes(self._identity)


class Scorer:
    def __init__(self, method, models_dir, threads, cache_bytes):
        start = perf_counter()
        self._method, self._threads = method, threads
        self._model, self._spec = models(method, models_dir, threads)
        self._owner = object()
        self._cache = _Cache(cache_bytes)
        self._lock = threading.Lock()
        self.model_load_seconds = perf_counter() - start

    @property
    def method(self):
        return self._method

    @property
    def model_id(self):
        return self._spec["model_id"]

    @property
    def threshold(self):
        return self._spec["threshold"]

    @contextmanager
    def _serial(self):
        require(self._lock.acquire(blocking=False), "one resident scorer permits one active call at a time")
        try:
            yield
        finally:
            self._lock.release()

    def cache_info(self):
        return self._cache.info()

    def prepare_sources(self, source_evidence, *, version_binding):
        start = perf_counter()
        with self._serial():
            require(isinstance(source_evidence, dict), "ordered source ID mapping required")
            check_binding(version_binding, source_evidence)
            b = version_binding
            sources = {}
            for name, (image, frame, cam) in source_evidence.items():
                sources[name] = source_identity(image, frame, cam)
                if b["source_files"] is not None:
                    declared = b["source_files"][name]
                    ready = declared["self_render"]["status"] == "ready"
                    require(
                        ready == (frame is not None), "source render disposition differs from supplied arrays"
                    )
                    require(
                        sha(encoded(sources[name])) == declared["decoded_source_sha256"],
                        "source buffers or camera differ from authenticated bundle",
                    )
            primary = b["primary_source_id"]
            consumed = [primary] if self.method == "B-L1-transfer" else list(source_evidence)
            identity = encoded(
                {
                    "binding": b,
                    "source_arrays": sources,
                    "model": self._spec,
                    "method": self.method,
                    "scorer_source_ids": consumed,
                    "release_index_sha256": RELEASE_INDEX_SHA256,
                }
            )
            key = sha(identity)
            previous = self._cache.get(("sources", key))
            if previous is not None:
                return replace(previous, prepare_seconds=perf_counter() - start, cache_hit=True)
            snapshots = {}
            size = len(identity)
            for name, item in source_evidence.items():
                snapshot = _snapshot_source(item)
                if name in consumed:
                    image, frame, cam = snapshots[name] = snapshot
                    size += image.nbytes + cam.K.nbytes + cam.c2w.nbytes
                    if frame is not None:
                        size += sum(a.nbytes for a in frame.values())
            residuals = None
            if self.method != "B-L1-transfer":
                extra = sum(
                    image.shape[0] * image.shape[1] * 4
                    for image, frame, _ in snapshots.values()
                    if frame is not None
                )
                if size + extra <= self._cache.capacity:
                    residuals = MappingProxyType(
                        {
                            s: _immutable(source_self_mae(image, frame))
                            for s, (image, frame, _) in snapshots.items()
                            if frame is not None
                        }
                    )
                    size += extra
            context = PreparedSources(
                key,
                self._owner,
                identity,
                tuple(b[k] for k in VERSIONS),
                sha(encoded(b)),
                MappingProxyType(snapshots),
                residuals,
                primary,
                b["depth_verified"],
            )
            self._cache.put(("sources", key), context, size)
            return replace(context, prepare_seconds=perf_counter() - start)

    def score_frame(self, context, target_prediction, target_camera, *, frame_meta):
        """Compute a fresh result. The caller keeps input buffers fixed for the duration of the call."""
        start = perf_counter()
        with self._serial():
            require(
                isinstance(context, PreparedSources) and context._owner is self._owner,
                "source context belongs to another scorer or model",
            )
            if self.method in ("MS-frozen", "MS-ft"):
                import torch

                require(
                    torch.get_num_threads() == self._threads,
                    "PyTorch process thread configuration changed; isolate different settings in separate processes",
                )
            cam = camera_identity(target_camera)
            check_frame_meta(frame_meta, context._versions, cam, target_prediction)
            if frame_meta.get("source_binding_sha256") is not None:
                require(
                    frame_meta["source_binding_sha256"] == context.binding_sha256,
                    "frame requires a different source binding",
                )
            if target_prediction is not None:
                validate_render(target_prediction, target_camera)
            elif frame_meta.get("render_status") != "render_failed" or not frame_meta.get("render_failure"):
                raise ValueError("missing target prediction requires an explicit render failure")
            prediction_id = render_identity(target_prediction)
            prediction_sha = sha(encoded(prediction_id))
            if frame_meta.get("target_array_sha256") is not None:
                require(
                    frame_meta["target_array_sha256"] == prediction_sha,
                    "target content differs from expected arrays",
                )
            # The K-dependent ray grid has no dependency on target pixels, pose or depth.
            # Context membership invalidates even this invariant when the registered map/source changes.
            rays = None
            if self._cache.capacity and self.method != "B-L1-transfer" and target_prediction is not None:
                ray_key = ("rays", context.key, cam["K_sha256"], target_camera.shape)
                rays = self._cache.get(ray_key)
                if rays is None:
                    rays = camera_rays(target_camera)
                    self._cache.put(ray_key, rays, rays.nbytes)
            bound = perf_counter()
            timing = {"identity_and_cache_seconds": bound - start}
            result = calculate_frame(
                target_prediction,
                target_camera,
                context._sources,
                context._primary,
                context._verified,
                self.method,
                self._model,
                self.threshold,
                rays=rays,
                self_mae_maps=context._self_mae,
                timing=timing,
            )
            after_kernel = perf_counter()
            # A fresh JSON copy separates mutable calling metadata from returned evidence.
            meta = json_bytes(encoded(frame_meta))
            identity = {
                "source_context_sha256": context.key,
                **dict(zip(VERSIONS, context._versions)),
                "frame_meta": meta,
                "target_camera": cam,
                "target_prediction": prediction_id,
                "target_array_sha256": prediction_sha,
                "model_id": self.model_id,
                "method": self.method,
                "weights_sha256": self._spec["weights_sha256"],
                "feature_schema_sha256": self._spec["feature_schema_sha256"],
                "preprocessing_sha256": self._spec["preprocessing_sha256"],
                "model_index_sha256": self._spec["model_index_sha256"],
            }
            identity["frame_input_sha256"] = sha(encoded(identity))
            identity["output_arrays_sha256"] = {name: array_sha(array) for name, array in result.items()}
            identity["output_sha256"] = sha(encoded(identity["output_arrays_sha256"]))
            profile = {
                "id": PROFILE,
                "original_threshold": self.threshold,
                "calibration_profile": None,
                "applicability": "supported_native_input_contract",
                "external_calibration": "unestablished",
            }
            result.update(identity=identity, profile=profile, timing=timing)
            end = perf_counter()
            timing["postprocess_seconds"] += end - after_kernel
            timing["service_seconds"] = end - start
            return result


def load_scorer(method="MS-L1", device="cpu", *, models_dir=None, threads=4, cache_bytes=64 << 20):
    """Load the exact frozen release once. A zero-byte cache uses the original per-frame feature path."""
    require(device == "cpu", "only the CPU risk backend is supported", Unsupported)
    require(type(threads) is int and 1 <= threads <= 256, "CPU thread count must be 1–256")
    require(
        type(cache_bytes) is int and cache_bytes >= 0, "cache byte capacity must be a nonnegative integer"
    )
    return Scorer(method, models_dir, threads, cache_bytes)
