from copy import deepcopy

import numpy as np
import pytest

from test_bundle import S, T, fixture
from evidence_lens_bundle.bindings import array_sha, binding_from_bundle, frame_meta_from_bundle
from evidence_lens_bundle.bundle import Bundle
from evidence_lens_bundle.numerics import CHANNELS, compare_arrays, matched_sets, METHODS
from evidence_lens_bundle.resident import load_scorer
from evidence_lens_bundle.scoring import calculate, models


def test_reference_exact_and_array_lifetimes(tmp_path):
    bundle = Bundle(fixture(tmp_path / "bundle"))
    sources = {s: deepcopy(bundle.source(s)) for s in bundle.order}
    b = binding_from_bundle(bundle)
    frame, camera = bundle.frame(T[0]), bundle.cameras[T[0]]
    meta = frame_meta_from_bundle(bundle, T[0], binding=b)
    model, spec = models("MS-L1", None, 1)
    expected = calculate(bundle, T[0], frame, sources, "MS-L1", model, spec["threshold"])
    cached = load_scorer()
    ctx = cached.prepare_sources(sources, version_binding=b)
    assert cached.prepare_sources(sources, version_binding=b).cache_hit
    straight = load_scorer(cache_bytes=0)
    ctx0 = straight.prepare_sources(sources, version_binding=b)
    for engine, context in ((cached, ctx), (straight, ctx0)):
        result = engine.score_frame(context, frame, camera, frame_meta=meta)
        assert compare_arrays(expected, result, spec["threshold"])["bit_exact"]
        assert result["identity"]["output_arrays_sha256"] == {k: array_sha(expected[k]) for k in CHANNELS}
        assert result["timing"]["feature_seconds"] > 0 and result["timing"]["model_seconds"] > 0
        assert result["profile"]["original_threshold"] == spec["threshold"]
    original = cached.score_frame(ctx, frame, camera, frame_meta=meta)
    meta["frame_id"] = "second delivery"
    sources[S[0]][0][10, 10, 0] = 0.9
    sources[S[1]][1]["rgb"][20, 20, 0] = 0.8
    sources[S[2]][2].c2w[0, 3] += 0.1
    with pytest.raises(ValueError, match="authenticated bundle"):
        cached.prepare_sources(sources, version_binding=b)
    ctx2 = cached.prepare_sources(sources, version_binding={**b, "source_files": None})
    assert ctx.key != ctx2.key
    original["risk"][:] = 0
    again = cached.score_frame(ctx, frame, camera, frame_meta=meta)
    assert compare_arrays(expected, again, spec["threshold"])["bit_exact"]
    assert original["identity"]["frame_meta"]["frame_id"] != again["identity"]["frame_meta"]["frame_id"]
    before = again["identity"]["target_array_sha256"]
    frame["rgb"][0, 0, 0] = 0.1
    live = {
        k: v
        for k, v in meta.items()
        if k not in ("target_file_sha256", "target_array_sha256", "bundle_sha256")
    }
    assert (
        cached.score_frame(ctx, frame, camera, frame_meta=live)["identity"]["target_array_sha256"] != before
    )
    bad = {**meta, "target_array_sha256": before}
    with pytest.raises(ValueError, match="target content"):
        cached.score_frame(ctx, frame, camera, frame_meta=bad)
    missing = cached.score_frame(
        ctx, None, bundle.cameras[T[1]], frame_meta=frame_meta_from_bundle(bundle, T[1])
    )
    assert np.all(missing["unknown_reason"] == 7)
    with pytest.raises(ValueError, match="another scorer"):
        straight.score_frame(ctx, frame, camera, frame_meta=meta)


def test_version_cache_and_role_boundaries(tmp_path):
    bundle = Bundle(fixture(tmp_path / "bundle"))
    sources = {s: bundle.source(s) for s in bundle.order}
    b = binding_from_bundle(bundle)
    engine = load_scorer(cache_bytes=13 << 20)
    base = engine.prepare_sources(sources, version_binding=b)
    assert base.key == engine.prepare_sources(sources, version_binding=b).key
    for role in (
        "map_revision",
        "source_revision",
        "renderer_revision",
        "renderer_receipt_sha256",
        "renderer_config_sha256",
        "checkpoint_sha256",
        "bundle_config_sha256",
    ):
        variant = deepcopy(b)
        variant[role] = "c" * 64
        assert engine.prepare_sources(sources, version_binding=variant).key != base.key
    for mutate in ("rgb", "self_render", "K", "pose"):
        items = {s: deepcopy(bundle.source(s)) for s in bundle.order}
        photo, render, camera = items[S[0]]
        if mutate == "rgb":
            photo[1, 1, 0] = 0.8
        elif mutate == "self_render":
            render["z"][1, 1] += 0.2
        elif mutate == "K":
            camera.K[0, 0] += 1
        else:
            camera.c2w[0, 3] += 0.01
        with pytest.raises(ValueError, match="authenticated bundle"):
            engine.prepare_sources(items, version_binding=b)
        assert engine.prepare_sources(items, version_binding={**b, "source_files": None}).key != base.key
    for role in ("primary_source_id", "endpoint_source_ids"):
        variant = {**b, role: S[0] if role == "primary_source_id" else [S[0], S[1]]}
        assert engine.prepare_sources(sources, version_binding=variant).key != base.key
    reverse = dict(reversed(sources.items()))
    with pytest.raises(ValueError, match="order"):
        engine.prepare_sources(reverse, version_binding=b)
    allowed = {**b, "source_order": list(reverse)}
    assert engine.prepare_sources(reverse, version_binding=allowed).key != base.key
    with pytest.raises(ValueError, match="metric"):
        engine.prepare_sources(
            sources, version_binding={**b, "geometry": {**b["geometry"], "prepared_camera_and_z_unit": "mm"}}
        )
    for k in ("ReferenceBundle", "target_reference", "real_labels", "errors"):
        with pytest.raises(ValueError, match="metadata"):
            engine.score_frame(
                base,
                bundle.frame(T[0]),
                bundle.cameras[T[0]],
                frame_meta={**frame_meta_from_bundle(bundle, T[0]), k: "do not open"},
            )
    meta = frame_meta_from_bundle(bundle, T[0])
    with pytest.raises(ValueError, match="version"):
        engine.score_frame(
            base, bundle.frame(T[0]), bundle.cameras[T[0]], frame_meta={**meta, "map_revision": "later"}
        )
    with pytest.raises(ValueError, match="target camera"):
        engine.score_frame(base, bundle.frame(T[0]), bundle.cameras[T[1]], frame_meta=meta)
    other_config = engine.prepare_sources(sources, version_binding={**b, "renderer_config_sha256": "e" * 64})
    with pytest.raises(ValueError, match="different source binding"):
        engine.score_frame(other_config, bundle.frame(T[0]), bundle.cameras[T[0]], frame_meta=meta)
    with pytest.raises(ValueError, match="risk backend"):
        load_scorer(device="cuda")
    with pytest.raises(ValueError, match="needs --models"):
        load_scorer("MS-ft")
    info = engine.cache_info()
    assert info["evictions"] > 0 and info["peak_bytes"] <= info["capacity_bytes"]
    opaque = {
        **meta,
        "pose_covariance": {
            "matrix": np.eye(6).tolist(),
            "coordinates": "camera SE(3)",
            "units": "m and rad",
        },
        "dynamic_mask": {"reference": "opaque-id", "source": "caller", "time": None, "clock_domain": None},
    }
    out = engine.score_frame(base, bundle.frame(T[0]), bundle.cameras[T[0]], frame_meta=opaque)
    assert out["identity"]["frame_meta"]["pose_covariance"] == opaque["pose_covariance"]
    with engine._serial(), pytest.raises(ValueError, match="one active"):
        engine.score_frame(base, None, bundle.cameras[T[0]], frame_meta=meta)


def test_bit_identity_detects_equal_state_sort_changes():
    risk = np.linspace(0.1, 0.9, 65536, dtype=np.float32).reshape(256, 256)
    a = {
        "risk": risk,
        "valid": np.ones((256, 256), bool),
        "state": (risk >= 0.5).astype(np.uint8),
        "unknown_reason": np.zeros((256, 256), np.uint8),
    }
    b = {k: v.copy() for k, v in a.items()}
    k = (2 * 20 * 65536 + 99) // 200
    b["risk"].flat[k - 1], b["risk"].flat[k] = risk.flat[k], risk.flat[k - 1]
    diff = compare_arrays(a, b, 0.5)
    assert diff["changed_pixels"] == dict(zip(CHANNELS, (2, 0, 0, 0)))
    assert diff["threshold_crossings_common_finite"] == 0
    sets = matched_sets(dict.fromkeys(METHODS, a), dict.fromkeys(METHODS, b), "control", "M2", "v01", 0.5)
    twenty = next(r for r in sets["methods"]["MS-L1"] if r["anchor"] == "20%_canvas")
    assert twenty["added_pixels"] == twenty["removed_pixels"] == 1


def test_closure_binds_failed_slots_and_old_scores(tmp_path):
    from evidence_lens_bundle.bridge_history import check_joint_closure
    from evidence_lens_bundle.common import encoded, sha, write

    payload = tmp_path / "payload"
    payload.mkdir()
    cases = []
    roster = []
    for n in range(6):
        for cond in ("M2", "M4"):
            obj = f"registered-{n}"
            old = {
                "object_id": obj,
                "condition": cond,
                "status": "failed" if n == 0 else "succeeded",
                "reason": "probe failed" if n == 0 else None,
            }
            new = {"object_id": obj, "layout": cond + "-fixed"}
            for role, new_role in (("case", "case"), ("readback", "readback"), ("scores", "historical")):
                old[role] = {"sha256": sha((obj + cond + role).encode())}
                new[new_role] = {"sha256": old[role]["sha256"], "directory": "unopened"}
            if n == 0:
                new = {
                    "object_id": obj,
                    "layout": cond + "-fixed",
                    "terminal_failure": {"reason": "probe failed"},
                }
            cases.append(old)
            roster.append(new)
    raw = encoded(
        {
            "schema": "el-t1-closed-matrix-1",
            "run_id": "closure",
            "objects": [f"registered-{n}" for n in range(6)],
            "cells": [None] * 42,
            "fixed_score_terminals": [None] * 288,
            "fixed_cells": cases,
        }
    )
    (payload / "sealed-matrix.json").write_bytes(raw)
    m = encoded(
        {
            "run_id": "closure",
            "files": [{"path": "payload/sealed-matrix.json", "bytes": len(raw), "sha256": sha(raw)}],
        }
    )
    (tmp_path / "manifest.json").write_bytes(m)
    write(
        tmp_path / "completion.json",
        {"run_id": "closure", "manifest_sha256": sha(m), "execution_status": "succeeded"},
    )
    closure = {
        "directory": str(payload),
        "run_id": "closure",
        "manifest_sha256": sha(m),
        "matrix_sha256": sha(raw),
    }
    for cell in roster:
        if "terminal_failure" in cell:
            cell["terminal_failure"]["receipt_sha256"] = sha(raw)
    assert check_joint_closure(closure, roster)["matrix_sha256"] == sha(raw)
    roster[-1]["historical"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="do not belong"):
        check_joint_closure(closure, roster)
