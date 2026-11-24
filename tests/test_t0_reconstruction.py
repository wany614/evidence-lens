"""Exercise the T0 source-only fallback, its D exception, and strict transport membership."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from t0_worker_pack import dispatch, encoded


def test_dispatch_rejects_other_source_camera_before_model_files(tmp_path):
    ids = ["s00", "s03"]
    paths = {k: tmp_path / (k + ".json") for k in ("initmeta", "rendercameras", "cameras", "recipe")}
    for name, path in paths.items():
        path.write_bytes(encoded({"views": {s: {} for s in (*ids, "s01")}} if name == "cameras" else {}))
    cfg = {
        "scene_family_id": "set2",
        "condition_id": "M2",
        "generator_source_ids": ids,
        "scientific_trial_id": "t0-frozen-set2-M2",
    }
    with pytest.raises(ValueError, match="Incorrect condition cameras"):
        dispatch(cfg, paths, tmp_path)


def test_unqualified_natural_branch_and_engineering_fixture(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from test_recovery_triangulate import camera_document
    import t0_worker_init as worker

    g = worker.geometry
    protocol = ROOT / "protocol/recovery-initialization-v1.json"
    monkeypatch.setenv("EL_RUN_ID", "20260101T000000000000Z-0123456789abcdef")
    for s in g.SETS["M4"]:
        Image.new("RGB", (256, 256), (33, 44, 55)).save(tmp_path / (s + ".png"))
    digests = []
    for condition, forced in [("M2", False), ("M4", False), ("M2", True)]:
        ids = g.SETS[condition]
        folder = tmp_path / (condition + str(forced))
        folder.mkdir()
        cameras = folder / "cameras.json"
        cameras.write_bytes(encoded(camera_document(ids)))
        sources = [(s, str(tmp_path / (s + ".png"))) for s in ids]
        pixel_hashes = {s: g.sha(f) for s, f in sources}
        cfg = {
            "forced_fallback_fixture": forced,
            "condition_id": condition,
            "scene_family_id": "set1" if forced else "synthetic",
            "trial_id": "t0-d-forced-fallback-set1-M2-init" if forced else "synthetic-only",
            "source_rgb_sha256": pixel_hashes,
            "source_camera_sha256": g.sha(cameras),
            "initialization_protocol_sha256": g.PROTOCOL_SHA,
            "generator_source_ids": list(ids),
            "source_parent": {},
            "registration": {},
            "recipe_sha256": g.PROTOCOL_SHA,
        }
        config = folder / "config.json"
        config.write_bytes(encoded(cfg))
        out = folder / "out"
        out.mkdir()
        previous = tmp_path / "M2False/out/init.json"
        a = SimpleNamespace(
            config=config, protocol=protocol, cameras=cameras, source=sources, out=out, previous=previous
        )
        worker.initialize(a)
        result = g.load(out / "init.json")
        assert (
            result["N"] == result["fallback"]["total"] == 2048
            and result["actual_method"] == "endpoint_stratified_inverse_depth"
        )
        assert (
            not result["eligibility"]["qualified"]
            and "fewer_than_24_distinct_triangulated_tracks" in result["eligibility"]["failed_gates"]
        )
        assert result["fallback"]["samples_per_endpoint"] == 1024
        with np.load(out / "init.npz", allow_pickle=False) as data:
            assert set(data) == set(g.NAMES) and all(
                a.dtype == np.dtype("<f4") and np.isfinite(a).all() for a in data.values()
            )
            assert data["means"].shape == (2048, 3) and len(np.unique(data["means"], axis=0)) == 2048
        digests.append(result["parameter_sha256"])
    assert len(set(digests)) == 1
    cfg["scene_family_id"] = "set2"
    config.write_bytes(encoded(cfg))
    with pytest.raises(ValueError, match="reserved for one D cell"):
        worker.initialize(a)
