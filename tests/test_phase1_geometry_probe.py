"""Independent analytic witnesses. These tests make no GPU or learned-prediction claim."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("el_probe_oracle", ROOT / "scripts/phase1_geometry_oracle.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
REGISTRY = ROOT / "protocol/phase1-geometry-probe-v1.json"
FROZEN = "fa16f5a196d8cd95b61e01288537ca016c402b1d6715744e9eed77d11e5dafb6"


def test_official_ply_image_metadata_writes_numpy_scalars(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import phase1_geometry_probe as probe

    official = SimpleNamespace(
        resolution_px=(np.uint32(95), np.uint32(63)),
        focal_length_px=np.float32(106.25),
        color_space="linearRGB",
    )
    dest = tmp_path / "ply_roundtrip.json"
    probe.write(dest, {"asymmetric": probe.ply_image_metadata(official)})
    assert probe.read(dest)["asymmetric"] == {
        "resolution_wh": [95, 63],
        "f_px": 106.25,
        "color_space": "linearRGB",
    }


def test_registry_geometry_and_layers():
    data = json.loads(REGISTRY.read_bytes())
    assert hashlib.sha256(REGISTRY.read_bytes()).hexdigest() == FROZEN
    assert set(data["scenes"][0]["expected"]) == set(oracle.NAMES)
    assert set(data["scenes"][1]["expected"]) == set(oracle.NAMES)
    for scene in data["scenes"]:
        source = scene["cameras"]["source"]
        for cam, v in scene["cameras"].items():
            for p, expect in zip(data["means_m"], scene["expected"][cam], strict=True):
                # A separate homogeneous reference supplies a second check on the physical-basis construction.
                q = np.linalg.solve(v["c2w"], np.asarray(source["c2w"]) @ np.r_[p, 1])
                u, vv, z = np.asarray(v["K"]) @ q[:3]
                assert np.allclose(expect["uv"], [u / z, vv / z], atol=9e-5, rtol=0)
                assert abs(expect["z_m"] - q[2]) < 4e-7
    for b, r in zip(
        data["scenes"][0]["expected"]["source"], data["scenes"][1]["expected"]["source"], strict=True
    ):
        assert np.allclose((np.array(b["uv"]) + 0.5) / 2 - 0.5, r["uv"])
        assert b["z_m"] == r["z_m"]
    case = data["layered"]
    a, b = case["opacity"]
    weights = np.array([a, (1 - a) * b])
    z = weights @ case["z_m"]
    assert (
        weights.sum() == case["expected"]["alpha"]
        and z == case["expected"]["D"]
        and z / weights.sum() == case["expected"]["ED"]
    )


def test_opacity_centroid_detects_adversarial_half_pixel():
    yy, xx = np.indices((29, 35))
    centre = np.array([14.13, 16.79])

    def raster(xy):
        x, y = xx - xy[0], yy - xy[1]
        return np.where(
            x * x + y * y < 64, 0.43 * np.exp(-0.4 * x * x - 0.3 * x * y - 0.2 * y * y), 0
        ).astype(np.float32)

    assert np.allclose(oracle.centroid(raster(centre)), centre, atol=3e-6)
    assert np.allclose(oracle.centroid(raster(centre - 0.5)) - centre, [-0.5, -0.5], atol=3e-6)
    with pytest.raises(ValueError):
        oracle.centroid(np.zeros((4, 5), np.float32))
