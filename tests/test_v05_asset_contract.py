"""Test the v2 upstream source gate, its calibrated resampling and malformed checkpoint rejection."""

import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from b2_asset_fixture import build
from evidence_lens_asset.gaussian import RECOVERY, load
from evidence_lens_asset.permissions import source_grant
from evidence_lens_asset.prepare import inspect
from evidence_lens_bundle.common import BundleError, encoded, sha, write


def test_ordered_grants_follow_each_branch():
    registration = {"schema": "el-upstream-layout-authority-1", "fixed_layouts": {"M2": ["24", "22"], "M4": ["24", "23", "17", "22"]},
                    "optional_pool": ["17", "23", "31"]}
    failure = {"schema": "el-upstream-optional-choice-1", "registration_sha256": sha(encoded(registration)),
               "status": "failed", "reason": "registered probe failed", "selected_ids": None}
    assert source_grant(registration, "fixed-M4", failure) == ["24", "23", "17", "22"]
    assert source_grant(registration, "optional-M4", failure) == ["24", "22"]
    choice = {**failure, "status": "selected", "reason": None, "selected_ids": ["17", "23"]}
    assert source_grant(registration, "optional-M4", choice) == ["24", "17", "23", "22"]
    with pytest.raises(BundleError, match="selected pair"):
        source_grant(registration, "optional-M4", {**choice, "selected_ids": ["23", "22"]})
    with pytest.raises(BundleError, match="sealed optional"):
        source_grant(registration, "optional-M4", {**choice, "registration_sha256": "f" * 64})


def test_original_camera_to_analysis_sources(tmp_path):
    build(tmp_path)
    path = tmp_path / "plan.json"
    plan = json.loads(path.read_bytes())
    reg = {"schema": "el-upstream-layout-authority-1", "fixed_layouts": {"M2": ["41", "5"], "M4": ["41", "70", "109", "5"]},
           "optional_pool": []}
    plan.update(schema="el-asset-view-plan-2", analysis_profile="analysis-448x252-v1",
                source_authorization={"branch": "fixed-M2", "registration": reg, "registration_sha256": sha(encoded(reg))})
    write(path, plan)
    args = (tmp_path / "colmap/text", tmp_path / "images", tmp_path / "project.npz", path, "text", "evidence-lens-npz-v1")
    stats, cameras, images, _ = inspect(*args)
    assert [*images] == ["41", "5"]
    assert set(stats["images"]) == set(images)
    assert stats["grids"]["analysis_size"] == [448, 252]
    for view, photo in images.items():
        with Image.open(io.BytesIO(photo)) as image:
            assert image.size == (448, 252)
        rec = stats["images"][view]
        assert sha((tmp_path / "images" / rec["name"]).read_bytes()) == rec["sha256"]
        assert sha(photo) == rec["analysis_png_sha256"]
    for name, camera in cameras["views"].items():
        grid = stats["grids"]["views"][name]
        np.testing.assert_allclose(camera["K"], np.asarray(grid["original_to_analysis"]) @ np.asarray(grid["original_K"]))
    assert not (tmp_path / "images/unobserved").exists()
    plan["sources"].reverse()
    write(path, plan)
    with pytest.raises(BundleError, match="order exceeds"):
        inspect(*args)


def test_native_count_and_recipe_rejections(tmp_path):
    torch = pytest.importorskip("torch")
    params = {k: torch.zeros((2, dim), dtype=torch.float32) for k, dim in (("means", 3), ("colors", 3), ("scales", 3), ("quats", 4))}
    params["quats"][:, 0] = 1
    params["opacities"] = torch.zeros(2)
    recipe = {"topology": {"cap": 20}, "eps2d": .3, "checkpoints": [5]}
    point = {"schema": "el-recovery-checkpoint-1", "step": 5, "recipe": recipe, "params": params}
    path = tmp_path / "native.pt"
    torch.save(point, path)
    meta = {}
    assert len(load(path, RECOVERY, 1, metadata=meta)["means"]) == 2
    assert meta["registered_cap"] == 20 and meta["eps2d"] == .3
    for malformed in (None, [], {**recipe, "topology": []}, {**recipe, "topology": {"cap": 0}},
                      {**recipe, "topology": {"cap": True}}, {**recipe, "checkpoints": [True]},
                      {**recipe, "eps2d": float("nan")}):
        torch.save({**point, "recipe": malformed}, path)
        with pytest.raises(BundleError, match="recipe|cap"):
            load(path, RECOVERY, 1)
    torch.save({**point, "params": {**params, "opacities": torch.zeros(1)}}, path)
    with pytest.raises(BundleError, match="shape"):
        load(path, RECOVERY, 1)
