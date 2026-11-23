"""Portable prediction-only renderings from tiny in-memory score products."""

import json

import numpy as np
import pytest
from PIL import Image

from evidence_lens.l1 import KEYS, risk_state
from evidence_lens.score_maps import export
from evidence_lens.scoring_io import TARGETS, file_hash, write_json


def fixture(directory, *, state_error=None):
    score = np.array([[0, 0.5, 1, np.nan]], np.float32)
    state = risk_state(score, 0.5)
    valid = np.isfinite(score)
    if state_error == "code":
        state[0, 0] = 18
    if state_error == "threshold tie":
        state[0, 1] = 0
    if state_error == "valid":
        valid[0, -1] = True
    if state_error == "shape":
        valid = valid.T
    if state_error == "range":
        score[0, 1] = 2
    if state_error == "infinity":
        score[0, -1] = np.inf
    np.savez(directory / "v01.npz", l1_full=score, l1_full_valid=valid, l1_full_state=state)
    row = {"sha256": file_hash(directory / "v01.npz"), "l1_scoreable": 3, "status": "scored"}
    receipt = {
        "schema": "el-l1-score-1",
        "run_id": "synthetic",
        "model_id": "fixture-model",
        "model_files": {"model_index": {"sha256": "1" * 64}},
        "inputs": {"input_png": {"sha256": "2" * 64}, "cameras": {"sha256": "3" * 64}},
        "methods": KEYS,
        "state_codes": {"0": "low", "1": "high", "255": "unknown"},
        "thresholds": {m: 0.5 for m in KEYS if m != "L0"},
        "views": {v: dict(row) for v in TARGETS},
        "status": "scored",
    }
    write_json(directory / "l1_score.json", receipt)
    return receipt


def test_only_selected_view_is_read_and_indexed_pngs_roundtrip(tmp_path):
    fixture(tmp_path)
    dest = tmp_path / "portable"
    receipt_sha = file_hash(tmp_path / "l1_score.json")
    result = export(tmp_path, dest, view="v01", receipt_sha256=receipt_sha)
    metadata = json.loads((dest / "v01.json").read_text())
    assert list(result["views"]) == ["v01"]
    assert result["views"]["v01"]["sha256"] == file_hash(dest / "v01.json")
    assert metadata["source"]["receipt"]["sha256"] == receipt_sha
    assert metadata["source"]["scores"]["sha256"] == file_hash(tmp_path / "v01.npz")
    assert metadata["counts"] == {"total": 4, "low": 1, "high": 2, "valid": 3, "unknown": 1}
    assert metadata["width"] == 4 and metadata["height"] == 1 and metadata["method"] == "L1-full"
    for field, codes in (("continuous", [0, 127, 254, 255]), ("state", [0, 1, 1, 255])):
        p = dest / metadata["maps"][field]["file"]
        assert metadata["maps"][field]["sha256"] == file_hash(p)
        with Image.open(p) as im:
            assert im.mode == "P" and np.asarray(im).tolist() == [codes]
            assert im.convert("RGB").getpixel((3, 0)) == (113, 118, 127)
            if field == "continuous":
                assert im.convert("RGB").getpixel((0, 0)) == (34, 49, 79)
                assert im.convert("RGB").getpixel((2, 0)) == (233, 104, 49)
    for output in (dest, tmp_path):
        with pytest.raises(FileExistsError):
            export(tmp_path, output, view="v01")
    with pytest.raises(ValueError, match="supplied SHA"):
        export(tmp_path, tmp_path / "bad-receipt", view="v01", receipt_sha256="0" * 64)
    with (tmp_path / "v01.npz").open("ab") as fp:
        fp.write(b"changed")
    with pytest.raises(ValueError, match="receipt SHA"):
        export(tmp_path, tmp_path / "bad-file", view="v01")
    assert not (tmp_path / "bad-file").exists()


@pytest.mark.parametrize("error", ["shape", "valid", "code", "threshold tie", "range", "infinity"])
def test_invalid_numeric_product_rejected_even_with_matching_digest(tmp_path, error):
    fixture(tmp_path, state_error=error)
    with pytest.raises(ValueError):
        export(tmp_path, tmp_path / "out", view="v01")
    assert not (tmp_path / "out").exists()


def test_declared_unsupported_dimensions_do_not_invent_an_image(tmp_path):
    receipt = fixture(tmp_path)
    receipt["status"] = "unsupported"
    receipt["views"] = {v: {"status": "unknown"} for v in TARGETS}
    (tmp_path / "v01.npz").unlink()
    write_json(tmp_path / "l1_score.json", receipt)
    index = export(tmp_path, tmp_path / "unknown")
    assert set(index["views"]) == set(TARGETS)
    assert {r["map_status"] for r in index["views"].values()} == {"unavailable_dimensions"}
    assert not list((tmp_path / "unknown").glob("*.png"))
