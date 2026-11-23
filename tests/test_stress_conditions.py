"""Small original fixtures for the preregistered two held-out interventions."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from inspect_sharp_ply import SUPPLEMENTS, VERTICES, read_ply
from stress_condition import COLOUR_FIELDS, half_source, round_colour, sha


def test_box_ties_and_pixel_centres():
    image = np.zeros((2, 4, 3), np.uint8)
    image[0, :2] = [1, 3, 5]
    camera = {"width": 4, "height": 2, "K": [[4, 0, 1.5], [0, 4, .5], [0, 0, 1]],
              "K_edge": [[4, 0, 2], [0, 4, 1], [0, 0, 1]], "c2w": np.eye(4).tolist()}
    cameras = {"schema": "el-camera-1", "pixel_center": "integer", "views": {"source": camera, "v01": camera}}
    down, changed = half_source(image, cameras)
    np.testing.assert_array_equal(down[0], [[0, 2, 2], [0, 0, 0]])
    np.testing.assert_array_equal(changed["views"]["source"]["K"], [[2, 0, .5], [0, 2, 0], [0, 0, 1]])
    assert changed["views"]["v01"] == camera and camera["width"] == 4


def small_ply(path, value):
    header = ["ply", "format binary_little_endian 1.0", "element vertex 3"]
    header += ["property float " + field for field in VERTICES]
    vertex = np.arange(3 * len(VERTICES), dtype="<f4").reshape(3, -1) * np.float32(.12)
    vertex[:, 3] = value
    extras = b""
    for key, n, decl, dtype in SUPPLEMENTS:
        header.extend([f"element {key} {n}", f"property {decl} {key}"])
        extras += np.zeros(n, dtype=dtype).tobytes()
    path.write_bytes(("\n".join([*header, "end_header"]) + "\n").encode() + vertex.tobytes() + extras)


@pytest.mark.parametrize("value,overflow", [(0.1234567, False), (1e10, True)])
def test_q1_preserves_raw_noncolour_fields_and_rejects_overflow(tmp_path, value, overflow):
    ply, generation = tmp_path / "base.ply", tmp_path / "generation.json"
    small_ply(ply, value)
    original = sha(ply)
    generation.write_text(json.dumps({"schema": "el-generation-1", "producer_run_id": "fixture",
                                      "ply_bytes": ply.stat().st_size, "ply_sha256": original, "gaussian_count": 3,
                                      "generator_commit": "c" * 40, "checkpoint_sha256": "a" * 64,
                                      "camera_sha256": "b" * 64, "input_sha256": "c" * 64}))
    out = tmp_path / "out"
    out.mkdir()
    if overflow:
        with pytest.raises(ValueError, match="overflow"):
            round_colour(ply, generation, out, "fixture-test")
        assert not (out / "generation.json").exists()
    else:
        round_colour(ply, generation, out, "fixture-test")
        before, _ = read_ply(ply)
        after, _ = read_ply(out / "scene.ply")
        for name in VERTICES:
            want = before[name].astype("<f2").astype("<f4") if name in COLOUR_FIELDS else before[name]
            np.testing.assert_array_equal(after[name], want)
        meta = json.loads((out / "generation.json").read_text())
        assert meta["forward_calls_in_this_stage"] == 0 and meta["ply_sha256"] == sha(out / "scene.ply")
    assert original == sha(ply)
