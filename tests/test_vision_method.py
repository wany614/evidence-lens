import json

import numpy as np

import vision_method as method
from vision_cases import case_board
from vision_evaluate import TWO
from PIL import Image


def metrics(ft, frozen, lr):
    return {
        s: {"L1-full": {"value": lr[i]}, "L2-frozen": {"value": frozen[i]}, "L2-vision-ft": {"value": ft[i]}}
        for i, s in enumerate(method.SEEDS)
    }


def test_default_uses_all_three_pairs_and_sample_standard_deviation():
    cost = {"hot_p95_seconds": 5.0, "rss_bytes": 2147483648}
    out = method.deploy_rule(metrics([0.81, 0.82, 0.83], [0.70, 0.72, 0.73], [0.8, 0.8, 0.8]), cost)
    assert out["default"] == "L2-vision-ft" and np.allclose(out["paired_ft_minus_frozen"], [0.11, 0.10, 0.10])
    assert np.isclose(out["sample_sd_difference"], np.std([0.11, 0.10, 0.10], ddof=1))
    tied = method.deploy_rule(metrics([0.8, 0.82, 0.83], [0.68, 0.68, 0.68], [0.8, 0.8, 0.8]), cost)
    assert tied["default"] == "L1-full" and not tied["checks"]["ft_exceeds_lr_every_seed"]
    noisy = method.deploy_rule(metrics([0.81, 0.82, 0.83], [0.92, 0.70, 0.83], [0.8, 0.8, 0.8]), cost)
    assert noisy["default"] == "L1-full" and not noisy["checks"]["paired_mean_exceeds_sample_sd"]
    assert (
        method.deploy_rule(metrics([0.9] * 3, [0.7] * 3, [0.8] * 3), {**cost, "hot_p95_seconds": 5.01})[
            "default"
        ]
        == "L1-full"
    )
    assert method.deploy_rule(metrics([0.9, None, 0.9], [0.7] * 3, [0.8] * 3), cost)["default"] == "L1-full"


def test_cpu_ulp_and_threshold_crossings_are_diagnostics():
    before = np.float32(0.5)
    after = np.nextafter(before, np.float32(1.0))
    a = np.array([before, after, np.nan, 0.1], np.float32)
    b = np.array([after, after, 0.2, np.nan], np.float32)
    r = method.cpu_numbers(a, b, (float(before) + float(after)) / 2)
    assert (
        r["common_finite"] == 2
        and r["max_ulps"] == 1
        and r["threshold_crossings"] == 1
        and r["support_disagreement"] == 2
    )


def test_case_panel_uses_distinct_unknown_and_nonevaluable_colors(tmp_path):
    from evidence_lens.comparison_board import RISK_COLORS, INVALID_COLOR

    shape = (3, 3)
    valid = np.ones(shape, bool)
    valid[1, 0] = False
    t = {
        "reference_valid": np.ones(shape, bool),
        "rgb_evaluable": valid,
        "rgb_high_error": np.where(valid, 1, 255).astype(np.uint8),
        "rgb_error": np.where(valid, 0.1, np.nan).astype(np.float32),
        "visibility": np.ones(shape, np.uint8),
    }
    scores = {m: np.full(shape, 0.2, np.float32) for m in ("L1-full", *TWO)}
    for risk in scores.values():
        risk[0, 0] = np.nan
    out = tmp_path / "board"
    rgb = np.zeros((*shape, 3), np.float32)
    hashes = case_board(
        out,
        {"family": "engineering-fixture", "condition": "B0", "cohort": "fixture"},
        Image.fromarray(np.zeros((*shape, 3), np.uint8)),
        {"rgb": rgb},
        rgb,
        t,
        scores,
        {m: 0.3 for m in scores},
        {},
    )
    assert len(hashes) == 13
    with Image.open(out / "L2-vision-ft_state.png") as img:
        a = np.asarray(img)
        assert np.array_equal(a[0, 0], RISK_COLORS["unknown"]) and np.array_equal(a[1, 0], INVALID_COLOR)
    assert json.loads((out / "board.json").read_text())["view"] == "v01"


def test_independent_heavy_ties_one_class_zero_v_and_exact_half_choice():
    import pytest
    from evidence_lens.l1_freeze import threshold_for
    from vision_evaluate import ranking
    from vision_replay import ap_readback, threshold_readback

    rng = np.random.default_rng(843)
    raw = rng.integers(0, 7, size=24576).astype(np.float32) / 8
    y = rng.integers(0, 2, len(raw)).astype(bool)
    for area in (np.ones(len(y), bool), y, ~y, np.zeros(len(y), bool)):
        a, b = ranking(raw[area], y[area]), ap_readback(raw[area], y[area])
        assert (
            a["missing_reason"] == b["missing_reason"] and a["positive"] == b["positive"] and a["n"] == b["n"]
        )
        assert (a["ap"] is None and b["ap"] is None) or np.isclose(a["ap"], b["ap"], rtol=0, atol=1e-13)
    for vals, n in [
        ({"a": raw[:15000], "b": raw[15000:], "empty": raw[:0]}, {"a": 16321, "b": 11018, "empty": 0}),
        ({"a": np.array([0.2] * 6 + [0.8] * 4, np.float32)}, {"a": 10}),
        ({"a": raw[:0]}, {"a": 106}),
        ({"a": np.ones(3000, np.float32)}, {"a": 3000}),
    ]:
        a, b = threshold_for(vals, n), threshold_readback(vals, n)
        assert all(a[k] == v for k, v in b.items())
    with pytest.raises(ValueError):
        threshold_readback({"dead": raw[:0]}, {"dead": 0})


def test_exported_tensor_bytes_and_registered_tensor_identity(tmp_path):
    import hashlib
    import struct
    import pytest
    from vision_replay import verify_tensor_payloads

    a = np.array([0.5, 1.0], dtype="<f4").tobytes()
    b = np.array([3, -4, 7], dtype="<i8").tobytes()
    fields = {
        "__metadata__": {"format": "pt"},
        "frozen": {"shape": [2], "dtype": "F32", "data_offsets": [0, 8]},
        "learned": {"shape": [3], "dtype": "I64", "data_offsets": [8, 32]},
    }
    header = json.dumps(fields).encode()
    header += b" " * (-len(header) % 8)
    path = tmp_path / "weights.safetensors"
    raw = struct.pack("<Q", len(header)) + header + a + b
    path.write_bytes(raw)
    expected = {"frozen": hashlib.sha256(a).hexdigest(), "learned": hashlib.sha256(b).hexdigest()}
    assert verify_tensor_payloads(path, expected) == 2
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="actual saved tensor SHA differs"):
        verify_tensor_payloads(path, expected)
    path.write_bytes(raw + b"x")
    with pytest.raises(ValueError, match="unaccounted"):
        verify_tensor_payloads(path, expected)
