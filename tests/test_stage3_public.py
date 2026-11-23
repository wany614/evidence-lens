"""Publication evidence, alpha-independent RGB labels, and true scope controls."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage3_public_cpu as public
import stage3_release as release


def test_independent_reference_native_grid_including_undefined(tmp_path):
    v = np.ones((256, 256), bool)
    e = v.copy()
    y = np.zeros_like(v, dtype=np.uint8)
    ref = np.zeros((256, 256, 3), np.float32)
    pred = ref.copy()
    ref[0, 0] = np.nan
    pred[0, 1] = 1.01
    pred[0, 2] = 0.05
    pred[0, 3] = 0.08
    v[0, 0] = False
    e[0, :2] = False
    y[0, :2] = 255
    y[0, 3] = 1
    mae = np.zeros_like(v, dtype=np.float32)
    mae[0, :2] = np.nan
    mae[0, 2] = 0.05
    mae[0, 3] = 0.08
    np.savez_compressed(tmp_path / "ref.npz", rgb=ref)
    np.savez_compressed(tmp_path / "prediction.npz", rgb=pred)
    np.savez_compressed(tmp_path / "expected.npz", V=v, E=e, Y=y, rgb_mae=mae)
    row = {"V": 65535, "E": 65534, "Y": 1}
    assert (
        public.check_error(tmp_path / "ref.npz", tmp_path / "prediction.npz", tmp_path / "expected.npz", row)
        == row
    )
    y[0, 2] = 1
    np.savez_compressed(tmp_path / "altered.npz", V=v, E=e, Y=y, rgb_mae=mae)
    with pytest.raises(ValueError, match="RGB error/label"):
        public.check_error(tmp_path / "ref.npz", tmp_path / "prediction.npz", tmp_path / "altered.npz", row)


def test_native_comparison_reports_threshold_crossing_and_refuses_support_loss():
    x = np.array([[0.25, 0.50001, np.nan]], np.float32)
    y = np.array([[0.25, 0.49999, np.nan]], np.float32)

    def vals(a):
        v = np.isfinite(a)
        return {"risk": a, "valid": v, "state": np.where(v, a >= 0.5, 255).astype(np.uint8)}

    row = public.comparison(vals(x), vals(y), "M2/v01", 0.0001, False)
    assert row["threshold_crossings"] == 1 and row["unknown"] == 1 and row["equal_values"] == 1
    with pytest.raises(ValueError, match="threshold"):
        public.comparison(vals(x), vals(y), "M2/v01", 0.0001, True)
    with pytest.raises(ValueError, match="support"):
        public.comparison(vals(np.nan_to_num(x)), vals(y), "M2/v01", 0.0001, False)


def test_release_notes_show_unavailable_and_real_defined_denominators():
    metric = {"mean": None, "defined_families": 0, "planned_families": 4}
    summary = {
        "dispositions": {"planned_target_views": 48, "returned_target_views": 33},
        "family_equal_overall": [
            {
                "condition": c,
                "scope": "all",
                "method": m,
                "common_ap": metric,
                "low_coverage": dict(metric, mean=0.3, defined_families=2),
                "low_error": metric,
            }
            for c in release.CONS
            for m in release.METHODS
        ],
    }
    candidate = {
        "default_method": "MS-L1",
        "assets": {"cpu": {"index_sha256": "a" * 64}, "example": {"index_sha256": "b" * 64}},
    }
    notes = release.release_notes(summary, candidate).decode()
    assert (
        "33 of 48" in notes
        and "undefined (0/4)" in notes
        and "0.3000 (2/4)" in notes
        and notes.count("| M4 |") == 4
    )
    assert "a" * 64 in notes and "b" * 64 in notes


def test_comparator_independently_checks_data_and_states(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from stage3_neural_evidence import compare_maps

    one = np.zeros((256, 256), np.float32)
    valid = np.ones_like(one, bool)
    state = np.zeros_like(one, np.uint8)
    one[7, 9] = np.nan
    valid[7, 9] = False
    state[7, 9] = 255
    np.savez_compressed(tmp_path / "good.npz", risk=one, valid=valid, state=state)
    entry = compare_maps(tmp_path / "good.npz", tmp_path / "good.npz", 0.5)
    assert entry["valid"] == 65535 and entry["max_abs_error"] == 0
    state[0, 0] = 1
    np.savez_compressed(tmp_path / "bad.npz", risk=one, valid=valid, state=state)
    with pytest.raises(ValueError, match="threshold decision"):
        compare_maps(tmp_path / "bad.npz", tmp_path / "good.npz", 0.5)


def test_single_view_outputs_exact_native_png_and_threshold(tmp_path):
    from types import SimpleNamespace
    from PIL import Image

    risk = np.full((256, 256), np.float32(0.0))
    risk[0, 1] = 1
    risk[0, 2] = np.nan
    risk[0, 3] = 0.5
    valid = np.isfinite(risk)
    state = np.where(valid, risk >= 0.5, 255).astype(np.uint8)
    fake = SimpleNamespace(
        meta={"version": "fixture"},
        specs={"MS-L1": {"threshold": 0.5}},
        score=lambda *args: {"risk": risk, "valid": valid, "state": state},
    )
    case = SimpleNamespace(sha256="c" * 64, condition="M2", generator=("s00", "s03"))
    rec = public.risk_output(fake, case, "MS-L1", "v01", "one", tmp_path)
    assert set(rec["images"]) == {"risk", "state"} and rec["display"]["spatial_resampling"] is False
    for n, e in rec["images"].items():
        assert public.member(tmp_path.resolve(), e) == tmp_path / (n + ".png")
    with Image.open(tmp_path / "state.png") as im:
        assert im.size == (256, 256) and [im.getpixel((i, 0)) for i in range(4)] == [
            (75, 168, 184),
            (230, 99, 67),
            (113, 118, 127),
            (230, 99, 67),
        ]
    with Image.open(tmp_path / "risk.png") as im:
        assert im.size == (256, 256) and [im.getpixel((i, 0)) for i in range(3)] == [
            (23, 71, 96),
            (207, 63, 68),
            (113, 118, 127),
        ]


def test_terminal_recovery_rejects_new_updates_and_unrelated_weight_bytes():
    from stage3_terminal_recovery import AFTER, BEFORE, EXPERIMENT, FAILED, STATE, WEIGHTS, recovery_identity

    run = {"run_kind": "derivation", "experiment_id": EXPERIMENT}
    fit = {
        "method": "MS-frozen",
        "seed": 1729,
        "optimizer_steps": 288,
        "optimizer_steps_in_this_record": 0,
        "optimization_parent": FAILED,
        "weights_sha256": WEIGHTS,
    }
    index = {
        "worker_zip_sha256": AFTER,
        "optimization_worker_zip_sha256": BEFORE,
        "optimization_parent": FAILED,
    }
    proof = {
        "optimization_parent": FAILED,
        "optimizer_steps_in_this_record": 0,
        "terminal_checkpoint_sha256": STATE,
        "original_weights_sha256": WEIGHTS,
        "verified_terminal_tensors": 181,
        "original_all_scope_maps": 48,
        "source_equivalence": {
            "original_optimizer_worker_sha256": BEFORE,
            "corrected_export_worker_sha256": AFTER,
        },
    }
    recovery_identity(run, fit, index, proof)
    for changed in (
        {"optimizer_steps_in_this_record": 1},
        {"optimizer_steps": 0},
        {"seed": 2718},
        {"weights_sha256": "0" * 64},
    ):
        with pytest.raises(ValueError, match="lineage"):
            recovery_identity(run, dict(fit, **changed), index, proof)
    with pytest.raises(ValueError, match="lineage"):
        recovery_identity(run, fit, index, dict(proof, terminal_checkpoint_sha256="1" * 64))
    with pytest.raises(ValueError, match="lineage"):
        recovery_identity(dict(run, run_kind="model_fit"), fit, index, proof)
