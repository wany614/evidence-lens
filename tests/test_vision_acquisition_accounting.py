import numpy as np
import pytest

from evidence_lens.scoring_io import file_hash
from test_vision_data import seal_fixture
from vision_public_study import acquisition_accounting


def test_repeated_generation_failed_attempt_and_reused_inputs(tmp_path):
    store = tmp_path / "store"
    roots = (store, tmp_path / "shelf")
    native = tmp_path / "native.npz"
    np.savez(native, rgb=np.zeros((2, 2, 3), np.float32))
    g = {"run_id": "20030101T000000000000Z-0000000000000010", "manifest_sha256": "1" * 64}
    retry = {"run_id": "20030101T000000000000Z-0000000000000011", "manifest_sha256": "2" * 64}

    def render(n, wrong=False, missing_bytes=False):
        filenames = ("source", *(f"v{i:02}" for i in range(1, 7)))
        views = {v: {"sha256": file_hash(native), "bytes": native.stat().st_size} for v in filenames}
        if wrong:
            views["source"]["sha256"] = "c" * 64
        if missing_bytes:
            del views["source"]["bytes"]
        return seal_fixture(
            store,
            f"20030101T000000000000Z-{n:016x}",
            "render",
            "succeeded",
            {},
            {
                "payload/prediction.json": {"views": views},
                **{"payload/" + v + ".npz": native for v in filenames},
            },
        )

    r1, r2, no, missing = render(1), render(2), render(3, True), render(4, missing_bytes=True)

    def c(cond, pin, cohort="new_confirmation"):
        return {
            "cohort": cohort,
            "family": "synthetic_family",
            "condition": cond,
            "status": "selected",
            "roles": {"generation": g, "prediction": pin},
        }

    rows = [
        c("B0", r1),
        c("Q1", r2),
        {"cohort": "new_confirmation", "family": "synthetic_family", "condition": "R1", "status": "failed"},
        c("B0", r1, "historical_regression"),
    ]
    history = [
        {"cohort": "new_confirmation", "stage": kind, "run_id": p["run_id"]}
        for kind, p in (
            ("generation", g),
            ("generation", g),
            ("generation", retry),
            ("render", r1),
            ("render", r2),
        )
    ]
    cost = {
        p["run_id"]: {"run_id": p["run_id"], "run_kind": k, "status": status, "elapsed_seconds_monotonic": t}
        for p, k, status, t in (
            (g, "generation", "succeeded", 5.0),
            (retry, "generation", "failed", 1.0),
            (r1, "render", "succeeded", 1.1),
            (r2, "render", "succeeded", 1.2),
            (no, "render", "succeeded", 1.5),
            (missing, "render", "succeeded", 1.5),
        )
    }
    sampling = lambda f, sec: {
        "selected_unique_runs": f,
        "attempts_including_failures": f,
        "attempts_with_wall_reading": f,
        "sum_attempt_wall_seconds": sec,
    }
    led = [
        {
            "scope": "preopening_resources",
            "receipt_sha256": "b" * 64,
            "learning_summary": [
                {
                    "origin": "v11_new",
                    "families": 2,
                    "prediction_views": 14,
                    "workers": {k: sampling(2, 10.0) for k in ("generation", "render")},
                },
                {
                    "origin": "original_learning",
                    "families": 4,
                    "prediction_views": 28,
                    "workers": {k: sampling(4, 87.0) for k in ("generation", "render")},
                },
            ],
        }
    ]
    result = acquisition_accounting(roots, led, rows, history, cost)
    assert result["new_acquisition_conditions"] == 5 and result["preopening_reused_B0"] == 4
    assert result["selected_native_prediction_views"] == 28
    data = result["workers"]["generation"]
    assert data["confirmation_selected_runs"] == 1 and data["confirmation_unsuccessful_attempts"] == 1
    assert data["new_acquisition_attempts"] == 4 and data["new_acquisition_attempt_wall_seconds"] == 16.0
    assert np.isclose(result["workers"]["render"]["new_acquisition_attempt_wall_seconds"], 12.3)
    assert data["new_acquisition_attempts_with_wall_reading"] == 4
    cost[retry["run_id"]]["elapsed_seconds_monotonic"] = None
    gstat = acquisition_accounting(roots, led, rows, history, cost)["workers"]["generation"]
    assert gstat["new_acquisition_attempt_wall_seconds"] == 15.0
    assert gstat["new_acquisition_attempts_with_wall_reading"] == 3
    assert gstat["new_acquisition_attempts_missing_wall_reading"] == 1
    assert acquisition_accounting(roots, [], rows, history, cost, allow_fixture_without_learning=True) is None
    with pytest.raises(ValueError, match="pre-opening acquisition resources"):
        acquisition_accounting(roots, [], rows, history, cost)
    with pytest.raises(ValueError, match="duplicate new confirmation"):
        acquisition_accounting(roots, led, [*rows, rows[0]], history, cost)
    with pytest.raises(ValueError, match="original byte count"):
        acquisition_accounting(roots, led, [c("B0", missing)], history, cost)
    with pytest.raises(ValueError, match="native view receipt differs"):
        acquisition_accounting(roots, led, [c("B0", no)], history, cost)
    # A one-byte change to the actual NPZ preserves both signed indexes and its byte count.
    source = store / "runs" / r1["run_id"] / "payload/source.npz"
    raw = source.read_bytes()
    source.unlink()
    source.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="(checksum|hash|changed|mismatch|identity)"):
        acquisition_accounting(roots, led, [c("B0", r1)], history, cost)
