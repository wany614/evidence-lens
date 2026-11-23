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

    def render(n, wrong=False):
        filenames = ("source", *(f"v{i:02}" for i in range(1, 7)))
        views = {v: {"sha256": file_hash(native), "bytes": native.stat().st_size} for v in filenames}
        if wrong:
            views["source"]["sha256"] = "c" * 64
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

    r1, r2, no = render(1), render(2), render(3, True)

    def c(cond, pin, cohort="new_confirmation"):
        return {
            "cohort": cohort,
            "condition": cond,
            "status": "selected",
            "roles": {"generation": g, "prediction": pin},
        }

    rows = [
        c("B0", r1),
        c("Q1", r2),
        {"cohort": "new_confirmation", "condition": "R1", "status": "failed"},
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
        )
    }
    sampling = lambda f, sec: {
        "selected_unique_runs": f,
        "attempts_including_failures": f,
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
    assert acquisition_accounting(roots, [], rows, history, cost) is None
    with pytest.raises(ValueError, match="native view receipt differs"):
        acquisition_accounting(roots, led, [c("B0", no)], history, cost)
