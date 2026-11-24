import csv
from types import SimpleNamespace

import numpy as np
import pytest

from evidence_lens_bundle.replay import LatestQueue, Sequence, replay, words
from evidence_lens_bundle.replay_sidecar import SCHEMA, optional_metadata
from evidence_lens_bundle.resident import load_scorer
from evidence_lens_bundle.scoring import calculate, models
from test_bundle import fixture


def test_latest_queue_discards_pending_and_expires_a_superseded_inflight_result():
    q = LatestQueue()
    a = SimpleNamespace(sequence_id="old", version_key="first-bound-snapshot")
    b = SimpleNamespace(sequence_id="new", version_key="second-bound-snapshot")

    def item(index, seq):
        return seq, "view", {"frame_id": index, "input_ready_time": 0.0}

    q.offer(item(1, a))
    in_service = q.take()
    q.offer(item(2, a))
    q.offer(item(3, b))
    assert q.rows[1]["status"] == "dropped" and q.rows[1]["replaced_by"] == 3
    assert not q.finish(in_service, in_service[2]["start_time"] + 0.01, float("inf"))
    assert in_service[2]["reason"] == "superseded_map_or_source"
    last = q.take()
    q.close()
    assert q.take() is None
    assert not q.finish(last, last[2]["start_time"] + 0.01, 0.001)
    assert last[2]["reason"] == "result_age_limit"


def test_clock_covariance_and_mask_provenance_are_stored_without_reading_files():
    data = {
        "schema": SCHEMA,
        "clock_domain": "test clock",
        "acquisition_timestamp": 123.0,
        "pose_covariance": {
            "coordinates": "camera tangent x y z roll pitch yaw",
            "units": "m and rad",
            "matrix": np.eye(6).tolist(),
        },
        "dynamic_mask": {
            "source": "manual",
            "reference": "absent-mask.npy",
            "time": None,
            "clock_domain": None,
        },
    }
    result = optional_metadata(data)
    assert result["dynamic_mask"] == data["dynamic_mask"]
    assert result["pose_covariance"] == data["pose_covariance"]
    data["pose_covariance"]["matrix"][0][0] = 3.0
    assert result["pose_covariance"]["matrix"][0][0] == 1.0
    assert optional_metadata()["pose_covariance"] is None
    with pytest.raises(ValueError, match="unsupported"):
        optional_metadata({"schema": SCHEMA, "target_reference": "unopened.png"})
    with pytest.raises(ValueError, match="clock domain"):
        optional_metadata({"schema": SCHEMA, "acquisition_timestamp": 12.0})


def test_scored_replay_matches_file_calculation_with_unknown_denominator(tmp_path):
    root = tmp_path / "data"
    fixture(root)
    seq = Sequence(root, sequence_id="fixture")
    model, spec = models("MS-L1", None, 4)
    expected = {
        (seq.sequence_id, v): words(
            calculate(seq.bundle, v, seq.frame(v), seq.sources, "MS-L1", model, spec["threshold"])
        )
        for v in seq.bundle.targets
    }
    result = replay(
        [seq],
        [(0, seq)],
        load_scorer(),
        "MS-L1",
        tmp_path / "out",
        hz=4,
        duration=0.5,
        max_age=4,
        expected=expected,
        sidecar={
            "schema": SCHEMA,
            "source_selection_policy": "predeclared original source roster",
            "views": {
                "fixture": {
                    seq.bundle.targets[0]: {"clock_domain": "source clock", "acquisition_timestamp": 42.5}
                }
            },
        },
    )
    assert result["accepted"] == result["status_counts"]["finished"] == 2
    assert result["pixel_denominators"]["canvas"] == 2 * 256 * 256
    assert result["pixel_denominators"]["unknown"] >= 256 * 256
    assert result["sidecar"]["acquisition_timestamp"] is None
    with (tmp_path / "out" / "frames.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["acquisition_timestamp"] == "42.5" and not rows[1]["acquisition_timestamp"]
    assert rows[0]["sidecar_sha256"] != rows[1]["sidecar_sha256"]
    assert (tmp_path / "out" / "replay.html").is_file()
