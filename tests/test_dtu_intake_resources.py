"""Portable regressions for DTU off-axis readback and resource accounting."""

import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dtu_receive
import dtu_resources
from evidence_lens._bundle import sha


def signed(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return sha(path)


def test_off_axis_probe_authenticates_numeric_result_and_pose(tmp_path):
    k = [[241.0, 0.0, 113.1], [0.0, 241.0, 145.6], [0.0, 0.0, 1.0]]
    pose = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    camera = tmp_path / "camera.json"
    camera_sha = signed(camera, {"views": {"s00": {"K": k, "c2w": pose}}})
    pred_sha = signed(tmp_path / "prediction.json", {"source": "test"})
    depth_sha = signed(tmp_path / "depth-probe.json", {"passed": True})
    sample = {
        "actual_ED_m": 0.73,
        "expected_ED_m": 0.73,
        "actual_D_m": 0.5256,
        "expected_D_m": 0.5256,
        "actual_alpha": 0.72,
        "expected_alpha": 0.72,
        "neighbour_max_alpha": 0.68,
        "actual_linear_rgb": [0.2, 0.45, 0.62],
        "expected_linear_rgb": [0.2, 0.45, 0.62],
    }
    probed = {
        "passed": True,
        "measured_empty_alpha": 0,
        "integer_K": k,
        "c2w": pose,
        "measured_gsplat_edge_K": [[241.0, 0.0, 113.6], [0.0, 241.0, 146.1], [0.0, 0.0, 1.0]],
        "cases": [sample.copy(), sample.copy()],
    }
    report = {
        "schema": "el-dtu-native-gpu-probe-1",
        "passed": True,
        "render_camera_sha256": camera_sha,
        "prediction_sha256": pred_sha,
        "original_C_probe_sha256": depth_sha,
        "cameras": {"s00": probed},
    }
    job = {"files": {"render_cameras.json": {"sha256": camera_sha}}}
    signed(tmp_path / "dtu-camera-probe.json", report)
    dtu_receive.camera_probe(tmp_path, camera, job)
    probed["cases"][0]["actual_ED_m"] = 0.41
    signed(tmp_path / "dtu-camera-probe.json", report)
    with pytest.raises(ValueError, match="expectation"):
        dtu_receive.camera_probe(tmp_path, camera, job)
    probed["cases"][0]["actual_ED_m"] = 0.73
    probed["measured_gsplat_edge_K"][0][2] = 113.1
    signed(tmp_path / "dtu-camera-probe.json", report)
    with pytest.raises(ValueError, match="calibration"):
        dtu_receive.camera_probe(tmp_path, camera, job)


def test_stage_rollup_keeps_failed_status_and_different_memory_peaks(tmp_path):
    base = {
        "work_package": "D1",
        "stage": "render",
        "scope": "primary_256",
        "platform": "Windows",
        "status": "succeeded",
        "recorder_elapsed_s": 1.5,
        "payload_logical_B": 32,
        "sampled_process_tree_rss_peak_B": 300,
        "sampled_process_gpu_peak_B": None,
        "sampled_device_wide_vram_peak_MiB": 89.0,
        "torch_allocated_peak_B": 11,
        "torch_reserved_peak_B": 64,
        "linux_container_peak_B": None,
    }
    dtu_resources.summarize(
        tmp_path,
        [
            base,
            {**base, "recorder_elapsed_s": 2.5, "sampled_process_tree_rss_peak_B": 150},
            {**base, "status": "failed", "recorder_elapsed_s": 1.0},
        ],
    )
    with (tmp_path / "stages.csv").open() as stream:
        rows = {r["status"]: r for r in csv.DictReader(stream)}
    assert set(rows) == {"succeeded", "failed"}
    good = rows["succeeded"]
    assert good["runs"] == "2" and good["sum_of_recorded_stage_wall_s"] == "4.0"
    assert good["sampled_process_tree_rss_peak_B"] == "300" and good["sampled_process_gpu_peak_B"] == ""
    assert good["torch_allocated_peak_B"] == "11" and good["torch_reserved_peak_B"] == "64"
    assert rows["failed"]["runs"] == "1"


def test_archive_receipt_requires_local_and_remote_container_identity(tmp_path):
    rid = "20261123T194437433991Z-8fac72aeaecb4df2"
    archive = {"sha256": "a" * 64, "bytes": 123}
    local = {
        "run_id": rid,
        "manifest_sha256": "b" * 64,
        "container_sha256": archive["sha256"],
        "bytes": 123,
        "status": "accepted",
        "elapsed_seconds": 2.3,
        "local_input_storage": {"linked_input_bytes": 34},
    }
    receipt = {
        "run_id": rid,
        "manifest_sha256": local["manifest_sha256"],
        "archive_receive": local,
        "container_sha256": archive["sha256"],
        "container_bytes": 123,
    }
    dest = tmp_path / "batch" / "M2" / "transfers" / (rid + ".json")
    dest.parent.mkdir(parents=True)
    spec = {"condition": "M2", "sha256": signed(dest, receipt)}
    batch = {"batch": "batch", "archive_receipts": {rid: spec}}
    got = dtu_resources.archive_readback(tmp_path, batch, rid + ".zip", archive)
    assert got["mac_archive_elapsed_s"] == 2.3 and got["mac_linked_input_B"] == 34
    with pytest.raises(ValueError, match="returned bytes"):
        dtu_resources.archive_readback(tmp_path, batch, rid + ".zip", {**archive, "sha256": "c" * 64})
    receipt["archive_receive"]["elapsed_seconds"] = 0
    signed(dest, receipt)
    with pytest.raises(ValueError, match="Changed Mac"):
        dtu_resources.archive_readback(tmp_path, batch, rid + ".zip", archive)
