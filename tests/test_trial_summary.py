"""Arithmetic and archive evidence for the P2 readout, using only temporary fixtures."""

import csv
import importlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from evidence_lens._bundle import sha


@pytest.fixture
def summary(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("summarize_trial")


def test_pooled_counts_are_sums_and_keep_unknown_errors(tmp_path, summary):
    rows = []
    for i, v in enumerate(summary.VIEWS):
        size = i + 1
        values = [("output", "prediction_valid_fraction", 5, 6), ("output", "rgb_error_fraction_gt_0.05", 2, 5),
                  ("output", "invalid_output_fraction", 1, 6), ("output", "rgb_mae", 0.4, 5)]
        for method in summary.METHODS:
            values.extend((method, k, n, d) for k, n, d in (
                ("warning_fraction", 2, 5), ("overall_low_risk_coverage", 2, 6), ("unknown_fraction", 1, 5),
                ("high_risk_recall", 1, 2), ("low_risk_error_rate", 0, 2), ("error_fraction_within_unknown", 1, 1),
                ("average_precision_common", None, 4)))
        for m, k, n, d in values:
            rows.append(dict(run_id="fixture", camera_id=v, region_name="all", method_id=m, metric_name=k,
                             numerator=n * size if n is not None else None, denominator=d * size,
                             value=n / d if n is not None else 0.5))
    p = tmp_path / "metrics.csv"
    def write():
        with p.open("w", encoding="utf8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    write()
    totals = summary.result_counts(p, "fixture")["family_sum"]
    assert (totals["V"], totals["E"], totals["Y"], totals["invalid_output"]) == (126, 105, 42, 21)
    assert totals["methods"]["L0"] == {"H": 42, "L": 42, "U": 21, "Y_in_H": 21, "Y_in_L": 0, "Y_in_U": 21}
    for row in rows:
        if row["camera_id"] != "v01":
            continue
        metric = row["metric_name"]
        if metric in ("rgb_error_fraction_gt_0.05", "low_risk_error_rate", "error_fraction_within_unknown"):
            row.update(numerator=0, value=0)
        if metric == "high_risk_recall":
            row.update(numerator=None, denominator=0, value=None)
        if metric == "average_precision_common":
            row["value"] = None
    write()
    empty = summary.result_counts(p, "fixture")["by_view"]["v01"]
    assert empty["Y"] == 0 and empty["E"] == 5 and empty["methods"]["L0"]["common_ap"] is None
    rows[6]["numerator"] = 0
    write()
    with pytest.raises(ValueError, match="inconsistent L0"):
        summary.result_counts(p, "fixture")


def test_projection_counts_shared_once_and_exposes_missing_stage_zip(tmp_path, summary):
    directory = tmp_path / "runs" / "fixture"
    directory.mkdir(parents=True)
    (directory / "payload").mkdir()
    (directory / "payload/generation.json").write_text(json.dumps({"checkpoint_sha256": "fixture"}))
    (directory / "config.resolved.yaml").write_text(json.dumps({"generator": {"checkpoint_sha256": "fixture", "checkpoint_bytes": 11}}))
    checked = {"generation": {"run": {"run_id": "fixture"}, "shared": [{"sha256": "fixture", "bytes": 11}]}}
    stages = {k: {"logical_bytes": 90, "archive": {"bytes": 100}, "artifacts": {}} for k in summary.STAGES}
    stages["generation"]["artifacts"]["payload/scene.ply"] = {"bytes": 13}
    report = summary.budget(stages, checked, tmp_path)
    assert sum(r["count"] for r in report["stages"].values()) == 208
    assert report["stages"]["q1_derivation"]["unit_zip_bytes"] == 113
    assert report["totals"]["zip_bytes"]["all_stages_estimate_bytes"] == 20863
    assert report["totals"]["zip_bytes"]["with_50pct_margin_bytes"] == 31295
    stages["readback"]["archive"]["bytes"] = None
    missing = summary.budget(stages, checked, tmp_path)["totals"]["zip_bytes"]
    assert missing["all_stages_estimate_bytes"] is None and missing["with_50pct_margin_bytes"] is None
    assert missing["known_subtotal_bytes_including_shared"] == 18463 and missing["missing_stages"] == ["readback"]


def test_report_authenticates_data_member_in_original_zip(tmp_path, summary):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name, data in (("sample.bin", b"abc"), ("manifest.json", b"{}"), ("completion.json", b"{}")):
        (bundle / name).write_bytes(data)
    verified = {"manifest": {"files": [{"path": "sample.bin", "bytes": 3, "sha256": sha(bundle / "sample.bin")}]}}
    def pack(sample):
        with zipfile.ZipFile(tmp_path / "fixture.zip", "w") as z:
            for name in ("manifest.json", "completion.json"):
                z.write(bundle / name, name)
            z.writestr("sample.bin", sample)
    pack(b"abc")
    receipt = summary.container_info("fixture", verified, bundle, [tmp_path], tmp_path / "report")
    assert receipt["bytes"] == (tmp_path / "fixture.zip").stat().st_size and receipt["paths"] == ["../fixture.zip"]
    pack(b"cba")
    with pytest.raises(ValueError, match="changed ZIP member"):
        summary.container_info("fixture", verified, bundle, [tmp_path], tmp_path / "report")


def test_links_resolve_for_workspace_symlink_to_private_store(tmp_path, summary):
    store = tmp_path / "private" / "store"
    store.mkdir(parents=True)
    alias = tmp_path / "project" / ".research"
    alias.parent.mkdir()
    alias.symlink_to(store, target_is_directory=True)
    assert summary.relative(alias / "incoming/fixture.zip", store / "reports/B0") == "../../incoming/fixture.zip"


def test_report_cli_from_sealed_chain_and_seven_containers(tmp_path, monkeypatch, capsys):
    from evidence_lens.archive import pack
    from test_trial_selection import build_frozen_fixture

    store, selection = build_frozen_fixture(tmp_path, monkeypatch, capsys)
    frozen = json.loads(selection.read_text(encoding="utf8"))
    containers = tmp_path / "original-zips"
    for row in frozen["selected"].values():
        rid = row["run_id"]
        pack(store / "runs" / rid, containers / f"{rid}.zip", store)
    note = tmp_path / "handoff-preflight.json"
    note.write_text(json.dumps({"stage": "handoff_preflight", "status": "failed", "reason": "synthetic packaging failure",
                                "scientific_attempt": False}), encoding="utf8")
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "private-report"
    command = [sys.executable, str(root / "scripts/summarize_trial.py"), "--store", str(store),
               "--selection", str(selection), "--containers", str(containers),
               "--preflight-evidence", str(note), "--output", str(out)]
    subprocess.run(command, env=dict(os.environ, PYTHONPATH=str(root / "src")), cwd=tmp_path, check=True)
    result = json.loads((out / "report.json").read_text(encoding="utf8"))
    assert result["metrics"]["by_view"]["v01"]["V"] == 6
    assert result["metrics"]["by_view"]["v01"]["E"] == 0
    assert result["metrics"]["by_view"]["v01"]["rgb_mae"] is None
    counts = result["metrics"]["family_sum"]
    assert (counts["V"], counts["E"], counts["invalid_output"]) == (36, 25, 11)
    assert all(s["archive"]["bytes"] > 0 for s in result["stages"].values())
    assert result["failed_started_stage_count"] == 1 and len(result["preflights"]) == 1
    evaluation = store / "runs" / frozen["selected"]["evaluation"]["run_id"] / "payload"
    for label, name in (("Evaluation metrics CSV", "metrics.csv"), ("Evaluation metrics Parquet", "metrics.parquet")):
        assert (out / result["links"][label]).resolve() == evaluation / name
        assert result["evaluation_metric_sources"][name] == sha(evaluation / name)
    assert all((out / result["links"][k]).is_file() for k in ("v01 reference array", "v01 prediction array", "FP32 labels", "Raw L0"))
    assert (out / "report.md").stat().st_size > 0
