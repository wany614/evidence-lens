"""Four test families share a single authenticated opening with a sealed method."""

import json
from pathlib import Path

import pytest

from evidence_lens._bundle import BundleError
from evidence_lens.stage3_gate import require_opening

CAT = "b" * 64


def test_ordinary_folder_or_wrong_index_does_not_open_test(tmp_path):
    with pytest.raises(BundleError, match="still sealed"):
        require_opening(tmp_path, CAT)
    with pytest.raises(BundleError, match="roster SHA"):
        require_opening(tmp_path, "../unregistered")


def test_a_marker_needs_its_original_method_and_external_digest(tmp_path):
    p = tmp_path / "protocol" / "stage3" / CAT / "test-opening.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({
        "catalog_sha256": CAT, "schema": "el-stage3-test-opening-1", "conditions": ["M2", "M4"],
        "method": {"run_id": "20260101T140000000000Z-1234567890abcdef", "manifest_sha256": "c" * 64},
        "method_sha256": "a" * 64, "test_families": ["A", "B", "C", "D"], "opened_at_utc": "2026-01-01T14:30:00Z",
    }))
    with pytest.raises(BundleError, match="Unexpected Stage C opening"):
        require_opening(tmp_path, CAT, opening_sha256="f" * 64)
    with pytest.raises((FileNotFoundError, BundleError)):
        require_opening(tmp_path, CAT)


def test_default_recomputed_from_both_seeds_and_original_producer():
    from copy import deepcopy
    from evidence_lens.stage3_decision import check_selection

    study = json.loads((Path(__file__).resolve().parents[1] / "protocol/stage3-study-v1.json").read_bytes())
    names = study["methods"]
    seeds = ["1729", "2718"]
    producer = {m: {"model_id": m, "weights_sha256": str(i)*64, "model_index_sha256": str(i+4)*64,
                    "threshold": .4} for i, m in enumerate(names)}
    rows = {s: {m: {"ap": .8 if m == "MS-ft" else .75, "complete": True,
                     "low_error": .1, "low_coverage": .8} for m in names} for s in seeds}
    decisions = {k: True for k in ("MS-L1_AP", "MS-frozen_AP", "1729_coverage", "2718_coverage",
                                    "1729_low_error", "2718_low_error")}
    frozen = {"models": producer, "catalog_sha256": CAT, "study_sha256": "a"*64,
              "thresholds": {m: .4 for m in names}, "default_method": "MS-ft",
              "components": {"neural_selection": {"run_id": "selection"},
                             "development_dataset": {"sha256": "d"*64}, "thresholds": {"sha256": "f"*64}}}
    selection = {"schema": "el-stage3-neural-selection-1", "run_id": "selection", "cohort": "development",
                 "study_sha256": "a"*64, "catalog_sha256": CAT, "development_dataset_sha256": "d"*64,
                 "methods": names, "models": producer, "files": {"thresholds.json": {"sha256": "f"*64}},
                 "summaries": rows, "decision": {"release_seed": 1729, "default": "MS-ft", "checks": decisions}}
    cuts = {"schema": "el-stage3-threshold-recalculation-1", "dataset_sha256": "d"*64, "catalog_sha256": CAT,
            "methods": {m: {"tau": .4} for m in names}, "all_seed_models": {s: producer for s in seeds}}
    light = {"schema": "el-stage3-light-development-1", "development_dataset_sha256": "d"*64,
             "run_id": "MS-L1", "model_index_sha256": producer["MS-L1"]["model_index_sha256"],
             "weights_sha256": producer["MS-L1"]["weights_sha256"], "threshold": {"tau": .4}}
    check_selection(frozen, study, selection, cuts, light)
    broken = deepcopy(selection)
    broken["summaries"]["2718"]["MS-ft"]["ap"] = .74
    with pytest.raises(BundleError, match="registered rule"):
        check_selection(frozen, study, broken, cuts, light)
    changed = deepcopy(cuts)
    changed["methods"]["MS-ft"]["tau"] = .41
    with pytest.raises(BundleError, match="threshold calculation"):
        check_selection(frozen, study, selection, changed, light)
