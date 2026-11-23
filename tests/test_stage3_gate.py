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
