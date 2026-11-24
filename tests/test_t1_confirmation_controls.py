"""Regression cases for the five-receipt gate and the original probe's terminal cap guard."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from t1_access import POOL, S0, STRATEGIES, decisions
from t1_probe_terminal import planned_random
from t1_study import stage

from evidence_lens._bundle import save, sha
from evidence_lens._verify import BundleError


def valid_receipts(tmp_path):
    common = {"run_id": "test-run", "object_id": "set45", "m2_checkpoint_sha256": "f" * 64}
    files = {}
    for name in STRATEGIES:
        path = tmp_path / (name + ".json")
        save(path, dict(**common, strategy_id=name, selected_official_ids=[17, 23]))
        files["choice-" + name] = path
    joint = dict(
        **common,
        schema="el-b2-joint-nbv-choices-1",
        pool=POOL,
        legal_photo_ids_before_selection=S0,
        results={name: [17, 23] for name in STRATEGIES},
        selections={name: sha(files["choice-" + name]) for name in STRATEGIES},
    )
    return joint, files


def test_both_camera_ids_must_be_committed_for_every_strategy(tmp_path):
    joint, files = valid_receipts(tmp_path)
    decisions(joint, files, "set45")
    missing = copy.deepcopy(joint)
    missing["results"]["risk"] = None
    with pytest.raises(BundleError):
        decisions(missing, files, "set45")
    missing = copy.deepcopy(joint)
    del missing["selections"]["random-137103"]
    with pytest.raises(BundleError):
        decisions(missing, files, "set45")


def test_unordered_camera_knowledge_cannot_replace_the_original_fixed_slot_order(tmp_path):
    from t1_access import FIXED

    joint, files = valid_receipts(tmp_path)
    decisions(joint, files, "set45")
    selected = [24, *joint["results"]["risk"], 22]
    assert selected == [24, 17, 23, 22]
    assert sorted(selected) == sorted(FIXED)
    assert selected != FIXED


def test_genuine_point_count_failure_cannot_release_even_the_fixed_pair(tmp_path):
    joint, files = valid_receipts(tmp_path)
    joint.update(
        schema="el-t1-probe-terminal-1",
        results={k: None for k in STRATEGIES},
        completed_pair_decisions=0,
        candidate_photo_access_authorized=False,
    )
    with pytest.raises(BundleError):
        decisions(joint, files, "set45")
    assert planned_random("uniform", "set45") is None
    assert planned_random("random-137101", "set45") == [17, 20]


def test_opened_study_cannot_dispatch_a_missing_scientific_job(tmp_path):
    opening = tmp_path / "work/controller/t1-opening"
    opening.mkdir(parents=True)
    save(opening / "selected.json", {"test": True})
    with pytest.raises(BundleError, match="joint opening"):
        stage(
            (tmp_path, tmp_path), "t1-set45-M4-fixed-dispatch", "dataset_build", {}, {}, [], "unused.json", 0
        )
