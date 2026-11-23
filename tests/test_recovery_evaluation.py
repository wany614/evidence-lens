"""Tiny deterministic fixtures for the preregistered recovery evaluation. No experimental images or fits."""

from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recovery_eval_coverage import counts_at, matched, ranks
from recovery_eval_quality import closest_axis, measurement, source_mean
from recovery_eval_select import selection
from dtu_score import METHODS, VIEWS


def test_native_strict_label_and_broken_prediction():
    ref = np.zeros((1, 4, 3), np.float32)
    pred = ref.copy()
    pred[0, 1] = np.float32(0.05)
    pred[0, 2] = np.float32(0.050001)
    pred[0, 3, 0] = np.nan
    c, (v, e, y, err) = measurement(ref, pred, np.ones((1, 4), bool))
    assert (c["V"], c["E"], c["V_minus_E"], c["Y"]) == (4, 3, 1, 1)
    assert y.tolist() == [[0, 0, 1, 255]]
    assert np.isnan(err[0, 3])
    missing, _ = measurement(ref, None, np.ones((1, 4), bool))
    assert missing["V"] == 4 and missing["E"] == 0 and missing["MAE"] is None and missing["Y_over_E"] is None


def test_input_role_boundary_rejects_extra_scoring_or_quality_data_before_opening():
    from recovery_eval_record import check_input_roles, needs_freeze

    score = dict.fromkeys(
        ("object_id", "condition", "protocol", "recipe", "freeze", "case", "provenance", "release")
    )
    score["schema"] = "el-recovery-score-request-1"
    score.update(object_id="set1", condition="M2")
    check_input_roles("score", score)
    assert needs_freeze("baseline", {"cells": [{"object_id": "set6", "condition": "M4"}]})
    assert not needs_freeze("baseline", {"cells": [{"object_id": "set1", "condition": "M4"}]})
    with pytest.raises(ValueError, match="source-only"):
        check_input_roles("score", {**score, "references": {"member": "payload/private-truth"}})
    with pytest.raises(ValueError, match="unregistered"):
        check_input_roles("quality", {"schema": "quality", "old_scores": {}})
    with pytest.raises(ValueError, match="reconstruction quality"):
        check_input_roles(
            "select", {"schema": "selection", "protocol": {}, "quality": [{"member": "payload/risk.json"}]}
        )


def test_source_only_mean_and_closest_axis_lexical_tie():
    src = {"s03": np.ones((256, 256, 3), np.float32), "s00": np.zeros((256, 256, 3), np.float32)}
    im, channels = source_mean(src)
    assert np.array_equal(im[10, 10], [0.5, 0.5, 0.5]) and channels == [0.5] * 3
    mats = {s: SimpleNamespace(c2w=np.eye(4)) for s in ("s03", "s00", "v01")}
    mats["s03"].c2w[:3, 3] = 100
    selected, angles = closest_axis(mats, src, "v01")
    assert selected == "s00" and angles == dict.fromkeys(src, 0)
    mats["s00"].c2w[:3, 2] = [0, 1, 0]
    assert closest_axis(mats, src, "v01")[0] == "s03"


def curve(recipe, cond, values, completed=True):
    return {
        "schema": "el-recovery-quality-1",
        "object_id": "set1",
        "role": "development_selection",
        "condition": cond,
        "risk_inputs_read": 0,
        "recipe_id": recipe,
        "recipe_sha256": recipe + "-digest",
        "execution_status": "completed" if completed else "truncated",
        "schedule_steps": 5000,
        "completed_steps": 5000 if completed else 1500,
        "steps": {
            str(s): {
                "status": "measured",
                "targets": {v: {"full_crop": {"MAE": mae}} for v in VIEWS},
                "checkpoint_sha256": str(s),
                "phase": "fixed",
            }
            for s, mae in values.items()
        },
    }


def test_selection_eligibility_and_ties_ignore_control_and_regression():
    rs = [
        curve(r, c, {500: 0.25, 1500: 0.25, 3000: 0.3, 5000: 0.5}) for r in ("B", "A") for c in ("M2", "M4")
    ]
    rows, winner = selection(rs)
    assert (winner["recipe_id"], winner["step"]) == ("A", 500) and len(rows) == 8
    rs[-1] = curve("A", "M4", {500: 0.01}, False)
    _, w = selection(rs)
    assert w["recipe_id"] == "B"
    rs[1]["steps"]["500"]["targets"]["v04"]["full_crop"]["MAE"] = None
    _, w = selection(rs)
    assert w["step"] == 1500
    rs[0]["object_id"] = "set6"
    pytest.raises(ValueError, selection, rs)


def test_quality_report_pairs_candidate_rows_by_the_same_update(tmp_path):
    from recovery_eval_report import quality_text

    settings = ("r2-scale-decay-02", "r2-geometry-01")
    data = [
        curve(name, cond, {500: 0.12, 1500: 0.25, 3000: 0.11, 5000: 0.15})
        for name in settings
        for cond in ("M2", "M4")
    ]
    rows, winner = selection(data)
    quality_text(tmp_path, [], {**winner, "stop_step": winner["step"], "candidates": rows}, [])
    lines = (tmp_path / "README.md").read_text().splitlines()
    cells = [line.split(" | ")[:2] for line in lines if " | r2-" in line]
    assert cells == [[f"| {step:,}", name] for step in (500, 1500, 3000, 5000) for name in reversed(settings)]
    assert "one run per setting" in " ".join(lines)


def test_exact_same_coverage_unattainable_and_partial_tie_limits():
    score = np.ones(8, np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], np.uint8)
    r = ranks((2, 4), "set1", "M2", "v01").reshape(-1)
    assert np.array_equal(r, ranks((2, 4), "set1", "M2", "v01").ravel())
    z = counts_at(score, y, r, 10, [("x", 3), ("empty", 0), ("too_big", 9)])
    m = z[0]
    assert m["accepted"] == 3 and m["cutoff_group_size"] == 8 and m["accepted_in_cutoff_group"] == 3
    assert (
        m["min_possible_accepted_positive"] == 0
        and m["max_possible_accepted_positive"] == 3
        and m["accepted_positive"] == int(y[np.argsort(r)[:3]].sum())
    )
    assert z[1]["reason"] == "empty_low" and z[2]["reason"] == "outside_support"
    valid = np.ones((1, 8), bool)
    raw = score.reshape(1, -1)
    a = {m: raw.copy() for m in METHODS}
    b = {m: raw.copy() for m in METHODS}
    b["MS-ft"][0, 1:3] = np.nan
    t = dict.fromkeys(METHODS, 1.4)
    c = matched(
        "set1",
        "M2",
        "v01",
        {
            "old": {"V": valid, "E": valid, "Y": y.reshape(1, -1), "scores": a, "thresholds": t},
            "new": {"V": valid, "E": valid, "Y": y.reshape(1, -1), "scores": b, "thresholds": t},
        },
        valid,
    )
    assert {o["eligible"] for o in c if o["domain"] == "paired_old_new"} == {6}
    same = [o for o in c if o["domain"] == "paired_old_new" and o["anchor"] == "40%_V"]
    assert len(same) == 8 and len({(z["accepted"], z["accepted_positive"]) for z in same}) == 1


def test_risk_gate_validates_archived_registration_before_score(monkeypatch):
    import recovery_eval_io as api

    fake = {
        "schema": "el-recovery-final-freeze-1",
        "recipe_sha256": "w",
        "registered_selection": True,
        "run_id": "f",
        "schedule_steps": 5000,
        "stop_step": 500,
        "registration": {"run_id": "unrelated"},
        "risk_fits": 0,
        "risk_recalibrations": 0,
        "scopes": {"B-L1-transfer": "one", "MS-L1": "all", "MS-frozen": "all", "MS-ft": "all"},
    }
    monkeypatch.setattr(api, "original", lambda roots, spec: (None, None, fake))
    req = {"freeze": {"run_id": "f"}, "recipe": {"sha256": "w"}, "protocol": {"run_id": "actual"}}
    pytest.raises(ValueError, api.permit_frozen, (), req)
    fake["registration"] = req["protocol"]
    assert api.permit_frozen((), req)["stop_step"] == 500


def test_independent_order_statistics_replay_all_matched_anchors():
    pytest.importorskip("sklearn")
    from recovery_eval_audit import coverage_replay, count_replay

    rng = np.random.default_rng(9184)
    v = np.ones((5, 8), bool)
    e = v.copy()
    e[0, 0] = False
    versions = {}
    expected = {}
    for version, cut in (("old", 3), ("new", 1)):
        risks = {}
        for m in METHODS:
            value = rng.integers(0, 5, size=v.shape).astype(np.float32) / 4
            value[:cut, 0] = np.nan
            risks[m] = value
        y = rng.integers(0, 2, size=v.shape, dtype=np.uint8)
        y[~e] = 255
        tau = dict.fromkeys(METHODS, 0.25)
        common = e & np.logical_and.reduce([np.isfinite(x) for x in risks.values()])
        versions[version] = {"V": v, "E": e, "Y": y, "scores": risks, "thresholds": tau}
        expected[version] = {"V": v, "y": y, "scores": risks, "tau": tau, "common": common}
        counts, aps = count_replay(
            v,
            e,
            y,
            np.zeros_like(e, dtype=np.float32),
            risks["MS-L1"],
            np.logical_and.reduce([np.isfinite(x) for x in risks.values()]),
            0.25,
        )
        assert counts["V"] == 40 and counts["E"] == 39
        assert counts["E"] == counts["L"] + counts["H"] + counts["U"]
        assert 0 < aps["common"] <= 1
    rows = matched("fixture", "M4", "v03", versions, v)
    assert coverage_replay(rows, expected, "fixture", "M4", "v03") == 152
    next(row for row in rows if row["accepted"])["accepted_positive"] += 1
    with pytest.raises(ValueError, match="cutoff or counts"):
        coverage_replay(rows, expected, "fixture", "M4", "v03")


def test_zero_accepted_view_counts_in_coverage_but_conditional_risk_remains_undefined():
    from recovery_eval_risk import coverage_totals

    rows = [
        {"reason": None, "accepted": 2, "accepted_positive": 1, "V": 10},
        {"reason": "empty_low", "accepted": 0, "accepted_positive": 0, "V": 10},
        {"reason": "outside_support", "accepted": None, "accepted_positive": None, "V": 10},
    ]
    result = coverage_totals(rows)
    assert result["coverage_defined_views"] == 2
    assert result["risk_defined_views"] == 1
    assert result["V_of_coverage_defined_views"] == 20
    assert (result["actual_coverage"], result["risk"]) == (0.1, 0.5)
    empty = coverage_totals([rows[1]])
    assert empty["actual_coverage"] == 0 and empty["risk"] is None


def test_native_quality_fixture_keeps_each_source_and_target_and_binds_the_visual_board(
    tmp_path, monkeypatch
):
    import recovery_eval_quality as worker
    from recovery_eval_report import quality_tables, inspect_visual_notes
    from evidence_lens.scoring_io import file_hash

    img = np.full((256, 256, 3), 0.4, dtype=np.float32)
    before = np.zeros_like(img)
    sources = ("s00", "s03")
    ids = (*sources, *VIEWS)
    meta = {
        "checkpoint_sha256": "old-fixture",
        "reconstruction_run_id": "test-old",
        "source_images": {s: {"sha256": s + "-original"} for s in sources},
        "renders": {v: {"sha256": v + "-old"} for v in ids},
    }
    case = SimpleNamespace(
        sha256="fixture-case",
        generator=sources,
        cameras={v: SimpleNamespace(c2w=np.eye(4)) for v in ids},
        frame=lambda v: {"rgb": before},
        meta=meta,
    )
    originals = SimpleNamespace(
        case=case,
        sources={s: img.copy() for s in sources},
        refs={v: img.copy() for v in VIEWS},
        masks={v: np.ones(img.shape[:2], bool) for v in VIEWS},
        object_masks=dict.fromkeys(VIEWS),
        object_reason=dict.fromkeys(VIEWS, "no fixture mask"),
        hashes={v: v + "-ref" for v in VIEWS},
    )
    checkpoint = {
        "step": 5000,
        "phase": "fixed",
        "gaussians": 5,
        "checkpoint_sha256": "example",
        "views": {v: {"sha256": v + "-output"} for v in ids},
    }
    output = SimpleNamespace(
        data={
            "status": "completed",
            "completed_steps": 5000,
            "schedule_steps": 5000,
            "checkpoints": {"5000": checkpoint},
            "recipe_sha256": "example",
            "protocol_sha256": "fixture",
        },
        rgb=lambda step, v: img.copy(),
    )
    monkeypatch.setattr(worker, "authority", lambda *_: None)
    monkeypatch.setattr(worker, "recipe", lambda *_: None)
    monkeypatch.setattr(worker, "Outputs", lambda *_: output)
    monkeypatch.setattr(worker, "Originals", lambda *_: originals)
    req = {
        "schema": "el-recovery-quality-request-1",
        "object_id": "set1",
        "condition": "M2",
        "recipe_id": "fixture",
    }
    root = tmp_path / "quality"
    result = worker.work((), req, root, "test-fixture")
    assert result["risk_inputs_read"] == 0
    tabs, missing = quality_tables([result])
    assert len(missing) == 4
    assert len(tabs["paired-targets"]) == 6
    assert len(tabs["source-views"]) == 4
    for row in tabs["target-views"]:
        if row["appearance"] == "recovery" and row["domain"] == "full_crop":
            assert (row["V"], row["E"], row["Y"], row["MAE"]) == (256**2, 256**2, 0, 0)
    note = {
        "cells": [
            {
                "object_id": "set1",
                "condition": "M2",
                "step": 5000,
                "source_fit": "Synthetic unit fixture.",
                "interpretation": "Synthetic unit fixture.",
                "source_board_sha256": file_hash(root / "boards/source-5000.png"),
                "target_board_sha256": file_hash(root / "boards/target-5000.png"),
                "views": dict.fromkeys(VIEWS, "Synthetic fixture colour."),
            }
        ]
    }
    inspect_visual_notes(note, [result], 5000, tmp_path)
    assert "v06" in (tmp_path / "visual-readback.md").read_text()
    note["cells"][0]["target_board_sha256"] = "other"
    with pytest.raises(ValueError, match="another board"):
        inspect_visual_notes(note, [result], 5000, tmp_path)


def test_numeric_plots_and_private_gallery_are_portable(tmp_path):
    from xml.etree.ElementTree import parse
    from recovery_eval_figures import quality_curves, risk_curves
    from recovery_eval_report import gallery
    from evidence_lens.scoring_io import file_hash
    from PIL import Image

    pytest.importorskip("matplotlib")
    qrows = [
        {
            "object_id": "set1",
            "appearance": "recovery",
            "condition": c,
            "recipe_id": "fixture",
            "view_role": role,
            "step": step,
            "macro_view_MAE": step / 10000,
        }
        for c in ("M2", "M4")
        for role in ("target", "source")
        for step in (0, 500, 1500)
    ]
    rrows = [
        {
            "object_id": o,
            "condition": c,
            "domain": "paired_old_new",
            "version": version,
            "method": m,
            "anchor": str(p) + "%_V",
            "risk": None if p == 40 else p / 100,
        }
        for o in ("set1", "set6")
        for c in ("M2", "M4")
        for version in ("old", "new")
        for m in METHODS
        for p in (1, 10, 40, 80)
    ]
    quality_curves(qrows, tmp_path / "quality.svg")
    risk_curves(rrows, tmp_path / "risk.svg")
    for name in ("quality", "risk"):
        xml = parse(tmp_path / (name + ".svg"))
        assert not list(xml.getroot().iter("{http://www.w3.org/2000/svg}image"))
    img = tmp_path / "known-fixture.png"
    Image.new("RGB", (2, 2), (24, 160, 191)).save(img)
    gallery(tmp_path, [("fixture", "M4", "original fixture", "test.png", img, file_hash(img))])
    html = (tmp_path / "private/index.html").read_text()
    assert 'href="fixture/M4/test.png"' in html
    assert file_hash(tmp_path / "private/fixture/M4/test.png") == file_hash(img)


def test_new_checkpoint_adapter_uses_its_own_ed_and_opacity_for_every_source_and_target(
    tmp_path, monkeypatch
):
    import json
    from test_stage3_multisource import case_fixture, camera, render
    from evidence_lens.multisource_io import Case
    from evidence_lens.scoring_io import file_hash
    import recovery_score_case as adapter

    prior = tmp_path / "old"
    case_fixture(prior, "M2")
    original_case = prior / "case.json"
    digest = file_hash(original_case)
    raw = tmp_path / "new-frames"
    raw.mkdir()
    ids = ("s00", "s03", *VIEWS)
    for i, v in enumerate(ids):
        np.savez_compressed(raw / (v + ".npz"), **render(camera(), tone=0.1 + i / 25, depth=4 + i, alpha=0.3))
    pin = {"id": "fixture-depth", "sha256": "8" * 64}
    new_model = "6" * 64
    old_pin = {"run_id": "fixture-original", "manifest_sha256": "1" * 64}
    case_pin = {**old_pin, "member": "payload/case/case.json", "sha256": digest}
    proof_path = tmp_path / "old-provenance.json"
    proof_member = "payload/case-provenance.json"
    archive = SimpleNamespace(
        pin=old_pin, files={}, open=lambda name: proof_path if name == proof_member else None
    )

    def provenance(obj):
        c = Case(prior, digest)
        proof_path.write_text(
            json.dumps(
                {
                    "schema": "el-dtu-case-provenance-1",
                    "run_id": old_pin["run_id"],
                    "case_sha256": digest,
                    "object_id": obj,
                    "condition": "M2",
                    "arm": "main-256",
                    "optimization_steps": 256,
                    "source_images": {s: {"rgb": value} for s, value in c.meta["source_images"].items()},
                }
            )
        )
        sha = file_hash(proof_path)
        archive.files[proof_member] = {"sha256": sha}
        return {**old_pin, "member": proof_member, "sha256": sha}

    metadata = {
        "original_case": case_pin,
        "original_case_provenance": provenance("set1"),
        "object_id": "set1",
        "condition": "M2",
        "protocol_sha256": "fixture",
        "status": "completed",
        "completed_steps": 5000,
        "checkpoints": {"500": {"step": 500, "checkpoint_sha256": new_model}},
        "depth_verified": True,
        "depth_probe": pin,
        "depth_semantics": "expected_camera_z_m",
        "rgb_encoding": "display_srgb_0_1",
        "alpha_semantics": "accumulated_opacity",
        "reconstruction_run_id": "fixture-new",
        "recipe_sha256": "fixture-recipe",
    }
    wrap = SimpleNamespace(data=metadata, asset=lambda step, v: raw / (v + ".npz"))
    monkeypatch.setattr(adapter, "authority", lambda *_: {"protocol_sha256": "fixture"})
    monkeypatch.setattr(adapter, "recipe", lambda *_: None)
    monkeypatch.setattr(
        adapter,
        "permit_frozen",
        lambda *_: {
            "stop_step": 500,
            "selection": {"winner": {"checkpoint_sha256": {"M2": new_model}}},
            "recipe_sha256": "fixture-recipe",
            "protocol_sha256": "fixture",
            "authorized_new_conditions": {"set6": ["M2", "M4"]},
        },
    )
    monkeypatch.setattr(adapter, "Outputs", lambda *_: wrap)
    req = {
        "schema": "el-recovery-case-request-1",
        "object_id": "set1",
        "condition": "M2",
        "protocol": {},
        "recipe": {"sha256": "fixture-recipe"},
        "freeze": {},
        "original_case": case_pin,
        "outputs": {},
        "depth_probe": pin,
    }
    monkeypatch.setattr(
        adapter, "original", lambda roots, p: (archive, original_case, {"passed": True} if p == pin else {})
    )
    dst = tmp_path / "adapter-out"
    d = adapter.work((), req, dst, "test-fixture")
    c = Case(dst / "case", d["case_sha256"])
    assert d["reference_reads"] == 0 and c.meta["checkpoint_sha256"] == new_model
    for i, v in enumerate(ids):
        a = c.frame(v)
        assert np.all(a["z"] == 4 + i) and np.all(a["alpha"] == np.float32(0.3))
        assert file_hash(raw / (v + ".npz")) == c.meta["renders"][v]["sha256"]
    metadata["depth_verified"] = False
    with pytest.raises(ValueError, match="semantic evidence"):
        adapter.work((), req, tmp_path / "wrong", "test-fixture")
    metadata["depth_verified"] = True
    req["object_id"] = "set6"
    with pytest.raises(ValueError, match="original D1 identity"):
        adapter.work((), req, tmp_path / "foreign-original", "test-fixture")
    metadata["original_case_provenance"] = provenance("set6")
    with pytest.raises(ValueError, match="same cell"):
        adapter.work((), req, tmp_path / "foreign-normalization", "test-fixture")
    metadata.update(object_id="set6", completed_steps=500)
    metadata["recipe_sha256"] = "other-recipe"
    with pytest.raises(ValueError, match="same cell"):
        adapter.work((), req, tmp_path / "foreign-recipe", "test-fixture")
    metadata["recipe_sha256"] = "fixture-recipe"
    metadata["original_case"] = {**case_pin, "sha256": "2" * 64}
    with pytest.raises(ValueError, match="same cell"):
        adapter.work((), req, tmp_path / "foreign-seal", "test-fixture")
    metadata["original_case"] = case_pin
    assert adapter.work((), req, tmp_path / "closed-regression", "test-fixture")["object_id"] == "set6"


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.0001, 1.0001, True, "0.2", 1])
def test_selection_excludes_illegal_mae_and_records_every_registered_checkpoint(invalid):
    values = {0: 0.0, 500: 0.2, 1500: 0.3, 3000: 0.4, 5000: 0.5}
    data = [curve(name, c, values) for name in ("r1", "r2") for c in ("M2", "M4")]
    data[0]["steps"]["500"]["targets"]["v06"]["full_crop"]["MAE"] = invalid
    del data[3]["steps"]["1500"]
    records, winner = selection(data)
    assert [(r["recipe_id"], r["step"]) for r in records] == [
        (name, s) for name in ("r1", "r2") for s in (500, 1500, 3000, 5000)
    ]
    assert records[0]["macro_target_MAE"] is None and records[0]["condition_exclusions"] == {
        "M2": "invalid_native_rgb_mae"
    }
    assert (
        records[5]["reason"] == "missing_comparable_target"
        and winner["recipe_id"] == "r2"
        and winner["step"] == 500
    )


def test_coverage_rejects_different_v_or_e_outside_v_and_uses_half_down_rounding():
    valid = np.ones((5, 10), bool)
    risk = np.arange(valid.size, dtype=np.float32).reshape(valid.shape) / valid.size
    versions = {
        side: {
            "V": valid.copy(),
            "E": valid.copy(),
            "Y": np.zeros(valid.shape, np.uint8),
            "scores": {m: risk.copy() for m in METHODS},
            "thresholds": dict.fromkeys(METHODS, 0.4),
        }
        for side in ("old", "new")
    }
    versions["new"]["scores"]["MS-ft"][2, 3] = np.nan
    rows = matched("set1", "M2", "v01", versions, valid)
    paired = [r for r in rows if r["domain"] == "paired_old_new"]
    assert len({r["eligible"] for r in paired}) == 1 and paired[0]["eligible"] == 49
    assert {r["requested_k"] for r in paired if r["anchor"] == "1%_V"} == {0}
    assert {r["requested_k"] for r in paired if r["anchor"] == "5%_V"} == {2}
    versions["old"]["V"][0, 0] = False
    with pytest.raises(ValueError, match="same V"):
        matched("set1", "M2", "v01", versions, valid)
    valid[0, 0] = False
    versions["new"]["V"][0, 0] = False
    with pytest.raises(ValueError, match="within"):
        matched("set1", "M2", "v01", versions, valid)
    for z in versions.values():
        z["E"][0, 0] = False
    versions["new"]["Y"][0, 1] = 255
    with pytest.raises(ValueError, match="binary labels"):
        matched("set1", "M2", "v01", versions, valid)
