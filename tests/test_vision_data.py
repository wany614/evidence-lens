"""Deterministic disposable fixtures. They never read the research store or contain real scientific weights."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from evidence_lens.l1 import ALL_COLUMNS, KEYS, schema_digest
from evidence_lens.l1_freeze import threshold_for
from evidence_lens.scoring_io import TARGETS, file_hash, write_json
import vision_baselines as baselines
import vision_data as data
import vision_evaluate as ev
import vision_sources as sources


def fixture_truth(e=None):
    e = np.ones((2, 4), bool) if e is None else e
    v = np.ones(e.shape, bool)
    rgb = np.array([[0.01, 0.05, 0.05001, 0.20], [0.01, 0.01, 0.4, 0.06]], np.float32)
    rgb = np.where(e, rgb, np.float32(np.nan))
    high = np.where(e, rgb > np.float32(0.05), 255).astype(np.uint8)
    return {
        "reference_valid": v,
        "rgb_evaluable": e,
        "rgb_high_error": high,
        "rgb_error": rgb,
        "reference_surface": v,
        "depth_evaluable": np.zeros(e.shape, bool),
        "depth_high_error": np.full(e.shape, 255, np.uint8),
        "depth_error": np.full(e.shape, np.nan, np.float32),
        "signed_depth_error": np.full(e.shape, np.nan, np.float32),
        "relative_depth_error": np.full(e.shape, np.nan, np.float32),
        "signed_relative_depth_error": np.full(e.shape, np.nan, np.float32),
        "visibility": np.array([[0, 1, 1, 1], [2, 2, 3, 3]], np.uint8),
        "edge": np.array([[0, 0, 1, 0], [1, 0, 1, 0]], bool),
        "low_support": np.zeros(e.shape, bool),
        "low_alpha": np.zeros(e.shape, bool),
        "alpha_valid": v,
    }


def base_scores():
    return {m: np.array([[0, 0, 0, 1], [1, 1, 1, 1]], np.float32) for m in sources.METHODS}


def test_native_counts_common_masks_and_ties():
    mask = np.ones((2, 4), bool)
    mask[0, 3] = False
    truth = fixture_truth(mask)
    scores = base_scores()
    scores["L2-vision-ft"][1, 2] = np.nan
    scores["L0"][1, 3] = np.nan
    tau = {m: 0.5 for m in scores}
    result = ev.system_counts(truth, scores, tau, np.ones(mask.shape, bool))
    dense = result["L2-vision-ft"]["counts"]
    assert (dense["V"], dense["E"], dense["Y"], dense["S5"], dense["S7"]) == (8, 7, 3, 6, 5)
    assert (dense["L"], dense["H"], dense["U"], dense["LY"], dense["HY"], dense["UY"]) == (3, 3, 1, 1, 1, 1)
    assert dense["missed_errors"] == 2
    assert result["L2-frozen"]["ranking"]["five_common"] is None
    for m in ev.FIVE:
        assert result[m]["ranking"]["five_common"]["n"] == 6
        assert result[m]["ranking"]["seven_common"]["n"] == 5
    support = mask & np.isfinite(scores["L0"]) & np.isfinite(scores["L2-vision-ft"])
    y = truth["rgb_high_error"][support] == 1
    a = average_precision_score(y, scores["L1-full"][support])
    assert np.isclose(a, result["L1-full"]["ranking"]["seven_common"]["ap"])
    no = ev.system_counts(fixture_truth(np.zeros((2, 4), bool)), scores, tau, np.ones(mask.shape, bool))
    assert no["L0"]["counts"]["V"] == 8 and no["L0"]["counts"]["E"] == 0
    assert (
        no["L0"]["rates"]["low_coverage"]["value"] == 0
        and no["L0"]["rates"]["low_risk_error"]["value"] is None
    )


def test_family_weight_tie_and_empty_masks():
    assert ev.ranking(np.array([0.3, 0.3], np.float32), np.array([0, 1]))["ap"] == 0.5
    for y, reason in (([], "empty_mask"), ([0, 0], "all_negative"), ([1, 1], "all_positive")):
        assert ev.ranking(np.zeros(len(y), np.float32), np.array(y))["missing_reason"] == reason
    choice = threshold_for(
        {"a": np.array([0] * 3 + [1] * 2, np.float32), "b": np.array([0] * 3 + [1] * 2, np.float32)},
        {"a": 5, "b": 5},
    )
    assert choice["coverage"] == 0.6 and choice["tau"] == 0.5
    curve = ev.compact_curve(
        [ev.groups(np.array([0.1, 0.1, 0.9, 0.9], np.float32), np.array([1, 0, 1, 0]))], 5, 0.5
    )
    assert curve["frozen_L"] == 2 and curve["frozen_LY"] == 1
    assert all(p["accepted"] in (0, 2, 4) for p in curve["points"])


def test_no_test_transport_or_scores_read_before_real_opening(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    shelf = tmp_path / "shelf"
    shelf.mkdir()
    rows = [{"scene_family_id": f"test{i}", "split": "test"} for i in range(8)]
    catalog = tmp_path / "catalog.json"
    write_json(catalog, {"schema": "el-vision-family-roster-1", "families": rows})
    pin = (catalog, file_hash(catalog))
    absent = (tmp_path / "DO-NOT-READ", "a" * 64)
    with pytest.raises(ValueError, match="test remains sealed"):
        sources.make_index(root, shelf, *pin, "test", tmp_path / "output", *absent)
    with pytest.raises(ValueError, match="test remains sealed"):
        data.prepare([absent], [root, shelf], "test", tmp_path / "output", gate="b" * 64, test_catalog=pin)
    with pytest.raises(ValueError, match="test remains sealed"):
        ev.evaluate(absent[0], absent[1], "test", [absent], absent, tmp_path / "output", [root, shelf], [pin])
    with pytest.raises(ValueError, match="test remains sealed"):
        baselines.score(
            absent[0], absent[1], absent[0], absent[1], tmp_path / "output", "test", [root, shelf], pin
        )
    with pytest.raises(ValueError, match="test-opening check"):
        data.open_dataset(absent[0], absent[1], "test")


def test_legacy_four_can_never_be_confirmation(tmp_path):
    p = tmp_path / "old.json"
    write_json(p, {"schema": "el-formal-families-1", "families": []})
    with pytest.raises(ValueError, match="historical test material"):
        sources.make_index(tmp_path, tmp_path, p, file_hash(p), "test", tmp_path / "out")


def dataset_fixture(tmp, split, families, with_scores=False, formal=False):
    root = tmp / split
    root.mkdir(parents=True)
    a = fixture_truth()

    def add_npz(name, **arrays):
        path = root / name
        np.savez_compressed(path, **arrays)
        return data.assetspec(path, root)

    x = np.zeros((8, 26), np.float32)
    x[:, 0] = np.linspace(0.1, 0.9, 8)
    x[:, 12] = 1
    x[0, 1] = np.nan
    x[0, 16] = 1
    sample = add_npz(
        "sample.npz",
        index=np.arange(8, dtype=np.int32),
        X=x,
        Y=(a["rgb_high_error"].ravel() == 1).astype(np.uint8),
    )
    rgb = root / "rgb.npy"
    np.save(rgb, np.zeros((2, 4, 3), np.float32))
    scores = base_scores()
    scores["constant"][:] = 0.4
    dep = add_npz(
        "dep.npz",
        X=x.reshape(2, 4, 26),
        scoreable=np.ones((2, 4), bool),
        l0=scores["L0"],
        alpha_score=scores["1-alpha"],
    )
    tr = add_npz("truth.npz", **a)
    row = {
        "condition": "B0",
        "status": "completed",
        "reason": None,
        "rgb": data.assetspec(rgb, root),
        "samples": sample,
        "full": None if split == "train" else {"deploy": dep, "truth": tr},
        "height": 512 if formal else 2,
        "width": 512 if formal else 4,
        "sample_count": 8,
        "eligible": 8,
        "E": 8,
        "V": 8,
        "positive_E": int((a["rgb_high_error"] == 1).sum()),
    }
    roster = []
    vs = []
    for n, f in enumerate(families):
        fold = n % 4 if split == "train" else None
        roster.append({"scene_family_id": f, "split": split, "cv_fold": fold, "trials": {"B0": f + "B0"}})
        vs.extend(
            [
                {
                    **row,
                    "family": f,
                    "view": v,
                    "cv_fold": fold,
                    "source_input_sha256": f.encode().hex().ljust(64, "0"),
                }
                for v in TARGETS
            ]
        )
    catalog = tmp / (split + ".json")
    write_json(catalog, {"schema": "synthetic-fixture", "families": roster})
    ds = {
        "schema": "el-vision-dataset-1",
        "split": split,
        "columns": ALL_COLUMNS,
        "label_version": "el-labels-1",
        "scope": "specified_materials",
        "feature_schema_digest": schema_digest(),
        "sample_cap": 4096,
        "views": vs,
        "test_unlock_sha256": None,
        "material_indices": [{"catalog_sha256": file_hash(catalog), "sha256": "3" * 64, "cohort": "fixture"}],
    }
    write_json(root / "dataset.json", ds)
    ds_sha = file_hash(root / "dataset.json")
    pin = (catalog, file_hash(catalog))
    if not with_scores:
        return root, ds_sha, pin
    score_pins = []
    for i, tag in enumerate(ev.SCORE_SETS):
        folder = tmp / tag
        folder.mkdir()
        if tag == "original-five":
            arrays = {KEYS[m]: scores[m] for m in ev.FIVE}
        else:
            s = scores[tag]
            final = threshold_for({"f": s.ravel()}, {"f": s.size})["tau"]
            arrays = {
                "risk": s,
                "valid": np.isfinite(s),
                "state": (s.astype(np.float64) >= final).astype(np.uint8),
            }
        path = folder / "fixture.npz"
        np.savez_compressed(path, **arrays)
        entries = [
            {
                **{k: r[k] for k in ("family", "condition", "view")},
                "status": "scored",
                "reason": None,
                "score": data.assetspec(path, folder),
                "rgb_sha256": row["rgb"]["sha256"],
                "deploy_sha256": dep["sha256"],
            }
            for r in vs
        ]
        idx = folder / "scores.json"
        write_json(
            idx,
            {
                "schema": "el-vision-scores-1",
                "dataset_sha256": ds_sha,
                "method": tag,
                "model_sha256": str(i + 1) * 64,
                "views": entries,
            },
        )
        score_pins.append((idx, file_hash(idx)))
    return root, ds_sha, pin, score_pins


def test_small_independent_development_workflow(tmp_path, monkeypatch):
    # Only this disposable numerical test relaxes registered family count/image size.
    root, sha, pin, indices = dataset_fixture(tmp_path, "development", ["a", "b"], True)
    monkeypatch.setattr(baselines, "full_cohort", lambda *args: None)
    f = tmp_path / "freeze"
    e = tmp_path / "evaluation"
    dev = ev.freeze_dev(root, sha, indices, [pin], f, "fixture-dev")
    r = ev.evaluate(
        root,
        sha,
        "development",
        indices,
        (f / "thresholds.json", file_hash(f / "thresholds.json")),
        e,
        None,
        [pin],
    )
    assert dev["thresholds"]["L0"] == 0.5 and r["planned_target_views"] == 12 and len(r["files"]) > 16
    macro = json.loads((e / "macro.json").read_text())
    assert macro[0]["methods"]["L1-full"]["five_common_ap"]["defined_families"] == 2
    assert (e / "a/B0/v01.json").exists() and (e / "b/B0/curves.json").exists()
    scores = json.loads(indices[2][0].read_text())
    scores["views"][0]["rgb_sha256"] = "0" * 64
    write_json(indices[2][0], scores)
    updated = ev.score_sources(
        data.open_dataset(root, sha, "development"),
        sha,
        [*indices[:2], (indices[2][0], file_hash(indices[2][0]))],
    )
    with pytest.raises(ValueError, match="another source"):
        ev.read_scores(data.open_dataset(root, sha, "development")["views"][0], updated)


def test_preprocessing_shares_numeric_schema_with_lr_and_checks_full_roster(tmp_path):
    root, sha, pin = dataset_fixture(tmp_path, "train", [f"f{i:02d}" for i in range(28)], formal=True)
    pre = data.preprocess_training(root, sha, tmp_path / "pre", [pin])
    assert pre["columns"] == ALL_COLUMNS and pre["number_of_rows"] == 28 * 6 * 8 and len(pre["medians"]) == 11
    fit = baselines.fit(root, sha, [pin], tmp_path / "fit", "synthetic-disposable-fit")
    assert fit["status"] == "fitted" and max(fit["max_abs_parity"].values()) < 1e-6
    ds = data.open_dataset(root, sha, "train")
    ds["views"][0]["height"] = 252
    with pytest.raises(ValueError, match="native target size"):
        baselines.full_cohort(ds, [pin], "train")
    with pytest.raises(ValueError, match="split"):
        data.open_dataset(root, sha, "development")


def seal_fixture(root, rid, kind, status, meta, files):
    dr = root / "runs" / rid
    dr.mkdir(parents=True)
    write_json(dr / "run.json", {"run_id": rid, "run_kind": kind, "execution_status": status, **meta})
    for rel, source in files.items():
        p = dr / rel
        p.parent.mkdir(exist_ok=True, parents=True)
        if isinstance(source, Path):
            p.hardlink_to(source)
        else:
            write_json(p, source)
    members = [
        {"path": p.relative_to(dr).as_posix(), **sources.signature(p)} for p in dr.rglob("*") if p.is_file()
    ]
    write_json(dr / "manifest.json", {"run_id": rid, "files": members})
    sha = file_hash(dr / "manifest.json")
    write_json(dr / "completion.json", {"run_id": rid, "manifest_sha256": sha, "execution_status": status})
    return {"run_id": rid, "manifest_sha256": sha}


def test_whole_trial_failure_keeps_reference_and_no_training_targets(tmp_path):
    store = tmp_path / "store"
    shelf = tmp_path / "shelf"
    f = "fixture"
    trial = f + "-B0"
    meta = {"scene_family_id": f, "trial_id": trial, "condition_id": "B0"}
    a = tmp_path / "native.npz"
    rgb = np.zeros((512, 512, 3), np.float32)
    rgb[0, 0] = np.nan
    plane = np.ones((512, 512), np.float32)
    np.savez_compressed(
        a,
        rgb=rgb,
        z=plane,
        position=np.ones_like(rgb),
        visibility=np.ones(plane.shape, np.uint8),
        edge=np.zeros(plane.shape, bool),
    )
    validation = tmp_path / "val.json"
    write_json(validation, {"fixture": True})
    fields = {
        "payload/reference.json": {
            "schema": "el-reference-1",
            "validation_passed": True,
            "scene_family_id": f,
            "scene_id": f + "/v1",
            "validation_sha256": file_hash(validation),
            "views": {v: {"height": 512, "width": 512, "reference_sha256": file_hash(a)} for v in TARGETS},
        },
        "payload/validation/results.json": validation,
        **{f"payload/reference/{v}.npz": a for v in TARGETS},
    }
    ref = seal_fixture(
        store, "20000101T000000000000Z-0000000000000001", "scene_render", "succeeded", meta, fields
    )
    fail = seal_fixture(store, "20000101T000000000000Z-0000000000000002", "generation", "failed", meta, {})
    row = {"scene_family_id": f, "scene_id": f + "/v1", "split": "development", "trials": {"B0": trial}}
    sel = {
        "schema": "el-vision-failed-trial-1",
        "scene_family_id": f,
        "trial_id": trial,
        "condition_id": "B0",
        "selection_status": "failed",
        "reference": ref,
        "failure": fail,
        "reason": "fixture generation terminated",
        "planned_targets": list(TARGETS),
    }
    src = sources.failed_cell([store, shelf], sel, "a" * 64, row, "B0")
    out = tmp_path / "pack"
    out.mkdir()
    result = data.emit_cell(src, "b" * 64, [store, shelf], out)
    assert len(result) == 6
    for r in result:
        assert (
            r["V"] == 512 * 512 - 1
            and r["E"] == 0
            and r["sample_count"] == 0
            and r["rgb"] is None
            and r["full"]["deploy"] is None
        )
        assert r["failure_kind"] == "generation" and r["truth_origin"] == "terminal_reference"
        with np.load(data.file_for(out, r["full"]["truth"]), allow_pickle=False) as truth:
            assert (truth["rgb_high_error"] == 255).all()


def test_reduced_source_uses_preparation_and_original_target_members(tmp_path):
    store = tmp_path / "store"
    rows = {v: {"height": 512, "width": 512, "reference_sha256": "d" * 64} for v in TARGETS}
    original = {
        "schema": "el-reference-1",
        "validation_passed": True,
        "validation_sha256": "f" * 64,
        "input_sha256": "1" * 64,
        "camera_sha256": "2" * 64,
        "views": rows,
    }
    ref = seal_fixture(
        store,
        "20000101T000000000000Z-0000000000000011",
        "scene_render",
        "succeeded",
        {},
        {"payload/reference.json": original},
    )
    roots = [store]
    rr = sources.find_run(roots, ref)
    old_sha = file_hash(rr.open("payload/reference.json"))
    pred = {"input_sha256": "3" * 64, "camera_sha256": "4" * 64}
    reduced = {**original, **pred, "source_condition": "R1", "parent_reference_sha256": old_sha}
    prep = seal_fixture(
        store,
        "20000101T000000000000Z-0000000000000012",
        "derivation",
        "succeeded",
        {},
        {"payload/reference.json": reduced},
    )
    pr = sources.find_run(roots, prep)
    member = "payload/reference.json"
    new = sources.signature(pr.open(member))
    sel = {"sources": {"reference_metadata": {**prep, **new, "path": member}}}
    effective, _, native, _ = sources.effective_reference(
        roots, "R1", sel, {"preparation": prep}, rr, new, pred
    )
    assert effective["input_sha256"] != native["input_sha256"] and effective["views"] == native["views"]
    with pytest.raises(ValueError, match="producer/consumer"):
        sources.effective_reference(roots, "R1", sel, {"preparation": prep}, rr, {"sha256": old_sha}, pred)
    bad = {**reduced, "views": {**rows, "v01": {**rows["v01"], "reference_sha256": "e" * 64}}}
    other = seal_fixture(
        store,
        "20000101T000000000000Z-0000000000000013",
        "derivation",
        "succeeded",
        {},
        {"payload/reference.json": bad},
    )
    m = sources.signature(sources.find_run(roots, other).open(member))
    q = {"sources": {"reference_metadata": {**other, **m, "path": member}}}
    with pytest.raises(ValueError, match="preserving six native targets"):
        sources.effective_reference(roots, "R1", q, {"preparation": other}, rr, m, pred)


def test_seven_method_freeze_retains_prior_five_thresholds(tmp_path, monkeypatch):
    root, sha, pin, indices = dataset_fixture(tmp_path, "development", ["a", "b"], True)
    monkeypatch.setattr(baselines, "full_cohort", lambda *args: None)
    five = tmp_path / "five"
    result = ev.freeze_dev(root, sha, indices[:1], [pin], five, "disposable-five", True)
    assert set(result["thresholds"]) == set(ev.FIVE) and result["thresholds"]["L0"] == 0.5
    prior = (five / "thresholds.json", file_hash(five / "thresholds.json"))
    all_seven = ev.freeze_dev(
        root, sha, indices, [pin], tmp_path / "seven", "disposable-seven", baseline_pin=prior
    )
    assert all_seven["independent_original_thresholds_sha256"] == prior[1]
    altered = {**result, "thresholds": {**result["thresholds"], "L1-full": 0.0}}
    wrong = tmp_path / "wrong.json"
    write_json(wrong, altered)
    with pytest.raises(ValueError, match="changed an original operating point"):
        ev.freeze_dev(
            root,
            sha,
            indices,
            [pin],
            tmp_path / "bad",
            "disposable-seven",
            baseline_pin=(wrong, file_hash(wrong)),
        )


def test_confirmed_opening_needs_actual_models_and_paired_seeds(tmp_path):
    from stage2_opening import open_cohort, verify_opening
    from vision_gate_fixture import gate_fixture

    g = gate_fixture(tmp_path)
    pin, method, method_sha = g["freeze"]()
    store, shelf, cat, sha = (g[k] for k in ("store", "shelf", "catalog", "catalog_sha"))
    result = open_cohort(store, cat, sha, pin["run_id"], pin["manifest_sha256"])
    assert result["method_sha256"] == method_sha and len(method["training_receipts"]) == 8
    assert verify_opening(store, json.loads(cat.read_bytes()), sha) == result
    marker = store / "protocol/stage2" / sha / "test-opening.json"
    assert sources.test_authority([store, shelf], (cat, sha)) == file_hash(marker)
    for c in method["cpu_costs"].values():
        assert c["native_cold_seconds"] == 2 and c["cold_seconds"] == 2.5
        assert c["native_process_tree_peak_rss_bytes"] == 100 and c["rss_bytes"] == 115
        assert len(c["canonical_win_to_mac"]) == 6


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("omit_second_A", "eight-candidate"),
        ("change_repeat_lr", "learning rate"),
        ("replace_seed_A", "selected seed-A"),
        ("change_native_pin", "original member digest mismatch"),
        ("change_external_pin", "original member digest mismatch"),
        ("forge_rss", "cold/RSS projection"),
        ("forge_hot_p95", "wall order statistics"),
        ("forge_device_parity", "numerical diagnostics"),
        ("wrong_display", "thresholds, seeds or display"),
        ("change_tau", "operating point"),
    ],
)
def test_freeze_and_recheck_reject_changed_scientific_proofs(tmp_path, change, reason):
    from stage2_opening import open_cohort
    from vision_gate_fixture import gate_fixture

    g = gate_fixture(tmp_path)

    def mutate(method, roster):
        rec = method["training_receipts"]
        m = "L2-frozen"
        cost = method["cpu_costs"][m]
        if change == "omit_second_A":
            rec.pop(1)
            roster["candidate_runs"].pop(1)
        elif change == "change_repeat_lr":
            rec[2]["head_lr"] = rec[1]["head_lr"]
        elif change == "replace_seed_A":
            method["models"][m].update({k: rec[2][k] for k in ("run_id", "manifest_sha256", "model_sha256")})
        elif change == "change_native_pin":
            cost["native_cost_sha256"] = "a" * 64
        elif change == "change_external_pin":
            cost["process_cost_sha256"] = "9" * 64
        elif change == "forge_rss":
            cost["rss_bytes"] = 99
        elif change == "forge_hot_p95":
            cost["hot_p95_seconds"] = 0.5
        elif change == "forge_device_parity":
            cost["canonical_win_to_mac"]["v02"]["threshold_crossings"] = 12
        elif change == "wrong_display":
            method["display_rule"] = "v04"
        elif change == "change_tau":
            method["thresholds"]["L0"]["tau"] = 0.6

    pin, _, method_sha = g["freeze"](mutate)
    store, shelf, cat, sha = (g[k] for k in ("store", "shelf", "catalog", "catalog_sha"))
    marker = store / "protocol/stage2" / sha / "test-opening.json"
    with pytest.raises(ValueError, match=reason):
        open_cohort(store, cat, sha, pin["run_id"], pin["manifest_sha256"])
    assert not marker.exists()
    marker.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        marker,
        {
            "schema": "el-stage2-test-opening-1",
            "catalog_sha256": sha,
            "test_family_ids": g["names"],
            "freeze_run_id": pin["run_id"],
            "freeze_manifest_sha256": pin["manifest_sha256"],
            "method_member": "payload/method-freeze.json",
            "method_sha256": method_sha,
        },
    )
    with pytest.raises(ValueError, match=reason):
        sources.test_authority([store, shelf], (cat, sha))


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("swapped_external", "external process cost has a different"),
        ("wrong_raster", "native risk, support and threshold states"),
    ],
)
def test_sealed_but_unrelated_or_non_native_cost_cannot_open(tmp_path, fault, reason):
    from stage2_opening import open_cohort
    from vision_gate_fixture import gate_fixture

    g = gate_fixture(tmp_path, cost_fault=fault)
    pin, _, _ = g["freeze"]()
    with pytest.raises(ValueError, match=reason):
        open_cohort(g["store"], g["catalog"], g["catalog_sha"], pin["run_id"], pin["manifest_sha256"])
    assert not (g["store"] / "protocol/stage2" / g["catalog_sha"] / "test-opening.json").exists()


def test_original_native_and_process_costs_reach_the_scientific_freeze(tmp_path):
    from vision_method import cost_evidence
    from vision_gate_fixture import gate_fixture

    g = gate_fixture(tmp_path)
    _, method, _ = g["freeze"]()
    ds = json.loads(g["dev_data"].read_bytes())
    for mode in ("L2-frozen", "L2-vision-ft"):
        model = method["models"][mode]
        prior = method["cpu_costs"][mode]
        root = g["shelf"] / "runs" / model["run_id"] / "payload/development-scores"
        views = json.loads((root / "scores.json").read_bytes())["views"]
        produced = cost_evidence(
            [g["store"], g["shelf"]],
            prior["pin"],
            prior["family"],
            {"fit": {**model, "branch": mode, "threshold": 0.5}},
            ds,
            (root, {(r["family"], r["condition"], r["view"]): r for r in views}, None),
        )
        assert produced == prior
        assert (
            produced["external_sample_interval_seconds"]
            == produced["native_rss_sample_interval_seconds"]
            == 0.01
        )
