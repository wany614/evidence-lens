"""Only disposable fixture evidence; these tests never touch the scientific store."""

import json
from pathlib import Path

import numpy as np
import pytest

from evidence_lens.scoring_io import TARGETS, file_hash, write_json
from test_vision_data import dataset_fixture, fixture_truth
from vision_confirmation import COHORTS, history_evaluation, numeric_difference, paired_perturbations, work
from vision_data import assetspec
from vision_evaluate import METHODS, FIVE, SCORE_SETS, ranking, score_sources
from vision_method import independent_development, SEEDS
from vision_public_study import flatten, table
from vision_replay import threshold_readback


def read(p):
    return json.loads(Path(p).read_text())


def test_history_and_new_reject_test_before_any_selection_or_dataset_open(tmp_path):
    cat = tmp_path / "catalog.json"
    write_json(
        cat,
        {
            "schema": "el-vision-family-roster-1",
            "families": [{"scene_family_id": f"new{i}", "split": "test"} for i in range(8)],
        },
    )
    for cohort in COHORTS:
        for action in ("materials", "transport", "score", "evaluate", "boards"):
            with pytest.raises(ValueError, match="test remains sealed"):
                work(
                    action,
                    {"cohort": cohort, "catalog_sha256": file_hash(cat)},
                    (tmp_path, tmp_path / "cold"),
                    cat,
                    tmp_path / "missing_old_catalog",
                    tmp_path / "missing_output",
                )
    assert not (tmp_path / "missing_output").exists()


def test_all_eight_native_candidates_independent_domains_and_one_bad_member(tmp_path):
    root, ds_sha, _cat, pins = dataset_fixture(tmp_path, "development", ["A", "B"], True)
    ds = read(root / "dataset.json")
    y = (fixture_truth()["rgb_high_error"] == 1).ravel()
    chosen = {}
    cands = []
    for i, (tag, src) in enumerate(zip(SCORE_SETS[1:], pins[1:])):
        native = read(src[0])
        p = src[0].parent / native["views"][0]["score"]["path"]
        risk = np.load(p)["risk"].ravel()
        ap = ranking(risk, y)
        tau = threshold_readback({"a": risk}, {"a": len(risk)})
        for j, (seed, rate) in enumerate(
            [(SEEDS[0], 1e-4), (SEEDS[0], 3e-4), (SEEDS[1], 1e-4), (SEEDS[2], 1e-4)]
        ):
            rid = f"c{i}-{j}"
            fit = {
                "run_id": rid,
                "model_sha256": native["model_sha256"],
                "threshold": tau["tau"],
                "branch": tag,
                "seed": seed,
                "head_lr": rate,
                "backbone_lr": 0,
                "best_epoch": 1,
                "best_development_family_equal_ap": ap["ap"],
            }
            cands.append(
                {
                    "fit": fit,
                    "score_pin": src,
                    "dev": {
                        "threshold": tau,
                        "views": {
                            f: {v: {**ap, "V": len(y), "E": len(y)} for v in TARGETS} for f in ("A", "B")
                        },
                    },
                }
            )
            if j == 0:
                chosen[tag] = fit
    r = independent_development(root, ds, ds_sha, pins[0], cands, {"chosen": chosen})
    assert len(r["candidates"]) == 8 and set(r["primary_domains"]) == set(METHODS)
    for m, rec in r["primary_domains"].items():
        assert (
            rec["own"]["macro"]["defined_families"] == 2 and rec["seven_common"]["views"]["B"]["v06"]["n"] == 8
        )
        assert (rec["five_common"] is None) == (m not in FIVE)
    assert r["original_thresholds"]["constant"]["coverage"] == 1
    cands[0]["dev"]["views"]["A"]["v04"]["n"] = 7
    with pytest.raises(ValueError, match="AP or support"):
        independent_development(root, ds, ds_sha, pins[0], cands, {"chosen": chosen})


def test_history_family_condition_curve_export_and_failed_view(tmp_path):
    root, _, _cat, initial = dataset_fixture(tmp_path, "development", ["past"], True)
    ds = read(root / "dataset.json")
    ref = ds["views"]
    ds["views"] = [
        {**r, "condition": c, "scientific_disposition": "selected", "failure_kind": None}
        for c in ("B0", "R1", "Q1")
        for r in ref
    ]
    fail = ds["views"][-1]
    blank = fixture_truth(np.zeros((2, 4), bool))
    np.savez_compressed(root / "failed.npz", **blank)
    fail.update(
        status="render_failed",
        rgb=None,
        E=0,
        reason="deliberate fixture failure",
        scientific_disposition="terminal_upstream_failure",
        failure_kind="render",
        full={"deploy": None, "truth": assetspec(root / "failed.npz", root)},
    )
    write_json(root / "hist.json", ds)
    h = file_hash(root / "hist.json")
    pins = []
    tau = {m: 0.5 for m in METHODS}
    for tag, pin in zip(SCORE_SETS, initial):
        start = read(pin[0])
        r = start["views"][0]
        doc = {
            **start,
            "dataset_sha256": h,
            "views": [
                {**r, "family": q["family"], "condition": q["condition"], "view": q["view"]}
                for q in ds["views"]
            ],
        }
        doc["views"][-1].update(
            status="render_failed",
            score=None,
            rgb_sha256=None,
            deploy_sha256=None,
            reason="deliberate fixture failure",
        )
        pp = pin[0].parent / "hist-scores.json"
        write_json(pp, doc)
        pins.append((pp, file_hash(pp)))
        if tag != "original-five":
            a = np.load(pp.parent / r["score"]["path"])["risk"]
            tau[tag] = threshold_readback({"a": a.ravel()}, {"a": a.size})["tau"]
    srcs = score_sources(ds, h, pins)
    out = tmp_path / "out"
    history_evaluation(root, ds, h, srcs, tau, "9" * 64, "8" * 64, out)
    fam = read(out / "families.json")
    status = read(out / "status.json")
    rows = flatten(COHORTS[1], {("past", c): None for c in ("B0", "R1", "Q1")}, fam, status, tau)
    assert len(rows) == 21
    row = next(r for r in rows if r["condition_id"] == "Q1" and r["method"] == "L2-frozen")
    assert (row["V"], row["E"], row["invalid"], row["status"], row["terminal_upstream_views"]) == (
        48,
        40,
        8,
        "partial",
        1,
    )
    assert row["five_common_ap"] is None and row["seven_common_defined_views"] == 5
    assert read(out / "past/Q1/curves.json")["L2-frozen"]["frozen_L"] == row["L"]
    table(tmp_path / "rows.csv", rows)
    pairs = paired_perturbations(root, ds, srcs, tau)
    q = next(r for r in pairs if r["condition"] == "Q1")
    assert q["paired"]["rgb"]["defined_views"] == 5 and q["paired"]["rgb"]["bitwise_identical"]
    a = np.array([0.0, -0.0, 0.9, np.nan], np.float32)
    b = np.array([0.0, 0.0, 0.8, 1], np.float32)
    z = numeric_difference(a, b, np.isfinite(a) & np.isfinite(b))
    assert z["paired_values"] == 3 and z["changed_bits"] == 2
    bad = read(out / "families.json")
    bad[0]["methods"]["L0"]["counts"]["L"] += 3
    with pytest.raises(ValueError, match="denominators"):
        flatten(COHORTS[1], {("past", c): None for c in ("B0", "R1", "Q1")}, bad, status, tau)


def test_fitted_lr_export_keeps_native_scores_and_whole_ties(tmp_path, monkeypatch):
    import vision_baselines
    from evidence_lens.l1 import load_package
    from vision_evaluate import freeze_dev
    from vision_method import export_lr

    train, tsha, tc = dataset_fixture(tmp_path, "train", [f"tf{i:02}" for i in range(28)], formal=True)
    fit = tmp_path / "fit"
    receipt = vision_baselines.fit(train, tsha, [tc], fit, "disposable-original-fit")
    assert receipt["status"] == "fitted"
    dev, dsha, dc = dataset_fixture(tmp_path, "development", [f"df{i}" for i in range(8)], formal=False)
    monkeypatch.setattr(
        vision_baselines, "full_cohort", lambda *a: None
    )  # Keep a two-by-four native fixture grid.
    outputs = []
    for label, package in (("before", fit / "model"), ("after", tmp_path / "frozen/model")):
        if label == "after":
            threshold = tmp_path / "before-threshold/thresholds.json"
            export_lr(
                fit / "model", threshold, dev, dsha, tmp_path / "frozen", "disposable-logistic-deployment"
            )
            reopened = load_package(package)
            assert reopened["index"]["model_id"] == "disposable-logistic-deployment"
            assert all(
                file_hash(package / n) == file_hash(fit / "model" / n)
                for n in ("feature_schema.json", "L1-full.json", "L1-geometry.json")
            )
        sc = tmp_path / (label + "-score")
        result = vision_baselines.score(
            dev, dsha, package, file_hash(package / "model.json"), sc, "development"
        )
        t = freeze_dev(
            dev,
            dsha,
            [(sc / "scores.json", file_hash(sc / "scores.json"))],
            [dc],
            tmp_path / (label + "-threshold"),
            "fixture-" + label,
            baseline_only=True,
        )
        outputs.append((result, t, sc))
    assert outputs[0][1]["choices"] == outputs[1][1]["choices"]
    for a, b in zip(outputs[0][0]["views"], outputs[1][0]["views"]):
        with (
            np.load(outputs[0][2] / a["score"]["path"]) as first,
            np.load(outputs[1][2] / b["score"]["path"]) as second,
        ):
            assert first.files == second.files and all(
                np.array_equal(first[n], second[n], equal_nan=True) for n in first.files
            )
