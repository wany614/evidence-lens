"""Tiny synthetic full-matrix tests. No project research store or scientific prediction is read."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from test_cpu_p2 import deployable, reference

from evidence_lens import l1
from evidence_lens.evaluation import run_evaluation
from evidence_lens.formal_evaluation import field_change, run
from evidence_lens.formal_inputs import CONDITIONS, protocol, signature, stage
from evidence_lens.formal_metrics import (
    METHODS,
    curve_arrays,
    pool_family,
    rank,
    view_statistics,
)
from evidence_lens.formal_report import assemble, build
from evidence_lens.formal_roster import roster
from evidence_lens.labels import label_view
from evidence_lens.risk_model import run_score
from evidence_lens.scoring_io import VIEWS, file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]


def test_ties_both_directions_single_class_and_empty():
    y = np.array([1, 0, 1, 0], bool)
    mask = np.ones(4, bool)
    perfect = np.array([1, 0.1, 1, 0.1], np.float32)
    assert rank(perfect, y, mask)[0]["ap"] == 1
    rec, group = rank(np.ones(4, np.float32), y, mask)
    assert rec["ap"] == 0.5 and rec["unique_score_count"] == 1
    assert rank(perfect, np.ones(4, bool), mask)[0]["ap"] is None
    assert rank(perfect, y, ~mask)[0]["status"] == "empty_mask"
    curves = curve_arrays(group, 6, 1.0)
    assert (
        curves["accepted"].tolist() == [0, 4]
        and curves["errors"].tolist() == [0, 2]
        and curves["frozen_index"] == 0
    )
    assert curve_arrays(group, 6, float(np.nextafter(1.0, 2.0)))["frozen_index"] == 1
    assert field_change(np.array([0.0], np.float32), np.array([-0.0], np.float32))["bitwise_different"] == 1


def test_family_pools_system_counts_but_gives_each_two_class_view_one_vote():
    samples = {}
    for idx, width in enumerate((2, 8)):
        shape = (1, width)
        ref = {
            "rgb": np.zeros((*shape, 3), np.float32),
            "z": np.ones(shape, np.float32),
            "position": np.zeros((*shape, 3), np.float32),
            "visibility": np.ones(shape, np.uint8),
            "edge": np.zeros(shape, bool),
        }
        pred = {"rgb": ref["rgb"].copy(), "z": ref["z"].copy(), "alpha": np.ones(shape, np.float32)}
        pred["rgb"][0, 0] = 0.1
        labels = label_view(ref, pred, True)
        score = (labels["rgb_high_error"] == 1).astype(np.float32)
        if idx:
            score = 1 - score
        scores = {m: score for m in METHODS}
        rows, _ = view_statistics(labels, scores, np.zeros(shape, np.uint8), dict.fromkeys(METHODS, 0.5))
        samples[str(idx)] = {"regions": rows}
    result = pool_family(samples)["all"]
    assert result["counts"]["V"] == 10 and result["outcomes"]["rgb_error_incidence"]["value"] == 0.2
    assert result["methods"]["L0"]["common_ranking"]["ap"] == pytest.approx((1 + 1 / 8) / 2)
    assert result["methods"]["L0"]["common_ranking"]["eligible_view_count"] == 2


def rid(i):
    return f"20200101T000000000000Z-{i:016x}"


def seal(store, index, kind, identity, files, failed=False):
    run_id = rid(index)
    folder = store / "runs" / run_id
    folder.mkdir(parents=True)
    for path, original in files.items():
        dest = folder / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, dest)
    status = "failed" if failed else "succeeded"
    write_json(
        folder / "run.json",
        dict(
            identity,
            run_id=run_id,
            run_kind=kind,
            execution_status=status,
            elapsed_seconds_monotonic=0.1,
            failure_reason="test whole failure" if failed else None,
        ),
    )
    rows = [
        dict(path=p.relative_to(folder).as_posix(), **signature(p))
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    ]
    write_json(folder / "manifest.json", {"run_id": run_id, "files": rows})
    pin = {"run_id": run_id, "manifest_sha256": file_hash(folder / "manifest.json")}
    write_json(folder / "completion.json", dict(pin, execution_status=status))
    return pin


def original_files(directory):
    return {"payload/" + p.relative_to(directory).as_posix(): p for p in directory.rglob("*") if p.is_file()}


def write_fake_model(folder, model_id, catalog_sha):
    folder.mkdir()
    write_json(folder / "feature_schema.json", l1.FEATURE_SCHEMA)
    for name in l1.MODELS:
        numeric, cols = l1.columns(name)
        coeff = [0.5 if c in ("local_rgb_difference", "c_behind") else 0 for c in cols]
        write_json(
            folder / (name + ".json"),
            {
                "schema": "el-l1-coefficients-1",
                "method": name,
                "columns": cols,
                "numeric_columns": numeric,
                "feature_schema_digest": l1.schema_digest(),
                "medians": [0] * len(numeric),
                "means": [0] * len(numeric),
                "scales": [1] * len(numeric),
                "coefficients": coeff,
                "intercept": -0.12,
                "classes": [0, 1],
                "C": 0.1,
            },
        )
    thresholds = {m: 0.5 for m in METHODS if m != "L0"}
    write_json(
        folder / "thresholds.json",
        {
            "schema": "el-l1-thresholds-1",
            "fit_run_id": rid(991),
            "threshold_run_id": model_id,
            "split_manifest_sha256": catalog_sha,
            "thresholds": thresholds,
        },
    )
    write_json(
        folder / "model.json",
        {
            "schema": "el-l1-package-1",
            "status": "frozen",
            "model_id": model_id,
            "fit_run_id": rid(991),
            "threshold_run_id": model_id,
            "split_manifest_sha256": catalog_sha,
            "training_manifest_sha256": "0" * 64,
            "feature_schema_digest": l1.schema_digest(),
            "files": {p.name: file_hash(p) for p in folder.iterdir()},
            "constant_risk": 0.4,
        },
    )


def build_fixture_report(
    tmp_path,
    *,
    frozen_model_dir=None,
    catalog_path=None,
    study_path=None,
    method_freeze_path=None,
    model_pin=None,
    fit_pin=None,
    store=None,
    checkout=ROOT,
    report_run_id=None,
):
    """Exercise the real P6/P7 numerical pipeline with 3x2 synthetic inputs and four explicit family rows."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = Path(store) if store is not None else tmp_path / "fixture-store"
    work = tmp_path / "examples"
    work.mkdir()
    template = tmp_path / "source"
    template.mkdir()
    deployable(template)
    if catalog_path is None:
        catalog = tmp_path / "catalog.json"
        families = [f"fixture-test-{i}" for i in range(4)]
        rows = [
            {
                "scene_family_id": f,
                "scene_id": f + "/v001",
                "split": "test",
                "example_view": "v01",
                "camera_sha256": file_hash(template / "cameras.json"),
                "trials": {c: f + "-" + c for c in CONDITIONS},
            }
            for f in families
        ]
        write_json(catalog, {"dataset_id": "synthetic-fixture", "families": rows})
    else:
        catalog = Path(catalog_path)
    rows = {
        r["scene_family_id"]: r for r in json.loads(catalog.read_text())["families"] if r["split"] == "test"
    }
    ids = list(rows)
    assert len(ids) == 4 and all(
        r["camera_sha256"] == file_hash(template / "cameras.json") for r in rows.values()
    )
    if study_path is None:
        study = tmp_path / "study.json"
        write_json(study, {"test_methods": list(METHODS)})
    else:
        study = Path(study_path)
    if frozen_model_dir is None:
        model = tmp_path / "model"
        write_fake_model(model, rid(1), file_hash(catalog))
        model_pin = seal(
            store,
            1,
            "threshold_select",
            {"trial_id": "fixture-model"},
            {"payload/model/" + p.name: p for p in model.iterdir()},
        )
    else:
        model = Path(frozen_model_dir)
        assert model_pin and fit_pin
    if method_freeze_path is None:
        method = tmp_path / "method.json"
        write_json(
            method,
            {
                "schema": "el-method-freeze-1",
                "catalog_sha256": file_hash(catalog),
                "study_sha256": file_hash(study),
                "comparison_protocol_sha256": protocol(checkout)["comparison_protocol_sha256"],
                "model_id": model_pin["run_id"],
                "threshold_select_run_id": model_pin["run_id"],
                "threshold_select_manifest_sha256": model_pin["manifest_sha256"],
            },
        )
    else:
        method = Path(method_freeze_path)
    assert protocol(checkout)["comparison_protocol_sha256"] == protocol(ROOT)["comparison_protocol_sha256"]
    protocol_inputs = {
        k: {"path": str(v), "sha256": file_hash(v)}
        for k, v in (("catalog", catalog), ("study", study), ("method_freeze", method))
    }
    report_rows = []
    trial_markers = {}
    for fi, family in enumerate(ids):
        base = {}
        for ci, c in enumerate(CONDITIONS):
            prefix = 20 + fi * 30 + ci * 8
            trial = rows[family]["trials"][c]
            ident = {
                "trial_id": trial,
                "scene_family_id": family,
                "scene_id": rows[family]["scene_id"],
                "condition_id": c,
                "dataset_hash": file_hash(catalog),
                "split": "test",
            }
            local = work / trial
            local.mkdir()
            pred = deployable(local)
            prediction_file = pred / "prediction.json"
            pred_doc = json.loads(prediction_file.read_text())
            pred_doc["producer_run_id"] = rid(prefix)
            if c == "Q1":
                vv = pred / "v01.npz"
                with np.load(vv) as z:
                    t = {k: z[k] for k in z.files}
                t["rgb"][0, 0, 0] += 0.01
                np.savez_compressed(vv, **t)
            if fi == 2 and c == "R1":
                pred_doc["failed_views"] = {"v01": "fixture target missing"}
                (pred / "v01.npz").unlink()
            pred_doc["views"] = {
                v: {"sha256": file_hash(pred / (v + ".npz"))} for v in VIEWS if (pred / (v + ".npz")).exists()
            }
            write_json(prediction_file, pred_doc)
            ref = reference(local)
            ref_doc = json.loads((ref / "reference.json").read_text())
            ref_doc.update(scene_family_id=family, scene_id=rows[family]["scene_id"])
            write_json(ref / "reference.json", ref_doc)
            if c == "B0":
                bound_ref = seal(
                    store,
                    prefix + 1,
                    "scene_render",
                    ident,
                    original_files(ref)
                    | {
                        "payload/input.png": local / "input.png",
                        "payload/cameras.json": local / "cameras.json",
                    },
                )
            whole = fi == 3 and c == "R1"
            ppin = seal(store, prefix, "render", ident, {} if whole else original_files(pred), failed=whole)
            failure = "test whole failure" if whole else None
            parents = {"reference": bound_ref, "model": model_pin, "prediction": ppin}
            if not whole:
                scores = local / "rules"
                run_score(local / "input.png", local / "cameras.json", pred, scores, rid(prefix + 2))
                learned = local / "learned"
                l1.run(
                    local / "input.png",
                    local / "cameras.json",
                    pred,
                    model,
                    learned,
                    rid(prefix + 3),
                    file_hash(model / "model.json"),
                )
                parents["rules"] = seal(store, prefix + 2, "model_score", ident, original_files(scores))
                parents["learned"] = seal(store, prefix + 3, "model_score", ident, original_files(learned))
            ev = local / "labels"
            run_evaluation(
                local / "input.png",
                local / "cameras.json",
                ref,
                None if whole else pred,
                None if whole else scores,
                ev,
                ident | {"run_id": rid(prefix + 4)},
                render_failed=failure,
            )
            parents["labels"] = seal(store, prefix + 4, "evaluation", ident, original_files(ev))
            if c == "B0":
                base = {k: parents[k] for k in ("prediction", "learned", "rules")}
            if c == "Q1":
                parents.update({"paired_" + k: v for k, v in base.items()})
            sel = {
                "schema": "el-formal-comparison-selection-1",
                "kind": "synthetic_fixture",
                "scene_family_id": family,
                "condition_id": c,
                "parents": parents,
                "protocol_files": protocol_inputs,
                "source": {"parent_role": "reference", "path": "payload/input.png"},
                "cameras": {"parent_role": "reference", "path": "payload/cameras.json"},
                "render_failure_reason": failure,
            }
            sp = local / "selection.json"
            write_json(sp, sel)
            staged = local / "staged"
            stage(store, sp, staged, checkout)
            result = run(staged, local / "result", rid(prefix + 5))
            stats = result["family"]["all"]
            if whole:
                assert stats["counts"]["V"] == 36 and stats["counts"]["E"] == 0
                assert stats["outcomes"]["invalid_fraction"]["value"] == 1
                assert stats["outcomes"]["confirmed_error_fraction_of_V"]["value"] == 0
                assert stats["outcomes"]["rgb_error_incidence"]["value"] is None
                assert stats["methods"]["L1-full"]["system"]["low_coverage"]["value"] == 0
                assert stats["methods"]["L1-full"]["common_ranking"]["ap"] is None
            elif c == "Q1":
                assert result["views"]["v01"]["paired_Q1_minus_B0"]["geometry_bitwise_equal"]
                assert result["views"]["v01"]["paired_Q1_minus_B0"]["rgb_mae"] > 0
                checks = [
                    result["source_Q1_minus_B0"],
                    *(v["paired_Q1_minus_B0"] for v in result["views"].values()),
                ]
                expected_fields = {
                    "L0",
                    "1-alpha",
                    "l0_reason",
                    "scoreable",
                    "frame_relation",
                    *l1.columns("L1-geometry")[1],
                }
                for trial_check in checks:
                    assert expected_fields <= set(trial_check["fields"])
                    assert all(trial_check["fields"][k]["bitwise_different"] == 0 for k in expected_fields)
            else:
                assert stats["counts"]["V"] == 36 and stats["counts"]["invalid"] >= 6
            if fi == 2 and c == "R1":
                assert (
                    result["views"]["v01"]["failure_reason"] == "fixture target missing"
                    and (local / "result/board/board_v01.png").is_file()
                )
            if fi == 0 and c == "B0":
                # This target has five E pixels, three common-scoreable pixels, one unknown true error and one invalid output.
                view = result["views"]["v01"]["regions"]["all"]
                assert view["counts"]["E"] == 5 and view["counts"]["V"] == 6 and view["counts"]["Y"] == 2
                assert view["methods"]["L0"]["counts"]["UY"] == 1
                assert len({m["common_ranking"]["n"] for m in view["methods"].values()}) == 1
                assert view["methods"]["constant"]["common_ranking"]["ap"] == pytest.approx(
                    view["outcomes"]["common_prevalence"]["value"]
                )
                with np.load(local / "result/curves/family.npz") as z:
                    assert len(z["constant__accepted"]) == 2 and int(z["constant__V"]) == 36
            pin = seal(store, prefix + 5, "evaluation", ident, original_files(local / "result"))
            report_rows.append(dict(pin, scene_family_id=family, condition_id=c))
            if not whole:
                selection = {
                    "schema": "el-p2-trial-selection-1" if c == "B0" else "el-condition-selection-1",
                    "selection_status": "selected",
                    "scene_family_id": family,
                    "trial_id": trial,
                    "condition_id": c,
                    "selected": {
                        dst: parents[src]
                        for src, dst in (
                            ("reference", "reference"),
                            ("prediction", "render"),
                            ("rules", "score"),
                            ("labels", "evaluation"),
                        )
                    },
                    "attempts": [parents[k] for k in ("reference", "prediction", "rules", "labels")],
                }
                marker = local / "canonical.json"
                write_json(marker, selection)
                trial_markers[family, c] = {"path": str(marker), "sha256": file_hash(marker)}
    proof = tmp_path / "recovery.json"
    write_json(
        proof,
        {
            "index": {"runs": 10, "receipts": 24},
            "compared_files": 21,
            "private_hostname": "confidential-machine-fixture",
        },
    )
    timing_spec = tmp_path / "timing.json"
    timing_id = {
        "trial_id": ids[0] + "-B0-timing-01",
        "scene_family_id": ids[0],
        "condition_id": "B0",
        "experiment_id": "p6-timing-repeats",
    }
    timed_gen = seal(store, 940, "generation", timing_id, {})
    timed_render = seal(store, 941, "render", timing_id, {})
    write_json(
        timing_spec,
        {
            "schema": "el-timing-repeat-1",
            "trial_id": timing_id["trial_id"],
            "scene_family_id": ids[0],
            "condition_id": "B0",
            "timing_repeat": 1,
            "included_in_quality_or_training": False,
            "warmup": {"selection_sha256": trial_markers[ids[0], "B0"]["sha256"]},
            "catalog_sha256": file_hash(catalog),
            "method_freeze_sha256": file_hash(method),
            "attempts": [timed_gen, timed_render],
            "producer": timed_gen,
            "render": timed_render,
            "producer_elapsed_seconds": 0.1,
            "render_elapsed_seconds": 0.2,
        },
    )
    fitpin = fit_pin or seal(store, 991, "model_fit", {"trial_id": "fixture-fit"}, {})
    timing_cpu_config = tmp_path / "timing-model-config.json"
    write_json(
        timing_cpu_config,
        {
            "timing_repeat": 1,
            "quality_result": False,
            "timing_marker_sha256": file_hash(timing_spec),
            "render": timed_render,
        },
    )
    timed_cpu = seal(
        store,
        942,
        "model_score",
        dict(timing_id, experiment_id="p6-timing-1-model-score"),
        {"config.resolved.yaml": timing_cpu_config},
    )
    definitions = []
    for row in report_rows:
        f, c = row["scene_family_id"], row["condition_id"]
        item = {
            "scene_family_id": f,
            "condition_id": c,
            "comparison": {k: row[k] for k in ("run_id", "manifest_sha256")},
        }
        if (f, c) in trial_markers:
            item["quality"] = trial_markers[f, c]
        definitions.append(item)
    config = {
        "schema": "el-study-report-roster-1",
        "kind": "synthetic_fixture",
        "trials": definitions,
        "protocol_files": protocol_inputs,
        "inventory_scope": "synthetic test matrices only",
        "model_fit": fitpin,
        "threshold_select": model_pin,
        "proofs": [
            {
                "id": "fixture-restore",
                "kind": "recovery",
                "path": str(proof),
                "sha256": file_hash(proof),
                "measures": {"verified_runs": "/index/runs", "compared_files": "/compared_files"},
            }
        ],
        "timing_markers": [{"path": str(timing_spec), "sha256": file_hash(timing_spec)}],
        "cpu_timing": [timed_cpu],
    }
    roster_in = tmp_path / "roster.json"
    write_json(roster_in, config)
    report_sel = tmp_path / "report-selection.json"
    rs = roster(store, roster_in, report_sel)
    assert len(rs["scientific_markers"]) == 11 and len(rs["trials"]) == 12
    package = tmp_path / "report-bundle"
    assemble(store, report_sel, package, checkout)
    output = tmp_path / "report"
    result = build(package, output, report_run_id or rid(900))
    return {
        "output": output,
        "staged": package,
        "store": store,
        "result": result,
        "selection": report_sel,
        "roster": roster_in,
        "families": ids,
    }


def test_formal_staging_scoring_failure_and_explicit_report(tmp_path):
    info = build_fixture_report(tmp_path)
    output, package, store, result, report_sel, ids = (
        info[k] for k in ("output", "staged", "store", "result", "selection", "families")
    )
    rs = json.loads(report_sel.read_text())
    assert (
        result["kind"] == "synthetic_fixture"
        and result["quality_trials"] == 12
        and result["completed_targets"] == 65
    )
    assert (output / "figures/coverage-B0.png").is_file() and (
        output / "figures/fixture-test-2-R1-v01.png"
    ).is_file()
    public = output / "public-numerical"
    assert all(b"confidential-machine-fixture" not in p.read_bytes() for p in public.iterdir())
    assert (output / "numeric/proofs/fixture-restore.json").is_file()
    assert "verified_runs,10" in (public / "measured-receipts.csv").read_text()
    assert result["evidence_availability"]["Recovery receipt"]
    assert "Synthetic software validation" in (output / "report.md").read_text()
    assert result["files"]["numeric/Q1-field-response.csv"]["sha256"] == file_hash(
        output / "numeric/Q1-field-response.csv"
    )
    with (public / "family-results.csv").open() as fp:
        family_rows = list(csv.DictReader(fp))
    with (public / "family-equal-summary.csv").open() as fp:
        macro = list(csv.DictReader(fp))
    fsel = [r for r in family_rows if r["condition"] == "R1" and r["region"] == "all" and r["method"] == "L0"]
    overall = next(
        r
        for r in macro
        if r["condition"] == "R1"
        and r["region"] == "all"
        and r["method"] == "L0"
        and r["metric"] == "confirmed_error_fraction_of_V"
    )
    assert len(fsel) == 4 and float(overall["mean"]) == pytest.approx(
        np.mean([float(r["Y"]) / float(r["V"]) for r in fsel])
    )
    for idx, key in enumerate(("timing_marker_sha256", "render")):
        bad_cfg = json.loads((tmp_path / "timing-model-config.json").read_text())
        bad_cfg[key] = (
            "0" * 64 if key == "timing_marker_sha256" else {"run_id": rid(940), "manifest_sha256": "0" * 64}
        )
        bp = tmp_path / f"unmatched-cpu-{idx}.json"
        write_json(bp, bad_cfg)
        badpin = seal(
            store,
            948 + idx,
            "model_score",
            {
                "trial_id": ids[0] + "-B0-timing-01",
                "scene_family_id": ids[0],
                "condition_id": "B0",
                "experiment_id": "p6-timing-1-model-score",
            },
            {"config.resolved.yaml": bp},
        )
        config = json.loads(info["roster"].read_text())
        config["cpu_timing"] = [badpin]
        path = tmp_path / f"bad-roster-{idx}.json"
        write_json(path, config)
        with pytest.raises(ValueError, match="matching Windows repeat"):
            roster(store, path, tmp_path / f"bad-output-{idx}.json")
    rs["trials"] = rs["trials"][:-1]
    write_json(report_sel, rs)
    with pytest.raises(ValueError, match="all 12"):
        assemble(store, report_sel, tmp_path / "truncated", ROOT)
    bad = package / "trials" / ids[0] / "B0/curves/family.npz"
    bad.write_bytes(bad.read_bytes() + b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        build(package, tmp_path / "tampered", rid(901))
