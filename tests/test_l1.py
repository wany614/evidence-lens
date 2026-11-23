"""Ephemeral synthetic validation of P4/P5. These cases never use the research store."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("sklearn", minversion="1.9.1")
pytest.importorskip("pyarrow")
from evidence_lens.evaluation import run_evaluation
from evidence_lens.l1 import ALL_COLUMNS, MODELS, load_package, risk_state, transform
from evidence_lens.l1_data import Dataset, prepare, sample_indices
from evidence_lens.l1_fit import choose_c, fit, folds_for, preprocess, weights
from evidence_lens.l1_freeze import freeze, threshold_for
from evidence_lens.risk_model import run_score
from evidence_lens.scoring_io import TARGETS, file_hash, write_json
from test_cpu_p2 import deployable, reference


def test_sampling_weights_fallback_and_fold_preprocessing():
    eligible = np.ones((100, 100), bool)
    a = sample_indices(eligible, "fixture-a", "v01", "0"*64)
    assert len(a) == 4096 and (np.diff(a) > 0).all()
    assert np.array_equal(a, sample_indices(eligible, "fixture-a", "v01", "0"*64))
    seed_bytes = b"9184\n" + b"0" * 64 + b"\nfixture-a\nv01"
    oracle_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:16], "big")
    oracle = np.sort(np.random.Generator(np.random.PCG64(oracle_seed)).choice(np.arange(10000), 4096, replace=False))
    np.testing.assert_array_equal(a, oracle)
    assert not np.array_equal(a, sample_indices(eligible, "fixture-a", "v02", "0"*64))
    w = weights(np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0]))
    np.testing.assert_allclose(w, [.5, .5, 1, 2])
    x = np.zeros((4, len(ALL_COLUMNS)), np.float32)
    x[:, 0] = [0, 1, 3, np.nan]
    x[:, 1] = np.nan
    x[:, -1] = [1, 0, 0, 1]
    p = preprocess(x, w, "L1-full")
    assert p["medians"][:2] == [1, 0] and p["all_missing"][1] and p["scales"][1] == 1
    assert p["means"][0] == pytest.approx((.5 + 3 + 2) / 4)
    np.testing.assert_array_equal(transform(x, p)[:, -1], x[:, -1])
    ds = SimpleNamespace(families={str(i): {"pin": {"cv_fold": fold}} for i, fold in enumerate([0, 1, 2, 0, 1, 2, 3, 3])})
    f = np.arange(8)
    y = np.isin(f, [0, 3]).astype(np.uint8)
    groups, tries = folds_for(ds, f, y)
    assert [op["usable_training_folds"] for op in tries] == [False, False, True] and groups.max() == 1
    assert folds_for(ds, f, np.zeros(8))[0] is None
    assert choose_c({0.1: .7, 1: .7000005, 10: .6}) == .1 and choose_c({.1: None, 1: None}) is None


def test_exact_family_thresholds_and_unknown():
    values = {"fixture": np.array([.1]*7 + [.3]*2 + [.9], np.float32)}
    r = threshold_for(values, {"fixture": 10})
    assert r["coverage"] == .7 and (risk_state(values["fixture"], r["tau"]) == 0).sum() == 7
    r = threshold_for({"a": np.array([.1]*8, np.float32), "b": np.array([.5]*95, np.float32)}, {"a": 10, "b": 100})
    assert r["coverage"] == .875  # (8/10 + 95/100)/2, each family gets one vote
    tau = threshold_for({"a": np.ones(3, np.float32)}, {"a": 3})["tau"]
    assert tau > 1 and risk_state(np.array([1, np.nan], np.float32), tau).tolist() == [0, 255]
    assert threshold_for({"a": np.empty(0, np.float32)}, {"a": 20})["coverage"] == 0
    between = np.array([0.5, np.nextafter(np.float32(0.5), np.float32(1.0))], np.float32)
    tau = threshold_for({"a": np.repeat(between, [4, 1])}, {"a": 5})["tau"]
    assert risk_state(between, tau).tolist() == [0, 1]


def make_extract(raw, dest, catalog, row, color, single_class=False):
    family, trial = row["scene_family_id"], row["trial_id"]
    pred = deployable(raw)
    if single_class:
        for view in TARGETS:
            path = pred / (view + ".npz")
            with np.load(path) as doc:
                arrays = {k: doc[k] for k in doc.files}
            arrays["rgb"].fill(0)
            np.savez_compressed(path, **arrays)
    image = raw / "input.png"
    Image.fromarray(np.full((2, 3, 3), color, np.uint8)).save(image)
    p = json.loads((pred / "prediction.json").read_text())
    p.update(input_sha256=file_hash(image), producer_run_id=f"fixture-render-{family}")
    write_json(pred / "prediction.json", p)
    ref = reference(raw)
    r = json.loads((ref / "reference.json").read_text())
    r.update(scene_id=row["scene_id"], scene_family_id=family)
    write_json(ref / "reference.json", r)
    scored, evaluated = raw / "scores", raw / "evaluated"
    sid, eid = f"fixture-scores-{family}", f"fixture-eval-{family}"
    run_score(image, raw / "cameras.json", pred, scored, sid)
    identity = dict(run_id=eid, trial_id=trial, scene_id=row["scene_id"], scene_family_id=family, condition_id="B0")
    run_evaluation(image, raw / "cameras.json", ref, pred, scored, evaluated, identity)
    pin = {"schema": "el-learning-family-1", "split": "train" if row["split"] == "train" else "dev",
           "scene_id": row["scene_id"], "scene_family_id": family, "condition_id": "B0", "trial_id": trial,
           "cv_fold": row["cv_fold"], "score_run_id": sid, "evaluation_run_id": eid, "split_manifest_sha256": file_hash(catalog),
           "score_receipt_sha256": file_hash(scored / "score.json"), "evaluation_receipt_sha256": file_hash(evaluated / "evaluation.json"),
           "selection_sha256": "a"*64, "score_manifest_sha256": "b"*64, "evaluation_manifest_sha256": "c"*64}
    write_json(raw / "family-pin.json", pin)
    prepare(raw / "family-pin.json", scored, evaluated, dest, catalog)
    return {"scene_family_id": family, "extract_manifest_sha256": file_hash(dest / "family.json")}


def test_fit_freeze_and_base_only_scorer(tmp_path):
    train, dev = tmp_path / "train", tmp_path / "dev"
    train.mkdir()
    dev.mkdir()
    rows = [{"scene_family_id": f"fixture-{i:02}", "scene_id": f"fixture-{i:02}/v001",
             "trial_id": f"fixture-{i:02}-B0", "split": "train" if i < 16 else ("development" if i < 20 else "test"),
             "cv_fold": i % 4 if i < 16 else None} for i in range(24)]
    catalog = tmp_path / "catalog.json"
    write_json(catalog, {"schema": "el-formal-families-1", "families": rows})
    sources = {"train": [], "dev": []}
    deployed = tmp_path / "deployed"
    deployed.mkdir()
    for i, row in enumerate(rows[:20]):
        key = "train" if i < 16 else "dev"
        folder = train if i < 16 else dev
        with tempfile.TemporaryDirectory(dir=tmp_path) as work:
            raw = Path(work)
            entry = make_extract(raw, folder / row["scene_family_id"], catalog, row, i)
            sources[key].append(entry)
            if i == 19:
                for name in ("input.png", "cameras.json"):
                    shutil.copyfile(raw / name, deployed / name)
                shutil.copytree(raw / "prediction", deployed / "prediction")
                invalid = json.loads((raw / "family-pin.json").read_text())
                invalid.update(scene_family_id=rows[-1]["scene_family_id"], split="train", cv_fold=0)
                write_json(raw / "bad.json", invalid)
                with pytest.raises(ValueError, match="contradicts frozen"):
                    prepare(raw / "bad.json", raw / "absent", raw / "absent", raw / "rejected", catalog)
    for split, folder in (("train", train), ("dev", dev)):
        write_json(folder / "dataset.json", {"schema": "el-learning-index-1", "split": split, "condition_id": "B0",
                   "split_manifest_sha256": file_hash(catalog), "families": sources[split]})
        shutil.copyfile(catalog, folder / "catalog.json")
    fitted, frozen = tmp_path / "fit", tmp_path / "cutoff"
    result = fit(train, fitted, "synthetic-fit-only")
    assert result["status"] == "fitted" and all(v <= 1e-6 for v in result["max_sklearn_numpy_error"].values())
    assert result["rows"] == 16 * 6 * 4  # 4 of the 5 RGB-evaluable pixels have finite valid target inputs
    cv = json.loads((fitted / "cv_results.json").read_text())
    assert set(cv["models"]["L1-full"]["candidates"]) == {"0.1", "1.0", "10.0"}
    with pytest.raises(ValueError, match="development-frozen"):
        load_package(fitted / "model")
    selected = freeze(dev, fitted / "model", frozen, "synthetic-freeze-only")
    assert selected["status"] == "frozen" and set(selected["methods"]) == {*MODELS, "1-alpha", "constant"}
    assert len(selected["methods"]["constant"]["families"]) == 4
    package = load_package(frozen / "model")
    assert package["index"]["constant_risk"] == pytest.approx(2/5)
    assert "local_rgb_difference" not in package["models"]["L1-geometry"]["columns"]
    root = Path(__file__).resolve().parents[1]
    interpreter = root / ".venv-cpu/bin/python"
    guard = """
import runpy,sys
runpy.run_module('evidence_lens.l1', run_name='__main__')
assert not any(k in sys.modules for k in ('sklearn','pyarrow','torch','bpy','evidence_lens.labels','evidence_lens.evaluation'))
"""
    args = [str(interpreter if interpreter.is_file() else sys.executable), "-c", guard,
            "--model", str(frozen / "model"), "--input", str(deployed / "input.png"), "--cameras", str(deployed / "cameras.json"),
            "--predictions", str(deployed / "prediction"), "--run-id", "synthetic-public-score", "--output", str(deployed / "risks")]
    subprocess.run(args, check=True, cwd=deployed, env=dict(os.environ, PYTHONPATH=str(root / "src")))
    if os.environ.get("EL_TEST_CLEAN_WHEEL") == "1":
        grade_wheel(tmp_path, frozen, deployed)
    with np.load(deployed / "risks/v01.npz") as scored:
        assert np.array_equal(scored["l1_full_valid"], scored["l1_geometry_valid"])
        assert scored["l1_full_state"][1, 0] == 255 and np.isnan(scored["l1_geometry"][0, 2])
        assert np.isfinite(scored["l1_full"][0, 1])  # low alpha remains eligible
        last = rows[19]["scene_family_id"]
        with np.load(frozen / "development_scores" / last / "v01.npz") as development:
            for method, key in (("L1-full", "l1_full"), ("L1-geometry", "l1_geometry"),
                                ("constant", "constant"), ("1-alpha", "alpha_score")):
                np.testing.assert_allclose(scored[key].ravel()[development["index"]], development[key], atol=1e-6, rtol=0)
                assert int((scored[key + "_state"].ravel()[development["index"]] == 0).sum()) == selected["methods"][method]["views"][last]["v01"]["L"]
    p = deployed / "prediction/prediction.json"
    meta = json.loads(p.read_text())
    meta["depth_validation"]["off_axis_plane_passed"] = False
    write_json(p, meta)
    args[-1] = str(deployed / "unsupported")
    subprocess.run(args, check=True, cwd=deployed, env=dict(os.environ, PYTHONPATH=str(root / "src")))
    with np.load(deployed / "unsupported/v01.npz") as scored:
        assert (scored["l1_full_state"] == 255).all() and np.isfinite(scored["alpha_score"][0, 1])
    meta["depth_validation"]["off_axis_plane_passed"] = True
    meta["failed_views"] = {"v02": "declared synthetic render failure"}
    (deployed / "prediction/v02.npz").unlink()
    write_json(p, meta)
    args[-1] = str(deployed / "failed-view")
    subprocess.run(args, check=True, cwd=deployed, env=dict(os.environ, PYTHONPATH=str(root / "src")))
    receipt = json.loads((deployed / "failed-view/l1_score.json").read_text())
    assert receipt["views"]["v02"]["status"] == "render_failed"
    with np.load(deployed / "failed-view/v02.npz") as scored:
        assert all((scored[k + "_state"] == 255).all() for k in receipt["methods"].values())
    meta["depth_semantics"] = "otherwise_valid_unsupported_synthetic_semantics"
    write_json(p, meta)
    args[-1] = str(deployed / "incompatible")
    subprocess.run(args, check=True, cwd=deployed, env=dict(os.environ, PYTHONPATH=str(root / "src")))
    unsupported = json.loads((deployed / "incompatible/l1_score.json").read_text())
    assert unsupported["status"] == "unsupported" and len(unsupported["views"]) == 6
    path = frozen / "model/L1-full.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest"):
        load_package(frozen / "model")
    wrong = json.loads((train / "dataset.json").read_text())
    wrong["families"][-1]["scene_family_id"] = rows[-1]["scene_family_id"]
    write_json(train / "dataset.json", wrong)
    with pytest.raises(ValueError, match="roster"):
        Dataset(train, "train")


def test_recorded_single_class_failure_keeps_counts(tmp_path):
    catalog = tmp_path / "catalog.json"
    entries = []
    rows = [{"scene_family_id": f"synthetic-{i:02}", "scene_id": f"synthetic-{i:02}/v001",
             "trial_id": f"synthetic-{i:02}-B0", "split": "train", "cv_fold": i % 4} for i in range(16)]
    write_json(catalog, {"schema": "el-formal-families-1", "families": rows})
    for i, row in enumerate(rows):
        with tempfile.TemporaryDirectory(dir=tmp_path) as work:
            raw = Path(work)
            folder = tmp_path / row["scene_family_id"]
            # This deliberate same-class fixture processes actual all-zero synthetic renders through the P2 evaluator.
            saved = make_extract(raw, folder, catalog, row, i, single_class=True)
            entries.append(saved)
    write_json(tmp_path / "dataset.json", {"schema": "el-learning-index-1", "split": "train", "condition_id": "B0",
               "split_manifest_sha256": file_hash(catalog), "families": entries})
    output = tmp_path / "single-class-fit"
    result = fit(tmp_path, output, "synthetic-single-class")
    assert result["status"] == "insufficient_data" and result["reason"] == "two_class_training_folds_unavailable"
    assert result["positives"] == 0 and result["rows"] > 0
    folds = json.loads((output / "cv_results.json").read_text())["fold_attempts"]
    assert [r["k"] for r in folds] == [4, 3, 2]
    assert not (output / "model").exists()


def grade_wheel(tmp_path, frozen, deployed):
    from evidence_lens.l1_demo import grade, seal
    repo = Path(__file__).resolve().parents[1]
    public = tmp_path / "release-fixture"
    example = public / "example"
    example.mkdir(parents=True)
    shutil.copytree(frozen / "model", public / "model")
    shutil.copytree(deployed / "prediction", example / "predictions")
    for name in ("input.png", "cameras.json"):
        shutil.copyfile(deployed / name, example / name)
    receipt = file_hash(deployed / "risks/l1_score.json")
    # The calling subprocess must identify its archived synthetic example before it can be sealed.
    manifest = seal(public, deployed / "risks", receipt, ["v01", "v06"])
    pkg = tmp_path / "wheel-source"
    pkg.mkdir()
    for name in ("README.md", "pyproject.toml"):
        shutil.copyfile(repo / name, pkg / name)
    shutil.copytree(repo / "src/evidence_lens", pkg / "src/evidence_lens", ignore=shutil.ignore_patterns("__pycache__"))
    wheeldir, env = tmp_path / "wheels", tmp_path / "clean-public"
    subprocess.run(["uv", "build", "--offline", "--wheel", "--out-dir", str(wheeldir), str(pkg)], check=True, capture_output=True)
    subprocess.run(["uv", "venv", "--python", sys.executable, str(env)], check=True, capture_output=True)
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(["uv", "pip", "install", "--offline", "--python", str(python), str(next(wheeldir.glob("*.whl")))],
                   check=True, capture_output=True)
    result = grade(public, tmp_path / "graded", manifest["manifest_sha256"], python=python)
    assert result["passed"], result.get("error")
    assert result["runtime"]["optional_training_render_packages_present"] == [] and set(result["views"]) == {"v01", "v06"}
    wrong = public / "expected/v01.npz"
    wrong.write_bytes(wrong.read_bytes() + b" changed")
    with pytest.raises(ValueError, match="changed expected"):
        grade(public, tmp_path / "tampered", manifest["manifest_sha256"], python=python)
