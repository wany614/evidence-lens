import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from evidence_lens.features import extract_features
from evidence_lens.multisource import (
    COLUMNS,
    NUMERIC,
    CONDITIONS,
    TARGETS,
    extract,
    schema_bytes,
    schema_sha256,
    source_ids,
)
from evidence_lens.multisource_io import Case, load_native, cameras
from evidence_lens.multisource_model import transform, validate_rows
from evidence_lens.scoring_io import Camera, file_hash, write_json
from stage3_risk_sources import asset
from stage3_risk_learning import balanced, preprocessor, threshold_for, weights


def camera(w=6, h=4, tx=0.0):
    k = np.array([[2.0, 0, (w - 1) / 2], [0, 2.0, (h - 1) / 2], [0, 0, 1]])
    c = np.eye(4)
    c[0, 3] = tx
    return Camera(w, h, k, c)


def render(cam, tone=0.25, depth=2.0, alpha=0.8):
    shape = cam.shape
    return {
        "rgb": np.full((*shape, 3), tone, np.float32),
        "z": np.full(shape, depth, np.float32),
        "alpha": np.full(shape, alpha, np.float32),
    }


def case_fixture(p, c="M4"):
    from PIL import Image

    p.mkdir()
    (p / "predictions").mkdir()
    (p / "sources").mkdir()
    ids = CONDITIONS[c]
    cam = camera()
    spec = {"width": cam.width, "height": cam.height, "K": cam.K.tolist(), "c2w": cam.c2w.tolist()}
    sizes = {}
    for role, vs in (("source_cameras", ids), ("render_cameras", (*ids, *TARGETS))):
        filepath = p / (role + ".json")
        write_json(
            filepath,
            {"schema": "el-camera-1", "unit": "m", "pixel_center": "integer", "views": {v: spec for v in vs}},
        )
        sizes[role] = asset(p, filepath)
    src = {}
    images = {}
    views = {}
    for i, s in enumerate(ids):
        fp = p / "sources" / (s + ".png")
        Image.new("RGB", (cam.width, cam.height), (32 + i * 16,) * 3).save(fp)
        src[s] = asset(p, fp)
    for s in (*ids, *TARGETS):
        fp = p / "predictions" / (s + ".npz")
        np.savez_compressed(fp, **render(cam))
        images[s] = asset(p, fp)
        views[s] = {k: images[s][k] for k in ("bytes", "sha256")}
    pr = {
        "schema": "el-stage3-prediction-1",
        "generator_source_ids": list(ids),
        "target_ids": list(TARGETS),
        "checkpoint_sha256": "5" * 64,
        "reconstruction_run_id": "test-reconstruction",
        "views": views,
        "depth_verified": True,
        "source_camera_sha256": sizes["source_cameras"]["sha256"],
        "render_camera_sha256": sizes["render_cameras"]["sha256"],
        "source_rgb_sha256": {s: v["sha256"] for s, v in src.items()},
    }
    write_json(p / "prediction.json", pr)
    write_json(
        p / "case.json",
        {
            "schema": "el-stage3-deploy-case-1",
            "condition": c,
            "generator_source_ids": list(ids),
            "target_ids": list(TARGETS),
            "prediction": asset(p, p / "prediction.json"),
            "reconstruction_run_id": pr["reconstruction_run_id"],
            "checkpoint_sha256": pr["checkpoint_sha256"],
            "source_images": src,
            "renders": images,
            **sizes,
        },
    )
    return Case(p, file_hash(p / "case.json"))


def test_schema_and_source_order():
    assert len(NUMERIC) == 22 and len(COLUMNS) == len(set(COLUMNS)) == 39
    p = Path(__file__).resolve().parents[1] / "protocol/multisource-v1.json"
    assert p.read_bytes() == schema_bytes() and file_hash(p) == schema_sha256()
    cam = camera()
    t = render(cam)
    defs = {}
    for i, s in enumerate(CONDITIONS["M4"]):
        im = np.full((*cam.shape, 3), i * 0.2, np.float32)
        defs[s] = (im, render(cam, tone=i * 0.2), cam)
    x = extract(t, cam, defs)
    y = extract(t, cam, dict(reversed(list(defs.items()))))
    assert all(np.array_equal(a, y[k], equal_nan=True) for k, a in x.items())
    a = x["X"]
    validate_rows(a)
    assert x["scoreable"].all() and (x["support_count"] == 4).all()
    assert np.allclose(a[..., COLUMNS.index("source_pair_mae_max")], 0.6)
    assert np.allclose(a[..., COLUMNS.index("target_source_mae_mean")], 0.2)
    assert (x["colour_conflict_count"] == 3).all()
    assert np.all(a[..., COLUMNS.index("depth_residual_mean")] == 0)


def test_projection_missing_and_conflict_separate():
    cam = camera()
    shifted = camera(tx=1.0)
    t = render(cam)
    s = render(shifted)
    im = np.zeros((*cam.shape, 3), np.float32)
    im[:, 0] = 0.5
    new = extract(t, cam, {"s00": (im, s, shifted)})
    old = extract_features(im, s, t, shifted, cam, True)
    assert np.array_equal(new["in_frame_count"] == 1, old["frame_inside"])
    assert not new["scoreable"][:, 0].any() and (new["unknown_reason"][:, 0] == 5).all()
    assert np.allclose(new["X"][:, 1, COLUMNS.index("target_source_mae_mean")], 0.25)
    assert np.all(new["X"][..., COLUMNS.index("fewer_than_two_support")] == 1)
    assert np.isnan(new["X"][..., COLUMNS.index("source_pair_mae_max")]).all()
    s["z"][:] = 8
    x = extract(t, cam, {"s00": (im, s, cam)})
    assert x["scoreable"].all() and not x["support_count"].any() and (x["depth_conflict_count"] == 1).all()
    s["alpha"][:] = 0.2
    x = extract(t, cam, {"s00": (im, s, cam)})
    assert x["scoreable"].all() and not x["support_count"].any() and not x["conflict_count"].any()
    x = extract(t, cam, {"s00": (im, None, cam)})
    assert not x["scoreable"].any() and (x["unknown_reason"] == 6).all()
    assert np.isnan(x["X"][..., COLUMNS.index("depth_residual_mean")]).all()
    t["rgb"][0, 0] = np.nan
    t["z"][0, 1] = -1
    t["alpha"][0, 2] = np.nan
    x = extract(t, cam, {"s00": (im, s, cam)})
    assert list(x["unknown_reason"][0, :3]) == [2, 4, 3]
    x = extract(t, cam, {"s00": (im, s, cam)}, depth_verified=False)
    assert np.all(x["unknown_reason"] == 1)


def test_disk_excludes_unallowed_sources_and_truth(tmp_path):
    case = case_fixture(tmp_path / "case")
    for s in ("s01", "s02", "s03"):
        (case.root / "sources" / (s + ".png")).unlink()
        (case.root / "predictions" / (s + ".npz")).unlink()
    (case.root / "reference.json").write_bytes(b"bad truth")
    rgb, x = load_native(case.root, "v01", "one", expected_case_sha256=case.sha256)
    assert x["scoreable"].all() and rgb.shape == (4, 6, 3)
    with pytest.raises(ValueError):
        load_native(case.root, "v01", "two")
    with pytest.raises(ValueError):
        load_native(case.root, "v01", "one", expected_case_sha256="0" * 64)
    (case.root / "sources/s00.png").write_bytes(b"substituted pixels")
    with pytest.raises(ValueError, match="identity"):
        load_native(case.root, "v01", "one")


def test_m2_alias_and_bad_camera(tmp_path):
    c = case_fixture(tmp_path / "M2", "M2")
    a, b = [c.features("v01", s)[1] for s in ("two", "all")]
    assert all(np.array_equal(v, b[k], equal_nan=True) for k, v in a.items())
    assert source_ids("M2", "all") == source_ids("M4", "two")
    d = json.loads((c.root / "render_cameras.json").read_bytes())
    d["views"]["v01"]["K"][0][0] = -1
    with pytest.raises(ValueError, match="intrinsics"):
        cameras(d, d["views"])


def test_fraction_ties_denominators_and_weights():
    vs = {
        ("A", "M2"): np.array([0.2, 0.2, 0.2, 0.3], np.float32),
        ("B", "M2"): np.array([0.1, 0.1, 0.1, 0.9], np.float32),
    }
    cutoff = threshold_for(vs, {("A", "M2"): 4, ("B", "M2"): 4, ("A", "M4"): 0})
    assert cutoff["coverage"] == 0.75 and 0.2 < cutoff["tau"] < 0.3
    zero = threshold_for({}, {("A", "M2"): 9})
    assert zero["tau"] == zero["coverage"] == 0
    groups = np.array([[0, 0, 0]] * 20 + [[0, 0, 1]] * 10 + [[0, 1, 0]] * 2 + [[1, 1, 0]] * 4)
    w = weights(groups)
    assert w[groups[:, 0] == 0].sum() == pytest.approx(w[groups[:, 0] == 1].sum())
    assert w[:30].sum() == pytest.approx(w[30:32].sum())
    assert w[:20].sum() == pytest.approx(w[20:30].sum())
    assert balanced(
        {("a", "M2", "v01"): 0.2, ("a", "M2", "v02"): 0.6, ("a", "M4", "v01"): 1.0, ("b", "M2", "v01"): 0.0}
    )[0] == pytest.approx(0.35)


def test_preproc_train_only_and_binary():
    cam = camera()
    t = render(cam)
    x = extract(t, cam, {"s00": (t["rgb"], t, cam)})["X"].reshape(-1, 39)
    w = np.ones(len(x))
    p = preprocessor(x, w)
    orig = np.array(p["means"])
    change = x.copy()
    change[:, 0] = 0.1
    z = transform(change, p)
    assert (z[:, 0] < 0).all() and np.isfinite(z).all()
    assert np.array_equal(z[:, 22:], change[:, 22:]) and np.array_equal(p["means"], orig)
    bad = x.copy()
    bad[1, 22] = 1 - bad[1, 22]
    with pytest.raises(ValueError, match="flag"):
        transform(bad, p)


def test_scientific_selection_rejects_engineering_and_repeat(tmp_path):
    from stage3_risk_sources import pin_selections, trial_location

    family = "training-fixture"
    row = {
        "scene_family_id": family,
        "split": "train",
        "pilot": True,
        "trials": {c: family + "-" + c for c in CONDITIONS},
    }
    roots = (tmp_path, tmp_path / "cold")
    for cohort in ("formal", "pilot"):
        for c in CONDITIONS:
            trial, path, phase = trial_location(tmp_path, row, c, cohort)
            path.parent.mkdir(parents=True)
            write_json(
                path, {"family": family, "condition": c, "trial_id": trial, "selected": {}, "phase": phase}
            )
        out = pin_selections(roots, [row], cohort)
        assert all(x["cohort"] == cohort for x in out)
    trial, path, _ = trial_location(tmp_path, row, "M2", "formal")
    doc = json.loads(path.read_bytes())
    doc["phase"] = "engineering-pilot"
    write_json(path, doc)
    with pytest.raises(ValueError, match="trial"):
        pin_selections(roots, [row], "formal")
    with pytest.raises(ValueError, match="pilot"):
        trial_location(tmp_path, {**row, "split": "development"}, "M2", "pilot")
