import io
import json

import numpy as np
from PIL import Image
import pytest

from evidence_lens_bundle.common import encoded, row, sha
from evidence_lens_bundle.grids import build_grids
from evidence_lens_many import (
    BASELINE_ID,
    Bank,
    FEATURE_NAMES,
    FEATURE_SCHEMA_SHA256,
    NUMERIC_NAMES,
    Plan,
    audit_subset,
    build_manifest,
    legacy_q4,
    score_lr,
    select_sources,
    simple_ranks,
    stream_features,
)
from evidence_lens_many.cli import main
from evidence_lens_many.stream import _Moments


def bank_fixture(path):
    path.mkdir()
    sources = {}
    cameras = {}
    names = ["a", "b", "c", "d", "e", "duplicate-a"]
    original = {
        "width": 378,
        "height": 252,
        "K": [[220, 0, 188.5], [0, 210, 125.5], [0, 0, 1]],
        "distortion": "none",
    }
    grids = build_grids("analysis-378x252-v1", {n: original for n in [*names, "t"]})

    def put(name, data):
        (path / name).write_bytes(data)
        return row(name, data)

    for i, n in enumerate([*names, "t"]):
        pose = np.eye(4)
        pose[0, 3] = 200 if n in ("a", "b", "c", "d", "duplicate-a") else 0
        cameras[n] = {**original, "c2w": pose.tolist()}
        c = 0 if n == "duplicate-a" else i
        photo = np.full((252, 378, 3), 40 + c, np.uint8)
        png = io.BytesIO()
        Image.fromarray(photo).save(png, "PNG")
        buff = io.BytesIO()
        z = np.full((252, 378), 2, np.float32)
        a = np.ones_like(z)
        if n == "t":
            z[0, 0] = np.nan
            a[1, 1] = 0.005
        np.savez_compressed(buff, rgb=photo.astype(np.float32) / 255, z=z, alpha=a)
        render = put(n + ".npz", buff.getvalue())
        if n == "t":
            targets = {n: {"render": render}}
        else:
            asset = put(n + ".png", png.getvalue())
            sources[n] = {"original_sha256": asset["sha256"], "photo": asset, "render": render}
    data = build_manifest(
        scene_id="analytic",
        checkpoint_sha256="1" * 64,
        renderer_id="analytic-plane",
        reconstruction_provenance="verified",
        reconstruction_ids=names,
        accessible_ids=names,
        target_ids=["t"],
        metric_anchor_ids=["a", "d"],
        frame_receipt={"unit": "scene_units", "matrix": np.eye(4).tolist()},
        cameras=cameras,
        grids=grids,
        sources=sources,
        targets=targets,
        depth_verified=True,
        depth_probe_sha256="2" * 64,
        permission={"source_ids": names, "terms": "generated numeric interface fixture"},
    )
    (path / "bank.json").write_bytes(encoded(data))
    return path


def test_geometry_only_selection_aliases_partial_scans_and_gap_witness(tmp_path):
    b = Bank(bank_fixture(tmp_path / "bank"), cache_bytes=0)
    assert b.ids == ("a", "b", "c", "d", "e") and b.aliases == {"duplicate-a": "a"}
    assert ["a", "b"] in b.inspect()["near_duplicate_pairs"]
    plan = select_sources(b, "t")
    assert plan.ids[0] == "e" and plan.cost["source_photos_decoded"] == plan.cost["target_rgb_decoded"] == 0
    selected = next(stream_features(b, "t", plan, budgets=(4,)))
    assert b.counters["source_photos_decoded"] == 4 and b.counters["source_renders_decoded"] == 4
    assert selected.comparable_count.dtype == np.uint32 and selected.comparable_count[5, 5] == 1
    assert selected.first_witness[5, 5] == b.ids.index("e")
    assert not selected.finite[0, 0] and not selected.finite[1, 1] and selected.finite[5, 5]
    fixed = Plan(b.sha256, "t", "explicit-control", b.ids, (), {})
    four, all_views = stream_features(b, "t", fixed, budgets=(4, 8, 16, 32, "all"))
    assert all_views.nominal_budgets == (8, 16, 32, "all") and all_views.actual_k == 5
    assert four.comparable_count[5, 5] == 0 and four.finite[5, 5]
    assert audit_subset(four)["state"][5, 5] == 0
    audit = audit_subset(four, all_views)
    assert audit["full_complete"] and audit["state"][5, 5] == 1 and audit["gap_witness"][5, 5] == 4
    assert audit["state"][0, 0] == 4 and audit["gap_witness"][0, 0] == -1
    assert all_views.cost["source_photos_decoded"] == 5 and four.cost["source_photos_decoded"] == 4
    assert four.X.shape == (252, 378, len(FEATURE_NAMES)) and len(NUMERIC_NAMES) <= 32
    assert np.isnan(four.X[5, 5, FEATURE_NAMES.index("photo_mae_mean")])
    assert four.X[5, 5, FEATURE_NAMES.index("missing_photo_mae_mean")] == 1
    rank = simple_ranks(four)
    assert rank["residual"][5, 5] == 1 and all(
        np.array_equal(np.isfinite(a), four.finite) for a in rank.values()
    )
    model = dict(
        schema="el-b-cpu-lr-1",
        baseline_id=BASELINE_ID,
        feature_schema_sha256=FEATURE_SCHEMA_SHA256,
        columns=list(FEATURE_NAMES),
        impute=[0.0] * len(FEATURE_NAMES),
        mean=[0.0] * len(FEATURE_NAMES),
        scale=[1.0] * len(FEATURE_NAMES),
        coef=[0.0] * len(FEATURE_NAMES),
        intercept=0.0,
        tau=None,
    )
    inferred = score_lr(four, model)
    assert (inferred[four.finite] == 0.5).all() and np.isnan(inferred[~four.finite]).all()
    # Q4 transfer accepts a fixed frame anchor outside Q. The A geometry remains in its original scene units.
    transferred = legacy_q4(b, "t", ("a", "b", "c", "e"), "MS-L1")
    assert transferred["profile"]["metric_anchor_ids"] == ["a", "d"] and transferred["finite"][5, 5]
    n = _Moments((1, 1))
    for _ in range(300):
        n.update(np.ones((1, 1)), np.ones((1, 1), bool))
    assert n.n.item() == 300


def test_declared_missing_integrity_and_cache(tmp_path, capsys):
    p = bank_fixture(tmp_path / "bank")
    b = Bank(p, cache_bytes=32 << 20)
    first = b.source("e")
    second = b.source("e")
    assert first is second and b.counters["cache_hits"] == 1 and b.counters["cache_peak_bytes"] <= b.capacity
    with pytest.raises(TypeError):
        first[1]["rgb"] = np.zeros((1,))
    (p / "e.png").write_bytes(b"substituted")
    with pytest.raises(ValueError, match="changed source"):
        b.source("e")
    (p / "e.png").unlink()
    full = next(stream_features(b, "t", select_sources(b, "t", strategy="fixed"), budgets=("all",)))
    assert full.missing_source_ids == ("e",) and audit_subset(full, full)["state"][5, 5] == 3
    d = json.loads((p / "bank.json").read_bytes())
    d["targets"]["t"]["reference"] = {"path": "never-open"}
    (p / "bank.json").write_bytes(encoded(d))
    with pytest.raises(ValueError, match="target prediction"):
        Bank(p)
    ply = tmp_path / "unsupported.ply"
    ply.write_text("ply\nformat ascii 1.0\nelement vertex 2\nend_header\n")
    assert main(["inspect", "--bank", str(ply)]) == 0
    assert "structural_inspection_only" in capsys.readouterr().out


def test_operator_outputs_and_no_implicit_full_audit(tmp_path):
    b = bank_fixture(tmp_path / "bank")
    for action in ("score", "audit-bank"):
        dest = tmp_path / action
        status = main(
            [
                action,
                "--bank",
                str(b),
                "--target",
                "t",
                "--budget",
                "4",
                "--rule",
                "residual",
                "--out",
                str(dest),
            ]
        )
        assert status == 0
        doc = json.loads((dest / "scores.json").read_bytes())
        assert doc["audit"]["requested"] == (action == "audit-bank") and doc["arrays"]["sha256"] == sha(
            (dest / "scores.npz").read_bytes()
        )
        if action == "score":
            assert doc["audit"]["full_scope_cost"] is None and doc["query_cost"]["source_photos_decoded"] == 4
        else:
            assert doc["audit"]["full_scope_cost"]["source_photos_decoded"] == 5
        assert "Click a pixel" in (dest / "report.html").read_text()


def test_streamed_sums_conflict_and_empty_moments_match_an_explicit_plane(tmp_path):
    p = bank_fixture(tmp_path / "bank")
    metadata = json.loads((p / "bank.json").read_bytes())
    for name, camera in metadata["cameras"].items():
        camera["c2w"] = np.eye(4).tolist()
    with np.load(p / "a.npz") as prev:
        raw = dict(prev)
    raw["z"][:] = 3
    buf = io.BytesIO()
    np.savez_compressed(buf, **raw)
    (p / "a.npz").write_bytes(buf.getvalue())
    metadata["sources"]["a"]["render"] = row("a.npz", buf.getvalue())
    (p / "bank.json").write_bytes(encoded(metadata))
    b = Bank(p)
    out = next(stream_features(b, "t", select_sources(b, "t"), budgets=("all",)))
    y, x = 40, 50
    fields = dict(zip(FEATURE_NAMES, out.X[y, x]))
    assert (
        out.comparable_count[y, x] == 5
        and out.support_count[y, x] == 4
        and out.depth_conflict_count[y, x] == 1
    )
    assert fields["support_comparable_fraction"] == pytest.approx(0.8)
    assert fields["depth_conflict_comparable_fraction"] == pytest.approx(0.2)
    assert fields["relative_depth_std"] == pytest.approx(np.std([1 / 3, 0, 0, 0, 0]), abs=1e-7)
    assert fields["relative_depth_max"] == pytest.approx(1 / 3) and fields["angle_mean"] == 0
    errors = np.arange(2, 6) / 255
    assert fields["photo_mae_mean"] == pytest.approx(errors.mean(), abs=1e-7)
    assert fields["photo_mae_std"] == pytest.approx(errors.std(), abs=1e-7)
    assert fields["self_mae_max"] == 0 and fields["missing_photo_mae_mean"] == 0
    metadata["depth_verified"] = False
    (p / "bank.json").write_bytes(encoded(metadata))
    incomplete = Bank(p)
    with pytest.raises(ValueError, match="verified expected camera-Z"):
        next(stream_features(incomplete, "t", select_sources(incomplete, "t"), budgets=(4,)))
