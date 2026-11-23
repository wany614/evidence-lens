"""Synthetic native-grid supervision and frozen B initialization checks. No stage C study data."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from evidence_lens.multisource import COLUMNS, NUMERIC, schema_sha256
from evidence_lens_multirisk.data import Dataset, source_ids
from evidence_lens_multirisk.model import B_INDEX, B_WEIGHTS, transform
from evidence_lens_multirisk.proof import verify
from evidence_lens_multirisk.selection import hierarchical_ap, operating_point
from evidence_lens_multirisk.train import fit_core
from evidence_lens_vision.contract import digest, file_record, put, read

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))


def native_case(root):
    from PIL import Image

    h = w = 256
    shape = (h, w)
    ids = source_ids("M4", "all")
    views = [f"v{i:02d}" for i in range(1, 7)]
    for n in ("sources", "predictions"):
        (root / n).mkdir(parents=True)
    cam = {
        "height": h,
        "width": w,
        "K": [[250.0, 0.0, 127.5], [0.0, 250.0, 127.5], [0.0, 0.0, 1.0]],
        "c2w": np.eye(4).tolist(),
    }
    members = {}
    originals = {}
    renders = {}
    for n, chosen in (("source_cameras", ids), ("render_cameras", ids + views)):
        p = root / (n + ".json")
        put(
            p,
            {
                "schema": "el-camera-1",
                "unit": "m",
                "pixel_center": "integer",
                "views": dict.fromkeys(chosen, cam),
            },
        )
        members[n] = file_record(p, root)
    for i, s in enumerate(ids):
        p = root / "sources" / (s + ".png")
        Image.new("RGB", (w, h), (24 + i * 16, 64, 88)).save(p)
        originals[s] = file_record(p, root)
    for s in ids + views:
        p = root / "predictions" / (s + ".npz")
        np.savez_compressed(
            p,
            rgb=np.full((*shape, 3), 0.4, np.float32),
            alpha=np.full(shape, 0.8, np.float32),
            z=np.full(shape, 2.0, np.float32),
        )
        renders[s] = file_record(p, root)
    common = {
        "generator_source_ids": ids,
        "target_ids": views,
        "reconstruction_run_id": "synthetic-reconstruction",
        "checkpoint_sha256": "2" * 64,
    }
    p = root / "prediction.json"
    put(
        p,
        {
            "schema": "el-stage3-prediction-1",
            **common,
            "depth_verified": True,
            "source_rgb_sha256": {k: d["sha256"] for k, d in originals.items()},
            "views": {k: {j: d[j] for j in ("bytes", "sha256")} for k, d in renders.items()},
            "render_camera_sha256": members["render_cameras"]["sha256"],
            "source_camera_sha256": members["source_cameras"]["sha256"],
        },
    )
    put(
        root / "case.json",
        {
            "schema": "el-stage3-deploy-case-1",
            "condition": "M4",
            **common,
            "prediction": file_record(p, root),
            **members,
            "source_images": originals,
            "renders": renders,
        },
    )
    return digest(root / "case.json")


def raw(scope_count=2):
    x = np.zeros((256, 256, len(COLUMNS)), np.float32)
    y, xcoord = np.mgrid[:256, :256]
    for n, val in (
        ("target_alpha", 0.8),
        ("target_local_depth_range", 0.02),
        ("n_sources", scope_count),
        ("n_in_frame", scope_count),
        ("n_supported", scope_count),
        ("n_comparable", scope_count),
        ("comparable_fraction", 1.0),
        ("support_fraction", 1.0),
        ("source_alpha_mean", 0.8),
        ("depth_residual_mean", (xcoord % 7) / 10),
        ("view_angle_mean", xcoord / 256),
        ("target_source_mae_mean", (xcoord + y) % 11 / 12),
    ):
        x[..., COLUMNS.index(n)] = val
    return x


def files(root, split, family):
    root.mkdir()
    rng = np.random.default_rng(125)
    rgb = rng.uniform(0.05, 0.95, (256, 256, 3)).astype(np.float32)
    np.save(root / "rgb.npy", rgb)
    mask = np.zeros((256, 256), bool)
    mask[3:15, 3:18] = True
    ix = np.flatnonzero(mask).astype(np.int32)
    y = np.full((256, 256), 255, np.uint8)
    y[mask] = (ix % 5 < 2).astype(np.uint8)
    np.save(root / "common.npy", mask)
    np.savez_compressed(
        root / "truth.npz", reference_valid=np.ones_like(mask), rgb_evaluable=mask, rgb_high_error=y
    )
    ref = lambda n: file_record(root / n, root)
    for num in (1, 2, 4):
        x = raw(num)
        np.savez_compressed(
            root / f"d{num}.npz",
            X=x,
            scoreable=mask,
            support_count=mask.astype(np.uint8),
            conflict_count=np.zeros_like(y),
        )
        np.savez_compressed(
            root / f"s{num}.npz",
            index=ix,
            X=x.reshape(-1, len(COLUMNS))[ix],
            Y=y[mask],
            inclusion_probability=np.array(1.0, np.float64),
        )
    rows = []
    for c in ("M2", "M4"):
        for i in range(1, 7):
            modes = {}
            for s in ("one", "two", "all"):
                src = source_ids(c, s)
                num = len(src)
                modes[s] = {
                    "source_ids": src,
                    "alias_of": "two" if (c, s) == ("M2", "all") else None,
                    "eligible": len(ix),
                    "scoreable_pixels": len(ix),
                    "deploy": ref(f"d{num}.npz") if split == "development" else None,
                    "samples": ref(f"s{num}.npz")
                    if split == "train" and (s == "all" or (c, s) == ("M2", "two"))
                    else None,
                }
            rows.append(
                {
                    "family": family,
                    "condition": c,
                    "view": f"v{i:02d}",
                    "status": "ready",
                    "failure": None,
                    "width": 256,
                    "height": 256,
                    "rgb": ref("rgb.npy"),
                    "truth": ref("truth.npz"),
                    "selection_support": ref("common.npy"),
                    "modes": modes,
                }
            )
    path = root / "dataset.json"
    put(
        path,
        {
            "schema": "el-stage3-risk-dataset-1",
            "cohort": "formal",
            "split": split,
            "catalog_sha256": "1" * 64,
            "feature_schema_sha256": schema_sha256(),
            "columns": list(COLUMNS),
            "families": [family],
            "views": rows,
        },
    )
    return path


def test_lowest_tie_is_attainable_with_family_balancing_and_missing_cells():
    family = ["a", "b"]
    lists = {(f, c): [] for f in family for c in ("M2", "M4")}
    V = {k: 0 for k in lists}
    V["a", "M2"] = 4
    V["b", "M2"] = 4
    V["b", "M4"] = 4
    lists["a", "M2"] = [np.array([0.25, 0.25, 0.5, 0.5], np.float32)]
    lists["b", "M2"] = lists["a", "M2"]
    lists["b", "M4"] = [np.array([], np.float32)]
    got = operating_point(lists, V, family)
    assert got["coverage"] == 0.75 and got["tau"] > 0.5
    six = [{"family": "a", "condition": "M2", "ap": v} for v in (0.1, 0.3, None)] + [
        {"family": "a", "condition": "M4", "ap": 0.8},
        {"family": "b", "condition": "M4", "ap": 0.4},
    ]
    assert hierarchical_ap(six, family)["value"] == pytest.approx(0.45)


def test_shared_preprocessing_rejects_a_false_missing_flag():
    x = raw().reshape(-1, len(COLUMNS))[:2].copy()
    pre = {
        "schema": "el-stage3-risk-preprocessing-1",
        "feature_schema_sha256": schema_sha256(),
        "columns": list(COLUMNS),
        "numeric_count": len(NUMERIC),
        "medians": [0.0] * len(NUMERIC),
        "means": [0.0] * len(NUMERIC),
        "scales": [1.0] * len(NUMERIC),
    }
    x[:, COLUMNS.index("view_angle_mean")] = np.nan
    with pytest.raises(ValueError, match="missing flag"):
        transform(x, pre)
    x[:, COLUMNS.index("missing_view_angle_mean")] = 1
    assert np.isfinite(transform(x, pre)).all()


def test_sealed_heldout_before_any_dataset_read(tmp_path, monkeypatch):
    import evidence_lens.stage3_gate as gate
    from evidence_lens_multirisk.inference import batch

    class Closed(RuntimeError):
        pass

    def no_data(*args, **kwargs):
        raise Closed("no release marker")

    monkeypatch.setattr(gate, "require_opening", no_data)
    a = SimpleNamespace(
        split="test",
        opening_sha256="a" * 64,
        store=tmp_path / "store",
        shelf=None,
        dataset=tmp_path / "absent.json",
    )
    with pytest.raises(Closed):
        batch(a)


def test_synthetic_full_training_proof_and_resume_both_branches(tmp_path):

    core = os.environ.get("EL_C_B_PACKAGE")
    source = os.environ.get("EL_C_FEATURE_SCHEMA")
    if not core or not source:
        pytest.skip("set EL_C_B_PACKAGE and EL_C_FEATURE_SCHEMA for published-B CPU integration")
    parent = Path(core)
    schema_path = Path(source)
    assert digest(parent / "model.json") == B_INDEX and digest(parent / "weights.safetensors") == B_WEIGHTS
    cat = {
        "families": [
            {"scene_family_id": "synth_train", "split": "train"},
            {"scene_family_id": "synth_dev", "split": "development"},
        ]
    }
    t = files(tmp_path / "train", "train", "synth_train")
    d = files(tmp_path / "dev", "development", "synth_dev")
    pre = {
        "schema": "el-stage3-risk-preprocessing-1",
        "feature_schema_sha256": schema_sha256(),
        "columns": list(COLUMNS),
        "numeric_count": len(NUMERIC),
        "medians": [0.0] * len(NUMERIC),
        "means": [0.0] * len(NUMERIC),
        "scales": [1.0] * len(NUMERIC),
        "training_dataset_sha256": digest(t),
    }
    prep = tmp_path / "preprocessing.json"
    put(prep, pre)
    tr = Dataset(t, digest(t), "train", cat, "1" * 64, pre)
    dv = Dataset(d, digest(d), "development", cat, "1" * 64, pre)
    protocol = {
        "head_lr": 0.0003,
        "finetune_backbone_lr": 0.00001,
        "accumulate": 4,
        "epochs": 1,
        "patience": 3,
    }
    ident = {
        "schema":"el-stage3-neural-fit-1",
        "seed": 1729,
        "training_dataset_sha256": tr.sha,
        "development_dataset_sha256": dv.sha,
        "feature_schema_sha256": schema_sha256(),
        "catalog_sha256": "1" * 64,
    }
    a = SimpleNamespace(
        preprocessing=prep,
        feature_schema=schema_path,
        B_index=parent / "model.json",
        B_weights=parent / "weights.safetensors",
        device="cpu",
        threads=2,
        resume=None,
        resume_sha256=None,
    )
    proofs = []
    from stage3_fit_audit import first_step_replay,verify_base
    roster=verify_base(a.B_index,a.B_weights)
    assert roster["block11_tensors"]==14 and roster["final_norm_tensors"]==2 and roster["tail_scalars"]==1776000
    data=SimpleNamespace(train_sha=tr.sha,supervised=[],samples={})
    for row in tr.rows:
        ix,_,labels=tr.samples(row);key=tuple(row[k] for k in ("family","condition","view"))
        data.supervised.append(key)
        spec=row["modes"]["all"]["samples"]
        data.samples[key]={"sample":tr.file(spec),"rgb":tr.file(row["rgb"]),"spec":spec,"n":len(ix),"positive":int(labels.sum())}
    for branch in ("MS-frozen", "MS-ft"):
        a.output = tmp_path / branch
        meta = dict(ident, method=branch, model_id="fixture-" + branch)
        fit = fit_core(a, protocol, pre, tr, dv, meta, "engineering_fixture")
        assert read(a.output/"package/model.json")["schema"]=="el-multisource-risk-model-1"
        proof = verify(
            a.B_index,
            a.B_weights,
            a.output,
            tmp_path / ("check-" + branch),
            digest(a.output / "training.json"),
        )
        assert (
            fit["optimizer_steps"] == 3
            and proof["verified_tensors"] == 181
            and min(proof["nonzero_task_gradient_minibatches"].values()) >= 3
        )
        assert all(d["status"] == "ready" for d in read(a.output / "development-scores/scores.json")["views"])
        for row in read(a.output / "development-scores/scores.json")["views"]:
            if row["condition"] == "M2":
                assert row["modes"]["two"]["file"] == row["modes"]["all"]["file"]
        source=SimpleNamespace(open=lambda member:a.output/member.removeprefix("payload/"))
        audited=first_step_replay(source,fit,a.B_index,a.B_weights,data,2)
        assert audited["within_numerical_check"] and audited["first_step_replayed"]==4 and audited["original_witnesses_checked"]==8
        assert [w["upcoming_optimizer_step"] for w in audited["witnesses"]]==[1]*4+[2]*4
        assert all(max(g["absolute_l2_difference"] for g in w["gradients"].values())<1e-5 for w in audited["witnesses"][:4])
        proofs.append(proof)
    assert (
        proofs[0]["initial_head_sha256"] == proofs[1]["initial_head_sha256"]
        and proofs[0]["paired_design_sha256"] == proofs[1]["paired_design_sha256"]
    )
    assert proofs[0]["backbone_changed_scalars"] == 0 and proofs[1]["backbone_changed_scalars"] > 0
    a.resume = tmp_path / "MS-ft/training-state.pt"
    a.resume_sha256 = digest(a.resume)
    a.output = tmp_path / "recovered"
    fit = fit_core(
        a, protocol, pre, tr, dv, dict(ident, method="MS-ft", model_id="fixture-retry"), "engineering_fixture"
    )
    assert fit["optimizer_steps"] == 3 and fit["weights_sha256"] == proofs[1]["weights_sha256"]
    # Run the exported application in an isolated interpreter from an empty cwd.
    # A missing truth and missing withheld source rasters must leave one-source scoring usable.
    pk = tmp_path / "MS-ft/package"
    msha = proofs[1]["model_index_sha256"]
    case = tmp_path / "native-case"
    csha = native_case(case)
    common = ["--model-sha256", msha, "--case", str(case), "--case-sha256", csha, "--threads", "2"]
    isolated = tmp_path / "empty"
    isolated.mkdir()
    cmd = [sys.executable, "-I", "-B", str(pk / "score-native.py"), *common]
    clean = dict(os.environ)
    clean.pop("PYTHONPATH", None)
    scored = subprocess.run(
        [*cmd, "--view", "v01", "--scope", "all", "--output", str(tmp_path / "native-all")],
        cwd=isolated,
        env=clean,
        check=True,
        capture_output=True,
    )
    rec = json.loads(scored.stdout)
    assert (
        rec["supported"] == 65536
        and rec["scorer_source_ids"] == source_ids("M4", "all")
        and rec["condition"] == "M4"
    )
    from evidence_lens_multirisk.inference import costs

    if os.environ.get("EL_C_MEASURE_FIXTURE"):
        import evidence_lens_multirisk

        script = Path(evidence_lens_multirisk.__file__).resolve().parents[2] / "scripts/multi_risk_worker.py"
        measurement = costs(
            SimpleNamespace(
                output=tmp_path / "timing",
                model=pk,
                model_sha256=msha,
                case=case,
                case_sha256=csha,
                threads=2,
                scopes=["one", "all"],
                repetitions=1,
            ),
            script,
        )
        assert measurement["cold_warm_v01_equal"] and all(
            m["timed_native_views"] == 6 for m in measurement["warm_views"].values()
        )
        assert (
            min(m["sampled_peak_process_tree_rss_bytes"] for m in measurement["cold_v01_by_scope"].values())
            > 0
        )
    for s in ("s01", "s02", "s03"):
        (case / "sources" / (s + ".png")).unlink()
        (case / "predictions" / (s + ".npz")).unlink()
    scored = subprocess.run(
        [*cmd, "--view", "v01", "--scope", "one", "--output", str(tmp_path / "native-one")],
        cwd=isolated,
        env=clean,
        check=True,
        capture_output=True,
    )
    rec = json.loads(scored.stdout)
    assert rec["scorer_source_ids"] == ["s00"] and rec["supported"] == 65536
    with np.load(tmp_path / "native-one/scores.npz", allow_pickle=False) as grid:
        assert set(grid.files) == {"risk", "valid", "state"} and np.isfinite(grid["risk"]).all()
        assert np.array_equal(grid["state"], grid["risk"].astype(np.float64) >= rec["threshold"])
