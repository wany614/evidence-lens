"""Negative controls for the restricted PyTorch readback of actual Windows checkpoint/index formats."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evidence_lens.stage3_gaussians import PARAMS, digest_values
from recovery_outputs import WINDOWS_PARAM, checkpoint_audit, png_readback, trace_summary, verified_semantics
from evidence_lens._bundle import sha


def test_literal_cv_param_content_and_sampler():
    assert WINDOWS_PARAM == {
        "means": "means",
        "log_scales": "scales",
        "quats": "quats",
        "opacity_logits": "opacities",
        "color_logits": "colors",
    }


def test_semantics_require_the_admitted_cuda_build_and_raw_probe(tmp_path, monkeypatch):
    import json
    import recovery_outputs

    def asset(name, contents):
        path = tmp_path / name
        path.write_text(json.dumps(contents))
        return path

    parent = {"run_id": "projection", "manifest_sha256": "a" * 64}
    depth = {**parent, "member": "payload/ed-probe.json", "sha256": "b" * 64}
    resource_pin = {"run_id": "readback", "sha256": "c" * 64}
    software = {"sha256": "d" * 64}
    req = {"resource_probe": resource_pin, "depth_probe": depth, "worker": software, "recipe": {"run_id": "admit"}}
    worker = {"source_sha256": "e" * 64, "gsplat": "1.5.3", "extension_sha256": "f" * 64}
    actual = {"passed": True, "resources": {"gsplat": "1.5.3"}}
    raw = {**actual, "expected_z_m": 0.656835443}
    probe = asset("ed.json", raw)
    resource = asset("resource.json", {
        "schema": "el-recovery-resource-probe-1", "passed": True, "worker_tree_sha256": "tree",
        "worker_zip_sha256": software["sha256"], "actual_worker_snapshots": {"projection": worker["source_sha256"]},
        "raw_runs": {"projection": parent, "continuous": {"run_id": "optimizer"}},
        "geometry": {"actual": raw},
    })
    admission = asset("admission.json", {"inputs": {"resource_probe": resource_pin, "worker_source": software}})
    reconstruction = asset("fit.json", {"identity": {"worker": worker}})
    monkeypatch.setattr(recovery_outputs, "pinned_member", lambda s, p: resource if p == resource_pin else probe)
    monkeypatch.setattr(recovery_outputs, "member", lambda s, p, n: admission if p == req["recipe"] else reconstruction)
    assert verified_semantics(tmp_path, req, worker, "tree")["depth_probe"] == depth
    with pytest.raises(ValueError, match="admitted worker"):
        verified_semantics(tmp_path, req, {**worker, "extension_sha256": "0" * 64}, "tree")
    probe.write_text(json.dumps(actual))
    with pytest.raises(ValueError, match="admitted worker"):
        verified_semantics(tmp_path, req, worker, "tree")


def test_restricted_pt_loader_rechecks_step_zero_parameters(tmp_path):
    torch = pytest.importorskip("torch")
    identity = {"condition": "M2"}
    n = 7
    arrays = {
        k: np.full((n, *tail), i + 0.2, np.float32)
        for i, (k, tail) in enumerate(zip(PARAMS, [(3,), (3,), (4,), (), (3,)], strict=True))
    }
    original = {
        "N": n,
        "parameter_sha256": digest_values(arrays),
        "actual_method": "known_pose_sift_triangulation",
        "scene_scale_m": 0.37,
    }
    recipe = {"topology": {"start": 500, "stop": 3000}}
    params = {WINDOWS_PARAM[k]: torch.as_tensor(v) for k, v in arrays.items()}
    ck = {
        "schema": "el-recovery-checkpoint-1",
        "identity": identity,
        "producer_run_id": "synthetic",
        "step": 0,
        "horizon": 5000,
        "phase": "initial",
        "recipe": recipe,
        "source_order": ["s00", "s03"],
        "next_source_index": 0,
        "counts": {"s00": 0, "s03": 0},
        "actual_method": original["actual_method"],
        "scene_scale_m": 0.37,
        "params": params,
        "rng": dict.fromkeys(["torch_cpu", "torch_cuda0", "python", "numpy"]),
        "strategy": {},
        "adam": {k: {"param_groups": [{"params": [0]}], "state": {}} for k in params},
    }
    p = tmp_path / "step-0000.pt"
    torch.save(ck, p)
    args = dict(
        genid="synthetic",
        identity=identity,
        original=original,
        recipe=recipe,
        step=0,
        n=n,
        counts=ck["counts"],
    )
    assert checkpoint_audit(p, **args) == original["parameter_sha256"]
    ck["params"]["scales"][0, 0] += 0.01
    torch.save(ck, p)
    with pytest.raises(ValueError, match="Initial parameters"):
        checkpoint_audit(p, **args)


def test_fp32_checked_against_actual_png_and_expected_depth_convention(tmp_path):
    a = np.zeros((256, 256), np.float32)
    rgb = np.zeros((256, 256, 3), np.float32)
    depth = np.full_like(a, np.nan)
    rgb[0, 2] = [0, 0.5, 1]
    a[0, 2] = 0.3
    depth[0, 2] = 1.37
    native = tmp_path / "v01.npz"
    png = tmp_path / "v01.png"
    np.savez(native, rgb=rgb, alpha=a, z=depth)
    q = np.floor(rgb * 255 + 0.5).astype(np.uint8)
    Image.fromarray(q).save(png)
    png_readback(png, native)
    q[0, 2, 1] -= 1
    Image.fromarray(q).save(png)
    with pytest.raises(ValueError, match="display PNG"):
        png_readback(png, native)


def test_actual_training_trace_cannot_use_unregistered_source(tmp_path):
    import json

    ids = ("s00", "s03")
    recipe = {"topology": {"start": 500, "stop": 3000, "cap": 20000}}
    p = tmp_path / "steps.jsonl"
    row = {
        "step": 1,
        "source": "s00",
        "pixels": 65536,
        "source_counts": {"s00": 1, "s03": 0},
        "phase": "warmup",
        "n_before": 17,
        "n_after": 17,
        "gradient_l2": dict.fromkeys(WINDOWS_PARAM.values(), 0.01),
    }
    p.write_text(json.dumps(row) + "\n")
    reconstruction = {
        "first_completed_step": 0,
        "last_completed_step": 1,
        "trace_sha256": sha(p),
        "updates_in_process": 1,
        "sampling_counts": {"s00": 1, "s03": 0},
    }
    s = trace_summary(p, reconstruction, ids, recipe)
    assert s["actual_source_pixels"] == 65536 and s["updates_with_nonzero_gradient"] == {"s00": 1}
    row["source"] = "v01"
    p.write_text(json.dumps(row) + "\n")
    reconstruction["trace_sha256"] = sha(p)
    with pytest.raises(ValueError, match="sampler"):
        trace_summary(p, reconstruction, ids, recipe)


def test_one_unsupported_positive_alpha_pixel_is_an_explicit_geometry_failure(tmp_path):
    opacity = np.zeros((256, 256), np.float32)
    opacity[1, 2] = np.nextafter(np.float32(0), np.float32(1))
    depth = np.full_like(opacity, np.nan)
    rgb = np.zeros((*opacity.shape, 3), np.float32)
    raw = tmp_path / "v02.npz"
    image = tmp_path / "v02.png"
    np.savez(raw, rgb=rgb, z=depth, alpha=opacity)
    Image.fromarray(np.zeros_like(rgb, dtype=np.uint8)).save(image)
    with pytest.raises(ValueError, match="1/1 positive-alpha pixels"):
        png_readback(image, raw)
