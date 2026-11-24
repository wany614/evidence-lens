"""Small generated interface fixtures. Published-example regression is a separate recorded operation."""

import io
import json

import numpy as np
import pytest
from PIL import Image

from evidence_lens.multisource import extract
from evidence_lens_bundle.bundle import Bundle
from evidence_lens_bundle.cli import main
from evidence_lens_bundle.common import (
    PROFILE,
    RELEASE_INDEX_SHA256,
    SEMANTICS,
    encoded,
    profile,
    row,
    sha,
    write,
)
from evidence_lens_bundle.report import render
from evidence_lens_bundle.scoring import score

S = ["camera 91", "rig/left.34", "camera:7", "A13"]
T = ["end/701", "17"]
H = "1" * 64


def fixture(root, *, analysis_profile=None):
    root.mkdir()

    def obj(name, data):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return row(name, data)

    camera = {
        "schema": "el-camera-1",
        "unit": "m",
        "pixel_center": "integer",
        "image_origin": "top_left",
        "axis_convention": "OpenCV x right, y down, z forward",
        "views": {},
    }
    frames = {}
    photos = {}
    intrinsics = [[125, 0, 127.5], [0, 125, 127.5], [0, 0, 1]]
    grids = None
    w = h = 256
    if analysis_profile is not None:
        from evidence_lens_bundle.grids import build_grids

        original = {"width": 256, "height": 256, "K": intrinsics, "distortion": "none"}
        grids = build_grids(analysis_profile, {n: original for n in S + T})
        w, h = grids["analysis_size"]
    for i, n in enumerate(S + T):
        pose = np.eye(4)
        pose[0, 3] = i * 0.01
        camera["views"][n] = {
            "K": grids["views"][n]["analysis_K"] if grids else intrinsics,
            "c2w": pose.tolist(),
            "width": w,
            "height": h,
            "projection": "pinhole",
            "distortion": "none",
        }
        if n in S:
            b = io.BytesIO()
            Image.new("RGB", (w, h), (20 + 12 * i, 30, 60)).save(b, "PNG")
            photos[n] = obj(f"sources/{i}.png", b.getvalue())
        if n == T[-1]:
            frames[n] = {"status": "render_failed", "reason": "declared camera render failure"}
            continue
        zz = np.full((h, w), 2, np.float32)
        if n in T:
            zz[0, 0] = np.nan
        a = io.BytesIO()
        np.savez_compressed(
            a, rgb=np.full((h, w, 3), 0.25 + i * 0.02, np.float32), z=zz, alpha=np.ones_like(zz)
        )
        frames[n] = {"status": "ready", "file": obj(f"renders/{i}.npz", a.getvalue())}
    c = obj("cameras.json", encoded(camera))
    producer = {
        "schema": "el-render-evidence-2" if grids else "el-render-evidence-1",
        **({"grids_sha256": sha(encoded(grids))} if grids else {}),
        "generator": {"run_id": "test-gen", "receipt_sha256": H},
        "renderer": {"run_id": "test-render", "receipt_sha256": H},
        "checkpoint_sha256": H,
        "camera_sha256": c["sha256"],
        "source_rgb_sha256": {s: photos[s]["sha256"] for s in S},
        "source_order": S,
        "target_order": T,
        "semantics": SEMANTICS,
        "depth_verified": True,
        "depth_probe_sha256": H,
        "frames": frames,
    }
    d = {
        "schema": "el-evidence-bundle-2" if grids else "el-evidence-bundle-1",
        **({"grids": grids} if grids else {}),
        "creation": {
            "run_id": "fixture-adapter",
            "implementation": "test only",
            "code_sha256": H,
            "config_sha256": H,
            "original_case_sha256": None,
        },
        "input_domain": "Generated test",
        "condition": "M4",
        "source_order": S,
        "primary_source_id": S[2],
        "endpoint_source_ids": [S[3], S[0]],
        "target_order": T,
        "camera": c,
        "geometry": {
            "prepared_camera_and_z_unit": "m",
            "meters_per_original_unit": 1,
            "metric_evidence": "test metric fixture",
            "crop": "none",
            "resize": "none",
            "undistortion": "none",
            "scoring_resampling": "none",
        },
        "sources": photos,
        "producer": obj("producer.json", encoded(producer)),
        "risk_profile": {
            "id": PROFILE,
            "release_index_sha256": RELEASE_INDEX_SHA256,
            "sha256": sha(encoded(profile())),
        },
    }
    write(root / "bundle.json", d)
    return root


def replace(root, file, key, val):
    b = json.loads((root / "bundle.json").read_bytes())
    obj = b if file == "bundle.json" else json.loads((root / file).read_bytes())
    ref = obj
    for k in key[:-1]:
        ref = ref[k]
    ref[key[-1]] = val
    write(root / file, obj)
    if file == "bundle.json":
        return
    role = "camera" if file == "cameras.json" else "producer"
    b[role] = row(file, (root / file).read_bytes())
    if role == "camera":
        p = json.loads((root / "producer.json").read_bytes())
        p["camera_sha256"] = b[role]["sha256"]
        write(root / "producer.json", p)
        b["producer"] = row("producer.json", (root / "producer.json").read_bytes())
    write(root / "bundle.json", b)


def test_external_order_primary_and_deployment_html(tmp_path):
    root = fixture(tmp_path / "in")
    b = Bundle(root)
    assert b.validate()["primary_source_id"] == S[2]
    with pytest.raises(ValueError, match="source IDs"):
        extract(b.frame(T[0]), b.cameras[T[0]], {s: b.source(s) for s in S})
    score_dir = tmp_path / "scores"
    r = score(root, score_dir)
    assert r["source_order"] == S and r["target_order"] == T and r["primary_source_id"] == S[2]
    assert r["scorer_source_ids"] == S and r["endpoint_source_ids"] == [S[3], S[0]]
    assert r["counts"]["canvas"] == 2 * 256**2 and r["views"][1]["counts"]["unknown"] == 256**2
    assert r["views"][0]["unknown_reasons"]["invalid_target_z"] == 1
    html = tmp_path / "a.html"
    assert render(score_dir, html)["views"] == 2
    body = html.read_text()
    assert (
        "data:image/png;base64," in body
        and "data:application/json;base64," in body
        and "addEventListener" in body
    )
    assert "rig/left.34" in body and "end/701" in body and "http://" not in body and "https://" not in body
    assert "MAE" not in body and "accuracy" not in body
    with pytest.raises(FileExistsError):
        render(score_dir, html)
    # Producer reads explicit members only. A separately kept reference cannot change the score product.
    (root / "ReferenceBundle").mkdir()
    (root / "ReferenceBundle" / "DONT_READ").write_bytes(b"unrelated evidence")
    assert Bundle(root).validate()["status"] == "valid"
    with pytest.raises(ValueError):
        Bundle(root, "0" * 64)
    array = score_dir / r["views"][0]["arrays"]["path"]
    array.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes differ"):
        render(score_dir, tmp_path / "bad.html")


@pytest.mark.parametrize(
    ("file", "key", "val", "message"),
    [
        ("bundle.json", ["reference"], {"path": "ReferenceBundle"}, "bundle"),
        ("bundle.json", ["primary_source_id"], "unknown-source", "primary"),
        ("bundle.json", ["endpoint_source_ids"], [S[0], S[0]], "duplicate"),
        ("bundle.json", ["source_order"], list(reversed(S)), "order"),
        ("bundle.json", ["geometry", "meters_per_original_unit"], 0, "scale"),
        ("bundle.json", ["risk_profile", "id"], "new cutoff", "profile"),
        ("producer.json", ["semantics", "z"], "euclidean_range", "semantics"),
        ("producer.json", ["semantics", "alpha"], "density", "semantics"),
        ("producer.json", ["source_rgb_sha256", S[0]], "0" * 64, "source photo hashes"),
        ("producer.json", ["frames", T[1]], {"status": "render_failed"}, "failed render"),
        ("producer.json", ["frames", T[0], "file", "path"], "../GT.npz", "unsafe"),
        ("cameras.json", ["views", T[0], "distortion"], "brown", "pinhole"),
        ("cameras.json", ["views", T[0], "width"], 257, "256"),
        ("cameras.json", ["views", S[0], "w2c"], (-np.eye(4)).tolist(), "w2c"),
    ],
)
def test_reject_malformed_or_unsupported(tmp_path, file, key, val, message):
    root = fixture(tmp_path / "in")
    replace(root, file, key, val)
    with pytest.raises(ValueError, match=message):
        Bundle(root).validate()


def test_unknown_and_missing_artifact_codes(tmp_path, capsys):
    root = fixture(tmp_path / "in")
    replace(root, "producer.json", ["depth_verified"], False)
    r = score(root, tmp_path / "score")
    assert r["views"][0]["unknown_reasons"]["depth_unverified"] == 256**2
    (root / "renders/4.npz").unlink()
    assert main(["validate", "--bundle", str(root)]) == 2 and "missing" in capsys.readouterr().err
    replace(root, "producer.json", ["semantics", "z"], "unsupported")
    assert main(["validate", "--bundle", str(root)]) == 3 and "unsupported" in capsys.readouterr().err


def test_legacy_proof_rejects_unpinned_or_unrecognized_roles(tmp_path):
    from evidence_lens_bundle.proof import verified_docs

    proof = {
        "schema": "el-legacy-deploy-proof-1",
        "kind": "recovery",
        "original_case_sha256": H,
        "members": {"reference_rgb": {}},
        "seals": {},
    }
    write(tmp_path / "proof.json", proof)
    with pytest.raises(ValueError, match="proof SHA-256"):
        verified_docs(tmp_path, "0" * 64)
    with pytest.raises(ValueError, match="evidence roles"):
        verified_docs(tmp_path, sha((tmp_path / "proof.json").read_bytes()))


@pytest.mark.parametrize("role", ("case", "prediction", "views"))
def test_invalid_legacy_producer_gets_a_field_error(tmp_path, capsys, role):
    from test_stage3_multisource import case_fixture

    root = tmp_path / "input"
    case_fixture(root)
    cm, pm = root / "case.json", root / "prediction.json"
    c, p = json.loads(cm.read_bytes()), json.loads(pm.read_bytes())
    if role == "views":
        p["views"]["v01"] = []
        message = "prediction.views[v01]"
    else:
        (c if role == "case" else p)["reconstruction_run_id"] = " "
        message = role + ".json reconstruction_run_id"
    write(pm, p)
    c["prediction"] = row("prediction.json", pm.read_bytes())
    write(cm, c)
    target = tmp_path / "output"
    assert (
        main(
            ["adapt-legacy", "--case", str(root), "--case-sha256", sha(cm.read_bytes()), "--out", str(target)]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "invalid" and message in error["message"]
    assert not target.exists()


def test_validate_bad_npz_reports_invalid(tmp_path, capsys):
    root = fixture(tmp_path / "input")
    f = Bundle(root).producer["frames"][T[0]]["file"]
    path = root / f["path"]
    path.write_bytes(b"PK\x03\x04corrupt")
    replace(root, "producer.json", ["frames", T[0], "file"], row(f["path"], path.read_bytes()))
    assert main(["validate", "--bundle", str(root)]) == 2
    err = json.loads(capsys.readouterr().err)
    assert err["status"] == "invalid" and "NPZ" in err["message"]
