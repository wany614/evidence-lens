"""Private release integration, using real fixture fits, CPU outputs and independently computed report tables."""

import base64
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens.scoring_io import VIEWS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import release_model as release


def test_private_paths_and_email_are_excluded():
    for bad in (
        "/Volumes/Drive/archive",
        "/private/var/folders/u",
        "/var/folders/u",
        "/mnt/research/x",
        r"\\machine\share",
        "file:///research/x",
        "test.person@example.invalid",
    ):
        with pytest.raises(ValueError):
            release.safe_text("MODEL_CARD.md", bad.encode())
    release.safe_text("LICENSE", b"Copyright Example Holder <example@example.invalid>")
    for value in (r"\\machine\share", "file:///private/data", "/mnt/records", "example@example.invalid"):
        with pytest.raises(ValueError):
            release.safe_text(
                "demo.json",
                json.dumps({"notes": [value]}).replace("/", r"\/").replace("@", r"\u0040").encode(),
            )


def test_reference_proof_requires_the_sealed_native_scene_and_validation(tmp_path):
    payload = tmp_path / "payload"
    (payload / "scene").mkdir(parents=True)
    (payload / "validation").mkdir()
    # The digest/linkage fixture has no renderer or numerical study result.
    native = payload / "scene/unit.blend"
    native.write_bytes(b"private test of the native source pin")
    validation = payload / "validation/results.json"
    save(validation, {"schema": "el-reference-validation-1", "passed": True})
    source = {"cameras": {"sha256": "1" * 64}, "rgb": {"sha256": "2" * 64}}
    meta = {
        "schema": "el-reference-1",
        "input_sha256": source["rgb"]["sha256"],
        "camera_sha256": source["cameras"]["sha256"],
        "validation_passed": True,
        "views": {v: {} for v in VIEWS},
        "scene_path": "scene/unit.blend",
        "validation_sha256": sha(validation),
        "blend_sha256": sha(native),
    }
    meta["views"]["source"]["png_sha256"] = meta["input_sha256"]
    path = payload / "reference.json"
    save(path, meta)
    material = SimpleNamespace(open=lambda name: tmp_path / name)
    verified = release.original_reference_proof(material, source, "heldout_study")
    assert verified["native_blend_sha256"] == sha(native) and verified["validation_sha256"] == sha(validation)
    native.write_bytes(b"different local bytes")
    with pytest.raises(ValueError, match="native scene differs"):
        release.original_reference_proof(material, source, "heldout_study")
    meta["schema"] = "el-synthetic-plane-source-1"
    save(path, meta)
    with pytest.raises(ValueError, match="original Blender"):
        release.original_reference_proof(material, source, "heldout_study")


def substitute_wheel_source(wheel, destination):
    """Change a module and recompute the wheel RECORD as an independent distributor could."""
    target = "evidence_lens/l1.py"
    destination.parent.mkdir()
    with zipfile.ZipFile(wheel) as original, zipfile.ZipFile(destination, "x") as changed:
        record = next(n for n in original.namelist() if n.endswith(".dist-info/RECORD"))
        rows = list(csv.reader(io.StringIO(original.read(record).decode())))
        edited = original.read(target) + b"\n"
        for row in rows:
            if row[0] == target:
                digest = base64.urlsafe_b64encode(hashlib.sha256(edited).digest()).rstrip(b"=").decode()
                row[1:] = ["sha256=" + digest, str(len(edited))]
        replacement = io.StringIO(newline="")
        csv.writer(replacement, lineterminator="\n").writerows(rows)
        replacements = {target: edited, record: replacement.getvalue().encode()}
        for member in original.infolist():
            payload = (
                replacements[member.filename] if member.filename in replacements else original.read(member)
            )
            changed.writestr(member, payload)


def recorder_run(store, repo, cfg, kind, parents, inputs, worker, capsys):
    from evidence_lens import recorder

    args = [
        "--root",
        str(store),
        "--source-root",
        str(repo),
        "--config",
        str(cfg),
        "--kind",
        kind,
        "--experiment",
        "private-release-fixture",
        "--trial",
        load(cfg)["trial_id"],
    ]
    for p in parents:
        args.extend(["--parent", p["run_id"] + "=" + p["manifest_sha256"]])
    for role, (path, owner) in inputs.items():
        args.extend(["--input", role + "=" + str(path)])
        if owner:
            args.extend(["--input-parent", role + "=" + owner["run_id"]])
    code = recorder.main([*args, "--", "{python}", *worker])
    result = capsys.readouterr()
    assert code == 0, (result.out, result.err)
    rec = json.loads(result.out)
    return {k: rec[k] for k in ("run_id", "manifest_sha256")}


def authored_example(work, fixture, capsys):
    store, repo = fixture["store"], fixture["repo"]
    raw = work / "authored-texture"
    raw.mkdir()
    yy, xx = np.indices((12, 16))
    Image.fromarray(np.stack((xx * 13, yy * 17, (xx + yy) * 7), axis=-1).astype(np.uint8)).save(
        raw / "input.png"
    )
    k = np.array([[17.0, 0.0, 7.5], [0.0, 17.0, 5.5], [0.0, 0.0, 1.0]])
    edge = np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]]) @ k
    views = {}
    for i, key in enumerate(VIEWS):
        c = np.eye(4)
        c[0, 3] = i * 0.035
        views[key] = {"width": 16, "height": 12, "K": k.tolist(), "K_edge": edge.tolist(), "c2w": c.tolist()}
    save(
        raw / "cameras.json",
        {"schema": "el-camera-1", "unit": "m", "pixel_center": "integer", "views": views},
    )
    save(
        raw / "reference.json",
        {
            "schema": "el-synthetic-plane-source-1",
            "input_sha256": sha(raw / "input.png"),
            "camera_sha256": sha(raw / "cameras.json"),
        },
    )
    cfg = work / "example-stage.json"
    save(cfg, {"trial_id": "unit-authored-plane", "condition_id": "plane-example"})
    parent = recorder_run(
        store,
        repo,
        cfg,
        "scene_render",
        [],
        {"input": (raw, None)},
        [
            "-c",
            "import shutil,sys;shutil.copytree(sys.argv[1],sys.argv[2],dirs_exist_ok=True)",
            "{input:input}",
            "{out}",
        ],
        capsys,
    )
    src = store / "runs" / parent["run_id"] / "payload"
    inputs = {"rgb": (src / "input.png", parent), "cameras": (src / "cameras.json", parent)}
    example = recorder_run(
        store,
        repo,
        cfg,
        "derivation",
        [parent],
        inputs,
        [
            "{source}/scripts/example_plane.py",
            "--input",
            "{input:rgb}",
            "--cameras",
            "{input:cameras}",
            "--output",
            "{out}",
            "--stride",
            "2",
        ],
        capsys,
    )
    example_root = store / "runs" / example["run_id"] / "payload/example"
    threshold = fixture["threshold"]
    model = store / "runs" / threshold["run_id"] / "payload/model"
    inputs = {
        "source": (example_root / "input.png", example),
        "cameras": (example_root / "cameras.json", example),
        "predictions": (example_root / "predictions", example),
        "model": (model, threshold),
    }
    score = recorder_run(
        store,
        repo,
        cfg,
        "model_score",
        [example, threshold],
        inputs,
        [
            "-m",
            "evidence_lens.l1",
            "--input",
            "{input:source}",
            "--cameras",
            "{input:cameras}",
            "--predictions",
            "{input:predictions}",
            "--model",
            "{input:model}",
            "--model-sha256",
            sha(model / "model.json"),
            "--run-id",
            "{run_id}",
            "--output",
            "{out}",
        ],
        capsys,
    )
    return parent, example, score


def scientific_report(work, f, capsys):
    from test_formal_evaluation import build_fixture_report

    model = f["store"] / "runs" / f["threshold"]["run_id"] / "payload/model"
    result = build_fixture_report(
        work / "report-case",
        store=f["store"],
        checkout=f["repo"],
        frozen_model_dir=model,
        catalog_path=f["catalog"],
        study_path=f["study"],
        method_freeze_path=f["method"],
        model_pin=f["threshold"],
        fit_pin=f["fit"],
    )
    selection = load(result["selection"])
    external_store = work / "remote-comparisons"
    external_store.mkdir()
    cfg = work / "report-stage.json"
    save(cfg, {"trial_id": "unit-study-report", "model_id": f["threshold"]["run_id"]})
    parents = [{k: r[k] for k in ("run_id", "manifest_sha256")} for r in selection["trials"]]
    report = recorder_run(
        external_store,
        f["repo"],
        cfg,
        "report",
        parents,
        {"report": (result["staged"], None)},
        [
            "{source}/scripts/report_study.py",
            "build",
            "--bundle",
            "{input:report}",
            "--output",
            "{out}",
            "--run-id",
            "{run_id}",
        ],
        capsys,
    )
    shutil.copytree(external_store / "runs" / report["run_id"], f["store"] / "runs" / report["run_id"])
    return report


def test_released_wheel_original_proofs_and_zip(tmp_path, capsys, monkeypatch):
    if os.environ.get("EL_TEST_CLEAN_WHEEL") != "1":
        pytest.skip("set EL_TEST_CLEAN_WHEEL=1 for the offline wheel/recorder/release integration")
    pytest.importorskip("sklearn", minversion="1.9.1")
    pytest.importorskip("pyarrow")
    from test_recorded_l1 import build_recorded_learning_fixture, fixture_commit

    f = build_recorded_learning_fixture(tmp_path, capsys, monkeypatch, full_protocol=True)
    repo, store = f["repo"], f["store"]
    monkeypatch.setattr(release, "ROOT", repo)
    import test_formal_evaluation

    from evidence_lens import formal_evaluation, formal_report

    monkeypatch.setattr(test_formal_evaluation, "ROOT", repo)
    monkeypatch.setattr(formal_evaluation, "__file__", str(repo / "src/evidence_lens/formal_evaluation.py"))
    monkeypatch.setattr(formal_report, "__file__", str(repo / "src/evidence_lens/formal_report.py"))
    original, example, score = authored_example(tmp_path, f, capsys)
    report = scientific_report(tmp_path, f, capsys)
    for name in ("README.md", "pyproject.toml"):
        shutil.copyfile(REPO / name, repo / name)
    version = "v0.0.1"
    docs = repo / "docs/releases" / version
    docs.mkdir(parents=True)
    for name in ("MODEL_CARD.md", "RELEASE_NOTES.md", "README.md"):
        (docs / name).write_text(
            "# Private synthetic fixture\n\nThis package tests the release implementation.\n"
        )
    for name in release.LICENSES.values():
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "Private synthetic fixture wording supplied for this test. No license decision for the owner's work.\n"
        )
    addendum = repo / "protocol/analysis-addendum-v1.json"
    save(addendum, {"schema": "el-preregistered-analysis-clarifications-1", "study_sha256": sha(f["study"])})
    save(
        repo / "protocol/operating-points-v1.json",
        {
            "schema": "el-operating-points-clarification-1",
            "parent_study_sha256": sha(f["study"]),
            "parent_addendum_sha256": sha(addendum),
        },
    )
    names = {str(p.relative_to(repo)) for p in (repo / "src/evidence_lens").rglob("*.py")}
    names |= {str(p.relative_to(repo)) for p in (repo / "docs/releases").rglob("*.md")}
    names |= {
        *release.LICENSES.values(),
        "README.md",
        "pyproject.toml",
        "protocol/analysis-addendum-v1.json",
        "protocol/operating-points-v1.json",
    }
    commit = fixture_commit(repo, names, parent=f["commit"])
    wheels = tmp_path / "wheel-out"
    wheel_source = tmp_path / "committed-wheel-source"
    wheel_source.mkdir()
    import tarfile

    tar = subprocess.check_output(["git", "-C", str(repo), "archive", commit])
    with tarfile.open(fileobj=io.BytesIO(tar)) as archive:
        archive.extractall(wheel_source, filter="data")
    subprocess.run(
        ["uv", "build", "--offline", "--wheel", "--out-dir", str(wheels), str(wheel_source)],
        check=True,
        capture_output=True,
    )
    wheel = next(wheels.glob("*.whl"))
    substituted = tmp_path / "changed-wheel" / wheel.name
    substitute_wheel_source(wheel, substituted)
    with pytest.raises(ValueError, match="wheel source differs from release commit"):
        release.read_wheel(substituted, (repo / "pyproject.toml").read_bytes(), checkout=repo, commit=commit)

    def arg(p):
        return p["run_id"] + "=" + p["manifest_sha256"]

    def cli(*args):
        command = [sys.executable, str(repo / "scripts/release_model.py"), *map(str, args)]
        p = subprocess.run(
            command,
            check=False,
            cwd=tmp_path,
            env=dict(os.environ, PYTHONPATH=str(repo / "src")),
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, p.stderr
        return json.loads(p.stdout)

    output = tmp_path / "release"
    a = SimpleNamespace(
        store=store,
        method=f["method"],
        method_sha256=sha(f["method"]),
        version=version,
        commit=commit,
        wheel=wheel,
        example=arg(example),
        example_source=arg(original),
        scores=arg(score),
        report=arg(report),
        output=output,
        fixture=False,
    )
    with pytest.raises(ValueError, match="completed held-out study"):
        release.build(a)
    built = cli(
        "build",
        "--fixture",
        "--store",
        store,
        "--method",
        f["method"],
        "--method-sha256",
        sha(f["method"]),
        "--version",
        version,
        "--commit",
        commit,
        "--wheel",
        wheel,
        "--example",
        a.example,
        "--example-source",
        a.example_source,
        "--scores",
        a.scores,
        "--report",
        a.report,
        "--output",
        output,
    )
    assert built["kind"] == "synthetic_fixture" and built["wheel_sha256"] == sha(wheel)
    members = release.check(output, built["release_sha256"], built["checksums_sha256"])
    archive = tmp_path / "fixture-model.zip"
    pins = ["--release-sha256", built["release_sha256"], "--checksums-sha256", built["checksums_sha256"]]
    packed = cli("zip", "--folder", output, "--output", archive, *pins)
    assert packed == release.package(output, archive) and packed["sha256"] == sha(archive)
    assert (tmp_path / packed["checksum_file"]).read_text() == packed["sha256"] + "  fixture-model.zip\n"
    verified = cli(
        "verify", "--folder", output, *pins, "--archive", archive, "--archive-sha256", packed["sha256"]
    )
    assert verified["archive"]["members"] == len(members) == verified["verified_files"]
    assert verified["externally_pinned"] and verified["wheel_sha256"] == sha(wheel)
    with pytest.raises(ValueError, match="external pin"):
        release.check(output, "0" * 64)
    unpacked = tmp_path / "extracted-only"
    with zipfile.ZipFile(archive) as zipf:
        zipf.extractall(unpacked)
    env = tmp_path / "wheel-only-env"
    subprocess.run(["uv", "venv", "--python", sys.executable, str(env)], check=True, capture_output=True)
    exe = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--offline",
            "--python",
            str(exe),
            str(unpacked / "distribution" / wheel.name),
        ],
        check=True,
        capture_output=True,
    )
    graded = tmp_path / "wheel-release-grade"
    subprocess.run(
        [
            str(exe),
            "-I",
            "-m",
            "evidence_lens.l1_demo",
            "grade",
            "--bundle",
            str(unpacked),
            "--manifest-sha256",
            built["demo_sha256"],
            "--output",
            str(graded),
        ],
        check=True,
        capture_output=True,
        cwd=unpacked,
    )
    result = load(graded / "grade.json")
    assert (
        result["passed"]
        and result["numerical_l1_parity_exercised"]
        and result["runtime"]["optional_training_render_packages_present"] == []
    )
    maps = tmp_path / "wheel-release-maps"
    subprocess.run(
        [
            str(exe),
            "-I",
            "-m",
            "evidence_lens.score_maps",
            "--scores",
            str(graded / "scores"),
            "--receipt-sha256",
            sha(graded / "scores/l1_score.json"),
            "--view",
            "v01",
            "--method",
            "L1-full",
            "--output",
            str(maps),
        ],
        cwd=unpacked,
        check=True,
        capture_output=True,
    )
    view = load(maps / "v01.json")
    assert load(maps / "index.json")["views"]["v01"]["sha256"] == sha(maps / "v01.json")
    assert view["source"]["model_id"] == built["model_id"] and view["method"] == "L1-full"
    with np.load(graded / "scores/v01.npz", allow_pickle=False) as numerical:
        state, risk, valid = (numerical[k] for k in ("l1_full_state", "l1_full", "l1_full_valid"))
    assert view["counts"] == {
        "total": risk.size,
        "valid": int(valid.sum()),
        "low": int(np.sum(state == 0)),
        "high": int(np.sum(state == 1)),
        "unknown": int(np.sum(state == 255)),
    }
    color = np.full(risk.shape, 255, dtype=np.uint8)
    color[valid] = np.rint(254 * risk[valid].astype(np.float64)).astype(np.uint8)
    for field, expected in (("state", state), ("continuous", color)):
        record = view["maps"][field]
        path = maps / record["file"]
        assert record["sha256"] == sha(path)
        with Image.open(path) as png:
            assert png.format == "PNG" and np.array_equal(np.asarray(png), expected)
    # Rehash an unrelated study selection to show that internal model/study bindings remain independently enforced.
    sel = unpacked / "results/selection.json"
    s = load(sel)
    s["model_id"] = "unrelated-test-model"
    save(sel, s)
    idx = load(unpacked / "release.json")
    idx["files"]["results/selection.json"].update(release.signature(sel))
    proof = unpacked / "proofs/report-public.json"
    q = load(proof)
    q["files"]["public-numerical/selection.json"] = release.signature(sel)
    save(proof, q)
    idx["files"]["proofs/report-public.json"].update(release.signature(proof))
    idx["payload_members_sha256"] = release.aggregate(
        {n: release.signature(unpacked / n) for n in idx["files"]}
    )
    (unpacked / "release.json").write_bytes(encoded(idx))
    (unpacked / "SHA256SUMS").write_text(
        "".join(sha(unpacked / n) + "  " + n + "\n" for n in sorted(set(members) - {"SHA256SUMS"}))
    )
    with pytest.raises(ValueError, match="different model"):
        release.check(unpacked)
