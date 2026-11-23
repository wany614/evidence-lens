"""Fresh-checkout reconstruction without rendering, network access or historical research data."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from evidence_lens._bundle import encoded, load, new_id, sha
from evidence_lens.blender_reference.formal_catalog import CATALOG, LOCK, STUDY, resolved, validate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rebuild_study as rebuild


def test_original_worker_from_clean_git_and_supplied_zip(tmp_path):
    available = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", rebuild.WORKER_COMMIT + "^{commit}"], capture_output=True)
    if available.returncode:
        pytest.skip("this source distribution omits the original worker commit")
    checkout, store = tmp_path / "clean", tmp_path / "store"
    subprocess.run(["git", "clone", "--quiet", "--no-local", "--no-checkout", str(ROOT), str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "switch", "--quiet", "--detach", rebuild.WORKER_COMMIT], check=True)
    (checkout / "untracked-private.txt").write_text("fixture outside the source inventory")
    commit, source, _ = rebuild.committed_source(checkout)
    assert commit == rebuild.WORKER_COMMIT and "untracked-private.txt" not in source
    catalog = ROOT / CATALOG
    result = rebuild.worker(store, catalog, checkout)
    bundle = Path(result["path"])
    before = bundle.stat()
    assert sha(bundle) == rebuild.WORKER_SHA and result["members"] == 19
    receipt = load(bundle.with_name("worker-source.json"))
    assert len(receipt["files"]) == 19 and "scripts/windows_stage.py" in receipt["files"]
    assert all(rebuild.sha_bytes(source[name]) == digest for name, digest in receipt["files"].items())
    assert rebuild.worker(store, catalog, tmp_path / "no-git", bundle) == result
    assert bundle.stat().st_ino == before.st_ino and bundle.stat().st_mtime_ns == before.st_mtime_ns
    other = tmp_path / "other"
    pin = other / "formal" / sha(catalog) / "worker-source.json"
    pin.parent.mkdir(parents=True)
    pin.write_bytes(b"existing independent freeze")
    with pytest.raises(ValueError, match="already frozen"):
        rebuild.worker(other, catalog, checkout)
    assert pin.read_bytes() == b"existing independent freeze" and not pin.with_name("worker-source.zip").exists()


def test_changed_worker_rejected_before_store_creation(tmp_path):
    changed = tmp_path / "worker.zip"
    changed.write_bytes(b"different source")
    with pytest.raises(ValueError, match="worker bytes differ"):
        rebuild.worker(tmp_path / "store", ROOT / CATALOG, tmp_path, changed)
    assert not (tmp_path / "store").exists()


def test_zip_only_checkout_reconstructs_all_historical_blobs(tmp_path):
    checkout = tmp_path / "export"
    descriptor = load(ROOT / rebuild.WORKER_DESCRIPTOR)
    for name in descriptor["files"]:
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (checkout / "pyproject.toml").write_text('[project]\nversion = "later-release"\n')
    result = rebuild.worker(tmp_path / "store", ROOT / CATALOG, checkout)
    assert result["worker_source_sha256"] == rebuild.WORKER_SHA and not (checkout / ".git").exists()
    changed = next(name for name, row in descriptor["files"].items() if "edits" in row)
    with (checkout / changed).open("ab") as out:
        out.write(b"# independent local change\n")
    with pytest.raises(ValueError, match="differs from both"):
        rebuild.worker(tmp_path / "another-store", ROOT / CATALOG, checkout)
    assert not (tmp_path / "another-store").exists()


def test_replication_uses_new_lineage_and_same_authored_geometry(tmp_path, monkeypatch):
    source, store = tmp_path / "source", tmp_path / "research"
    parents = load(ROOT / LOCK)
    names = {*parents["frozen_files"], LOCK, rebuild.ADDENDUM, rebuild.POINTS}
    for name in names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    blobs = {name: (source / name).read_bytes() for name in names}
    entries = {name: (0o100644, "fixture") for name in names}
    monkeypatch.setattr(rebuild, "committed_source", lambda p: ("a" * 40, blobs, entries))
    # Schema-identical role pins stand in for a newly selected real pilot. The selector's independent suite
    # exercises its recorded seven-stage graph; this test exercises registration and the original validator.
    orig = load(source / CATALOG)["families"][0]
    pins = {k: (new_id() if k.endswith("_run_id") else "4" * 64) for k in orig["reuse"]}
    pins.update(reference_config_sha256="5" * 64, readback_config_sha256="6" * 64)
    with pytest.raises(FileNotFoundError):
        rebuild.register(store, source, tmp_path / "missing", "independent-test")
    monkeypatch.setattr(rebuild, "pilot_evidence", lambda *args: pins)
    with pytest.raises(ValueError, match="new project directory"):
        rebuild.register(store, source, source, "independent-test")
    result = rebuild.register(store, source, tmp_path / "replication", "independent-test")
    derived = Path(result["project"])
    old, designs, base, original_lock = validate(source)
    new, _, _, lock = validate(derived)
    assert new["replication"]["name"] == "independent-test" and new["families"][0]["reuse"] == pins
    assert result["catalog_sha256"] == lock["catalog_sha256"] != original_lock["catalog_sha256"]
    assert sha(source / CATALOG) == rebuild.ORIGINAL_CATALOG and old["families"][0]["reuse"] == orig["reuse"]
    assert {n for n in names if (derived / n).read_bytes() != blobs[n]} == {STUDY, CATALOG, LOCK, rebuild.ADDENDUM, rebuild.POINTS}
    assert new["families"][1:] == old["families"][1:]
    assert resolved(old["families"][1], old, designs, base, original_lock)["reference"] == resolved(new["families"][1], new, designs, base, lock)["reference"]
    assert load(derived / rebuild.ADDENDUM)["study_sha256"] == sha(derived / STUDY)
    assert load(derived / rebuild.POINTS)["parent_addendum_sha256"] == sha(derived / rebuild.ADDENDUM)
    assert load(store / "protocol/replication-registration.json")["catalog_sha256"] == result["catalog_sha256"]
    duplicate = rebuild.register(store, source, tmp_path / "second-path", "independent-test")
    assert duplicate["catalog_sha256"] == result["catalog_sha256"]
    with pytest.raises(ValueError, match="different registration"):
        rebuild.register(store, source, tmp_path / "other", "other-study")
    with pytest.raises(ValueError, match="new project directory"):
        rebuild.register(store, source, derived, "independent-test")


def test_pilot_rejects_historical_or_changed_source(tmp_path, monkeypatch):
    catalog, _, base, lock = validate(ROOT)
    row = dict(catalog["families"][0])
    trial = row["trial_id"]
    store = tmp_path / "store"
    path = store / "trials" / trial / "trial_selection.json"
    path.parent.mkdir(parents=True)
    ids = {role: new_id() for role in rebuild.KINDS}
    selected = {"scene_id": row["scene_id"], "scene_family_id": row["scene_family_id"], "trial_id": trial,
                "experiment_id": "p2-single-image", "condition_id": "B0",
                "selected": {role: {"run_id": rid, "manifest_sha256": "e" * 64} for role, rid in ids.items()}}
    path.write_bytes(encoded(selected))
    recorded = []
    monkeypatch.setattr(rebuild, "freeze", lambda *args: recorded.append(args))
    historic = dict(row, reuse=dict(row["reuse"], selection_sha256=sha(path)))
    with pytest.raises(ValueError, match="historical private tabletop"):
        rebuild.pilot_evidence(store, ROOT, historic, lock)
    assert not recorded
    windows = load(ROOT / "configs/windows-b0.json")
    cpu = load(ROOT / "configs/p2-cpu.json")
    for role in ("generation", "render", "score", "evaluation", "recompute"):
        root = store / "runs" / ids[role]
        root.mkdir(parents=True)
        (root / "config.resolved.yaml").write_bytes(encoded(windows if role in ("generation", "render") else cpu))
    generated = store / "runs" / ids["generation"] / "payload"
    generated.mkdir()
    (generated / "generation.json").write_bytes(encoded({"checkpoint_sha256": windows["generator"]["checkpoint_sha256"]}))
    for role in ("reference", "readback"):
        root = store / "runs" / ids[role]
        root.mkdir(parents=True)
        (root / "config.resolved.yaml").write_bytes(encoded(base))
        for name in lock["frozen_files"]:
            if name.startswith("src/evidence_lens/blender_reference/"):
                dst = root / "source" / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dst)
    payload = store / "runs" / ids["reference"] / "payload"
    (payload / "scene").mkdir(parents=True)
    (payload / "input.png").write_bytes(b"input fixture")
    (payload / "cameras.json").write_bytes(b"camera fixture")
    blend = payload / "scene/scene_001_tabletop_v001.blend"
    blend.write_bytes(b"blend fixture")
    row["camera_sha256"] = sha(payload / "cameras.json")
    data = {"camera_sha256": row["camera_sha256"], "blend_sha256": sha(blend), "input_sha256": sha(payload / "input.png")}
    (payload / "reference.json").write_text(json.dumps(data))
    pins = rebuild.pilot_evidence(store, ROOT, row, lock)
    assert recorded == [(store, trial, ids)] and pins["blend_sha256"] == sha(blend)
    (store / "runs" / ids["readback"] / "source/src/evidence_lens/blender_reference/verify_reference.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="implementation differs"):
        rebuild.pilot_evidence(store, ROOT, row, lock)
