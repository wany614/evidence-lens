"""Exercise formal handoff and dispatch with tiny, sealed local unit fixtures."""

import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

from evidence_lens._bundle import BundleError, load, save, sha
from evidence_lens import recorder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import formal_lineage
import make_handoff
import windows_stage
import worker_jobs


def seal(store, project, config, kind, payload, capsys, parents=(), inputs=()):
    args = ["--root", str(store), "--source-root", str(project), "--config", str(config), "--kind", kind,
            "--experiment", formal_lineage.EXPERIMENT, "--trial", load(config)["trial_id"]]
    for parent in parents:
        args += ["--parent", parent]
    for i, path in enumerate(inputs):
        args += ["--input", f"input{i}={path}"]
    run = recorder.main([*args, "--", "{python}", "-c", "import shutil,sys;shutil.copytree(sys.argv[1],sys.argv[2],dirs_exist_ok=True)",
                         str(payload), "{out}"])
    output = capsys.readouterr()
    assert run == 0, (output.out, output.err)
    return json.loads(output.out)


def test_formal_handoff_isolation_resumption_and_pins(tmp_path, capsys, monkeypatch):
    project, store = tmp_path / "project", tmp_path / "store"
    (project / "scripts").mkdir(parents=True)
    (project / "scripts/stub.py").write_text("# synthetic fixture\n")
    row = {"scene_family_id":"fixture", "scene_id":"fixture/v001", "scene_version":"v001", "trial_id":"fixture-B0", "split":"train",
           "cv_fold":0, "example_view":"v01", "trials":{"B0":"fixture-B0"}}
    catalog, freeze, cfg = (tmp_path / n for n in ("families-v1.json", "freeze.json", "config.json"))
    camera_file = tmp_path / 'cameras.json'
    save(camera_file, {"views":dict.fromkeys(formal_lineage.VIEWS, {})})
    row.update(camera_sha256=sha(camera_file), recipe_sha256="c"*64)
    save(catalog, {"schema":"el-formal-families-1", "dataset_id":"fixture-dataset", "families":[row]})
    save(freeze, {"catalog_sha256":sha(catalog)})
    family = formal_lineage.family_record(catalog, freeze, "fixture")
    save(cfg, family)
    scene, proof = tmp_path / "scene", tmp_path / "proof"
    scene.mkdir(); proof.mkdir()
    (scene / "input.png").write_bytes(b"synthetic image bytes")
    (scene / "test.blend").write_bytes(b"private synthetic truth never transferred")
    (scene / "private-truth.bin").write_bytes(b"must not transfer")
    save(scene / "cameras.json", {"views":dict.fromkeys(formal_lineage.VIEWS, {})})
    source_hash, camera_hash = sha(scene / "input.png"), sha(scene / "cameras.json")
    save(scene / "reference.json", {"schema":"el-reference-1", "scene_id":row["scene_id"], "scene_family_id":"fixture", "validation_passed":True,
         "input_sha256":source_hash, "camera_sha256":camera_hash, "blend_sha256":sha(scene / "test.blend"), "scene_path":"test.blend",
         "formal":{"recipe_sha256":row["recipe_sha256"]}, "views":dict.fromkeys(formal_lineage.VIEWS, {})})
    ref = Path(seal(store, project, cfg, "scene_render", scene, capsys)["path"])
    save(proof / "reference_readback.json", {"schema":"el-reference-readback-1", "scientific_acceptance":True, "integrity_checked":True,
         "verification_version":"1.2", "dataset_hash":family["dataset_hash"], "failed_scientific_checks":[], "reference_json_sha256":sha(ref / "payload/reference.json"), "views":dict.fromkeys(formal_lineage.VIEWS, {})})
    rb = Path(seal(store, project, cfg, "archive_verify", proof, capsys, [ref.name], [ref / "payload/reference.json"])["path"])
    ref_selections = store / "formal-references" / family["dataset_hash"]
    ref_selections.mkdir(parents=True)
    save(ref_selections / "fixture.json", {"schema":"el-formal-reference-selection-1", **family,
         "reference_run_id":ref.name, "readback_run_id":rb.name, "reference_manifest_sha256":sha(ref/"manifest.json"),
         "readback_manifest_sha256":sha(rb/"manifest.json"), "camera_sha256":camera_hash,"input_sha256":source_hash,
         "blend_sha256":sha(scene/"test.blend")})
    archive = tmp_path / "handoff.zip"
    result = make_handoff.build(ref, archive, rb, catalog, freeze)
    base = tmp_path / "unpacked"
    with zipfile.ZipFile(archive) as z:
        assert {n for n in z.namelist() if not n.startswith('code/')} == {'handoff.json', 'input.png','cameras.json'}
        assert not any('blender_reference' in n or n.endswith('families-v1.json') or n.endswith('README.md') for n in z.namelist())
        z.extractall(base)
    job, digest = windows_stage.read_handoff(base)
    assert job['identity']['scene_id'] == 'fixture/v001' and job['identity']['dataset_hash'] == sha(catalog)
    assert job['readback']['run_id'] == rb.name and result['handoff_sha256'] == digest
    monkeypatch.setattr(make_handoff, 'source_files', lambda **kwargs: pytest.fail('must use frozen worker source'))
    assert make_handoff.build(ref, archive, rb, catalog, freeze) == result
    (base / 'unexpected.png').write_bytes(b'truth');
    with pytest.raises(BundleError, match='undeclared'):
        windows_stage.read_handoff(base)
    (base / 'unexpected.png').unlink()
    save(cfg, dict(job['identity'], formal_handoff_sha256=digest))
    gen = tmp_path / 'gen'; gen.mkdir(); (gen / 'opaque').write_bytes(b'synthetic')
    outgoing = tmp_path / 'outgoing'; outgoing.mkdir()
    calls = []
    def launch():
        calls.append(1)
        return seal(store, project, cfg, 'generation', gen, capsys, [ref.name])
    orig = worker_jobs.outgoing_receipt
    def interrupted(*args):
        raise OSError('synthetic packaging interruption')
    monkeypatch.setattr(worker_jobs, 'outgoing_receipt', interrupted)
    with pytest.raises(OSError, match='packaging'):
        worker_jobs.dispatch(store, outgoing, job, digest, 'generation', None, None, None, launch)
    monkeypatch.setattr(worker_jobs, 'outgoing_receipt', orig)
    result = worker_jobs.dispatch(store, outgoing, job, digest, 'generation', None, None, None, launch)
    assert result['status'] == 'succeeded' and result['sha256'] == sha(outgoing / result['package'])
    assert load(outgoing / (result['run'] + '.receipt.json')) == result
    assert worker_jobs.dispatch(store, outgoing, job, digest, 'generation', None, None, None, launch) == result
    assert calls == [1]
    with pytest.raises(BundleError, match='conflicting'):
        worker_jobs.dispatch(store, outgoing, job, 'f'*64, 'generation', None, None, None, launch)
    with pytest.raises(BundleError, match='successful producer'):
        worker_jobs.check_generation(store, ref.name, job, digest)


def test_selected_learning_extract_uses_only_frozen_train_features_and_labels(tmp_path, monkeypatch, capsys):
    import formal_batch
    from test_trial_selection import build_frozen_fixture

    store, selected_file = build_frozen_fixture(tmp_path, monkeypatch, capsys)
    s = load(selected_file)
    row = {"scene_family_id": s["scene_family_id"], "scene_id":s["scene_id"], "scene_version":"v001", "trial_id":s["trial_id"],
           "split":"train", "cv_fold":0, "example_view":"v01", "trials":{"B0":s["trial_id"]}, "reuse":{"selection_sha256":sha(selected_file)},
           "camera_sha256":s["evidence"]["camera_set_sha256"], "recipe_sha256":"9"*64}
    catalog, freeze = tmp_path / 'families-v1.json', tmp_path / 'frozen.json'
    save(catalog, {"schema":"el-formal-families-1", "dataset_id":"learning-test", "families":[row]})
    save(freeze, {"catalog_sha256":sha(catalog)})
    result = formal_batch.prepare(store, catalog, freeze, row['scene_family_id'])
    assert result['family_pin']['score_run_id'] == s['selected']['score']['run_id']
    sealed = store / 'runs' / result['run_id']
    family = load(sealed / 'payload/family.json')
    assert set(family['views']) == set(formal_batch.TARGETS) and family['views']['v01']['E'] == 0
    assert {i['role'] for i in load(sealed / 'inputs.json')['inputs']} == {'scores', 'evaluation', 'family', 'catalog'}
    assert all(not n.endswith('.ply') and 'prediction' not in n for n in formal_batch.files(sealed/'inputs'))
    assert formal_batch.prepare(store, catalog, freeze, row['scene_family_id']) == result
    import formal_archive
    vault = tmp_path / 'offline-volume'
    archive = formal_archive.archive(store, vault, catalog, freeze, row['scene_family_id'])
    retrieved = formal_archive.verify_vault(vault, Path(archive['manifest']), archive['manifest_sha256'])
    assert retrieved['selection_sha256'] == sha(selected_file)
    assert formal_archive.archive(store, vault, catalog, freeze, row['scene_family_id']) == archive
    obj = formal_archive.asset_path(vault, retrieved['selection_sha256'])
    obj.chmod(0o600)
    obj.write_bytes(b'corrupt archive fixture')
    with pytest.raises(BundleError, match='missing or changed retrieved'):
        formal_archive.verify_vault(vault, Path(archive['manifest']), archive['manifest_sha256'])
    row.update(split='test', cv_fold=None)
    save(catalog, {"schema":"el-formal-families-1", "dataset_id":"learning-test", "families":[row]})
    with pytest.raises(BundleError, match='catalog differs'):
        formal_batch.prepare(store, catalog, freeze, row['scene_family_id'])
    save(freeze, {"catalog_sha256":sha(catalog)})
    with pytest.raises(BundleError, match='test families'):
        formal_batch.prepare(store, catalog, freeze, row['scene_family_id'])
