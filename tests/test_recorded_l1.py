"""Tiny synthetic producer bundles for the recorder and freeze-command integration."""

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
pytest.importorskip("psutil")
from evidence_lens import formal_inputs, recorder
from evidence_lens._bundle import load, save, sha
from evidence_lens.l1_data import prepare
from test_cpu_p2 import deployable
from test_l1 import make_extract

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import formal_learning as fl
import formal_lineage
from verify_l1_development import pilot_masks, record_audit


def fixture_commit(repo, names, parent=None):
    """Create a temporary Git object for the method-byte check without staging a working tree."""
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    text = bytearray(b"commit refs/heads/fixture\ncommitter Fixture <fixture@example.invalid> 1000000000 +0000\ndata 7\nfixture\n")
    if parent:
        text.extend(b"from " + parent.encode() + b"\n")
    for name in sorted(names):
        content = (repo / name).read_bytes()
        text.extend(f"M 100644 inline {name}\ndata {len(content)}\n".encode() + content + b"\n")
    text.extend(b"\ndone\n")
    subprocess.run(["git", "-C", str(repo), "fast-import", "--done", "--quiet"], input=text, check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "symbolic-ref", "HEAD", "refs/heads/fixture"], check=True)
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def fixture_seal(store, project, cfg, payload, kind, capsys, parents=()):
    command = ["--root", str(store), "--source-root", str(project), "--config", str(cfg), "--kind", kind,
               "--experiment", "unit-fixture-only", "--trial", load(cfg)["trial_id"], "--input", f"assets={payload}"]
    for parent in parents:
        command += ["--parent", parent["run_id"] + "=" + parent["manifest_sha256"]]
    worker = """
import json,os,shutil,sys
from pathlib import Path
source,out = map(Path,sys.argv[1:3])
shutil.copytree(source,out,dirs_exist_ok=True)
meta = out/'prediction.json'
if meta.is_file():
    obj=json.loads(meta.read_bytes())
    obj['producer_run_id']=os.environ['EL_RUN_ID']
    meta.chmod(0o644)
    meta.write_text(json.dumps(obj))
"""
    code = recorder.main([*command, "--", "{python}", "-c", worker, "{input:assets}", "{out}"])
    capture = capsys.readouterr()
    result = json.loads(capture.out)
    assert code == 0, (result.get("failure_reason"), (Path(result["path"]) / "stderr.log").read_text())
    return fl.run_pin(result)


def prepare_recorded_development(store, project, repo, catalog, cfg, starter, raw, row, digest, capsys):
    """Prepare the audit family's actual synthetic P2 calculation with recorder-assigned source run IDs."""
    ident = {k:row[k] for k in ("scene_id", "scene_family_id", "trial_id", "split")}
    save(cfg, dict(ident, condition_id="B0", dataset_hash=digest, dataset_id="unit-only"))
    pngcam = raw / "pngcam"
    pngcam.mkdir()
    for name in ("input.png","cameras.json"):
        shutil.copyfile(raw / name, pngcam / name)
    src = fixture_seal(store, project, cfg, pngcam, "scene_render", capsys)
    gen = fixture_seal(store, project, cfg, starter, "generation", capsys, [src])
    render = fixture_seal(store, project, cfg, raw / "prediction", "render", capsys, [src,gen])
    def record(kind, parents, inputs, worker):
        args = ["--root",str(store),"--source-root",str(repo),"--config",str(cfg),"--kind",kind,
                "--experiment","unit-fixture-only","--trial",row["trial_id"]]
        for p in parents:
            args += ["--parent",p["run_id"]+"="+p["manifest_sha256"]]
        for role, path in inputs.items():
            args += ["--input",role+"="+str(path)]
        code = recorder.main([*args,"--","{python}","-m",*worker])
        captured = capsys.readouterr()
        assert code == 0, captured
        return fl.run_pin(json.loads(captured.out))
    def payload(p):
        return store / "runs" / p["run_id"] / "payload"
    roles = {"rgb":payload(src)/"input.png","cameras":payload(src)/"cameras.json","predictions":payload(render)}
    common = ["--input","{input:rgb}","--cameras","{input:cameras}","--predictions","{input:predictions}","--output","{out}","--run-id","{run_id}"]
    s = record("model_score", [src,render], roles, ["evidence_lens.risk_model",*common])
    e = record("evaluation", [src,render,s], dict(roles, scores=payload(s), reference=raw/"private_reference"),
               ["evidence_lens.evaluation",*common,"--scores","{input:scores}","--references","{input:reference}",
                "--scene-family-id", row["scene_family_id"], "--trial-id",row["trial_id"], "--scene-id",row["scene_id"], "--condition-id","B0"])
    tr = store / "trials" / row["trial_id"]
    tr.mkdir(parents=True)
    selection = tr / "trial_selection.json"
    save(selection, dict(ident, schema="el-p2-trial-selection-1", condition_id="B0", selection_status="selected",
                        selected={"reference":src,"generation":gen,"render":render,"score":s,"evaluation":e}))
    family = load(raw / "family-pin.json")
    family["selection_sha256"] = sha(selection)
    for role, pin in (("score",s),("evaluation",e)):
        family[role+"_run_id"] = pin["run_id"]
        family[role+"_manifest_sha256"] = pin["manifest_sha256"]
        family[role+"_receipt_sha256"] = sha(payload(pin)/(role+".json"))
    file = raw / "recorded-family-pin.json"
    save(file, family)
    output = raw / "recorded-extract"
    prepare(file, payload(s), payload(e), output, catalog)
    return output


def build_recorded_learning_fixture(tmp_path, capsys, monkeypatch, *, full_protocol=False):
    repo, project, store = tmp_path / "source", tmp_path / "fixture-producer", tmp_path / "fixture-store"
    repo.mkdir()
    store.mkdir()
    (project / "scripts").mkdir(parents=True)
    (project / "scripts/stub.py").write_text("# Synthetic fixture only\n")
    shutil.copytree(REPO / "src/evidence_lens", repo / "src/evidence_lens", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "scripts", repo / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    (repo / "protocol").mkdir()
    # The compact comparator here tests publication and digest binding. Scientific metrics have separate fixtures.
    comparison_names = tuple(formal_inputs.PROTOCOL_FILES) if full_protocol else ("protocol/study-v1.json", "src/evidence_lens/formal_inputs.py")
    monkeypatch.setattr(formal_inputs, "PROTOCOL_FILES", comparison_names)
    monkeypatch.setattr(fl, "ROOT", repo)
    monkeypatch.setattr(formal_lineage, "__file__", str(repo / "scripts/formal_lineage.py"))
    study = repo / "protocol/study-v1.json"
    save(study, {"schema": "el-study-preregistration-1", "kind": "synthetic_fixture",
                 "numerical_protocol_version":"1.0", "labels":{"version":"el-labels-1"},
                 "test_methods":["constant","1-alpha","L0","L1-full","L1-geometry"],
                 "timing_repeats":{"family":"lexicographically first test family", "repetitions":3, "conditions":["B0","R1","Q1"]}})
    if full_protocol:
        add = repo / "protocol/analysis-addendum-v1.json"
        save(add, {"schema":"el-preregistered-analysis-clarifications-1", "study_sha256":sha(study)})
        save(repo / "protocol/operating-points-v1.json", {"schema":"el-operating-points-clarification-1",
             "parent_study_sha256":sha(study),"parent_addendum_sha256":sha(add)})
    (repo / "configs").mkdir()
    catalog, lock, cfg = repo / "protocol/families-v1.json", repo / "configs/formal-freeze-v1.json", tmp_path / "config.json"
    camera_probe = tmp_path / "fixture-calibration"
    camera_probe.mkdir()
    deployable(camera_probe)
    camera_sha = sha(camera_probe / "cameras.json")
    rows = []
    for i in range(24):
        name = f"unit-family-{i:02}"
        rows.append({"scene_family_id":name, "scene_id":name + "/v001", "scene_version":"v001", "trial_id":name + "-B0",
                     "split":"train" if i < 16 else ("development" if i < 20 else "test"), "cv_fold":i % 4 if i < 16 else None,
                     "example_view":"v01", "trials":{c:name + "-" + c for c in (("B0","R1","Q1") if i >= 20 else ("B0",))}, "camera_sha256":camera_sha, "recipe_sha256":"1"*64})
    save(catalog, {"schema":"el-formal-families-1", "dataset_id":"unit-only", "families":rows})
    digest = sha(catalog)
    save(lock, {"catalog_sha256":digest, "preregistration_sha256":sha(study)})
    commit = fixture_commit(repo, set(comparison_names) | set(fl.METHOD_SOURCES) | {"protocol/families-v1.json", "configs/formal-freeze-v1.json"})
    starter = tmp_path / "seed"
    starter.mkdir()
    (starter / "synthetic.txt").write_text("Unit fixture")
    save(cfg, {"trial_id":"unit-parent"})
    parent = fixture_seal(store, project, cfg, starter, "generation", capsys)
    receipts = store / "formal" / digest / "extracts"
    receipts.mkdir(parents=True)
    for i, row in enumerate(rows[:20]):
        with tempfile.TemporaryDirectory(dir=tmp_path) as work:
            raw = Path(work)
            dest = raw / "extract"
            make_extract(raw, dest, catalog, row, i)
            if i == 16:
                dest = prepare_recorded_development(store, project, repo, catalog, cfg, starter, raw, row, digest, capsys)
            meta = load(dest / "family.json")
            save(cfg, {"trial_id":row["trial_id"], "split":row["split"], "dataset_hash":digest})
            ref = fixture_seal(store, project, cfg, dest, "dataset_build", capsys, [parent])
            save(receipts / (row["scene_family_id"] + ".json"), {**ref, "family_pin":meta["pin"], "extract_manifest_sha256":sha(dest / "family.json")})
    python = Path(sys.executable)
    inspection = pilot_masks(SimpleNamespace(store=store, families=catalog, freeze=lock, family=rows[16]["scene_family_id"],
                                             output=tmp_path / "pilot-masks"))
    assert inspection["V_minus_E"] == sum(inspection["V_minus_E_raw_L0_" + k] for k in ("low","high","unknown")) == 6
    fitted = fl.record_learning(store, catalog, lock, "train", python)
    assert fl.record_learning(store, catalog, lock, "train", python) == fitted
    with pytest.raises(ValueError, match="explicitly named selected fit"):
        fl.record_learning(store, catalog, lock, "dev", python, fit_run=parent["run_id"])
    cpu = REPO / ".venv-cpu/bin/python"
    cpu = cpu if cpu.is_file() else python
    frozen = fl.record_learning(store, catalog, lock, "dev", cpu, fit_run=fitted["run_id"])
    method = fl.freeze_method(store, catalog, lock, study, frozen["run_id"], commit)
    method_file = Path(method["method_freeze"])
    assert fl.freeze_method(store, catalog, lock, study, frozen["run_id"], commit) == method
    audit = record_audit(SimpleNamespace(store=store, families=catalog, freeze=lock, study=study, method_freeze=method_file,
                                        threshold_run=frozen["run_id"], cpu_python=cpu, retry_score_of=None, retry_audit_of=None))
    audited = load(store / "runs" / audit["run_id"] / "payload/development_parity.json")
    assert audited["status"] == "passed" and audited["scene_family_id"] == rows[16]["scene_family_id"] and audited["family_V_minus_E"] == 6
    f = rows[-1]
    identity = {"trial_id":f["trial_id"], "scene_id":f["scene_id"], "scene_family_id":f["scene_family_id"],
                "dataset_hash":digest, "dataset_id":"unit-only", "condition_id":"B0", "split":"test"}
    save(cfg, identity)
    source = tmp_path / "unseen-synthetic"
    source.mkdir()
    pred = deployable(source)
    pngcam = tmp_path / "only-source-and-camera"
    pngcam.mkdir()
    for name in ("input.png", "cameras.json"):
        shutil.copyfile(source / name, pngcam / name)
    src = fixture_seal(store, project, cfg, pngcam, "scene_render", capsys)
    gen = fixture_seal(store, project, cfg, starter, "generation", capsys, [src])
    rend = fixture_seal(store, project, cfg, pred, "render", capsys, [src, gen])
    tr = store / "trials" / f["trial_id"]
    tr.mkdir(parents=True)
    save(tr / "trial_selection.json", {"schema":"el-p2-trial-selection-1", "selection_status":"selected", "trial_id":f["trial_id"],
         "scene_family_id":f["scene_family_id"], "scene_id":f["scene_id"], "condition_id":"B0", "selected":{"reference":src,"generation":gen,"render":rend}})
    scored = fl.record_score(store, catalog, lock, study, method_file, f["scene_family_id"], "B0", cpu)
    receipt = load(store / "runs" / scored["run_id"] / "payload/l1_score.json")
    assert receipt["model_id"] == frozen["run_id"] and receipt["producer_run_id"] == rend["run_id"]
    assert set(receipt["inputs"]) == {"input_png","cameras","prediction_metadata","prediction/source", *("prediction/v%02d" % i for i in range(1, 7))}
    assert len(load(store / "runs" / fitted["run_id"] / "inputs.json")["parents"]) == 16
    assert len(load(store / "runs" / frozen["run_id"] / "inputs.json")["parents"]) == 5
    for condition in ("R1", "Q1"):
        trial = f["trials"][condition]
        save(cfg, dict(identity, trial_id=trial, condition_id=condition))
        folder = tmp_path / condition
        folder.mkdir()
        if condition == "R1":
            source_assets = folder / "source"
            source_assets.mkdir()
            cameras = load(pngcam / "cameras.json")
            cameras["views"]["source"].update(width=2, height=1, K=[[2,0,.5],[0,2,0],[0,0,1]])
            save(source_assets / "cameras.json", cameras)
            Image.new("RGB", (2, 1), (35, 72, 1)).save(source_assets / "input.png")
            prepared = fixture_seal(store, project, cfg, source_assets, "derivation", capsys, [src])
            producer = fixture_seal(store, project, cfg, starter, "generation", capsys, [prepared])
            bound_source = prepared
        else:
            source_assets = pngcam
            prepared = fixture_seal(store, project, cfg, starter, "derivation", capsys, [gen])
            producer, bound_source = prepared, src
        prediction = folder / "prediction"
        shutil.copytree(pred, prediction)
        meta = load(prediction / "prediction.json")
        meta.update(input_sha256=sha(source_assets / "input.png"), camera_sha256=sha(source_assets / "cameras.json"))
        save(prediction / "prediction.json", meta)
        if condition == "R1":
            with np.load(prediction / "source.npz") as a:
                reduced = {k:a[k][:1,:2] for k in a.files}
            np.savez_compressed(prediction / "source.npz", **reduced)
        rendered = fixture_seal(store, project, cfg, prediction, "render", capsys, [bound_source, producer])
        target = store / "trials" / trial
        target.mkdir()
        source_pins = {key: dict(bound_source, path="payload/" + name, sha256=sha(source_assets / name))
                       for key, name in (("input_png","input.png"),("cameras","cameras.json"))}
        save(target / "trial_selection.json", {"schema":"el-condition-selection-1", "selection_status":"selected", "trial_id":trial,
             "scene_family_id":f["scene_family_id"], "scene_id":f["scene_id"], "condition_id":condition,
             "dataset_hash":digest, "method_freeze_sha256":sha(method_file), "b0_selection_sha256":sha(tr / "trial_selection.json"),
             "sources":source_pins, "selected":{"reference":src,"render":rendered,"preparation":prepared,"producer":producer}})
        pinned = fl.record_score(store, catalog, lock, study, method_file, f["scene_family_id"], condition, cpu)
        result = load(store / "runs" / pinned["run_id"] / "payload/l1_score.json")
        assert result["producer_run_id"] == rendered["run_id"] and result["inputs"]["input_png"]["sha256"] == source_pins["input_png"]["sha256"]
    return dict(store=store, repo=repo, catalog=catalog, study=study, freeze=lock, method=method_file,
                fit=fl.run_pin(fitted), threshold=fl.run_pin(frozen), commit=commit)


def test_recorded_fit_threshold_freeze_and_score(tmp_path, capsys, monkeypatch):
    build_recorded_learning_fixture(tmp_path, capsys, monkeypatch)
