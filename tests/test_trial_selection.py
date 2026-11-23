"""Exercise the P2 selector only on sealed six-pixel unit fixtures."""

import runpy
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from evidence_lens import archive, recorder
from evidence_lens._bundle import BundleError, load, new_id, save, sha
from evidence_lens.evaluation import run_evaluation
from evidence_lens.risk_model import run_score
from evidence_lens.scoring_io import VIEWS
from test_cpu_p2 import deployable, reference

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
selector = runpy.run_path(str(SCRIPTS / "select_trial.py"))
recompute = runpy.run_path(str(SCRIPTS / "recompute_view.py"))["check"]


def build_frozen_fixture(tmp_path, monkeypatch, capsys):
    """Build the isolated six-pixel chain and return its store and selection file."""
    store, project = tmp_path / "store", tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "scripts/stub.py").write_text("# synthetic selector fixture only\n")
    tri = "selector-unit-fixture"
    scene = "synthetic-fixture-only"
    cfg = tmp_path / "config.json"
    identity = {"trial_id": tri, "scene_id": scene, "scene_family_id": scene, "condition_id": "B0"}
    save(cfg, identity)
    ids = {k: new_id() for k in selector["KINDS"]}
    rgb, cameras = tmp_path / "input.png", tmp_path / "cameras.json"
    render, src = deployable(tmp_path), reference(tmp_path)
    with np.load(render / "v01.npz") as data:
        unavailable = {k: data[k].copy() for k in data}
    unavailable["rgb"].fill(np.nan)
    unavailable["z"].fill(np.nan)
    unavailable["alpha"].fill(0)
    np.savez_compressed(render / "v01.npz", **unavailable)
    shutil.copyfile(rgb, src / "input.png")
    shutil.copyfile(cameras, src / "cameras.json")
    blend = src / "test.blend"
    blend.write_bytes(b"synthetic scene fixture only")
    ref = load(src / "reference.json")
    ref.update(scene_path=blend.name, blend_sha256=sha(blend))
    ref["views"]["source"] = {}
    save(src / "reference.json", ref)
    weights = tmp_path / "fixture.bin"
    weights.write_bytes(b"synthetic weights")
    generation, rb = tmp_path / "generation", tmp_path / "readback"
    generation.mkdir()
    rb.mkdir()
    (generation / "scene.ply").write_bytes(b"synthetic PLY fixture")
    gen = {"schema":"el-generation-1", "producer_run_id":ids["generation"], "gaussian_count":1,
           "checkpoint_sha256":sha(weights), "ply_sha256":sha(generation / "scene.ply"),
           "input_sha256":sha(rgb), "camera_sha256":sha(cameras)}
    save(generation / "generation.json", gen)
    save(rb / "reference_readback.json", {"schema":"el-reference-readback-1", "scientific_acceptance":True,
         "integrity_checked":True, "failed_scientific_checks":[], "views":dict.fromkeys(VIEWS, {}),
         "reference_json_sha256":sha(src / "reference.json")})
    probe = {"off_axis_plane_passed":True, "alpha_normalization_passed":True, "translation_passed":True,
             "linear_blend_passed":True, "empty_pixel_alpha":0.0}
    save(render / "depth-probe.json", probe)
    pred = load(render / "prediction.json")
    pred.update(producer_run_id=ids["render"], generation_run_id=ids["generation"], ply_sha256=gen["ply_sha256"],
                depth_validation=probe, depth_validation_receipt_sha256=sha(render / "depth-probe.json"), views={},
                source_rgb_check={"finite_pixels":6, "pixels":6, "mae":0.})
    for view in VIEWS:
        npz = render / (view + ".npz")
        with np.load(npz) as values:
            Image.fromarray(np.rint(255 * np.nan_to_num(values["rgb"])).astype("uint8")).save(render / (view + ".png"))
        pred["views"][view] = {"sha256":sha(npz), "bytes":npz.stat().st_size}
    save(render / "prediction.json", pred)
    scores, evaluation = tmp_path / "scores", tmp_path / "evaluation"
    run_score(rgb, cameras, render, scores, ids["score"])
    run_evaluation(rgb, cameras, src, render, scores, evaluation, dict(identity, run_id=ids["evaluation"]), parquet=True)
    independent = tmp_path / "independent"
    independent.mkdir()
    proof = recompute(SimpleNamespace(cameras=cameras, metadata=render / "prediction.json",
                      source_prediction=render / "source.npz", prediction=render / "v01.npz",
                      reference=src / "reference/v01.npz", score=scores / "v01.npz", labels=evaluation / "labels/v01.npz",
                      metrics=evaluation / "metrics.csv", view="v01"))

    def seal(role, output, parent_roles=(), inputs=(), extra=(), rid=None, fail=False):
        rid = rid or ids[role]
        with monkeypatch.context() as patch:
            patch.setattr(recorder, "new_id", lambda: rid)
            args = ["--root", str(store), "--source-root", str(project), "--config", str(cfg),
                    "--kind", selector["KINDS"][role], "--experiment", "unit", "--trial", tri]
            for parent in parent_roles:
                args.extend(["--parent", ids.get(parent, parent)])
            for idx, (parent, name) in enumerate(inputs):
                path = store / "runs" / ids.get(parent, parent) / "payload" / name
                args.extend(["--input", f"selected{idx}={path}"])
            copy = "import shutil,sys;shutil.copytree(sys.argv[1],sys.argv[2],dirs_exist_ok=True)"
            cmd = ["{python}", "-c", "raise SystemExit(7)"] if fail else ["{python}", "-c", copy, str(output), "{out}"]
            code = recorder.main([*args, *extra, "--", *cmd])
            assert code == (7 if fail else 0), capsys.readouterr()
        return store / "runs" / rid

    seal("reference", src)
    seal("readback", rb, ["reference"], [("reference", "reference.json")])
    failed = seal("generation", generation, ["reference"], rid=new_id(), fail=True)
    ref_inputs = [("reference","input.png"), ("reference","cameras.json")]
    early_id = new_id()
    early = tmp_path / "earlier-generation"
    shutil.copytree(generation, early)
    save(early / "generation.json", dict(gen, producer_run_id=early_id))
    early_run = seal("generation", early, ["reference"], ref_inputs, ["--shared-input", f"weights={weights}"], rid=early_id)
    archive.receive(early_run, store)
    gen_run = seal("generation", generation, ["reference"], ref_inputs, ["--shared-input", f"weights={weights}"])
    archive.receive(gen_run, store)
    rendered = seal("render", render, ["reference","generation"],
                    [*ref_inputs, ("generation","generation.json"), ("generation","scene.ply")])
    archive.receive(rendered, store)
    score_parents = ["reference","readback","render"]
    seal("score", scores, score_parents, [*ref_inputs, ("render",".")])
    eval_parents = [*score_parents,"score"]
    eval_inputs = [("reference","."), ("render","."), ("score",".")]
    seal("evaluation", evaluation, eval_parents, eval_inputs)
    check_parents = [*eval_parents,"evaluation"]
    check_inputs = [("reference","cameras.json"), ("reference","reference/v01.npz"), ("render","prediction.json"),
                    ("render","source.npz"), ("render","v01.npz"), ("score","v01.npz"),
                    ("evaluation","labels/v01.npz"), ("evaluation","metrics.csv")]
    # The operator verifies a later evaluation first. Neither the earlier
    # generation nor the earlier scientific evaluation can be silently bypassed.
    alt_eid, alt_check = new_id(), new_id()
    other_eval = tmp_path / "later-evaluation"
    run_evaluation(rgb, cameras, src, render, scores, other_eval, dict(identity, run_id=alt_eid), parquet=True)
    seal("evaluation", other_eval, eval_parents, eval_inputs, rid=alt_eid)
    alternate = dict(ids, evaluation=alt_eid, recompute=alt_check)
    save(independent / selector["CHECK_FILE"], proof)
    alt_inputs = [(alt_eid if p == "evaluation" else p, n) for p,n in check_inputs]
    seal("recompute", independent, [*eval_parents, alt_eid], alt_inputs, rid=alt_check)
    freeze = selector["freeze"]
    with pytest.raises(BundleError, match=f"earlier unresolved generation {early_id}"):
        freeze(store, tri, alternate)
    failed_render = seal("render", render, ["reference", early_id], rid=new_id(), fail=True)
    with pytest.raises(BundleError, match=f"earlier unresolved evaluation {ids['evaluation']}"):
        freeze(store, tri, alternate)
    save(independent / selector["CHECK_FILE"], dict(proof, passed=False))
    bad = seal("recompute", independent, check_parents, check_inputs, rid=new_id())
    save(independent / selector["CHECK_FILE"], proof)
    seal("recompute", independent, check_parents, check_inputs)
    with pytest.raises(BundleError, match="earlier verified chain"):
        freeze(store, tri, alternate)
    with monkeypatch.context() as patch:
        module = selector["scientific_gates"].__globals__
        reader = module["load"]
        def temporarily_unreadable(path):
            if path.name == selector["CHECK_FILE"] and path.parent.parent.name == bad.name:
                raise OSError("temporary read failure in an earlier outcome")
            return reader(path)
        patch.setitem(module, "load", temporarily_unreadable)
        with pytest.raises(OSError, match="temporary read failure"):
            freeze(store, tri, ids)
    assert not (store / "trials" / tri / "trial_selection.json").exists()
    path, status = freeze(store, tri, ids)
    frozen = load(path)
    assert status == "frozen" and frozen["selected"]["recompute"]["run_id"] == ids["recompute"]
    assert frozen["evidence"]["target_denominators"]["v01"] == {
        "reference_valid": 6, "rgb_evaluable": 0, "depth_evaluable": 0}
    assert {a["run_id"] for a in frozen["attempts"] if a["execution_status"] == "failed"} == {failed.name, failed_render.name}
    assert [a["recompute_run_id"] for a in frozen["earlier_ineligible_chains"]] == [bad.name]
    original = sha(path)
    seal("generation", generation, ["reference"], rid=new_id(), fail=True)
    assert freeze(store, tri, ids) == (path, "identical") and sha(path) == original
    with pytest.raises(BundleError, match="immutable selection conflicts"):
        freeze(store, tri, alternate)
    return store, path


def test_only_first_complete_chain_can_be_frozen(tmp_path, monkeypatch, capsys):
    build_frozen_fixture(tmp_path, monkeypatch, capsys)
