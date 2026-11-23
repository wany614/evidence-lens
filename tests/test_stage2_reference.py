"""Pre-acquisition registrations, split controls and selected reference settings."""

import importlib
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

ASSETS = Path(__file__).resolve().parents[1]


def paths(monkeypatch):
    # Keep this test working both in the development overlay and after promotion to the project.
    project=next(d for d in (Path.cwd(),*Path.cwd().parents) if (d/"configs/reference.json").is_file())
    monkeypatch.syspath_prepend(str(ASSETS/"scripts"))
    return project,importlib.import_module("stage2_reference_catalog")


def test_roster_and_independent_split_signatures(monkeypatch):
    project,engine=paths(monkeypatch)
    catalog,study,lock=engine.validation(ASSETS,project)
    assert lock["catalog_sha256"]==engine.digest(ASSETS/engine.CATALOG)
    assert Counter(r["split"] for r in catalog["families"])=={"train":12,"development":4,"test":8}
    assert len(set(study["novelty_audit"]["historical_foreground_pair_fingerprints"].values()))==24
    assert {k for r in catalog["families"] if r["ordinal"]<45 for k in r["reserved_geometry"]}==set()
    assert [r["reserved_geometry"] for r in catalog["families"][-4:]]==[[k] for k in engine.NOVEL]
    assert len({r["pair_structure_sha256"] for r in catalog["families"]})==24
    for r in catalog["families"]:
        assert r["example_view"]=="v01" and r["camera_sha256"]==engine.digest(project/engine.CAMERA)


def test_test_opening_blocks_before_any_selected_recipe_read(monkeypatch,tmp_path):
    project,engine=paths(monkeypatch)
    catalog,_,lock=engine.validation(ASSETS,project)
    def deny(*args):
        raise ValueError("test-opening receipt absent")
    def unexpected(*args):
        raise AssertionError("opened data before checking the gate")
    monkeypatch.setitem(sys.modules,"stage2_opening",SimpleNamespace(verify_opening=deny))
    monkeypatch.setattr(engine,"read",unexpected)
    for r in catalog["families"][-8:]:
        with pytest.raises(ValueError,match="test-opening receipt absent"):
            engine.resolve_selected(project,ASSETS,r,catalog,lock,tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_selected_pilots_preserve_native_scene_protocol(monkeypatch,tmp_path):
    project,engine=paths(monkeypatch)
    cat,_,lock=engine.validation(ASSETS,project)
    for r in cat["families"][:4]:
        cfg=engine.resolve_selected(project,ASSETS,r,cat,lock,tmp_path)
        ref=cfg["reference"]
        assert (ref["width"],ref["height"],ref["scene_name"])==(512,512,r["scene_family_id"]+"_v001")
        assert set(ref["camera"]["positions"])=={"source",*engine.TARGETS}
        assert ref["validation"]["occlusion_profile"]=="opaque_mesh_pairs"
        assert cfg["dataset_hash"]==lock["catalog_sha256"] and cfg["formal"]["recipe_sha256"]==r["recipe_sha256"]
        assert len(ref["objects"])==r["object_count"]+2
        assert len({obj["name"] for obj in ref["objects"]})==len(ref["objects"])
