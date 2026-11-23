"""Disposable sealed integration of the public aggregates, with explicitly synthetic science and source files."""

import csv
import json

import numpy as np
import pytest
from evidence_lens.formal_inputs import signature
from evidence_lens.scoring_io import TARGETS, file_hash, write_json
from test_vision_data import base_scores, fixture_truth, seal_fixture
from vision_confirmation import COHORTS, CONDITIONS
from vision_evaluate import (
    METHODS,
    SCORE_SETS,
    aggregate_method,
    compact_curve,
    groups,
    macro_families,
    paired_report,
    system_counts,
)
import vision_public_study as public


def test_opening_to_pinned_public_report_no_private_results(tmp_path, monkeypatch):
    store = tmp_path / "store"
    shelf = tmp_path / "cold"
    serial = iter(range(1000))

    def rid():
        return "20000101T000000000000Z-" + f"{next(serial):016x}"

    def seal(kind, files, meta=None, id_=None):
        return seal_fixture(store, id_ or rid(), kind, "succeeded", meta or {}, files)

    def asset(pin, rel):
        return store / "runs" / pin["run_id"] / "payload" / rel

    def doc(path, name, body):
        dest = path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_json(dest, body)
        return dest

    def catalog(name, count):
        names = [name + str(i) for i in range(count)]
        cat = {
            "schema": "el-vision-family-roster-1",
            "families": [
                {"scene_family_id": f, "split": "test", "trials": {c: f + "-" + c for c in CONDITIONS}}
                for f in names
            ],
        }
        return doc(tmp_path, name + ".json", cat), names

    new, newnames = catalog("syntheticnew", 8)
    old, oldnames = catalog("syntheticpast", 4)
    monkeypatch.setattr(public, "ORIGINAL_CATALOG_SHA", file_hash(old))
    from vision_gate_fixture import gate_fixture
    from stage2_opening import open_cohort

    fixture = gate_fixture(tmp_path, names=newnames)
    new, shelf = fixture["catalog"], fixture["shelf"]
    methodpin, method, mh = fixture["freeze"]()
    models, values = method["models"], fixture["values"]
    opened = open_cohort(
        store, new, fixture["catalog_sha"], methodpin["run_id"], methodpin["manifest_sha256"]
    )
    assert opened["method_sha256"] == mh
    marker = store / "protocol/stage2" / fixture["catalog_sha"] / "test-opening.json"
    spec = {
        "catalog_sha256": file_hash(new),
        "method": methodpin,
        "method_sha256": mh,
        "cohorts": {},
        "engineering": [],
    }
    t = fixture_truth()
    scores = base_scores()
    raw = system_counts(t, scores, values, np.ones((2, 4), bool))
    measure = {m: aggregate_method([raw[m]] * 6) for m in METHODS}
    curves = {
        m: compact_curve([groups(scores[m].ravel(), (t["rgb_high_error"].ravel() == 1))] * 6, 48, values[m])
        for m in METHODS
    }
    for cohort, names in zip(COHORTS, (newnames, oldnames)):
        materials = []
        families = []
        status = []
        case = []
        files = {}
        scope = tmp_path / cohort
        ds = seal("dataset_build", {"payload/dataset.json": {"synthetic": True, "cohort": cohort}})
        ds_sha = file_hash(asset(ds, "dataset.json"))
        for f in names:
            for c in CONDITIONS:
                sel = doc(
                    store / "trials" / (f + "-" + c),
                    "trial_selection.json",
                    {"selection_status": "selected", "selected": {}, "attempts": []},
                )
                materials.append(
                    {
                        "family": f,
                        "condition": c,
                        "trial_id": f + "-" + c,
                        "selection_sha256": file_hash(sel),
                        "parents": {},
                    }
                )
                families.append({"family": f, "condition": c, "group": None, "methods": measure})
                for v in TARGETS:
                    status.append(
                        {
                            "family": f,
                            "condition": c,
                            "view": v,
                            "V": 8,
                            "E": 8,
                            "render": "completed",
                            "scientific_disposition": "selected",
                            "methods": {
                                m: {"status": "scored", "reason": "private/path never exported"}
                                for m in METHODS
                            },
                        }
                    )
                case.append({"family": f, "condition": c, "view": "v01"})
                n = f + "/" + c + "/curves.json"
                files[n] = doc(scope, n, curves)
        macro = [
            {
                "condition": c,
                "cohort": "all",
                "methods": {m: macro_families([measure[m]] * len(names)) for m in METHODS},
            }
            for c in CONDITIONS
        ]
        for n, o in (
            ("families", families),
            ("status", status),
            ("macro", macro),
            ("paired", paired_report(families)),
        ):
            files[n + ".json"] = doc(scope, n + ".json", o)
        mat = seal(
            "dataset_build",
            {"payload/materials.json": {"test_unlock_sha256": file_hash(marker), "cells": materials}},
        )
        sin = {}
        ss = {}
        for tag in SCORE_SETS:
            m = "L1-full" if tag == "original-five" else tag
            sr = seal("model_score", {"payload/scores.json": {"method_freeze_sha256": mh}})
            ss[tag] = sr
            sin[tag] = {
                "model_sha256": models[m]["model_sha256"],
                "sha256": file_hash(asset(sr, "scores.json")),
            }
        out = {
            "schema": "el-vision-comparison-1"
            if cohort == COHORTS[0]
            else "el-vision-historical-regression-1",
            "thresholds": values,
            "test_opening_sha256": file_hash(marker),
            "methods": list(METHODS),
            "dataset_sha256": ds_sha,
            "files": {n: signature(p) for n, p in files.items()},
            "inputs": sin,
        }
        ev = seal(
            "evaluation", {"payload/evaluation.json": out, **{"payload/" + n: p for n, p in files.items()}}
        )
        response = doc(
            scope,
            "response.json",
            {"cohort": cohort, "method_sha256": mh, "dataset_sha256": ds_sha, "families": []},
        )
        boards = seal(
            "report",
            {
                "payload/boards.json": {
                    "cohort": cohort,
                    "count": len(case),
                    "cases": case,
                    "method_sha256": mh,
                    "dataset_sha256": ds_sha,
                    "paired_response": signature(response),
                },
                "payload/paired-response.json": response,
            },
        )
        spec["cohorts"][cohort] = {
            "materials": mat,
            "dataset": ds,
            "scores": ss,
            "boards": boards,
            "evaluation": ev,
        }
    out = tmp_path / "public"
    public.publish_report((store, shelf), spec, new, old, out, "synthetic-local-report")
    pub = out / "public-numerical"
    manifest = json.loads((pub / "manifest.json").read_text())
    assert manifest["method_sha256"] == mh and all(
        signature(pub / n) == sig for n, sig in manifest["files"].items()
    )
    table = list(csv.DictReader((pub / "family-metrics.csv").open()))
    assert len(table) == (8 + 4) * 3 * 7 and all(int(r["V"]) == 48 and int(r["E"]) == 48 for r in table)
    assert set(manifest["files"]) == {
        "study-summary.json",
        "selected-method.json",
        "family-metrics.csv",
        "paired-changes.csv",
        "execution-costs.json",
    }
    assert "Historical four-family regression" in (out / "study.md").read_text()
    assert not any("private/path" in p.read_text() or str(tmp_path) in p.read_text() for p in pub.iterdir())
    spec["method_sha256"] = "3" * 64
    with pytest.raises(ValueError, match="another method"):
        public.publish_report((store, shelf), spec, new, old, tmp_path / "bad", "synthetic-local-bad")
