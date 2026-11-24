"""Offline evaluator rehearsal on one authored legacy case, including a deliberately absent prediction."""

import argparse
import csv
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from dtu_eval import read_pinned
from dtu_score import METHODS, VIEWS
from t0_evaluate import evaluate


def run(history, cfg, out):
    pins = cfg["sources"]
    _, cp = read_pinned((history,), pins["case"])
    case = load(cp)
    _, sp = read_pinned((history,), pins["scores"])
    scores = load(sp)
    require(
        scores["case_sha256"] == sha(cp) and case["condition"] == "M4", "wrong authored rehearsal materials"
    )
    example = "authored-offset-bell-cage-fixture"
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        refdir = root / "references"
        refdir.mkdir()
        evaluated = root / "numerical"
        evaluated.mkdir()
        roles = {}
        database = {}
        items = {v: {"name": "surrogate-" + v} for v in VIEWS}

        def document(role, path, value):
            save(path, value)
            sel = {"fixture": role, "sha256": sha(path)}
            roles[role] = sel
            database[role] = path
            return sel

        r = document(
            "roster", root / "roster.json", {"objects": [{"object_id": example, "target_members": items}]}
        )
        f = document(
            "freeze",
            root / "freeze.json",
            {
                "object_ids": [example],
                "families": {example: example},
                "expected_scores": 48,
                "sources": {"roster": r},
                "models": scores["models"],
                "protocol": {"camera": {"targets": {v: i for i, v in enumerate(VIEWS)}}},
            },
        )
        opening = document(
            "opening",
            root / "opening.json",
            {
                "schema": "el-t0-reference-opening-1",
                "permitted_reader": "independent_evaluator",
                "freeze": f,
                "object_ids": [example],
                "scheduled_entries": 48,
                "scored_entries": 24,
                "conditions": ["M2", "M4"],
                "terminal_cells": [
                    {
                        "object_id": example,
                        "condition": "M2",
                        "status": "failed",
                        "reason": "intentional fixture absence",
                    },
                    {
                        "object_id": example,
                        "condition": "M4",
                        "status": "scored",
                        "case": pins["case"],
                        "scores": pins["scores"],
                    },
                ],
            },
        )
        views = {}
        source = cp.parent / "sources/s00.png"
        for i, v in enumerate(VIEWS):
            path = refdir / (v + ".png")
            path.write_bytes(source.read_bytes())
            views[v] = {
                "official_view_id": i,
                "archive_member": items[v]["name"],
                "original_sha256": sha(path),
                "rgb": {"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)},
            }
        refs = document(
            "reference",
            refdir / "index.json",
            {"schema": "el-t0-dtu-references-1", "object_id": example, "opening": opening, "views": views},
        )

        def original(roots, pin):
            if "fixture" not in pin:
                return read_pinned(roots, pin)
            name = pin["fixture"]
            require(
                pin == roles[name] and sha(database[name]) == pin["sha256"],
                "changed isolated fixture document",
            )
            return None, database[name]

        # Real-data authority never accepts these fixture-only keys. Nothing is written into the official roster or open registry.
        with patch("t0_evaluate.read_pinned", original):
            evaluate((history,), opening, {example: refs}, evaluated)
        with (evaluated / "public/method-views.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        failed = [r for r in rows if r["condition"] == "M2"]
        passed = [r for r in rows if r["condition"] == "M4"]
        require(
            len(failed) == len(passed) == 24
            and all(r["E"] == "0" and r["V"] == "65536" and r["common_AP"] == "" for r in failed),
            "failure accounting differs",
        )
        require(all(r["V"] == r["E"] == "65536" for r in passed), "authored pixel denominators differ")
        require(len(list((evaluated / "private").glob("*.png"))) == 12, "missing fixture boards")
        unit = load(evaluated / "public/evaluation.json")
        require(
            unit["scored_entries"] == 24 and unit["scheduled_entries"] == 48, "fixture denominator differs"
        )
    save(
        out / "fixture.json",
        {
            "schema": "el-t0-evaluation-engineering-fixture-1",
            "run_id": os.environ["EL_RUN_ID"],
            "physical_T0_objects": 0,
            "target_reference_files_opened": 0,
            "evaluation_code_sha256": sha(Path(__file__).with_name("t0_evaluate.py")),
            "authored_source_used_as_surrogate_for_shape": True,
            "complete_maps_tested": 24,
            "deliberately_absent_maps_tested": 24,
            "panels_rendered": 12,
            "scientific_metrics_exported": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("history", "config", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    run(a.history, load(a.config), a.out)
