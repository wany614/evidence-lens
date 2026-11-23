"""Tiny deterministic representation fixtures. No scientific source, unpublished model or held-out scene is opened."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from evidence_lens.l1 import ALL_COLUMNS, NUMERIC
from evidence_lens.multisource import COLUMNS, CONDITIONS, METHODS, SCOPES, SOURCES, TARGETS, schema_sha256
from evidence_lens.scoring_io import file_hash, write_json
from stage3_risk_data import pack_view, read_asset
from stage3_risk_evaluate import study
from stage3_risk_learning import fit, freeze
from stage3_risk_sources import CATALOG, STUDY, LEARNING, TRANSFER, asset
from test_stage3_multisource import case_fixture


def synthetic_dataset(root, split):
    root.mkdir()
    catalog = Path.cwd() / "protocol/stage3-families-v1.json"
    if not catalog.exists():
        catalog = Path(__file__).resolve().parents[3] / "reference/protocol/stage3-families-v1.json"
    assert file_hash(catalog) == CATALOG
    shutil.copyfile(catalog, root / "catalog.json")
    names = [
        r["scene_family_id"] for r in json.loads(catalog.read_bytes())["families"] if r["split"] == split
    ]
    cells = []
    views = []
    raw = root / "synthetic-original.npz"
    surface = np.ones((4, 6), bool)
    rgb = np.full((4, 6, 3), 0.25, np.float32)
    rgb[0:2] = 0.43
    real = {
        "rgb": rgb,
        "z": np.ones((4, 6), np.float32) * 2,
        "position": np.ones((4, 6, 3), np.float32),
        "reference_surface": surface,
        "object_id": np.ones((4, 6), np.uint16),
        "edge": np.zeros((4, 6), bool),
        **{"visibility_" + s: np.ones((4, 6), np.uint8) for s in SOURCES},
    }
    np.savez_compressed(raw, **real)
    fixture = {
        "models": {
            "L1-full": {
                "columns": ALL_COLUMNS,
                "coefficients": [0.0] * len(ALL_COLUMNS),
                "intercept": 0.0,
                "medians": [0.0] * len(NUMERIC),
                "means": [0.0] * len(NUMERIC),
                "scales": [1.0] * len(NUMERIC),
            }
        }
    }
    for fi, f in enumerate(names):
        for c in CONDITIONS:
            folder = root / "cases" / f / c
            folder.parent.mkdir(parents=True, exist_ok=True)
            case = case_fixture(folder, c)
            cell = {
                "family": f,
                "condition": c,
                "selection_sha256": "6" * 64,
                "failure": None,
                "selected": {},
                "case": asset(root, folder / "case.json"),
            }
            cells.append(cell)
            for v in TARGETS:
                views.append(
                    pack_view(
                        root,
                        cell,
                        v,
                        case,
                        (raw, asset(root, raw)),
                        case.cameras[v],
                        fixture,
                        split,
                        "synthetic",
                        fi % 4 if split == "train" else None,
                    )
                )
    doc = {
        "schema": "el-stage3-risk-dataset-1",
        "run_id": "private-fixture-only",
        "split": split,
        "cohort": "formal",
        "catalog_sha256": CATALOG,
        "study_sha256": STUDY,
        "learning_sha256": LEARNING,
        "feature_schema_sha256": schema_sha256(),
        "columns": list(COLUMNS),
        "families": names,
        "transfer": {
            "run": TRANSFER,
            "model_index_sha256": "a" * 64,
            "weights_sha256": "b" * 64,
            "threshold": 0.43483029305934906,
        },
        "cells": cells,
        "views": views,
    }
    write_json(root / "dataset.json", doc)
    return doc, file_hash(root / "dataset.json")


def test_synthetic_training_and_four_method_evaluation(tmp_path):
    pytest.importorskip("sklearn")
    train, ds = tmp_path / "train", tmp_path / "dev"
    _, th = synthetic_dataset(train, "train")
    d, dh = synthetic_dataset(ds, "development")
    fr = fit(train, th, tmp_path / "fit", "fixture-only-fit")
    assert fr["status"] == "fit" and fr["C"] == 0.1 and fr["parity_rows"] == 2304
    pre = json.loads((tmp_path / "fit/package/preprocessing.json").read_bytes())
    assert pre["training_dataset_sha256"] == th
    ff = freeze(
        ds, dh, tmp_path / "fit/package", fr["model_index_sha256"], tmp_path / "frozen", "fixture-only-freeze"
    )
    assert ff["threshold"]["coverage"] == 1.0
    roster = {}
    for k in METHODS:
        # Substitute identical controlled score grids for the two neural branches in this protocol-format test.
        source = tmp_path / "frozen" / (k if k in ("B-L1-transfer", "MS-L1") else "MS-L1")
        folder = tmp_path / ("fixture-" + k)
        shutil.copytree(source, folder)
        entry = folder / "scores.json"
        doc = json.loads(entry.read_bytes())
        doc["method"] = k
        if k.startswith("MS-") and k != "MS-L1":
            doc.update(model_id="fixture-only-" + k)
        write_json(entry, doc)
        roster[k] = {
            "directory": str(folder),
            "sha256": file_hash(entry),
            **{n: doc[n] for n in ("model_id", "model_index_sha256", "weights_sha256", "threshold")},
        }
    result = study(ds, dh, "development", roster, tmp_path / "evaluation", boards=False)
    assert result["dispositions"] == {
        "planned_cells": 8,
        "planned_target_views": 48,
        "returned_target_views": 48,
        "scheduled_scorer_modes": 144,
        "nonredundant_scorer_modes": 120,
    }
    fam = json.loads((tmp_path / "evaluation/families.json").read_bytes())
    assert len(fam) == 96 and fam[0]["regions"]["all"]["counts"]["E"] == 144
    assert fam[0]["regions"]["all"]["ranking"]["common"]["ap"] == pytest.approx(0.5)
    assert (
        json.loads((tmp_path / "evaluation/reconstruction.json").read_bytes())["families"][d["families"][0]][
            "M4_minus_M2"
        ]["rgb_mae"]
        == 0
    )
    from stage3_risk_boards import save_case
    from PIL import Image

    first = d["views"][0]
    truth = read_asset(ds, first["truth"])
    meta = {}
    for s in SCOPES:
        meta[s] = {}
        for k in METHODS:
            dr = Path(roster[k]["directory"])
            j = json.loads((dr / "scores.json").read_bytes())
            meta[s][k] = read_asset(dr, j["views"][0]["modes"][s]["file"])["risk"]
    b = save_case(ds, tmp_path / "boards", first, truth, meta, {m: roster[m]["threshold"] for m in METHODS})
    with Image.open(tmp_path / "boards" / b["file"]["path"]) as im:
        assert im.mode == "RGB" and im.width >= 1200
