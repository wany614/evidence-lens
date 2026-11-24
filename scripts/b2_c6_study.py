"""Score the fixed C6 reconstructions once and authorize an independent calibration reader."""

import argparse
import json
import os
from pathlib import Path

from b2_records import bind, selected, stage
from dtu_eval import score_map
from dtu_score import METHODS, RELEASE_SHA, SCOPES, VIEWS
from evidence_lens._bundle import load, save, sha, utc
from evidence_lens._verify import require
from evidence_lens.multisource_io import Case
from t0_records import resolve

PROJECT = Path(__file__).resolve().parents[1]
RELEASE = "20261123T193505072501Z-6046afc1fce4480b"
CONDITIONS = ("M2", "M4")


def verify_reconstruction(roots, obj, cond, cp, rp, cohort, recon):
    r = load(resolve(roots, rp))
    path = resolve(roots, cp)
    require(
        r["schema"] == "el-b2-reconstruction-readback-1"
        and r["verified"] is True
        and r["object_id"] == obj
        and r["condition"] == cond
        and r["case_sha256"] == cp["sha256"]
        and rp["run_id"] == cp["run_id"]
        and r["cohort"] == cohort
        and r["freeze"] == recon
        and r["completed_steps"] == 5000
        and r["optimizer_seed"] == 43171
        and r["target_reference_files_opened"] == 0,
        "unregistered C6 reconstruction",
    )
    case = Case(path.parent, cp["sha256"])
    require(
        case.condition == cond and case.meta["target_ids"] == list(VIEWS),
        "fixed-layout Case target order differs",
    )
    return case, path


def review(roots, cfg, matrix, candidates, dest):
    pins = cfg["sources"]
    cohort = load(resolve(roots, pins["cohort"]))
    release = load(resolve(roots, pins["release"]))
    require(
        sha(resolve(roots, pins["release"])) == RELEASE_SHA,
        "incorrect published risk package",
    )
    method = resolve(roots, pins["calibration_method"])
    candidate_path = resolve(roots, pins["score_candidates"])
    scored = load(candidate_path)
    policy = load(method)
    require(
        sha(candidates) == sha(candidate_path)
        and sha(matrix) == pins["closed_matrix"]["sha256"]
        and policy["schema"] == "el-b2-calibration-method-1"
        and policy["cohort"] == pins["cohort"]
        and scored["schema"] == "el-b2-score-candidates-1"
        and scored["matrix"] == pins["candidate_matrix"]
        and sha(resolve(roots, scored["matrix"])) == sha(matrix)
        and scored["cohort"] == pins["cohort"]
        and scored["calibration_method"] == pins["calibration_method"]
        and scored["C6_reference_reads"] == scored["T1_photo_reads"] == 0,
        "wrong calibration protocol or candidates",
    )
    cells = load(matrix)["cells"]
    objects = cohort["splits"]["C6"]
    require(
        set(cells) == {o + "-" + c for o in objects for c in CONDITIONS}
        and len(objects) == 6
        and not set(objects).intersection(cohort["splits"]["T1"]),
        "the matrix must cover only the six registered calibration objects",
    )
    terminal = []
    n = 0
    for obj in objects:
        for cond in CONDITIONS:
            cell = cells[obj + "-" + cond]
            if cell["status"] == "scored":
                case, _ = verify_reconstruction(
                    roots,
                    obj,
                    cond,
                    cell["case"],
                    cell["readback"],
                    pins["cohort"],
                    pins["reconstruction"],
                )
                path = resolve(roots, cell["scores"])
                risk = load(path)
                require(
                    risk["schema"] == "el-dtu-frozen-scores-1"
                    and risk["case_sha256"] == case.sha256
                    and risk["checkpoint_sha256"] == case.meta["checkpoint_sha256"]
                    and risk["reconstruction_run_id"]
                    == case.meta["reconstruction_run_id"]
                    and risk["models"] == release["models"]
                    and risk["method_scopes"] == SCOPES
                    and risk["release_sha256"] == RELEASE_SHA
                    and risk["condition"] == cond
                    and set(risk["views"]) == set(VIEWS),
                    "unbound C6 scores",
                )
                for v in VIEWS:
                    entry = risk["views"][v]
                    if v in case.failed:
                        require(
                            entry["status"] == "render_failed"
                            and entry["reason"] == case.failed[v]
                            and not entry["methods"],
                            "unbound failed rendering",
                        )
                    else:
                        require(
                            entry["status"] == "ready"
                            and set(entry["methods"]) == set(METHODS),
                            "incomplete per-view score matrix",
                        )
                        for m in METHODS:
                            score_map(
                                path.parent,
                                risk,
                                m,
                                v,
                                case,
                                release["models"][m]["threshold"],
                            )
                            n += 1
            else:
                require(
                    cell["status"] in ("failed", "not_executed")
                    and bool(cell.get("reason")),
                    "missing failure reason",
                )
                disposition = load(resolve(roots, cell["disposition"]))
                require(
                    disposition["object_id"] == obj
                    and disposition["condition"] == cond
                    and disposition["cohort"] == pins["cohort"]
                    and disposition["reason"] == cell["reason"]
                    and disposition["status"] == cell["status"],
                    "unlinked failed terminal cell",
                )
            terminal.append({"object_id": obj, "condition": cond, **cell})
    save(
        dest / "opening.json",
        {
            "schema": "el-b2-c6-opening-1",
            "run_id": os.environ["EL_RUN_ID"],
            "utc": utc(),
            "cohort": pins["cohort"],
            "calibration_method": pins["calibration_method"],
            "score_candidates": pins["score_candidates"],
            "risk_release": pins["release"],
            "reconstruction": pins["reconstruction"],
            "object_ids": objects,
            "conditions": list(CONDITIONS),
            "target_ids": list(VIEWS),
            "scheduled_entries": len(cells) * len(VIEWS) * len(METHODS),
            "scored_entries": n,
            "terminal_cells": terminal,
            "sealed_matrix_sha256": sha(matrix),
            "predictions_and_risks_sealed": True,
            "permitted_reader": "independent_C6_calibration_evaluator",
            "T1_photographs_permitted": False,
        },
    )


def run(a):
    store, history = a.store.resolve(), a.history.resolve()
    roots = (store, history)
    cohort = selected(store, "b2-dtu-cohort")
    co = load(resolve(roots, cohort))
    objects = co["splits"]["C6"]
    recon = selected(store, "b2-reconstruction-seed-43171")
    rel = bind(history, RELEASE, "payload/cpu/release.json")
    rfile = resolve(roots, rel)
    require(sha(rfile) == RELEASE_SHA, "incorrect RC1 runtime")
    matrix = store / "work/controller/c6-closed-matrix.json"
    if a.action == "scores":
        cells = {}
        for obj in objects:
            for cond in CONDITIONS:
                name = obj + "-" + cond
                readback = (
                    store
                    / "work/controller"
                    / ("b2-c6-readback-" + name)
                    / "selected.json"
                )
                if not readback.exists():
                    fail = (
                        store
                        / "work/controller"
                        / ("b2-c6-failure-" + name)
                        / "selected.json"
                    )
                    if fail.exists():
                        pin = load(fail)
                        dispo = load(resolve(roots, pin))
                        cells[name] = {
                            "status": dispo["status"],
                            "reason": dispo["reason"],
                            "disposition": pin,
                        }
                    continue
                rp = load(readback)
                cp = bind(store, rp["run_id"], "payload/case/case.json")
                _, path = verify_reconstruction(roots, obj, cond, cp, rp, cohort, recon)
                stagespec = load(PROJECT / "protocol/t0-confirmation-v1.json")
                require(
                    sha(PROJECT / "scripts/dtu_score.py")
                    == stagespec["source_sha256"]["scripts/dtu_score.py"],
                    "wrong frozen CPU scorer",
                )
                pins = {
                    "case": cp,
                    "readback": rp,
                    "release": rel,
                    "cohort": cohort,
                    "reconstruction": recon,
                }
                sc = stage(
                    store,
                    history,
                    "c6-score-" + name,
                    "model_score",
                    pins,
                    {"object_id": obj, "condition": cond, "method_scopes": SCOPES},
                    [
                        str(a.cpu_python.absolute()),
                        "-I",
                        "-B",
                        "{source}/scripts/dtu_score.py",
                        "--case",
                        str(path.parent),
                        "--case-sha256",
                        cp["sha256"],
                        "--cpu-dir",
                        str(rfile.parent),
                        "--release-sha256",
                        RELEASE_SHA,
                        "--threads",
                        "4",
                        "--output",
                        "{out}",
                    ],
                    "payload/scores.json",
                    20000000,
                )
                cells[name] = {
                    "status": "scored",
                    "case": cp,
                    "readback": rp,
                    "scores": sc,
                }
        if len(cells) == 12:
            doc = {
                "schema": "el-b2-c6-closed-matrix-1",
                "cohort": cohort,
                "cells": cells,
            }
            if matrix.exists():
                require(load(matrix) == doc, "closed calibration register differs")
            else:
                save(matrix, doc)
            parents = {"cohort": cohort, "reconstruction": recon, "release": rel}
            for name, cell in cells.items():
                for role in ("case", "readback", "scores", "disposition"):
                    if role in cell:
                        parents[role + name] = cell[role]
            stage(
                store,
                history,
                "c6-matrix",
                "archive_verify",
                parents,
                {
                    "role": "seal complete C6 risk and failure register before reading any labels"
                },
                [
                    "{python}",
                    "{source}/scripts/b2_c6_study.py",
                    "snapshot",
                    "--store",
                    str(store),
                    "--history",
                    str(history),
                    "--matrix",
                    "{input:matrix}",
                    "--config",
                    "{config}",
                    "--out",
                    "{out}",
                ],
                "payload/matrix.json",
                100000,
                {"matrix": matrix},
            )
        return {"terminal": len(cells), "expected": 12, "closed": matrix.exists()}
    if a.action == "snapshot":
        book = load(a.matrix)
        src = load(a.config)["sources"]
        require(
            book["cohort"] == src["cohort"]
            and set(book["cells"])
            == {o + "-" + c for o in objects for c in CONDITIONS},
            "incomplete role-valid prediction register",
        )
        for name, cell in book["cells"].items():
            for role in ("case", "readback", "scores", "disposition"):
                if role in cell:
                    require(
                        cell[role] == src[role + name],
                        "unbound member in registered matrix",
                    )
        save(a.out / "matrix.json", book)
        require(sha(a.out / "matrix.json") == sha(a.matrix), "matrix snapshot changed")
        return
    if a.action == "worker":
        return review(roots, load(a.config), a.matrix, a.candidates, a.out)
    cal = load(a.method)
    candidates = load(a.candidate_pin)
    cells = load(matrix)["cells"]
    pins = {
        "cohort": cohort,
        "reconstruction": recon,
        "release": rel,
        "calibration_method": cal,
        "score_candidates": candidates,
        "closed_matrix": selected(store, "c6-matrix"),
        "candidate_matrix": selected(store, "b2-C6-score-only-matrix"),
    }
    for name, cell in cells.items():
        for key in ("case", "readback", "scores", "disposition"):
            if key in cell:
                pins[key + name] = cell[key]
    return stage(
        store,
        history,
        "c6-opening",
        "archive_verify",
        pins,
        {"purpose": "one complete C6 opening after score-only candidate construction"},
        [
            "{python}",
            "{source}/scripts/b2_c6_study.py",
            "worker",
            "--config",
            "{config}",
            "--store",
            str(store),
            "--history",
            str(history),
            "--matrix",
            "{input:matrix}",
            "--candidates",
            "{input:score_candidates}",
            "--out",
            "{out}",
        ],
        "payload/opening.json",
        500000,
        {"matrix": matrix},
        a.retry_of,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("scores", "open", "worker", "snapshot"))
    for name in (
        "store",
        "history",
        "matrix",
        "method",
        "candidate-pin",
        "candidates",
        "out",
        "config",
        "cpu-python",
    ):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--retry-of")
    print(json.dumps(run(p.parse_args()), sort_keys=True))
