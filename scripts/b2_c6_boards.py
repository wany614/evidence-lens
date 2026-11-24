"""Make private readback boards for the first registered calibration subject, after its labels open."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from b2_records import selected, stage
from dtu_eval import photo, score_map
from dtu_eval_board import board
from dtu_eval_metrics import labels
from dtu_score import METHODS, VIEWS
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from evidence_lens.multisource_io import Case
from t0_records import resolve


def draw(roots, pins, out):
    opening = load(resolve(roots, pins["opening"]))
    obj = opening["object_ids"][0]
    cal = load(resolve(roots, pins["calibration"]))
    require(cal["opening"] == pins["opening"] and cal["references"][obj] == pins["references"],
            "calibration references differ from this opening")
    refpath = resolve(roots, pins["references"])
    references = load(refpath)
    require(references["object_id"] == obj and references["opening"] == pins["opening"],
            "private reference lineage differs")
    method = load(resolve(roots, cal["calibration_method"]))
    thresholds = {m: method["models"][m]["threshold"] for m in METHODS}
    examples = []
    for row in (r for r in opening["terminal_cells"] if r["object_id"] == obj):
        k = row["condition"]
        if row["status"] != "scored":
            examples.append({"layout": k, "status": row["status"], "reason": row["reason"]})
            continue
        cp, sp = row["case"], row["scores"]
        case = Case(resolve(roots, cp).parent, cp["sha256"])
        scorepath = resolve(roots, sp)
        scores = load(scorepath)
        sources = {s: f[0] for s, f in case.sources("all").items()}
        for view in VIEWS:
            ref = photo(refpath.parent, references["views"][view]["rgb"])
            frame = case.frame(view)
            require(scores["case_sha256"] == case.sha256 and frame is not None,
                    "predictions and risk do not share a case")
            risk = {m: score_map(scorepath.parent, scores, m, view, case, thresholds[m]) for m in METHODS}
            label = labels(ref, frame["rgb"], np.ones(ref.shape[:2], dtype=bool))
            name = f"{obj}-{k}-{view}.png"
            board(out / name, obj, k, view, sources, frame["rgb"], ref, label, risk, thresholds, frame["alpha"])
            examples.append({"layout": k, "status": "scored", "view": view, "name": name,
                             "pixel_canvas": int(label[0].sum()), "comparable": int(label[1].sum()),
                             "errors": int((label[2] == 1).sum()), "sha256": sha(out / name)})
    require({x["layout"] for x in examples} == {"M2", "M4"}, "one registered layout is missing")
    links = [f'<li><a href="{x["name"]}">{x["name"]}</a></li>' for x in examples if "name" in x]
    (out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>C6 photographic readback</title>'
                                   '<h1>Private C6 readback</h1><p>First registered subject, both layouts and all six targets. '
                                   'States use the original published operating thresholds.</p><ul>' + "".join(links)
                                   + "</ul>", encoding="utf-8")
    save(out / "boards.json", {"schema": "el-b2-private-calibration-boards-1", "run_id": os.environ["EL_RUN_ID"],
                               "display_rule": "first registered subject with all scheduled layouts and views",
                               "object_id": obj, "inputs": pins, "examples": examples})


def main(a):
    roots = (a.store.resolve(), a.history.resolve())
    if a.action == "worker":
        return draw(roots, load(a.config)["sources"], a.out)
    opening = selected(roots[0], "c6-opening")
    cal = load(a.calibration)
    d = load(resolve(roots, opening))
    obj = d["object_ids"][0]
    pins = {"opening": opening, "calibration": cal, "references": load(resolve(roots, cal))["references"][obj]}
    for r in d["terminal_cells"]:
        if r["object_id"] == obj and r["status"] == "scored":
            pins.update({f + r["condition"]: r[f] for f in ("case", "scores")})
    return stage(*roots, "c6-private-boards", "report", pins,
                 {"display_rule": "first registered calibration subject, every scheduled condition and view"},
                 ["{python}", "{source}/scripts/b2_c6_boards.py", "worker", "--store", str(roots[0]),
                  "--history", str(roots[1]), "--config", "{config}", "--out", "{out}"],
                 "payload/boards.json", 14000000, retry=a.retry_of)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("record", "worker"))
    for f in ("store", "history", "calibration", "config", "out"):
        p.add_argument("--" + f, type=Path)
    p.add_argument("--retry-of")
    print(json.dumps(main(p.parse_args()), sort_keys=True))
