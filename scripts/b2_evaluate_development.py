"""Evaluate every completed development trajectory and fixed-seed control after its prediction seal."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageDraw
from b2_records import selected, stage
from dtu_eval import photo
from dtu_score import VIEWS
from evidence_lens._bundle import encoded, load, save, sha, utc
from evidence_lens._verify import require
from evidence_lens.multisource_io import Case
from recovery_eval_quality import measurement
from stage3_risk_boards import heat, rgb
from t0_evaluate import table
from t0_records import bind, resolve

OBJECTS = ("set1", "set6")
RANDOM = tuple("random-" + str(n) for n in (137101, 137102, 137103))
SELECTED = ("uniform", "risk", *RANDOM)
EXTRA = ("seed-43172", "seed-43173")
BRANCHES = ("M2", "M4-fixed", *SELECTED, *EXTRA)
PROJECT = Path(__file__).resolve().parents[1]


def observed(roots, manifest, pins):
    cells = {}
    recipes = {n: load(resolve(roots, pins["recipe" + str(n)])) for n in (43171, 43172, 43173)}
    for n, recipe in recipes.items():
        require(recipe["seed"] == n and {**recipe, "seed": 43171} == recipes[43171]
                and recipe["stop_step"] == 5000, "sensitivity changes more than the optimizer seed")
    orders = {(o, b) for o in OBJECTS for b in (*SELECTED, *EXTRA)}
    rows = manifest["rows"]
    require(
        len(rows) == 14 and {(x["object_id"], x["branch"]) for x in rows} == orders,
        "missing development trajectory",
    )
    for o in OBJECTS:
        choice = load(resolve(roots, pins["choices" + o]))
        for b in ("M2", "M4-fixed"):
            spec = pins[o + b]
            _, folder = spec, resolve(roots, spec)
            case = Case(folder.parent, spec["sha256"])
            require(
                case.condition == ("M4" if b == "M4-fixed" else "M2"),
                "historical Case sources differ",
            )
            cells[o, b] = (
                case,
                {
                    "case": spec,
                    "status": "succeeded",
                    "reuse_of": "previously exposed D recovery endpoint",
                },
            )
        for row in (x for x in rows if x["object_id"] == o):
            b = row["branch"]
            if b in SELECTED:
                sp = row["selection"]
                require(
                    sp["run_id"] == pins["choices" + o]["run_id"]
                    and sp["sha256"] == choice["selections"][b],
                    "selection receipt differs from frozen joint allocation",
                )
                dec = load(resolve(roots, sp))
                require(
                    dec["object_id"] == o
                    and dec["strategy_id"] == b
                    and dec["selected_official_ids"] == choice["results"][b],
                    "invalid exposure ledger",
                )
            if row["status"] in ("failed", "not_executed"):
                require(
                    row["reason"]
                    and row.get("disposition")
                    and resolve(roots, row["disposition"]).is_file(),
                    "missing terminal evidence",
                )
                cells[o, b] = (None, row)
                continue
            cp, rp = row["case"], row["readback"]
            rec = load(resolve(roots, rp))
            case = Case(resolve(roots, cp).parent, cp["sha256"])
            require(
                rec["object_id"] == o
                and rec["verified"]
                and rec["case_sha256"] == case.sha256
                and rec["completed_steps"] == 5000
                and rec["condition"] == case.condition == "M4"
                and rec["target_reference_files_opened"] == 0,
                "invalid source-only development reconstruction",
            )
            ids = rec["official_by_slot"]
            order = [ids[f"s{i:02}"] for i in range(4)]
            if b in SELECTED:
                require(
                    order == [24, *choice["results"][b], 22]
                    and rec["optimizer_seed"] == 43171
                    and rec["layout"] == "M4-selected",
                    "cold reconstruction differs from selected photo pair",
                )
            else:
                require(
                    order == [24, 23, 17, 22]
                    and rec["optimizer_seed"] == int(b[5:])
                    and rec["layout"] == "M4-fixed",
                    "wrong fixed-layout sensitivity seed",
                )
                seed = rec["optimizer_seed"]
                original = load(resolve(roots, pins[o + "fixedGeneration"]))
                new = load(resolve(roots, pins[o + b + "Generation"]))
                keys = ("condition", "generator_source_ids", "object_id", "init_sha256",
                        "source_camera_sha256", "source_rgb_sha256", "worker")
                require(original["identity"]["recipe_sha256"] == pins["recipe43171"]["sha256"]
                        and new["identity"]["recipe_sha256"] == pins["recipe" + str(seed)]["sha256"]
                        and new["last_completed_step"] == original["last_completed_step"] == 5000
                        and all(original["identity"][k] == new["identity"][k] for k in keys)
                        and original["run_id"] == cells[o, "M4-fixed"][0].meta["reconstruction_run_id"]
                        and new["run_id"] == case.meta["reconstruction_run_id"],
                        "fixed seed controls differ in initialization, source, worker, topology or schedule")
            cells[o, b] = (case, row)
    return cells


def page(items, reference):
    tile = 194
    w = len(items) * tile
    sheet = Image.new("RGB", (w, 2 * tile + 37), "white")
    d = ImageDraw.Draw(sheet)
    for i, (title, im, err, e) in enumerate(items):
        x = i * tile
        if im is not None:
            sheet.paste(Image.fromarray(rgb(im)).resize((tile - 4, tile - 4)), (x, 22))
        if err is not None:
            sheet.paste(
                Image.fromarray(heat(err, 0.25, e)).resize((tile - 4, tile - 4)),
                (x, tile + 24),
            )
        d.text((x + 2, 5), title, fill="black")
    d.text(
        (2, 2 * tile + 24),
        "Top: display RGB. Bottom: mean absolute RGB error, colour scale 0 to 0.25.",
        fill="black",
    )
    return sheet


def evaluate(roots, cfg, manifest, out):
    pins = cfg["sources"]
    cells = observed(roots, load(manifest), pins)
    # All registrations, selections and completions above are verified before reference image access.
    opened = utc()
    public, private = out / "public", out / "private"
    public.mkdir()
    private.mkdir()
    rows = []
    totals = []
    paired = []
    panels = []
    for o in OBJECTS:
        refpath = resolve(roots, pins["reference" + o])
        book = load(refpath)
        require(
            book["schema"] == "el-dtu-references-1"
            and book["object_id"] == o
            and set(book["views"]) == set(VIEWS),
            "wrong independent photographic references",
        )
        for view in VIEWS:
            ref = photo(refpath.parent, book["views"][view]["rgb"])
            cards = [("Reference", ref, None, None)]
            for b in BRANCHES:
                case, meta = cells[o, b]
                frame = case.frame(view) if case is not None else None
                prediction = frame["rgb"] if frame else None
                stats, (_, e, _, err) = measurement(
                    ref, prediction, np.ones(ref.shape[:2], bool)
                )
                rows.append(
                    {
                        "object_id": o,
                        "branch": b,
                        "view": view,
                        "official_target_id": book["views"][view]["official_view_id"],
                        **stats,
                        "E_mask_sha256": hashlib.sha256(e.tobytes(order="C")).hexdigest(),
                        "status": meta["status"],
                        "case_sha256": case.sha256 if case else None,
                        "reuse_of": meta.get("reuse_of"),
                        "reference_sha256": book["views"][view]["rgb"]["sha256"],
                    }
                )
                if b in ("M2", "M4-fixed", "uniform", "risk", *RANDOM):
                    cards.append((b, prediction, err, e))
            path = private / (o + "-" + view + ".png")
            page(cards, ref).save(path)
            panels.append(path.name)
        for b in BRANCHES:
            sub = [x for x in rows if (x["object_id"], x["branch"]) == (o, b)]
            counts = {k: sum(x[k] for x in sub) for k in ("V", "E", "Y", "rgb_mae_sum")}
            totals.append(
                {
                    "object_id": o,
                    "branch": b,
                    **counts,
                    "MAE": counts["rgb_mae_sum"] / counts["E"] if counts["E"] else None,
                    "error_rate": counts["Y"] / counts["E"] if counts["E"] else None,
                    "scheduled_views": 6,
                    "comparable_views": sum(x["E"] > 0 for x in sub),
                    "missing_reference_valid_pixels": counts["V"] - counts["E"],
                    "status": sub[0]["status"],
                    "reuse_of": sub[0]["reuse_of"],
                }
            )
        by = {x["branch"]: x for x in totals if x["object_id"] == o}
        a, base = by["risk"]["MAE"], by["uniform"]["MAE"]
        masks = {b: [r["E_mask_sha256"] for r in rows if r["object_id"] == o and r["branch"] == b]
                 for b in BRANCHES}
        same = masks["risk"] == masks["uniform"]
        randomness = [by[b]["MAE"] for b in RANDOM]
        seeds = [by[b]["MAE"] for b in ("M4-fixed", *EXTRA)]
        complete = all(v is not None for v in randomness)
        ss = all(v is not None for v in seeds)
        paired.append(
            {
                "object_id": o,
                "risk_MAE": a,
                "uniform_MAE": base,
                "risk_minus_uniform_MAE": a - base
                if a is not None and base is not None and same
                else None,
                "same_risk_uniform_E_masks": same,
                "all_branch_E_masks_identical": all(m == masks["M2"] for m in masks.values()),
                "random_scheduled": 3,
                "random_defined": sum(v is not None for v in randomness),
                "random_mean_MAE": mean(randomness) if complete else None,
                "random_min_MAE": min(randomness) if complete else None,
                "random_max_MAE": max(randomness) if complete else None,
                "fixed_three_seed_mean_MAE": mean(seeds) if ss else None,
                "fixed_three_seed_range_MAE": max(seeds) - min(seeds) if ss else None,
            }
        )
    table(public, "views.csv", rows)
    table(public, "branches.csv", totals)
    table(public, "paired.csv", paired)
    (private / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Development predictions and reference error</title><h1>DTU development views</h1><ul>'
        + "".join(f'<li><a href="{name}">{name[:-4]}</a></li>' for name in panels)
        + "</ul>",
        encoding="utf-8",
    )
    save(
        public / "evaluation.json",
        {
            "schema": "el-b2-d-evaluation-1",
            "run_id": os.environ["EL_RUN_ID"],
            "new_predictions_sealed_before_read": True,
            "input_manifest_sha256": sha(manifest),
            "independent_reference_open_utc": opened,
            "objects": list(OBJECTS),
            "views": list(VIEWS),
            "branches": list(BRANCHES),
            "T1_photographs_read": 0,
            "original_and_new_camera_counts": {
                o: {
                    b: {k: by[k] for k in ("V", "E", "Y")}
                    for by in totals
                    if by["object_id"] == o
                    for b in [by["branch"]]
                }
                for o in OBJECTS
            },
            "tables": {
                n: sha(public / n) for n in ("views.csv", "branches.csv", "paired.csv")
            },
        },
    )


def main(a):
    s, h = a.store.resolve(), a.history.resolve()
    if a.action == "worker":
        return evaluate((s, h), load(a.config), a.matrix, a.out)
    historical = load(
        PROJECT / "docs/results/reconstruction-recovery/risk/evidence.json"
    )
    pins = {"method": selected(s, "nbv-method")}
    for seed in (43171, 43172, 43173):
        reg = selected(s, "b2-reconstruction-seed-" + str(seed))
        pins["recipe" + str(seed)] = bind(s, reg["run_id"], "payload/recipe.json")
    for r in historical["scoring_sources"]:
        o = r["object_id"]
        b = "M4-fixed" if r["condition"] == "M4" else "M2"
        pins[o + b] = r["recovery_case"]
        pins["reference" + o] = r["reference"]
        if b == "M4-fixed":
            rid = load(resolve((s, h), r["recovery_case"]))["reconstruction_run_id"]
            pins[o + "fixedGeneration"] = bind(h, rid, "payload/reconstruction.json")
    for o in OBJECTS:
        pins["choices" + o] = selected(s, "nbv-choices-" + o)
    rows = load(a.matrix)["rows"]
    for i, r in enumerate(rows):
        if r["branch"] in EXTRA and r.get("case"):
            rid = load(resolve((s, h), r["case"]))["reconstruction_run_id"]
            pins[r["object_id"] + r["branch"] + "Generation"] = bind(s, rid, "payload/reconstruction.json")
        for field in ("case", "readback", "selection", "disposition"):
            if r.get(field):
                pins[field + str(i)] = r[field]
    return stage(
        s,
        h,
        "d-development-evaluation",
        "evaluation",
        pins,
        {
            "role": "independent paired endpoint evaluator; both objects are exposed development material"
        },
        [
            "{python}",
            "{source}/scripts/b2_evaluate_development.py",
            "worker",
            "--store",
            str(s),
            "--history",
            str(h),
            "--config",
            "{config}",
            "--matrix",
            "{input:matrix}",
            "--out",
            "{out}",
        ],
        "payload/public/evaluation.json",
        30000000,
        {"matrix": a.matrix},
        a.retry_of,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("record", "worker"))
    for n in ("store", "history", "matrix", "out", "config"):
        p.add_argument("--" + n, type=Path)
    p.add_argument("--retry-of")
    print(json.dumps(main(p.parse_args()), sort_keys=True))
