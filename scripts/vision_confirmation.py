"""Read only explicitly opened confirmation or historical-regression inputs, bound to the same method."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from evidence_lens.formal_inputs import compact_pin, quality_parents, require, signature
from evidence_lens.l1 import ALL_COLUMNS, load_package, schema_digest
from evidence_lens.scoring_io import TARGETS, file_hash, new_output, write_json
from vision_cases import case_board
from vision_data import TRUTH, assetspec, emit_cell, file_for, open_dataset, packed, prepare
from vision_evaluate import (
    TWO,
    SCORE_SETS,
    aggregate_method,
    compact_curve,
    groups,
    deploy_contract,
    evaluate,
    guard_test,
    identity,
    macro_families,
    model_matches,
    paired_report,
    read_scores,
    score_sources,
    system_counts,
    test_access,
)
from vision_pipeline import ORIGINAL_CATALOG_SHA
from vision_registry import registry
from vision_sources import METHODS, find_run, make_index, member, pinned_cell, signed

COHORTS = ("new_confirmation", "historical_regression")
CONDITIONS = ("B0", "R1", "Q1")
PRIOR_REPORT = {
    "run_id": "20261123T050620457525Z-a311c1632b7d4b1b",
    "manifest_sha256": "209fb7f045a7e79aaa2113eb50a8823c4f0dbf66098e058ae4b201a826653aba",
}


def successful(roots, pin, name):
    run = find_run(roots, compact_pin(pin))
    require(run.run["execution_status"] == "succeeded", "required original stage did not finish")
    p, spec = member(run, "payload/" + name)
    return p, spec["sha256"]


def catalog_grid(catalog):
    return {
        (r["scene_family_id"], condition): r
        for r in catalog["families"]
        if r["split"] == "test"
        for condition in CONDITIONS
    }


def authenticated(roots, new, new_sha):
    marker_sha, method = test_access(roots, (new, new_sha))
    marker = signed(roots[0] / "protocol/stage2" / new_sha / "test-opening.json", marker_sha)
    pin = {"run_id": marker["freeze_run_id"], "manifest_sha256": marker["freeze_manifest_sha256"]}
    p, sha = successful(roots, pin, "method-freeze.json")
    require(
        sha == marker["method_sha256"]
        and method["run_id"] == pin["run_id"]
        and method["display_rule"] == "v01",
        "different active method",
    )
    return marker_sha, method, pin, sha


def historical_materials(roots, old, marker_sha, output):
    cat = signed(old, ORIGINAL_CATALOG_SHA)
    grid = catalog_grid(cat)
    require(len(grid) == 12, "four historical families with all three conditions required")
    initial, initial_sha = successful(roots, PRIOR_REPORT, "public-numerical/selection.json")
    sealed = signed(initial, initial_sha)
    previous = {
        (r["scene_family_id"], r["condition_id"]): compact_pin(r) for r in sealed["comparison_trials"]
    }
    require(
        sealed["catalog_sha256"] == ORIGINAL_CATALOG_SHA and set(previous) == set(grid),
        "historical report roster changed",
    )
    cells = []
    for (family, condition), row in sorted(grid.items()):
        path = roots[0] / "trials" / row["trials"][condition] / "trial_selection.json"
        sha = file_hash(path)
        q = signed(path, sha)
        rec, rs = successful(roots, previous[family, condition], "formal_evaluation.json")
        original = signed(rec, rs)
        require(
            original["scene_family_id"] == family
            and original["condition_id"] == condition
            and all(original["parents"][k] == p for k, p in quality_parents(q).items()),
            "historical selection differs from original held-out run",
        )
        if condition != "B0":
            b = roots[0] / "trials" / row["trials"]["B0"] / "trial_selection.json"
            base = signed(b, q["b0_selection_sha256"])
            require(
                compact_pin(q["selected"]["reference"]) == compact_pin(base["selected"]["reference"]),
                "wrong historical condition pairing",
            )
            if condition == "Q1":
                require(
                    compact_pin(q["selected"]["generation"]) == compact_pin(base["selected"]["generation"])
                    and compact_pin(q["selected"]["preparation"]) == compact_pin(q["selected"]["producer"]),
                    "wrong historical precision intervention",
                )
        cell = pinned_cell(roots, path, sha, row, condition)
        require(
            cell["input_sha256"] == original["input_sha256"]
            and cell["camera_sha256"] == original["camera_sha256"]
            and all(
                cell["views"][v]["reference"]["sha256"] == original["target_reference_sha256"][v]
                for v in TARGETS
            ),
            "historical originals differ",
        )
        cells.append(
            {
                **cell,
                "split": "historical_regression",
                "previous_comparison": previous[family, condition] | {"receipt_sha256": rs},
            }
        )
    from vision_sources import paired_render_sources

    for family in {c["family"] for c in cells}:
        paired_render_sources([x for x in cells if x["family"] == family])
    with new_output(output) as out:
        write_json(
            out / "materials.json",
            {
                "schema": "el-vision-history-materials-1",
                "cohort": "historical_regression",
                "catalog_sha256": ORIGINAL_CATALOG_SHA,
                "test_unlock_sha256": marker_sha,
                "prior_report": PRIOR_REPORT,
                "prior_public_selection_sha256": initial_sha,
                "cells": cells,
            },
        )


def validated_dataset(roots, spec, new, old, marker_sha, method):
    fp, ds_sha = successful(roots, spec["dataset"], "dataset.json")
    if spec["cohort"] == COHORTS[0]:
        ds = open_dataset(fp.parent, ds_sha, "test", marker_sha)
        require(guard_test(ds, roots) == method, "different test opening")
        catalog = signed(new, method["catalog_sha256"])
        grid = catalog_grid(catalog)
        slots = {(f, c, v) for f, c in grid for v in TARGETS}
        require(
            len(grid) == 24
            and len(ds["views"]) == len(slots)
            and {identity(r) for r in ds["views"]} == slots
            and ds["scope"] == "specified_materials"
            and all(
                r["height"] == r["width"] == 512
                and r["cv_fold"] is None
                and r.get("group") == grid[r["family"], r["condition"]].get("test_group")
                for r in ds["views"]
            ),
            "confirmation score inputs differ from the complete native eight-family registry",
        )
    else:
        ds = signed(fp, ds_sha)
        require(
            ds["schema"] == "el-vision-historical-dataset-1"
            and ds["split"] == COHORTS[1]
            and ds["test_unlock_sha256"] == marker_sha
            and ds["catalog_sha256"] == ORIGINAL_CATALOG_SHA
            and ds["columns"] == ALL_COLUMNS
            and ds["feature_schema_digest"] == schema_digest(),
            "historical regression transport mismatch",
        )
        slots = {(f, c, v) for f, c in catalog_grid(signed(old, ORIGINAL_CATALOG_SHA)) for v in TARGETS}
        require(
            {identity(r) for r in ds["views"]} == slots and len(ds["views"]) == len(slots),
            "missing historical cell/view",
        )
    signed_inputs = method["learning_split_audit"]["source_input_sha256"]
    learning = {h for seen in signed_inputs.values() for h in seen}
    require(
        learning.isdisjoint({r["source_input_sha256"] for r in ds["views"] if r["source_input_sha256"]}),
        "confirmation or historical source PNG duplicates learning material",
    )
    return fp.parent, ds, ds_sha


def original_package(roots, method, name):
    role = "L1-full" if name == "original-five" else name
    assert role in ("L1-full", *TWO)
    model = method["models"][role]
    if name == "original-five":
        require(
            model["model_sha256"] == method["models"]["L1-geometry"]["model_sha256"], "unmatched LR models"
        )
    run = find_run(roots, compact_pin(model))
    for record in model["files"]:
        member(run, record["path"], record)
    entry = next(x for x in model["files"] if Path(x["path"]).name == "model.json")
    path = run.open(entry["path"])
    require(
        entry["sha256"] == model["model_sha256"] and model_matches(method, role, model["model_sha256"]),
        "wrong frozen package",
    )
    return path.parent, model["model_sha256"]


def score_deploy(root, ds, ds_sha, method, method_sha, tag, package, model_sha, outpath):
    """This stage opens only target RGB and deployable features, plus the model package."""
    require(tag in SCORE_SETS, "unregistered method")
    if tag == "original-five":
        from vision_baselines import infer_five

        mod = load_package(package, expected_index_sha256=model_sha)
        require(
            not set(mod["index"]["training_sources"]).intersection(r["family"] for r in ds["views"])
            and not {h for h in mod["index"]["training_sources"].values() if h}.intersection(
                r["source_input_sha256"] for r in ds["views"]
            ),
            "training source in independent cohort",
        )
    else:
        from evidence_lens_vision.deploy import open_model, score

        mod, pre, idx = open_model(package, model_sha, "cpu")
        require(
            idx["branch"] == tag
            and idx["model_id"] == method["models"][tag]["run_id"]
            and idx["threshold"] == method["thresholds"][tag]["tau"]
            and idx["seed"] == 1729,
            "wrong primary CPU package",
        )
    views = []
    with new_output(outpath) as out:
        for row in ds["views"]:
            rec = {k: row[k] for k in ("family", "condition", "view", "reason")}
            rgb, dep = row["rgb"], row["full"]["deploy"]
            rec.update(
                **{
                    k: row[k]
                    for k in (
                        "source_input_sha256",
                        "source_camera_sha256",
                        "source_prediction_sha256",
                        "target_prediction_sha256",
                        "render_pin",
                    )
                },
                rgb_sha256=rgb["sha256"] if rgb else None,
                deploy_sha256=dep["sha256"] if dep else None,
                score=None,
            )
            if row["status"] == "render_failed":
                require(rgb is None and dep is None and rec["reason"], "unexplained missing prediction")
                rec["status"] = "render_failed"
            else:
                require(row["status"] == "completed" and rgb and dep, "missing deployable source")
                pixels = np.load(file_for(root, rgb), allow_pickle=False)
                data = packed(file_for(root, dep), ("X", "scoreable", "l0", "alpha_score"))
                require(pixels.shape == (row["height"], row["width"], 3), "wrong target raster")
                if tag == "original-five":
                    states = infer_five(pixels, data, mod)
                else:
                    states, _ = score(mod, pixels, data["X"], data["scoreable"], pre, "cpu", idx["threshold"])
                path = out / row["family"] / row["condition"] / (row["view"] + ".npz")
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **states)
                rec.update(status="scored", reason=None, score=assetspec(path, out))
            views.append(rec)
        write_json(
            out / "scores.json",
            {
                "schema": "el-vision-scores-1",
                "method": tag,
                "model_sha256": model_sha,
                "method_freeze_sha256": method_sha,
                "dataset_sha256": ds_sha,
                "views": views,
            },
        )


def history_evaluation(root, ds, ds_sha, sources, tau, method_sha, marker_sha, outpath):
    families = defaultdict(list)
    statuses = []
    current = None
    seen = set()
    with new_output(outpath) as out:

        def save_curves(key, bag, den):
            if key is not None:
                write_json(
                    out / key[0] / key[1] / "curves.json",
                    {m: compact_curve(bag[m], den, tau[m]) for m in METHODS},
                )

        bag = defaultdict(list)
        den = 0
        for row in ds["views"]:
            t = packed(file_for(root, row["full"]["truth"]), TRUTH)
            e = t["rgb_evaluable"]
            v = t["reference_valid"]
            err = t["rgb_error"]
            require(
                int(v.sum()) == row["V"]
                and int(e.sum()) == row["E"]
                and not (e & ~v).any()
                and np.array_equal(np.isfinite(err), e)
                and np.array_equal(t["rgb_high_error"][e], err[e] > np.float32(0.05))
                and np.all(t["rgb_high_error"][~e] == 255),
                "historical RGB/error/V/E disagreement",
            )
            scores, disp = read_scores(row, sources, tau, deploy_contract(root, row))
            measures = system_counts(t, scores, tau, np.ones_like(e))
            f, c, vv = identity(row)
            if (f, c) != current:
                save_curves(current, bag, den)
                require((f, c) not in seen, "historical transport revisits a completed family/condition")
                current = f, c
                seen.add(current)
                bag, den = defaultdict(list), 0
            den += int(v.sum())
            for m, s in scores.items():
                ok = e & np.isfinite(s)
                bag[m].append(groups(s[ok], (t["rgb_high_error"][ok] == 1)))
            folder = out / f / c
            folder.mkdir(parents=True, exist_ok=True)
            write_json(
                folder / (vv + ".json"),
                {"region": {"all": measures}, "status": disp, "truth_sha256": row["full"]["truth"]["sha256"]},
            )
            families[f, c].append(measures)
            statuses.append(
                {
                    "family": f,
                    "condition": c,
                    "view": vv,
                    "render": row["status"],
                    "reason": row["reason"],
                    "scientific_disposition": row["scientific_disposition"],
                    "failure_kind": row["failure_kind"],
                    "methods": disp,
                    "V": row["V"],
                    "E": row["E"],
                }
            )
        save_curves(current, bag, den)
        by = [
            {
                "family": f,
                "condition": c,
                "group": None,
                "methods": {m: aggregate_method([v[m] for v in views]) for m in METHODS},
            }
            for (f, c), views in sorted(families.items())
        ]
        macro = [
            {
                "condition": c,
                "cohort": "all",
                "methods": {
                    m: macro_families([r["methods"][m] for r in by if r["condition"] == c]) for m in METHODS
                },
            }
            for c in CONDITIONS
        ]
        for name, doc in (
            ("families", by),
            ("macro", macro),
            ("paired", paired_report(by)),
            ("status", statuses),
        ):
            write_json(out / (name + ".json"), doc)
        write_json(
            out / "evaluation.json",
            {
                "schema": "el-vision-historical-regression-1",
                "split": COHORTS[1],
                "dataset_sha256": ds_sha,
                "methods": list(METHODS),
                "thresholds": tau,
                "method_sha256": method_sha,
                "family_count": len({f for f, c in families}),
                "planned_target_views": len(statuses),
                "test_opening_sha256": marker_sha,
                "inputs": {tag: src[2] for tag, src in sources.items()},
                "files": {p.relative_to(out).as_posix(): signature(p) for p in sorted(out.rglob("*.json"))},
            },
        )


def numeric_difference(a, b, ok):
    if a is None or b is None or not np.any(ok):
        return {"paired_values": 0, "changed_bits": 0, "sum_abs": 0.0, "max_abs": None}
    av, bv = a[ok], b[ok]
    require(
        av.dtype == bv.dtype == np.float32 and np.isfinite(av).all() and np.isfinite(bv).all(),
        "unusable paired array",
    )
    dif = np.abs(av.astype(np.float64) - bv)
    return {
        "paired_values": int(av.size),
        "changed_bits": int(np.count_nonzero(av.view(np.uint32) != bv.view(np.uint32))),
        "sum_abs": float(dif.sum(dtype=np.float64)),
        "max_abs": float(dif.max()),
    }


def paired_perturbations(root, ds, sources, tau):
    rows = {identity(r): r for r in ds["views"]}
    output = []

    def read(row):
        e = packed(file_for(root, row["full"]["truth"]), ("rgb_evaluable",))["rgb_evaluable"]
        rgb = np.load(file_for(root, row["rgb"]), mmap_mode="r", allow_pickle=False) if row["rgb"] else None
        scores, _ = read_scores(row, sources, tau)
        return e, rgb, scores

    for f in sorted({r["family"] for r in ds["views"]}):
        cells = {c: defaultdict(list) for c in ("R1", "Q1")}
        for v in TARGETS:
            b = rows[f, "B0", v]
            be, br, bs = read(b)
            for c, stats in cells.items():
                row = rows[f, c, v]
                ce, cr, cs = read(row)
                e = be & ce
                stats["rgb"].append(numeric_difference(br, cr, e))
                for m in METHODS:
                    stats[m].append(
                        numeric_difference(bs[m], cs[m], e & np.isfinite(bs[m]) & np.isfinite(cs[m]))
                    )
        for c, stats in cells.items():
            final = {}
            for tag, values in stats.items():
                n = sum(q["paired_values"] for q in values)
                changes = sum(q["changed_bits"] for q in values)
                final[tag] = {
                    "paired_values": n,
                    "bitwise_changed_values": changes,
                    "bitwise_identical": changes == 0 if n else None,
                    "mean_absolute_difference": sum(q["sum_abs"] for q in values) / n if n else None,
                    "max_absolute_difference": max(
                        (q["max_abs"] for q in values if q["max_abs"] is not None), default=None
                    ),
                    "defined_views": sum(q["paired_values"] > 0 for q in values),
                    "scheduled_views": 6,
                }
            output.append({"family": f, "condition": c, "baseline": "B0", "paired": final})
    return output


def boards(roots, mat, root, ds, ds_sha, sources, tau, cohort, method_sha, outpath):
    rows = {identity(r): r for r in ds["views"]}
    result = []
    with new_output(outpath) as out:
        for cell in mat["cells"]:
            f, c = cell["family"], cell["condition"]
            row = rows[f, c, "v01"]
            require(row["selection_sha256"] == cell["selection_sha256"], "case selection differs")
            chosen = signed(
                roots[0] / "trials" / cell["trial_id"] / "trial_selection.json", cell["selection_sha256"]
            )
            ref = find_run(roots, cell["parents"]["reference"])
            refview = cell["views"]["v01"]["reference"]
            exact_ref, _ = member(ref, refview["path"], refview)
            ref_rgb = packed(exact_ref, ("rgb",))
            source = None
            source_sha = None
            if c == "B0" or c == "Q1":
                source_path, sp = member(
                    ref, "payload/input.png", cell["input_sha256"] if cell["input_sha256"] else None
                )
            elif chosen["selection_status"] == "selected":
                a = chosen["sources"]["input_png"]
                source_path, sp = member(find_run(roots, compact_pin(a)), a["path"], a["sha256"])
            else:
                source_path = None
            if source_path:
                with Image.open(source_path) as im:
                    source = im.convert("RGB")
                source_sha = sp["sha256"]
            predict = np.load(file_for(root, row["rgb"]), allow_pickle=False) if row["rgb"] else None
            truth = packed(file_for(root, row["full"]["truth"]), TRUTH)
            scores, _ = read_scores(row, sources, tau, deploy_contract(root, row))
            pins = {
                "selection_sha256": cell["selection_sha256"],
                "input_sha256": source_sha,
                "reference_sha256": refview["sha256"],
                "rgb_sha256": row["rgb"]["sha256"] if row["rgb"] else None,
                "prediction_member_sha256": (cell["views"]["v01"]["prediction"] or {}).get("sha256"),
                "truth_sha256": row["full"]["truth"]["sha256"],
                "method_sha256": method_sha,
                "scores": {
                    m: entries[identity(row)]["score"] for m, (_src, entries, _spec) in sources.items()
                },
                "dataset_sha256": ds_sha,
            }
            path = out / f / c
            case_board(
                path,
                {"cohort": cohort, "family": f, "condition": c},
                source,
                ref_rgb,
                predict,
                truth,
                scores,
                tau,
                pins,
            )
            result.append(
                {
                    "family": f,
                    "condition": c,
                    "view": "v01",
                    "index": assetspec(path / "board.json", out),
                    "board": assetspec(path / "board.png", out),
                }
            )
        pair = out / "paired-response.json"
        write_json(
            pair,
            {
                "schema": "el-vision-native-response-1",
                "cohort": cohort,
                "method_sha256": method_sha,
                "dataset_sha256": ds_sha,
                "domain": "Both condition and B0 RGB evaluable. Risk changes also require both native scores finite. RGB counts channel values.",
                "families": paired_perturbations(root, ds, sources, tau),
            },
        )
        write_json(
            out / "boards.json",
            {
                "schema": "el-vision-native-cases-1",
                "cohort": cohort,
                "method_sha256": method_sha,
                "dataset_sha256": ds_sha,
                "count": len(result),
                "cases": result,
                "distribution": "private research only",
                "paired_response": assetspec(pair, out),
            },
        )


def work(action, spec, roots, new, old, outpath, approved_registry=None):
    cohort = spec["cohort"]
    require(cohort in COHORTS, "unknown case series")
    new_sha = spec["catalog_sha256"]
    marker, method, method_pin, method_sha = authenticated(roots, new, new_sha)
    require(
        method_pin == compact_pin(spec["method"]) and method_sha == spec["method_sha256"],
        "requested method differs from canonical opening",
    )
    if action == "materials":
        if cohort == COHORTS[1]:
            return historical_materials(roots, old, marker, outpath)
        require(approved_registry, "acquisition registry must be staged with its pin")
        approved = signed(approved_registry, spec["acquisition_registry_sha256"])
        require(
            approved["schema"] == "el-vision-selected-trials-1"
            and approved["catalog_sha256"] == new_sha
            and approved["split"] == "test"
            and {t: {k: r[k] for k in ("selection_sha256", "status")} for t, r in approved["trials"].items()}
            == spec["trials"],
            "staged acquisition registry differs",
        )
        with new_output(outpath) as out:
            rr = registry(*roots, (new, new_sha), "test", out / "registry")
            require(rr["trials"] == spec["trials"], "confirmation selections changed during acquisition")
            index = out / "registry/selections.json"
            make_index(
                *roots, new, new_sha, "test", out / "index", index, file_hash(index), test_gate_sha=marker
            )
            (out / "index/materials.json").replace(out / "materials.json")
            (out / "index").rmdir()
        return
    if action == "transport":
        matpath, matsha = successful(roots, spec["materials"], "materials.json")
        source = signed(matpath, matsha)
        require(source["test_unlock_sha256"] == marker, "material registered at a different opening")
        if cohort == COHORTS[0]:
            return prepare(
                [(matpath, matsha)], roots, "test", outpath, gate=marker, test_catalog=(new, new_sha)
            )
        require(
            source["schema"] == "el-vision-history-materials-1"
            and source["catalog_sha256"] == ORIGINAL_CATALOG_SHA,
            "wrong historical materials",
        )
        with new_output(outpath) as out:
            views = []
            for cell in source["cells"]:
                views.extend(emit_cell(cell, ORIGINAL_CATALOG_SHA, roots, out))
            write_json(
                out / "dataset.json",
                {
                    "schema": "el-vision-historical-dataset-1",
                    "split": cohort,
                    "catalog_sha256": ORIGINAL_CATALOG_SHA,
                    "materials_sha256": matsha,
                    "test_unlock_sha256": marker,
                    "columns": ALL_COLUMNS,
                    "feature_schema_digest": schema_digest(),
                    "views": views,
                },
            )
        return
    root, ds, ds_sha = validated_dataset(roots, spec, new, old, marker, method)
    if action == "score":
        tag = spec["score_set"]
        return score_deploy(
            root, ds, ds_sha, method, method_sha, tag, *original_package(roots, method, tag), outpath
        )
    srcpins = [successful(roots, spec["scores"][tag], "scores.json") for tag in SCORE_SETS]
    sources = score_sources(ds, ds_sha, srcpins)
    for row in ds["views"]:
        required = (
            "source_input_sha256",
            "source_camera_sha256",
            "source_prediction_sha256",
            "target_prediction_sha256",
            "render_pin",
        )
        for _, records, _ in sources.values():
            original = records[identity(row)]
            require(
                all(k in row and k in original and original[k] == row[k] for k in required),
                "confirmation/historical scorer and original condition/source hashes differ",
            )
    for tag, (scorefile, digest) in zip(SCORE_SETS, srcpins):
        doc = signed(scorefile, digest)
        require(
            doc["method"] == tag
            and doc["method_freeze_sha256"] == method_sha
            and model_matches(method, "L1-full" if tag == "original-five" else tag, doc["model_sha256"]),
            "wrong scored model",
        )
    tau = {k: v["tau"] for k, v in method["thresholds"].items()}
    if action == "boards":
        mat, msha = successful(roots, spec["materials"], "materials.json")
        return boards(roots, signed(mat, msha), root, ds, ds_sha, sources, tau, cohort, method_sha, outpath)
    require(action == "evaluate", "unknown evaluation mode")
    if cohort == COHORTS[1]:
        return history_evaluation(root, ds, ds_sha, sources, tau, method_sha, marker, outpath)
    p, h = successful(roots, spec["thresholds"], "thresholds.json")
    require(h == method["thresholds_sha256"], "threshold source differs from joint freeze")
    return evaluate(root, ds_sha, "test", srcpins, (p, h), outpath, roots, [(new, new_sha)])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("materials", "transport", "score", "evaluate", "boards", "report"))
    for n in ("roster", "store", "shelf", "catalog", "original-catalog", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--run-id")
    p.add_argument("--selection-registry", type=Path)
    a = p.parse_args()
    spec = signed(a.roster, a.sha256)
    roots = (a.store.resolve(), a.shelf.resolve())
    if a.action == "report":
        from vision_public_study import publish_report

        publish_report(roots, spec, a.catalog, a.original_catalog, a.output, a.run_id)
    else:
        work(a.action, spec, roots, a.catalog, a.original_catalog, a.output, a.selection_registry)
