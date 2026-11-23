"""Publish numerical recovery measurements from named seals and bind the private all-view comparison boards."""

import argparse
import csv
import html
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from dtu_score import METHODS, VIEWS
from evidence_lens.formal_inputs import require, verified_file
from evidence_lens.scoring_io import file_hash, new_output, write_json
from recovery_eval_figures import quality_curves, risk_curves
from recovery_eval_io import SYNTH, authority, original, permit_frozen
from recovery_eval_quality import sum_views
from recovery_eval_select import selection
from vision_sources import signed

CELLS = {(o, c) for o in ("set1", "set6") for c in ("M2", "M4")}
QKEYS = ("V", "E", "V_minus_E", "Y", "rgb_mae_sum", "MAE", "Y_over_E")


def csv_file(folder, name, rows):
    require(rows, f"unexpected empty table {name}")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (folder / (name + ".csv")).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def quality_tables(records):
    target, source, aggregates, pairs, missing, comparisons = [], [], [], [], [], []
    for doc in records:
        base = {k: doc[k] for k in ("object_id", "condition", "role", "recipe_id", "recipe_sha256")}
        source_raw = doc["original_source_hashes"]
        raw_old = doc["old_render_hashes"]

        def project(view, role, appearance, dom, data, step, original_hash, prediction_hash=None, **extra):
            if appearance == "old":
                checkpoint = doc["old_checkpoint_sha256"]
            elif appearance == "recovery":
                checkpoint = doc["steps"][str(step)]["checkpoint_sha256"]
            else:
                checkpoint = None
            row = {
                **base,
                "view": view,
                "view_role": role,
                "domain": dom,
                "appearance": appearance,
                "step": step,
                "reference_sha256": original_hash,
                "prediction_sha256": prediction_hash,
                "checkpoint_sha256": checkpoint,
                "original_case_sha256": doc["original_case_sha256"],
                **extra,
            }
            if data is not None:
                row.update({k: data[k] for k in QKEYS})
            (source if role == "source" else target).append(row)

        prior = doc["old_and_zero_training_targets"]
        for v in VIEWS:
            for domain in ("full_crop", "object"):
                t = prior[v][domain]
                for appearance in ("old", "mean_color", "nearest_axis_copy"):
                    src = doc["angular_source_choices"][v]["source_id"]
                    predictor = (
                        raw_old[v]
                        if appearance == "old"
                        else source_raw[src]
                        if appearance == "nearest_axis_copy"
                        else None
                    )
                    info = {"source_copy_id": src} if appearance == "nearest_axis_copy" else {}
                    if t.get("status") == "unavailable":
                        project(
                            v,
                            "target",
                            appearance,
                            domain,
                            None,
                            None,
                            doc["original_target_hashes"][v],
                            predictor,
                            reason=t["reason"],
                            **info,
                        )
                    else:
                        project(
                            v,
                            "target",
                            appearance,
                            domain,
                            t[appearance],
                            None,
                            doc["original_target_hashes"][v],
                            predictor,
                            **info,
                        )
        for appearance in ("old", "mean_color", "nearest_axis_copy"):
            aggregates.append(
                {
                    **base,
                    "appearance": appearance,
                    "step": None,
                    "view_role": "target",
                    "status": "measured",
                    **sum_views([prior[v]["full_crop"][appearance] for v in VIEWS]),
                }
            )
        for s, m in doc["old_source"].items():
            project(s, "source", "old", "source_crop", m, None, source_raw[s], raw_old[s])
        aggregates.append(
            {
                **base,
                "appearance": "old",
                "step": None,
                "view_role": "source",
                "status": "measured",
                **sum_views(list(doc["old_source"].values())),
            }
        )
        for ss, point in doc["steps"].items():
            step = int(ss)
            meta = {
                **base,
                "step": step,
                "appearance": "recovery",
                "status": point["status"],
                "checkpoint_sha256": point.get("checkpoint_sha256"),
                "gaussians": point.get("gaussians"),
                "phase": point.get("phase"),
                "endpoint_role": point.get("endpoint_role"),
                "reason": point.get("reason"),
            }
            if point["status"] != "measured":
                missing.append(meta)
                for role in ("source", "target"):
                    aggregates.append({**meta, "view_role": role})
                continue
            for v, m in point["targets"].items():
                for dom in ("full_crop", "object"):
                    project(
                        v,
                        "target",
                        "recovery",
                        dom,
                        m[dom],
                        step,
                        m["reference_sha256"],
                        m["prediction_sha256"],
                        reason=m.get("object_reason") if dom == "object" else None,
                    )
                pr = m["paired_comparable"]
                paired = {
                    k + "_" + direction: val
                    for direction in ("old", "new")
                    for k, val in pr[direction].items()
                    if k in QKEYS
                }
                pairs.append(
                    {
                        **meta,
                        "view": v,
                        "original_V": pr["planned_reference_V"],
                        "paired_E": pr["paired_E"],
                        **paired,
                    }
                )
            for s, m in point["source"].items():
                project(
                    s, "source", "recovery", "source_crop", m, step, source_raw[s], m["prediction_sha256"]
                )
            for role in ("source", "target"):
                aggregates.append({**meta, "view_role": role, **point[role + "_summary"]})
            measure = point["targets"]
            changes = [v for v in VIEWS if measure[v]["paired_comparable"]["paired_E"] > 0]
            for baseline in ("old", "mean_color", "nearest_axis_copy"):
                # Restrict comparator direction to views with identical numerical support for the coarse references.
                # A true old/new pair is already recomputed by the evaluation worker on their native E intersection.
                if baseline == "old":
                    vals = [
                        (
                            measure[v]["paired_comparable"]["old"]["MAE"],
                            measure[v]["paired_comparable"]["new"]["MAE"],
                        )
                        for v in changes
                    ]
                else:
                    vals = [
                        (prior[v]["full_crop"][baseline]["MAE"], measure[v]["full_crop"]["MAE"])
                        for v in VIEWS
                        if measure[v]["full_crop"]["E"] == measure[v]["full_crop"]["V"] > 0
                    ]
                comparisons.append(
                    {
                        **meta,
                        "comparator": baseline,
                        "views_scheduled": len(VIEWS),
                        "views_comparable": len(vals),
                        "lower_MAE_views": sum(n < o for o, n in vals),
                        "equal_MAE_views": sum(n == o for o, n in vals),
                        "higher_MAE_views": sum(n > o for o, n in vals),
                    }
                )
    return {
        "target-views": target,
        "source-views": source,
        "quality-curves": aggregates,
        "paired-targets": pairs,
        "comparator-directions": comparisons,
    }, missing


def risk_tables(records):
    tables = defaultdict(list)
    for doc in records:
        base = {k: doc[k] for k in ("object_id", "condition", "role")}
        base["old_checkpoint_sha256"] = doc["checkpoints"]["old"]
        base["new_checkpoint_sha256"] = doc["checkpoints"]["new"]
        base["recovery_stop_step"] = doc["step"]

        def state(version):
            return {
                **base,
                "version": version,
                "step": 256 if version == "old" else doc["step"],
                "parameter_run_id": doc["parameter_runs"][version],
                "checkpoint_sha256": doc["checkpoints"][version],
            }

        for row in doc["rows"]:
            for version, evidence in row["versions"].items():
                ids = {
                    **state(version),
                    "view": row["view"],
                    "domain": "full_crop",
                    "object_mask_status": row["object_mask_status"],
                    "object_mask_reason": row["object_mask_reason"],
                    "reference_sha256": row["target_reference_sha256"],
                    "prediction_sha256": evidence["target_prediction_sha256"],
                }
                for method, result in evidence["metrics"].items():
                    detail = {
                        k: result[dom][f]
                        for dom in ("common", "own")
                        for k, f in ((dom + "_AP", "ap"), (dom + "_missing_reason", "missing_reason"))
                    }
                    tables["method-views"].append(
                        {
                            **ids,
                            "method": method,
                            "scope": doc["method_scopes"][method],
                            "tau": doc["thresholds"][method],
                            "score_sha256": evidence["score_source_sha256"].get(method),
                            **result["counts"],
                            **{k: r["value"] for k, r in result["rates"].items()},
                            **detail,
                        }
                    )
                for method, counts in evidence["unknown_reasons"].items():
                    for cause, v in counts.items():
                        ref = evidence["metrics"][method]["counts"]
                        tables["unknown-reasons"].append(
                            {
                                **ids,
                                "method": method,
                                "cause": cause,
                                "eligible_E": ref["E"],
                                "unknown_total": ref["U"],
                                **v,
                            }
                        )
                for measure, groups in evidence["support"].items():
                    for g in groups:
                        tables["source-support"].append({**ids, "measure": measure, **g})
                if evidence.get("geometry_on_E") is not None:
                    tables["render-diagnostics"].append({**ids, **evidence["geometry_on_E"]})
        for version, group in doc["aggregate"].items():
            for method, result in group.items():
                info = {
                    **state(version),
                    "domain": "full_crop",
                    "method": method,
                    "scope": doc["method_scopes"][method],
                    "tau": doc["thresholds"][method],
                }
                dims = {}
                for dom in ("common", "own"):
                    rank = result["AP"][dom]
                    prev = result["matched_view_prevalence"][dom]
                    require(prev["defined_views"] == rank["defined_views"], "AP/prevalence view sets differ")
                    dims.update(
                        {
                            dom + "_AP": rank["ap"],
                            dom + "_AP_defined_views": rank["defined_views"],
                            dom + "_matched_view_prevalence": prev["value"],
                            "scheduled_views": rank["planned_views"],
                        }
                    )
                tables["method-conditions"].append(
                    {
                        **info,
                        **result["counts"],
                        **dims,
                        **{k: v["value"] for k, v in result["rates"].items()},
                        "ece10": result["ece10"],
                    }
                )
                for row in result["risk_coverage"]["points"]:
                    tables["whole-tie-coverage"].append({**info, **row})
        tables["matched-coverage"].extend({**state(row["version"]), **row} for row in doc["matched_coverage"])
        tables["matched-conditions"].extend(
            {**state(row["version"]), **row} for row in doc["matched_summary"]
        )
    return tables


def pct(val):
    return "N/A" if val is None else f"{100 * val:.2f}%"


def num(val):
    return "N/A" if val is None else f"{val:.4f}"


def quality_text(out, qs, sel, comparisons):
    step = sel["stop_step"]
    text = [
        "# Reconstruction quality",
        "",
        f"The development rule selected **{sel['recipe_id']} at {step:,} updates** on the registered 5,000-update schedule. Selection used the mean of the six target RGB MAEs in each of set1 M2 and M4, giving both conditions equal weight. set6 was previously exposed in D1 and serves as a two-condition regression. The synthetic control uses its original floating-point reference RGB.",
        "",
        "| Object | Sources | Target MAE, old → recovery | Errors/E, old → recovery | Zero-fit mean colour / closest source MAE | Views improved vs D1 | Source MAE, old → recovery |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for d in sorted(qs, key=lambda x: (x["object_id"], x["condition"])):
        if d["recipe_sha256"] != sel["recipe_sha256"]:
            continue
        o, c = d["object_id"], d["condition"]
        point = d["steps"][str(step)]
        comp = d["old_and_zero_training_targets"]

        def base(k):
            return sum_views([v["full_crop"][k] for v in comp.values()])

        old = base("old")
        mean = base("mean_color")
        near = base("nearest_axis_copy")
        new = point.get("target_summary", {})
        oldsrc = sum_views(list(d["old_source"].values()))
        count = next(
            (
                r
                for r in comparisons
                if r["object_id"] == o
                and r["condition"] == c
                and r["recipe_sha256"] == d["recipe_sha256"]
                and r["comparator"] == "old"
                and r["step"] == step
            ),
            {},
        )
        text.append(
            f"| {o} | {c} | {num(old['MAE'])} → {num(new.get('MAE'))} | {pct(old['Y_over_E'])} → {pct(new.get('Y_over_E'))} | {num(mean['MAE'])} / {num(near['MAE'])} | {count.get('lower_MAE_views', 0)}/{count.get('views_comparable', 0)} | {num(oldsrc['MAE'])} → {num(point.get('source_summary', {}).get('MAE'))} |"
        )
    text += [
        "",
        "Errors are pixels whose mean absolute display-RGB difference exceeds 0.05. The full 256 × 256 crop includes background and low-opacity pixels. Each rate uses observed RGB-comparable pixels E. The tables retain reference-valid V, missing observations and object-mask status. DTU supplies no independent 2D object mask for these views, so the image-wide figures cannot separate object from background.",
        "",
        "The mean-colour comparator broadcasts the per-channel mean of the allowed source photographs. The closest-source comparator copies the full crop whose optical axis has the smallest angle to the target, breaking ties by source ID. Both use source images only.",
        "",
        "[Visual readback](visual-readback.md) · [Development curves](quality-curves.svg) · [Source views](source-views.csv) · [Target views](target-views.csv) · [Paired pixels](paired-targets.csv) · [Candidate decisions](stopping-candidates.csv) · [Provenance](evidence.json)",
    ]
    (out / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def risk_text(out, table, matched):
    text = [
        "# Frozen risk on the same camera views",
        "",
        "Four released CPU methods scored both generations of reconstruction. B-L1-transfer always reads s00. The three multisource methods use all photographs permitted by M2 or M4. Weights, preprocessing and thresholds follow v0.3.0-rc.1. Each entry compares the original D1 measurement with the selected recovery reconstruction.",
        "",
        "| Object | Sources | Method | Common AP, old → new | Common prevalence, old → new | Low coverage L/V, old → new | Error in low LY/L, old → new | Unknown U/E, old → new |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    by = {(r["object_id"], r["condition"], r["method"], r["version"]): r for r in table}
    for o, c in sorted(CELLS):
        for m in METHODS:
            a = by.get((o, c, m, "old"), {})
            b = by.get((o, c, m, "new"), {})
            vals = []
            for key in (
                "common_AP",
                "common_matched_view_prevalence",
                "low_coverage",
                "low_error",
                "unknown",
            ):
                form = num if key == "common_AP" else pct
                vals.append(f"{form(a.get(key))} → {form(b.get(key))}")
            text.append("| " + " | ".join([o, c, m, *vals]) + " |")
    text += [
        "",
        "## Errors at matched coverage",
        "",
        "The two percentages below are points from the registered 1–80% grid. Every method and both reconstructions use the same number of accepted pixels per attainable view on the eight-score intersection. Rates pool errors across the accepted pixels.",
        "",
        "| Object | Sources | Requested coverage | Views attained | Accepted pixels | B-L1-transfer, old → new | MS-L1, old → new | MS-frozen, old → new | MS-ft, old → new |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lookup = {
        (r["object_id"], r["condition"], r["method"], r["version"], r["anchor"]): r
        for r in matched
        if r["domain"] == "paired_old_new"
    }
    for obj, cond in sorted(CELLS):
        for percent in (5, 20):
            anchor = f"{percent}%_V"
            sample = lookup.get((obj, cond, "MS-L1", "new", anchor))
            values = []
            for method in METHODS:
                rates = []
                for version in ("old", "new"):
                    r = lookup.get((obj, cond, method, version, anchor))
                    if sample is not None:
                        require(
                            r is not None
                            and r["accepted"] == sample["accepted"]
                            and r["risk_defined_views"] == sample["risk_defined_views"],
                            "matched-coverage populations differ across models or reconstructions",
                        )
                    rates.append(pct(r["risk"] if r is not None else None))
                values.append(" → ".join(rates))
            views = f"{sample['risk_defined_views']}/6" if sample is not None else "N/A"
            count = f"{sample['accepted']:,}" if sample is not None else "N/A"
            text.append("| " + " | ".join([obj, cond, f"{percent}%", views, count, *values]) + " |")
    text += [
        "",
        "AP ranks larger scores as greater RGB error and groups identical scores. For each reconstruction the common domain is E intersected with finite scores from all four methods. The table gives each defined view equal weight for AP and for its companion positive prevalence. The method-conditions table also gives own-domain AP and prevalence, defined-view counts and pooled numerators. A single-class or empty view has undefined AP.",
        "",
        "E contains pixels having a usable reference and a comparable target prediction. Among E, low means finite score strictly below the frozen threshold, high includes equality, and unknown means no finite score. Counts satisfy E = L + H + U and Y = LY + HY + UY. The coverage denominator V includes every reference-valid pixel. Source support and alpha describe the reconstruction only.",
        "",
        "The matched-coverage figure compares all four methods on the same pixels across old and new renders. Each view uses a fixed target fraction of V and a shared label-independent order for ties. Points pool the accepted and positive counts of views reaching the requested coverage. The per-view table gives missing points, support, cutoff tie bounds, and counts at the original MS-L1 operating coverage. These are descriptive comparisons of a development object and a previously exposed regression object.",
        "",
        "[Matched-coverage figure](matched-coverage.svg) · [Frozen rates](method-conditions.csv) · [All views](method-views.csv) · [Matched points and tie bounds](matched-coverage.csv) · [Source support](source-support.csv) · [Unknown reasons](unknown-reasons.csv) · [Provenance](evidence.json)",
    ]
    (out / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def load_set(roots, specs, schema, registration):
    results = []
    for pin in specs:
        _, path, doc = original(roots, pin)
        require(
            doc["schema"] == schema
            and doc["run_id"] == pin["run_id"]
            and doc["protocol_sha256"] == registration["protocol_sha256"],
            "different study or run identity",
        )
        results.append((pin, path.parent, doc))
    return results


def add_board_index(result, category, obj, cond, name, spec, root, parent):
    source = verified_file(root, name, spec)
    result.append(
        {
            "owner": category,
            "object_id": obj,
            "condition": cond,
            "run_id": parent["run_id"],
            "manifest_sha256": parent["manifest_sha256"],
            "member": "payload/" + name,
            "sha256": file_hash(source),
            "bytes": source.stat().st_size,
        }
    )


def validate_baselines(quality, baseline_runs):
    lookup = {}
    for _, _, record in baseline_runs:
        require(
            record["model_calls"] == 0 and record["new_optimizations"] == 0, "invalid source-only comparator"
        )
        for row in record["cells"]:
            key = (row["object_id"], row["condition"])
            require(key not in lookup, "duplicate baseline condition")
            lookup[key] = row
    require(set(lookup) == CELLS | {(SYNTH, "M4")}, "incomplete original baseline roster")
    for q in quality:
        old = lookup[q["object_id"], q["condition"]]
        require(
            old["source"] == q["old_source"] and old["allowed_source_hashes"] == q["original_source_hashes"],
            "historical source measurements differ from the independently sealed original",
        )
        for v in VIEWS:
            target = old["target"][v]
            require(
                target["reference_sha256"] == q["original_target_hashes"][v]
                and target["old_render_sha256"] == q["old_render_hashes"][v]
                and target["full_crop"] == q["old_and_zero_training_targets"][v]["full_crop"]
                and old["angular_copy_choice"][v]["source"] == q["angular_source_choices"][v]["source_id"],
                "different original or zero-fit measurement on the same target",
            )


def inspect_visual_notes(review, quality, selected_step, output):
    expected = {
        (q["object_id"], q["condition"]): q
        for q in quality
        if q["steps"][str(selected_step)]["status"] == "measured"
    }
    seen = set()
    lines = [
        "# Inspection of all six target views",
        "",
        "Each observation refers to the archived source and target comparison boards at the selected checkpoint. "
        "The original target photographs and comparison images remain in the private research archive.",
        "",
    ]
    for cell in review["cells"]:
        key = (cell["object_id"], cell["condition"])
        require(key in expected and key not in seen, "duplicate or unobserved visual assessment")
        seen.add(key)
        q = expected[key]
        for role in ("source", "target"):
            name = f"boards/{role}-{selected_step:04}.png"
            require(
                cell[role + "_board_sha256"] == q["boards"][name]["sha256"],
                "visual assessment uses another board",
            )
        require(
            cell["step"] == selected_step
            and set(cell["views"]) == set(VIEWS)
            and all(
                isinstance(s, str) and s.strip()
                for s in [cell["source_fit"], cell["interpretation"], *cell["views"].values()]
            ),
            "an observation is missing its actual source or target inspection",
        )
        lines.extend(
            [
                f"## {key[0]} · {key[1]}",
                "",
                cell["interpretation"],
                "",
                cell["source_fit"],
                "",
                "| View | Appearance in the matched crops |",
                "|---|---|",
            ]
        )
        lines.extend(f"| {v} | {cell['views'][v]} |" for v in VIEWS)
        lines.extend(
            [
                "",
                f"Target board SHA-256 `{cell['target_board_sha256']}`. "
                f"Source board SHA-256 `{cell['source_board_sha256']}`.",
                "",
            ]
        )
    require(seen == set(expected), "not all measured conditions have a recorded visual inspection")
    (output / "visual-readback.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def gallery(dest, copies):
    folder = dest / "private"
    folder.mkdir()
    sections = defaultdict(list)
    for obj, cond, label, name, path, sha in copies:
        target = folder / obj / cond / name
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(path, target)
        require(file_hash(target) == sha, "private gallery copy differs from its original")
        href = html.escape(target.relative_to(folder).as_posix(), quote=True)
        sections[obj, cond].append(f'<li><a href="{href}">{html.escape(label)}</a></li>')
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Recovery comparison boards</title>",
        "<style>body{font:16px/1.55 system-ui;color:#20353c;max-width:68rem;margin:2rem auto;padding:0 1rem}"
        "ul{columns:2}a{color:#075d61}h2{margin:2rem 0 .4rem}@media(max-width:40rem){ul{columns:1}}</style>",
        "<h1>Recovery comparison boards</h1>",
        "<p>Open a source or target sheet to compare both reconstructions on the same native pixels. "
        "The risk sheets show one target camera and all four models with the released thresholds. "
        "DTU photographs and these comparison sheets are for private research use.</p>",
    ]
    for (obj, cond), links in sorted(sections.items()):
        parts.extend([f"<h2>{html.escape(obj)} · {html.escape(cond)}</h2>", "<ul>", *links, "</ul>"])
    parts.append("</html>")
    (folder / "index.html").write_text("\n".join(parts) + "\n", encoding="utf-8")


def work(roots, req, out, rid):
    require(req["schema"] == "el-recovery-report-request-1", "wrong report request")
    reg = authority(roots, req)
    freeze = permit_frozen(roots, req)
    _, _, sel = original(roots, req["selection"])
    require(
        req["selection"] == freeze["registered_selection"] and sel == freeze["selection"],
        "report selection differs from final freeze",
    )
    qrecords = load_set(roots, req["quality"], "el-recovery-quality-1", reg)
    rrecords = load_set(roots, req["risk"], "el-recovery-frozen-risk-evaluation-1", reg)
    quality = [r[2] for r in qrecords]
    risk = [r[2] for r in rrecords]
    validate_baselines(quality, load_set(roots, req["baselines"], "el-recovery-original-rgb-1", reg))
    require(
        len({(r["object_id"], r["condition"], r["recipe_id"]) for r in quality}) == len(quality),
        "duplicate quality trajectory",
    )
    selected = [r for r in quality if r["recipe_sha256"] == freeze["recipe_sha256"]]
    require(
        {(q["object_id"], q["condition"]) for q in selected} == CELLS | {(SYNTH, "M4")},
        "the frozen recipe needs an original disposition in the synthetic control and four DTU cells",
    )
    rows, winner = selection([d for d in quality if d["object_id"] == "set1"])
    require(
        rows == sel["candidates"]
        and winner == sel["winner"]
        and {p["sha256"] for p, _, d in qrecords if d["object_id"] == "set1"}
        == {p["sha256"] for p in sel["quality"]},
        "different original development trajectories or decision",
    )
    successful = {
        (q["object_id"], q["condition"])
        for q in selected
        if q["object_id"] != SYNTH
        and q["execution_status"] == "completed"
        and q["steps"][str(freeze["stop_step"])]["status"] == "measured"
    }
    require(
        {(d["object_id"], d["condition"]) for d in risk} == successful and len(risk) == len(successful),
        "a frozen-endpoint risk cell is missing or duplicated",
    )
    require(
        all(
            r["input_pins"]["freeze"] == req["freeze"]
            and r["step"] == freeze["stop_step"]
            and r["thresholds"] == {m: freeze["models"][m]["threshold"] for m in METHODS}
            for r in risk
        ),
        "different model threshold or freeze in a risk cell",
    )
    boards = []
    copies = []
    for pin, base, q in qrecords:
        for name, spec in q["boards"].items():
            add_board_index(boards, "quality", q["object_id"], q["condition"], name, spec, base, pin)
        if q["recipe_sha256"] == freeze["recipe_sha256"]:
            for role in ("source", "target"):
                name = f"boards/{role}-{freeze['stop_step']:04}.png"
                if name in q["boards"]:
                    copies.append(
                        (
                            q["object_id"],
                            q["condition"],
                            role.capitalize() + " · reference, D1 and recovery",
                            "quality-" + role + ".png",
                            base / name,
                            q["boards"][name]["sha256"],
                        )
                    )
    for pin, base, r in rrecords:
        for v in r["rows"]:
            spec = v["new_risk_board"]
            add_board_index(
                boards, "risk-recovery", r["object_id"], r["condition"], spec["path"], spec, base, pin
            )
            copies.append(
                (
                    r["object_id"],
                    r["condition"],
                    v["view"] + " · recovery risk",
                    "risk-recovery-" + v["view"] + ".png",
                    base / spec["path"],
                    spec["sha256"],
                )
            )
    qa = []
    for pin in req["audits"]:
        _, _, d = original(roots, pin)
        require(
            d["schema"] == "el-recovery-independent-audit-1"
            and d["passed"]
            and d["freeze"] == req["freeze"]
            and d["evaluation"] in req["risk"],
            "independent raw-array audit belongs to different results",
        )
        hp = d["historical_evaluation"]
        _, old_path, old_eval = original(roots, hp)
        require(
            (old_eval["object_id"], old_eval["condition"]) == (d["object_id"], d["condition"]),
            "historical risk boards belong to a different scientific cell",
        )
        for view in old_eval["rows"]:
            spec = view["board"]
            add_board_index(
                boards, "risk-D1", d["object_id"], d["condition"], spec["path"], spec, old_path.parent, hp
            )
            copies.append(
                (
                    d["object_id"],
                    d["condition"],
                    view["view"] + " · original D1 risk",
                    "risk-D1-" + view["view"] + ".png",
                    old_path.parent / spec["path"],
                    spec["sha256"],
                )
            )
        qa.append(d)
    require(
        {d["evaluation"]["sha256"] for d in qa} == {p["sha256"] for p in req["risk"]}
        and len(qa) == len(risk),
        "missing independent original-array readback",
    )
    with new_output(out) as dest:
        gallery(dest, copies)
        qdir = dest / "public" / "quality"
        rdir = dest / "public" / "risk"
        qdir.mkdir(parents=True)
        rdir.mkdir()
        qt, missing = quality_tables(quality)
        for name, data in qt.items():
            csv_file(qdir, name, data)
        candidates = [
            {
                **{
                    k: v
                    for k, v in row.items()
                    if k not in ("condition_means", "checkpoint_sha256", "phase", "condition_exclusions")
                },
                **{c + "_MAE": (row["condition_means"] or {}).get(c) for c in ("M2", "M4")},
                **{c + "_checkpoint_sha256": row["checkpoint_sha256"].get(c) for c in ("M2", "M4")},
                **{c + "_phase": row["phase"].get(c) for c in ("M2", "M4")},
                **{c + "_exclusion": row["condition_exclusions"].get(c) for c in ("M2", "M4")},
                "selected": row == winner,
            }
            for row in rows
        ]
        csv_file(qdir, "stopping-candidates", candidates)
        quality_curves(qt["quality-curves"], qdir / "quality-curves.svg")
        quality_text(qdir, quality, sel, qt["comparator-directions"])
        inspect_visual_notes(req["visual_readback"], selected, freeze["stop_step"], qdir)
        write_json(
            qdir / "evidence.json",
            {
                "schema": "el-recovery-quality-report-1",
                "report_run_id": rid,
                "study": req["protocol"],
                "final_freeze": req["freeze"],
                "selection": req["selection"],
                "quality": req["quality"],
                "original_baselines": req["baselines"],
                "missing": missing,
                "original_inputs": [
                    {"object_id": q["object_id"], "condition": q["condition"], "inputs": q["input_pins"]}
                    for q in quality
                ],
            },
        )
        rt = risk_tables(risk)
        for name, data in rt.items():
            csv_file(rdir, name, data)
        risk_curves(rt["matched-conditions"], rdir / "matched-coverage.svg")
        risk_text(rdir, rt["method-conditions"], rt["matched-conditions"])
        write_json(
            rdir / "evidence.json",
            {
                "schema": "el-recovery-risk-report-1",
                "report_run_id": rid,
                "study": req["protocol"],
                "final_freeze": req["freeze"],
                "risk": req["risk"],
                "audits": req["audits"],
                "release": freeze["release"],
                "models": freeze["models"],
                "scopes": freeze["scopes"],
                "scoring_sources": [
                    {
                        "object_id": r["object_id"],
                        "condition": r["condition"],
                        "original_case": r["input_pins"]["original_case"],
                        "original_scores": r["input_pins"]["old_scores"],
                        "recovery_case": r["input_pins"]["new_case"],
                        "recovery_scores": r["input_pins"]["new_scores"],
                        "reference": r["input_pins"]["references"],
                    }
                    for r in risk
                ],
                "unavailable": [
                    {
                        "object_id": q["object_id"],
                        "condition": q["condition"],
                        "status": q["execution_status"],
                        "reason": q["output_reason"],
                        "original_reference_V": sum(
                            q["old_and_zero_training_targets"][v]["full_crop"]["old"]["V"] for v in VIEWS
                        ),
                    }
                    for q in selected
                    if q["object_id"] != SYNTH and (q["object_id"], q["condition"]) not in successful
                ],
            },
        )
        write_json(
            dest / "private-boards.json", {"schema": "el-recovery-private-board-index-1", "boards": boards}
        )
        files = {
            p.relative_to(dest).as_posix(): {"sha256": file_hash(p), "bytes": p.stat().st_size}
            for p in sorted((dest / "public").rglob("*"))
            if p.is_file()
        }
        summary = {
            "schema": "el-recovery-final-report-1",
            "run_id": rid,
            "inputs": req,
            "quality_trajectories": len(quality),
            "successful_risk_cells": len(risk),
            "planned_risk_cells": len(CELLS),
            "compared_versions": 2,
            "audited_cells": len(qa),
            "private_board_count": len(boards),
            "private_gallery": {
                "path": "private/index.html",
                "sha256": file_hash(dest / "private/index.html"),
                "linked_images": len(copies),
            },
            "public_files": files,
        }
        write_json(dest / "report.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "shelf", "request", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    result = work(
        (args.store, args.shelf),
        signed(args.request, args.request_sha256),
        args.output,
        os.environ["EL_RUN_ID"],
    )
    print(json.dumps({k: result[k] for k in ("run_id", "quality_trajectories", "audited_cells")}))
