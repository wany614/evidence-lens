"""Build private case-series tables, figures and narrative from an explicit manifest."""

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ._bundle import LABEL, load, local_path
from .formal_inputs import (
    CONDITIONS,
    PinnedRun,
    Staged,
    protocol,
    require,
    signature,
)
from .formal_metrics import METHODS, OUTCOMES, REGIONS, SYSTEM, pool_family, ranking_ap
from .formal_roster import measured_pointer, pinned_file
from .scoring_io import TARGETS, file_hash, new_output, write_json

COLORS = {
    "constant": "#33383d",
    "1-alpha": "#c07728",
    "L0": "#9052a6",
    "L1-full": "#007b85",
    "L1-geometry": "#4d6eb0",
}
FAMILY_COLORS = ("#147b80", "#a75b30", "#7158a0", "#355f95")
COLUMNS = (
    "rgb_mae",
    "rgb_error_incidence",
    "confirmed_error_fraction_of_V",
    "evaluable_fraction",
    "common_coverage",
    "common_prevalence",
    "invalid_fraction",
    "common_ap",
    "low_risk_error",
    "low_coverage",
    "unknown_fraction",
    "detection",
    "warning_precision",
    "warning_coverage",
    "unknown_error_incidence",
    "own_ap",
)


def assemble(store, selection, destination, checkout):
    """Stage only explicit runs. Stage status and optional evidence are independently pinned."""
    selection = Path(selection).resolve()
    sel = load(selection)
    require(
        sel.get("schema") == "el-formal-report-selection-1"
        and sel["kind"] in ("heldout_study", "synthetic_fixture"),
        "invalid report selection",
    )
    pfiles = sel["protocol_files"]
    require(set(pfiles) == {"catalog", "study", "method_freeze"}, "report requires three protocol pins")
    sources = {}
    for key, entry in pfiles.items():
        path = (selection.parent / entry["path"]).resolve()
        require(file_hash(path) == entry["sha256"], "mismatched report protocol file")
        sources["protocol/" + key + ".json"] = path
    catalog, method = load(sources["protocol/catalog.json"]), load(sources["protocol/method_freeze.json"])
    require(
        method["catalog_sha256"] == pfiles["catalog"]["sha256"]
        and method["study_sha256"] == pfiles["study"]["sha256"]
        and method["comparison_protocol_sha256"] == protocol(checkout)["comparison_protocol_sha256"],
        "report differs from P5",
    )
    families = [r["scene_family_id"] for r in catalog["families"] if r["split"] == "test"]
    rows = sel["trials"]
    require(
        len(families) == 4
        and len(rows) == 12
        and {(r["scene_family_id"], r["condition_id"]) for r in rows}
        == {(f, c) for f in families for c in CONDITIONS},
        "report requires all 12 planned rows including retained failures",
    )
    runs, trial_pins = {}, []
    for r in rows:
        pin = {k: r[k] for k in ("run_id", "manifest_sha256")}
        parent = PinnedRun(store, pin)
        f, c = r["scene_family_id"], r["condition_id"]
        require(f == local_path(f) and "/" not in f and c in CONDITIONS, "unsafe scientific identity")
        record = load(parent.open("payload/formal_evaluation.json"))
        require(
            parent.run["run_kind"] == "evaluation"
            and parent.run["execution_status"] == "succeeded"
            and record["scene_family_id"] == f
            and record["condition_id"] == c
            and record["kind"] == sel["kind"]
            and record["run_id"] == pin["run_id"]
            and record["comparison_protocol_sha256"] == method["comparison_protocol_sha256"]
            and record["method_freeze_sha256"] == pfiles["method_freeze"]["sha256"],
            "wrong P6 result or method",
        )
        prefix = f"trials/{f}/{c}/"
        for name in (
            "formal_evaluation.json",
            "curves/family.npz",
            "board/board_v01.png",
            "board/board_v01.json",
            "reason_visibility.json",
        ):
            path = parent.open("payload/" + name)
            if name == "curves/family.npz":
                require(file_hash(path) == record["curves_sha256"][name], "curve differs from P6")
            if name == "board/board_v01.png":
                require(file_hash(path) == record["board_sha256"], "board differs from P6")
            if name == "reason_visibility.json":
                require(
                    file_hash(path) == record["reason_visibility_sha256"], "L0 cross-counts differ from P6"
                )
            sources[prefix + name] = path
        runs[pin["run_id"]] = parent
        trial_pins.append(dict(pin, scene_family_id=f, condition_id=c))
    # Evidence rows make all attempted/timing/recovery selections explicit. No directory scan is used.
    evidence = []
    ids = set()
    for e in sel.get("evidence", []):
        rid = e["run_id"]
        require(
            rid not in ids
            and e["purpose"]
            in {"scientific", "timing_repeat", "engineering", "transfer", "archive", "recovery", "training"},
            "invalid or duplicate evidence run",
        )
        ids.add(rid)
        parent = runs.get(rid) or PinnedRun(store, {k: e[k] for k in ("run_id", "manifest_sha256")})
        require(parent.pin["manifest_sha256"] == e["manifest_sha256"], "conflicting seal for evidence run")
        rec = dict(
            e,
            run=parent.run,
            payload_bytes=sum(
                item["bytes"] for name, item in parent.files.items() if name.startswith("payload/")
            ),
            logical_bundle_bytes=sum(item["bytes"] for item in parent.files.values()),
            environment=load(parent.open("environment.json")) if "environment.json" in parent.files else {},
        )
        evidence.append(rec)
        known = {
            "generation.json": "payload/generation.json",
            "prediction.json": "payload/prediction.json",
            "score.json": "payload/score.json",
            "l1_score.json": "payload/l1_score.json",
            "fit.json": "payload/fit.json",
            "cv_results.json": "payload/cv_results.json",
            "development.json": "payload/development.json",
            "model.json": "payload/model/model.json",
            "L1-full.json": "payload/model/L1-full.json",
            "L1-geometry.json": "payload/model/L1-geometry.json",
            "thresholds.json": "payload/model/thresholds.json",
            "feature_schema.json": "payload/model/feature_schema.json",
        }
        rec["documents"] = {k: v for k, v in known.items() if v in parent.files} | e.get("documents", {})
        for alias, rel in rec["documents"].items():
            require(
                alias == local_path(alias)
                and "/" not in alias
                and Path(alias).suffix in {".json", ".md", ".csv"},
                "invalid report evidence document",
            )
            sources[f"evidence/{rid}/{alias}"] = parent.open(rel)
    proof_rows = []
    for spec in sel.get("proofs", []):
        identifier, kind = spec["id"], spec["kind"]
        require(LABEL.fullmatch(identifier) and LABEL.fullmatch(kind), "invalid external proof identity")
        source = pinned_file(selection.parent, spec)
        relative = "proofs/" + identifier + ".json"
        require(relative not in sources, "duplicate external proof ID")
        data = load(source)
        measures = {}
        for name, pointer in spec.get("measures", {}).items():
            require(LABEL.fullmatch(name), "invalid public numeric measure name")
            measures[name] = measured_pointer(data, pointer)
        sources[relative] = source
        proof_rows.append(
            {
                "id": identifier,
                "kind": kind,
                "sha256": spec["sha256"],
                "measures": measures,
                "selectors": spec.get("measures", {}),
            }
        )
    with new_output(Path(destination)) as out:
        files = {}
        for rel, source in sources.items():
            dest = out / local_path(rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            files[rel] = signature(dest)
        result = {
            "schema": "el-formal-report-input-1",
            "kind": sel["kind"],
            "trials": trial_pins,
            "evidence": evidence,
            "proofs": proof_rows,
            "scientific_markers": sel.get("scientific_markers", []),
            "roster_sha256": sel.get("roster_sha256"),
            "inventory_scope": sel.get("inventory_scope", "explicitly listed runs only"),
            "method_freeze_sha256": pfiles["method_freeze"]["sha256"],
            "selection_sha256": file_hash(selection),
            "comparison_protocol_sha256": method["comparison_protocol_sha256"],
            "files": files,
        }
        write_json(out / "report_inputs.json", result)
    return result


def family_rows(trials):
    rows, per_view = [], []

    def extract(region, method):
        result = {k: v["value"] for k, v in region["outcomes"].items()}
        result.update({k: v["value"] for k, v in method["system"].items()})
        result.update(region["counts"], **method["counts"])
        for key, field in (("common_ap", "common_ranking"), ("own_ap", "own_ranking")):
            record = method[field]
            result[key] = ranking_ap(record) if "status" in record else record["ap"]
        return result

    for (f, c), trial in trials.items():
        for region in REGIONS:
            group = trial["family"][region]
            for m, method in group["methods"].items():
                base = {"family": f, "condition": c, "region": region, "method": m}
                rows.append(
                    base
                    | extract(group, method)
                    | {
                        "common_two_class_views": method["common_ranking"]["eligible_view_count"],
                        "own_two_class_views": method["own_ranking"]["eligible_view_count"],
                        "completed_views": trial["completed_targets"],
                        "planned_views": trial["planned_targets"],
                        "tau": trial["thresholds"][m],
                    }
                )
                for v, target in trial["views"].items():
                    vr = target["regions"][region]
                    mr = vr["methods"][m]
                    per_view.append(
                        base
                        | {
                            "camera": v,
                            "render_status": target["render_status"],
                            "failure_reason": target["failure_reason"],
                            "learned_status": target["learned_status"],
                            "common_status": mr["common_ranking"]["status"],
                            "own_status": mr["own_ranking"]["status"],
                            "S_mask": vr["S_mask_sha256"],
                            "own_mask": mr["own_mask_sha256"],
                        }
                        | extract(vr, mr)
                    )
    return rows, per_view


def describe(values):
    a = [v for v in values if v is not None]
    return {
        "eligible_family_count": len(a),
        "mean": float(np.mean(a)) if a else None,
        "median": float(np.median(a)) if a else None,
        "minimum": min(a) if a else None,
        "maximum": max(a) if a else None,
        "missing_reason": None if a else "no_eligible_families",
    }


def summarize(rows, per_view):
    valid_views = Counter()
    for row in per_view:
        for metric in (*OUTCOMES, *SYSTEM, "common_ap", "own_ap"):
            valid_views[row["condition"], row["region"], row["method"], metric] += row[metric] is not None
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"], row["region"], row["method"]].append(row)
    results = []
    for (condition, region, method), parts in groups.items():
        for metric in (*OUTCOMES, *SYSTEM, "common_ap", "own_ap"):
            results.append(
                {
                    "condition": condition,
                    "region": region,
                    "method": method,
                    "metric": metric,
                    "planned_families": len(parts),
                    "eligible_views": valid_views[condition, region, method, metric],
                    "planned_views": sum(p["planned_views"] for p in parts),
                    "eligible_families": "|".join(p["family"] for p in parts if p[metric] is not None),
                    **describe([p[metric] for p in parts]),
                }
            )
    return results


def paired_rows(trials, rows):
    lookup = {(r["family"], r["condition"], r["region"], r["method"]): r for r in rows}
    paired = []
    for (f, c, region, m), row in lookup.items():
        if c == "B0":
            continue
        b = lookup[f, "B0", region, m]
        tv, bv = trials[f, c]["views"], trials[f, "B0"]["views"]
        matched = [
            v
            for v in TARGETS
            if all(
                ranking_ap(t[v]["regions"][region]["methods"][m]["common_ranking"]) is not None
                for t in (tv, bv)
            )
        ]
        delta = {
            "family": f,
            "variant": c,
            "region": region,
            "method": m,
            "paired_camera_ids": "|".join(matched),
        }
        for metric in COLUMNS:
            a, z = row[metric], b[metric]
            if metric == "common_ap":
                values = [
                    [result[v]["regions"][region]["methods"][m]["common_ranking"]["ap"] for v in matched]
                    for result in (tv, bv)
                ]
                a, z = (float(np.mean(group)) if group else None for group in values)
            delta.update(
                {
                    metric + "_variant": a,
                    metric + "_B0": z,
                    metric + "_delta": a - z if a is not None and z is not None else None,
                }
            )
        paired.append(delta)
    return paired


def csv_write(path, rows, fields=None):
    fields = list(fields or dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def draw_curves(out, condition, families, curves):
    w, h = 1232, 956
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    font, small, title = (ImageFont.load_default(size=n) for n in (17, 14, 28))
    d.text((22, 14), f"Attainable risk and coverage  /  {condition}", font=title, fill="#202732")
    d.text(
        (22, 55),
        "One dot per occupied display pixel containing an attainable whole-tie endpoint. Outlined circles mark the frozen thresholds.",
        font=small,
        fill="#394452",
    )
    x = 24
    for name, color in COLORS.items():
        d.ellipse((x, 86, x + 10, 96), fill=color)
        d.text((x + 16, 83), name, font=small, fill="#222")
        x += 158
    # Fixed axes make the same coordinate comparable across all families and conditions.
    for i, f in enumerate(families):
        x0, y0 = 68 + (i % 2) * 609, 158 + (i // 2) * 390
        ww, hh = 501, 295
        d.text((x0, y0 - 35), f, font=font, fill="#182532")
        for j in range(5):
            z = j / 4
            x, y = x0 + round(ww * z), y0 + round(hh * (1 - z))
            d.line((x, y0, x, y0 + hh), fill="#dde0e2")
            d.line((x0, y, x0 + ww, y), fill="#dde0e2")
            d.text((x, y0 + hh + 8), f"{z:.2g}", font=small, anchor="mt", fill="#303d4a")
            d.text((x0 - 8, y), f"{z:.2g}", font=small, anchor="rm", fill="#303d4a")
        patch = np.asarray(im.crop((x0, y0, x0 + ww + 1, y0 + hh + 1))).copy()
        marks, empty_thresholds = [], []
        for method_index, (name, key) in enumerate(METHODS.items()):
            with np.load(curves[f, condition], allow_pickle=False) as data:
                n, err, V = data[key + "__accepted"], data[key + "__errors"], int(data[key + "__V"])
                j = int(data[key + "__frozen_index"])
            ok = (n > 0) & (V > 0)
            if V and ok.any():
                xx = np.rint(ww * n[ok] / V).astype(np.intp)
                yy = np.rint(hh * (1 - err[ok] / n[ok])).astype(np.intp)
                col = bytes.fromhex(COLORS[name][1:])
                patch[yy, xx] = tuple(col)
                if ok.sum() <= 3:
                    for x, y in zip(xx, yy):
                        marks.append((x0 + x, y0 + y, COLORS[name], 2))
            if n[j] and V:
                marks.append(
                    (
                        x0 + round(ww * n[j] / V),
                        y0 + round(hh * (1 - err[j] / n[j])),
                        COLORS[name],
                        4 + 2 * method_index,
                    )
                )
            else:
                empty_thresholds.append(name)
        im.paste(Image.fromarray(patch), (x0, y0))
        if empty_thresholds:
            d.text(
                (x0, y0 + hh + 29),
                "No pixels accepted at the frozen threshold for " + ", ".join(empty_thresholds),
                font=small,
                fill="#394452",
            )
        for x, y, c, r in marks:
            d.ellipse((x - r, y - r, x + r, y + r), outline=c, width=2)
    d.text(
        (23, h - 35),
        "Coverage equals accepted pixels / full V. Risk equals true errors / accepted pixels. Accept-none has undefined risk.",
        font=small,
        fill="#293a45",
    )
    im.save(out)


def draw_paired(out, families, all_rows):
    lookup = {(r["family"], r["condition"], r["method"]): r for r in all_rows if r["region"] == "all"}
    im = Image.new("RGB", (1190, 850), "white")
    d = ImageDraw.Draw(im)
    font, small, title = (ImageFont.load_default(size=n) for n in (18, 14, 28))
    d.text((24, 15), "The same four families under three conditions", font=title, fill="#24303b")
    for i, f in enumerate(families):
        x = 24 + (i % 2) * 554
        y = 64 + (i // 2) * 24
        d.line((x, y + 8, x + 19, y + 8), fill=FAMILY_COLORS[i], width=3)
        d.text((x + 26, y), f, font=small, fill="#333")
    measures = (
        ("confirmed_error_fraction_of_V", "Confirmed RGB errors / full V"),
        ("rgb_mae", "Display-sRGB mean absolute error"),
        ("common_ap", "L1-full AP on common S"),
        ("low_coverage", "L1-full accepted / V"),
    )
    for i, (metric, label) in enumerate(measures):
        x, y = 73 + (i % 2) * 590, 155 + (i // 2) * 328
        w, h = 461, 227
        d.text((x, y - 32), label, font=font, fill="#182532")
        peak = max([lookup[f, c, "L1-full"][metric] or 0 for f in families for c in CONDITIONS])
        top = max(0.01, min(1.0, np.ceil(peak * 10.0) / 10.0)) if metric == "rgb_mae" else 1.0
        for j in range(5):
            yy = y + round(h * (1 - j / 4))
            d.line((x, yy, x + w, yy), fill="#e0e1e3")
            d.text((x - 6, yy), f"{top * j / 4:.2f}", font=small, anchor="rm", fill="#34404f")
        for j, c in enumerate(CONDITIONS):
            d.text((x + j * w / 2, y + h + 8), c, font=small, anchor="mt", fill="#34404f")
        for j, f in enumerate(families):
            prev = None
            for column, c in enumerate(CONDITIONS):
                value = lookup[f, c, "L1-full"][metric]
                point = (
                    (int(x + column * w / 2), int(y + h * (1 - value / top))) if value is not None else None
                )
                if point:
                    if prev:
                        d.line((*prev, *point), fill=FAMILY_COLORS[j], width=2)
                    xx, yy = point
                    d.ellipse((xx - 4, yy - 4, xx + 4, yy + 4), fill=FAMILY_COLORS[j])
                prev = point
    d.text(
        (24, 817),
        "Family identities connect across conditions. Empty positions indicate an undefined quantity. Each axis names its denominator.",
        font=small,
        fill="#343e49",
    )
    im.save(out)


def draw_architecture(path):
    i = Image.new("RGB", (1190, 357), "white")
    d = ImageDraw.Draw(i)
    t, f, s = (ImageFont.load_default(size=n) for n in (26, 18, 15))
    d.text((22, 14), "What each component can read", font=t, fill="#263546")
    boxes = [
        (24, 82, 290, 184, "Condition input + cameras", "One authored frame and known cameras"),
        (331, 82, 597, 184, "SHARP + Gaussian renderer", "Target RGB, opacity, camera-Z"),
        (641, 82, 919, 184, "L0 + independently fitted L1", "Risk maps from permitted inputs"),
        (24, 242, 391, 325, "Private original Blender scene", "Reference RGB, surface and visibility"),
        (685, 242, 1159, 325, "Independent evaluation", "RGB labels, ranking and full-V system measures"),
    ]
    for x, y, x2, y2, title, body in boxes:
        d.rounded_rectangle(
            (x, y, x2, y2), radius=7, fill="#edf2f2" if y < 200 else "#f6efe5", outline="#657681"
        )
        d.text((x + 10, y + 11), title, font=f, fill="#263543")
        d.text((x + 10, y + 47), body, font=s, fill="#3c4b52")
    for prev, next_x, y in ((290, 331, 130), (597, 641, 130), (391, 685, 280)):
        d.line((prev, y, next_x - 5, y), fill="#40596a", width=2)
        d.polygon(((next_x - 5, y - 5), (next_x - 5, y + 5), (next_x, y)), fill="#40596a")
    d.line((775, 185, 775, 235), fill="#40596a", width=2)
    d.polygon(((770, 235), (780, 235), (775, 241)), fill="#40596a")
    d.line((500, 185, 500, 213, 681, 213, 681, 252), fill="#40596a", width=2)
    d.text((600, 207), "prediction", font=s, fill="#354554", anchor="ms")
    i.save(path)


def evidence_tables(staged, rows):
    stages, phases, documents = [], [], {}
    for e in staged.meta["evidence"]:
        r = e["run"]
        b = {
            "run_id": r["run_id"],
            "purpose": e["purpose"],
            "family": r.get("scene_family_id"),
            "condition": r.get("condition_id"),
            "trial_id": r["trial_id"],
            "kind": r["run_kind"],
            "status": r["execution_status"],
            "retry_of": r.get("retry_of_run_id"),
            "failure_reason": r.get("failure_reason"),
            "elapsed_seconds": r.get("elapsed_seconds_monotonic"),
            "peak_sampled_rss_bytes": r.get("sampled_peak_tree_rss_bytes"),
            "logical_bundle_bytes": e["logical_bundle_bytes"],
            "payload_bytes": e["payload_bytes"],
            "manifest_sha256": e["manifest_sha256"],
            "platform": e["environment"].get("platform"),
            "timing_index": e.get("timing_index"),
            "startup": e.get("startup"),
            "source": "recorder",
        }
        stages.append(b)
        for alias in e["documents"]:
            rel = f"evidence/{r['run_id']}/{alias}"
            path = staged.open(rel)
            if path.suffix != ".json":
                continue
            data = load(path)
            documents.setdefault(data.get("schema", alias), []).append((rel, data))
            for name, phase in data.get("phases", {}).items():
                if isinstance(phase, dict):
                    phases.append(
                        {
                            **b,
                            "phase": name,
                            "camera": None,
                            "seconds": phase.get("seconds"),
                            "cuda_peak_allocated_bytes": phase.get("cuda_allocated_peak_bytes"),
                            "cuda_peak_reserved_bytes": phase.get("cuda_reserved_peak_bytes"),
                        }
                    )
            if data.get("schema") == "el-generation-1":
                for name in ("preprocess_seconds", "forward_seconds", "unproject_seconds"):
                    if data.get(name) is not None:
                        phases.append({**b, "phase": name, "camera": None, "seconds": data[name]})
            if data.get("schema") == "el-l1-cv-1":
                for method, rec in data["models"].items():
                    for C, candidate in rec.get("candidates", {}).items():
                        for fold in candidate.get("folds", []):
                            phases.append(
                                {
                                    **b,
                                    "phase": "cv_fit",
                                    "method": method,
                                    "C": float(C),
                                    "fold": fold["fold"],
                                    "camera": None,
                                    "seconds": fold.get("seconds"),
                                    "iterations": fold.get("iterations"),
                                }
                            )
            if data.get("schema") in ("el-score-1", "el-l1-score-1"):
                for v, vr in data["views"].items():
                    for name, seconds in vr.items():
                        if name.endswith("_seconds"):
                            phases.append({**b, "phase": name, "camera": v, "seconds": seconds})
            if data.get("schema") == "el-gaussian-derivation-1":
                phases.append(
                    {
                        **b,
                        "phase": "FP16 colour round trip and file verification",
                        "camera": None,
                        "seconds": data.get("derivation_seconds"),
                    }
                )
    return stages, phases, documents


def q1_records(trials, families):
    detail, summary, ply = [], [], []
    for f in families:
        tr = trials[f, "Q1"]
        paired = [v["paired_Q1_minus_B0"] for v in tr["views"].values()]
        available = [r for r in paired if r["status"] == "compared"]
        n = sum(r["paired_valid_rgb"] for r in available)
        summary.append(
            {
                "family": f,
                "paired_views": len(available),
                "total_views": len(paired),
                "paired_rgb_pixels": n,
                "mean_Q1_B0_rgb_response": sum(r["rgb_abs_sum"] for r in available) / n if n else None,
                "changed_rgb_pixels": sum(r["rgb_changed_pixel_count"] for r in available),
                "max_rgb_pixel_response": max(
                    (r["rgb_max_pixel_mae"] for r in available if r["rgb_max_pixel_mae"] is not None),
                    default=None,
                ),
                "geometry_bitwise_equal_views": sum(r["geometry_bitwise_equal"] for r in available),
                "geometry_changed_values": sum(
                    field["bitwise_different"]
                    for r in available
                    for k, field in r["fields"].items()
                    if k != "L1-full"
                ),
                "full_score_changed_values": sum(
                    r["fields"]["L1-full"]["bitwise_different"] for r in available
                ),
                "true_Q1_reference_MAE": tr["family"]["all"]["outcomes"]["rgb_mae"]["value"],
            }
        )
        self_render = tr.get("source_Q1_minus_B0")
        if self_render and self_render["status"] == "compared":
            for field, entry in self_render["fields"].items():
                detail.append(
                    {
                        "family": f,
                        "camera": "source",
                        "field": field,
                        "status": "compared",
                        "reason": None,
                        **entry,
                    }
                )
        else:
            detail.append(
                {
                    "family": f,
                    "camera": "source",
                    "field": None,
                    "status": "unavailable",
                    "reason": self_render["reason"] if self_render else "source diagnostic unavailable",
                }
            )
        summary[-1].update(
            {
                "source_geometry_bitwise_equal": self_render["geometry_bitwise_equal"]
                if self_render and self_render["status"] == "compared"
                else None,
                "source_geometry_changed_values": sum(
                    v["bitwise_different"] for v in self_render["fields"].values()
                )
                if self_render and self_render["status"] == "compared"
                else None,
            }
        )
        for v, vr in tr["views"].items():
            r = vr["paired_Q1_minus_B0"]
            if r["status"] != "compared":
                detail.append(
                    {"family": f, "camera": v, "field": None, "status": r["status"], "reason": r["reason"]}
                )
                continue
            for field, d in r["fields"].items():
                detail.append(
                    {"family": f, "camera": v, "field": field, "status": "compared", "reason": None, **d}
                )
        if tr.get("intervention"):
            change = tr["intervention"]
            for field, v in change.get("schema_fields", {}).items():
                ply.append(
                    {
                        "family": f,
                        "field": field,
                        "protected": field not in change["colour_fields"],
                        "exact_equal": v["input_sha256"] == v["output_sha256"],
                        **v,
                    }
                )
    return detail, summary, ply


def paired_costs(trials, families, stages):
    by_run = {r["run_id"]: r for r in stages}
    rows = []
    for f in families:
        base = trials[f, "B0"]

        def origin_generation(t):
            return t["parents"].get("generation", {}).get("run_id") or t.get("generation_run_id")

        for condition in CONDITIONS:
            trial = trials[f, condition]
            gen = origin_generation(base if condition == "Q1" else trial)
            selected = {
                "generation": gen,
                "preparation": None,
                "render": None,
                "rules_features": None,
                "learned_features_and_scores": None,
                "label_evaluation": None,
                "five_method_comparison": trial["run_id"],
            }
            for name, role in (
                ("preparation", "preparation"),
                ("render", "prediction"),
                ("rules_features", "rules"),
                ("learned_features_and_scores", "learned"),
                ("label_evaluation", "labels"),
            ):
                selected[name] = trial["parents"].get(role, {}).get("run_id")
            if condition == "Q1" and selected["preparation"] is None:
                selected["preparation"] = trial["parents"].get("derivation", {}).get("run_id")
            row = {
                "family": f,
                "condition": condition,
                "generation_cost_basis": "inherited B0 generation" if condition == "Q1" else "native forward",
                "generation_run_id": gen,
            }
            for name, rid in selected.items():
                rec = by_run.get(rid, {})
                row[name + "_seconds"] = rec.get("elapsed_seconds")
                row[name + "_missing_reason"] = (
                    None
                    if rec.get("elapsed_seconds") is not None
                    else (
                        "no_preparation_in_B0"
                        if condition == "B0" and name == "preparation"
                        else "selected_run_absent_from_inventory"
                    )
                )
                row[name + "_run_id"] = rid
            rows.append(row)
    for row in rows:
        b = next(r for r in rows if r["family"] == row["family"] and r["condition"] == "B0")
        for stage in (
            "render",
            "rules_features",
            "learned_features_and_scores",
            "label_evaluation",
            "five_method_comparison",
        ):
            current, original = row[stage + "_seconds"], b[stage + "_seconds"]
            row[stage + "_delta_B0_seconds"] = (
                current - original if current is not None and original is not None else None
            )
        row["native_generation_delta_B0_seconds"] = (
            row["generation_seconds"] - b["generation_seconds"]
            if row["condition"] != "Q1"
            and row["generation_seconds"] is not None
            and b["generation_seconds"] is not None
            else None
        )
    return rows


def format_num(value):
    return "N/A" if value is None else f"{value:.4g}"


def render_report(out, s, trials, families, rows, summary, paired, q1, stages, costs, docs, safety):
    catalog, study, method = (
        s.json("protocol/catalog.json"),
        s.json("protocol/study.json"),
        s.json("protocol/method_freeze.json"),
    )
    main = [r for r in rows if r["region"] == "all"]
    location = out / "report.md"

    def table(headers, values):
        return [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *[
                "| " + " | ".join(str(x).replace("|", " / ").replace("\n", " ") for x in row) + " |"
                for row in values
            ],
            "",
        ]

    def avg(c, r, m, k):
        return next(
            x
            for x in summary
            if x["condition"] == c and x["region"] == r and x["method"] == m and x["metric"] == k
        )

    md = [
        "# Evidence Lens",
        "",
        "Local risk of display-sRGB error in single-image Gaussian reconstruction",
        "",
    ]
    if s.meta["kind"] == "synthetic_fixture":
        md += [
            "**Synthetic software validation.** All following values come from deterministic test fixtures.",
            "",
        ]
    complete = sum(t["completed_targets"] for t in trials.values())
    full, standard = avg("B0", "all", "L1-full", "common_ap"), avg("B0", "all", "constant", "common_ap")
    fit_docs = docs.get("el-model-fit-1", [])
    fit = fit_docs[0][1] if len(fit_docs) == 1 else {}
    md += [
        "## Abstract",
        "",
        f"The held-out matrix contains four authored static opaque families, 12 scientific conditions and {complete} completed target images out of 72 planned. The five comparisons use a frozen training-incidence baseline, inverse opacity, L0 rules, L1-full and L1-geometry. Under B0, family-mean common-mask AP was {format_num(full['mean'])} for L1-full and {format_num(standard['mean'])} for the constant baseline across {full['eligible_family_count']} eligible families. B0 L1-full admitted {format_num(avg('B0', 'all', 'L1-full', 'low_coverage')['mean'])} of valid-reference pixels at the frozen threshold. The error fraction among admitted pixels was {format_num(avg('B0', 'all', 'L1-full', 'low_risk_error')['mean'])}.",
        "",
        "## Research question and registered predictions",
        "",
        "Can evidence available from a single source image and its predicted views locate novel-view RGB errors in held-out layouts? The source image, camera matrices and reconstruction passes enter the predictor. Blender target answers enter the private evaluator.",
        "",
        "![Information flow](figures/architecture.png)",
        "",
    ]
    md += table(
        ["Before acquisition", "What would contradict it"],
        [(p["expectation"], p["falsifier"]) for p in study.get("predictions_before_formal_acquisition", [])],
    )
    md += [
        "## Dataset, outcomes and methods",
        "",
        f"The frozen catalog assigns {sum(x['split'] == 'train' for x in catalog['families'])} families to training, {sum(x['split'] == 'development' for x in catalog['families'])} to development and four to final testing. Every condition retains the six fixed target cameras. B0 uses an original source. R1 halves its width and height and runs the generator again. Q1 rounds the selected B0 Gaussian colour coefficients to binary16, restores binary32 and rerenders the seven views.",
        "",
        "For finite reference and prediction RGB, the per-pixel absolute channel differences are averaged in display-sRGB units. An error requires this average to exceed 0.05. The sensitivity tables also retain 0.025 and 0.10. A rendered-depth error exceeds the larger of 0.02 m and 2% of reference camera-Z on the supported surface-depth mask. Signed depth and relative-depth quantiles are archived with each trial.",
        "",
        "V includes valid reference RGB, including valid background. E additionally requires valid prediction RGB. Each method partitions E into low risk, high risk and unknown. Confirmed high-error area over V and error incidence over E have their own columns. Low coverage divides by V and detection divides by every error on E, including errors in unknown areas.",
        "",
        "The public L1 mapping uses a logistic score formed from geometry, opacity, motion, correspondence and RGB features. L1-geometry removes the three appearance measurements and their missing-data indicators. Both models retain identical eligibility. Each available score below its development threshold is low risk. L0 uses its fixed rule states. Thresholds govern risk decisions. The scores rank pixels within each target, and their probability calibration remains unmeasured.",
        "",
    ]
    marked = {(r["scene_family_id"], r["condition_id"]) for r in s.meta.get("scientific_markers", [])}
    completion_rows = []
    for condition in CONDITIONS:
        planned = sum(condition in r.get("trials", {}) for r in catalog["families"])
        total = sum(t["completed_targets"] for (family, c), t in trials.items() if c == condition)
        attempts = [r for r in stages if r["purpose"] == "scientific" and r["condition"] == condition]
        completion_rows.append(
            (
                condition,
                planned,
                sum(c == condition for _, c in marked),
                total,
                24 - total,
                len(attempts),
                sum(r["status"] != "succeeded" for r in attempts),
                sum(bool(r["retry_of"]) for r in attempts),
            )
        )
    md += table(
        [
            "Condition",
            "Planned families",
            "Canonical quality selections",
            "Completed held-out targets",
            "Missing held-out targets",
            "Listed science-stage runs",
            "Failed stage runs",
            "Retry runs",
        ],
        completion_rows,
    )
    if fit:
        md += [
            f"The archived fit contains {fit.get('rows')} sampled rows from {fit.get('families')} B0 training families and {fit.get('positives')} positive labels. The maximum NumPy-versus-training-library score differences are {fit.get('max_sklearn_numpy_error')}.",
            "",
        ]
    else:
        md += [
            "The explicit report evidence inventory contains no matching fitted-model receipt. The frozen package identity and applied thresholds come from the P6 trials.",
            "",
        ]
    cvs = docs.get("el-l1-cv-1", [])
    if cvs:
        cvpath, cv = cvs[0]
        md += table(
            ["Model", "Chosen C", "Grouped candidates"],
            [
                (name, row.get("selected_C"), ", ".join(str(c) for c in cv["C_grid"]))
                for name, row in cv["models"].items()
            ],
        )
        md += [f"[Grouped cross-validation scores and fit diagnostics](numeric/{cvpath})", ""]
    frozen_parameters = [
        (path, data)
        for path, data in docs.get("el-l1-coefficients-1", [])
        if path.startswith("evidence/" + method["model_id"] + "/")
    ]
    for path, data in frozen_parameters:
        md += [
            f"{data['method']} uses {len(data['coefficients'])} fitted coefficients and intercept {format_num(data['intercept'])}. [Coefficients, feature order, imputation and scaling](numeric/{path})",
            "",
        ]
    md += table(
        ["Method", "Frozen operating threshold"],
        [(m, format_num(trials[families[0], "B0"]["thresholds"][m])) for m in METHODS],
    )
    md += [
        "Training samples were uniform among eligible pixels, capped per target. Training families received equal total weight and their nonempty views received equal shares. Grouped cross-validation selected the penalty within training. B0 development families determined the operating thresholds. The full split, camera matrix conventions and protocol are linked in the source manifest.",
        "",
        "## System performance and ranking",
        "",
        "Counts were pooled across six planned views within a family. The summary gives families equal weight. AP first uses whole score ties on the same five-method intersection S, then averages eligible two-class views and finally eligible families. The diagnostics tables preserve each method’s own mask, the positives excluded from S, class counts and every excluded single-class view.",
        "",
    ]
    md += table(
        [
            "Condition",
            "Method",
            "Eligible families",
            "AP mean",
            "Median",
            "Min–max",
            "Error / E",
            "S / V",
            "Low error",
            "Low / V",
            "U / E",
            "Invalid / V",
        ],
        [
            (
                c,
                m,
                avg(c, "all", m, "common_ap")["eligible_family_count"],
                format_num(avg(c, "all", m, "common_ap")["mean"]),
                format_num(avg(c, "all", m, "common_ap")["median"]),
                f"{format_num(avg(c, 'all', m, 'common_ap')['minimum'])}–{format_num(avg(c, 'all', m, 'common_ap')['maximum'])}",
                *[
                    format_num(avg(c, "all", m, k)["mean"])
                    for k in (
                        "rgb_error_incidence",
                        "common_coverage",
                        "low_risk_error",
                        "low_coverage",
                        "unknown_fraction",
                        "invalid_fraction",
                    )
                ],
            )
            for c in CONDITIONS
            for m in METHODS
        ],
    )
    for c in CONDITIONS:
        md += [f"### {c} family results", ""]
        md += table(
            [
                "Family",
                "Method",
                "Two-class targets",
                "Error / E",
                "AP",
                "Low error",
                "Low / V",
                "U / E",
                "Invalid / V",
            ],
            [
                (
                    r["family"],
                    r["method"],
                    r["common_two_class_views"],
                    *[
                        format_num(r[k])
                        for k in (
                            "rgb_error_incidence",
                            "common_ap",
                            "low_risk_error",
                            "low_coverage",
                            "unknown_fraction",
                            "invalid_fraction",
                        )
                    ],
                )
                for r in main
                if r["condition"] == c
            ],
        )
        md += [f"![Attainable risk and coverage for {c}](figures/coverage-{c}.png)", ""]
    observed = avg("B0", "observed", "constant", "rgb_error_incidence")
    dis = avg("B0", "disoccluded", "constant", "rgb_error_incidence")
    matched_geo = []
    for f in families:
        a, b = (
            trials[f, "B0"]["family"][r]["outcomes"]["rgb_error_incidence"]["value"]
            for r in ("disoccluded", "observed")
        )
        if a is not None and b is not None:
            matched_geo.append(a - b)
    md += [
        "## Spatial error and paired interventions",
        "",
        f"Across the four preassigned B0 test families, family-mean error incidence on E was {format_num(dis['mean'])} in disoccluded regions ({dis['eligible_family_count']} families) and {format_num(observed['mean'])} on observed surfaces ({observed['eligible_family_count']} families). The mean within-family difference among the {len(matched_geo)} families with both defined was {format_num(describe(matched_geo)['mean'])}. Geometric visibility comes from reference surfaces and is independent of the actual colour difference.",
        "",
        "![Paired outcomes](figures/paired-conditions.png)",
        "",
    ]
    md += table(
        [
            "Family",
            "Variant",
            "Change in RGB MAE",
            "Change in error / E",
            "Change in confirmed error / V",
            "Change in edge confirmed error / V",
            "Paired L1-full AP change",
            "Change in low coverage",
        ],
        [
            (
                r["family"],
                r["variant"],
                *[
                    format_num(r[k + "_delta"])
                    for k in ("rgb_mae", "rgb_error_incidence", "confirmed_error_fraction_of_V")
                ],
                format_num(
                    next(
                        d["confirmed_error_fraction_of_V_delta"]
                        for d in paired
                        if d["region"] == "edge"
                        and d["family"] == r["family"]
                        and d["variant"] == r["variant"]
                        and d["method"] == "L1-full"
                    )
                ),
                format_num(r["common_ap_delta"]),
                format_num(r["low_coverage_delta"]),
            )
            for r in paired
            if r["region"] == "all" and r["method"] == "L1-full"
        ],
    )
    effects = [
        r["confirmed_error_fraction_of_V_delta"]
        for r in paired
        if r["variant"] == "R1" and r["region"] == "all" and r["method"] == "L1-full"
    ]
    effect = describe(effects)
    md += [
        f"The registered R1 outcome is confirmed error pixels divided by all V. Its family-equal paired change from B0 was {format_num(effect['mean'])} across {effect['eligible_family_count']} eligible families. The edge analysis is descriptive.",
        "",
    ]
    md += table(
        [
            "Family",
            "Q1/B0 paired targets",
            "Q1–B0 RGB response",
            "Q1 reference MAE",
            "Changed RGB pixels",
            "Bitwise-identical geometry targets",
            "Changed full-score pixels",
        ],
        [
            (
                r["family"],
                r["paired_views"],
                format_num(r["mean_Q1_B0_rgb_response"]),
                format_num(r["true_Q1_reference_MAE"]),
                r["changed_rgb_pixels"],
                r["geometry_bitwise_equal_views"],
                r["full_score_changed_values"],
            )
            for r in q1
        ],
    )
    md += [
        "Prediction-to-prediction RGB response measures the numerical perturbation. Q1 reference MAE measures reconstruction error. The per-field appendix tests native protected PLY attributes, target opacity, expected camera-Z, contribution masks, geometric features and L1-geometry scores using exact bytes. L1-full may respond to the appearance changes.",
        "",
        "## Fixed visual examples",
        "",
        "Every family contributes its preregistered v01 target under all three conditions. Continuous error and learned risk use fixed, separate scales. Gray denotes an unknown risk on E and magenta denotes unevaluable RGB.",
        "",
    ]
    for f in families:
        md += [f"### {f}", ""]
        for c in CONDITIONS:
            md += [f"![{f} {c} fixed first target](figures/{f}-{c}-v01.png)", ""]
    md += [
        "## Failures, cost and reproducibility",
        "",
        f"The matrix retains {72 - complete} planned target renders without an output. Invalid pixels remain in V. Low-risk misses, warnings without a true error, unknown-error counts, empty strata and missing AP are retained for every method and target. The probability of error outside these static authored layouts is unmeasured.",
        "",
        f"The explicit run inventory covers {len(stages)} unique runs. Its declared scope is “{s.meta['inventory_scope']}”. See the stage-cost table for host, elapsed time, sampled RSS, logical bytes, retry status and missing measurements. Dedicated timing repeats are separated from quality scores.",
        "",
    ]
    groups = defaultdict(list)
    for r in stages:
        groups[r["kind"]].append(r)
    md += table(
        [
            "Stage",
            "Listed runs",
            "Succeeded",
            "Failed or interrupted",
            "Retries",
            "Median wall seconds",
            "Logical payload bytes",
        ],
        [
            (
                k,
                len(v),
                sum(r["status"] == "succeeded" for r in v),
                sum(r["status"] != "succeeded" for r in v),
                sum(bool(r["retry_of"]) for r in v),
                format_num(describe([r["elapsed_seconds"] for r in v])["median"]),
                sum(r["payload_bytes"] for r in v),
            )
            for k, v in sorted(groups.items())
        ],
    )
    md += table(
        [
            "Family",
            "Condition",
            "Generation basis",
            "Generation (s)",
            "Preparation (s)",
            "Render (s)",
            "Render change from B0 (s)",
        ],
        [
            (
                r["family"],
                r["condition"],
                r["generation_cost_basis"],
                format_num(r["generation_seconds"]),
                format_num(r["preparation_seconds"]),
                format_num(r["render_seconds"]),
                format_num(r["render_delta_B0_seconds"]),
            )
            for r in costs
        ],
    )
    md += [
        "Q1 inherits the baseline model generation. Only the binary16 round trip and fresh views are incremental. Recorded process and phase costs are distinct. Missing driver/process measurements remain unavailable. Shared logical payloads can appear in multiple run bundles, so the storage column is distinct from physical disk or archive occupancy.",
        "",
        "### Evidence and limits",
        "",
    ]
    for label, done in safety.items():
        md += [f"- {label} — {'available' if done else 'unavailable in the explicit evidence inventory'}"]
    measured = [
        (p["id"], p["kind"], k, v) for p in s.meta.get("proofs", []) for k, v in p["measures"].items()
    ]
    if measured:
        md += table(["Selected proof", "Kind", "Measured field", "Recorded value"], measured)
    md += [
        "",
        f"The pinned model ID is `{method['model_id']}`. Comparison protocol digest `{method['comparison_protocol_sha256']}`. The report input manifest, all selected P6 run IDs, numeric tables, original figure inputs, output hashes and exact rendering choices are in this report package.",
        "",
        "[Numerical tables and figure sources](numeric/) · [Figure provenance](figures.json) · [Report receipt](report.json)",
        "",
    ]
    location.write_text("\n".join(md), encoding="utf-8")


def publish_numerical(
    out, s, trials, catalog, rows, summary, paired_macro, q1, stages, phases, costs, timing
):
    """Create a numerical release candidate using a fixed field allowlist, independently of private receipts."""
    public = out / "public-numerical"
    public.mkdir()
    families = {r["scene_family_id"] for r in catalog["families"]}
    markers = {(r["scene_family_id"], r["condition_id"]): r for r in s.meta.get("scientific_markers", [])}
    completion = []
    for family in catalog["families"]:
        f = family["scene_family_id"]
        for c in family.get("trials", {}):
            key = f, c
            rec = trials.get(key, {})
            mark = markers.get(key, {})
            completion.append(
                {
                    "family": f,
                    "split": family["split"],
                    "condition": c,
                    "planned_targets": 6,
                    "canonical_quality_selection_sha256": mark.get("sha256"),
                    "marked_attempt_count": mark.get("attempt_count"),
                    "heldout_completed_targets": rec.get("completed_targets"),
                    "heldout_planned_targets": rec.get("planned_targets"),
                    "availability": "selected"
                    if mark
                    else "P6 retained"
                    if rec
                    else "selection unavailable in inventory",
                }
            )
    csv_write(public / "data-completion.csv", completion)
    safe_costs = []
    for r in stages:
        platform = (r["platform"] or "").lower()
        group = next(
            (
                n
                for n, match in (("Windows", "windows"), ("macOS", "macos"), ("Linux", "linux"))
                if match in platform
            ),
            "unreported",
        )
        safe_costs.append(
            {
                "run_id": r["run_id"],
                "manifest_sha256": r["manifest_sha256"],
                "purpose": r["purpose"],
                "stage": r["kind"],
                "family": r["family"] if r["family"] in families else None,
                "condition": r["condition"] if r["condition"] in CONDITIONS else None,
                "platform_family": group,
                "elapsed_seconds": r["elapsed_seconds"],
                "peak_sampled_rss_bytes": r["peak_sampled_rss_bytes"],
                "logical_bundle_bytes": r["logical_bundle_bytes"],
                "payload_bytes": r["payload_bytes"],
                "status": r["status"],
                "is_retry": bool(r["retry_of"]),
                "timing_index": r["timing_index"],
            }
        )
    csv_write(
        public / "stage-times.csv", safe_costs, None if safe_costs else ("run_id", "stage", "elapsed_seconds")
    )
    # Each field here is authored by this evaluator or belongs to the frozen family/method vocabulary.
    phase_labels = {
        "load_model",
        "preprocess_forward_unproject",
        "export_ply",
        "calibration",
        "load_ply",
        "preprocess_seconds",
        "forward_seconds",
        "unproject_seconds",
        "target_read_seconds",
        "prediction_read_seconds",
        "features_seconds",
        "scoring_seconds",
        "output_seconds",
        "output_write_and_hash_seconds",
        "cv_fit",
        "FP16 colour round trip and file verification",
        *("view_" + v for v in ("source", *TARGETS)),
    }
    safe_phases = [
        {
            "run_id": r["run_id"],
            "phase": r["phase"],
            "camera": r.get("camera") if r.get("camera") in TARGETS else None,
            "seconds": r.get("seconds"),
            "cuda_peak_allocated_bytes": r.get("cuda_peak_allocated_bytes"),
            "cuda_peak_reserved_bytes": r.get("cuda_peak_reserved_bytes"),
            "C": r.get("C"),
            "fold": r.get("fold"),
            "iterations": r.get("iterations"),
        }
        for r in phases
        if r["phase"] in phase_labels
    ]
    csv_write(
        public / "worker-phase-times.csv",
        safe_phases,
        None if safe_phases else ("run_id", "phase", "seconds"),
    )
    csv_write(public / "family-results.csv", rows)
    csv_write(public / "family-equal-summary.csv", summary)
    csv_write(public / "paired-family-equal.csv", paired_macro)
    csv_write(public / "Q1-numerical-response.csv", q1)
    csv_write(public / "paired-stage-times.csv", costs)
    csv_write(
        public / "timing-repeats.csv",
        timing,
        None if timing else ("family", "condition", "stage", "listed_repeats", "median_seconds"),
    )
    measurements = {
        "verified_runs",
        "run_count",
        "restored_runs",
        "compared_files",
        "identical_outputs",
        "verified_outputs",
        "logical_files",
        "container_bytes",
        "archive_bytes",
        "unpacked_bytes",
        "physical_bytes",
        "download_seconds",
        "upload_seconds",
        "index_runs",
        "index_receipts",
        "restored_receipts",
        "receipt_count",
        "passed",
        "elapsed_seconds",
        "elapsed_minutes",
        "logical_bytes",
        "file_count",
        "bytes",
        "producer_elapsed_seconds",
        "render_elapsed_seconds",
        "producer_seconds",
    }
    safe_kinds = {
        "scientific_marker",
        "windows_timing",
        "cpu_timing",
        "learning_marker",
        "transfer",
        "archive",
        "storage",
        "validation",
        "resource",
        "recovery",
        "engineering",
    }
    numeric_proofs = []
    for proof in s.meta.get("proofs", []):
        if proof["kind"] in safe_kinds:
            for key, value in proof["measures"].items():
                if key in measurements:
                    numeric_proofs.append(
                        {
                            "receipt_kind": proof["kind"],
                            "receipt_sha256": proof["sha256"],
                            "field": key,
                            "measured_value": value,
                        }
                    )
    csv_write(
        public / "measured-receipts.csv",
        numeric_proofs,
        None if numeric_proofs else ("receipt_kind", "receipt_sha256", "field", "measured_value"),
    )
    status = {
        "schema": "el-numerical-study-release-candidate-1",
        "kind": s.meta["kind"],
        "scope": "Four fixed original opaque scene families and six held-out targets under B0, R1 and Q1. Three timing repeats measure engineering costs.",
        "model_id": next(iter(trials.values()))["model_id"],
        "model_index_sha256": s.json("protocol/method_freeze.json").get("model_index_sha256"),
        "method_code_commit": s.json("protocol/method_freeze.json").get("method_code_commit"),
        "catalog_sha256": s.files["protocol/catalog.json"]["sha256"],
        "study_sha256": s.files["protocol/study.json"]["sha256"],
        "method_freeze_sha256": s.meta["method_freeze_sha256"],
        "comparison_protocol_sha256": s.meta["comparison_protocol_sha256"],
        "roster_sha256": s.meta.get("roster_sha256"),
        "input_selection_sha256": s.meta["selection_sha256"],
        "comparison_trials": [
            {k: r[k] for k in ("scene_family_id", "condition_id", "run_id", "manifest_sha256")}
            for r in s.meta["trials"]
        ],
        "files": {p.name: signature(p) for p in sorted(public.iterdir()) if p.is_file()},
    }
    write_json(public / "selection.json", status)


def build(bundle, output, run_id):
    s = Staged(bundle, "report_inputs.json")
    require(s.meta["schema"] == "el-formal-report-input-1", "incorrect input format")
    catalog, study, method = (
        s.json("protocol/" + k + ".json") for k in ("catalog", "study", "method_freeze")
    )
    require(study["test_methods"] == list(METHODS), "report method list differs from preregistration")
    require(
        s.meta["comparison_protocol_sha256"]
        == method["comparison_protocol_sha256"]
        == protocol(Path(__file__).resolve().parents[2])["comparison_protocol_sha256"]
        and s.files["protocol/catalog.json"]["sha256"] == method["catalog_sha256"]
        and s.files["protocol/study.json"]["sha256"] == method["study_sha256"]
        and s.files["protocol/method_freeze.json"]["sha256"] == s.meta["method_freeze_sha256"],
        "report freeze changed",
    )
    families = [r["scene_family_id"] for r in catalog["families"] if r["split"] == "test"]
    require(len(families) == len(set(families)) == 4, "report requires the four frozen test families")
    trials, curves, boards, cross_rows = {}, {}, {}, []
    for r in s.meta["trials"]:
        f, c = r["scene_family_id"], r["condition_id"]
        prefix = f"trials/{f}/{c}/"
        rec = s.json(prefix + "formal_evaluation.json")
        require(
            rec["schema"] == "el-formal-comparison-1"
            and rec["kind"] == s.meta["kind"]
            and (f, c) not in trials
            and rec["run_id"] == r["run_id"]
            and rec["condition_id"] == c
            and rec["scene_family_id"] == f
            and rec["dataset_hash"] == method["catalog_sha256"]
            and rec["study_sha256"] == method["study_sha256"]
            and rec["model_id"] == method["model_id"]
            and rec["method_freeze_sha256"] == s.meta["method_freeze_sha256"]
            and rec["comparison_protocol_sha256"] == s.meta["comparison_protocol_sha256"]
            and set(rec["views"]) == set(TARGETS),
            "mismatched explicit P6 result",
        )
        curves[f, c] = s.open(prefix + "curves/family.npz")
        require(
            pool_family(rec["views"]) == rec["family"],
            "P6 family summary differs from its six target reductions",
        )
        require(
            file_hash(curves[f, c]) == rec["curves_sha256"]["curves/family.npz"], "P6 curve digest differs"
        )
        boards[f, c] = s.open(prefix + "board/board_v01.png")
        detail = s.json(prefix + "board/board_v01.json")
        require(
            detail["camera_id"] == "v01"
            and detail["board_sha256"] == file_hash(boards[f, c]) == rec["board_sha256"],
            "example selection mismatch",
        )
        cross = s.json(prefix + "reason_visibility.json")
        cross_rows.extend({"family": f, "condition": c, **v} for v in [*cross["views"], *cross["family"]])
        trials[f, c] = rec
    require(
        set(trials) == {(f, c) for f in families for c in CONDITIONS},
        "report omits a planned family or condition",
    )
    frozen_thresholds = trials[families[0], "B0"]["thresholds"]
    for f in families:
        base = trials[f, "B0"]
        for c in CONDITIONS:
            require(
                trials[f, c]["target_reference_sha256"] == base["target_reference_sha256"]
                and trials[f, c]["thresholds"] == frozen_thresholds,
                "conditions do not share the same reference and thresholds",
            )
        q = trials[f, "Q1"]
        require(
            q["input_sha256"] == base["input_sha256"]
            and q["camera_sha256"] == base["camera_sha256"]
            and q["parents"]["paired_prediction"] == base["parents"]["prediction"],
            "Q1 compared with the wrong baseline",
        )
        if not q.get("render_failure_reason") and "paired_learned" in q["parents"]:
            require(
                all(q["parents"]["paired_" + k] == base["parents"][k] for k in ("learned", "rules")),
                "wrong Q1 score baseline",
            )
    rows, per_view = family_rows(trials)
    summary, paired = summarize(rows, per_view), paired_rows(trials, rows)
    detail, q1, ply = q1_records(trials, families)
    stages, phases, documents = evidence_tables(s, rows)
    costs = paired_costs(trials, families, stages)
    with new_output(Path(output)) as out:
        numeric, fig = out / "numeric", out / "figures"
        numeric.mkdir()
        fig.mkdir()
        csv_write(numeric / "family-system.csv", rows)
        csv_write(numeric / "per-view-system.csv", per_view)
        csv_write(numeric / "family-equal-summary.csv", summary)
        csv_write(numeric / "paired.csv", paired)
        csv_write(numeric / "Q1-field-response.csv", detail)
        csv_write(numeric / "Q1-family-response.csv", q1)
        csv_write(
            numeric / "Q1-ply-fields.csv",
            ply or [],
            None if ply else ("family", "field", "protected", "exact_equal"),
        )
        csv_write(numeric / "reason-visibility.csv", cross_rows)
        csv_write(numeric / "paired-stage-costs.csv", costs)
        csv_write(
            numeric / "costs.csv", stages or [], None if stages else ("run_id", "kind", "elapsed_seconds")
        )
        csv_write(
            numeric / "phases.csv", phases or [], None if phases else ("run_id", "phase", "camera", "seconds")
        )
        operating = []
        durations = defaultdict(list)
        for r in stages:
            if r["purpose"] == "timing_repeat":
                durations[r["family"], r["condition"], r["kind"]].append(r)
        for (family, cond, kind), series in durations.items():
            successful = [r for r in series if r["status"] == "succeeded"]
            by_index = {r["timing_index"]: r["elapsed_seconds"] for r in successful}
            ids = {r["timing_index"] for r in series}
            require(
                ids <= {1, 2, 3} and len(by_index) == len(successful),
                "timing series has a missing index or duplicate successful stage",
            )
            seconds = [by_index[i] for i in sorted(by_index)]
            operating.append(
                {
                    "family": family,
                    "condition": cond,
                    "stage": kind,
                    "listed_attempts": len(series),
                    "listed_repeats": len(ids),
                    "successful_repeats": len(seconds),
                    "successful_indices": "|".join(map(str, sorted(by_index))),
                    "missing_indices": "|".join(map(str, sorted({1, 2, 3} - set(by_index)))),
                    "times_seconds": json.dumps(seconds),
                    "median_seconds": describe(seconds)["median"],
                    "minimum_seconds": min(seconds, default=None),
                    "maximum_seconds": max(seconds, default=None),
                }
            )
        csv_write(
            numeric / "timing-repeats.csv",
            operating or [],
            None if operating else ("family", "condition", "stage", "listed_repeats", "median_seconds"),
        )
        by_fam = defaultdict(list)
        for r in paired:
            by_fam[r["variant"], r["region"], r["method"]].append(r)
        paired_macro = []
        for (c, region, method_name), vals in by_fam.items():
            for metric in COLUMNS:
                paired_macro.append(
                    {
                        "variant": c,
                        "region": region,
                        "method": method_name,
                        "metric": metric,
                        **describe([r[metric + "_delta"] for r in vals]),
                    }
                )
        csv_write(numeric / "paired-family-equal.csv", paired_macro)
        unknown = []
        depth = []
        for (f, c), tr in trials.items():
            for view, t in tr["views"].items():
                for region, r in t["regions"].items():
                    for m, d in r["methods"].items():
                        for reason, n in d["unknown_reasons"].items():
                            unknown.append(
                                {
                                    "family": f,
                                    "condition": c,
                                    "view": view,
                                    "region": region,
                                    "method": m,
                                    "reason": reason,
                                    **n,
                                }
                            )
            for view, d in [
                ("family", tr["depth_distribution"]),
                *((v, r["depth_distribution"]) for v, r in tr["views"].items()),
            ]:
                for key, data in d.items():
                    depth.extend(
                        {
                            "family": f,
                            "condition": c,
                            "view": view,
                            "measurement": key,
                            "count": data["count"],
                            "q": prob,
                            "value": data["values"][i] if data["values"] else None,
                            "missing_reason": data["missing_reason"],
                        }
                        for i, prob in enumerate(data["probabilities"])
                    )
        csv_write(
            numeric / "unknown-reasons.csv",
            unknown or [],
            None if unknown else ("family", "condition", "view", "region", "method", "reason", "U", "UY"),
        )
        csv_write(numeric / "signed-depth-distributions.csv", depth)
        figure_index = {}

        def register(name, sources):
            figure_index[name] = {
                "sha256": file_hash(fig / name),
                "sources": {rel: file_hash(out / rel) for rel in sources},
            }

        for f in families:
            for c in CONDITIONS:
                rel = f"trials/{f}/{c}/"
                for key, name in (
                    ("curves/family.npz", f"numeric/{f}-{c}-curve.npz"),
                    ("board/board_v01.json", f"numeric/{f}-{c}-board.json"),
                ):
                    shutil.copyfile(s.open(rel + key), out / name)
                name = f"{f}-{c}-v01.png"
                shutil.copyfile(boards[f, c], fig / name)
                register(name, [f"numeric/{f}-{c}-board.json"])
        for c in CONDITIONS:
            name = f"coverage-{c}.png"
            draw_curves(fig / name, c, families, curves)
            register(name, [f"numeric/{f}-{c}-curve.npz" for f in families])
        draw_paired(fig / "paired-conditions.png", families, rows)
        register("paired-conditions.png", ["numeric/family-system.csv"])
        draw_architecture(fig / "architecture.png")
        for name in ("catalog", "study", "method_freeze"):
            shutil.copyfile(s.open("protocol/" + name + ".json"), numeric / (name + ".json"))
        register("architecture.png", ["numeric/study.json"])
        for rel in s.files:
            if rel.startswith(("evidence/", "proofs/")):
                dest = out / "numeric" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(s.open(rel), dest)
        safety = {
            "Fitted coefficients and preprocessing": bool(documents.get("el-l1-coefficients-1")),
            "Grouped model selection": bool(documents.get("el-l1-cv-1")),
            "Final training/export comparison": bool(documents.get("el-model-fit-1")),
            "B0 development threshold selection": bool(documents.get("el-development-freeze-1")),
            "Three timing repeats per condition": all(
                any(
                    r["condition"] == c and r["stage"] == k and r["successful_repeats"] == 3
                    for r in operating
                )
                for c in CONDITIONS
                for k in (("derivation" if c == "Q1" else "generation"), "render", "model_score")
            ),
            "Recovery receipt": any(r["purpose"] == "recovery" and r["status"] == "succeeded" for r in stages)
            or any(p["kind"] == "recovery" and p["measures"] for p in s.meta.get("proofs", [])),
        }
        publish_numerical(
            out, s, trials, catalog, rows, summary, paired_macro, q1, stages, phases, costs, operating
        )
        render_report(out, s, trials, families, rows, summary, paired, q1, stages, costs, documents, safety)
        write_json(
            out / "figures.json",
            {
                "schema": "el-study-figures-1",
                "kind": s.meta["kind"],
                "figures": figure_index,
                "risk_coverage": "All numeric endpoints archived. A plotted raster pixel contains one or more actual endpoints. No interpolation or splitting of ties.",
            },
        )
        result = {
            "schema": "el-study-report-1",
            "run_id": run_id,
            "kind": s.meta["kind"],
            "scope": "Four fixed held-out families with paired B0/R1/Q1",
            "comparison_protocol_sha256": s.meta["comparison_protocol_sha256"],
            "report_input_sha256": file_hash(s.path / "report_inputs.json"),
            "selected_trials": s.meta["trials"],
            "quality_trials": len(trials),
            "planned_targets": 72,
            "completed_targets": sum(v["completed_targets"] for v in trials.values()),
            "model_id": method["model_id"],
            "evidence_availability": safety,
            "source_manifest": s.meta,
            "files": {
                p.relative_to(out).as_posix(): signature(p) for p in sorted(out.rglob("*")) if p.is_file()
            },
        }
        write_json(out / "report.json", result)
    return result
