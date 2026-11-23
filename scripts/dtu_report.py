"""Aggregate D1's registered matrix into public numbers and a separate private photographic gallery."""

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from evidence_lens.formal_inputs import require, verified_file
from evidence_lens.scoring_io import file_hash, new_output, write_json
from dtu_eval import DOMAINS, read_pinned
from dtu_eval_metrics import RATES
from dtu_score import METHODS, SCOPES, VIEWS, RELEASE_SHA
from vision_sources import signed

COUNT_KEYS = (
    "V",
    "E",
    "V_minus_E",
    "Y",
    "C",
    "CY",
    "S",
    "SY",
    "L",
    "LY",
    "H",
    "HY",
    "U",
    "UY",
    "missed_Y",
    "Y3",
    "rgb_mae_sum",
    "aux3_mae_sum",
    "brier_sum",
)


def csv_file(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    require(bool(columns), "refusing empty table " + path.name)
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def public_rates(metric):
    return {
        **{k: None if metric is None else metric["counts"][k] for k in COUNT_KEYS},
        **{k: None if metric is None else metric["rates"][k]["value"] for k in RATES},
    }


def ratio(v, percentage=False):
    if v is None:
        return "N/A"
    return f"{v * 100:.2f}%" if percentage else f"{v:.4f}"


def method_rows(identity, measurement, domain, out):
    g = measurement["aggregate"][domain]
    reasons = sorted(
        {row["domain_reasons"][domain] for row in measurement["rows"] if domain in row["domain_reasons"]}
    )
    reason = " | ".join(reasons) or None
    for m in METHODS:
        z = g["methods"].get(m)
        base = {
            **identity,
            "domain": domain,
            "method": m,
            "scope": SCOPES[m],
            "tau": measurement["thresholds"][m],
        }
        agg = {
            **base,
            "planned_views": len(VIEWS),
            "reference_present_views": g["reference_present_views"],
            "domain_missing_reason": reason if z is None else None,
            **public_rates(z),
        }
        for k in ("own", "common"):
            agg["AP_" + k] = z["AP"][k]["ap"] if z else None
            agg[k + "_AP_defined_views"] = z["AP"][k]["defined_views"] if z else 0
        agg["ece10"] = z["ece10"] if z else None
        out["object"].append(agg)
        for row in measurement["rows"]:
            obs = (row["domains"][domain] or {}).get(m)
            entry = {
                **base,
                "view": row["view"],
                "official_view_id": row.get("official_view_id"),
                "status": row["disposition"],
                **public_rates(obs),
            }
            for k in ("own", "common"):
                entry["AP_" + k] = obs[k]["ap"] if obs else None
                entry[k + "_AP_missing_reason"] = (
                    obs[k]["missing_reason"] if obs else row["domain_reasons"][domain]
                )
            out["view"].append(entry)
        if z:
            cur = z["risk_coverage"]
            out["curve"].extend({**base, "type": "whole_group", **p} for p in cur["points"])
            out["curve"].append(
                {
                    **base,
                    "type": "frozen",
                    "requested_coverage": None,
                    "accepted": cur["frozen_L"],
                    "positive": cur["frozen_LY"],
                    "V": z["counts"]["V"],
                    "coverage": z["rates"]["low_coverage"]["value"],
                    "risk": z["rates"]["low_error"]["value"],
                }
            )
            out["bins"].extend(
                {**base, "bin_lower": b["bin"] / 10, "bin_upper": (b["bin"] + 1) / 10, **b}
                for b in z["bin_scores"]
            )


def summaries(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["role"], r["arm"], r["condition"], r["domain"], r["method"])].append(r)
    result = []
    for keys, objects in grouped.items():
        base = dict(zip(("role", "arm", "condition", "domain", "method"), keys))
        base["scheduled_objects"] = len(objects)
        base["physical_families"] = len({r["family_id"] for r in objects})
        base["scheduled_views"] = sum(r["planned_views"] for r in objects)
        present = [r for r in objects if r["E"] is not None]
        base["known_domain_objects"] = len(present)
        for k in ("AP_common", "AP_own", *RATES, "ece10"):
            vals = [r[k] for r in present if r[k] is not None]
            base[k + "_object_equal"] = fmean(vals) if vals else None
            base[k + "_defined_objects"] = len(vals)
        for k in ("own", "common"):
            base[k + "_AP_defined_views"] = sum(r[k + "_AP_defined_views"] for r in objects)
        for k in COUNT_KEYS:
            base[k + "_pool"] = sum(r[k] for r in present) if present else None
        for k, (n, d) in RATES.items():
            base[k + "_pool"] = (
                base[n + "_pool"] / base[d + "_pool"] if present and base[d + "_pool"] else None
            )
        result.append(base)
    return result


def paired(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["role"], row["object_id"], row["family_id"], row["arm"], row["domain"], row["method"])][
            row["condition"]
        ] = row
    records = []
    for k, pair in groups.items():
        require(set(pair) == {"M2", "M4"}, "incomplete registered paired conditions")
        a, b = pair["M2"], pair["M4"]
        base = dict(zip(("role", "object_id", "family_id", "arm", "domain", "method"), k))
        for metric in (
            "AP_common",
            "AP_own",
            "rgb_error_fraction",
            "rgb_mae",
            "low_coverage",
            "low_error",
            "unknown",
            "validity",
            "common_prevalence",
        ):
            aa, bb = a[metric], b[metric]
            base[metric + "_M2"] = aa
            base[metric + "_M4"] = bb
            base[metric + "_M4_minus_M2"] = bb - aa if aa is not None and bb is not None else None
        records.append(base)
    return records


def run(roots, request, out, rid):
    require(request["schema"] == "el-dtu-report-request-1", "wrong report request")
    objects = {o["object_id"]: o for o in request["objects"]}
    require(
        len(objects) == len(request["objects"]) == 2
        and set(objects) == {"set1", "set6"}
        and objects["set1"]["role"] == "debug"
        and objects["set6"]["role"] == "external",
        "unexpected formal roster",
    )
    planned = {(o, c, "main-256") for o in objects for c in ("M2", "M4")}
    diagnostics = request.get("diagnostics", [])
    require(
        (not diagnostics and not request.get("debug_1024"))
        or (
            request.get("debug_1024")
            and len(diagnostics) == 2
            and {d["condition"] for d in diagnostics} == {"M2", "M4"}
        ),
        "incomplete separate development budget diagnostic",
    )
    cells = request["cells"]
    require(
        len(cells) == len(planned) and {(c["object_id"], c["condition"], c["arm"]) for c in cells} == planned,
        "incomplete or duplicate scientific matrix",
    )
    gate = None
    authority_pins = []
    for pin in request["authorities"]:
        _, p = read_pinned(roots, pin)
        doc = json.loads(p.read_bytes())
        authority_pins.append(pin)
        if doc["schema"] == "el-dtu-reference-gate-1":
            gate = (pin, doc)
    require(
        gate and gate[1]["object_ids"] == ["set6"],
        "external joint results require the separate complete-reference gate",
    )
    tables = {
        k: []
        for k in (
            "view",
            "object",
            "curve",
            "bins",
            "diag",
            "resource",
            "cells",
            "budget",
            "reconstruction",
            "sources",
        )
    }
    pins = []
    board_links = []
    with new_output(out) as dest:
        pub = dest / "public-numerical"
        pub.mkdir()
        gallery = dest / "private-cases"
        gallery.mkdir()
        for c in cells:
            oid, cond, arm = c["object_id"], c["condition"], c["arm"]
            o = objects[oid]
            role = o["role"]
            require(c["role"] == role, "role differs from enrollment")
            rr, p = read_pinned(roots, c["evaluation"])
            _, q = read_pinned(roots, c["replay"])
            d = json.loads(p.read_bytes())
            proof = json.loads(q.read_bytes())
            require(
                d["schema"] == "el-dtu-condition-evaluation-1"
                and (d["object_id"], d["condition"], d["arm"], d["role"]) == (oid, cond, arm, role)
                and d["release_sha256"] == RELEASE_SHA
                and d["method_scopes"] == SCOPES
                and [x["view"] for x in d["rows"]] == list(VIEWS)
                and proof["schema"] == "el-dtu-original-pixel-replay-1"
                and proof["passed"] is True
                and proof["evaluation"] == c["evaluation"],
                "mismatched scientific evaluation/replay",
            )
            if role == "external":
                require(
                    d["input_pins"]["authority"] == gate[0],
                    "external condition evaluated with different authority",
                )
            identity = {
                "object_id": oid,
                "family_id": o["family_id"],
                "role": role,
                "condition": cond,
                "arm": arm,
            }
            pins.append(
                {**identity, "evaluation": c["evaluation"], "replay": c["replay"], "inputs": d["input_pins"]}
            )
            for domain in DOMAINS:
                method_rows(identity, d, domain, tables)
            for r in d["rows"]:
                tables["diag"].append(
                    {
                        **identity,
                        "view": r["view"],
                        "official_view_id": r.get("official_view_id"),
                        "disposition": r["disposition"],
                        **r.get("engineering_observations", {}),
                    }
                )
                if "board" in r:
                    old = verified_file(p.parent, r["board"]["path"], r["board"])
                    rel = Path(oid) / arm / cond / (r["view"] + ".png")
                    local = gallery / rel
                    local.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(old, local)
                    require(file_hash(local) == r["board"]["sha256"], "private board differs from evaluation")
                    board_links.append(f"- [{oid} {arm} {cond} {r['view']}]({rel.as_posix()})")
            tables["cells"].append(
                {
                    **identity,
                    "execution_status": d["status"],
                    "failure_reason": d.get("failure_reason"),
                    "planned_targets": 6,
                    "reference_readable_targets": d["aggregate"]["full_crop"]["reference_present_views"],
                    "ready_targets": sum(x["disposition"] == "ready" for x in d["rows"]),
                    "expected_target_method_pairs": 6 * len(METHODS),
                    "scored_target_method_pairs": len(METHODS)
                    * sum(x["disposition"] == "ready" for x in d["rows"]),
                    "maximum_replay_AP_gap": proof["max_abs_ap_gap"],
                }
            )
            eng = d.get("reconstruction_readback")
            if eng:
                reconstruction_rows(identity, eng, tables)
        for entry in diagnostics:
            _, pp = read_pinned(roots, entry["diagnostic"])
            proof = json.loads(pp.read_bytes())
            cond = entry["condition"]
            require(
                proof["schema"] == "el-dtu-budget-diagnostic-1"
                and proof["object_id"] == "set1"
                and proof["condition"] == cond
                and proof["risk_scores_computed"] == 0
                and proof["input_pins"]["evaluation"]
                == next(
                    c["evaluation"] for c in cells if c["condition"] == cond and c["object_id"] == "set1"
                ),
                "unrelated optional budget diagnostic",
            )
            for row in [*proof["views"], {"view": "all", **proof["totals"]}]:
                for arm_key in ("main_256", "diagnostic_1024"):
                    tables["budget"].append(
                        {
                            "role": "debug",
                            "object_id": "set1",
                            "condition": cond,
                            "view": row["view"],
                            "arm": arm_key,
                            **row[arm_key],
                        }
                    )
            identity = {
                "object_id": "set1",
                "family_id": objects["set1"]["family_id"],
                "role": "debug",
                "condition": cond,
                "arm": "diagnostic-1024",
            }
            reconstruction_rows(identity, proof["readback_1024"], tables)
        for spec in request["declared_stages"]:
            r, p = read_pinned(
                roots, spec["original"], success=spec.get("status", "succeeded") == "succeeded"
            )
            raw = r.run
            tables["resource"].append(
                {
                    "stage": spec["stage"],
                    "object_id": spec.get("object_id"),
                    "condition": spec.get("condition"),
                    "arm": spec.get("arm"),
                    "run_id": raw["run_id"],
                    "manifest_sha256": r.pin["manifest_sha256"],
                    "run_kind": raw["run_kind"],
                    "status": raw["execution_status"],
                    "elapsed_seconds": raw.get("elapsed_seconds_monotonic"),
                    "source_commit": raw.get("code_commit"),
                    "source_snapshot_sha256": raw.get("code_snapshot_sha256"),
                    "payload_logical_bytes": sum(
                        v["bytes"] for k, v in r.files.items() if k.startswith("payload/")
                    ),
                    "manifested_bytes": sum(v["bytes"] for v in r.files.values()),
                    "public_disposition": spec.get("public_disposition"),
                    "sampled_peak_tree_rss_bytes": raw.get("sampled_peak_tree_rss_bytes"),
                    "member": spec["original"]["member"],
                    "member_sha256": spec["original"]["sha256"],
                }
            )
        require(tables["resource"], "expected explicit acquisition and evaluation resource receipts")
        summary = summaries(tables["object"])
        tables["summary"] = summary
        tables["paired"] = paired(tables["object"])
        names = {
            "view": "target-metrics",
            "object": "object-metrics",
            "summary": "cohort-metrics",
            "paired": "paired-conditions",
            "curve": "risk-coverage",
            "bins": "score-bins",
            "diag": "render-diagnostics",
            "resource": "resource-readback",
            "cells": "matrix",
            "budget": "development-budget",
            "reconstruction": "reconstruction",
            "sources": "source-supervision",
        }
        for key, name in names.items():
            if tables[key]:
                csv_file(pub / (name + ".csv"), tables[key])
        gallery.joinpath("index.md").write_text(
            "# Local photographic comparisons\n\n" + "\n".join(board_links) + "\n", encoding="utf-8"
        )
        provenance = {
            "schema": "el-dtu-numerical-provenance-1",
            "run_id": rid,
            "release": "v0.3.0-rc.1",
            "release_sha256": RELEASE_SHA,
            "method_scopes": SCOPES,
            "denominators": {k: list(v) for k, v in RATES.items()},
            "objects": request["objects"],
            "cells": pins,
            "diagnostics": diagnostics,
            "authorities": authority_pins,
            "rights": "DTU photographs, renderings, maps and photographic boards are retained for local research. This directory contains measurements and source digests.",
        }
        write_json(pub / "provenance.json", provenance)
        write_text(pub / "study.md", tables, summary, request)
        files = {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in sorted(pub.iterdir())}
        write_json(
            pub / "manifest.json",
            {"schema": "el-dtu-numerical-package-1", "report_run_id": rid, "files": files},
        )
        result = {
            "schema": "el-dtu-study-report-1",
            "run_id": rid,
            "release_sha256": RELEASE_SHA,
            "completed_cells": len(cells),
            "scientific_matrix": tables["cells"],
            "resource_receipts": len(tables["resource"]),
            "public_numerical": {
                "member": "public-numerical/manifest.json",
                "sha256": file_hash(pub / "manifest.json"),
            },
            "local_board_count": len(board_links),
            "private_gallery": "private-cases/index.md",
            "source_evaluations": pins,
        }
        write_json(dest / "report.json", result)
    return result


def reconstruction_rows(identity, eng, tables):
    tables["reconstruction"].append(
        {
            **identity,
            **{
                k: eng.get(k)
                for k in (
                    "passed",
                    "steps",
                    "initial_gaussians",
                    "final_gaussians",
                    "all_parameters_unchanged_gaussians",
                    "sampled_source_pixels",
                    "optimization_seconds",
                    "render_seconds",
                )
            },
            **eng["runtime"],
            "probe_max_ED_error_m": eng["GPU_probe"]["max_ED_error_m"],
        }
    )
    for source, row in eng["sources"].items():
        tables["sources"].append(
            {
                **identity,
                "source": source,
                **{
                    k: row.get(k)
                    for k in (
                        "source_updates",
                        "source_pixels",
                        "initial",
                        "optimizer_final",
                        "rendered_final",
                    )
                },
            }
        )


def write_text(path, tables, summary, request):
    lines = [
        "# Frozen risk scoring on DTU photographs",
        "",
        "The 256-step arm scores six target views per object and reconstruction. M2 uses the two endpoint sources. M4 uses all four. B-L1-transfer reads s00. The three multisource methods read every source allowed by their reconstruction. All coefficients, preprocessing and thresholds come from Evidence Lens v0.3.0-rc.1.",
        "",
        "Set1 was used for integration. Set6 was the registered external acquisition. One external object supports a within-object comparison. The six cameras and the four methods share that object. A cross-object interval and performance on other physical scenes remain undetermined.",
        "",
        "| Role and object | Arm | Condition | Method | Common AP (views) | Common prevalence | RGB errors / E | Low / V | Errors / low | Unknown / E |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    external = {
        r["condition"]: r
        for r in tables["object"]
        if r["role"] == "external" and r["domain"] == "full_crop" and r["method"] == "MS-L1"
    }
    if all(external[c]["E"] for c in ("M2", "M4")):
        a, b = external["M2"], external["M4"]
        extra = f"DTU 实拍外测 set6 采用发布包四个冻结模型。MS-L1 在 M2／M4 的共同域 AP 为 {ratio(a['AP_common'])}／{ratio(b['AP_common'])}，共同域错误基率为 {ratio(a['common_prevalence'], True)}／{ratio(b['common_prevalence'], True)}，低风险组错误为 {ratio(a['low_error'], True)}／{ratio(b['low_error'], True)}。单个外测对象的排序与拒判须分开解读。数值与代码可公开，实拍图板仅供本地研究。"
    else:
        extra = "本轮用 DTU 调试对象和登记外测对象检验四个冻结模型。外测条件存在缺失，原始失败和已知分母保留。确认实拍迁移仍缺少完整、独立的对象证据。数值与代码可公开，图板仅供本地研究。"
    lines[2:2] = [extra, ""]
    for row in tables["object"]:
        if row["domain"] != "full_crop":
            continue
        vals = [
            row["role"] + " " + row["object_id"],
            row["arm"],
            row["condition"],
            row["method"],
            f"{ratio(row['AP_common'])} ({row['common_AP_defined_views']}/6)",
        ]
        vals.extend(
            ratio(row[k], True)
            for k in ("common_prevalence", "rgb_error_fraction", "low_coverage", "low_error", "unknown")
        )
        lines.append("| " + " | ".join(vals) + " |")
    lines += [
        "",
        "## What the counts measure",
        "",
        "V is the independently valid photographic crop. E contains its pixels with valid predicted RGB. Y counts pixels of E where the mean absolute display-RGB channel error exceeds 0.05. S contains finite scores on E, U is its complement on E and low means finite risk strictly below the released threshold. All ratios keep their original numerator and denominator.",
        "",
        "Common AP uses E intersected with the four finite score masks. Own AP uses each method’s deployment mask. Exact score ties are grouped. Only views containing positive and negative labels have an AP. Views with defined AP carry equal weight within an object, and objects with defined AP carry equal weight in their cohort. A zero count of RGB errors gives a zero incidence when E is positive. Empty rate denominators give N/A.",
        "",
        "The object-only region is unavailable for photographs without an independently verified image-space mask. The full crop includes background. The auxiliary 3×3 test compares each centre prediction with the closest colour in its valid reference neighbourhood. It can hide misalignment or texture errors. Predicted expected camera depth and source support are engineering diagnostics. Surface-depth supervision is unavailable.",
        "",
        "The accompanying tables distinguish failed acquisition, missing reference, invalid RGB and unknown risk. A poor reconstruction stays in the registered matrix. The source-only scoring run and the separate target-photograph evaluations are pinned in [provenance](provenance.json).",
        "",
        "Read the [registered matrix](matrix.csv), [individual targets](target-metrics.csv), [object totals and rates](object-metrics.csv), [cohort estimates](cohort-metrics.csv), [paired changes](paired-conditions.csv), [engineering diagnostics](render-diagnostics.csv) and [recorded resources](resource-readback.csv).",
    ]
    if tables["curve"]:
        lines += [
            "",
            "[Risk–coverage points](risk-coverage.csv) preserve whole score groups and the actual frozen operating point. The [descriptive score bins](score-bins.csv) use ten equal-width bins.",
        ]
    if tables["budget"]:
        lines += [
            "",
            "The [development budget comparison](development-budget.csv) uses the same set1 source images, calibration, Gaussian topology and initial parameters. Each 1024-step endpoint was run independently. It measures the change in source and six target RGB errors. Frozen risk was evaluated only on the 256-step arms.",
        ]
    if tables["reconstruction"]:
        lines += [
            "",
            "The [reconstruction ledger](reconstruction.csv) records measured GPU update time, rendering time, source pixels and PyTorch CUDA allocator peaks. The [source ledger](source-supervision.csv) gives initial and final source-view error and updates per source.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("store", "shelf", "request", "output"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--request-sha256", required=True)
    a = p.parse_args()
    d = run((a.store, a.shelf), signed(a.request, a.request_sha256), a.output, os.environ.get("EL_RUN_ID"))
    print(json.dumps({k: d[k] for k in ("completed_cells", "local_board_count", "public_numerical")}))
