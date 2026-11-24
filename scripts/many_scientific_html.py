"""Publish an offline summary from the terminal numerical tables and installed reason dictionaries."""

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile

TABLES = (
    "views",
    "domain-ranks",
    "unknown-reasons",
    "comparability",
    "geometric-visibility",
    "view-priority",
)
PUBLIC = (
    "report.html",
    "study-summary.json",
    "unknown-reason-counts.csv",
    "reason-codebook.json",
    "interface-check.json",
)
EXTERNAL = ("living_room", "bridge")
DOMAIN_KEYS = ("scene", "R", "view", "cell", "method")
COUNT_KEYS = (
    "analysis_canvas",
    "full_reference",
    "independently_labelled",
    "method_own_labelled",
    "finite_canvas",
    "unknown_canvas",
    "unknown_full_reference",
    "unknown_labelled_reference",
)
PHYSICAL = (
    "probe_pixels",
    "independent_target_surface",
    "both",
    "only_model",
    "only_true",
    "neither",
    "true_visible",
    "model_comparable",
)


def check(value, msg):
    if not value:
        raise ValueError(msg)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(p):
    return json.loads(Path(p).read_bytes())


def save(p, v):
    with p.open("x", encoding="utf-8") as out:
        json.dump(v, out, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")


def rows(p):
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def i(d, k):
    return int(d[k]) if d.get(k) not in ("", None) else None


def f(d, k):
    return float(d[k]) if d.get(k) not in ("", None) else None


def key(row):
    return tuple(row[k] for k in DOMAIN_KEYS)


def field_tuple(z, member, field, prior=()):
    doc = ast.parse(z.read(member))
    target = next(
        a.value
        for a in doc.body
        if isinstance(a, ast.Assign) and any(isinstance(t, ast.Name) and t.id == field for t in a.targets)
    )
    result = []
    for e in target.elts:
        if isinstance(e, ast.Starred):
            check(
                isinstance(e.value, ast.Name) and e.value.id == "REASONS" and prior,
                "unexpected legacy reason prefix",
            )
            result.extend(prior)
        else:
            v = ast.literal_eval(e)
            check(isinstance(v, str), "non-string reason")
            result.append(v)
    return tuple(result)


def codebook(a):
    with ZipFile(a.base) as z:
        pref = field_tuple(z, "evidence_lens/multisource.py", "REASONS")
        old = field_tuple(z, "evidence_lens_bundle/scoring.py", "REASON_NAMES", pref)
    with ZipFile(a.prototype) as z:
        current = field_tuple(z, "evidence_lens_many/schema.py", "REASONS")
        legacy = ast.parse(z.read("evidence_lens_many/legacy.py"))
    code = []
    for n in ast.walk(legacy):
        if isinstance(n, ast.Dict):
            code.extend(
                [
                    ast.literal_eval(v)
                    for k, v in zip(n.keys, n.values)
                    if isinstance(k, ast.Constant) and k.value == "unavailable_source_reason_code"
                ]
            )
    check(code == [254], "unrecognized missing legacy Q4 source code")
    return {
        "schema": "el-b-study-reason-codebook-1",
        "legacy": {**{str(k): v for k, v in enumerate(old)}, "254": "unavailable_source_reason_code"},
        "many_source": {str(k): v for k, v in enumerate(current)},
        "legacy_extension": "254 is the named missing-source sentinel from the prototype legacy_q4 adapter. The original view reason codes come from the Stage A wheel.",
        "base_wheel_sha256": sha(a.base),
        "prototype_wheel_sha256": sha(a.prototype),
        "scope": "Code 0 marks available ranking. A nonzero code is specific to its family and belongs to a nonfinite rank. The independent RGB error label has its own mask. Whole-view terminal failures have no issued pixel reason code.",
    }


def intersection(raw, total_unknown, masked_unknown):
    if raw == 0 or masked_unknown == 0:
        return 0
    if masked_unknown == total_unknown:
        return raw
    if raw == total_unknown:
        return masked_unknown
    return None


def reason_tables(metrics, reasons, books, path):
    counts = {}
    aggregate = {}
    native = {}
    for r in reasons:
        d = native.setdefault(key(r), {})
        code = r["reason_code"]
        check(code not in d, "duplicate native reason row")
        d[code] = int(r["pixels"])
    fields = [
        *DOMAIN_KEYS,
        "family",
        "code",
        "reason_name",
        "analysis_canvas_count",
        "full_reference_count",
        "labelled_reference_count",
        "method_own_labelled_count",
        *COUNT_KEYS,
        "intersection_status",
    ]
    with path.open("x", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fields, lineterminator="\n")
        writer.writeheader()
        total = 0
        for r in metrics:
            if r["domain"] != "own":
                continue
            ident = key(r)
            family = "legacy" if r["method"] in ("MS-L1", "MS-ft") else "many_source"
            names = books[family]
            raw = native.pop(ident, {})
            can, ref, lab, fin, own = (
                int(r[k]) for k in ("canvas", "reference", "labelled", "finite_scores", "denominator")
            )
            check(
                0 <= own <= lab <= ref <= can
                and 0 <= own <= fin <= can
                and set(raw) <= set(names)
                and "0" not in raw
                and sum(raw.values()) == can - fin,
                "rank or reason partition differs",
            )
            total_unknown = can - fin
            label_unknown = lab - own
            full_finite = (
                fin
                if ref == can
                else own
                if lab == ref
                else ref
                if can == fin
                else 0
                if not ref or not fin
                else None
            )
            full_unknown = ref - full_finite if full_finite is not None else None
            den = dict(
                zip(
                    COUNT_KEYS,
                    (can, ref, lab, own, fin, total_unknown, full_unknown, label_unknown),
                    strict=True,
                )
            )
            counts[ident] = den
            scene_group = (r["scene"], r["cell"], r["method"])
            agg = aggregate.setdefault(
                scene_group, {"denominators": {k: 0 for k in den}, "reason_counts": {}, "views": 0}
            )
            agg["views"] += 1
            for k, num in den.items():
                agg["denominators"][k] = (
                    agg["denominators"][k] + num
                    if num is not None and agg["denominators"][k] is not None
                    else None
                )
            for code, n in names.items():
                candidate = raw.get(code, 0) if code != "0" else fin
                rc = intersection(candidate, total_unknown, full_unknown) if code != "0" else full_finite
                lc = intersection(candidate, total_unknown, label_unknown) if code != "0" else own
                own_count = 0 if code != "0" else own
                row = {k: r[k] for k in DOMAIN_KEYS}
                writer.writerow(
                    {
                        **row,
                        "family": family,
                        "code": code,
                        "reason_name": n,
                        "analysis_canvas_count": candidate,
                        "full_reference_count": rc,
                        "labelled_reference_count": lc,
                        "method_own_labelled_count": own_count,
                        **den,
                        "intersection_status": "exact"
                        if rc is not None and lc is not None
                        else "not_identified_by_sealed_domain_totals",
                    }
                )
                total += 1
                accum = agg["reason_counts"].setdefault(
                    n,
                    {
                        "code": int(code),
                        "canvas": 0,
                        "full_reference": 0,
                        "labelled_reference": 0,
                        "method_own_labelled": 0,
                    },
                )
                for k, num in (
                    ("canvas", candidate),
                    ("full_reference", rc),
                    ("labelled_reference", lc),
                    ("method_own_labelled", own_count),
                ):
                    accum[k] = accum[k] + num if accum[k] is not None and num is not None else None
    check(not native, "orphaned reason rows outside measured domains")
    return counts, aggregate, total


def total(rs, key):
    vals = [i(s, key) for s in rs]
    return sum(vals) if vals and all(v is not None for v in vals) else None


def scene_data(outcome, tables, agg):
    all_scenes = {s["scene"]: s for s in outcome["scenes"]}
    external = {}
    views = tables["views"]
    domains = tables["domain-ranks"]
    compare = tables["comparability"]
    for name in EXTERNAL:
        case = all_scenes[name]
        vv = sorted((v for v in views if v["scene"] == name), key=lambda v: v["view"])
        check(len(vv) == case["views"] == 10, "wrong external target plan")
        n = int(vv[0]["R"])
        check(all(int(v["R"]) == n for v in vv), "mixed external checkpoint")
        source = {(r["view"], r["cell"]): r for r in compare if r["scene"] == name}
        primary = []
        for v in vv:
            row = {
                k: i(v, k)
                for k in ("reference", "labelled", "errors", "common", "quota", "new_errors", "rule_errors")
            }
            row.update(
                view=v["view"],
                status=v["status"],
                delta=f(v, "new_minus_rule"),
                G_difference=f(v, "coverage4_minus_fixed4"),
                g={},
            )
            for cell in ("fixed-4", "coverage-4", "coverage-16"):
                r = source.get((v["view"], cell))
                row["g"][cell] = (
                    None
                    if not r or f(r, "correct_gap_rate") is None
                    else {
                        "missed_correct": i(r, "correct_with_gap"),
                        "correct": i(r, "all_reference_correct"),
                    }
                )
            primary.append(row)
        for k, prefix in (
            ("fixed-4", "fixed4"),
            ("coverage-4", "coverage4"),
            ("coverage-16", "coverage16"),
        ):
            if case[prefix + "_gap_pixels"] is not None:
                check(
                    all(v["g"][k] is not None for v in primary)
                    and sum(v["g"][k]["missed_correct"] for v in primary) == case[prefix + "_gap_pixels"]
                    and sum(v["g"][k]["correct"] for v in primary)
                    == case[prefix + "_correct_label_denominator"],
                    "per-view G linkage differs",
                )
        domain_view = {}
        for method, cell in [
            ("MS-L1", "coverage-4"),
            ("MS-ft", "coverage-4"),
            ("B-CPU-LR", "coverage-4"),
            ("B-CPU-LR", "coverage-16"),
            (outcome["strong_simple_rule"], "coverage-16"),
        ]:
            single = {}
            for domain in ("own", "same_query_methods", "all_checkpoint_ranks"):
                dd = [
                    r
                    for r in domains
                    if r["scene"] == name
                    and r["cell"] == cell
                    and r["method"] == method
                    and r["domain"] == domain
                ]
                single[domain] = {
                    "measured_views": len(dd),
                    "scoreable_labelled": total(dd, "denominator"),
                    "error_pixels": total(dd, "positive"),
                    "full_reference": total(dd, "reference"),
                    "labelled": total(dd, "labelled"),
                }
            reason = agg.get((name, cell, method))
            domain_view[cell + " " + method] = {
                "method": method,
                "cell": cell,
                "domains": single,
                "reasons": reason,
            }
        scopes = {}
        for cell in ("fixed-4", "coverage-4", "coverage-16", "coverage-" + str(n)):
            ss = [r for r in compare if r["scene"] == name and r["cell"] == cell]
            scopes[cell] = {
                "views": len(ss),
                "audits_complete": sum(r["audit_complete"] == "True" for r in ss),
                **{
                    k: total(ss, k)
                    for k in (
                        "selected_gap",
                        "labelled_selected_gap",
                        "library_no_comparable",
                        "library_incomplete",
                        "invalid_geometry",
                        "query_has_comparable",
                        "depth_conflicts",
                        "colour_conflicts",
                    )
                },
            }
        proxy = [
            {
                **{k: p[k] for k in ("method", "first", "second")},
                **{
                    k: i(p, k)
                    for k in (
                        "planned",
                        "opened",
                        "captured",
                        "omitted",
                        "total_label_errors",
                        "no_score_views",
                    )
                },
            }
            for p in tables["view-priority"]
            if p["scene"] == name
        ]
        external[name] = {
            "R": n,
            "native_endpoint": case,
            "planned": len(vv),
            "ready": sum(v["status"] == "ready" for v in vv),
            "evaluable_G": sum(v["G_difference"] is not None for v in primary),
            "evaluable_P2": case["complete_p2_views"],
            "view_counts": primary,
            "methods": domain_view,
            "audit_scopes": scopes,
            "view_priority": proxy,
        }
    author = []
    for s in outcome["scenes"]:
        if not s["scene"].startswith("b0"):
            continue
        for n in sorted({int(t["R"]) for t in tables["geometric-visibility"] if t["scene"] == s["scene"]}):
            vv = [
                t
                for t in tables["geometric-visibility"]
                if t["scene"] == s["scene"] and int(t["R"]) == n and t["cell"] == "coverage-4"
            ]
            if not vv:
                continue
            values = {k: total(vv, k) for k in PHYSICAL}
            check(
                values["independent_target_surface"]
                == sum(values[k] for k in ("both", "only_model", "only_true", "neither")),
                "physical contingency differs",
            )
            author.append({"family": s["scene"], "role": s["role"], "R": n, "views": len(vv), **values})
    check(
        all(r["scene"] not in EXTERNAL for r in tables["geometric-visibility"]),
        "unexpected external true geometry",
    )
    return external, author


def emit(a):
    cfg = read(a.config)
    pins = cfg["sources"]
    outcome = read(a.endpoints)
    check(
        outcome["schema"] == "el-many-scientific-endpoints-1"
        and pins["endpoints"]["sha256"] == sha(a.endpoints),
        "different scientific endpoints",
    )
    tables = {}
    for tag, fp in a.table:
        p = Path(fp)
        f = outcome["files"][tag + ".csv"]
        check(f["sha256"] == sha(p) and f["bytes"] == p.stat().st_size, "different scientific table " + tag)
        tables[tag] = rows(p)
        check(len(tables[tag]) == outcome["rows"][tag], "changed numeric row count")
    check(set(tables) == set(TABLES), "missing tables")
    books = codebook(a)
    save(a.out / "reason-codebook.json", books)
    denominators, aggregate, n = reason_tables(
        tables["domain-ranks"], tables["unknown-reasons"], books, a.out / "unknown-reason-counts.csv"
    )
    ext, physical = scene_data(outcome, tables, aggregate)
    deployment = read(a.deployment)
    performance = rows(a.perf)
    check(sha(a.perf) == deployment["tables"]["cpu-performance.csv"]["sha256"], "different CPU receipts")
    demo = read(a.example)
    check(
        demo["run_id"] == deployment["parents"]["example"]["run_id"]
        and sha(a.example) == deployment["parents"]["example"]["sha256"],
        "different authored example",
    )
    selection = demo["reports"]["audit-bank"]
    cpu = []
    for r in performance:
        if r["stage"] == "fitted":
            cpu.append(
                {
                    k: float(r[k])
                    for k in (
                        "thread_ceiling",
                        "spawn_to_cold_s",
                        "hot_p50_s",
                        "hot_p95_s",
                        "native_peak_rss_bytes",
                        "source_cache_peak_bytes",
                    )
                }
            )
    check(len(cpu) == 3 and set(ext) == set(EXTERNAL), "incomplete source records")
    missing_views = [
        {k: r[k] for k in ("scene", "R", "view", "status", "reference", "labelled")}
        for r in tables["views"]
        if r["status"] != "ready"
    ]
    meta = {
        "schema": "el-b-scientific-summary-html-data-1",
        "comparison_status": outcome["comparison_status"],
        "runtime_disposition": outcome["cost_and_implementation_acceptance"],
        "strong_simple_rule": outcome["strong_simple_rule"],
        "external": ext,
        "physical_visibility": physical,
        "authored_example": {
            "scene": demo["scene_id"],
            "target": demo["target"],
            "actual_query": demo["actual_q"],
            "scopes": selection["scope_counts"],
            "added_witness": selection["gap_witness_counts_by_source"],
        },
        "fitted_cpu": cpu,
        "unranked_view_terminals": missing_views,
        "reason_export_rows": n,
        "rank_domains_with_own_counts": len(denominators),
        "all_study_families": outcome["all_families"],
        "all_checkpoints": outcome["checkpoints"],
        "run_ids": {k: pins[k]["run_id"] for k in ("endpoints", "deployment", "example")},
        "scope": "Numerical research records only. External photographs and forecasts of physical visibility are outside this report. Authoritative endpoints use an equal mean over targets within checkpoint and equal checkpoint weights. Rendered illustration and reference images are excluded.",
    }
    save(a.out / "study-summary.json", meta)
    inline = json.dumps(
        meta, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False
    ).replace("<", "\\u003c")
    template = a.template.read_text(encoding="utf-8")
    check(template.count("__DATA__") == 1, "wrong template")
    html = template.replace("__DATA__", inline)
    (a.out / "report.html").write_text(html, encoding="utf-8")
    published = (a.out / "report.html").read_text(encoding="utf-8")
    blocks = re.findall(r'<script type="application/json" id="study-data">(.*?)</script>', published, re.DOTALL)
    check(
        len(blocks) == 1
        and blocks[0] == inline
        and json.loads(blocks[0]) == read(a.out / "study-summary.json"),
        "embedded study data differ from the independent JSON download",
    )
    browser = subprocess.run(
        [a.node, a.driver, a.chrome, a.out / "report.html"],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    check(browser.returncode == 0, "offline controls failed " + browser.stderr[-900:])
    smoke = json.loads(browser.stdout)
    save(a.out / "interface-check.json", smoke)
    check(
        smoke["external_http_requests_observed"] == 0
        and smoke["images"] == 0
        and smoke["verified_scenes"] == list(EXTERNAL),
        "unverified external-scene interface",
    )
    styles = re.findall(r"<style>(.*?)</style>", html, re.DOTALL)
    check(len(styles) == 1, "unregistered styles")
    inventory = {n: {"bytes": (a.out / n).stat().st_size, "sha256": sha(a.out / n)} for n in PUBLIC}
    check({x.name for x in a.out.iterdir()} == set(inventory), "unlisted outgoing artifact")
    save(
        a.out / "report.json",
        {
            "schema": "el-b-scientific-summary-html-1",
            "run_id": os.environ["EL_RUN_ID"],
            "parents": pins,
            "positive_list": inventory,
            "full_output_list": [*PUBLIC, "report.json"],
            "css_sha256": hashlib.sha256(styles[0].encode()).hexdigest(),
            "embedded_json_sha256": hashlib.sha256(inline.encode()).hexdigest(),
            "embedded_data_contract": "the compact JSON block parses to exactly study-summary.json",
            "reason_counts_exported": n,
            "rights": {
                "html_css_browser_script": "Original Evidence Lens work, wany614, Apache-2.0",
                "numeric_findings": "Project measurements and published numeric identifiers. ETH3D is credited as the external benchmark.",
                "visual_material": "No photographs, external renders, fonts or third-party visual assets. Browser uses local fonts.",
                "authored_demonstration": "Numeric counts only from wany614’s CC BY 4.0 authored dataset.",
            },
            "actual_software_paths": {
                "source_template_sha256": sha(a.template),
                "browser_test_program_sha256": sha(a.driver),
            },
        },
    )


def controller(a):
    from dual_records import bind, selected, stage

    s = a.store.resolve()
    p = selected(s, "b-scientific-results")
    deploy = selected(s, "b-bank-deployment-public")
    rt = selected(s, "b-score-runtime")
    pins = {
        "endpoints": p,
        "deployment": deploy,
        "example": selected(s, "b-bank-public-example"),
        "perf": bind(s, deploy["run_id"], "payload/cpu-performance.csv"),
        "base": bind(s, rt["run_id"], "inputs/basewheel/evidence_lens-0.5.0-py3-none-any.whl"),
        "prototype": bind(
            s, rt["run_id"], "inputs/prototype/evidence_lens_many_source-0.1.0.dev0-py3-none-any.whl"
        ),
    }
    cmd = [
        "{python}",
        "-B",
        "{source}/scripts/many_scientific_html.py",
        "worker",
        "--config",
        "{config}",
        "--out",
        "{out}",
        "--node",
        shutil.which("node"),
        "--chrome",
        str(a.chrome),
        "--template",
        "{input:template}",
        "--driver",
        "{input:driver}",
    ]
    for k in pins:
        cmd += ["--" + k, "{input:" + k + "}"]
    for tag in TABLES:
        role = tag.replace("-", "_")
        pins[role] = bind(s, p["run_id"], "payload/" + tag + ".csv")
        cmd += ["--table", tag, "{input:" + role + "}"]
    folder = Path(__file__).parent
    return stage(
        s,
        "b-scientific-summary-html",
        "report",
        pins,
        {
            "numerical_only": True,
            "source_figure_rights": "Original CSS, no embedded or remote images",
            "program_sha256": sha(Path(__file__)),
        },
        cmd,
        "payload/report.json",
        15000000,
        extra_files={
            "template": folder / "many_study_overview.html.tmpl",
            "driver": folder / "many_study_overview_smoke.mjs",
        },
        retry=a.retry,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "worker"))
    for n in (
        "store",
        "config",
        "out",
        "base",
        "prototype",
        "endpoints",
        "deployment",
        "example",
        "perf",
        "template",
        "driver",
        "node",
    ):
        p.add_argument("--" + n, type=Path)
    p.add_argument(
        "--chrome", type=Path, default=Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    )
    p.add_argument("--retry")
    p.add_argument("--table", nargs=2, action="append")
    a = p.parse_args()
    if a.action == "run":
        print(json.dumps(controller(a)))
    else:
        emit(a)
