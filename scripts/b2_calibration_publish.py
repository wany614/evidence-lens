"""Build a shareable numerical and graphical report from an explicitly pinned C6 fit and independent readback."""

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from b2_intake_publish import emit_table, put_copy, selected
from b2_intake_record import ROOT, locate, record_step, sealed_member
from t0_dtu_archive import check, digest, write_json


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(s):
    return float(s) if s not in ("", None) else None


def pct(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def plot(path, curves):
    image = Image.new("RGB", (1400, 735), "white")
    d = ImageDraw.Draw(image)
    font = lambda px: ImageFont.load_default(size=px)
    fg = "#1b2942"
    muted = "#43546b"
    grid = "#d6dfe7"
    d.text((38, 26), "C6 empirical error and coverage", fill=fg, font=font(34))
    d.text(
        (40, 75),
        "Full six-object fits. Each point is one frozen candidate. Curves connect thresholds in order.",
        fill=muted,
        font=font(20),
    )
    plotted = {
        c: [
            r
            for r in curves
            if r["condition"] == c
            and r["fold"] == "full"
            and r["mean_low_error"] != ""
            and r["mean_coverage"] != ""
        ]
        for c in ("M2", "M4")
    }
    observed = [
        float(r["mean_low_error"]) for series in plotted.values() for r in series
    ]
    upper = min(1, max(0.2, math.ceil(10 * max(observed, default=0.2)) / 10))
    for i, (c, series) in enumerate(plotted.items()):
        left = 102 + 685 * i
        right = left + 540
        top = 180
        bottom = 574
        xx = lambda v: left + v * (right - left)
        yy = lambda v: bottom - v / upper * (bottom - top)
        d.text((left, 122), c + " fixed layout", fill=fg, font=font(25))
        d.rectangle((xx(0.2), yy(0.05), right, bottom), fill="#e5f3e9")
        for p in (0, 0.2, 0.4, 0.6, 0.8, 1):
            d.line((xx(p), top, xx(p), bottom), fill=grid, width=1)
            d.text(
                (xx(p), bottom + 9),
                f"{100 * p:.0f}%",
                fill=muted,
                font=font(15),
                anchor="mt",
            )
        for p in (0, upper / 4, upper / 2, 3 * upper / 4, upper):
            d.line((left, yy(p), right, yy(p)), fill=grid, width=1)
            d.text(
                (left - 9, yy(p)),
                f"{100 * p:.0f}%",
                fill=muted,
                font=font(15),
                anchor="rm",
            )
        d.line((left, yy(0.05), right, yy(0.05)), fill="#438455", width=2)
        d.line((xx(0.2), top, xx(0.2), bottom), fill="#438455", width=2)
        for k in ("left", "bottom"):
            xy = (
                (left, top, left, bottom)
                if k == "left"
                else (left, bottom, right, bottom)
            )
            d.line(xy, fill=fg, width=2)
        pts = [
            (xx(float(r["mean_coverage"])), yy(float(r["mean_low_error"])))
            for r in series
        ]
        color = ("#2458a6", "#a3481e")[i]
        if len(pts) > 1:
            d.line(pts, fill=color, width=3)
        for x, y in pts:
            d.ellipse((x - 1.8, y - 1.8, x + 1.8, y + 1.8), fill=color)
        if not pts:
            d.text(
                (left + 20, top + 25),
                "No all-object nonempty points",
                fill=muted,
                font=font(17),
            )
        d.text(
            ((left + right) / 2, bottom + 43),
            "Mean coverage across objects (L/V)",
            fill=fg,
            font=font(20),
            anchor="mt",
        )
    d.text(
        (40, 655),
        "Vertical scale is mean confirmed error among accepted pixels (LY/L).",
        fill=muted,
        font=font(18),
    )
    d.text(
        (40, 688),
        "Green region requires at most 5% mean error, at least 20% mean coverage, and a nonempty acceptance set in every object.",
        fill="#386648",
        font=font(18),
    )
    image.save(path, format="PNG")


def export(cfg, out):
    pins = cfg["pinned"]
    roots = [Path(x) for x in cfg["read_only_roots"]]
    files = {role: locate(roots, pin) for role, pin in pins.items()}
    fit = read(files["fit"])
    audit = read(files["audit"])
    co = read(files["cohort"])
    check(
        audit["passed"]
        and audit["fit"] == pins["fit"]
        and fit["cohort"] == pins["cohort"]
        and audit["checked_object_anchor_counts"] == fit["per_object_candidate_rows"]
        and fit["T1_access"] == 0,
        "C6 readback does not verify this fit",
    )
    for name, h in fit["numerical_files"].items():
        check(
            digest((files["fit"].parent / name).read_bytes()) == h,
            "changed formal table",
        )
        put_copy(files["fit"].parent / name, out / name)
    for role, name in (
        ("fit", "calibration-results.json"),
        ("audit", "calibration-audit.json"),
        ("method", "calibration-method.json"),
    ):
        put_copy(files[role], out / name)
    for c in ("M2", "M4"):
        put_copy(files[c], out / "profiles" / (c + ".json"))
    curves = rows(out / "candidate-curves.csv")
    counts = rows(out / "candidate-object-counts.csv")
    original = rows(out / "original-method-object-counts.csv")
    original_rows = []
    for c in ("M2", "M4"):
        for m in ("B-L1-transfer", "MS-L1", "MS-frozen", "MS-ft"):
            rr = [r for r in original if (r["condition"], r["method"]) == (c, m)]
            rvals = [number(x["low_error"]) for x in rr]
            cvals = [number(x["coverage"]) for x in rr]
            check(len(rr) == 6, "original methods need all planned objects")
            originals = {
                k: sum(int(x[k]) for x in rr)
                for k in ("V", "E", "Y", "S", "L", "LY", "H", "HY", "U", "V_minus_E")
            }
            original_rows.append(
                {
                    "condition": c,
                    "method": m,
                    "objects": len(rr),
                    "nonempty_objects": sum(x is not None for x in rvals),
                    "mean_coverage": sum(cvals) / 6 if None not in cvals else None,
                    "mean_low_error_all_objects": sum(rvals) / 6
                    if None not in rvals
                    else None,
                    **originals,
                    "pooled_coverage": originals["L"] / originals["V"]
                    if originals["V"]
                    else None,
                    "pooled_low_error": originals["LY"] / originals["L"]
                    if originals["L"]
                    else None,
                }
            )
    emit_table(out / "original-method-family-means.csv", original_rows)
    outcomes = {}
    diagnostics = []
    for c in ("M2", "M4"):
        rr = [r for r in curves if r["condition"] == c and r["fold"] == "full"]
        nonempty = [
            r
            for r in rr
            if int(r["nonempty_objects"]) == 6
            and number(r["mean_low_error"]) is not None
        ]
        eligible = [r for r in nonempty if float(r["mean_coverage"]) >= 0.2]
        best = (
            min(
                eligible,
                key=lambda r: (
                    float(r["mean_low_error"]),
                    -float(r["mean_coverage"]),
                    float(r["tau"]),
                    int(r["anchor"]),
                ),
            )
            if eligible
            else None
        )
        low = [r for r in nonempty if number(r["mean_low_error"]) <= 0.05]
        highest = max((float(r["mean_coverage"]) for r in low), default=None)
        minimum = min((float(r["mean_low_error"]) for r in nonempty), default=None)
        selected = read(out / "profiles" / (c + ".json"))
        outcomes[c] = {
            "profile_status": selected["status"],
            "tau": selected["tau"],
            "best_at_least_20_percent": best,
            "maximum_coverage_at_5_percent": highest,
            "minimum_error_any_nonempty_coverage": minimum,
            "selected": selected["selected_estimates"],
            "loo_training_feasible": fit["outcomes"][c]["loo_feasible_folds"],
        }
        if best:
            diagnostics.extend(
                {
                    "diagnostic": "minimum mean error among all-object-nonempty cutoffs with mean coverage at least 0.2",
                    **r,
                }
                for r in counts
                if (r["condition"], r["fold"], r["anchor"])
                == (c, "full", best["anchor"])
            )
    if diagnostics:
        emit_table(out / "diagnostic-object-counts.csv", diagnostics)
    plot(out / "risk-coverage.png", curves)
    main = []
    for c, o in outcomes.items():
        r = o["best_at_least_20_percent"]
        xx = o["selected"]
        main.append(
            "| "
            + " | ".join(
                [
                    c,
                    "`" + o["profile_status"] + "`",
                    pct(xx["mean_coverage"]) if xx else "N/A",
                    pct(xx["mean_low_error"]) if xx else "N/A",
                    pct(float(r["mean_low_error"])) if r else "N/A",
                    pct(float(r["mean_coverage"])) if r else "N/A",
                    str(o["loo_training_feasible"]) + "/6",
                ]
            )
            + " |"
        )
    raw_main = []
    for t in original_rows:
        if t["method"] == "MS-L1":
            raw_main.append(
                f"{t['condition']} {pct(t['mean_coverage'])} coverage and {pct(t['mean_low_error_all_objects'])} accepted-pixel error"
            )
    extremes=[]
    for c in ("M2","M4"):
        rr=[v for v in diagnostics if v["condition"]==c]
        if rr:
            worst=max(rr,key=lambda r:float(r["low_error"]))
            smallest=min(rr,key=lambda r:float(r["coverage"]))
            extremes.append(f"{c} had its highest object error on {worst['object_id']} ({pct(float(worst['low_error']))}) and its lowest object coverage on {smallest['object_id']} ({pct(float(smallest['coverage']))}).")
    summary = f"""# C6 empirical operating points

Six physical objects supplied calibration labels for each fixed layout. The same six objects contribute to both layouts. The released MS-L1 score, 39 feature columns and valid/unknown decisions were fixed before C6. A pixel is a confirmed error when its mean absolute display-sRGB discrepancy exceeds 0.05. The empirical operating target requires mean accepted-pixel error at most 5%, mean coverage at least 20%, and at least one accepted pixel in every object. [Cohort and acquisition](../cohort/README.md) · [Frozen algorithm](calibration-method.json)

| Layout | Profile status | Selected mean coverage | Selected mean error | Lowest mean error at ≥20% coverage | Coverage at that diagnostic point | Feasible training folds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(main)}

Coverage is L/V and accepted-pixel error is LY/L. Counts pool the six camera views within each object, then the table equally weights the six objects. Empty acceptance gives undefined error. The diagnostic point shown above is one measured point on the full curve. Where fitting produced no feasible operating point, the profile contains a null threshold. The original released MS-L1 cutoff gave {" and ".join(raw_main)}. [Original four-method totals](original-method-family-means.csv) · [Object counts at every candidate](candidate-object-counts.csv)

At the diagnostic points, {" ".join(extremes)} [All six objects at both diagnostic points](diagnostic-object-counts.csv)

![C6 risk and coverage across all candidate cutoffs](risk-coverage.png)

Each full fit and each leave-one-object-out fit independently formed 201 candidates from the relevant objects’ frozen deployment scores. The six leave-one-object-out fits per layout never used the excluded object's scores or labels to select its threshold. The score-only commitment preceded collective target opening. An independent readback checked all 2,814 candidate cutoffs and 14,472 object-by-cutoff rows against the native predictions, risk maps and registered photographs. [All curves](candidate-curves.csv) · [All 12 held-out evaluations](leave-one-out.csv) · [Independent readback](calibration-audit.json)

The two profiles apply to the registered fixed M2 or M4 layout, the frozen reconstruction recipe and gsplat 1.5.3. C6 supplies empirical calibration evidence. T1 remains unevaluated here. Models, thresholds and domains for earlier roles retain their published identities. The numerical evidence supplies no probability calibration or cross-asset guarantee. [M2 profile](profiles/M2.json) · [M4 profile](profiles/M4.json) · [Numerical provenance](public-evidence.json)
"""
    (out / "README.md").write_text(summary)
    write_json(
        out / "public-evidence.json",
        {
            "schema": "el-b2-public-calibration-evidence-1",
            "cohort": pins["cohort"],
            "scientific_results": pins["fit"],
            "independent_audit": pins["audit"],
            "algorithm": pins["method"],
            "profiles": {c: pins[c] for c in ("M2", "M4")},
            "outcomes": outcomes,
            "source_figure": "candidate-curves.csv",
            "files": {
                p.relative_to(out).as_posix(): digest(p.read_bytes())
                for p in sorted(out.rglob("*"))
                if p.is_file()
            },
            "T1_evaluated": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("export", "worker"))
    for k in ("store", "batch-one", "history", "fit", "audit", "dest", "config", "out"):
        p.add_argument("--" + k, type=Path)
    a = p.parse_args()
    if a.action == "worker":
        export(read(a.config), a.out)
    else:
        roots = (
            a.store.resolve(),
            a.batch_one.resolve(),
            (a.history or ROOT / ".research").resolve(),
        )
        f = read(a.fit)
        outcome = read(locate(roots, f))
        pins = {
            "fit": f,
            "audit": read(a.audit),
            "cohort": outcome["cohort"],
            "method": outcome["calibration_method"],
            **{
                c: sealed_member(roots, f["run_id"], f"payload/profiles/{c}.json")
                for c in ("M2", "M4")
            },
        }
        pf = record_step(
            roots,
            "b2-public-calibration-v2",
            "b2_calibration_publish.py",
            "payload/public-evidence.json",
            {"read_only_roots": list(map(str, roots))},
            pins,
            worker_args=("worker",),
            extra_code=[
                "scripts/b2_intake_publish.py",
                "scripts/b2_intake_record.py",
                "src/evidence_lens/_bundle.py",
                "src/evidence_lens/_verify.py",
            ],
            estimate=8 * 1024**2,
        )
        if a.dest:
            file = locate(roots, pf)
            for name in [*read(file)["files"], "public-evidence.json"]:
                put_copy(file.parent / name, a.dest / name)
        print(json.dumps(pf))
