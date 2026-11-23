"""Export a public, aggregate-only study report from an explicitly pinned successful run."""

import argparse
import csv
import io
import json
import math
import re
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from statistics import mean, median

from PIL import Image

from evidence_lens._bundle import HASH, RUN_ID, load, sha
from evidence_lens.formal_inputs import PinnedRun, require
from evidence_lens.formal_report import draw_curves
from evidence_lens.scoring_io import new_output, write_json

METHODS = ("constant", "1-alpha", "L0", "L1-geometry", "L1-full")
LABEL = {
    "constant": "Constant prior",
    "1-alpha": "1 − alpha",
    "L0": "L0 rule",
    "L1-geometry": "L1 geometry",
    "L1-full": "L1 full",
}
CONDITIONS = ("B0", "R1", "Q1")
VISIBILITY = {
    "observed": "Visible in the source view",
    "disoccluded": "Revealed by a target view",
    "outside_input_fov": "Outside the source field of view",
    "background_or_invalid": "Background or invalid reference geometry",
}
COUNT_COLUMNS = ("V", "E", "Y", "S", "SY", "L", "LY", "H", "HY", "U", "UY")
FAMILY_COLUMNS = (
    "family",
    "condition",
    "method",
    "common_two_class_views",
    "planned_views",
    "completed_views",
    *COUNT_COLUMNS,
    "common_ap",
    "own_ap",
    "low_coverage",
    "low_risk_error",
    "unknown_fraction",
    "tau",
)
PAIRED_METRICS = ("confirmed_error_fraction_of_V", "common_ap", "low_coverage", "rgb_mae")


def signature(path):
    return {"bytes": path.stat().st_size, "sha256": sha(path)}


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, records, fields):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def n(value):
    if value in (None, ""):
        return None
    num = float(value)
    require(math.isfinite(num), "nonfinite public statistic")
    return num


def fmt(value, digits=3, *, scale=1, signed=False):
    value = n(value)
    if value is None:
        return "—"
    return format(value * scale, ("+" if signed else "") + "." + str(digits) + "f")


def pct(value):
    return fmt(value, 1, scale=100)


def pp(value):
    return fmt(value, 2, scale=100, signed=True)


def family_name(family):
    words = re.sub(r"^scene_[0-9]+_", "", family).replace("_", " ").replace("-", " ")
    return words.replace("potters wheel", "potter’s wheel").capitalize()


def table(labels, body):
    return ["| " + " | ".join(map(str, line)) + " |" for line in (labels, ["---"] * len(labels), *body)]


def selected_pin(method, prefix):
    return {"run_id": method[prefix + "_run_id"], "manifest_sha256": method[prefix + "_manifest_sha256"]}


class Tables:
    def __init__(self, store, pin, fixture):
        self.store = store
        self.supplemental = {}
        self.run = PinnedRun(store, pin)
        rec = self.run.open("payload/report.json")
        self.receipt = load(rec)
        self.kind = "synthetic_fixture" if fixture else "heldout_study"
        require(
            self.run.run["run_kind"] == "report"
            and self.run.run["execution_status"] == "succeeded"
            and self.receipt["schema"] == "el-study-report-1"
            and self.receipt["kind"] == self.kind
            and self.receipt["run_id"] == pin["run_id"],
            "expected the pinned successful study report",
        )
        self.sources = {"report.json": signature(rec)}
        self.selection = load(self.open("public-numerical/selection.json"))
        self.method_path = self.open("numeric/method_freeze.json")
        self.catalog = load(self.open("numeric/catalog.json"))
        self.study = load(self.open("numeric/study.json"))
        self.method = load(self.method_path)
        self.families = sorted(f["scene_family_id"] for f in self.catalog["families"] if f["split"] == "test")
        self.trials = {(r["scene_family_id"], r["condition_id"]) for r in self.selection["comparison_trials"]}
        require(
            len(self.families) == 4
            and all(re.fullmatch(r"[A-Za-z0-9_-]+", f) for f in self.families)
            and self.trials == {(f, c) for f in self.families for c in CONDITIONS}
            and self.selection["kind"] == self.kind
            and self.receipt["quality_trials"] == 12
            and self.receipt["planned_targets"] == 72,
            "incomplete paired held-out roster",
        )
        require(
            self.selection["model_id"] == self.method["model_id"] == self.receipt["model_id"]
            and self.selection["method_freeze_sha256"] == self.sources["numeric/method_freeze.json"]["sha256"]
            and self.selection["comparison_protocol_sha256"]
            == self.receipt["comparison_protocol_sha256"]
            == self.method["comparison_protocol_sha256"]
            and self.selection["catalog_sha256"] == self.sources["numeric/catalog.json"]["sha256"]
            and self.selection["study_sha256"] == self.sources["numeric/study.json"]["sha256"]
            and self.selection["model_index_sha256"] == self.method["model_index_sha256"]
            and self.selection["method_code_commit"] == self.method["method_code_commit"],
            "report, method and public numerical selection disagree",
        )
        train = PinnedRun(store, selected_pin(self.method, "model_fit"))
        train_result = train.open("payload/fit.json")
        self.fit = load(train_result)
        require(
            train.run["run_kind"] == "model_fit"
            and train.run["execution_status"] == "succeeded"
            and self.fit["status"] == "fitted"
            and self.fit["fit_run_id"] == self.method["model_fit_run_id"],
            "unexpected fit receipt",
        )
        self.fit_receipt = signature(train_result)
        dev_member = f"numeric/evidence/{self.method['threshold_select_run_id']}/development.json"
        self.development = load(self.open(dev_member))
        require(
            self.sources[dev_member]["sha256"] == self.method["development_receipt_sha256"]
            and self.development["schema"] == "el-development-freeze-1"
            and self.development["status"] == "frozen"
            and self.development["run_id"] == self.method["threshold_select_run_id"]
            and self.development["fit_run_id"] == self.method["model_fit_run_id"]
            and self.development["model_index_sha256"] == self.method["model_index_sha256"],
            "development operating points differ from the frozen method",
        )

    def open(self, member):
        spec = self.receipt["files"][member]
        sealed = self.run.files["payload/" + member]
        require(
            spec == {k: sealed[k] for k in ("bytes", "sha256")}, "report/file seal disagreement: " + member
        )
        path = self.run.open("payload/" + member)
        self.sources[member] = spec
        return path

    def public(self, member):
        path = self.open("public-numerical/" + member)
        require(
            self.selection["files"][member] == self.sources["public-numerical/" + member],
            "public selection/file disagreement: " + member,
        )
        return rows(path)


def parse_family_tables(source):
    all_regions = source.public("family-results.csv")
    family = [r for r in all_regions if r["region"] == "all"]
    regional = {
        (r["family"], r["region"]): (int(r["Y"]), int(r["E"]))
        for r in all_regions
        if r["condition"] == "B0" and r["method"] == "constant" and r["region"] in (*VISIBILITY, "all")
    }
    by_family = {(r["family"], r["condition"], r["method"]): r for r in family}
    expected = {(f, c, m) for f, c in source.trials for m in METHODS}
    require(set(by_family) == expected and len(by_family) == len(family), "incorrect family/method inventory")
    for row in family:
        v, e, y, s, sy, l, ly, h, hy, u, uy = [int(row[k]) for k in COUNT_COLUMNS]
        require(
            0 <= s <= e <= v and 0 <= sy <= s and e == l + h + u and y == ly + hy + uy,
            "inconsistent reference, ranking or state counts",
        )
    aggregates = {
        (r["condition"], r["method"], r["metric"]): r
        for r in source.public("family-equal-summary.csv")
        if r["region"] == "all"
    }
    paired_rows = [
        r
        for r in rows(source.open("numeric/paired.csv"))
        if r["region"] == "all" and r["method"] == "L1-full"
    ]
    pairs = {(r["family"], r["variant"]): r for r in paired_rows}
    require(
        set(pairs) == {(f, c) for f in source.families for c in ("R1", "Q1")}
        and len(paired_rows) == len(pairs),
        "incorrect family-level paired inventory",
    )
    paired = {}
    for (f, c), row in pairs.items():
        cameras = row["paired_camera_ids"].split("|") if row["paired_camera_ids"] else []
        require(
            len(set(cameras)) == len(cameras)
            and all(v in ("v01", "v02", "v03", "v04", "v05", "v06") for v in cameras),
            "incorrect matched target count",
        )
        record = {"family": f, "variant": c, "matched_AP_views": len(cameras)}
        for metric in PAIRED_METRICS:
            a, b, d = [n(row[metric + suffix]) for suffix in ("_variant", "_B0", "_delta")]
            require(
                d is None
                if a is None or b is None
                else d is not None and math.isclose(d, a - b, abs_tol=1e-12),
                "invalid aggregate paired change",
            )
            record.update({metric + "_B0": b, metric + "_variant": a, metric + "_delta": d})
        paired[f, c] = record
    return by_family, aggregates, paired, regional


def visibility_text(families, regional):
    require(
        set(regional) == {(f, k) for f in families for k in (*VISIBILITY, "all")},
        "incomplete B0 visibility regions",
    )
    for f in families:
        parts = [regional[f, region] for region in VISIBILITY]
        require(all(0 <= y <= e for y, e in parts), "invalid regional error count")
        totals = (sum(y for y, e in parts), sum(e for y, e in parts))
        require(totals == regional[f, "all"], "B0 visibility regions do not partition Y and E")
    conditional = {}
    panel = []
    for region, label in {"all": "All evaluable pixels", **VISIBILITY}.items():
        shares = []
        errors, evaluable = 0, 0
        rates = []
        for f in families:
            y, e = regional[f, region]
            errors += y
            evaluable += e
            if regional[f, "all"][1]:
                shares.append(e / regional[f, "all"][1])
            if e:
                conditional[f, region] = y / e
                rates.append(y / e)
        share = mean(shares) if shares else None
        share_text = "<0.01" if share is not None and 0 < share < 0.0001 else fmt(share, 2, scale=100)
        panel.append(
            (
                label,
                f"{errors:,}",
                f"{evaluable:,}",
                share_text,
                fmt(mean(rates) if rates else None, 2, scale=100),
                f"{len(rates)}/{len(families)}",
            )
        )
    paired = [
        conditional[f, "disoccluded"] - conditional[f, "observed"]
        for f in families
        if (f, "disoccluded") in conditional and (f, "observed") in conditional
    ]
    text = [
        "The B0 counts below sum pixels across all four families. The share of E divides the region’s evaluable pixels by all evaluable pixels in that family. Error incidence divides errors in the region by evaluable pixels in that same region. Each percentage gives eligible families equal weight.",
        "",
        *table(
            (
                "B0 visibility region",
                "Errors (Y)",
                "Evaluable pixels (E)",
                "Share of E %",
                "Within-region errors / E %",
                "Families with region E",
            ),
            panel,
        ),
    ]
    if paired:
        text += [
            "",
            f"Surfaces revealed by a target view were occluded in the source. The mean within-family difference in error incidence, revealed minus visible, was {pp(mean(paired))} percentage points across {len(paired)} families with both regions.",
        ]
    return text


def draw_changes(path, families, paired):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    fig = Figure(figsize=(10.2, 3.45), dpi=135, facecolor="white", layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3, sharey=True)
    for ax, metric, title in zip(
        axes,
        ("confirmed_error_fraction_of_V", "common_ap", "low_coverage"),
        ("Confirmed RGB errors / V", "L1 full common AP", "L1 full low-risk coverage"),
        strict=True,
    ):
        values = []
        for condition, shift, marker, colour in (("R1", -0.13, "o", "#007681"), ("Q1", 0.13, "D", "#A34E1B")):
            for i, f in enumerate(families):
                value = paired[f, condition][metric + "_delta"]
                if value is not None:
                    values.append(100 * value)
                    ax.scatter(
                        100 * value,
                        len(families) - 1 - i + shift,
                        s=32,
                        marker=marker,
                        color=colour,
                        edgecolor="white",
                        linewidth=0.55,
                        zorder=3,
                    )
            if ax is axes[0]:
                ax.scatter([], [], s=32, marker=marker, color=colour, label=condition + " minus B0")
        extent = max([0.1, *(abs(v) * 1.14 for v in values)])
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-0.55, len(families) - 0.45)
        ax.axvline(0, color="#6A727A", linewidth=0.9)
        ax.grid(axis="x", linewidth=0.5, color="#D6DADF")
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
        ax.set_xlabel(
            "AP difference × 100" if metric == "common_ap" else "Percentage-point difference", fontsize=8
        )
        ax.tick_params(axis="both", labelsize=8, length=0)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#B6BBC0")
    axes[0].set_yticks(range(len(families)), [family_name(f) for f in reversed(families)])
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=9
    )
    fig.suptitle("Change from the same family’s baseline", fontsize=15, fontweight="bold", ha="left", x=0.01)
    raw = io.BytesIO()
    fig.savefig(raw, format="png")
    raw.seek(0)
    with Image.open(raw) as png:
        png.convert("RGB").save(path, optimize=True)


def magnitude(value):
    value = n(value)
    return "—" if value is None else format(value, ".3e")


def results_text(source, fam, agg, paired, regional):
    def averaged(condition, method, metric):
        return agg[condition, method, metric]["mean"]

    lead = ["# Which pixels in a new view can we trust?", ""]
    if source.kind == "synthetic_fixture":
        lead += ["**Synthetic integration fixture.**", ""]
    counts = Counter(f["split"] for f in source.catalog["families"])
    eligible = [(n(averaged("B0", m, "common_ap")), m) for m in ("constant", "1-alpha", "L0")]
    eligible = [(v, m) for v, m in eligible if v is not None]
    baseline = ""
    if eligible:
        best, m = max(eligible)
        baseline = f" The highest AP among the three simple baselines was {fmt(best)} ({LABEL[m]})."
    dev = source.development["methods"]["L1-full"]
    require(
        all(n(fam[f, "B0", "L1-full"]["tau"]) == n(dev["tau"]) for f in source.families),
        "held-out L1 full thresholds differ from development",
    )
    lead += [
        f"On four held-out scenes, L1 full ranked pixels by the risk of an RGB reconstruction error with an average precision (AP) of **{fmt(averaged('B0', 'L1-full', 'common_ap'))}** under the baseline condition.{baseline}",
        "",
        f"At the frozen B0 threshold, L1 full accepted **{fmt(dev['coverage'], 2, scale=100)}%** of reference-valid pixels as low risk on the four development families. **{fmt(dev['family_macro_low_risk_error_rate'], 2, scale=100)}%** of accepted pixels were errors. On the four held-out families, coverage was **{fmt(averaged('B0', 'L1-full', 'low_coverage'), 2, scale=100)}%** and the error rate among accepted pixels was **{fmt(averaged('B0', 'L1-full', 'low_risk_error'), 2, scale=100)}%**. Both metrics give eligible families equal weight.",
        "",
        f"Each scene starts from one image. Apple’s SHARP estimates a Gaussian scene representation. gsplat renders six new camera views, and Blender supplies independent reference images. An error is a mean absolute RGB channel difference greater than 0.05 on a display-sRGB scale from 0 to 1. The study assigned {counts['train']} original, static, opaque scene families to training, {counts['development']} to development and {counts['test']} to testing. The development families set the decision thresholds. Performance on real photographs and other generators is unmeasured. [Study protocol](../../protocol/study-v1.json)",
        "",
        "## Render quality",
        "",
        "B0 used a 512 × 512 input. R1 reduced it to 256 × 256 before generation. Q1 reused the B0 scene and rounded its colour coefficients through FP16. All conditions evaluated the same six cameras per test family at 512 × 512.",
        "",
    ]
    cond_rows = []
    for c in CONDITIONS:
        completed = sum(int(fam[f, c, "constant"]["completed_views"]) for f in source.families)
        cond_rows.append(
            (
                c,
                f"{completed}/24",
                *(
                    pct(averaged(c, "constant", m))
                    for m in (
                        "rgb_error_incidence",
                        "confirmed_error_fraction_of_V",
                        "invalid_fraction",
                        "common_coverage",
                    )
                ),
            )
        )
    lead += table(
        (
            "Condition",
            "Completed targets",
            "Errors / valid prediction %",
            "Confirmed errors / reference %",
            "Invalid output / reference %",
            "Common ranking pixels / reference %",
        ),
        cond_rows,
    )
    lead += [
        "",
        "The reference population **V** contains all pixels with valid reference RGB. **E** also requires valid predicted RGB. Confirmed errors **Y** are defined on E. The common ranking set **S** contains the pixels in E that all five methods can score. Percentages pool pixels across each family’s cameras first, then give every eligible family equal weight.",
        "",
        *visibility_text(source.families, regional),
        "",
        "## Ranking and the low-risk decision",
        "",
        "L1 geometry uses camera, depth and opacity features. L1 full also uses appearance differences. The baselines are the training-family error prior, one minus predicted opacity, and L0’s fixed geometric rule. Higher AP means that true errors receive higher scores. For the four methods with development-selected thresholds, the operating point was the attainable coverage nearest to 80%, keeping equal scores together. The constant prior assigns every scored pixel to one tie group. L0 retained its 0.5 decision boundary.",
        "",
    ]
    items = []
    for c in CONDITIONS:
        for m in METHODS:
            ap = agg[c, m, "common_ap"]
            low = agg[c, m, "low_risk_error"]
            items.append(
                (
                    c,
                    LABEL[m],
                    fmt(ap["mean"]),
                    f"{ap['eligible_family_count']}/4 · {ap['eligible_views']}/24",
                    pct(averaged(c, m, "low_coverage")),
                    pct(low["mean"]),
                    int(low["eligible_family_count"]),
                    pct(averaged(c, m, "unknown_fraction")),
                )
            )
    lead += table(
        (
            "Condition",
            "Method",
            "Common AP",
            "AP families · views",
            "Low / V %",
            "Errors / low %",
            "Families with low pixels",
            "Unknown / E %",
        ),
        items,
    )
    lead += [
        "",
        "Common AP averages only cameras containing both error classes on S, then averages their family means. A dash marks an undefined quantity. A pixel enters the low-risk set **L** when its score is strictly below the threshold. Coverage divides L by full V. Accepted risk divides confirmed errors in L by L. Missing predictions remain in V. Unknowns stay outside L.",
        "",
        "### The same four families",
        "",
        "Each cell gives **common AP / low-risk coverage (%)**. The view count gives the number of the six targets eligible for common-mask AP. Exact pixel counts, high and unknown states, own-mask AP and thresholds are in the downloadable table. [Family and method counts](family-methods.csv)",
        "",
    ]
    panel = []
    for f in source.families:
        for c in CONDITIONS:
            panel.append(
                (
                    family_name(f),
                    c,
                    f"{fam[f, c, 'L1-full']['common_two_class_views']}/6",
                    *(
                        fmt(fam[f, c, m]["common_ap"]) + " / " + pct(fam[f, c, m]["low_coverage"])
                        for m in METHODS
                    ),
                )
            )
    lead += table(("Family", "Condition", "AP views", *(LABEL[m] for m in METHODS)), panel)
    lead += [
        "",
        "![B0 attainable risk and coverage by held-out family](../../examples/study-figures/risk-coverage-B0.png)",
        "",
        "The horizontal axis is accepted pixels divided by V. The vertical axis is errors among accepted pixels. Each plotted location contains at least one attainable whole-score-tie group. Outlined circles mark frozen operating points with at least one accepted pixel. [R1 risk and coverage](../../examples/study-figures/risk-coverage-R1.png) · [Q1 risk and coverage](../../examples/study-figures/risk-coverage-Q1.png)",
        "",
        "## Paired changes",
        "",
    ]
    paired_macro = {
        (r["variant"], r["metric"]): r
        for r in source.public("paired-family-equal.csv")
        if r["region"] == "all" and r["method"] == "L1-full"
    }
    effect = paired_macro["R1", "confirmed_error_fraction_of_V"]
    lead += [
        f"The preregistered R1 outcome measures confirmed errors over all V. Reducing input resolution changed that rate by **{pp(effect['mean'])} percentage points** across {effect['eligible_family_count']} matched families. L1 full’s matched common AP changed by {fmt(paired_macro['R1', 'common_ap']['mean'], signed=True)} and its low-risk coverage by {pp(paired_macro['R1', 'low_coverage']['mean'])} percentage points. The AP differences pair cameras eligible in both conditions. Coverage uses the whole planned family population.",
        "",
        f"The constant prior’s common-mask AP was {fmt(averaged('B0', 'constant', 'common_ap'))} under B0 and {fmt(averaged('R1', 'constant', 'common_ap'))} under R1. Its AP equals the error prevalence on each eligible camera’s common mask. The L1 AP change is unadjusted for this shift in prevalence. [Paired numerical changes](paired-changes.csv) · [Registered intervention](../../protocol/analysis-addendum-v1.json)",
        "",
    ]
    lead += table(
        (
            "Family",
            "B0 Y/V %",
            "R1 Y/V %",
            "R1 change (pp)",
            "Matched AP views",
            "R1 change in L1 full AP",
            "R1 coverage change (pp)",
        ),
        [
            (
                family_name(f),
                pct(p["confirmed_error_fraction_of_V_B0"]),
                pct(p["confirmed_error_fraction_of_V_variant"]),
                pp(p["confirmed_error_fraction_of_V_delta"]),
                f"{p['matched_AP_views']}/6",
                fmt(p["common_ap_delta"], signed=True),
                pp(p["low_coverage_delta"]),
            )
            for f in source.families
            for p in (paired[f, "R1"],)
        ],
    )
    lead += [
        "",
        "![Within-family changes from B0 in error incidence, L1 full AP and coverage](../../examples/study-figures/paired-changes.png)",
        "",
    ]
    q1 = {r["family"]: r for r in source.public("Q1-numerical-response.csv")}
    detail = defaultdict(list)
    for r in rows(source.open("numeric/Q1-field-response.csv")):
        if r["camera"] != "source" and r["field"] == "L1-full" and r["status"] == "compared":
            detail[r["family"]].append(r)
    lines = []
    for f in source.families:
        row = q1[f]
        valid = sum(int(r["joint_finite"]) for r in detail[f])
        delta = sum(float(r["sum_abs"]) for r in detail[f]) / valid if valid else None
        peak = max((float(r["max_abs"]) for r in detail[f] if r["max_abs"]), default=None)
        lines.append(
            (
                family_name(f),
                f"{row['paired_views']}/{row['total_views']}",
                magnitude(row["mean_Q1_B0_rgb_response"]),
                magnitude(row["max_rgb_pixel_response"]),
                magnitude(delta),
                magnitude(peak),
                f"{row['geometry_bitwise_equal_views']}/{row['paired_views']}",
                f"{int(row['full_score_changed_values']):,}",
            )
        )
    lead += [
        "### Colour precision",
        "",
        "Q1 compares the rounded-colour render with the original B0 render. RGB changes are absolute three-channel means per jointly valid pixel. L1 full score changes use jointly finite scores. The last two columns count views with byte-identical geometry fields and individual full scores whose stored bits changed.",
        "",
    ]
    lead += table(
        (
            "Family",
            "Paired views",
            "Mean RGB change",
            "Largest pixel RGB change",
            "Mean L1 score change",
            "Largest L1 score change",
            "Geometry identical",
            "Changed L1 scores",
        ),
        lines,
    )
    source_views = [
        r["source_geometry_bitwise_equal"] for r in q1.values() if r["source_geometry_bitwise_equal"]
    ]
    area = sum(int(r["paired_rgb_pixels"]) for r in q1.values())
    changed = sum(int(r["changed_rgb_pixels"]) for r in q1.values())
    lead += [
        "",
        f"RGB changed on {changed:,} of {area:,} paired target pixels. In the separate source-view diagnostic, geometry was byte-identical for {sum(v == 'True' for v in source_views)}/{len(source_views)} available families. These counts exclude unavailable pairs.",
        "",
    ]
    return lead


def engineering_and_hardware(source, stages):
    smoke = []
    hardware = ""
    for stage in stages:
        rid = stage["run_id"]
        if (
            stage["purpose"] == "engineering"
            and stage["status"] == "failed"
            and stage["stage"] == "scene_render"
        ):
            pin = {k: stage[k] for k in ("run_id", "manifest_sha256")}
            trial = PinnedRun(source.store, pin)
            if (
                trial.run.get("trial_id") != "fixture-failure"
                or trial.run.get("exit_code") != 7
                or trial.run.get("run_kind") != "scene_render"
            ):
                continue
            cfg = trial.open("config.resolved.yaml")
            cmd = trial.open("command.json")
            args = load(cmd)["argv_template"]
            config = load(cfg)
            if (
                config.get("purpose") == "recorder only"
                and config.get("scientific_result") is False
                and len(args) == 3
                and args[:2] == ["{python}", "-c"]
                and "intentional fixture exit 7" in args[2]
                and "sys.exit(7)" in args[2]
            ):
                source.supplemental["deliberate_recorder_smoke"] = {
                    "run": pin,
                    "config": signature(cfg),
                    "command": signature(cmd),
                }
                smoke.append(
                    f"The engineering entry `{rid}` was a deliberate recorder smoke test tagged `scene_render`. Its Python worker exited with code 7 before any Blender rendering."
                )
        if not hardware and (stage["purpose"], stage["stage"], stage["platform_family"], stage["status"]) == (
            "scientific",
            "generation",
            "Windows",
            "succeeded",
        ):
            rel = f"numeric/evidence/{rid}/generation.json"
            if rel not in source.receipt["files"]:
                continue
            data = load(source.open(rel))
            name = data.get("runtime", {}).get("gpu", "")
            capacities = {
                v["cuda_total_bytes"] for v in data.get("phases", {}).values() if v.get("cuda_total_bytes")
            }
            if len(capacities) == 1 and re.fullmatch(r"NVIDIA GeForce RTX [0-9A-Za-z +.-]+", name):
                hardware = f"The {name} has {next(iter(capacities)) / 2**30:.0f} GiB of dedicated VRAM. "
    return smoke, hardware


def cost_text(source):
    stages = source.public("stage-times.csv")
    phase = source.public("worker-phase-times.csv")
    repeats = source.public("timing-repeats.csv")
    costs = source.public("paired-stage-times.csv")
    completions = source.public("data-completion.csv")
    measured = source.public("measured-receipts.csv")
    results = source.receipt
    local = [r for r in completions if r["split"] == "test"]
    completed = sum(int(r["heldout_completed_targets"]) for r in local)
    require(
        completed == results["completed_targets"] and len(local) == 12, "inconsistent held-out target count"
    )
    failure = Counter((r["purpose"], r["stage"]) for r in stages if r["status"] != "succeeded")
    retries = sum(r["is_retry"] == "True" for r in stages)
    detail = ", ".join(
        f"{v} {k.replace('_', ' ')} {stage.replace('_', ' ')}" for (k, stage), v in sorted(failure.items())
    )
    text = [
        "## Completion, cost and recovery",
        "",
        f"**{completed}/{results['planned_targets']} held-out target renders completed.** The run roster contains {len(stages):,} stage records. Of these, {sum(failure.values())} ended unsuccessfully{(' (' + detail + ')') if detail else ''} and {retries} have an explicit retry link. Stage attempts and held-out target completion have different denominators.",
        "",
    ]
    smoke, hardware = engineering_and_hardware(source, stages)
    for sentence in smoke:
        text.extend([sentence, ""])
    quality_stages = (
        "generation",
        "preparation",
        "render",
        "learned_features_and_scores",
        "five_method_comparison",
    )
    by_cond = []
    for c in CONDITIONS:
        batch = [r for r in costs if r["condition"] == c]
        parts = []
        for k in quality_stages:
            values = [n(r[k + "_seconds"]) for r in batch if r[k + "_seconds"]]
            value = fmt(median(values), 2) if values else "—"
            if 0 < len(values) < len(source.families):
                value += f" ({len(values)}/{len(source.families)})"
            parts.append("reuses B0" if c == "Q1" and k == "generation" else value)
        by_cond.append((c, *parts))
    generation_ids = {
        r["run_id"]
        for r in stages
        if r["stage"] == "generation"
        and r["purpose"] in ("scientific", "timing_repeat")
        and r["family"] in source.families
        and r["condition"] in ("B0", "R1")
        and r["status"] == "succeeded"
    }
    worker_phases = defaultdict(set)
    for record in phase:
        worker_phases[record["run_id"]].add(record["phase"])
    generation_scope = ""
    expected = {"load_model", "preprocess_forward_unproject", "export_ply"}
    if generation_ids and all(expected <= worker_phases[rid] for rid in generation_ids):
        generation_scope = " Generation wall time includes recorder setup, launching a separate worker for each run, loading the model, running prediction and exporting the Gaussian scene."
    text += [
        "Median recorded wall time in seconds across the four selected quality families, using each stage’s available receipts. Q1 reuses B0 generation."
        + generation_scope,
        "",
    ]
    text += table(
        ("Condition", "Generation", "Preparation", "Render", "CPU learned scoring", "Five-method evaluation"),
        by_cond,
    )
    text += [
        "",
        "Dedicated timing runs repeated one preregistered family three times per condition. The following medians use successful repeats. These runs are separate from the quality selection.",
        "",
    ]
    timing_labels = {
        "generation": "generation",
        "derivation": "colour derivation",
        "render": "render",
        "model_score": "CPU scoring",
    }
    repeat_rows = []
    for r in sorted(repeats, key=lambda z: (CONDITIONS.index(z["condition"]), z["stage"])):
        repeat_rows.append(
            (
                r["condition"],
                timing_labels.get(r["stage"], r["stage"]),
                f"{r['successful_repeats']}/3",
                fmt(r["median_seconds"], 2),
                fmt(r["minimum_seconds"], 2),
                fmt(r["maximum_seconds"], 2),
            )
        )
    if repeat_rows:
        text += table(
            ("Condition", "Stage", "Successful repeats", "Median s", "Minimum s", "Maximum s"), repeat_rows
        )
    else:
        text += ["The sealed report contains no dedicated timing measurements."]
    logical = sum(int(r["logical_bundle_bytes"]) for r in stages if r.get("logical_bundle_bytes"))
    rss = []
    for platform in sorted({r["platform_family"] for r in stages if r.get("peak_sampled_rss_bytes")}):
        peak = max(
            int(r["peak_sampled_rss_bytes"])
            for r in stages
            if r["platform_family"] == platform and r.get("peak_sampled_rss_bytes")
        )
        rss.append(f"{platform} {peak / 2**30:.2f} GiB")
    allocated = max(
        (int(r["cuda_peak_allocated_bytes"]) for r in phase if r.get("cuda_peak_allocated_bytes")),
        default=None,
    )
    reserved = max(
        (int(r["cuda_peak_reserved_bytes"]) for r in phase if r.get("cuda_peak_reserved_bytes")), default=None
    )
    text += [
        "",
        f"The listed run bundles total {logical / 2**30:.2f} GiB in logical member bytes. This includes repeated and staged files. Sampled peak process-tree RSS by platform was {', '.join(rss) or 'unavailable'}. {hardware}Across worker phases, the independent maxima of PyTorch’s CUDA allocator counters were {fmt(allocated / 2**30 if allocated is not None else None, 2)} GiB allocated and {fmt(reserved / 2**30 if reserved is not None else None, 2)} GiB reserved. Physical GPU memory residency was unmeasured.",
        "",
    ]
    metrics = defaultdict(list)
    fields = {
        "restored_runs",
        "verified_runs",
        "compared_files",
        "identical_outputs",
        "verified_outputs",
        "logical_files",
        "file_count",
        "index_runs",
        "index_receipts",
        "restored_receipts",
        "download_seconds",
        "upload_seconds",
        "passed",
    }
    for r in measured:
        if r["receipt_kind"] in ("recovery", "archive", "transfer") and r["field"] in fields:
            metrics[r["receipt_kind"], r["field"]].append(r["measured_value"])
    if metrics:
        text += [
            "The sealed inventory contains the following recovery, archive or transfer checks. Ranges are per receipt so repeated inspections of an archive do not inflate totals.",
            "",
        ]
        outcomes = []
        for (kind, field), values in sorted(metrics.items()):
            nums = [1.0 if v == "True" else 0.0 if v == "False" else float(v) for v in values]
            show = lambda v: f"{v:,.0f}" if v.is_integer() else f"{v:,.2f}"
            lo, hi = min(nums), max(nums)
            outcome = (
                ("passed" if lo == 1 else "failed")
                if field == "passed" and lo == hi
                else show(lo)
                if lo == hi
                else show(lo) + "–" + show(hi)
            )
            outcomes.append((kind.capitalize(), field.replace("_", " "), len(nums), outcome))
        text += table(
            ("Receipt kind", "Recorded measurement", "Receipts", "Observed value or range"), outcomes
        )
    else:
        text += ["The sealed inventory contains no numerical recovery or archive checks."]
    return text


def render(source, fam, agg, paired, regional):
    result = results_text(source, fam, agg, paired, regional) + cost_text(source)
    method = source.method
    result += [
        "",
        "## Evidence and reproduction",
        "",
        f"The fitted model used {source.fit['rows']:,} sampled training pixels, including {source.fit['positives']:,} errors. The method commit is `{method['method_code_commit']}`. Training run `{method['model_fit_run_id']}` produced the coefficients. Development run `{method['model_id']}` froze the thresholds. Report run `{source.receipt['run_id']}` produced the sealed aggregates. [Source digests and figure inventory](provenance.json)",
        "",
        "The source checkout contains the two numerical tables linked above and all four aggregate plots. Regeneration uses the pinned recorder archive with NumPy, Pillow and Matplotlib. Inside the matching **model ZIP**, `results/family-results.csv` holds the full region tables, `results/paired-family-equal.csv` holds paired family summaries, and `model/thresholds.json` contains the exact decisions. [Evaluation definitions](../formal-evaluation.md) · [Study reconstruction](../rebuild-study.md) · [Release contents](../model-release.md)",
        "",
    ]
    return "\n".join(result)


def export(store, pin, output, *, fixture=False):
    source = Tables(store, pin, fixture)
    fam, agg, paired, regional = parse_family_tables(source)
    with new_output(output) as root:
        docs, figures = root / "docs/results", root / "examples/study-figures"
        docs.mkdir(parents=True)
        figures.mkdir(parents=True)
        write_rows(
            docs / "family-methods.csv",
            [fam[f, c, m] for c in CONDITIONS for f in source.families for m in METHODS],
            FAMILY_COLUMNS,
        )
        pair_rows = [paired[f, c] for f in source.families for c in ("R1", "Q1")]
        write_rows(docs / "paired-changes.csv", pair_rows, tuple(pair_rows[0]))
        for c in CONDITIONS:
            curves = {(f, c): source.open(f"numeric/{f}-{c}-curve.npz") for f in source.families}
            draw_curves(figures / ("risk-coverage-" + c + ".png"), c, source.families, curves)
        draw_changes(figures / "paired-changes.png", source.families, paired)
        report = render(source, fam, agg, paired, regional)
        (docs / "README.md").write_text(report, encoding="utf-8")
        binary = {}
        for path in sorted(figures.iterdir()):
            with Image.open(path) as image:
                require(
                    image.mode == "RGB" and image.format == "PNG" and not image.info,
                    "unapproved figure metadata",
                )
                image.verify()
            binary[path.relative_to(root).as_posix()] = {"format": "rgb8-png", **signature(path)}
        provenance = {
            "schema": "el-public-study-summary-1",
            "kind": source.kind,
            "report_run": pin,
            "method_code_commit": source.method["method_code_commit"],
            "model_id": source.method["model_id"],
            "model_index_sha256": source.method["model_index_sha256"],
            "method_freeze_sha256": source.selection["method_freeze_sha256"],
            "comparison_protocol_sha256": source.method["comparison_protocol_sha256"],
            "protocol_sha256": source.selection["study_sha256"],
            "catalog_sha256": source.selection["catalog_sha256"],
            "training_run": selected_pin(source.method, "model_fit"),
            "training_receipt": source.fit_receipt,
            "training_manifest_sha256": source.fit["training_manifest_sha256"],
            "sources": source.sources,
            "supplemental_receipts": source.supplemental,
            "figure_sources": {
                "examples/study-figures/paired-changes.png": ["numeric/paired.csv"],
                **{
                    f"examples/study-figures/risk-coverage-{c}.png": [
                        f"numeric/{f}-{c}-curve.npz" for f in source.families
                    ]
                    for c in CONDITIONS
                },
            },
            "figure_software": {"Pillow": version("Pillow"), "matplotlib": version("matplotlib")},
            "binary_examples": binary,
            "document_files": {p.name: signature(p) for p in sorted(docs.iterdir())},
        }
        write_json(docs / "provenance.json", provenance)
    return {"report_run_id": pin["run_id"], "binary_examples": binary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("--report", help="sealed report run_id=manifest_sha256")
    inp.add_argument("--receipt", type=Path, help="recorder JSON with run_id and manifest_sha256")
    parser.add_argument(
        "--output", type=Path, required=True, help="new directory for docs/results and examples/study-figures"
    )
    parser.add_argument("--fixture", action="store_true", help="only accepts an explicitly synthetic report")
    args = parser.parse_args()
    if args.receipt:
        receipt = load(args.receipt)
        pin = {k: receipt[k] for k in ("run_id", "manifest_sha256")}
    else:
        rid, sep, digest = args.report.partition("=")
        require(sep and RUN_ID.fullmatch(rid) and HASH.fullmatch(digest), "supply run_id=manifest_sha256")
        pin = {"run_id": rid, "manifest_sha256": digest}
    print(
        json.dumps(
            export(args.store, pin, args.output, fixture=args.fixture), sort_keys=True, separators=(",", ":")
        )
    )


if __name__ == "__main__":
    main()
