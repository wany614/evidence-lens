"""Render the registered operating points of every Stage B family from the public numerical report."""

import argparse
import hashlib
import io
import json
from pathlib import Path

COHORTS = {"new_confirmation": "New confirmation", "historical_regression": "Historical regression"}
CONDITIONS = {"B0": "native input", "R1": "half-resolution input", "Q1": "colour coefficients rounded to FP16"}
METHODS = {
    "constant": ("#555b64", "s"),
    "1-alpha": ("#8759a2", "+"),
    "L0": ("#a2661c", "^"),
    "L1-geometry": ("#5591aa", "x"),
    "L1-full": ("#006959", "o"),
    "L2-frozen": ("#4356b0", "v"),
    "L2-vision-ft": ("#bd365b", "."),
}


def coordinates(curve):
    """Keep distinct, attainable score groups and compute the frozen operating point from counts."""
    points = curve["points"]
    if [p["requested_coverage"] for p in points] != [i / 100 for i in range(0, 101, 5)]:
        raise ValueError("Expected the 21 registered coverage requests")
    valid = points[0]["V"]
    realized = {}
    for p in points:
        low, errors = p["accepted"], p["positive"]
        if p["V"] != valid or not 0 <= errors <= low <= valid:
            raise ValueError("Inconsistent risk–coverage counts")
        coverage = low / valid if valid else None
        risk = errors / low if low else None
        if p["coverage"] != coverage or p["risk"] != risk or (low in realized and realized[low] != risk):
            raise ValueError("Reported point differs from its denominator")
        if low:
            realized[low] = risk
    low, errors = curve["frozen_L"], curve["frozen_LY"]
    if not 0 <= errors <= low <= valid:
        raise ValueError("Invalid frozen operating counts")
    frozen = (low / valid, errors / low) if low else None
    return [(n / valid, value) for n, value in sorted(realized.items())], frozen


def plot(cohort, title, condition, output):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    from PIL import Image

    families = sorted(cohort["families"])
    if not families or len(families) % 2:
        raise ValueError("Expected a registered two-column family cohort")
    fig = Figure(figsize=(10.7, 2.28 * (len(families) // 2) + 1.55), dpi=148, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    axes = fig.subplots(len(families) // 2, 2, squeeze=False, sharex=True, sharey=True).flat
    fig.subplots_adjust(left=.075, right=.975, bottom=.68 / fig.get_figheight(),
                        top=1 - 1.12 / fig.get_figheight(), hspace=.43, wspace=.12)
    fig.text(.075, 1 - .25 / fig.get_figheight(), f"{title}  ·  {condition}", size=18, weight="bold", va="center")
    fig.text(.075, 1 - .49 / fig.get_figheight(), CONDITIONS[condition].capitalize() + "   |   Six targets per family", size=10)
    handles = [Line2D([], [], color=color, marker=marker, linestyle="none", markersize=6, label=name)
               for name, (color, marker) in METHODS.items()]
    handles.append(Line2D([], [], color="#262b32", marker="D", markerfacecolor="none", linestyle="none", label="Frozen threshold"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, 1 - .54 / fig.get_figheight()),
               frameon=False, ncol=8, fontsize=8, handletextpad=.22, columnspacing=.68)
    for ax, family in zip(axes, families, strict=True):
        series = cohort["family_risk_coverage"][family + "/" + condition]
        if set(series) != set(METHODS):
            raise ValueError("Missing registered method for " + family)
        observed = False
        for method, (color, marker) in METHODS.items():
            points, frozen = coordinates(series[method])
            if points:
                x, y = zip(*points)
                # Continuous ranking methods have many score groups. Discrete baselines show attainable points only.
                ax.plot(x, y, color=color, marker=marker, markersize=3.25, linewidth=.9,
                        linestyle="-" if len(points) > 3 else "none", alpha=.90)
                observed = True
            if frozen is not None:
                ax.plot(*frozen, marker="D", color=color, markerfacecolor="none", markeredgewidth=1.1, linestyle="none", markersize=6)
        if not observed:
            ax.text(.5, .48, "No defined accepted-risk values", ha="center", transform=ax.transAxes)
        _, number, label = family.split("_", 2)
        ax.set_title(number + "  " + label.replace("_", " ").capitalize(), loc="left", size=10, weight="bold", pad=5)
        ax.set(xlim=(-.025, 1.025), ylim=(-.025, 1.025), xticks=[0, .25, .5, .75, 1], yticks=[0, .25, .5, .75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(axis="both", labelsize=8, length=2, color="#a5aab0", pad=3)
        ax.grid(True, color="#dee2e6", linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#a6adb4")
    fig.supylabel("RGB errors among accepted pixels", x=.008, size=10)
    fig.text(.525, .32 / fig.get_figheight(), "Accepted pixels / reference-valid pixels", ha="center", size=10)
    fig.text(.075, .065 / fig.get_figheight(), "Points retain whole score ties. An empty accepted set has undefined risk. Diamonds use thresholds frozen on development.", size=8, color="#454c53")
    raw = io.BytesIO()
    canvas.print_png(raw)
    with Image.open(raw) as image:
        image.convert("RGB").save(output, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--sha256", required=True, help="study/study-summary.json digest in release.json")
    parser.add_argument("--output", type=Path, required=True, help="empty or new output directory")
    args = parser.parse_args()
    data = args.summary.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.sha256:
        raise SystemExit("Public study summary differs from the supplied SHA-256")
    doc = json.loads(data)
    if doc["schema"] != "el-vision-public-study-1" or set(doc["cohorts"]) != set(COHORTS):
        raise SystemExit("Expected the two Stage B cohorts")
    for key, expected in (("new_confirmation", 8), ("historical_regression", 4)):
        families = doc["cohorts"][key]["families"]
        if len(families) != expected or len(set(families)) != expected:
            raise SystemExit("Family count differs from the registered cohort")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise SystemExit("Use a new or empty plot directory")
    for cohort, title in COHORTS.items():
        for condition in CONDITIONS:
            plot(doc["cohorts"][cohort], title, condition, args.output / f"{cohort}-{condition}.png")


if __name__ == "__main__":
    main()
