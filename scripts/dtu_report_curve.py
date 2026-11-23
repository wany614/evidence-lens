"""Draw the four registered DTU risk–coverage panels from a pinned public, numeric CSV."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

METHOD_STYLE = {
    "B-L1-transfer": ("#375d98", "-."),
    "MS-L1": ("#006d68", "-"),
    "MS-frozen": ("#ac5b00", "--"),
    "MS-ft": ("#9f3769", ":"),
}
SECTIONS = [
    ("debug", "set1", "M2"),
    ("debug", "set1", "M4"),
    ("external", "set6", "M2"),
    ("external", "set6", "M4"),
]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def parse_curve(path, expected):
    if sha(path) != expected:
        raise ValueError("Risk–coverage CSV has different bytes")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    panels = {}
    for role, obj, cond in SECTIONS:
        methods = {}
        panels[obj, cond] = methods
        for m in METHOD_STYLE:
            selected = [
                r
                for r in rows
                if (r["role"], r["object_id"], r["condition"], r["method"], r["arm"], r["domain"])
                == (role, obj, cond, m, "main-256", "full_crop")
            ]
            if not selected:
                methods[m] = None
                continue
            curve = [r for r in selected if r["type"] == "whole_group"]
            mark = [r for r in selected if r["type"] == "frozen"]
            if (
                len(curve) != 21
                or len(mark) != 1
                or sorted(float(r["requested_coverage"]) for r in curve) != [i / 20 for i in range(21)]
            ):
                raise ValueError("Incomplete curve or frozen operating point")
            v = int(mark[0]["V"])
            tau = float(mark[0]["tau"])
            seen = {}
            previous = -1

            def point(r):
                n, y = int(r["accepted"]), int(r["positive"])
                if not (0 <= y <= n <= v and int(r["V"]) == v and float(r["tau"]) == tau):
                    raise ValueError("Invalid published counts or threshold")
                if (
                    r["coverage"] == ""
                    if not v
                    else r["coverage"] != "" and abs(float(r["coverage"]) - n / v) <= 1e-15
                ) and (
                    r["risk"] == "" if not n else r["risk"] != "" and abs(float(r["risk"]) - y / n) <= 1e-15
                ):
                    return (n, y)
                raise ValueError("Published risk/coverage denominator does not match its counts")

            for r in sorted(curve, key=lambda x: float(x["requested_coverage"])):
                n, y = point(r)
                if n < previous or (n in seen and seen[n] != y):
                    raise ValueError("Whole-score acceptance is inconsistent")
                previous = n
                seen[n] = y
            if any(a > b for a, b in zip(seen.values(), list(seen.values())[1:])):
                raise ValueError("Cumulative errors decreased")
            methods[m] = {
                "V": v,
                "groups": [{"accepted": n, "positive": y} for n, y in seen.items() if n],
                "frozen": dict(zip(("accepted", "positive"), point(mark[0]))),
                "tau": tau,
            }
    return panels


def draw(panels, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.102, right=0.977, bottom=0.15, top=0.787, hspace=0.31, wspace=0.13)
    fig.suptitle(
        "Frozen RGB risk on DTU photographs",
        x=0.102,
        y=0.959,
        ha="left",
        size=22,
        fontweight="bold",
        color="#17333c",
    )
    fig.text(
        0.102, 0.895, "Error incidence among accepted pixels as coverage increases", size=11, color="#42545c"
    )
    handles = [
        Line2D([0], [0], color=color, linestyle=style, lw=2, label=m)
        for m, (color, style) in METHOD_STYLE.items()
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#17282d",
            marker="D",
            linestyle="None",
            markersize=6,
            markerfacecolor="white",
            label="Released threshold",
        )
    )
    fig.legend(
        handles=handles,
        ncol=5,
        loc="upper left",
        bbox_to_anchor=(0.095, 0.865),
        frameon=False,
        fontsize=9.5,
        handlelength=2.6,
        columnspacing=1.6,
    )
    for ax, (role, obj, condition) in zip(axes.flat, SECTIONS):
        for m, (color, style) in METHOD_STYLE.items():
            series = panels[obj, condition][m]
            if not series or not series["V"]:
                continue
            denom = series["V"]
            points = series["groups"]
            n = series["frozen"]["accepted"]
            y = series["frozen"]["positive"]
            ax.plot(
                [r["accepted"] / denom for r in points],
                [r["positive"] / r["accepted"] for r in points],
                linestyle=style,
                color=color,
                lw=2.15 if m == "MS-L1" else 1.8,
            )
            if n:
                ax.scatter(
                    n / denom, y / n, marker="D", s=35, facecolor="white", edgecolor=color, lw=1.5, zorder=5
                )
        ax.set_title(
            f"{obj}  ·  {'Integration' if role == 'debug' else 'External confirmation'}  ·  {condition}",
            loc="left",
            size=11.5,
            fontweight="bold",
            color="#17333c",
            pad=7,
        )
        ax.set_xlim(-0.012, 1.012)
        ax.set_ylim(-0.038, 1.038)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1, 0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
        ax.grid(color="#dce3e4", lw=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9, color="#bdcbce", labelcolor="#3c5158")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#bdcbce")
    fig.supylabel("RGB errors / accepted pixels", x=0.024, y=0.452, size=11)
    fig.supxlabel("Accepted pixels / valid reference pixels", x=0.545, y=0.084, size=11)
    fig.text(
        0.102,
        0.037,
        "Six targets per panel. Lines connect distinct recorded points. Empty accepted sets have undefined risk.",
        size=9,
        color="#48575c",
    )
    fig.savefig(
        path,
        format="svg",
        metadata={
            "Date": None,
            "Creator": "Evidence Lens",
            "Title": "Frozen RGB risk on DTU photographs",
            "Description": "Four panels comparing four fixed methods on set1 and set6 using the recorded accepted-pixel and RGB-error counts.",
        },
    )
    plt.close(fig)


def main(path, digest, out):
    panels = parse_curve(path, digest)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Choose an empty or new output directory")
    with tempfile.TemporaryDirectory(prefix="dtu-numeric-plot-") as scratch:
        os.environ["MPLCONFIGDIR"] = scratch
        import matplotlib

        with matplotlib.rc_context(
            {"svg.fonttype": "none", "svg.hashsalt": digest, "font.family": "DejaVu Sans"}
        ):
            draw(panels, out / "risk-coverage.svg")
    file = out / "risk-coverage.svg"
    receipt = {
        "schema": "el-dtu-numeric-curve-figure-1",
        "run_id": os.environ.get("EL_RUN_ID"),
        "source_csv_sha256": digest,
        "arm": "main-256",
        "domain": "full_crop",
        "panels": {o + "-" + c: x for (o, c), x in panels.items()},
        "file": {"path": file.name, "sha256": sha(file), "bytes": file.stat().st_size},
    }
    (out / "figure.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(main(a.csv, a.sha256, a.output)["file"]))
