"""Render labelled numerical plots for the recovery study. No case imagery or pixel arrays enter this module."""

from collections import defaultdict

from dtu_score import METHODS

COLORS = dict(zip(METHODS, ("#52687a", "#006e61", "#a74717", "#783e89"), strict=True))


def plotting():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "el-recovery-1",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )
    import matplotlib.pyplot as plt

    return plt


def save(fig, path, plt):
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None, "Creator": "Evidence Lens"})
    plt.close(fig)


def quality_curves(rows, path):
    plt = plotting()
    groups = defaultdict(list)
    for row in rows:
        if row["object_id"] == "set1" and row["appearance"] == "recovery":
            groups[row["condition"], row["recipe_id"], row["view_role"]].append(row)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(9, 3.6), constrained_layout=True)
    recipes = sorted({g[1] for g in groups})
    for ax, cond in zip(axes, ("M2", "M4"), strict=True):
        for recipe in recipes:
            for role, color, style in (("target", "#006e61", "-"), ("source", "#994517", "--")):
                series = sorted(groups[cond, recipe, role], key=lambda r: r["step"])
                marker = "o" if recipes.index(recipe) == 0 else "s"
                ax.plot(
                    [r["step"] for r in series],
                    [r.get("macro_view_MAE") for r in series],
                    marker=marker,
                    ms=4,
                    ls=style,
                    color=color,
                    label=f"{recipe} • {role}",
                )
        ax.set(title=cond, xlabel="Optimizer updates", xticks=(0, 1500, 3000, 5000))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("Mean of per-view RGB MAEs")
    fig.suptitle("set1 development • source fitting and held-out camera views")
    save(fig, path, plt)


def risk_curves(rows, path):
    plt = plotting()
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 7), constrained_layout=True)
    for i, obj in enumerate(("set1", "set6")):
        for j, cond in enumerate(("M2", "M4")):
            ax = axes[i, j]
            usable = [
                r
                for r in rows
                if r["object_id"] == obj
                and r["condition"] == cond
                and r["domain"] == "paired_old_new"
                and r["anchor"].endswith("%_V")
            ]
            for method in METHODS:
                for version, line in (("old", "--"), ("new", "-")):
                    series = sorted(
                        (r for r in usable if r["version"] == version and r["method"] == method),
                        key=lambda r: float(r["anchor"].split("%")[0]),
                    )
                    # Keep empty points as gaps. Each point pools counts only over views attaining its K.
                    ax.plot(
                        [float(r["anchor"].split("%")[0]) for r in series],
                        [None if r["risk"] is None else 100 * r["risk"] for r in series],
                        ls=line,
                        color=COLORS[method],
                        marker="o" if version == "new" else None,
                        ms=3,
                        lw=1.6,
                    )
            ax.set(title=f"{obj}  /  {cond}", xlim=(0.9, 85), ylim=(0, 101), xscale="log")
            ax.set_xticks([1, 2, 5, 10, 20, 40, 80], [1, 2, 5, 10, 20, 40, 80])
            ax.grid(axis="y", alpha=0.2)
            if j == 0:
                ax.set_ylabel("Confirmed RGB errors among accepted (%)")
            if i == 1:
                ax.set_xlabel("Requested fraction of reference-valid pixels (%)")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=color, label=name) for name, color in COLORS.items()]
    handles.extend(
        [
            Line2D([], [], color="#303030", ls="--", label="D1"),
            Line2D([], [], color="#303030", marker="o", ms=3, label="Recovery"),
        ]
    )
    fig.legend(handles=handles, ncol=6, loc="outside lower center", fontsize=8)
    fig.suptitle("Risk at matched coverage • shared pixels across both reconstructions and four methods")
    save(fig, path, plt)
