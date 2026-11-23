"""Read the four registered Stage C train/development histories and reproduce their macro AP."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean


def load(path):
    return path, json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, why):
    if not value:
        raise ValueError(why)


def agree(left, right, what):
    require(math.isclose(left, right, abs_tol=1e-12, rel_tol=0), "Macro AP differs for " + what)


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compute(config, catalog, study, fits, out):
    require(not any(out.iterdir()), "Output directory must be empty")
    require(digest(catalog) == config["catalog_sha256"] and digest(study) == config["study_sha256"],
            "The registered protocol or catalog differs")
    _, registry = load(catalog)
    _, protocol = load(study)
    dev = sorted(f["scene_family_id"] for f in registry["families"] if f["split"] == "development")
    schedule = {(f, c, f"v{i:02d}") for f in dev for c in ("M2", "M4") for i in range(1, 7)}
    require(len(dev) == 4 and len(schedule) == 48, "Inconsistent Stage C development roster")
    neural = protocol["neural"]
    planned = {(m, seed) for m in neural["branches"] for seed in neural["candidate_seeds"]}
    require(len(fits) == len(planned) == 4 and neural["max_epochs"] == 12,
            "Expected the original four fits and registered 12-epoch cap")
    epoch_rows, family_rows, view_rows, inputs = [], [], [], []
    seen, paired, common = set(), {}, {}
    for curves_path, training_path in fits:
        _, training = load(training_path)
        key = training["method"], training["seed"]
        pin = config["fits"][training["run_id"]]
        require(key in planned and key not in seen and key == (pin["method"], pin["seed"])
                and training["phase"] == "formal", "Unknown, repeated or nonformal fit")
        require(digest(curves_path) == pin["curves_sha256"] and digest(training_path) == pin["training_sha256"],
                "Training receipt or curve differs from the source manifest")
        require(training["training_dataset_sha256"] == config["train_sha256"]
                and training["development_dataset_sha256"] == config["dev_sha256"],
                "Unexpected train or development input")
        seen.add(key)
        _, epochs = load(curves_path)
        require(len(epochs) == training["epochs"] == neural["max_epochs"], "Missing original epoch")
        design = (training["initial_head_sha256"], training["paired_design_sha256"],
                  tuple(e["view_order_sha256"] for e in epochs))
        require(paired.setdefault(key[1], design) == design, "Paired initialization or view order differs")
        means = []
        for i, epoch in enumerate(epochs, 1):
            result = epoch["development"]
            rows = result["views"]
            require(epoch["epoch"] == i and epoch["optimizer_steps"] == i * 24 and
                    len(rows) == len(schedule) and {(r["family"], r["condition"], r["view"]) for r in rows}
                    == schedule, "Incomplete curve or scheduled development views")
            base = {"method": key[0], "seed": key[1], "epoch": i}
            by_family = {f: {"M2": [], "M4": []} for f in dev}
            for r in rows:
                cell = r["family"], r["condition"], r["view"]
                support = (r["selection_support_sha256"], r["n"], r["positive"], r["negative"], r["E"], r["V"])
                require(common.setdefault(cell, support) == support and r["scope"] == "all"
                        and r["n"] == r["positive"] + r["negative"] and 0 < r["n"] <= r["E"] <= r["V"]
                        and r["status"] == "two_class" and r["positive"] > 0 and r["negative"] > 0
                        and 0 <= r["ap"] <= 1, "Invalid or changed score-all development domain")
                by_family[cell[0]][cell[1]].append(r["ap"])
                view_rows.append({**base, **{k: r[k] for k in ("family", "condition", "view", "V", "E", "n",
                                                                "positive", "negative", "ap")}})
            family_means = []
            for family, conditions in by_family.items():
                cond = {k: fmean(vs) for k, vs in conditions.items()}
                for k, ap in cond.items():
                    cell = result["macro"]["condition"][family][k]
                    require(cell["defined_views"] == cell["scheduled_views"] == 6, "Missing AP denominator")
                    agree(ap, cell["value"], family + "/" + k)
                val = fmean(cond.values())
                agree(val, result["macro"]["family"][family]["value"], family)
                family_rows.append({**base, "family": family, "M2_AP": cond["M2"], "M4_AP": cond["M4"],
                                    "family_AP": val})
                family_means.append(val)
            ap = fmean(family_means)
            agree(ap, result["family_equal_ap"], str(key))
            agree(ap, result["macro"]["value"], str(key))
            loss = epoch["weighted_mean_BCE"]
            require(math.isfinite(loss) and loss >= 0, "Invalid training loss")
            means.append(ap)
            epoch_rows.append({**base, "optimizer_steps": epoch["optimizer_steps"],
                               "weighted_train_BCE": loss, "development_family_equal_AP": ap})
        require(training["best_epoch"] == means.index(max(means)) + 1 and
                training["optimizer_steps"] == epochs[-1]["optimizer_steps"], "Selected checkpoint differs")
        agree(training["best_development_family_equal_ap"], max(means), "selected epoch")
        inputs.append({"run_id": training["run_id"], **pin, "optimization_parent": training.get("optimization_parent"),
                       "best_epoch": training["best_epoch"], "optimizer_steps": training["optimizer_steps"]})
    require(seen == planned, "Incomplete paired roster")
    write_csv(out / "epochs.csv", epoch_rows)
    write_csv(out / "families.csv", family_rows)
    write_csv(out / "views.csv", view_rows)
    return {"schema": "el-d0-curve-readback-1", "release_commit": config["release_commit"],
            "catalog_sha256": config["catalog_sha256"], "study_sha256": config["study_sha256"],
            "training_dataset_sha256": config["train_sha256"], "development_dataset_sha256": config["dev_sha256"],
            "retained_fits": inputs, "readback": {"fit_epochs": len(epoch_rows), "family_epochs": len(family_rows),
            "view_epochs": len(view_rows), "native_distinct_development_views": len(schedule),
            "common_pixel_denominator_per_epoch_per_fit": sum(s[1] for s in common.values()),
            "positive_events_per_epoch_per_fit": sum(s[2] for s in common.values())},
            "columns": {"weighted_train_BCE": "Training binary cross entropy, equally weighted by family, acquisition condition, then view.",
                        "ap": "Native average precision on the registered common finite development mask.",
                        "n": "Comparable score-all development pixels, including both classes.",
                        "positive": "RGB high-error events on that development mask.",
                        "family_AP": "Mean of the two acquisition-condition means of six view APs.",
                        "development_family_equal_AP": "Mean of the four development family APs."},
            "files": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in sorted(out.glob("*.csv"))}}


def plot(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with table.open(newline="", encoding="utf8") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for method, color in (("MS-frozen", "#14675e"), ("MS-ft", "#ac462c")):
        for seed, line in ((1729, "-"), (2718, "--")):
            curve = [r for r in rows if r["method"] == method and int(r["seed"]) == seed]
            x = [int(r["epoch"]) for r in curve]
            for ax, name in zip(axes, ("weighted_train_BCE", "development_family_equal_AP")):
                ax.plot(x, [float(r[name]) for r in curve], line, color=color, lw=2, label=f"{method}, {seed}")
    for ax, label in zip(axes, ("Weighted training BCE", "Development family-equal AP")):
        ax.set(xlabel="Epoch", ylabel=label, xticks=[1, 3, 6, 9, 12], xlim=(0.75, 12.25))
        ax.grid(axis="y", alpha=0.24)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=145, metadata={"Software": "Evidence Lens D0"})
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="mode", required=True)
    read = subs.add_parser("read")
    read.add_argument("--config", required=True, type=Path)
    read.add_argument("--study", required=True, type=Path)
    read.add_argument("--catalog", required=True, type=Path)
    read.add_argument("--fit", action="append", required=True, nargs=2, type=Path, metavar=("CURVES", "TRAINING"))
    read.add_argument("--output", type=Path, required=True)
    draw = subs.add_parser("plot")
    draw.add_argument("--epochs", type=Path, required=True)
    draw.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.mode == "plot":
        plot(args.epochs, args.output)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        result = compute(load(args.config)[1], args.catalog, args.study, args.fit, args.output)
        plot(args.output / "epochs.csv", args.output / "learning-curves.png")
        result["files"]["learning-curves.png"] = {"sha256": digest(args.output / "learning-curves.png"),
                                                  "bytes": (args.output / "learning-curves.png").stat().st_size}
        (args.output / "readback.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf8")
