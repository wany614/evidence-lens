"""Draw Gaussian population and sampled resource curves from the exported fit measurements."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path

from evidence_lens._bundle import asset_path, load, sha
from evidence_lens._verify import require


COLORS = {"M2": "#346492", "M4": "#ac4e28"}


def read_steps(path, allow_partial):
    result = []
    with path.open("rb") as trace:
        for raw in trace:
            try:
                result.append(json.loads(raw))
            except (ValueError, UnicodeDecodeError):
                require(
                    allow_partial and not raw.endswith(b"\n") and not trace.read(1),
                    "Invalid complete update row",
                )
                return result, len(raw)
    return result, 0


def parse_events(step):
    before, after = step["n_before"], step["n_after"]
    require(type(before) is type(after) is int and before > 0 and after > 0, "Invalid Gaussian population")
    current = before
    checked = 0
    admitted = None
    grown = 0
    rows = []
    for i, event in enumerate(step["topology"]):
        kind = event["operation"]
        if kind == "growth_budget":
            require(admitted is None, "Duplicate growth admission in one update")
            admitted = event["admitted"]
            candidates, room = event["candidates"], event["available"]
            require(
                min(candidates, room) >= 0
                and event["admitted"] == min(room, candidates)
                and event["blocked"] == candidates - event["admitted"],
                "Growth budget differs",
            )
        else:
            require(
                kind in {"prune", "split", "clone", "reset"} and event["adam_moments_checked"] is True,
                "Unverified Adam topology event",
            )
            delta = {"prune": -1, "clone": 1, "split": 1, "reset": 0}[kind]
            selected = event["selected"]
            require(
                type(selected) is int
                and 0 <= event["affected_values"] <= selected <= current
                and (kind != "reset" or current == selected)
                and event["before"] == current
                and event["after"] == current + delta * selected,
                "Topology event does not balance",
            )
            current = event["after"]
            if kind in ("clone", "split"):
                grown += selected
            checked += 1
        rows.append({"event_index": i, **event})
    require(grown == (admitted or 0), "Added Gaussian rows differ from the admitted growth budget")
    require(current == after, "Reported step transition differs from topology operations")
    return rows, checked


def fit_measurements(root, meta, cfg, membership, asset_store=None):
    """Join the original optimizer log, issued points and final summary. Partial tails stay marked."""
    trace_path = root / "payload/steps.jsonl"
    report_path = root / "payload/reconstruction.json"
    point_path = root / "payload/issued.json"
    successful = meta["execution_status"] == "succeeded"
    report = load(report_path) if report_path.is_file() else None
    require(not successful or report is not None, "Successful Gaussian run lacks reconstruction evidence")
    ident = cfg["science"]
    inputs = load(root / "inputs.json")["inputs"]
    signed = next((r for r in inputs if r["role"] == "recipe"), None)
    require(
        signed and len(signed["files"]) == 1 and signed["storage"] in ("embedded", "shared"),
        "Missing pinned optimizer recipe",
    )
    src = signed["files"][0]
    path = (
        root / src["path"]
        if signed["storage"] == "embedded"
        else asset_path(asset_store or root.parent.parent, src["sha256"])
    )
    require(
        path.is_file() and path.stat().st_size == src["bytes"] and src["sha256"] == ident["recipe_sha256"],
        "Optimizer recipe asset differs from signed input",
    )
    require(sha(path) == ident["recipe_sha256"], "Optimizer recipe bytes differ")
    topology = load(path)["topology"]
    rid = meta["run_id"]
    ids = ident["generator_source_ids"]
    prior = cfg.get("resume_origin")
    first = prior["step"] if prior else 0
    raw, truncated = read_steps(trace_path, not successful) if trace_path.is_file() else ([], 0)
    if report:
        require(
            report["schema"] == "el-recovery-reconstruction-1"
            and report["identity"] == ident
            and report["run_id"] == rid
            and report["first_completed_step"] == first
            and report["trace_sha256"] == sha(trace_path)
            and report["updates_in_process"] == len(raw)
            and report["last_completed_step"] == first + len(raw),
            "Optimizer summary and trace disagree",
        )
    owner = load(point_path) if point_path.is_file() else None
    points = owner["checkpoints"] if owner else {}
    require(
        owner is None or (owner["run_id"] == rid and owner["identity"] == ident),
        "Wrong issued checkpoint lineage",
    )
    require(
        report is None or report["checkpoints"] == points,
        "Issued checkpoint list differs from reconstruction",
    )
    obj = ident["object_id"]
    engineering = (
        obj == "generated-fixture"
        or "engineering" in meta["experiment_id"]
        or "preflight" in meta["trial_id"]
    )
    case = {
        "run_id": rid,
        "population": "engineering"
        if engineering
        else "authored_control"
        if obj == "scene_057_offset_bell_cage"
        else "dtu",
        "recipe_id": ident["recipe_sha256"][:12],
        "recipe_sha256": ident["recipe_sha256"],
        "object_id": obj,
        "condition": ident["condition"],
    }
    steps, events, checkpoints = [], [], []
    initial_n = points.get("0", {}).get("gaussians")
    total = Counter()
    if initial_n is not None:
        require(first == 0, "Resumed execution issued a new step zero")
    n = initial_n
    operations = 0
    for i, row in enumerate(raw, start=first + 1):
        expected_counts = {s: (i + len(ids) - 1 - k) // len(ids) for k, s in enumerate(ids)}
        require(
            i == row["step"]
            and row["source"] == ids[(i - 1) % len(ids)]
            and row["source_counts"] == expected_counts
            and (n is None or row["n_before"] == n),
            "Source sampling, step number or Gaussian continuity differs",
        )
        require(
            math.isfinite(row["loss"])
            and math.isfinite(row["display_l1"])
            and row["pixels"] > 0
            and 0 <= row["visible"] <= row["n_before"],
            "Invalid objective or support",
        )
        phase = (
            "warmup" if i < topology["start"] else "refinement" if i < topology["stop"] else "fixed_topology"
        )
        require(
            row["phase"] == phase
            and max(row["n_before"], row["n_after"]) <= topology["cap"]
            and (phase == "refinement" or not row["topology"]),
            "Step phase, registered cap or fixed topology differs",
        )
        mutations, checked = parse_events(row)
        for ev in mutations:
            events.append({**case, "step": i, **ev})
            if "selected" in ev:
                total[ev["operation"]] += ev["selected"]
        operations += checked
        n = row["n_after"]
        steps.append(
            {
                **case,
                "step": i,
                "phase": row["phase"],
                "source": row["source"],
                **{s + "_updates": expected_counts.get(s) for s in ("s00", "s01", "s02", "s03")},
                "n_before": row["n_before"],
                "gaussians": n,
                "pixels": row["pixels"],
                "visible": row["visible"],
                "loss": row["loss"],
                "display_l1": row["display_l1"],
                "ssim": row["ssim"],
                "max_radius_px": row["max_radius_px"],
                "max_abs_visible_means2d_px": row["max_abs_visible_means2d_px"],
                "topology_events": len(mutations),
                "adam_mutations_checked": checked,
            }
        )
        if str(i) in points:
            require(
                n == points[str(i)]["gaussians"] and row["phase"] == points[str(i)]["phase"],
                "Issued population or phase differs from optimizer trace",
            )
    if report:
        require(
            report["sampling_counts"]
            == {s: (first + len(raw) + len(ids) - 1 - k) // len(ids) for k, s in enumerate(ids)},
            "Final source counts differ",
        )
        if not prior:
            require(dict(total) == report["topology_totals"], "Reconstruction event totals disagree")
    for key, point in sorted(points.items(), key=lambda pair: int(pair[0])):
        rel = "payload/" + point["path"]
        require(
            rel == f"payload/checkpoints/step-{int(key):04d}.pt"
            and point["step"] == int(key)
            and point["producer_run_id"] == rid
            and point["gaussians"] > 0
            and rel in membership
            and membership[rel]["sha256"] == point["checkpoint_sha256"]
            and membership[rel]["bytes"] == point["bytes"],
            "Issued Gaussian checkpoint differs from sealed bytes",
        )
        checkpoints.append(
            {
                **case,
                "step": point["step"],
                "phase": point["phase"],
                "gaussians": point["gaussians"],
                "checkpoint_sha256": point["checkpoint_sha256"],
                "bytes": point["bytes"],
            }
        )
    summary = {
        "population": case["population"],
        "recipe_sha256": case["recipe_sha256"],
        "updates_this_process": len(raw) if report else None,
        "complete_logged_updates": len(raw),
        "first_prior_update": first,
        "last_complete_logged_update": first + len(raw),
        "start_N": initial_n if initial_n is not None else (raw[0]["n_before"] if raw else None),
        "end_N": n,
        "max_observed_N": max((max(r["n_before"], r["gaussians"]) for r in steps), default=initial_n),
        "topology_mutations_with_adam_witness": operations,
        **{k + "_selected_rows": total[k] for k in ("clone", "split", "prune", "reset")},
        "checkpoint_count": len(points),
        "trace_partial_tail_bytes": truncated,
        "schedule_horizon": None if not report else report["schedule_horizon"],
        "planned_stop": cfg.get("planned_stop_step"),
        "resume_run_id": None if not prior else prior["run_id"],
        "resume_checkpoint_sha256": None if not prior else prior["checkpoint_sha256"],
        "loop_wall_seconds": None if not report else report["loop_wall_seconds"],
    }
    return {"steps": steps, "events": events, "checkpoints": checkpoints, "summary": summary}


def read_table(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as reader:
        return list(csv.DictReader(reader))


def plot(steps, resources, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    observations = defaultdict(list)
    for row in read_table(steps):
        if row.get("population") == "engineering":
            continue
        groups[(row["recipe_id"], row["object_id"], row["condition"])].append(row)
    for row in read_table(resources):
        observations[row["run_id"]].append(row)
    experiments = sorted({(r, o) for r, o, _ in groups})
    if not experiments:
        raise ValueError("no recorded optimizations in step table")
    fig, grid = plt.subplots(len(experiments), 3, squeeze=False, figsize=(12, 2.5 * len(experiments) + 0.9))
    for axes, (recipe, obj) in zip(grid, experiments, strict=True):
        for condition, color in COLORS.items():
            rows = sorted(groups.get((recipe, obj, condition), []), key=lambda row: int(row["step"]))
            if not rows:
                continue
            for j, rid in enumerate(sorted({r["run_id"] for r in rows})):
                segment = [r for r in rows if r["run_id"] == rid]
                if len({r["step"] for r in segment}) != len(segment):
                    raise ValueError("The optimizer trace repeats an update in a single process")
                xs = [int(segment[0]["step"]) - 1, *(int(r["step"]) for r in segment)]
                ys = [int(segment[0]["n_before"]), *(int(r["gaussians"]) for r in segment)]
                axes[0].step(
                    xs, ys, where="post", color=color, label=condition if j == 0 else None, linewidth=1.15
                )
                obs = sorted(observations[rid], key=lambda r: float(r["elapsed_seconds"]))
                for ax, field in ((axes[1], "tree_rss_mib"), (axes[2], "device_used_mib")):
                    samples = [(float(r["elapsed_seconds"]), float(r[field])) for r in obs if r[field]]
                    if samples:
                        xs, ys = zip(*samples)
                        ax.plot(xs, ys, ".-", color=color, linewidth=0.8, markersize=2.2, alpha=0.85)
        label = "057 authored control" if obj == "scene_057_offset_bell_cage" else obj
        axes[0].set_ylabel(label + "\n" + recipe)
        for ax in axes:
            ax.grid(axis="y", color="#d6dadc", linewidth=0.55)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=8)
        for ax in axes[1:]:
            if not ax.lines:
                ax.text(0.5, 0.5, "No samples", transform=ax.transAxes, ha="center", color="#666666")
        axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    for ax, title in zip(
        grid[0], ("Gaussian count", "Process-tree RSS (MiB)", "Whole GPU in use (MiB)"), strict=True
    ):
        ax.set_title(title, loc="left", fontsize=10)
    for ax, title in zip(
        grid[-1], ("Optimizer updates", "Recorder elapsed (s)", "Recorder elapsed (s)"), strict=True
    ):
        ax.set_xlabel(title)
    fig.tight_layout()
    with plt.rc_context({"svg.hashsalt": "el-recovery-fit-v1", "svg.fonttype": "none"}):
        fig.savefig(out, format="svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("steps", "resources", "out"):
        p.add_argument("--" + flag, required=True, type=Path)
    a = p.parse_args()
    plot(a.steps, a.resources, a.out)
