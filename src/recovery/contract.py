"""Role-specific input contracts. The numerical worker never receives a cell directory."""
from __future__ import annotations

import math
import re
from pathlib import Path

from evidence_lens._bundle import load, sha
from evidence_lens._verify import require

SOURCES = {"M2": ("s00", "s03"), "M4": ("s00", "s01", "s02", "s03")}
TARGETS = tuple(f"v{i:02d}" for i in range(1, 7))
PARAMS = {"means": "means", "log_scales": "scales", "quats": "quats",
          "opacity_logits": "opacities", "color_logits": "colors"}
WIDTHS = {"means": 3, "log_scales": 3, "quats": 4, "opacity_logits": None, "color_logits": 3}
LR_NAMES = set(PARAMS)
HASH = re.compile(r"[a-f0-9]{64}")
RUN = re.compile(r"\d{8}T\d{12}Z-[a-f0-9]{16}")
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
HORIZON = 5000
MAX_N = 20000
WALL = 1800


def integer(value, low, high, label):
    require(type(value) is int and low <= value <= high, f"Invalid {label}")
    return value


def number(value, low, high, label):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high, f"Invalid {label}")
    return value


def validate_recipe(r):
    require(set(r) == {"schema", "seed", "stop_step", "checkpoints", "wall_seconds", "learning_rates",
                       "loss", "topology", "eps2d", "adam"} and r["schema"] == "el-recovery-recipe-1", "Recipe fields")
    end = integer(r["stop_step"], HORIZON, HORIZON, "fixed recipe horizon")
    integer(r["seed"], 0, 2**32 - 1, "seed")
    integer(r["wall_seconds"], 1, WALL, "aggregate wall limit")
    points = r["checkpoints"]
    require(isinstance(points, list) and all(type(i) is int and 0 <= i <= end for i in points)
            and points == sorted(set(points)) and {0, end, *(x for x in (500, 1500, 3000, 5000) if x <= end)} <= set(points),
            "Checkpoint schedule must include zero, the endpoint and registered milestones up to it")
    rates = r["learning_rates"]
    require(set(rates) == LR_NAMES, "Learning-rate groups")
    for name, pair in rates.items():
        require(isinstance(pair, list) and len(pair) == 2, "Each rate needs endpoints at updates 1 and 5000")
        for value in pair:
            number(value, 1e-15, 10, name)
    adam = r["adam"]
    require(set(adam) == {"betas", "eps"} and len(adam["betas"]) == 2, "Adam fields")
    for beta in adam["betas"]:
        number(beta, 0, 1 - 1e-12, "Adam beta")
    number(adam["eps"], 1e-15, 1e-2, "Adam epsilon")
    loss = r["loss"]
    require(set(loss) == {"display_l1", "one_minus_ssim"}, "Unsupported source loss")
    for v in loss.values():
        number(v, 0, 1, "source loss weight")
    require(abs(sum(loss.values()) - 1) < 1e-8, "Loss weights must sum to one")
    number(r["eps2d"], 1e-10, 10, "screen-space variance")
    t = r["topology"]
    require(set(t) == {"cap", "start", "stop", "every", "reset_steps", "reset_opacity", "prune_opacity",
                       "grow_grad2d", "clone_scale_fraction", "prune_scale_fraction", "prune_scale_from"}, "Topology fields")
    integer(t["cap"], 1, MAX_N, "point cap")
    integer(t["start"], 1, HORIZON - 1, "first refinement")
    integer(t["stop"], t["start"] + 1, HORIZON, "first frozen topology update")
    integer(t["every"], 1, HORIZON, "refinement period")
    integer(t["prune_scale_from"], 1, HORIZON, "first large-scale prune")
    require(t["reset_steps"] == sorted(set(t["reset_steps"])) and all(type(k) is int and t["start"] <= k < t["stop"]
            for k in t["reset_steps"]), "Reset outside topology window")
    number(t["prune_opacity"], 0, 0.99, "opacity floor")
    number(t["reset_opacity"], max(t["prune_opacity"], 1e-8), 1 - 1e-8, "reset opacity")
    for name in ("grow_grad2d", "clone_scale_fraction", "prune_scale_fraction"):
        number(t[name], 0, 1e6, name)
    return r


def phase(step, recipe):
    if step == 0:
        return "initial"
    t = recipe["topology"]
    return "warmup" if step < t["start"] else "refinement" if step < t["stop"] else "fixed_topology"


def field_names(condition):
    ids = SOURCES[condition]
    return {"source_cameras.json", "render_cameras.json", "init.npz", "init.json",
            *("sources/" + s + ".png" for s in ids)}


def pin(ref):
    require(isinstance(ref, dict) and set(ref) == {"run_id", "manifest_sha256"} and RUN.fullmatch(ref["run_id"])
            and HASH.fullmatch(ref["manifest_sha256"]), "Malformed run pin")
    return ref


def cell_manifest(cell, digest):
    path = cell / "job.json"
    require(path.is_file() and not path.is_symlink() and sha(path) == digest, "Cell manifest digest differs")
    j = load(path)
    require(j["schema"] == "el-recovery-cell-1" and j["condition"] in SOURCES and
            all(LABEL.fullmatch(j[k]) for k in ("object_id", "trial_id", "experiment_id"))
            and HASH.fullmatch(j["recipe_sha256"]) and set(j["files"]) == field_names(j["condition"]), "Cell role roster")
    for entry in j["files"].values():
        require(set(entry) == {"sha256", "bytes"} and HASH.fullmatch(entry["sha256"])
                and type(entry["bytes"]) is int and entry["bytes"] > 0, "Invalid input receipt")
    parents = [pin(p)["run_id"] for p in j["parents"]]
    require(parents and len(parents) == len(set(parents)), "Registration parent pins")
    require(set(j.get("input_parents", {})) <= set(j["files"]) | {"recipe"} and
            set(j.get("input_parents", {}).values()) <= set(parents), "Unknown input parent")
    return j


def checked_member(cell: Path, job, name):
    entry = job["files"][name]
    path = cell.joinpath(*name.split("/"))
    require(path.resolve() == path and path.is_file() and path.stat().st_size == entry["bytes"]
            and sha(path) == entry["sha256"], "Changed input " + name)
    return path
