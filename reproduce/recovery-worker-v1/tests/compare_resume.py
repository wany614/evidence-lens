"""Compare independent continuous and segmented runs using issued bytes; no external images."""
import argparse
import json
import os
from pathlib import Path

import torch

from evidence_lens._bundle import load, save
from evidence_lens._verify import require

p = argparse.ArgumentParser()
for k in ("full-mid", "seg-mid", "full-end", "seg-end", "full-log", "head-log", "tail-log", "render-all", "render-one", "fixture", "out"):
    p.add_argument("--" + k, type=Path, required=True)
a = p.parse_args()
require(os.environ.get("EL_RUN_KIND") == "archive_verify", "Record numerical comparison")
limits = load(a.fixture)
atol = limits["absolute_parameter_and_moment_tolerance"]
rtol = limits["relative_tolerance"]
details = {"tolerances": {"absolute": atol, "relative": rtol, "loss": limits["absolute_display_loss_tolerance"]}}
def compare(x, y, path, stats):
    if isinstance(x, torch.Tensor):
        require(x.shape == y.shape and x.dtype == y.dtype, "Tensor layout differs: " + path)
        if x.is_floating_point():
            diff = float((x - y).abs().max()) if x.numel() else 0.
            stats["max_abs"] = max(stats["max_abs"], diff)
            if not torch.allclose(x, y, rtol=rtol, atol=atol):
                stats["mismatches"].append([path, diff])
        elif not torch.equal(x, y):
            stats["mismatches"].append([path, "integer tensor differs"])
    elif isinstance(x, dict):
        require(x.keys() == y.keys(), "Mapping keys differ " + path)
        for key in x:
            compare(x[key], y[key], path + "/" + str(key), stats)
    elif isinstance(x, (list, tuple)):
        require(len(x) == len(y), "Sequence length differs " + path)
        for i, (left, right) in enumerate(zip(x, y, strict=True)):
            compare(left, right, path + "/" + str(i), stats)
    elif x != y:
        stats["mismatches"].append([path, "metadata differs"])
for tag, left, right in (("interruption_boundary", a.full_mid, a.seg_mid), ("endpoint", a.full_end, a.seg_end)):
    pair = [torch.load(x, weights_only=True, map_location="cpu") for x in (left, right)]
    for blob in pair:
        blob.pop("producer_run_id")
    stats = {"max_abs": 0., "mismatches": []}
    compare(*pair, "", stats)
    details[tag] = stats
def trace(path):
    return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines()]
full = trace(a.full_log)
seg = trace(a.head_log) + trace(a.tail_log)
require(len(full) == len(seg) == limits["end_step"], "Trace steps missing")
fields = ("step", "phase", "source", "source_counts", "n_before", "n_after", "topology", "lr")
details["same_trace_schedule_and_topology"] = all(all(x[k] == y[k] for k in fields) for x, y in zip(full, seg))
details["max_source_loss_difference"] = max(abs(x["loss"] - y["loss"]) for x, y in zip(full, seg))
all_views = load(a.render_all)["checkpoints"]
one = load(a.render_one)["checkpoints"]
k = str(limits["resume_at"])
details["all_checkpoint_steps"] = sorted(map(int, all_views))
details["same_selected_checkpoint_views"] = set(one) == {k} and one[k] == all_views[k]
details["view_ids"] = sorted(all_views[k]["views"])
details["passed"] = (not details["endpoint"]["mismatches"] and not details["interruption_boundary"]["mismatches"] and
                     details["same_trace_schedule_and_topology"] and details["same_selected_checkpoint_views"] and
                     details["max_source_loss_difference"] <= limits["absolute_display_loss_tolerance"] and
                     details["all_checkpoint_steps"] == [0, 4, 8, 12] and
                     set(details["view_ids"]) == {"s00", "s03", *(f"v{i:02d}" for i in range(1, 7))})
save(a.out / "resume-parity.json", details)
require(details["passed"], "Resume or selected render parity failed; see original resume-parity.json")
