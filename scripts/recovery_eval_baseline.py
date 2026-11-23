"""Read only the original references, allowed sources and old renderings; quantify two zero-fit RGB comparators."""

import argparse
import json
import os
from pathlib import Path
import numpy as np
from dtu_score import VIEWS
from evidence_lens.formal_inputs import require
from evidence_lens.scoring_io import new_output, write_json
from recovery_eval_io import Originals, SHAPE, SYNTH, authority, permit_frozen
from recovery_eval_quality import closest_axis, source_mean, measurement, sum_views
from vision_sources import signed


def work(roots, req, out, rid):
    require(req["schema"] == "el-recovery-original-rgb-request-1", "wrong original baseline request")
    auth = authority(roots, req)
    cells = req["cells"]
    requested = {(c["object_id"], c["condition"]) for c in cells}
    require(
        (
            requested == {("set1", "M2"), ("set1", "M4"), (SYNTH, "M4")}
            or requested == {("set6", "M2"), ("set6", "M4")}
        )
        and len(requested) == len(cells),
        "different registered comparison roster",
    )
    if ("set6", "M2") in requested:
        permit_frozen(roots, req)
    rows = []
    for cell in cells:
        old = Originals(roots, cell)
        mean, channels = source_mean(old.sources)
        views = {}
        sources = {}
        c = cell["condition"]
        choices = {}
        for v in VIEWS:
            s, angles = closest_axis(old.case.cameras, old.case.generator, v)
            choices[v] = {"source": s, "source_angles_radians": angles}
            view = {
                "reference_sha256": old.hashes[v],
                "old_render_sha256": old.case.meta["renders"].get(v, {}).get("sha256"),
            }
            frame = old.case.frame(v)
            methods = {
                "old": frame["rgb"] if frame is not None else None,
                "mean_color": mean,
                "nearest_axis_copy": old.sources[s],
            }
            for domain, valid in [("full_crop", old.masks[v]), ("object", old.object_masks[v])]:
                view[domain] = (
                    {k: measurement(old.refs[v], pred, valid)[0] for k, pred in methods.items()}
                    if valid is not None
                    else None
                )
            view["object_reason"] = old.object_reason[v]
            views[v] = view
        for s in old.case.generator:
            fr = old.case.frame(s)
            sources[s] = measurement(
                old.sources[s], fr["rgb"] if fr is not None else None, np.ones(SHAPE, bool)
            )[0]
        rows.append(
            {
                "object_id": old.obj,
                "condition": c,
                "target": views,
                "target_summary": {
                    k: sum_views([v["full_crop"][k] for v in views.values()])
                    for k in ("old", "mean_color", "nearest_axis_copy")
                },
                "source": sources,
                "source_summary": sum_views(list(sources.values())),
                "allowed_source_hashes": {k: v["sha256"] for k, v in old.case.meta["source_images"].items()},
                "zero_fit_per_channel_mean": channels,
                "angular_copy_choice": choices,
                "input_pins": cell,
            }
        )
    d = {
        "schema": "el-recovery-original-rgb-1",
        "run_id": rid,
        "protocol_sha256": auth["protocol_sha256"],
        "registration": req["protocol"],
        "model_calls": 0,
        "new_optimizations": 0,
        "cells": rows,
    }
    with new_output(out) as dest:
        write_json(dest / "baseline.json", d)
    return d


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("store", "shelf", "request", "output"):
        p.add_argument("--" + n, required=True, type=Path)
    p.add_argument("--request-sha256", required=True)
    a = p.parse_args()
    d = work((a.store, a.shelf), signed(a.request, a.request_sha256), a.output, os.environ.get("EL_RUN_ID"))
    print(json.dumps({"cells": len(d["cells"])}))
