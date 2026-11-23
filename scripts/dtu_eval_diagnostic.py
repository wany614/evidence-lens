"""Compare original 256/1024-step endpoints on the registered development object, without risk inference."""

import argparse
import json
import os
from pathlib import Path
import numpy as np
from evidence_lens.formal_inputs import require, verified_file
from evidence_lens.multisource_io import Case
from evidence_lens.scoring_io import new_output, write_json
from dtu_eval import read_pinned, photo, mask, IDENTITY
from dtu_score import VIEWS
from dtu_eval_replay import independent_labels
from vision_sources import signed


def run(roots, req, out, rid):
    require(
        req["schema"] == "el-dtu-source-budget-request-1"
        and req["object_id"] == "set1"
        and req["role"] == "debug"
        and req["condition"] in ("M2", "M4"),
        "diagnostic authorization is restricted to set1",
    )

    def original(key):
        _, p = read_pinned(roots, req[key])
        return p, json.loads(p.read_bytes())

    _, decision = original("authority")
    require(
        decision["schema"] == "el-dtu-development-diagnostic-1"
        and decision["object_id"] == "set1"
        and decision["updates"] == 1024
        and decision["external_updates"] == 256,
        "no registered development budget decision",
    )
    _, main = original("evaluation")
    require(
        main["role"] == "debug"
        and main["object_id"] == "set1"
        and main["condition"] == req["condition"]
        and main["arm"] == "main-256",
        "wrong comparison endpoint",
    )
    refs_path, refs = original("references")
    require(req["references"] == main["input_pins"]["references"], "different photographic reference")
    cp, meta = original("case")
    _, prov = original("case_provenance")
    _, eng = original("readback")
    case = Case(cp.parent, req["case"]["sha256"])
    require(
        prov["arm"] == "diagnostic-1024"
        and prov["optimization_steps"] == eng["steps"] == 1024
        and prov["case_sha256"] == case.sha256
        and prov["reconstruction_readback_sha256"] == req["readback"]["sha256"]
        and prov["object_id"] == "set1"
        and prov["condition"] == case.condition == req["condition"]
        and all(prov[k] == refs[k] for k in IDENTITY),
        "different sources, references or side-arm settings",
    )
    _, mainp = read_pinned(roots, main["input_pins"]["case"])
    base = Case(mainp.parent, main["input_pins"]["case"]["sha256"])
    e0 = main["reconstruction_readback"]
    require(
        base.meta["source_images"] == case.meta["source_images"]
        and eng["initial_parameter_sha256"] == e0["initial_parameter_sha256"]
        and base.meta["source_cameras"] == meta["source_cameras"]
        and base.meta["render_cameras"] == meta["render_cameras"]
        and eng["initial_gaussians"] == eng["final_gaussians"] == 4096
        and e0["steps"] == 256,
        "diagnostic changed registered sources, cameras, initialization or topology",
    )
    results = []
    for v in VIEWS:
        row = refs["views"][v]
        ref = photo(refs_path.parent, row["rgb"])
        valid = mask(refs_path.parent, row["valid_mask"])
        rec = {"view": v, "official_view_id": row["official_view_id"]}
        for key, x in [("main_256", base), ("diagnostic_1024", case)]:
            frame = x.frame(v)
            vv, e, y, err, aux = independent_labels(ref, frame["rgb"] if frame is not None else None, valid)
            n = int(e.sum())
            Y = int((y == 1).sum())
            rec[key] = {
                "V": int(vv.sum()),
                "E": n,
                "Y": Y,
                "rgb_mae_sum": float(err[e].sum(dtype=np.float64)),
                "rgb_mae": float(err[e].mean(dtype=np.float64)) if n else None,
                "rgb_error_fraction": Y / n if n else None,
                "aux3_mae": float(aux[e].mean(dtype=np.float64)) if n else None,
            }
        orig = next(x for x in main["rows"] if x["view"] == v)["domains"]["full_crop"]["MS-L1"]["counts"]
        require(
            all(orig[k] == rec["main_256"][k] for k in ("V", "E", "Y", "rgb_mae_sum")),
            "256 endpoint differs from original evaluation",
        )
        results.append(rec)

    def total(key):
        vals = {k: sum(x[key][k] for x in results) for k in ("V", "E", "Y", "rgb_mae_sum")}
        n = vals["E"]
        return {
            **vals,
            "rgb_mae": vals["rgb_mae_sum"] / n if n else None,
            "rgb_error_fraction": vals["Y"] / n if n else None,
        }

    d = {
        "schema": "el-dtu-budget-diagnostic-1",
        "run_id": rid,
        "object_id": "set1",
        "condition": req["condition"],
        "risk_scores_computed": 0,
        "input_pins": {
            k: req[k]
            for k in ("evaluation", "case", "case_provenance", "readback", "references", "authority")
        },
        "views": results,
        "totals": {k: total(k) for k in ("main_256", "diagnostic_1024")},
        "source_256": e0["sources"],
        "source_1024": eng["sources"],
        "readback_1024": eng,
    }
    with new_output(out) as dest:
        write_json(dest / "diagnostic.json", d)
    return d


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("store", "shelf", "request", "output"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--request-sha256", required=True)
    a = p.parse_args()
    d = run((a.store, a.shelf), signed(a.request, a.request_sha256), a.output, os.environ.get("EL_RUN_ID"))
    print(json.dumps(d["totals"]))
