"""Compare released risk maps on old and frozen recovery reconstructions, with originals confined to evaluation."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
from dtu_eval import score_map
from dtu_eval_board import board
from dtu_eval_metrics import labels, observation, aggregate_views
from dtu_score import METHODS, SCOPES, RELEASE_SHA, VIEWS
from evidence_lens.formal_inputs import require
from evidence_lens.multisource import REASONS
from evidence_lens.multisource_io import Case
from evidence_lens.scoring_io import new_output, write_json, file_hash, alpha_valid, depth_valid, rgb_valid
from recovery_eval_io import Originals, authority, recipe, permit_frozen, original
from recovery_eval_coverage import matched
from vision_sources import signed


def reasons(case, v, frame, risks, eligible, y):
    _, features = case.features(v, "all")
    positive = y == 1
    bwhy = np.select(
        [
            np.full(eligible.shape, not case.pred["depth_verified"]),
            ~rgb_valid(frame["rgb"]),
            ~alpha_valid(frame["alpha"]),
            ~depth_valid(frame["z"]),
        ],
        [1, 2, 3, 4],
        default=0,
    ).astype(np.uint8)
    out = {}
    for m, risk in risks.items():
        code = bwhy if m == "B-L1-transfer" else features["unknown_reason"]
        u = eligible & ~np.isfinite(risk)
        categories = {}
        for i, name in enumerate(REASONS):
            q = u & (code == i)
            categories[name if i else "unexplained_nonfinite_on_feature_support"] = {
                "U": int(q.sum()),
                "UY": int((q & positive).sum()),
            }
        require(
            sum(z["U"] for z in categories.values()) == int(u.sum())
            and sum(z["UY"] for z in categories.values()) == int((u & positive).sum()),
            "unknown reason partition differs from raw risk",
        )
        require(
            not (eligible & np.isfinite(risk) & (code != 0)).any(),
            "finite score outside its released input policy",
        )
        out[m] = categories
    hist = {}
    for k in ("in_frame_count", "comparable_count", "support_count", "conflict_count"):
        z = features[k]
        hist[k] = [
            {
                "count": i,
                "E": int((eligible & (z == i)).sum()),
                "Y": int((eligible & positive & (z == i)).sum()),
            }
            for i in range(len(case.generator) + 1)
        ]
    opacity = frame["alpha"]
    z = frame["z"]
    alpha_ok = eligible & alpha_valid(opacity)
    z_ok = eligible & depth_valid(z)
    geometry = {
        "E": int(eligible.sum()),
        "alpha_valid": int(alpha_ok.sum()),
        "alpha_zero": int((alpha_ok & (opacity == 0)).sum()),
        "alpha_half_or_more": int((alpha_ok & (opacity >= 0.5)).sum()),
        "finite_positive_expected_camera_z": int(z_ok.sum()),
        "positive_z_and_alpha_half_or_more": int((z_ok & alpha_ok & (opacity >= 0.5)).sum()),
    }
    return out, hist, geometry


def maps(root, req, version, case, models):
    _, p, meta = original(root, req[version + "_scores"])
    require(
        meta["schema"] == "el-dtu-frozen-scores-1"
        and meta["method_scopes"] == SCOPES
        and meta["release_sha256"] == RELEASE_SHA
        and meta["models"] == models
        and meta["condition"] == case.condition
        and meta["case_sha256"] == case.sha256
        and set(meta["views"]) == set(VIEWS),
        "score source identity or fixed model/threshold differs",
    )
    return p.parent, meta


def coverage_totals(records):
    """A zero accepted count defines coverage when V is positive. Its conditional error rate is undefined."""
    coverage_views = [r for r in records if r["accepted"] is not None and r["V"] > 0]
    risk_views = [r for r in coverage_views if r["accepted"] > 0]
    accepted = sum(r["accepted"] for r in coverage_views)
    positives = sum(r["accepted_positive"] for r in coverage_views)
    reference = sum(r["V"] for r in coverage_views)
    return {
        "scheduled_views": len(records),
        "defined_views": len(risk_views),
        "risk_defined_views": len(risk_views),
        "coverage_defined_views": len(coverage_views),
        "empty_low_views": sum(r["reason"] == "empty_low" for r in records),
        "outside_support_views": sum(r["reason"] == "outside_support" for r in records),
        "accepted": accepted,
        "accepted_positive": positives,
        "V_of_coverage_defined_views": reference,
        "actual_coverage": accepted / reference if reference else None,
        "risk": positives / accepted if accepted else None,
    }


def work(roots, req, out, rid):
    require(
        req["schema"] == "el-recovery-risk-eval-request-1" and req["object_id"] in ("set1", "set6"),
        "risk matrix is limited to development and exposed regression",
    )
    reg = authority(roots, req)
    recipe(roots, req)
    freeze = permit_frozen(roots, req)
    orig = Originals(roots, req)
    _, npth, _ = original(roots, req["new_case"])
    new = Case(npth.parent, req["new_case"]["sha256"])
    _, _, pr = original(roots, req["new_provenance"])
    require(
        pr["case_sha256"] == new.sha256
        and pr["freeze"] == req["freeze"]
        and pr["object_id"] == req["object_id"]
        and pr["condition"] == orig.case.condition == new.condition
        and pr["step"] == freeze["stop_step"],
        "new scores unrelated to frozen checkpoint",
    )
    for key in ("source_images", "source_cameras", "render_cameras", "generator_source_ids", "target_ids"):
        require(
            orig.case.meta[key] == new.meta[key], "calibration, source photographs, or view roster differ"
        )
    _, rp, release = original(roots, req["release"])
    require(file_hash(rp) == RELEASE_SHA, "wrong frozen published model")
    models = release["models"]
    taus = {m: models[m]["threshold"] for m in METHODS}
    cases = {"old": orig.case, "new": new}
    sr = {n: maps(roots, req, n, c, models) for n, c in cases.items()}
    groups = defaultdict(list)
    obs = defaultdict(list)
    rows = []
    matched_rows = []
    matched_valid = defaultdict(list)
    with new_output(out) as dest:
        pixels = dest / "pixels"
        pixels.mkdir()
        for v in VIEWS:
            versions = {}
            source_sha = orig.hashes[v]
            r = {
                "view": v,
                "target_reference_sha256": source_sha,
                "domain": "full_crop",
                "object_mask_status": "available" if orig.object_masks[v] is not None else "N/A",
                "object_mask_reason": orig.object_reason[v],
                "versions": {},
            }
            for n, c in cases.items():
                frame = c.frame(v)
                p, book = sr[n]
                pred = frame["rgb"] if frame is not None else None
                arrays = labels(orig.refs[v], pred, orig.masks[v])
                valid, e, y, err, aux = arrays
                pixel_path = pixels / f"{n}-{v}.npz"
                np.savez_compressed(pixel_path, V=valid, E=e, Y=y, rgb_mae=err, min3_mae=aux)
                rr = {
                    m: score_map(p, book, m, v, c, taus[m])
                    if frame is not None
                    else np.full(valid.shape, np.nan, np.float32)
                    for m in METHODS
                }
                z, g = observation(*arrays, rr, taus)
                object_observation = None
                if orig.object_masks[v] is not None:
                    region = orig.masks[v] & orig.object_masks[v]
                    region_arrays = labels(orig.refs[v], pred, region)
                    object_observation, _ = observation(*region_arrays, rr, taus)
                if frame is not None:
                    cause, support, geometry = reasons(c, v, frame, rr, e, y)
                else:
                    cause, support, geometry = {}, {}, None
                r["versions"][n] = {
                    "target_prediction_sha256": c.meta["renders"].get(v, {}).get("sha256"),
                    "score_source_sha256": {
                        m: book["views"][v]["methods"][m]["file"]["sha256"]
                        for m in METHODS
                        if m in book["views"][v]["methods"]
                    },
                    "metrics": z,
                    "object_metrics": object_observation,
                    "pixels": {
                        "path": pixel_path.relative_to(dest).as_posix(),
                        "sha256": file_hash(pixel_path),
                        "bytes": pixel_path.stat().st_size,
                    },
                    "unknown_reasons": cause,
                    "support": support,
                    "geometry_on_E": geometry,
                }
                versions[n] = {"V": valid, "E": e, "Y": y, "scores": rr, "thresholds": taus}
                for m in METHODS:
                    groups[n, m].append(g[m])
                    obs[n, m].append(z[m])
                if n == "new":
                    fig = dest / "boards" / (v + ".png")
                    board(
                        fig,
                        req["object_id"] + " recovery",
                        req["condition"],
                        v,
                        orig.sources,
                        pred,
                        orig.refs[v],
                        arrays,
                        rr,
                        taus,
                        frame["alpha"] if frame is not None else None,
                    )
                    r["new_risk_board"] = {
                        "path": fig.relative_to(dest).as_posix(),
                        "sha256": file_hash(fig),
                        "bytes": fig.stat().st_size,
                    }
            require(
                np.array_equal(versions["new"]["V"], versions["old"]["V"]),
                "paired mask differs between the same photographic reference",
            )
            coverage = matched(req["object_id"], req["condition"], v, versions, versions["new"]["V"])
            matched_rows.extend({"view": v, **x} for x in coverage)
            rows.append(r)
        ag = {}
        for n in cases:
            ag[n] = {}
            for m in METHODS:
                q = aggregate_views(obs[n, m], groups[n, m], taus[m], len(VIEWS))
                q["matched_view_prevalence"] = {}
                for dom, num, den in [("common", "CY", "C"), ("own", "SY", "S")]:
                    validrows = [r for r in obs[n, m] if r[dom]["ap"] is not None]
                    q["matched_view_prevalence"][dom] = {
                        "value": float(np.mean([r["counts"][num] / r["counts"][den] for r in validrows]))
                        if validrows
                        else None,
                        "defined_views": len(validrows),
                        "scheduled_views": len(VIEWS),
                    }
                ag[n][m] = q
        for r in matched_rows:
            matched_valid[r["domain"], r["version"], r["method"], r["anchor"]].append(r)
        coverage_summary = []
        for (dom, vn, m, anchor), record in matched_valid.items():
            require(len(record) == len(VIEWS), "missing matched-coverage target view")
            coverage_summary.append(
                {
                    "domain": dom,
                    "version": vn,
                    "method": m,
                    "anchor": anchor,
                    **coverage_totals(record),
                }
            )
        d = {
            "schema": "el-recovery-frozen-risk-evaluation-1",
            "run_id": rid,
            "object_id": req["object_id"],
            "role": "development_selection" if req["object_id"] == "set1" else "already_exposed_regression",
            "condition": req["condition"],
            "step": freeze["stop_step"],
            "recipe_sha256": pr["recipe_sha256"],
            "checkpoint_sha256": pr["checkpoint_sha256"],
            "checkpoints": {name: c.meta["checkpoint_sha256"] for name, c in cases.items()},
            "parameter_runs": {name: c.meta["reconstruction_run_id"] for name, c in cases.items()},
            "protocol_sha256": reg["protocol_sha256"],
            "risk_protocol_sha256": reg["registered_files"]["protocol/recovery-risk-comparison-v1.json"],
            "models": models,
            "thresholds": taus,
            "method_scopes": SCOPES,
            "release_sha256": RELEASE_SHA,
            "rows": rows,
            "aggregate": ag,
            "matched_coverage": matched_rows,
            "matched_summary": coverage_summary,
            "input_pins": {k: v for k, v in req.items() if isinstance(v, dict)},
        }
        write_json(dest / "risk.json", d)
    return d


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("store", "shelf", "request", "output"):
        p.add_argument("--" + n, required=True, type=Path)
    p.add_argument("--request-sha256", required=True)
    a = p.parse_args()
    d = work((a.store, a.shelf), signed(a.request, a.request_sha256), a.output, os.environ.get("EL_RUN_ID"))
    print(json.dumps({"object": d["object_id"], "condition": d["condition"], "views": len(d["rows"])}))
