"""Evaluate a predeclared D1 condition from original photographic PNGs, FP32 predictions and frozen risk maps."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from evidence_lens.formal_inputs import compact_pin, require, verified_file
from evidence_lens.multisource_io import Case
from evidence_lens.scoring_io import new_output, write_json, file_hash
from dtu_score import METHODS, SCOPES, VIEWS, RELEASE_SHA
from dtu_eval_metrics import labels, observation, aggregate_views, RATES
from dtu_eval_board import board
from vision_sources import find_run, member, signed

IDENTITY = (
    "object_id",
    "camera_plan_sha256",
    "camera_sha256",
    "view_set_sha256",
    "crop_sha256",
    "lighting_id",
    "source_view_ids",
    "target_view_ids",
)
DOMAINS = ("full_crop", "object")
SIZE = (256, 256)


def read_pinned(roots, spec, success=True):
    require(
        isinstance(spec, dict) and all(k in spec for k in ("run_id", "manifest_sha256", "member", "sha256")),
        "missing original run, member or digest",
    )
    pin = compact_pin(spec)
    if not success and not any((root / "runs" / pin["run_id"]).is_dir() for root in roots):
        from evidence_lens._verify import verify_bundle

        for root in roots:
            folder = root / "quarantine" / pin["run_id"]
            if not folder.is_dir():
                continue
            result = verify_bundle(folder)
            p = folder / "run.json"
            require(
                result["manifest_sha256"] == pin["manifest_sha256"]
                and result["run"]["execution_status"] in ("failed", "interrupted", "cancelled")
                and spec["member"] == "run.json"
                and file_hash(p) == spec["sha256"],
                "unverified original engineering failure",
            )
            return SimpleNamespace(
                pin=pin,
                root=folder,
                run=result["run"],
                files={r["path"]: r for r in result["manifest"]["files"]},
            ), p
    r = find_run(roots, pin)
    if not success:
        require(
            r.run["execution_status"] in ("failed", "interrupted", "cancelled")
            and spec["member"] == "run.json",
            "failure needs the original unsuccessful run receipt",
        )
    if success:
        require(r.run["execution_status"] == "succeeded", "unsuccessful input for " + spec["member"])
    p, _ = member(r, spec["member"], spec["sha256"])
    return r, p


def mask(root, spec):
    if spec["kind"] == "all_true":
        require(spec["width"] == spec["height"] == 256, "incorrect all-valid dimensions")
        return np.ones(SIZE, bool)
    require(spec["kind"] == "png", "unknown reference mask semantics")
    f = verified_file(root, spec["file"]["path"], spec["file"])
    with Image.open(f) as im:
        require(im.mode == "L" and im.size == SIZE, "expected original 256-square grayscale mask")
        a = np.asarray(im)
    require(np.all((a == 0) | (a == 255)), "expected binary reference mask")
    return a == 255


def photo(root, spec):
    p = verified_file(root, spec["path"], spec)
    with p.open("rb") as stream:
        b = stream.read(26)
    require(b[:8] == b"\x89PNG\r\n\x1a\n" and b[24:26] == b"\x08\x02", "expected original RGB8 PNG")
    with Image.open(p) as im:
        require(im.mode == "RGB" and im.size == SIZE, "incorrect reference resolution")
        return np.asarray(im, dtype=np.float32) / np.float32(255)


def score_map(root, ss, m, v, case, tau):
    row = ss["views"][v]["methods"][m]
    require(
        row["scope"] == SCOPES[m]
        and row["source_ids"] == (["s00"] if m == "B-L1-transfer" else list(case.generator))
        and row["target_prediction_sha256"] == case.meta["renders"][v]["sha256"],
        "risk belongs to different exposure or target",
    )
    file = verified_file(root, row["file"]["path"], row["file"])
    with np.load(file, allow_pickle=False) as z:
        require(set(z.files) == {"risk", "state", "valid"}, "unexpected score fields")
        risk, valid, state = z["risk"], z["valid"], z["state"]
    require(
        risk.dtype == np.float32
        and valid.dtype == bool
        and state.dtype == np.uint8
        and risk.shape == state.shape == valid.shape == SIZE
        and np.array_equal(valid, np.isfinite(risk))
        and np.all(np.isnan(risk[~valid]))
        and np.all((risk[valid] >= 0) & (risk[valid] <= 1))
        and np.all(state[~valid] == 255)
        and np.array_equal(state[valid], risk[valid].astype(np.float64) >= tau),
        "invalid archived finite support or threshold state",
    )
    return risk


def feature_diagnostics(case, v, frame):
    total = SIZE[0] * SIZE[1]
    if frame is None:
        return {"planned_pixels": total, "rendered": False}
    rgb = frame["rgb"]
    z = frame["z"]
    a = frame["alpha"]
    valid = np.isfinite(rgb).all(2) & (rgb >= 0).all(2) & (rgb <= 1).all(2)
    aa = np.isfinite(a) & (a >= 0) & (a <= 1)
    zz = np.isfinite(z) & (z > 0)
    _, feat = case.features(v, "all")
    return {
        "planned_pixels": total,
        "rendered": True,
        "rgb_valid": int(valid.sum()),
        "black_within_one_rgb8_code": int((valid & (rgb <= np.float32(1 / 255)).all(2)).sum()),
        "alpha_valid": int(aa.sum()),
        "alpha_zero": int((aa & (a == 0)).sum()),
        "alpha_half_or_more": int((aa & (a >= 0.5)).sum()),
        "positive_finite_expected_camera_Z": int(zz.sum()),
        "positive_finite_ED_at_alpha_half_or_more": int((zz & aa & (a >= 0.5)).sum()),
        "has_source_projection_in_frame": int((feat["in_frame_count"] > 0).sum()),
        "has_comparable_source": int((feat["comparable_count"] > 0).sum()),
        "has_depth_supported_source": int((feat["support_count"] > 0).sum()),
        "has_conflict": int((feat["conflict_count"] > 0).sum()),
        "geometry_is_diagnostic": "transparency-normalized expected camera Z, no independent first-surface ground truth",
    }


def assess(roots, req, out, rid):
    require(
        req["schema"] == "el-dtu-eval-request-1"
        and req["role"] in ("debug", "external")
        and req["arm"] in ("main-256", "diagnostic-1024")
        and req["condition"] in ("M2", "M4")
        and req["status"] in ("ready", "score_failed", "render_failed", "reference_unavailable"),
        "unregistered case",
    )
    require(
        req["object_id"] == {"debug": "set1", "external": "set6"}[req["role"]], "unregistered object role"
    )
    if req["role"] == "external":
        require(req["arm"] == "main-256", "diagnostic budget is restricted to the registered debug object")
    _, authority_file = read_pinned(roots, req["authority"])
    if req["role"] == "external":
        authority = json.loads(authority_file.read_bytes())
        require(
            authority.get("schema") == "el-dtu-reference-gate-1"
            and req["object_id"] in authority.get("object_ids", []),
            "external target references remain sealed",
        )
    if req["status"] != "ready":
        require(
            req.get("failure") and req.get("reason"), "failure requires an original disposition and reason"
        )
        read_pinned(roots, req["failure"], False)
    metadata = {}
    engineering = None
    refroot = None
    if req["status"] != "reference_unavailable":
        _, rp = read_pinned(roots, req["references"])
        refroot = rp.parent
        metadata = json.loads(rp.read_bytes())
        require(
            metadata["schema"] == "el-dtu-references-1"
            and metadata["object_id"] == req["object_id"]
            and set(metadata["views"]) == set(VIEWS)
            and set(metadata["target_view_ids"]) == set(VIEWS)
            and set(metadata["source_view_ids"]) == {"s00", "s01", "s02", "s03"}
            and len(set([*metadata["target_view_ids"].values(), *metadata["source_view_ids"].values()]))
            == 10,
            "incomplete or mislabeled photographic reference set",
        )
    if metadata and req["role"] == "external":
        require(
            metadata.get("opening_sha256") == req["authority"]["sha256"],
            "reference photo conversion has different reference gate",
        )
    case = None
    ss = None
    scorepath = None
    if req.get("case"):
        _, cp = read_pinned(roots, req["case"])
        case = Case(cp.parent, req["case"]["sha256"])
        _, pp = read_pinned(roots, req["case_provenance"])
        prov = json.loads(pp.read_bytes())
        require(
            prov["object_id"] == req["object_id"]
            and prov["condition"] == case.condition == req["condition"]
            and prov["case_sha256"] == case.sha256,
            "case provenance differs",
        )
        steps = 256 if req["arm"] == "main-256" else 1024
        require(
            prov.get("optimization_steps") == steps and prov.get("arm") == req["arm"],
            "wrong optimization arm",
        )
        require(
            prov["reconstruction_readback_sha256"] == req["readback"]["sha256"]
            and req["readback"]["run_id"] == req["case"]["run_id"],
            "incorrect independent reconstruction check",
        )
        _, rp_engineering = read_pinned(roots, req["readback"])
        engineering = json.loads(rp_engineering.read_bytes())
        require(
            engineering["schema"] == "el-dtu-reconstruction-readback-1" and engineering["steps"] == steps,
            "reconstruction evidence differs from declared arm",
        )
        if req["role"] == "external":
            require(prov.get("opening_sha256") == authority["opening_sha256"], "different source opening")
        if metadata:
            require(
                all(prov[k] == metadata[k] for k in IDENTITY),
                "reference and deployed case have different camera, crop or photographic identities",
            )
    else:
        require(
            req["status"] in ("render_failed", "reference_unavailable"), "missing declared deployment case"
        )
    _, lp = read_pinned(roots, req["release"])
    release = json.loads(lp.read_bytes())
    require(file_hash(lp) == RELEASE_SHA, "changed published release")
    tau = {m: release["models"][m]["threshold"] for m in METHODS}
    if req["role"] == "external" and req["status"] != "reference_unavailable":
        terminal = authority["terminal_cells"][req["condition"]]
        expected = req["scores"] if req["status"] == "ready" else req["failure"]
        require(
            terminal["status"] == req["status"]
            and all(terminal[k] == expected[k] for k in ("run_id", "manifest_sha256", "sha256")),
            "terminal condition differs from frozen complete score matrix",
        )
    if req["status"] == "ready":
        _, scorepath = read_pinned(roots, req["scores"])
        ss = json.loads(scorepath.read_bytes())
        require(
            ss["schema"] == "el-dtu-frozen-scores-1"
            and ss["release_sha256"] == RELEASE_SHA
            and ss["models"] == release["models"]
            and ss["method_scopes"] == SCOPES
            and ss["case_sha256"] == case.sha256
            and ss["condition"] == case.condition
            and set(ss["views"]) == set(VIEWS),
            "different frozen algorithms, weights, target roster or case",
        )
    source_images = {k: t[0] for k, t in case.sources("all").items()} if case else {}
    rows = []
    groups = {d: {m: [] for m in METHODS} for d in DOMAINS}
    ledger = []
    with new_output(out) as dest:
        for v in VIEWS:
            if not metadata:
                rows.append(
                    {
                        "view": v,
                        "disposition": "reference_unavailable",
                        "domains": {k: None for k in DOMAINS},
                        "domain_reasons": {k: req["reason"] for k in DOMAINS},
                    }
                )
                continue
            reference = metadata["views"][v]
            require(reference["official_view_id"] == metadata["target_view_ids"][v], "target ID differs")
            ref = photo(refroot, reference["rgb"])
            valid = mask(refroot, reference["valid_mask"])
            obj = reference.get("object_mask")
            domain = {"full_crop": valid, "object": valid & mask(refroot, obj) if obj is not None else None}
            require(
                obj is not None or bool(reference.get("object_mask_reason")),
                "missing stated object mask disposition",
            )
            if obj is not None:
                require(
                    bool(obj.get("evidence") or metadata.get("object_mask_evidence")),
                    "auxiliary object region needs independent official evidence",
                )
            frame = case.frame(v) if case else None
            if ss:
                require(
                    ss["views"][v]["status"] == ("ready" if frame is not None else "render_failed"),
                    "different render disposition",
                )
            prediction = frame["rgb"] if frame is not None else None
            risks = {
                m: (
                    score_map(scorepath.parent, ss, m, v, case, tau[m])
                    if ss and frame is not None
                    else np.full(SIZE, np.nan, np.float32)
                )
                for m in METHODS
            }
            diagnostics = feature_diagnostics(case, v, frame)
            row = {
                "view": v,
                "official_view_id": reference["official_view_id"],
                "disposition": (
                    "render_failed" if frame is None else "score_failed" if ss is None else "ready"
                ),
                "reference_sha256": reference["rgb"]["sha256"],
                "prediction_sha256": case.meta["renders"][v]["sha256"] if frame is not None else None,
                "domains": {},
                "domain_reasons": {},
                "engineering_observations": diagnostics,
            }
            saved = {}
            for d, region in domain.items():
                if region is None:
                    row["domains"][d] = None
                    row["domain_reasons"][d] = reference["object_mask_reason"]
                    continue
                arrays = labels(ref, prediction, region)
                vm, ev, yt, err, aux = arrays
                metrics, gr = observation(*arrays, risks, tau)
                row["domains"][d] = metrics
                for key, g in gr.items():
                    groups[d][key].append(g)
                for k, a in zip(("V", "E", "Y", "rgb_mae", "min3_mae"), arrays):
                    saved[d + "_" + k] = a
                if d == "full_crop":
                    fig = dest / "boards" / (v + ".png")
                    board(
                        fig,
                        req["object_id"],
                        req["condition"],
                        v,
                        source_images,
                        prediction,
                        ref,
                        arrays,
                        risks,
                        tau,
                        None if frame is None else frame["alpha"],
                    )
                    row["board"] = {
                        "path": fig.relative_to(dest).as_posix(),
                        "sha256": file_hash(fig),
                        "bytes": fig.stat().st_size,
                    }
            arr = dest / "labels" / (v + ".npz")
            arr.parent.mkdir(exist_ok=True)
            np.savez_compressed(arr, **saved)
            row["labels"] = {
                "path": arr.relative_to(dest).as_posix(),
                "sha256": file_hash(arr),
                "bytes": arr.stat().st_size,
            }
            rows.append(row)
        aggregate = {}
        for d in DOMAINS:
            present = [r for r in rows if r["domains"][d] is not None]
            aggregate[d] = {
                "planned_views": len(VIEWS),
                "reference_present_views": len(present),
                "methods": {},
            }
            if present:
                for m in METHODS:
                    aggregate[d]["methods"][m] = aggregate_views(
                        [r["domains"][d][m] for r in present], groups[d][m], tau[m], planned_views=len(VIEWS)
                    )
        result = {
            "schema": "el-dtu-condition-evaluation-1",
            "run_id": rid,
            "role": req["role"],
            "object_id": req["object_id"],
            "condition": req["condition"],
            "arm": req["arm"],
            "status": req["status"],
            "method_scopes": SCOPES,
            "thresholds": tau,
            "release_sha256": RELEASE_SHA,
            "failure_reason": req.get("reason"),
            "input_pins": {
                k: req[k]
                for k in (
                    "authority",
                    "references",
                    "case",
                    "case_provenance",
                    "readback",
                    "scores",
                    "release",
                    "failure",
                )
                if req.get(k)
            },
            "identity": {k: metadata[k] for k in IDENTITY} if metadata else None,
            "reconstruction_readback": engineering,
            "geometry_supervision": None,
            "geometry_reason": "No validated dense surface-depth supervision is registered.",
            "rows": rows,
            "aggregate": aggregate,
            "rates": RATES,
            "board_access": "private DTU research inspection",
        }
        write_json(dest / "evaluation.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("store", "shelf", "request", "output"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--request-sha256", required=True)
    a = p.parse_args()
    d = assess((a.store, a.shelf), signed(a.request, a.request_sha256), a.output, os.environ.get("EL_RUN_ID"))
    print(
        json.dumps(
            {
                "status": d["status"],
                "object": d["object_id"],
                "condition": d["condition"],
                "planned": len(d["rows"]),
            }
        )
    )
