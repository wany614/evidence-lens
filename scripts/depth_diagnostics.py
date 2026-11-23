"""Read pinned scientific runs in place and reproduce the frozen camera-Z results and private cases."""

import argparse
import csv
from pathlib import Path
from statistics import mean

import numpy as np
from depth_diagnostic_outputs import REGIONS, csv_out, private_board, strata_plot
from PIL import Image

from evidence_lens._bundle import load, sha
from evidence_lens.formal_inputs import PinnedRun, require
from evidence_lens.l1 import risk_state
from evidence_lens.labels import label_view
from evidence_lens.scoring_io import (
    VIEWS,
    ReadAudit,
    load_cameras,
    load_source_png,
    write_json,
)


class Evidence:
    """Track exact source members without unpacking an archive or duplicating scientific assets."""

    def __init__(self, roots):
        self.roots = [p.resolve() for p in roots]
        self.runs = {}
        self.locations = {}
        self.members = {}

    def run(self, pin):
        rid = pin["run_id"]
        if rid in self.runs:
            require(self.runs[rid].pin["manifest_sha256"] == pin["manifest_sha256"], "conflicting run pin")
        else:
            locations = [r for r in self.roots if (r / "runs" / rid).is_dir()]
            require(locations, f"missing sealed run {rid}")
            result = PinnedRun(locations[0], pin)
            require(result.run["execution_status"] == "succeeded", f"unusable upstream {rid}")
            self.runs[rid] = result
            self.locations[rid] = "active" if locations[0] == self.roots[0] else "cold_shelf"
        return self.runs[rid]

    def file(self, run, rel):
        rid = run.pin["run_id"]
        if (rid, rel) not in self.members:
            path = run.open(rel)
            self.members[rid, rel] = {
                "run_id": rid,
                "manifest_sha256": run.pin["manifest_sha256"],
                "member": rel,
                "location": self.locations[rid],
                **{k: run.files[rel][k] for k in ("bytes", "sha256")},
            }
            return path
        return run.root / rel

    def read(self, run, rel):
        return load(self.file(run, rel))

    def member(self, run, rel):
        self.file(run, rel)
        return self.members[run.pin["run_id"], rel]


def arrays(path, keys):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in keys}


def native_reference(path, cam, tol):
    ref = arrays(path, ("rgb", "z", "position", "reference_surface", "object_id", "edge", "visibility"))
    shape = cam.shape
    require(
        ref["rgb"].shape == (*shape, 3)
        and ref["position"].shape == (*shape, 3)
        and all(ref[k].shape == shape for k in ("z", "reference_surface", "object_id", "edge", "visibility")),
        "reference grid differs from native camera",
    )
    require(
        ref["rgb"].dtype == np.float32
        and ref["z"].dtype == np.float32
        and ref["position"].dtype == np.float32,
        "source RGB/Z/Position dtype",
    )
    require(
        ref["object_id"].dtype.kind in "iu"
        and ref["visibility"].dtype == np.uint8
        and np.all(ref["visibility"] <= 3),
        "reference identifiers",
    )
    surface = np.isfinite(ref["z"]) & (ref["z"] > 0)
    require(
        np.array_equal(surface, ref["reference_surface"])
        and np.array_equal(surface, np.isfinite(ref["position"]).all(-1)),
        "reference surface disagreement",
    )
    require(
        np.all((ref["edge"] == 0) | (ref["edge"] == 1)) and not np.any((ref["visibility"] != 0) & ~surface),
        "reference geometry labels",
    )
    ref["edge"] = ref["edge"].astype(bool, copy=False)
    if surface.any():
        w2c = np.linalg.inv(cam.c2w)
        xyz = ref["position"][surface] @ w2c[:3, :3].T + w2c[:3, 3]
        require(
            np.allclose(
                xyz[:, 2], ref["z"][surface], atol=tol["position_z_atol_m"], rtol=tol["position_z_rtol"]
            ),
            "reference camera-Z is inconsistent with Position",
        )
        y, x = np.nonzero(surface)
        xy = xyz @ cam.K.T
        uv = xy[:, :2] / xy[:, 2, None]
        require(
            float(np.hypot(uv[:, 0] - x, uv[:, 1] - y).max()) <= tol["position_projection_max_px"],
            "reference Position proj exceeds original tolerance",
        )
    return ref


def masks(ref):
    ids, vis, edge = ref["object_id"], ref["visibility"], ref["edge"]
    return dict(
        zip(
            REGIONS,
            (
                np.ones(edge.shape, bool),
                ids >= 3,
                ids == 2,
                ids == 1,
                ids == 0,
                edge,
                ~edge,
                vis == 1,
                vis == 2,
                vis == 3,
                vis == 0,
            ),
            strict=True,
        )
    )


def value(arr, fn=np.mean):
    return float(fn(arr)) if arr.size else None


def fraction(a, b):
    return a / b if b else None


def regional_statistics(ref, pred, labels, region, mask, verified=True):
    v = mask & labels["reference_valid"]
    e = mask & labels["rgb_evaluable"]
    ground = mask & labels["reference_surface"]
    z = mask & labels["depth_evaluable"]
    opacity_ok = labels["alpha_valid"]
    low_opacity = labels["low_alpha"]
    positive = ~labels["invalid_depth"]
    rgb_error = labels["rgb_error"][e]
    signed_rgb = (pred["rgb"][e] - ref["rgb"][e]).mean(-1) if pred is not None else np.empty(0)
    dz = labels["signed_depth_error"][z]
    rel = labels["signed_relative_depth_error"][z]
    ratio = pred["z"][z].astype(np.float64) / ref["z"][z] if pred is not None else np.empty(0)
    count = lambda x: int(np.count_nonzero(x))
    high = count(z & (labels["depth_high_error"] == 1))
    fails = {
        "failed_no_prediction": count(ground) if pred is None else 0,
        "failed_unverified_semantics": count(ground) if pred is not None and not verified else 0,
        "failed_invalid_predicted_z": count(ground & ~positive) if pred is not None and verified else 0,
        "failed_invalid_alpha": count(ground & positive & ~opacity_ok)
        if pred is not None and verified
        else 0,
        "failed_low_alpha": count(ground & positive & low_opacity) if pred is not None and verified else 0,
    }
    require(
        count(ground) - count(z) == sum(fails.values()), "depth failure reasons do not cover frozen domain"
    )
    q = (
        np.quantile(ratio, (0.05, 0.25, 0.50, 0.75, 0.95), method="linear").tolist()
        if ratio.size
        else [None] * 5
    )
    zsum = float(np.abs(dz).sum(dtype=np.float64))
    return {
        "region": region,
        "pixels": count(mask),
        "V": count(v),
        "E": count(e),
        "Y": count(e & (labels["rgb_high_error"] == 1)),
        "invalid_reference_rgb": count(mask & ~v),
        "invalid_prediction_given_V": count(v & ~e),
        "rgb_error_sum": float(rgb_error.sum(dtype=np.float64)),
        "rgb_mae": fraction(float(rgb_error.sum(dtype=np.float64)), count(e)),
        "rgb_median_abs": value(rgb_error, np.median),
        "rgb_signed_mean": value(signed_rgb.astype(np.float64)),
        "rgb_signed_median": value(signed_rgb, np.median),
        "rgb_negative": count(signed_rgb < 0),
        "rgb_zero": count(signed_rgb == 0),
        "rgb_positive": count(signed_rgb > 0),
        "surface": count(ground),
        "no_surface": count(mask & ~ground),
        "Z": count(z),
        "ZY": high,
        "Z_and_E": count(z & e),
        "Z_outside_E": count(z & ~e),
        "failed_depth": count(ground) - count(z),
        **fails,
        "z_fraction_of_surface": fraction(count(z), count(ground)),
        "depth_high_fraction": fraction(high, count(z)),
        "z_error_sum_m": zsum,
        "z_mae_m": fraction(zsum, count(z)),
        "z_median_abs_m": value(np.abs(dz), np.median),
        "z_signed_mean_m": value(dz.astype(np.float64)),
        "z_signed_median_m": value(dz, np.median),
        "z_negative": count(dz < 0),
        "z_zero": count(dz == 0),
        "z_positive": count(dz > 0),
        "z_signed_relative_mean": value(rel.astype(np.float64)),
        "z_signed_relative_median": value(rel, np.median),
        "z_relative_mae": value(np.abs(rel).astype(np.float64)),
        **dict(zip(("ratio_q05", "ratio_q25", "ratio_q50", "ratio_q75", "ratio_q95"), q, strict=True)),
    }


def same_camera(a, b, tol):
    return (
        a.shape == b.shape
        and np.allclose(a.K, b.K, rtol=0, atol=tol)
        and np.allclose(a.c2w, b.c2w, rtol=0, atol=tol)
    )


def read_ply_header(file):
    header = []
    with file.open("rb") as stream:
        while sum(len(x) + 1 for x in header) < 65536:
            line = stream.readline(4096)
            if not line:
                break
            text = line.decode("ascii").strip()
            header.append(text)
            if text == "end_header":
                break
    require(header[:1] == ["ply"] and header[-1:] == ["end_header"], "invalid or oversized PLY header")
    return [line for line in header if line.startswith(("format ", "element ", "property "))]


def assert_labels(as_saved, lab):
    with np.load(as_saved, allow_pickle=False) as saved:
        for k in (
            "reference_surface",
            "reference_valid",
            "rgb_evaluable",
            "depth_evaluable",
            "depth_high_error",
            "rgb_high_error",
            "rgb_error",
            "depth_error",
            "signed_depth_error",
            "signed_relative_depth_error",
        ):
            require(np.array_equal(saved[k], lab[k], equal_nan=True), f"frozen archived label differs: {k}")


def _source_supplement(ev, entry, case, cam, tol):
    pin = {k: entry[k] for k in ("run_id", "manifest_sha256")}
    run = ev.run(pin)
    receipt = ev.read(run, entry["receipt_path"])
    rel = entry["reference_path"]
    path = ev.file(run, rel)
    camera_file = ev.file(run, entry["cameras_path"])
    cams = load_cameras(camera_file, ReadAudit())
    require(same_camera(cams["source"], cam, tol), "R1 supplementary native camera mismatch")
    parents = set(run.run["parent_run_ids"])
    need = {case["selected"][k]["run_id"] for k in ("reference", "preparation")}
    require(
        need <= parents and receipt.get("validation_passed", receipt.get("passed")) is True,
        "R1 truth validation or lineage mismatch",
    )
    declared = entry["sha256"]
    require(
        receipt["family"] == case["family"]
        and receipt["parents"]
        == entry["parents"]
        == {k: case["selected"][k] for k in ("reference", "preparation")}
        and receipt["ground_truth_grid_wh"] == [cam.width, cam.height],
        "R1 native reference identity",
    )
    for path_key in ("receipt_path", "reference_path", "cameras_path"):
        member = entry[path_key]
        require(declared[Path(member).name] == run.files[member]["sha256"], "R1 source supplement file hash")
    require(
        receipt["outputs"]["source.npz"] == declared["source.npz"]
        and receipt["outputs"]["cameras.json"] == declared["cameras.json"],
        "R1 validation/file mismatch",
    )
    return path, ev.member(run, rel)


def process_case(ev, case, ctx):
    f, c = case["family"], case["condition"]
    ids = case["selected"]
    key = f + "-" + c
    ref, render, gen, evaluation = [
        ev.run(ids[k]) for k in ("reference", "render", "generation", "evaluation")
    ]
    cr = ev.run(ids["preparation"]) if c == "R1" else ref
    camera_rel, input_rel = "payload/cameras.json", "payload/input.png"
    cam_path = ev.file(cr, camera_rel)
    cams = load_cameras(cam_path, ReadAudit())
    orig = load_cameras(ev.file(ref, camera_rel), ReadAudit())
    png = ev.file(cr, input_rel)
    load_source_png(png, cams["source"], ReadAudit())
    source_cam = cams["source"]
    abs_tol = ctx["tolerances"]["camera_matrix_absolute"]
    rmeta = ev.read(ref, "payload/reference.json")
    valid = ev.read(ref, "payload/validation/results.json")
    predmeta = ev.read(render, "payload/prediction.json")
    depth_receipt = ev.member(render, "payload/depth-probe.json")
    geno = ev.read(gen, "payload/generation.json")
    evalmeta = ev.read(evaluation, "payload/evaluation.json")
    original_sha = ev.member(ref, camera_rel)["sha256"]
    require(
        rmeta.get("validation_passed")
        and valid.get("passed")
        and rmeta["validation_sha256"] == ref.files["payload/validation/results.json"]["sha256"]
        and rmeta["camera_sha256"] == original_sha,
        "reference receipt fails",
    )
    require(
        rmeta["depth_semantics"] == "positive_optical_axis_camera_z_m"
        and rmeta["rgb_encoding"] == "display_srgb_0_1",
        "reference units/semantics mismatch",
    )
    for data in (geno, predmeta):
        require(
            data["input_sha256"] == cr.files[input_rel]["sha256"]
            and data["camera_sha256"] == cr.files[camera_rel]["sha256"],
            "original input/camera provenance mismatch",
        )
    require(
        predmeta["producer_run_id"] == ids["render"]["run_id"]
        and geno["producer_run_id"] == ids["generation"]["run_id"]
        and predmeta["generation_run_id"] == ids["generation"]["run_id"],
        "upstream producer identity mismatch",
    )
    require(
        predmeta["depth_semantics"] == "expected_camera_z_m"
        and predmeta["alpha_semantics"] == "accumulated_opacity"
        and predmeta["rgb_encoding"] == "display_srgb_0_1",
        "prediction semantics unsupported",
    )
    require(
        predmeta["depth_validation_receipt_sha256"] == depth_receipt["sha256"],
        "renderer probe digest mismatch",
    )
    verified = all(
        predmeta["depth_validation"].get(k) is True
        for k in ("off_axis_plane_passed", "alpha_normalization_passed")
    )
    require(verified == evalmeta["depth_verified"], "evaluation and prediction depth status differ")
    ply = ev.file(gen, "payload/scene.ply")
    ply_meta = ev.member(gen, "payload/scene.ply")
    require(
        geno["ply_sha256"] == ply_meta["sha256"] == predmeta["ply_sha256"]
        and geno["ply_bytes"] == ply_meta["bytes"],
        "PLY identity mismatch",
    )
    meshes = ev.read(ref, "payload/scene/reference_config.json")["reference"]["objects"]
    names = {o["id"]: o["name"] for o in meshes}
    require(
        names[1] in ctx["aliases"]["mesh_id_1_names"] and names[2] in ctx["aliases"]["mesh_id_2_names"],
        "unrecognised studio wall/floor meshes",
    )
    require(
        geno["source_size_wh"] == [source_cam.width, source_cam.height]
        and abs(geno["focal_length_px"] - source_cam.K[0, 0]) <= abs_tol,
        "source input focal or dimensions mismatch",
    )
    if c == "R1":
        s = orig["source"]
        scale = np.array([[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1]])
        require(
            (s.width, s.height) == (2 * source_cam.width, 2 * source_cam.height)
            and np.allclose(scale @ s.K, source_cam.K, atol=abs_tol, rtol=0)
            and np.allclose(s.c2w, source_cam.c2w, rtol=0, atol=abs_tol),
            "R1 half-size pixel-centre camera relationship",
        )
    report = {
        "family": f,
        "condition": c,
        "selection_sha256": case["selection_sha256"],
        "originals": ids,
        "input_png": ev.member(cr, input_rel),
        "camera": ev.member(cr, camera_rel),
        "original_reference_camera": ev.member(ref, camera_rel),
        "generator": {
            k: geno[k]
            for k in (
                "generator_commit",
                "checkpoint_sha256",
                "native_frame",
                "source_size_wh",
                "focal_length_px",
                "network_input_wh",
                "resize",
                "channels",
                "ply_color",
                "gaussian_count",
            )
        },
        "render_runtime": {k: predmeta["runtime"].get(k) for k in ("gsplat", "torch", "torch_cuda")},
        "render_settings": predmeta["render_settings"],
        "depth_receipt": depth_receipt,
        "ply": ply_meta,
        "ply_header": read_ply_header(ply),
        "source_K": source_cam.K.tolist(),
        "source_c2w": source_cam.c2w.tolist(),
        "source_w2c": np.linalg.inv(source_cam.c2w).tolist(),
        "native_view_calibration": {},
    }
    if case.get("comparison"):
        comp = ev.run(case["comparison"])
        record = ev.read(comp, "payload/formal_evaluation.json")
        require(
            record["scene_family_id"] == f
            and record["condition_id"] == c
            and record["input_sha256"] == cr.files[input_rel]["sha256"],
            "formal comparison lineage",
        )
    else:
        record = None
    for view in VIEWS:
        cam = cams[view]
        report["native_view_calibration"][view] = {
            "width": cam.width,
            "height": cam.height,
            "K": cam.K.tolist(),
            "c2w": cam.c2w.tolist(),
            "w2c": np.linalg.inv(cam.c2w).tolist(),
        }
        rel = "payload/" + view + ".npz"
        visible = predmeta.get("views", {}).get(view)
        if visible:
            pf = ev.file(render, rel)
            require(render.files[rel]["sha256"] == visible["sha256"], "prediction array SHA")
            pred = arrays(pf, ("rgb", "z", "alpha"))
            require(
                all(v.dtype == np.float32 for v in pred.values())
                and pred["z"].shape == pred["alpha"].shape == cam.shape
                and pred["rgb"].shape == (*cam.shape, 3),
                "prediction native dimensions/dtype",
            )
        else:
            require(view in predmeta.get("failed_views", {}), "undeclared absent prediction")
            pred = None
        if view == "source" and c == "R1":
            if f not in ctx["supplements"]:
                ctx["missing"].append(
                    {
                        "family": f,
                        "condition": c,
                        "view": view,
                        "reason": "native_256_source_truth_unavailable",
                        "original_native_grid": [orig[view].width, orig[view].height],
                        "required_grid": [cam.width, cam.height],
                    }
                )
                continue
            rp, refent = _source_supplement(ev, ctx["supplements"][f], case, cam, abs_tol)
        else:
            require(same_camera(orig[view], cam, abs_tol), "using wrong native ground truth grid")
            refrel = "payload/reference/" + view + ".npz"
            rp = ev.file(ref, refrel)
            refent = ev.member(ref, refrel)
            require(
                refent["sha256"] == rmeta["views"][view]["reference_sha256"], "native truth hash mismatch"
            )
        gt = native_reference(rp, cam, ctx["tolerances"])
        lab = label_view(gt, pred, verified)
        if view != "source":
            labelrel = f"payload/labels/{view}.npz"
            require(
                evalmeta["views"][view]["label_sha256"] == evaluation.files[labelrel]["sha256"],
                "frozen evaluation label mismatch",
            )
            assert_labels(ev.file(evaluation, labelrel), lab)
        common = {
            "family": f,
            "condition": c,
            "view": view,
            "width": cam.width,
            "height": cam.height,
            "prediction_status": "available" if pred is not None else "missing",
        }
        for region, mask in masks(gt).items():
            ctx["rows"].append({**common, **regional_statistics(gt, pred, lab, region, mask, verified)})
        if pred is None:
            ctx["missing"].append({**common, "reason": predmeta["failed_views"][view]})
        if (
            f in ctx["selected_boards"]
            and c == "B0"
            and view == ctx["selected_boards"][f]
            and pred is not None
        ):
            learned = ev.run(case["learned"])
            lm = ev.read(learned, "payload/l1_score.json")
            require(
                lm["producer_run_id"] == render.pin["run_id"]
                and lm["model_id"] == record["model_id"]
                and learned.files[rel]["sha256"] == lm["views"][view]["sha256"],
                "learned case identity",
            )
            tau = lm["thresholds"]["L1-full"]
            scores = arrays(ev.file(learned, rel), ("l1_full", "l1_full_state", "l1_full_valid"))
            require(
                np.array_equal(scores["l1_full_valid"], np.isfinite(scores["l1_full"]))
                and np.array_equal(scores["l1_full_state"], risk_state(scores["l1_full"], tau)),
                "archived L1 state differs from frozen threshold",
            )
            board_sources = {
                "input": ev.member(cr, input_rel),
                "camera": ev.member(cr, camera_rel),
                "reference": refent,
                "prediction": ev.member(render, rel),
                "labels": ev.member(evaluation, labelrel),
                "learned_scores": ev.member(learned, rel),
                "thresholds": ev.member(learned, "payload/l1_score.json"),
                "prior_comparison": ev.member(comp, "payload/formal_evaluation.json"),
            }
            with Image.open(png) as artwork:
                b = private_board(
                    ctx["out"] / "private-cases" / f,
                    f,
                    ctx["rid"],
                    artwork.convert("RGB"),
                    gt,
                    pred,
                    lab,
                    scores["l1_full"],
                    scores["l1_full_state"],
                    tau,
                    board_sources,
                )
            ctx["boards"].append(
                {
                    "family": f,
                    "path": f"private-cases/{f}/board.png",
                    "sha256": b["sha256"],
                    "counts": b["counts"],
                }
            )
    if record and c == "B0":
        ctx["reproduction"][f] = record["family"]["all"]["counts"]
    ctx["cases"][key] = report


def scientific_summary(ev, plan, ctx):
    report = ev.run(plan["original_report"])
    family_file = ev.file(report, "payload/public-numerical/family-results.csv")
    summary_file = ev.file(report, "payload/public-numerical/family-equal-summary.csv")
    public_sel = ev.read(report, "payload/public-numerical/selection.json")
    require(
        public_sel["kind"] == "heldout_study"
        and all(public_sel["files"][f.name]["sha256"] == sha(f) for f in (family_file, summary_file)),
        "frozen result projections differ",
    )
    with family_file.open() as handle:
        archived = {
            r["family"]: r
            for r in csv.DictReader(handle)
            if r["condition"] == "B0" and r["region"] == "all" and r["method"] == "constant"
        }
    with summary_file.open() as handle:
        archived_macro = {
            r["metric"]: float(r["mean"])
            for r in csv.DictReader(handle)
            if r["condition"] == "B0"
            and r["region"] == "all"
            and r["method"] == "constant"
            and r["metric"] in ("depth_mae_m", "depth_high_fraction")
        }
    famrows = []
    tol = plan["tolerances"]
    for f, original in ctx["reproduction"].items():
        rows = [
            r
            for r in ctx["rows"]
            if r["family"] == f and r["condition"] == "B0" and r["view"] != "source" and r["region"] == "all"
        ]
        require(len(rows) == 6, "missing frozen target for reproduction")
        counts = {key: sum(r[key] for r in rows) for key in ("surface", "Z", "ZY", "V", "E", "Y")}
        total = sum(r["z_error_sum_m"] for r in rows)
        mae = fraction(total, counts["Z"])
        high = fraction(counts["ZY"], counts["Z"])
        require(
            all(int(archived[f][k]) == original[k] == v for k, v in counts.items()),
            "frozen family count changed",
        )
        require(
            abs(mae - float(archived[f]["depth_mae_m"])) <= tol["reported_mae_m_abs"]
            and abs(high - float(archived[f]["depth_high_fraction"])) <= tol["reported_depth_high_rate_abs"],
            "frozen family rates changed",
        )
        famrows.append(
            {
                "family": f,
                "completed_targets": len(rows),
                **counts,
                "z_error_sum_m": total,
                "depth_mae_m": mae,
                "depth_high_fraction": high,
                "frozen_mae_m": float(archived[f]["depth_mae_m"]),
                "frozen_high_fraction": float(archived[f]["depth_high_fraction"]),
            }
        )
    require(len(famrows) == 4, "four B0 frozen families required")
    macro = {k: mean(r[k] for r in famrows) for k in ("depth_mae_m", "depth_high_fraction")}
    require(
        abs(macro["depth_mae_m"] - archived_macro["depth_mae_m"]) <= tol["reported_mae_m_abs"]
        and abs(macro["depth_high_fraction"] - archived_macro["depth_high_fraction"])
        <= tol["reported_depth_high_rate_abs"],
        "original frozen macro mismatch",
    )
    sources = {
        (r["family"], r["condition"]): r
        for r in ctx["rows"]
        if r["view"] == "source" and r["region"] == "all"
    }
    pairs = []
    for f in ctx["selected_boards"]:
        b = sources.get((f, "B0"))
        r = sources.get((f, "R1"))
        row = {
            "family": f,
            "B0_native_source_wh": ctx["cases"][f + "-B0"]["generator"]["source_size_wh"],
            "R1_native_source_wh": ctx["cases"][f + "-R1"]["generator"]["source_size_wh"],
            "status": "compared" if b and r else "missing_native_source_truth",
        }
        for field in (
            "E",
            "surface",
            "Z",
            "ZY",
            "z_mae_m",
            "z_signed_mean_m",
            "z_signed_relative_mean",
            "ratio_q05",
            "ratio_q25",
            "ratio_q50",
            "ratio_q75",
            "ratio_q95",
            "rgb_mae",
        ):
            row[field + "_B0"] = b.get(field) if b else None
            row[field + "_R1"] = r.get(field) if r else None
            x, y = row[field + "_B0"], row[field + "_R1"]
            row[field + "_R1_minus_B0"] = y - x if y is not None and x is not None else None
        pairs.append(row)
    return famrows, macro, archived_macro, pairs


def build(args):
    p = load(args.plan)
    aliases = load(args.aliases)
    cfg = load(args.config)
    index = load(args.native_index)
    require(
        p["schema"] == "el-phase1-depth-plan-1"
        and sha(args.plan) == cfg["analysis_plan_sha256"] == aliases["plan_sha256"]
        and sha(args.aliases) == cfg["alias_evidence_sha256"]
        and sha(args.native_index) == cfg["native_index_sha256"],
        "preregistered plan or supplemental running inputs differ",
    )
    require(
        index["schema"] == "el-phase1-matched-source-index-1"
        and set(index["families"]) == set(p["views_for_private_boards"]),
        "wrong native reference roster",
    )
    ctx = {
        "rid": args.run_id,
        "out": args.out,
        "tolerances": p["tolerances"],
        "selected_boards": p["views_for_private_boards"],
        "aliases": aliases,
        "supplements": index["families"],
        "rows": [],
        "missing": [],
        "cases": {},
        "boards": [],
        "reproduction": {},
    }
    ev = Evidence((args.store, args.cold_shelf))
    roots = (args.store.resolve(), args.cold_shelf.resolve())
    require(roots[0] != roots[1], "native active and shelf roots should differ")
    args.out.mkdir(exist_ok=True)
    require(not any(args.out.iterdir()), "refuse populated output")
    for case in p["cases"]:
        path = args.store / "trials" / case["trial"] / "trial_selection.json"
        require(sha(path) == case["selection_sha256"], "selected trial has changed")
        sel = load(path)
        require(
            sel["selection_status"] == "selected"
            and all(
                all(sel["selected"][k][n] == v for n, v in item.items())
                for k, item in case["selected"].items()
            ),
            "selected upstream identity changed",
        )
        process_case(ev, case, ctx)
    rows = ctx["rows"]
    famrows, macro, expected, pairs = scientific_summary(ev, p, ctx)
    csv_out(args.out / "per-view-regions.csv", rows)
    csv_out(args.out / "per-view-all.csv", [r for r in rows if r["region"] == "all"])
    csv_out(args.out / "heldout-B0-reproduction.csv", famrows)
    csv_out(args.out / "B0-R1-native-source.csv", pairs)
    plots = args.out / "strata"
    plots.mkdir()
    figures = {}
    for c in p["cases"]:
        f, condition = c["family"], c["condition"]
        name = f"{f}-{condition}.png"
        file = plots / name
        figures["strata/" + name] = strata_plot(file, f, condition, rows, VIEWS)
    write_json(
        args.out / "materials.json",
        {
            "schema": "el-phase1-depth-materials-1",
            "plan_sha256": sha(args.plan),
            "native_index_sha256": sha(args.native_index),
            "cases": ctx["cases"],
            "source_files": list(ev.members.values()),
            "missing": ctx["missing"],
        },
    )
    write_json(
        args.out / "access.json",
        {
            "schema": "el-depth-owner-access-1",
            "owner_only_files": {
                "private-cases/": "Contains original formal input, reference render, SHARP predictions and derived image panels.",
                "materials.json": "Internal run lineage and exact source asset identities.",
            },
            "aggregate_readable_files": [
                "per-view-regions.csv",
                "per-view-all.csv",
                "heldout-B0-reproduction.csv",
                "B0-R1-native-source.csv",
                "strata/",
                "summary.json",
                "README.md",
            ],
            "publication_status": "Local study evidence. No upload, no distribution rights determination.",
        },
    )
    summary = {
        "schema": "el-phase1-depth-results-1",
        "run_id": args.run_id,
        "analysis_plan_sha256": sha(args.plan),
        "alias_addendum_sha256": sha(args.aliases),
        "native_index_sha256": sha(args.native_index),
        "source_report": p["original_report"],
        "planned_views": len(p["cases"]) * len(VIEWS),
        "observed_numeric_views": sum(r["region"] == "all" for r in rows),
        "region_rows": len(rows),
        "frozen_reproduction": macro,
        "frozen_archived": expected,
        "frozen_count_parity": True,
        "Z_outside_E_total": sum(r["Z_outside_E"] for r in rows if r["region"] == "all"),
        "source_pairs": pairs,
        "boards": ctx["boards"],
        "strata_figures": figures,
        "missing": ctx["missing"],
    }
    write_json(args.out / "summary.json", summary)
    parts = [
        "# Camera-Z investigation",
        "",
        f"All four original B0 families reproduced from the 24 original targets. Family-equal camera-Z MAE is {macro['depth_mae_m']:.9f} m and {100 * macro['depth_high_fraction']:.6f}% of comparable depths exceed max(0.02 m, 2% of reference Z). Source views are separate diagnostic observations.",
        "",
        "## Native source cameras",
        "",
        "Each row compares two separately rendered camera grids from the same family. Percentiles describe corresponding pixels on each native grid. Scene-surface correspondence remains unverified.",
        "",
        "| Family | B0 source ratio p05 / p50 / p95 | R1 source ratio p05 / p50 / p95 | Change in median |",
        "| --- | --- | --- | --- |",
    ]
    for r in pairs:

        def trio(c, r=r):
            return " / ".join(
                "N/A" if r[k + "_" + c] is None else f"{r[k + '_' + c]:.4f}"
                for k in ("ratio_q05", "ratio_q50", "ratio_q95")
            )

        d = r["ratio_q50_R1_minus_B0"]
        x = "N/A" if d is None else f"{d:+.4f}"
        parts.append(f"| {r['family']} | {trio('B0')} | {trio('R1')} | {x} |")
    parts += [
        "",
        "## Numerical evidence",
        "",
        "[All views and regions](per-view-regions.csv) · [Full view counts and distributions](per-view-all.csv) · [Original held-out reproduction](heldout-B0-reproduction.csv) · [Native source comparisons](B0-R1-native-source.csv) · [Signed source inventory](materials.json) · [Access classification](access.json)",
        "",
        "## Fixed v01 cases",
        "",
    ]
    for b in ctx["boards"]:
        parts.append(f"- [{b['family']}]({b['path']})")
    parts.extend(["", "## Fixed-scale ratio distributions", ""])
    for fig in figures:
        parts.append(f"- [{Path(fig).stem}]({fig})")
    (args.out / "README.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    return {
        k: summary[k]
        for k in ("run_id", "observed_numeric_views", "region_rows", "frozen_reproduction", "missing")
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("plan", "aliases", "config", "native-index", "store", "cold-shelf", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    import json

    print(json.dumps(build(args), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
