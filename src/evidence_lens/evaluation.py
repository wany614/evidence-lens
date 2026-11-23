"""Evaluate P2 predictions and scores against the private reference package."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .comparison_board import render_board
from .labels import LABEL_VERSION, RGB_THRESHOLDS, VISIBILITY, ReferenceBundle, label_view
from .risk_model import HIGH, REASONS
from .scoring_io import (
    TARGETS, PredictionBundle, ReadAudit, alpha_valid, file_hash, load_cameras, load_source_png,
    new_output, write_json,
)


METHODS = {"L0": "l0", "1-alpha": "alpha_score"}
METRIC_VERSION = "el-p2-metrics-1"


class ScoreBundle:
    def __init__(self, directory, bundle, audit):
        self.directory = directory
        self.audit = audit
        self.bundle = bundle
        self.meta = json.loads(audit.read("score_metadata", directory / "score.json"))
        if (self.meta.get("schema") != "el-score-1" or self.meta.get("depth_verified") is not bundle.depth_verified
                or self.meta.get("producer_run_id") != bundle.meta["producer_run_id"]
                or set(self.meta["views"]) != set(TARGETS)):
            raise ValueError("incompatible score.json")
        expected = {"cameras", "input_png", "prediction_metadata"}
        expected.update(f"prediction/{view}" for view in ("source", *TARGETS) if view not in bundle.failed)
        if set(self.meta["inputs"]) != expected:
            raise ValueError("score.json input roles differ from allowed scoring inputs")
        for role in ("cameras", "input_png", "prediction_metadata", "prediction/source"):
            if role in expected:
                self.check_input(role)

    def check_input(self, role):
        if self.meta["inputs"][role] != self.audit.files[role]:
            raise ValueError(f"{role}: score input digest or size mismatch")

    def read(self, view, pred):
        entry = self.meta["views"][view]
        if pred is None:
            if entry != {"status": "render_failed", "reason": self.bundle.failed[view]}:
                raise ValueError(f"{view}: conflicting render failure in score.json")
            return None
        self.check_input(f"prediction/{view}")
        role = f"score/{view}"
        data = self.audit.read(role, self.directory / f"{view}.npz")
        if entry.get("status") != "scored" or entry.get("score_sha256") != self.audit.sha(role):
            raise ValueError(f"{role}: score hash or status mismatch")
        names = ("l0", "l0_valid", "l0_reason", "alpha_score", "alpha_score_valid")
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            scores = {k: archive[k] for k in names}
        dtypes = (np.float32, np.bool_, np.uint8, np.float32, np.bool_)
        if any(scores[k].shape != pred["z"].shape or scores[k].dtype != t for k, t in zip(names, dtypes)):
            raise ValueError(f"{role}: invalid score array dimensions or dtype")
        r = scores["l0_reason"]
        expected_l0 = np.full(r.shape, np.nan, np.float32)
        expected_l0[r == 0], expected_l0[np.isin(r, HIGH)] = 0, 1
        expected_alpha = np.where(alpha_valid(pred["alpha"]), 1 - pred["alpha"], np.float32(np.nan))
        if (np.any(r > max(REASONS)) or not np.array_equal(scores["l0"], expected_l0, equal_nan=True)
                or not np.array_equal(scores["alpha_score"], expected_alpha, equal_nan=True)
                or not np.array_equal(scores["l0_valid"], np.isfinite(expected_l0))
                or not np.array_equal(scores["alpha_score_valid"], np.isfinite(expected_alpha))):
            raise ValueError(f"{role}: inconsistent score, reason code or availability")
        return scores


def _mask_hash(mask):
    h = hashlib.sha256()
    h.update(np.asarray(mask.shape, dtype="<u8").tobytes())
    h.update(np.packbits(mask, bitorder="little").tobytes())
    return h.hexdigest()


def _regions(labels):
    edge = labels["edge"]
    yield "all", np.ones(edge.shape, bool)
    yield "edge", edge
    yield "interior", ~edge
    for code, name in VISIBILITY.items():
        yield name, labels["visibility"] == code
    yield "prediction_low_contribution", labels["low_alpha"]
    yield "prediction_sufficient_contribution", labels["alpha_valid"] & ~labels["low_alpha"]
    yield "prediction_unknown_contribution", ~labels["alpha_valid"]


class RegionRows:
    def __init__(self, identity, sources, region, v, e):
        self.rows = []
        self.mask_cache = {}
        self.base = dict(identity, region_name=region, reference_pixel_count=int(v.sum()),
                         prediction_valid_count=int(e.sum()), metric_version=METRIC_VERSION,
                         label_version=LABEL_VERSION, threshold_version="p2-initial-1",
                         model_id=None, source_artifact_ids=sources)

    def add(self, method, name, mask_id, mask, *, numerator=None, value=None, unit="fraction",
            empty="empty_denominator", undefined=None, scoreable_count=None):
        if mask_id not in self.mask_cache:
            self.mask_cache[mask_id] = int(np.count_nonzero(mask)), _mask_hash(mask)
        n, mask_hash = self.mask_cache[mask_id]
        reason = undefined or (empty if n == 0 else None)
        if numerator is not None and n and reason is None:
            value = numerator / n
        self.rows.append(dict(
            self.base, method_id=method, metric_name=name, numerator=numerator, denominator=n,
            value=value if reason is None else None, unit=unit,
            status="defined" if reason is None else "undefined", missing_reason=reason,
            mask_id=mask_id, mask_hash=mask_hash, scoreable_count=scoreable_count,
        ))


def _groups(values, target):
    """Accumulate whole equal-score groups in ascending risk order."""
    if values.size == 0:
        return np.empty(0, np.float32), np.empty(0, np.int64), np.empty(0, np.int64)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    sizes = np.diff(np.r_[starts, sorted_values.size])
    errors = np.add.reduceat(target[order].astype(np.int64), starts)
    return sorted_values[starts], sizes, errors


def _ap(sizes, errors):
    positives = errors.sum()
    if not 0 < positives < sizes.sum():
        return None
    cumulative_p = np.cumsum(errors[::-1])
    cumulative_n = np.cumsum(sizes[::-1])
    return float(np.dot(cumulative_p / cumulative_n, errors[::-1]) / positives)


def _view_metrics(labels, scores, identity, sources, failure_reason):
    rows, curves = [], {}
    y = labels["rgb_high_error"] == 1
    for region, mask in _regions(labels):
        v, e = mask & labels["reference_valid"], mask & labels["rgb_evaluable"]
        m = RegionRows(identity, sources, region, v, e)
        no_rgb = failure_reason or "no_evaluable_rgb"
        m.add("output", "prediction_valid_fraction", "V", v, numerator=int(e.sum()), empty="no_reference_rgb")
        m.add("output", "invalid_output_fraction", "V", v, numerator=int((v & ~e).sum()), empty="no_reference_rgb")
        m.add("output", "rgb_mae", "E", e, numerator=float(labels["rgb_error"][e].sum(dtype=np.float64)),
              unit="display_srgb", empty=no_rgb)
        for threshold in RGB_THRESHOLDS:
            count = np.count_nonzero(e & (labels["rgb_error"] > np.float32(threshold)))
            m.add("output", f"rgb_error_fraction_gt_{threshold:g}", "E", e, numerator=int(count), empty=no_rgb)
        z_eval = mask & labels["depth_evaluable"]
        z_empty = failure_reason or "no_evaluable_depth"
        m.add("output", "depth_mae", "depth_evaluable", z_eval, unit="m", empty=z_empty,
              numerator=float(labels["depth_error"][z_eval].sum(dtype=np.float64)))
        m.add("output", "depth_error_fraction", "depth_evaluable", z_eval, empty=z_empty,
              numerator=int(((labels["depth_high_error"] == 1) & z_eval).sum()))
        for field, unit in (("signed_depth_error", "m"), ("relative_depth_error", "fraction"),
                            ("signed_relative_depth_error", "fraction")):
            value = float(np.median(labels[field][z_eval])) if z_eval.any() else None
            m.add("output", field + "_median", "depth_evaluable", z_eval, unit=unit, value=value, empty=z_empty)
        support_mask = mask & labels["reference_surface"] & labels["alpha_valid"]
        m.add("output", "low_support_fraction", "surface_with_valid_alpha", support_mask,
              numerator=int((mask & labels["low_support"]).sum()))
        common = e.copy()
        for key in METHODS.values():
            common &= np.isfinite(scores[key]) if scores is not None else False
        for method, key in METHODS.items():
            s = e & np.isfinite(scores[key]) if scores is not None else np.zeros_like(e)
            high = s & (scores[key] > 0.5) if scores is not None else np.zeros_like(e)
            low, unknown = s & ~high, e & ~s
            n_high_error = int((y & high).sum())
            n_low_error = int((y & low).sum())
            n_unknown_error = int((y & unknown).sum())
            n_s = int(s.sum())
            def ratio(name, count, mask_name, denominator, reason):
                m.add(method, name, f"{method}/{mask_name}", denominator, numerator=count,
                      empty=failure_reason or reason, scoreable_count=n_s)
            ratio("high_risk_recall", n_high_error, "E_positive", e & y, "no_positive_pixels")
            ratio("warning_precision", n_high_error, "H", high, "no_high_risk_pixels")
            ratio("warning_fraction", int(high.sum()), "E", e, "no_evaluable_rgb")
            ratio("low_risk_error_rate", n_low_error, "L", low, "no_low_risk_pixels")
            ratio("overall_low_risk_coverage", int(low.sum()), "V", v, "no_reference_rgb")
            ratio("unknown_fraction", int(unknown.sum()), "E", e, "no_evaluable_rgb")
            ratio("error_fraction_within_unknown", n_unknown_error, "U", unknown, "no_unknown_pixels")
            ratio("unknown_share_of_errors", n_unknown_error, "E_positive", e & y, "no_positive_pixels")
            if region == "all":
                ratio("common_scoreable_fraction", int(common.sum()), "V", v, "no_reference_rgb")
                ratio("common_error_fraction", int((y & common).sum()), "S", common, "no_common_pixels")
                for name, subset in (("common", common), ("own", s)):
                    vals = scores[key][subset] if scores is not None else np.empty(0, np.float32)
                    thresholds, sizes, errors = _groups(vals, y[subset])
                    value = _ap(sizes, errors)
                    m.add(method, "average_precision_" + name, f"{method}/{name}_scores", subset,
                          value=value, unit="average_precision", scoreable_count=n_s,
                          undefined=(failure_reason or "ranking_not_evaluable") if value is None else None)
                    if name == "own":
                        accepted = np.r_[0, np.cumsum(sizes)]
                        n_errors = np.r_[0, np.cumsum(errors)]
                        curves[key + "_threshold"] = np.r_[np.float32(np.nan), thresholds]
                        curves[key + "_accepted"] = accepted
                        curves[key + "_accepted_errors"] = n_errors
                        curves[key + "_coverage"] = accepted / int(v.sum()) if v.any() else np.full(accepted.shape, np.nan)
                        curves[key + "_error_rate"] = np.divide(n_errors, accepted, where=accepted > 0,
                                                               out=np.full(accepted.shape, np.nan))
        rows.extend(m.rows)
    return rows, curves


def _cross_counts(labels, score):
    v = labels["reference_valid"]
    reasons = score["l0_reason"] if score is not None else np.full(v.shape, 10, np.uint8)
    index = reasons[v].astype(np.intp) * 4 + labels["visibility"][v]
    return np.bincount(index, minlength=44).reshape(11, 4)


def _cross_rows(counts, camera):
    reasons = {**REASONS, 10: "render_failed"}
    return [
        {"camera_id": camera, "mask": "reference_valid", "l0_reason": reasons[i],
         "visibility": VISIBILITY[j], "count": int(counts[i, j]),
         "reason_total": int(counts[i].sum()), "visibility_total": int(counts[:, j].sum())}
        for i in range(counts.shape[0]) for j in range(counts.shape[1])
    ]


def _tables(rows, output, parquet):
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, source_artifact_ids=json.dumps(row["source_artifact_ids"])))
    if parquet:
        import pyarrow as pa
        import pyarrow.parquet as pq

        text_cols = ("run_id", "trial_id", "scene_family_id", "scene_id", "condition_id", "camera_id",
                     "method_id", "model_id", "metric_name", "metric_version", "unit", "status", "missing_reason",
                     "mask_id", "mask_hash", "region_name", "label_version", "threshold_version")
        count_cols = ("denominator", "reference_pixel_count", "prediction_valid_count", "scoreable_count")
        schema = pa.schema([(k, pa.string()) for k in text_cols] + [(k, pa.int64()) for k in count_cols] +
                           [("value", pa.float64()), ("numerator", pa.float64()), ("source_artifact_ids", pa.list_(pa.string()))])
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), output / "metrics.parquet")


def run_evaluation(input_path, camera_path, reference_path, prediction_path, score_path, output, identity,
                   *, render_failed=None, parquet=False):
    if render_failed is None and (prediction_path is None or score_path is None):
        raise ValueError("evaluation requires predictions and scores, or an explicit render failure")
    if render_failed is not None and (not render_failed.strip() or prediction_path is not None or score_path is not None):
        raise ValueError("render-failed requires a reason and excludes prediction/score paths")
    audit = ReadAudit()
    cameras = load_cameras(camera_path, audit)
    image = load_source_png(input_path, cameras["source"], audit)
    refs = ReferenceBundle(reference_path, audit)
    predictions = scores = None
    if render_failed is None:
        predictions = PredictionBundle(prediction_path, audit)
        predictions.read("source", cameras["source"])
        scores = ScoreBundle(score_path, predictions, audit)
    rows, cross_rows = [], []
    all_cross = np.zeros((11, 4), dtype=np.int64)
    with new_output(output) as out:
        (out / "labels").mkdir()
        (out / "curves").mkdir()
        views = {}
        for view in TARGETS:
            ref = refs.read(view, cameras[view])
            pred = predictions.read(view, cameras[view]) if predictions else None
            score = scores.read(view, pred) if scores else None
            failure = render_failed or (predictions.failed.get(view) if predictions else None)
            labels = label_view(ref, pred, predictions.depth_verified if predictions else False)
            label_path = out / "labels" / f"{view}.npz"
            np.savez_compressed(label_path, **labels)
            label_sha = file_hash(label_path)
            input_roles = ["input_png", "cameras", "reference_metadata", f"reference/{view}"]
            if predictions:
                input_roles.extend(["prediction_metadata", "score_metadata"])
                if "source" not in predictions.failed:
                    input_roles.append("prediction/source")
                if pred is not None:
                    input_roles.extend([f"prediction/{view}", f"score/{view}"])
            sources = [f"{role}:sha256:{audit.sha(role)}" for role in input_roles]
            sources.append(f"labels/{view}:sha256:{label_sha}")
            rowset, curves = _view_metrics(labels, score, dict(identity, camera_id=view), sources, failure)
            rows.extend(rowset)
            np.savez_compressed(out / "curves" / f"{view}.npz", **curves)
            counts = _cross_counts(labels, score)
            all_cross += counts
            cross_rows.extend(_cross_rows(counts, view))
            views[view] = {"render_status": "failed" if failure else "completed", "render_failure_reason": failure,
                           "label_sha256": label_sha, "reference_valid": int(labels["reference_valid"].sum()),
                           "rgb_evaluable": int(labels["rgb_evaluable"].sum()),
                           "depth_evaluable": int(labels["depth_evaluable"].sum())}
            if view == "v01":
                board = out / "board_v01.png"
                render_board(board, image, ref, pred, labels, score, identity["run_id"])
                write_json(out / "board_v01.json", {
                    "schema": "el-p2-board-1", **identity, "camera_id": view,
                    "selection": "first planned target camera v01", "source_artifact_ids": sources,
                    "rgb_error_palette_stops": [0, 0.025, 0.05, 0.10, 0.20],
                    "error_out_of_range": "saturate at 0.20", "risk_codes": ["low", "high", "unknown"],
                    "visibility_codes": VISIBILITY, "board_sha256": file_hash(board),
                })
        _tables(rows, out, parquet)
        write_json(out / "reason_visibility_counts.json", {
            "schema": "el-reason-visibility-1", **identity,
            "views": cross_rows, "family": _cross_rows(all_cross, "all_six_target_views"),
        })
        receipt = {
            "schema": "el-evaluation-1", **identity, "views": views, "inputs": audit.files,
            "producer_run_id": predictions.meta["producer_run_id"] if predictions else None,
            "scoring_run_id": scores.meta["run_id"] if scores else None,
            "depth_verified": predictions.depth_verified if predictions else False,
            "planned_target_views": len(TARGETS), "completed_target_views": sum(v["render_status"] == "completed" for v in views.values()),
            "metric_version": METRIC_VERSION, "label_version": LABEL_VERSION,
            "methods_in_common_ranking": list(METHODS), "metrics_parquet": parquet,
        }
        write_json(out / "evaluation.json", receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--cameras", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True, help="payload containing reference.json and reference/")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--predictions", type=Path, help="same prediction directory used for scoring")
    group.add_argument("--render-failed", metavar="REASON", help="document a known failed render; evaluation still requires its references")
    p.add_argument("--scores", type=Path, help="directory containing score.json and scored arrays")
    p.add_argument("--output", required=True, type=Path, help="new evaluation directory")
    p.add_argument("--parquet", action="store_true", help="also export typed Parquet (requires the record extra)")
    keys = ("run_id", "trial_id", "scene_family_id", "scene_id", "condition_id")
    for key in keys:
        p.add_argument("--" + key.replace("_", "-"), required=True)
    args = p.parse_args(argv)
    if bool(args.predictions) != bool(args.scores):
        p.error("--predictions and --scores must be supplied together")
    run_evaluation(args.input, args.cameras, args.references, args.predictions, args.scores, args.output,
                   {key: getattr(args, key) for key in keys}, render_failed=args.render_failed, parquet=args.parquet)


if __name__ == "__main__":
    main()
