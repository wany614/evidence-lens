"""Prepare one explicitly pinned train/dev family and read a bounded split index."""

import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np

from .features import FRAME_STATES
from .l1 import ALL_COLUMNS, FEATURE_SCHEMA, NUMERIC, feature_rows, schema_digest
from .scoring_io import TARGETS, ReadAudit, file_hash, new_output, write_json


SEED = 9184
SAMPLE_CAP = 4096
HASH = re.compile(r"[0-9a-f]{64}")
FAMILY = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}")
PIN_HASHES = ("split_manifest_sha256", "selection_sha256", "score_manifest_sha256", "score_receipt_sha256",
              "evaluation_manifest_sha256", "evaluation_receipt_sha256")


def check(ok, message):
    if not ok:
        raise ValueError(message)


def verified_bytes(path, expected, audit, role):
    check(isinstance(expected, str) and HASH.fullmatch(expected), f"missing SHA-256 for {role}")
    data = audit.read(role, path)
    check(audit.sha(role) == expected, f"digest mismatch for {role}")
    return data


def seed_for(family, view, split_hash):
    src = f"{SEED}\n{split_hash}\n{family}\n{view}".encode()
    return int.from_bytes(hashlib.sha256(src).digest()[:16], "big")


def sample_indices(eligible, family, view, split_hash):
    indices = np.flatnonzero(eligible)
    if indices.size <= SAMPLE_CAP:
        return indices
    rng = np.random.Generator(np.random.PCG64(seed_for(family, view, split_hash)))
    return np.sort(rng.choice(indices, size=SAMPLE_CAP, replace=False))


def _check_pin(pin, split=None):
    check(pin.get("schema") == "el-learning-family-1" and pin.get("split") in ("train", "dev")
          and pin.get("condition_id") == "B0" and (split is None or pin["split"] == split), "ineligible split or condition")
    name = pin.get("scene_family_id")
    check(isinstance(name, str) and FAMILY.fullmatch(name), "invalid family ID")
    for k in PIN_HASHES:
        check(isinstance(pin.get(k), str) and HASH.fullmatch(pin[k]), f"missing pinned {k}")
    for k in ("scene_id", "trial_id", "score_run_id", "evaluation_run_id"):
        check(isinstance(pin.get(k), str) and pin[k].strip(), f"missing {k}")
    if pin["split"] == "train":
        check(type(pin.get("cv_fold")) is int and pin["cv_fold"] in range(4), "train pin needs the frozen cv_fold 0..3")


def _validate_columns(features, shape):
    for name in [*ALL_COLUMNS, "scoreable", "alpha_score"]:
        value = features[name]
        dtype = np.float32 if name in NUMERIC or name == "alpha_score" else np.bool_
        check(value.shape == shape and value.dtype == dtype, f"invalid feature shape or dtype for {name}")
    for name in NUMERIC:
        check(np.array_equal(features["missing_" + name], ~np.isfinite(features[name])), f"incorrect missing flag for {name}")
    check((sum(features["frame_" + s].astype(np.uint8) for s in FRAME_STATES) == 1).all(), "invalid frame one-hot flags")
    check(np.array_equal(features["alpha_score"], 1 - features["target_alpha"], equal_nan=True), "inconsistent alpha risk")


def _catalog(path, digest, audit):
    data = json.loads(verified_bytes(path, digest, audit, "split_catalog"))
    check(data.get("schema") == "el-formal-families-1", "invalid frozen family catalog schema")
    rows = {r["scene_family_id"]: r for r in data["families"]}
    check(len(rows) == len(data["families"]), "duplicate family in frozen catalog")
    return rows


def _match_catalog(pin, rows):
    row = rows.get(pin["scene_family_id"], {})
    split = "development" if pin["split"] == "dev" else "train"
    check(row.get("split") == split and all(row.get(k) == pin.get(k) for k in ("scene_id", "trial_id", "cv_fold")),
          "family pin contradicts frozen assignment")


def prepare(family_path, scores, evaluation, output, catalog):
    audit = ReadAudit()
    pin = json.loads(audit.read("family_pin", family_path))
    _check_pin(pin)
    _match_catalog(pin, _catalog(catalog, pin["split_manifest_sha256"], audit))
    family, split = pin["scene_family_id"], pin["split"]
    s = json.loads(verified_bytes(scores / "score.json", pin["score_receipt_sha256"], audit, "score_receipt"))
    e = json.loads(verified_bytes(evaluation / "evaluation.json", pin["evaluation_receipt_sha256"], audit, "evaluation_receipt"))
    features_doc = json.loads(verified_bytes(scores / "feature_schema.json", s["feature_schema_sha256"], audit, "feature_schema"))
    check(features_doc == FEATURE_SCHEMA and s["schema"] == "el-score-1" and e["schema"] == "el-evaluation-1"
          and e["label_version"] == "el-labels-1", "incompatible feature or label semantics")
    check(s["run_id"] == pin["score_run_id"] == e["scoring_run_id"] and e["run_id"] == pin["evaluation_run_id"]
          and s["producer_run_id"] == e["producer_run_id"], "source run linkage mismatch")
    check(all(e[k] == pin[k] for k in ("scene_family_id", "scene_id", "trial_id", "condition_id")), "family identity mismatch")
    check(set(s["views"]) == set(e["views"]) == set(TARGETS), "six planned target views required")
    check(s["depth_verified"] is True and e["depth_verified"] is True, "incompatible provider depth validation")
    check(e["inputs"]["score_metadata"] == audit.files["score_receipt"]
          and s.get("depth_validation_receipt_sha256") == e.get("depth_validation_receipt_sha256")
          and all(value == e["inputs"].get(role) for role, value in s["inputs"].items()),
          "evaluation used different score, camera, prediction or depth-validation inputs")
    input_hash = s["inputs"]["input_png"]["sha256"]
    check(HASH.fullmatch(input_hash), "source image lineage mismatch")
    with new_output(output) as out:
        receipt = {"schema": "el-learning-extract-1", "pin": pin, "inputs": audit.files, "source_input_sha256": input_hash,
                   "feature_schema_digest": schema_digest(), "label_version": "el-labels-1", "views": {}}
        for view in TARGETS:
            se, ev = s["views"][view], e["views"][view]
            lb = verified_bytes(evaluation / "labels" / f"{view}.npz", ev["label_sha256"], audit, "labels/" + view)
            with np.load(io.BytesIO(lb), allow_pickle=False) as doc:
                v, both, label = (doc[k] for k in ("reference_valid", "rgb_evaluable", "rgb_high_error"))
            check(v.ndim == 2 and v.shape == both.shape == label.shape and v.dtype == both.dtype == np.bool_
                  and label.dtype == np.uint8 and not (both & ~v).any() and np.isin(label[both], [0, 1]).all()
                  and (label[~both] == 255).all(), f"{view}: invalid label masks")
            check(int(v.sum()) == ev["reference_valid"] and int(both.sum()) == ev["rgb_evaluable"], f"{view}: label counts differ")
            rows = np.empty(0, np.intp)
            x = np.empty((0, len(ALL_COLUMNS)), np.float32)
            available = np.empty(0, bool)
            alpha = np.empty(0, np.float32)
            population = 0
            if se["status"] == "scored":
                check(ev["render_status"] == "completed", f"{view}: contradictory render status")
                sb = verified_bytes(scores / f"{view}.npz", se["score_sha256"], audit, "features/" + view)
                check(e["inputs"]["score/" + view] == audit.files["features/" + view], f"{view}: score and evaluation array linkage mismatch")
                with np.load(io.BytesIO(sb), allow_pickle=False) as doc:
                    features = {k: doc[k] for k in [*ALL_COLUMNS, "scoreable", "alpha_score"]}
                _validate_columns(features, v.shape)
                eligible = both & features["scoreable"]
                population = int(eligible.sum())
                rows = sample_indices(eligible, family, view, pin["split_manifest_sha256"]) if split == "train" else np.flatnonzero(both)
                x = feature_rows(features, rows)
                available = features["scoreable"].ravel()[rows]
                alpha = features["alpha_score"].ravel()[rows]
            else:
                check(se["status"] == "render_failed" and ev["render_status"] == "failed" and not both.any(), f"{view}: unexplained missing view")
            y = label.ravel()[rows]
            path = out / f"{view}.npz"
            np.savez_compressed(path, index=rows.astype(np.int64), X=x, Y=y, scoreable=available, alpha_score=alpha)
            receipt["views"][view] = {
                "sha256": file_hash(path), "bytes": path.stat().st_size, "shape": list(v.shape), "V": int(v.sum()), "E": int(both.sum()),
                "positive_E": int((label == 1).sum()), "eligible": population, "rows": int(rows.size),
                "positive_rows": int(y.sum()), "sample_probability": (rows.size / population if population else None) if split == "train" else 1.0,
                "seed": str(seed_for(family, view, pin["split_manifest_sha256"])) if split == "train" else None,
            }
        write_json(out / "family.json", receipt)
    return receipt


class Dataset:
    """Access only named families and six numeric files underneath each explicit extract."""
    def __init__(self, root, split):
        self.root = Path(root)
        self.audit = ReadAudit()
        self.index = json.loads(self.audit.read("dataset_index", self.root / "dataset.json"))
        i = self.index
        check(i.get("schema") == "el-learning-index-1" and i.get("split") == split and i.get("condition_id") == "B0"
              and isinstance(i.get("split_manifest_sha256"), str) and HASH.fullmatch(i["split_manifest_sha256"]),
              "wrong dataset split, condition or split pin")
        entries = i["families"]
        check(len(entries) == {"train": 16, "dev": 4}[split], f"expected complete {split} family roster")
        names = [entry["scene_family_id"] for entry in entries]
        check(all(isinstance(n, str) and FAMILY.fullmatch(n) for n in names) and len(set(names)) == len(names), "invalid or duplicate family")
        roster = _catalog(self.root / "catalog.json", i["split_manifest_sha256"], self.audit)
        canonical_split = "development" if split == "dev" else "train"
        check(set(names) == {n for n, row in roster.items() if row["split"] == canonical_split}, "index differs from frozen split roster")
        self.families = {}
        for item in sorted(entries, key=lambda r: r["scene_family_id"]):
            name = item["scene_family_id"]
            path = self.root / name / "family.json"
            rec = json.loads(verified_bytes(path, item["extract_manifest_sha256"], self.audit, name + "/family"))
            pin = rec["pin"]
            _check_pin(pin, split)
            _match_catalog(pin, roster)
            check(rec["schema"] == "el-learning-extract-1" and rec["feature_schema_digest"] == schema_digest()
                  and rec["label_version"] == "el-labels-1" and pin["split_manifest_sha256"] == i["split_manifest_sha256"]
                  and pin["scene_family_id"] == name and set(rec["views"]) == set(TARGETS), "extract lineage mismatch")
            self.families[name] = rec
        check(len({r["source_input_sha256"] for r in self.families.values()}) == len(names), "duplicate source-image content across families")

    def read(self, family, view):
        info = self.families[family]["views"][view]
        path = self.root / family / f"{view}.npz"
        role = family + "/" + view
        data = verified_bytes(path, info["sha256"], self.audit, role)
        check(len(data) == info["bytes"], f"{role}: size mismatch")
        with np.load(io.BytesIO(data), allow_pickle=False) as file:
            arrays = {key: file[key] for key in ("X", "Y", "index", "scoreable", "alpha_score")}
        n = info["rows"]
        h, w = info["shape"]
        x, y, coords = arrays["X"], arrays["Y"], arrays["index"]
        check(x.shape == (n, len(ALL_COLUMNS)) and x.dtype == np.float32 and y.shape == (n,) and y.dtype == np.uint8
              and np.isin(y, [0, 1]).all() and int(y.sum()) == info["positive_rows"], f"{role}: invalid sample table")
        check(coords.shape == (n,) and coords.dtype == np.int64 and (n == 0 or (coords[0] >= 0 and coords[-1] < h * w
              and (np.diff(coords) > 0).all())), f"{role}: pixel coordinates do not represent uniform unique rows")
        check(arrays["scoreable"].shape == (n,) and arrays["scoreable"].dtype == np.bool_ and arrays["alpha_score"].shape == (n,)
              and arrays["alpha_score"].dtype == np.float32, f"{role}: invalid score availability")
        frame = x[:, len(NUMERIC):len(NUMERIC) + len(FRAME_STATES)]
        missing = x[:, len(NUMERIC) + len(FRAME_STATES):]
        check(((frame == 0) | (frame == 1)).all() and (frame.sum(axis=1) == 1).all()
              and np.array_equal(missing, ~np.isfinite(x[:, :len(NUMERIC)])), f"{role}: invalid feature flags")
        alpha = x[:, ALL_COLUMNS.index("target_alpha")]
        check(np.all(~np.isfinite(alpha) | ((alpha >= 0) & (alpha <= 1)))
              and np.array_equal(arrays["alpha_score"], 1 - alpha, equal_nan=True), f"{role}: invalid alpha scores")
        check(0 <= info["positive_E"] <= info["E"] <= info["V"] <= h*w and 0 <= info["eligible"] <= info["E"], f"{role}: invalid counts")
        if self.index["split"] == "train":
            expected_p = n / info["eligible"] if info["eligible"] else None
            check(n == min(SAMPLE_CAP, info["eligible"]) and arrays["scoreable"].all()
                  and info["sample_probability"] == expected_p
                  and info["seed"] == str(seed_for(family, view, self.index["split_manifest_sha256"])), "invalid training sampling")
        else:
            check(n == info["E"] and int(arrays["scoreable"].sum()) == info["eligible"]
                  and info["positive_rows"] == info["positive_E"], "dev must preserve every E pixel")
        return arrays
