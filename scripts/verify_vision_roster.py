"""Recompute tie-preserving seven-common family AP and verify its exact native render and frozen-model identities."""

import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

import numpy as np
from sklearn.metrics import average_precision_score

from evidence_lens._bundle import HASH, load, sha
from evidence_lens._verify import parent_directory, require
from evidence_lens.formal_inputs import PinnedRun, compact_pin
from evidence_lens.scoring_io import write_json

FIELDS = {"constant": "constant", "1-alpha": "alpha_score", "L0": "l0", "L1-full": "l1_full", "L1-geometry": "l1_geometry"}
VISUAL = ("L2-frozen", "L2-vision-ft")
METHODS = (*FIELDS, *VISUAL)
CONDITIONS = ("B0", "R1", "Q1")
VIEWS = tuple(f"v{i:02}" for i in range(1, 7))


def verify(args):
    store, shelf = args.root.resolve(), args.shelf.resolve()
    origins = {}

    def run(ref):
        pin = compact_pin(ref)
        key = (pin["run_id"], pin["manifest_sha256"])
        if key not in origins:
            path = parent_directory(store, pin, shelf)
            origins[key] = PinnedRun(path.parent.parent, pin)
            require(origins[key].run["execution_status"] == "succeeded", "Unsuccessful scientific parent")
        return origins[key]

    def checked(original, path, pin):
        digest = pin.get("sha256") if isinstance(pin, dict) else pin
        require(isinstance(digest, str) and HASH.fullmatch(digest), "Missing explicit source digest")
        f = original.open(path)
        require(original.files[path]["sha256"] == digest, "Original member differs")
        if isinstance(pin, dict):
            require(pin["bytes"] == f.stat().st_size, "Original size differs")
        return f

    def native(record, spec):
        with np.load(checked(record, spec["path"], spec), allow_pickle=False) as data:
            x = data["rgb"]
        require(x.dtype == np.float32 and x.shape == (512, 512, 3), "Native RGB must be FP32")
        return x

    q = load(args.request)
    cohort = q["cohort"]
    require(cohort in ("new_confirmation", "historical_regression"), "Unknown cohort")
    require(sha(args.catalog) == q["catalog_sha256"] and sha(args.summary) == args.summary_sha256,
            "Another registered roster or numerical report")
    new, old = load(args.catalog), load(args.historical_catalog)
    method = load(checked(run(q["method"]), "payload/method-freeze.json", q["method_sha256"]))
    marker = store / "protocol/stage2" / q["catalog_sha256"] / "test-opening.json"
    require(sha(marker) == q["opening_sha256"] and load(marker)["method_sha256"] == q["method_sha256"]
            and sha(args.historical_catalog) == method["original_catalog_sha256"], "Another original study or method")
    names = {r["scene_family_id"] for r in new["families"] if r["split"] == "test"}
    prior = {r["scene_family_id"] for r in old["families"] if r["split"] == "test"}
    training = {r["scene_family_id"] for r in new["families"] + old["families"] if r["split"] != "test"}
    require(len(names) == 8 and len(prior) == 4 and names == set(load(marker)["test_family_ids"])
            and not names & prior and not (names | prior) & training, "Overlapping or unexpected families")
    families = names if cohort == "new_confirmation" else prior
    expected = {(f, c, v) for f in families for c in CONDITIONS for v in VIEWS}
    stages = {k: run(r) for k, r in q["stages"].items()}
    docs = {k: load(checked(stages[k], "payload/" + r["receipt"], r["receipt_sha256"])) for k, r in q["stages"].items()}
    def roster(rows, columns):
        keys = [tuple(r[k] for k in columns) for r in rows]
        require(len(keys) == len(set(keys)), "Duplicate scientific row")
        return dict(zip(keys, rows))
    keys = ("family", "condition", "view")
    ds = roster(docs["dataset"]["views"], keys)
    cells = roster(docs["materials"]["cells"], keys[:2])
    require(set(ds) == expected and set(cells) == {k[:2] for k in expected}
            and all(set(c["views"]) == set(VIEWS) for c in cells.values()), "Native target grid differs")
    file_keys = {tuple(k[:-5].split("/")) for k in docs["evaluation"]["files"] if re.fullmatch(r"[^/]+/[^/]+/v\d{2}\.json", k)}
    require(file_keys == expected, "Evaluation target roster differs")
    selection = (("score-original-five", "L1-full", tuple(FIELDS)),
                 ("score-L2-frozen", "L2-frozen", ("L2-frozen",)), ("score-L2-vision-ft", "L2-vision-ft", ("L2-vision-ft",)))
    scores = {}
    for group, model, methods in selection:
        doc = docs[group]
        require(doc["model_sha256"] == method["models"][model]["model_sha256"] and
                doc["method_freeze_sha256"] == q["method_sha256"], "Wrong selected model")
        rows = roster(doc["views"], keys)
        require(set(rows) == expected, "Score roster differs")
        scores[group] = rows
    ap, ties, missing = defaultdict(list), Counter(), Counter()
    defined = {m: 0 for m in METHODS}
    undefined = {m: 0 for m in METHODS}
    label_ties = 0
    for ident, d in sorted(ds.items()):
        f, c, v = ident
        cell = cells[f, c]
        source, target = cell["receipts"]["source_prediction"], cell["views"][v]
        require(d["status"] == target["status"] == "completed" and d["source_input_sha256"] == cell["input_sha256"]
                and d["source_camera_sha256"] == cell["camera_sha256"] and d["source_prediction_sha256"] == source["sha256"]
                and d["target_prediction_sha256"] == target["prediction"]["sha256"]
                and d["render_pin"] == cell["parents"]["prediction"], "Material and score refer to different source or target")
        pred = native(run(cell["parents"]["prediction"]), target["prediction"])
        truth = native(run(cell["parents"]["reference"]), target["reference"])
        mask = (np.isfinite(truth) & np.isfinite(pred) & (truth >= 0) & (truth <= 1) & (pred >= 0) & (pred <= 1)).all(axis=-1)
        delta = np.abs(truth - pred).mean(axis=-1, dtype=np.float32)
        y = mask & (delta > np.float32(.05))
        label_ties += int(np.count_nonzero(mask & (delta == np.float32(.05))))
        spec = d["full"]["truth"]
        with np.load(checked(stages["dataset"], "payload/" + spec["path"], spec), allow_pickle=False) as truth_pack:
            event = truth_pack["rgb_high_error"]
            require(np.isin(event[mask], [0, 1]).all() and np.array_equal(event[mask] == 1, y[mask])
                    and np.array_equal(truth_pack["rgb_evaluable"], mask), "Native event coding differs")
        values = {}
        for group, _, methods in selection:
            row = scores[group][ident]
            require(row["status"] == "scored" and row["score"] is not None and
                    all(row[k] == d[k] for k in ("render_pin", "target_prediction_sha256", "source_prediction_sha256",
                                                   "source_camera_sha256", "source_input_sha256")), "Score lineage is incomplete")
            spec = row["score"]
            with np.load(checked(stages[group], "payload/" + spec["path"], spec), allow_pickle=False) as grid:
                for m in methods:
                    values[m] = grid[FIELDS.get(m, "risk")]
        require(all(x.shape == mask.shape for x in values.values()), "Non-native score geometry")
        common = mask & np.logical_and.reduce([np.isfinite(x) for x in values.values()])
        yy = y[common]
        n = len(yy)
        positives = int(yy.sum())
        missing["empty_common"] += int(n == 0)
        missing["zero_positives"] += int(n > 0 and positives == 0)
        missing["all_positives"] += int(n > 0 and positives == n)
        for m in METHODS:
            tau = docs["evaluation"]["thresholds"][m]
            arr = values[m]
            finite = mask & np.isfinite(arr)
            ties[m] += int(np.count_nonzero(finite & (arr.astype(np.float64) == tau)))
            if not 0 < positives < n:
                undefined[m] += 1
                continue
            fresh = float(average_precision_score(yy, arr[common]))
            if m == "constant":
                require(abs(fresh - positives/n) < 1e-14, "Constant-score AP differs from prevalence")
            ap[f, c, m].append(fresh)
            defined[m] += 1
    final = {}
    summary = load(args.summary)["cohorts"][cohort]
    for block in summary["family_equal"]:
        if block["cohort"] != "all":
            continue
        condition = block["condition"]
        final[condition] = {}
        for m in METHODS:
            estimates = {f: fmean(ap[f, condition, m]) for f in families if ap[f, condition, m]}
            mean = fmean(estimates.values()) if estimates else None
            want = block["methods"][m]["seven_common_ap"]
            require(len(estimates) == want["defined_families"] and
                    ((mean is None and want["value"] is None) or
                     (mean is not None and want["value"] is not None and abs(mean - want["value"]) < 2e-14)),
                    "Independently aggregated family-equal AP differs")
            final[condition][m] = {"family_mean": mean, "defined_views_by_family": {f: len(ap[f, condition, m]) for f in sorted(families)}}
    write_json(args.out / "roster-readback.json", {"schema": "el-b-independent-roster-and-averaging-1", "run_id": os.environ["EL_RUN_ID"],
               "cohort": cohort, "method_sha256": q["method_sha256"], "report_summary_sha256": args.summary_sha256,
               "registered_family_ids": sorted(families), "registered_targets": len(expected), "cell_count": len(cells),
               "defined_view_aps": defined, "undefined_view_aps": undefined, "undefined_reasons": dict(missing),
               "exact_rgb_threshold_ties": label_ties, "exact_score_threshold_ties": dict(ties), "seven_common_ap": final})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "shelf", "request", "catalog", "historical-catalog", "summary", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    verify(parser.parse_args())
