"""Read the opened test runs directly and independently check the four-method RGB headline and scorer subsets."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np

from evidence_lens.formal_inputs import require
from evidence_lens.multisource import CONDITIONS, METHODS, SCOPES, TARGETS
from evidence_lens.scoring_io import new_output, write_json
from evidence_lens.stage3_gate import get_frozen_method, require_opening
from stage3_risk_input_qa import bits, independent_y
from stage3_risk_sources import CATALOG, STUDY
from vision_sources import find_run, json_member, member, signed


def independent_counts(v, e, y, risk, state, finite_common, tau):
    from sklearn.metrics import average_precision_score

    positives = y == 1
    eligible = e & np.isfinite(risk)
    below = eligible & (risk.astype(np.float64) < tau)
    above = eligible & ~below
    unknown = e & ~eligible
    checked = np.full(risk.shape, 255, np.uint8)
    checked[np.isfinite(risk)] = (risk[np.isfinite(risk)].astype(np.float64) >= tau).astype(np.uint8)
    bits(checked, state, "published threshold state")

    def n(z):
        return int(np.count_nonzero(z))

    masks = {
        "V": v,
        "E": e,
        "Y": e & positives,
        "invalid": v & ~e,
        "A": eligible,
        "H": above,
        "L": below,
        "U": unknown,
        "S": finite_common,
        "SY": finite_common & positives,
        "HY": above & positives,
        "LY": below & positives,
        "UY": unknown & positives,
        "missed": (below | unknown) & positives,
    }
    out = {k: n(z) for k, z in masks.items()}
    ap = {}
    for tag, use in (("own", eligible), ("common", finite_common)):
        lab = positives[use]
        ap[tag] = float(average_precision_score(lab, risk[use])) if lab.any() and (~lab).any() else None
    return out, ap


def same_metric(a, b):
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= 1e-12)


def grade(roots, request, opening, destination, rid):
    marker = require_opening(roots[0], CATALOG, opening_sha256=opening, shelf=roots[1])
    method = get_frozen_method(roots[0], marker, roots[1])
    require(
        request["schema"] == "el-stage3-confirmation-replay-request-1"
        and request["opening_sha256"] == opening
        and set(request["scores"]) == set(METHODS),
        "separate test replay needs the frozen inputs",
    )

    def read(spec):
        r = find_run(roots, spec)
        require(r.run["execution_status"] == "succeeded", "unsuccessful test replay parent")
        return r, json_member(r, spec["member"], spec["sha256"])[0]

    ds, d = read(request["dataset"])
    er, ev = read(request["evaluation"])
    require(
        ev["schema"] == "el-stage3-risk-evaluation-1"
        and ev["split"] == d["split"] == "test"
        and d["cohort"] == "formal"
        and d["families"] == ev["families"] == marker["test_families"]
        and d["catalog_sha256"] == ev["catalog_sha256"] == CATALOG
        and d["opening_sha256"] == ev["opening_sha256"] == opening
        and d["method_sha256"] == ev["method_sha256"] == marker["method_sha256"]
        and ev["dataset_sha256"] == request["dataset"]["sha256"],
        "wrong original test scientific grade",
    )

    def lookup(r):
        return r["family"], r["condition"], r["view"]

    needed = [(f, c, v) for f in d["families"] for c in CONDITIONS for v in TARGETS]
    data = {lookup(r): r for r in d["views"]}
    require(
        len(data) == len(d["views"]) == len(needed) and set(data) == set(needed),
        "data dropped or duplicated a target",
    )
    ds_cells = {(c["family"], c["condition"]): c for c in d["cells"]}
    original_keys = {
        n: json_member(er, "payload/" + spec["path"], spec)[0]
        for n, spec in ev["files"].items()
        if n in ("views", "families", "macros", "same-pixels-across-scopes")
    }
    official = {(*lookup(x), x["scope"]): x for x in original_keys["views"]}
    require(len(official) == len(needed) * len(SCOPES), "incomplete original per-view grade")
    records = {m: read(s) for m, s in request["scores"].items()}
    stored = {}
    for m, (sr, j) in records.items():
        require(
            j["schema"] == "el-stage3-risk-scores-1"
            and j["dataset_sha256"] == ev["dataset_sha256"]
            and j["method"] == m
            and request["scores"][m]["sha256"] == ev["models"][m]["sha256"]
            and all(
                j[k] == method["models"][m][k]
                for k in ("threshold", "weights_sha256", "model_id", "model_index_sha256")
            ),
            "a regrade score belongs to another model or domain",
        )
        stored[m] = {lookup(r): r for r in j["views"]}
        require(
            len(stored[m]) == len(j["views"]) == len(needed) and set(stored[m]) == set(needed),
            "missing native scorer entry",
        )
    readers = {}

    def sealed(pin):
        if pin["run_id"] not in readers:
            readers[pin["run_id"]] = find_run(roots, pin)
        r = readers[pin["run_id"]]
        require(r.pin["manifest_sha256"] == pin["manifest_sha256"], "changing original seal")
        return r

    obs = []
    keys = defaultdict(list)
    all_common = {lookup(a): a for a in original_keys["same-pixels-across-scopes"]}
    gaps = []
    for f, c, v in needed:
        rr = data[f, c, v]
        cell = ds_cells[f, c]
        ref = sealed(cell["selected"]["reference"])
        p, _ = member(ref, "payload/reference/" + v + ".npz", rr["reference_member_sha256"])
        with np.load(p, allow_pickle=False) as z:
            rgb_ref = z["rgb"]
        rgb_pred = None
        if rr["status"] == "ready":
            ren = sealed(cell["selected"]["render"])
            p, _ = member(ren, "payload/" + v + ".npz", rr["prediction_member_sha256"])
            with np.load(p, allow_pickle=False) as z:
                rgb_pred = z["rgb"]
        else:
            require(rr["status"] == "render_failed", "unregistered target status")
        valid, e, y = independent_y(rgb_ref, rgb_pred)
        require(not (e & ~valid).any(), "E outside V")
        per = {}
        shared = e.copy()
        for s in SCOPES:
            for m in METHODS:
                spec = stored[m][f, c, v]["modes"][s]
                if spec["file"] is None:
                    require(rgb_pred is None, "ready target missing a method score")
                    risk = np.full(e.shape, np.nan, np.float32)
                    state = np.full(e.shape, 255, np.uint8)
                else:
                    require(rgb_pred is not None, "failed target unexpectedly has risk")
                    run = records[m][0]
                    prefix = Path(request["scores"][m]["member"]).parent
                    p, _ = member(run, (prefix / spec["file"]["path"]).as_posix(), spec["file"])
                    with np.load(p, allow_pickle=False) as z:
                        risk = z["risk"]
                        state = z["state"]
                        bits(z["valid"], np.isfinite(risk), "finite score support")
                per[s, m] = (risk, state)
                shared &= np.isfinite(risk)
            if c == "M2" and s == "all":
                for m in METHODS:
                    a = stored[m][f, c, v]["modes"]
                    require(
                        a["all"]["file"] == a["two"]["file"],
                        "M2 duplicated rather than aliased its endpoint exposure",
                    )
        fixed = all_common[f, c, v]
        require(
            fixed["common_all_scopes"] == int(shared.sum())
            and fixed["common_positives"] == int((shared & (y == 1)).sum()),
            "matched-source common domain or prevalence changed",
        )
        for s in SCOPES:
            cm = e & np.logical_and.reduce([np.isfinite(per[s, m][0]) for m in METHODS])
            for m in METHODS:
                ris, states = per[s, m]
                tau = method["models"][m]["threshold"]
                counts, ap = independent_counts(valid, e, y, ris, states, cm, tau)
                base = official[f, c, v, s]["regions"]["all"][m]
                require(
                    all(base["counts"][k] == v0 for k, v0 in counts.items()), "native AP grade count mismatch"
                )
                for k in ap:
                    val = base["ranking"][k]["ap"]
                    require(same_metric(ap[k], val), "independent scikit-learn AP differs")
                    if val is not None:
                        gaps.append(abs(val - ap[k]))
                from sklearn.metrics import average_precision_score

                lab = y[shared] == 1
                ap_shared = (
                    float(average_precision_score(lab, ris[shared])) if lab.any() and (~lab).any() else None
                )
                require(
                    same_metric(ap_shared, fixed["AP"][s][m]), "fixed identical cross-scope pixels disagree"
                )
                z = {
                    "family": f,
                    "condition": c,
                    "view": v,
                    "scope": s,
                    "method": m,
                    "alias_of": "two" if (c, s) == ("M2", "all") else None,
                    "counts": counts,
                    "own_ap": ap["own"],
                    "common_ap": ap["common"],
                    "all_scope_ap": ap_shared,
                }
                obs.append(z)
                keys[f, c, s, m].append(z)
    ftable = {}
    original_f = {
        (r["family"], r["condition"], r["scope"], r["method"]): r["regions"]["all"]
        for r in original_keys["families"]
    }
    macro_fields = {
        "common_coverage_E": ("S", "E"),
        "common_prevalence": ("SY", "S"),
        "low_coverage": ("L", "V"),
        "low_error": ("LY", "L"),
        "missed_error_fraction": ("missed", "Y"),
        "unknown_E": ("U", "E"),
        "rgb_error_fraction": ("Y", "E"),
        "error_detection": ("HY", "Y"),
    }

    def mean(xs):
        a = [x for x in xs if x is not None]
        return sum(a) / len(a) if a else None

    for identity, views in keys.items():
        a = original_f[identity]
        n = {k: sum(r["counts"][k] for r in views) for k in views[0]["counts"]}
        require(all(n[k] == a["counts"][k] for k in n), "family native count differs")
        row = {k: n[i] / n[j] if n[j] else None for k, (i, j) in macro_fields.items()}
        require(all(same_metric(v, a["rates"][k]) for k, v in row.items()), "family system rate differs")
        for k in ("own", "common"):
            row[k + "_ap"] = mean([r[k + "_ap"] for r in views])
            require(same_metric(row[k + "_ap"], a["ranking"][k]["ap"]), "original family AP differs")
        ftable[identity] = row
    macro_rows = []
    for r in original_keys["macros"]:
        c, s, m = r["condition"], r["scope"], r["method"]
        cells = [ftable[f, c, s, m] for f in d["families"]]
        means = {k: mean([a[k] for a in cells]) for k in cells[0]}
        expect = r["regions"]["all"]
        require(
            all(
                same_metric(a, expect[k]["mean"])
                and sum(x[k] is not None for x in cells) == expect[k]["defined_families"]
                for k, a in means.items()
            ),
            "final family-equal mean or defined family count differs",
        )
        macro_rows.append({"condition": c, "scope": s, "method": m, "independently_replayed": means})
    import sklearn

    summary = {
        "schema": "el-stage3-raw-confirmation-replay-1",
        "run_id": rid,
        "catalog_sha256": CATALOG,
        "study_sha256": STUDY,
        "method_sha256": marker["method_sha256"],
        "opening_sha256": opening,
        "passed": True,
        "scikit_learn": sklearn.__version__,
        "original_evaluation": request["evaluation"],
        "dataset": request["dataset"],
        "scores": request["scores"],
        "planned_target_views": len(needed),
        "native_method_scope_rows": len(obs),
        "distinct_method_scope_rows": sum(r["alias_of"] is None for r in obs),
        "max_view_ap_difference": max(gaps) if gaps else None,
        "family_method_scope_count": len(ftable),
        "macro_method_scope_count": len(macro_rows),
        "aggregates": macro_rows,
        "per_view_records": obs,
    }
    with new_output(destination) as out:
        write_json(out / "pixel-replay.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("store", "shelf", "request", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--request-sha256", required=True)
    p.add_argument("--opening-sha256", required=True)
    a = p.parse_args()
    j = grade(
        (a.store, a.shelf),
        signed(a.request, a.request_sha256),
        a.opening_sha256,
        a.output,
        os.environ["EL_RUN_ID"],
    )
    print(
        json.dumps(
            {
                k: j[k]
                for k in (
                    "passed",
                    "planned_target_views",
                    "native_method_scope_rows",
                    "max_view_ap_difference",
                )
            }
        )
    )
