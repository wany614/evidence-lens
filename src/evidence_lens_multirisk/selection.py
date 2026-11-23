"""Development-only model selection and the single operating threshold per checkpoint."""

from collections import defaultdict

import numpy as np

from evidence_lens.formal_metrics import rank, ranking_ap
from evidence_lens.l1_freeze import threshold_for
from evidence_lens_vision.contract import file_record, require
from .data import CONDITIONS
from .model import native_score


def hierarchical_ap(views, families):
    conditions = {}
    family_ap = {}
    for f in families:
        conditions[f] = {}
        for c in CONDITIONS:
            rows = [r for r in views if r["family"] == f and r["condition"] == c]
            ap = [r["ap"] for r in rows if r["ap"] is not None]
            conditions[f][c] = {
                "value": float(np.mean(ap)) if ap else None,
                "defined_views": len(ap),
                "scheduled_views": len(rows),
            }
        values = [r["value"] for r in conditions[f].values() if r["value"] is not None]
        family_ap[f] = {
            "value": float(np.mean(values)) if values else None,
            "defined_conditions": len(values),
            "scheduled_conditions": 2,
        }
    defined = [r["value"] for r in family_ap.values() if r["value"] is not None]
    return {
        "value": float(np.mean(defined)) if defined else None,
        "defined_families": len(defined),
        "scheduled_families": len(families),
        "family": family_ap,
        "condition": conditions,
    }


def operating_point(lists, denominators, families):
    # Two slots per valid family keep families equally weighted even when one condition has V=0.
    scores, counts = {}, {}
    cells = []
    for f in families:
        valid = [c for c in CONDITIONS if denominators[f, c] > 0]
        for c in CONDITIONS:
            values = np.concatenate(lists[f, c]) if lists[f, c] else np.empty(0, np.float32)
            n = denominators[f, c]
            require(np.isfinite(values).all() and len(values) <= n, "operating numerator exceeds native V")
            cells.append(
                {"family": f, "condition": c, "V": n, "finite_evaluable": len(values), "defined": n > 0}
            )
            if n:
                for i in range(2 // len(valid)):
                    key = f + "/" + c + "/" + str(i)
                    scores[key] = values
                    counts[key] = n
    require(counts, "empty development reference-valid denominator for every family")
    selected = threshold_for(scores, counts)
    selected.pop("eligible_families")
    return {
        **selected,
        "domain": "development-score-all-M2-M4",
        "target_coverage": 0.8,
        "decision": "low iff finite score < tau",
        "objective": "equal families, then equal defined acquisition conditions, pooled native low pixels / V",
        "cells": cells,
    }


def evaluate(model, dataset, pre, device, saved=None):
    require(dataset.split == "development", "checkpoint selection must use the development split")
    views = []
    rows = []
    lists = defaultdict(list)
    denominators = defaultdict(int)
    for row in dataset.rows:
        f, c, v = (row[k] for k in ("family", "condition", "view"))
        V, E, Y = dataset.dev_truth(row)
        denominators[f, c] += int(V.sum())
        meta = {
            "family": f,
            "condition": c,
            "view": v,
            "scope": "all",
            "source_ids": row["modes"]["all"]["source_ids"],
            "V": int(V.sum()),
            "E": int(E.sum()),
        }
        s = {
            "family": f,
            "condition": c,
            "view": v,
            "scope": "all",
            "source_ids": meta["source_ids"],
            "score": None,
            "status": row["status"],
            "reason": row.get("failure"),
        }
        if row["status"] == "ready":
            x, ok = dataset.deployment(row, "all")
            rgb = dataset.rgb(row)
            pred = native_score(model, rgb, x, ok, pre, device)
            common = np.load(dataset.file(row["selection_support"]), allow_pickle=False)
            require(
                common.dtype == np.bool_
                and common.shape == V.shape
                and not np.any(common & ~(E & pred["valid"])),
                "selection support is outside valid multi-source or RGB-evaluable native pixels",
            )
            value, _ = rank(pred["risk"], Y, common)
            meta.update(value)
            meta["ap"] = ranking_ap(value)
            meta["selection_support_sha256"] = row["selection_support"]["sha256"]
            if saved:
                lists[f, c].append(pred["risk"][E & pred["valid"]])
                path = saved / f / c / "all" / (v + ".npz")
                path.parent.mkdir(exist_ok=True, parents=True)
                np.savez_compressed(path, **pred)
                s.update(status="scored", path=path.relative_to(saved).as_posix())
        else:
            meta.update(ap=None, n=0, status=row["status"], reason=row.get("failure"))
        views.append(meta)
        rows.append(s)
    aggregate = hierarchical_ap(views, dataset.families)
    require(
        aggregate["value"] is not None,
        "no development target with both classes on the registered common support",
    )
    result = {"family_equal_ap": aggregate["value"], "macro": aggregate, "views": views}
    if saved:
        choice = operating_point(lists, denominators, dataset.families)
        tau = choice["tau"]
        for r in rows:
            if r["status"] == "scored":
                p = saved / r.pop("path")
                with np.load(p, allow_pickle=False) as z:
                    risk, valid = z["risk"], z["valid"]
                state = np.full(risk.shape, 255, np.uint8)
                state[valid] = (risk[valid].astype(np.float64) >= tau).astype(np.uint8)
                np.savez_compressed(p, risk=risk, valid=valid, state=state)
                r["score"] = file_record(p, saved)
        result.update(threshold=choice, score_rows=rows)
    return result
