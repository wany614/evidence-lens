"""Recover comparison populations and matched 20%-coverage errors from the sealed R2 tables."""

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean

from evidence_lens._bundle import sha
from evidence_lens.formal_inputs import require


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def audit(methods, matched, views, out, single_schema, multi_schema):
    conditions = {
        (r["object_id"], r["condition"], r["method"]): r for r in read(methods) if r["version"] == "new"
    }
    view_counts = {}
    for r in read(views):
        if r["version"] == "new" and r["anchor"] == "20%_V":
            key = (r["object_id"], r["condition"], r["method"], r["domain"])
            view_counts.setdefault(key, []).append(r)
    rows = []
    for r in read(matched):
        if r["version"] != "new" or r["anchor"] != "20%_V":
            continue
        key = (r["object_id"], r["condition"], r["method"])
        d = r["domain"]
        source = conditions[key]
        per_view = view_counts[(*key, d)]
        require(len({i["view"] for i in per_view}) == 6 == len(per_view), "incomplete scheduled views")
        count = lambda name: sum(int(row[name]) for row in per_view if row[name] != "")
        size, positives = count("eligible"), count("eligible_positive")
        require(
            count("accepted") == int(r["accepted"])
            and count("accepted_positive") == int(r["accepted_positive"]),
            "aggregate differs from the six views",
        )
        current = d == "within_new"
        if current:
            require(
                (size, positives) == (int(source["C"]), int(source["CY"])), "four-method population differs"
            )
        rows.append(
            {
                "object": key[0],
                "condition": key[1],
                "method": key[2],
                "domain": d,
                "V": count("V"),
                "candidate": size,
                "positive_candidates": positives,
                "pooled_prevalence": positives / size,
                "view_mean_prevalence": mean(
                    int(x["eligible_positive"]) / int(x["eligible"]) for x in per_view
                ),
                "common_AP": source["common_AP"] if current else "",
                "AP_matched_view_prevalence": source["common_matched_view_prevalence"] if current else "",
                "views_at_20pct": r["coverage_defined_views"],
                "accepted": count("accepted"),
                "accepted_errors": count("accepted_positive"),
                "matched_error": r["risk"],
            }
        )
    require(len(rows) == 32, "expected two objects, two conditions, four methods, two domains")
    for object_id, condition, _ in conditions:
        for domain in ("within_new", "paired_old_new"):
            same = [
                r
                for r in rows
                if (r["object"], r["condition"], r["domain"]) == (object_id, condition, domain)
            ]
            require(
                len({(i["V"], i["candidate"], i["positive_candidates"], i["accepted"]) for i in same}) == 1,
                "candidate labels or quota vary by method",
            )
    schemas = [json.loads(p.read_bytes()) for p in (single_schema, multi_schema)]
    cols = [len(s["columns"]) for s in schemas]
    require(cols == [26, 39], "published feature columns differ")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "comparison-domains.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema": "el-comparison-domain-audit-1",
        "run_id": os.environ["EL_RUN_ID"],
        "source_sha256": {
            name: sha(p)
            for name, p in zip(
                (
                    "method_conditions",
                    "matched_conditions",
                    "matched_views",
                    "B_feature_schema",
                    "MS_feature_schema",
                ),
                (methods, matched, views, single_schema, multi_schema),
            )
        },
        "feature_columns": {
            "B-L1-transfer": cols[0],
            "MS-L1": cols[1],
            "MS-frozen": cols[1],
            "MS-ft": cols[1],
        },
        "AP_domain": "within_new; paired_old_new AP was not measured",
        "counts": "one row per condition, scorer and candidate population, pooled over six independently ranked views",
        "table_sha256": sha(out / "comparison-domains.csv"),
    }
    (out / "domain-audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("methods", "matched", "views", "out", "single-schema", "multi-schema"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    audit(args.methods, args.matched, args.views, args.out, args.single_schema, args.multi_schema)
