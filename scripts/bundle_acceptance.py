"""Summarize explicit, sealed engineering checks and costs for the standard Bundle CLI."""

import argparse
import csv
import json
import os
from pathlib import Path

from bundle_original import check, put, sha


def read_selected(store, row):
    base = store / row["location"] / row["run_id"]
    manifest = (base / "manifest.json").read_bytes()
    check(sha(manifest) == row["manifest_sha256"], "unexpected recorded manifest")
    book = json.loads(manifest)
    idx = {v["path"]: v for v in book["files"]}
    check(book["run_id"] == row["run_id"], "unexpected run")

    def raw(n):
        data = (base / n).read_bytes()
        check(sha(data) == idx[n]["sha256"] and len(data) == idx[n]["bytes"], "changed recorded member: " + n)
        return data

    run = json.loads(raw("run.json"))
    check(run["execution_status"] == row["disposition"], "engineering result disposition differs")
    record = {
        "name": row["name"],
        "run_id": row["run_id"],
        "manifest_sha256": row["manifest_sha256"],
        "status": run["execution_status"],
        "stage_elapsed_seconds": run["elapsed_seconds_monotonic"],
        "sampled_peak_tree_rss_bytes": run.get("sampled_peak_tree_rss_bytes"),
        "sealed_payload_bytes": sum(v["bytes"] for name, v in idx.items() if name.startswith("payload/")),
        "sealed_file_logical_bytes": sum(v["bytes"] for v in idx.values()),
        "worker_exit_code": run.get("worker_exit_code"),
    }
    doc = None
    if row["receipt"]:
        data = raw("payload/" + row["receipt"])
        doc = json.loads(data)
        record.update(receipt="payload/" + row["receipt"], receipt_sha256=sha(data))
    return record, doc


def aggregate(a):
    selected = json.loads(a.spec.read_bytes())
    check(selected["schema"] == "el-bundle-acceptance-roster-1", "wrong roster")
    rows = []
    receipts = {}
    for item in selected["attempts"]:
        r, receipt = read_selected(a.store, item)
        rows.append(r)
        if receipt:
            receipts[item["name"]] = receipt
    check(len(set(x["run_id"] for x in rows)) == len(rows), "duplicate attempt")
    original = receipts["original-grade-v2"]
    recovered = receipts["recovery-grade-v2"]
    ids = receipts["external-ID-grade"]
    built = receipts["build-v2"]
    outputs = receipts["cpu-v2"]
    portable = receipts["portable-v2"]
    browser = receipts["browser-v2"]
    bounds = receipts["faults-v2"]
    check(
        original["fresh_legacy_strict_passed"]
        and original["published_expected_state_and_tolerance_passed"]
        and recovered["status"] == "passed"
        and ids["passed"],
        "failed compatibility",
    )
    check(
        outputs["wheel_sha256"] == portable["wheel_sha256"] == built["wheel"]["sha256"]
        and outputs["html"]["html_sha256"] == browser["html_sha256"],
        "different accepted build or report",
    )
    check(
        len(bounds["rejections"]) == 10 and len(bounds["honest_unknown_outcomes"]) == 2,
        "incomplete boundary QA",
    )

    def finite(dataset, key=None):
        return sum((r[key] if key else r)["finite_together"] for r in dataset["rows"])

    actual = []
    for m, obs in sorted(outputs["results"].items()):
        actual.append(
            {
                "condition_method": m,
                "counts": obs["counts"],
                "process_seconds": obs["process_seconds"],
                "score_sha256": obs["score_sha256"],
            }
        )
    detail = {
        "schema": "el-bundle-tooling-acceptance-1",
        "run_id": os.getenv("EL_RUN_ID"),
        "published_model_version": "v0.3.0-rc.1",
        "tool_version": "0.4.0a2",
        "scope": "Mac CPU engineering compatibility; synthetic original and previously exposed private examples.",
        "wheel": built["wheel"],
        "base_distributions": built["installed_distributions"],
        "optional_environment": outputs["optional_environment"],
        "portable_offline_example": portable["archive"],
        "bundle_sha256": portable["bundle_sha256"],
        "html": outputs["html"],
        "original_compatibility": {
            "conditions": 2,
            "targets_per_condition": 6,
            "methods": len(original["methods"]),
            "observations": len(original["rows"]),
            "canvas_method_pixels": original["canvas_method_pixels"],
            "finite_compared": finite(original, "fresh_legacy"),
            "changed_finite_float32_words_vs_published": sum(
                r["published_expected"]["finite_float32_bits_differ"] for r in original["rows"]
            ),
            "changed_finite_masks_vs_published": sum(
                r["published_expected"]["finite_disagreements"] for r in original["rows"]
            ),
            "changed_states_vs_published": sum(
                r["published_expected"]["state_disagreements"] for r in original["rows"]
            ),
            "maximum_difference_vs_published": max(
                r["published_expected"]["max_absolute_difference"] for r in original["rows"]
            ),
            "all_words_including_nan_vs_fresh_archive_identical": all(
                r["fresh_legacy_all_float32_bits_equal"] for r in original["rows"]
            ),
        },
        "known_recovery_compatibility": {
            "object": "set6",
            "condition": "M4",
            "method_views": len(recovered["rows"]),
            "canvas_method_pixels": recovered["canvas"],
            "finite_compared": finite(recovered),
            "all_words_masks_states_identical": recovered["status"] == "passed",
        },
        "official_camera_id_check": {
            "object": "set1",
            "condition": "M4",
            "source_order": ids["source_order"],
            "primary_source_id": ids["primary_source_id"],
            "endpoint_ids": ids["common_endpoints"],
            "target_order": ids["target_order"],
            "methods": ["B-L1-transfer", "MS-L1"],
            "method_views": len(ids["rows"]),
            "finite_compared": finite(ids),
            "all_words_masks_states_identical": ids["passed"],
        },
        "private_target_reference_reads_in_tool_checks": 0,
        "actual_optional_and_base_commands": actual,
        "offline_browser": {
            "product": browser["product"],
            "image_count": len(browser["images"]),
            "network_offline": browser["offline"],
            "request_schemes": browser["request_schemes"],
            "selected_views": browser["views"],
        },
        "controlled_faults": bounds["rejections"],
        "declared_unknowns": bounds["honest_unknown_outcomes"],
        "engineering_history": [
            {
                "run_id": "20261124T011914536604Z-b4c7e1a9b069422f",
                "event": "Initial writable fault copy inherited sealed read-only file modes; no fault injected.",
            },
            {
                "run_id": "20261124T012136918860Z-fcf0a2091cff46c8",
                "event": "Recorder rejected a changed-code retry before worker launch.",
            },
            {
                "run_id": bounds["run_id"],
                "event": "New engineering revision authenticated the original bytes and used writable temporary copies.",
            },
        ],
        "recorded_attempts": rows,
        "measurement_note": "Elapsed seconds use the recorder's monotonic timer; RSS is sampled process-tree memory. Payload and sealed-file byte counts are logical recorded bytes. Dataset comparisons reuse already exposed or synthetic assets.",
    }
    put(a.out / "verification.json", detail)
    with (a.out / "stage-costs.csv").open("x", newline="") as f:
        fields = [
            "name",
            "run_id",
            "manifest_sha256",
            "status",
            "stage_elapsed_seconds",
            "sampled_peak_tree_rss_bytes",
            "sealed_payload_bytes",
            "sealed_file_logical_bytes",
            "worker_exit_code",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("spec", "store", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    aggregate(p.parse_args())
