"""Original C measurements for redistribution, projected from the authenticated evaluator outputs."""

import csv
from pathlib import Path

from evidence_lens.multisource import SCOPES, METHODS
from evidence_lens.scoring_io import write_json
from stage3_risk_sources import asset


def table(path, rows):
    if not rows:
        raise ValueError("empty registered public measurement table")
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)


def public_tables(out, doc, d, families, macros, recon, paired, shared):
    dst = out / "public-numerical"
    dst.mkdir()
    results = []
    strata = []
    curves = []
    for r in families:
        identity = {k: r[k] for k in ("family", "condition", "scope", "alias_of", "method", "threshold")}
        for region, item in r["regions"].items():
            vals = {**identity, "region": region, **item["counts"], **item["rates"]}
            for domain in ("own", "common"):
                vals[domain + "_ap"] = item["ranking"][domain]["ap"]
                vals[domain + "_defined_views"] = item["ranking"][domain]["defined_views"]
                vals[domain + "_scheduled_views"] = item["ranking"][domain]["planned_views"]
            if region == "all":
                results.append(vals)
            else:
                strata.append(vals)
        curve = r["risk_coverage"]
        for p in curve["points"]:
            curves.append({**identity, **p, "frozen_L": curve["frozen_L"], "frozen_LY": curve["frozen_LY"]})
    table(dst / "family-metrics.csv", results)
    table(dst / "region-metrics.csv", strata)
    table(dst / "risk-coverage.csv", curves)
    lookup = {(r["family"], r["condition"], r["view"]): r for r in d["views"]}
    paired_z = []
    for r in recon["views"]:
        f, v = r["family"], r["view"]
        flat = {"family": f, "view": v, "V": r["V"], "paired_E": r["paired_E"], "paired_Z": r["paired_Z"]}
        for c in ("M2", "M4"):
            x = lookup[f, c, v]
            dist = x.get("source_camera_distances") or {}
            flat.update({c + "_" + k: a for k, a in r[c].items()})
            flat.update(
                {
                    c + "_" + k: x.get(k)
                    for k in (
                        "Y",
                        "surface",
                        "depth_comparable",
                        "predicted_ed_valid",
                        "alpha_valid",
                        "alpha_at_least_half",
                        "alpha_mean",
                    )
                }
            )
            flat[c + "_nearest_source_id"] = dist.get("nearest_source_id")
            flat[c + "_nearest_source_distance_m"] = dist.get("nearest_distance_m")
            flat[c + "_status"] = x["status"]
        paired_z.append(flat)
    table(dst / "paired-reconstruction.csv", paired_z)
    table(
        dst / "source-ablation.csv",
        [
            {
                **{k: a[k] for k in ("family", "condition", "method", "from", "to", "alias_same_result")},
                **{side + "_" + k: v for side in ("left", "right", "difference") for k, v in a[side].items()},
            }
            for a in paired
        ],
    )
    table(
        dst / "matched-all-scope-ap.csv",
        [
            {
                **{
                    k: a[k]
                    for k in ("family", "condition", "view", "E", "common_all_scopes", "common_positives")
                },
                "scope": s,
                "method": m,
                "ap": a["AP"][s][m],
            }
            for a in shared
            for s in SCOPES
            for m in METHODS
        ],
    )
    rows = []
    for c in d["cells"]:
        p = c.get("cost") or {}
        row = {
            "family": c["family"],
            "condition": c["condition"],
            "status": "failed" if c.get("failure") else "selected",
            "failure_stage": (c.get("failure") or {}).get("stage"),
            "failure_status": (c.get("failure") or {}).get("status"),
        }
        for n in (
            "gaussian_optimization_steps",
            "initial_gaussian_count",
            "final_gaussian_count",
            "sampled_source_pixels",
            "optimization_gpu_synchronized_seconds",
            "reconstruction_worker_seconds",
            "render_worker_seconds",
            "reconstruction_recorder_wall_seconds",
            "render_recorder_wall_seconds",
            "reconstruction_peak_tree_rss_bytes",
            "render_peak_tree_rss_bytes",
        ):
            row[n] = p.get(n)
        for s, x in p.get("source_sampling", {}).items():
            row[s + "_samples"] = x["samples"]
            row[s + "_pixels"] = x["pixels"]
        for r in ("reconstruction", "render"):
            runtime = p.get(r + "_runtime", {})
            row[r + "_cuda_allocated_peak_bytes"] = runtime.get("torch_cuda_allocated_peak_bytes")
            row[r + "_cuda_reserved_peak_bytes"] = runtime.get("torch_cuda_reserved_peak_bytes")
            row["gpu_device"] = runtime.get("device")
            row["gpu_total_device_bytes"] = runtime.get("gpu_total_bytes")
        rows.append(row)
    table(dst / "resources.csv", rows)
    summary = {
        "schema": "el-stage3-public-numerical-study-1",
        "split": doc["split"],
        "scientific_run_id": doc["run_id"],
        "catalog_sha256": doc["catalog_sha256"],
        "study_sha256": doc["study_sha256"],
        "dataset_sha256": doc["dataset_sha256"],
        "opening_sha256": doc["opening_sha256"],
        "method_sha256": doc["method_sha256"],
        "models": doc["models"],
        "families": doc["families"],
        "dispositions": doc["dispositions"],
        "family_equal_overall": [
            {k: r[k] for k in ("condition", "scope", "alias_of", "method")} | r["regions"]["all"]
            for r in macros
        ],
        "paired_reconstruction_by_family": recon["families"],
        "definitions": {
            "V": "reference RGB finite in the native frame",
            "E": "both original RGBs valid, a subset of V",
            "Y": "mean absolute display-sRGB difference over three channels exceeds float32 0.05",
            "S": "E with finite scores for all four methods at the same fixed source subset",
            "L": "E and finite own score strictly below its frozen threshold",
            "U": "E with no finite own score",
            "missed": "confirmed high-error in the low or unknown state",
            "depth": "positive optical-axis camera Z (metres), verified predicted expected-depth, alpha at least 0.5 and finite reference world Position",
            "depth_high": "absolute signed-Z error > max(0.02 m, 0.02 * reference Z)",
            "camera_distance": "Euclidean camera-centre separation in metres",
            "rates": "Pool native pixels within family and reconstruction. Macro rates give equal weight to families with defined rates.",
            "AP": "Original score ties stay together. Views with both classes define AP. Equal mean of defined view AP per family/condition; then equal mean of defined families.",
            "ablation": "M2 two/all use identical bytes. Common-domain prevalence and coverage are reported per subset. matched-all-scope-ap fixes the very same pixels across every subset.",
            "resources": "Wall times cover the identified recorder or worker stage. CUDA allocator counters are distinct from device capacity. Missing measurements remain empty.",
        },
    }
    write_json(dst / "study-summary.json", summary)
    files = sorted(dst.iterdir())
    manifest = {
        "schema": "el-stage3-public-numerical-manifest-1",
        "source_run_id": doc["run_id"],
        "files": [asset(dst, p) for p in files],
    }
    write_json(dst / "manifest.json", manifest)
    return asset(out, dst / "manifest.json")
