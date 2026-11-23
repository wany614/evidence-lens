"""Build a private P2 B0 readout from a pinned seven-run chain and its original ZIPs."""

import argparse
import csv
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote

from evidence_lens._bundle import LABEL, RUN_ID, load, sha
from evidence_lens._verify import require
from select_trial import KINDS as STAGES, RULE, attempt, chain_ids, scientific_gates, transfer_evidence, verified_trial


VIEWS = tuple(f"v{i:02d}" for i in range(1, 7))
METHODS = ("L0", "1-alpha")


def relative(path, output):
    return Path(os.path.relpath(Path(path).resolve(), Path(output).resolve())).as_posix()


def link(label, path):
    return f"[{label}]({quote(path, safe='/.-_')})"


def selected(args, store):
    source = args.selection
    if args.trial:
        require(LABEL.fullmatch(args.trial), "invalid trial ID")
        source = store / "trials" / args.trial / "trial_selection.json"
    record = None
    if source:
        record = load(source)
        require(record.get("schema") == "el-p2-trial-selection-1" and record.get("selection_status") == "selected"
                and record.get("rule_version") == RULE,
                "require a finalized el-p2-trial-selection-1")
        if args.trial:
            require(record["trial_id"] == args.trial, "selection belongs to a different trial")
        entries = record["selected"]
        require(isinstance(entries, dict) and set(entries) == set(STAGES), "selection must pin all seven stages")
        ids = {role: entry["run_id"] for role, entry in entries.items()}
    else:
        ids = {}
        for item in args.run:
            role, sep, rid = item.partition("=")
            require(sep and role in STAGES and role not in ids, "use each --run role=ID once")
            ids[role] = rid
    require(set(ids) == set(STAGES) and len(set(ids.values())) == len(STAGES), "seven distinct stage IDs are required")
    require(all(isinstance(rid, str) and RUN_ID.fullmatch(rid) for rid in ids.values()), "invalid run identity")
    trial = load(store / "runs" / ids["evaluation"] / "run.json")["trial_id"]
    inventory = [r["run_id"] for r in record["attempts"]] if record else list(ids.values())
    require(set(ids.values()) <= set(inventory) and len(set(inventory)) == len(inventory), "invalid attempt inventory")
    records = verified_trial(store, trial, inventory)
    require(chain_ids(records, ids["recompute"]) == ids, "selected stages do not form one complete P2 chain")
    experiment = record["experiment_id"] if record else records[ids["generation"]]["run"]["experiment_id"]
    evidence = scientific_gates(store, records, ids, transfer_evidence(store, records), experiment)
    if record:
        require(record["trial_id"] == trial and record["attempts"] == [attempt(records[r]) for r in inventory]
                and record["selected"] == {k: attempt(records[r]) for k, r in ids.items()}, "changed selection or attempts")
        for k in ("camera_set_sha256", "scene_sha256", "input_sha256", "board_sha256", "independent_view_check_sha256"):
            require(evidence[k] == record["evidence"][k], f"selection evidence changed for {k}")
    return ids, {k: records[v] for k, v in ids.items()}, records, record, source


def container_info(rid, verified, bundle, directories, output):
    found = [d / f"{rid}.zip" for d in directories if (d / f"{rid}.zip").is_file()]
    if not found:
        return {"bytes": None, "missing_reason": "no_selected_run_zip_in_supplied_directories", "paths": []}
    expected = {r["path"]: (r["bytes"], r["sha256"]) for r in verified["manifest"]["files"]}
    for name in ("manifest.json", "completion.json"):
        expected[name] = ((bundle / name).stat().st_size, sha(bundle / name))
    path = found[0]
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(expected) and set(names) == set(expected), f"{rid}: ZIP membership differs from seal")
        for item in archive.infolist():
            size, digest = expected[item.filename]
            with archive.open(item) as member:
                require(item.file_size == size and hashlib.file_digest(member, "sha256").hexdigest() == digest,
                        f"{rid}: changed ZIP member {item.filename}")
    digest = sha(path)
    for other in found[1:]:
        require(sha(other) == digest, f"conflicting ZIPs for {rid}")
    return {"bytes": path.stat().st_size, "sha256": digest, "paths": [relative(p, output) for p in found]}


def stage_info(role, checked, store, directories, out):
    run = checked["run"]
    rid = run["run_id"]
    bundle = store / "runs" / rid
    rows = list(checked["artifacts"].values())
    sizes = {key: 0 for key in ("payload", "embedded_inputs", "source", "run_evidence")}
    for file in checked["manifest"]["files"]:
        top = file["path"].split("/")[0]
        kind = {"payload": "payload", "inputs": "embedded_inputs", "source": "source"}.get(top, "run_evidence")
        sizes[kind] += file["bytes"]
    sizes["run_evidence"] += sum((bundle / name).stat().st_size for name in ("completion.json", "manifest.json"))
    telemetry = load(bundle / "telemetry_schema.json")
    gpu_process, driver_reasons, devices, samples = [], set(), {}, 0
    with (bundle / "telemetry.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            samples += 1
            if row["gpu_process_memory_bytes"]:
                gpu_process.append(int(row["gpu_process_memory_bytes"]))
            else:
                reason = json.loads(row["missing_reason"] or "{}").get("gpu_process_memory_bytes")
                if reason:
                    driver_reasons.add(reason)
            for gpu in json.loads(row["gpu_devices"] or "[]"):
                if gpu.get("memory.used") is not None:
                    key = gpu["uuid"]
                    devices[key] = max(devices.get(key, 0), gpu["memory.used"])
    result = {
        "run_id": rid, "kind": run["run_kind"], "status": run["execution_status"], "retry_of": run["retry_of_run_id"],
        "manifest_sha256": checked["manifest_sha256"], "config_sha256": run["config_sha256"],
        "bundle_path": relative(bundle, out), "elapsed_seconds": run["elapsed_seconds_monotonic"],
        "platform": load(bundle / "environment.json")["platform"], "sampled_peak_tree_rss_bytes": run.get("sampled_peak_tree_rss_bytes"),
        "telemetry_samples": samples, "interval_seconds": telemetry["interval_seconds"],
        "driver_process_gpu_peak_bytes": max(gpu_process, default=None), "driver_counter_missing_reasons": sorted(driver_reasons),
        "sampled_device_memory_mib_by_uuid": devices, "artifact_count": len(rows), "logical_bytes": sum(sizes.values()),
        "bytes_by_role": sizes, "archive": container_info(rid, checked, bundle, directories, out),
        "artifacts": {r["path"]: {"bytes": r["bytes"], "sha256": r["sha256"]} for r in rows},
    }
    if role in ("generation", "render"):
        name = "generation.json" if role == "generation" else "prediction.json"
        meta = load(bundle / "payload" / name)
        phases = meta.get("phases", {})
        result["worker_phases"] = phases
        result["worker_phase_status"] = "recorded" if phases else "provider_phase_measurements_unavailable"
        result["worker_runtime"] = meta.get("runtime", {})
        for memory in ("allocated", "reserved"):
            key = f"cuda_{memory}_peak_bytes"
            result[f"peak_cuda_{memory}_bytes"] = max((p[key] for p in phases.values() if p.get(key) is not None), default=None)
        if role == "generation":
            result["gaussian_count"] = meta["gaussian_count"]
            result["network_seconds"] = {k: meta.get(k) for k in ("preprocess_seconds", "forward_seconds", "unproject_seconds")}
        else:
            result["source_rgb_check"] = meta["source_rgb_check"]
            result["depth_validation"] = {k: meta["depth_validation"][k] for k in ("off_axis_plane_passed", "alpha_normalization_passed")}
    elif role == "score":
        meta = load(bundle / "payload/score.json")
        result["view_timings"] = {v: {k: val for k, val in entry.items() if k.endswith("_seconds")} for v, entry in meta["views"].items()}
    elif role == "reference":
        event_path = bundle / "payload/reference_events.jsonl"
        result["worker_phases"] = []
        if event_path.is_file():
            with event_path.open(encoding="utf8") as f:
                result["worker_phases"] = [r for line in f if (r := json.loads(line)).get("status") == "completed"]
        result["worker_phase_status"] = "recorded" if result["worker_phases"] else "provider_phase_measurements_unavailable"
    return result


def _metric_values(rows, evaluation_id):
    measures = {v: {} for v in VIEWS}
    for row in rows:
        if row["region_name"] != "all":
            continue
        v, key = row["camera_id"], (row["method_id"], row["metric_name"])
        require(row["run_id"] == evaluation_id and v in measures and key not in measures[v], "ambiguous sealed metric")
        a, value = row["numerator"], row["value"]
        measures[v][key] = (float(a) if a not in (None, "") else None, int(row["denominator"]),
                            float(value) if value not in (None, "") else None)
    return measures


def result_counts(path, evaluation_id, parquet=None):
    with path.open(encoding="utf-8", newline="") as f:
        measures = _metric_values(csv.DictReader(f), evaluation_id)
    if parquet is not None:
        import pyarrow.parquet as pq
        cols = ["run_id", "camera_id", "region_name", "method_id", "metric_name", "numerator", "denominator", "value"]
        table = pq.read_table(parquet, columns=cols, filters=[("region_name", "=", "all")])
        require(_metric_values(table.to_pylist(), evaluation_id) == measures, "evaluation payload CSV and Parquet metrics disagree")
    per = {}
    for v, metrics in measures.items():
        def counts(method, metric):
            n, d, _ = metrics[method, metric]
            if n is None and d == 0:
                return 0, 0
            require(n is not None and n.is_integer() and 0 <= n <= d, f"{v}: invalid count for {metric}")
            return int(n), d
        e, valid = counts("output", "prediction_valid_fraction")
        wrong, e2 = counts("output", "rgb_error_fraction_gt_0.05")
        bad, v2 = counts("output", "invalid_output_fraction")
        require(e == e2 and v2 == valid and bad == valid - e, f"{v}: inconsistent RGB masks")
        mae_sum, mae_den, _ = metrics["output", "rgb_mae"]
        if mae_sum is None and e == 0:
            mae_sum = 0.0  # Empty sum for pooling. The mean remains undefined.
        require(mae_den == e and mae_sum is not None, f"{v}: unavailable RGB error sum")
        per[v] = {"V": valid, "E": e, "Y": wrong, "invalid_output": bad, "sum_rgb_mae": mae_sum,
                  "rgb_mae": mae_sum / e if e else None, "methods": {}}
        for method in METHODS:
            h, he = counts(method, "warning_fraction")
            l, lv = counts(method, "overall_low_risk_coverage")
            u, ue = counts(method, "unknown_fraction")
            hit, positives = counts(method, "high_risk_recall")
            miss, l2 = counts(method, "low_risk_error_rate")
            uy, u2 = counts(method, "error_fraction_within_unknown")
            require(he == ue == e and lv == valid and (l, u, wrong) == (l2, u2, positives)
                    and h + l + u == e and hit + miss + uy == wrong, f"{v}: inconsistent {method} denominators")
            per[v]["methods"][method] = {"H": h, "L": l, "U": u, "Y_in_H": hit, "Y_in_L": miss, "Y_in_U": uy,
                                         "common_ap": metrics[method, "average_precision_common"][2]}
    totals = {k: sum(v[k] for v in per.values()) for k in ("V", "E", "Y", "invalid_output", "sum_rgb_mae")}
    totals["rgb_mae"] = totals["sum_rgb_mae"] / totals["E"] if totals["E"] else None
    totals["methods"] = {m: {k: sum(v["methods"][m][k] for v in per.values()) for k in ("H", "L", "U", "Y_in_H", "Y_in_L", "Y_in_U")} for m in METHODS}
    return {"by_view": per, "family_sum": totals}


def budget(stages, checked, store):
    gen = checked["generation"]
    resolved = load(store / "runs" / gen["run"]["run_id"] / "config.resolved.yaml")
    cfg = resolved.get("generator", {})
    metadata = load(store / "runs" / gen["run"]["run_id"] / "payload/generation.json")
    shared = {r["sha256"]: r["bytes"] for r in gen["shared"]}
    digest = metadata["checkpoint_sha256"]
    require(digest in shared and cfg.get("checkpoint_sha256", digest) == digest
            and cfg.get("checkpoint_bytes", shared[digest]) == shared[digest], "checkpoint size/hash differs from shared input")
    ply_bytes = stages["generation"]["artifacts"]["payload/scene.ply"]["bytes"]
    rows = {}
    for role, count in (("reference", 24), ("readback", 24), ("generation", 28), ("q1_derivation", 4),
                        ("render", 32), ("score", 32), ("evaluation", 32), ("recompute", 32)):
        derivation = role == "q1_derivation"
        stage = stages["generation" if derivation else role]
        extra = ply_bytes if derivation else 0
        zip_bytes = stage["archive"]["bytes"]
        rows[role] = {"count": count, "unit_logical_bytes": stage["logical_bytes"] + extra,
                      "unit_zip_bytes": None if zip_bytes is None else zip_bytes + extra,
                      "basis": "unmeasured Q1 proxy using B0 generation bundle plus one equal-sized embedded input PLY"
                               if derivation else "selected B0 exemplar"}
        for unit in ("logical_bytes", "zip_bytes"):
            size = rows[role]["unit_" + unit]
            estimate = None if size is None else count * size
            rows[role]["projected_" + unit] = estimate
            rows[role]["projected_" + unit + "_with_margin"] = None if estimate is None else (3 * estimate + 1) // 2
    totals = {}
    for unit in ("logical_bytes", "zip_bytes"):
        missing = [role for role, r in rows.items() if r["unit_" + unit] is None]
        known = sum(r["unit_" + unit] * r["count"] for r in rows.values() if r["unit_" + unit] is not None) + shared[digest]
        totals[unit] = {"known_subtotal_bytes_including_shared": known, "missing_stages": missing,
                        "all_stages_estimate_bytes": None if missing else known,
                        "with_50pct_margin_bytes": None if missing else (3 * known + 1) // 2}
    return {"conditions": 32, "families": 24, "measured_families": 1, "shared_checkpoint": {"bytes": shared[digest], "sha256": digest},
            "stages": rows, "totals": totals, "margin_fraction": 0.5,
            "exemplar_recorded_storage_limits": resolved.get("recording", {}).get("storage_limits")}


def attempts_info(records, store):
    result = []
    for rid, c in sorted(records.items()):
        run = c["run"]
        with (store / "runs" / rid / "events.jsonl").open(encoding="utf8") as f:
            launched = any(json.loads(line)["event"] == "process_started" for line in f)
        result.append({"run_id": rid, "kind": run["run_kind"], "status": run["execution_status"],
                       "manifest_sha256": c["manifest_sha256"], "worker_started": launched,
                       "retry_of_run_id": run.get("retry_of_run_id"), "failure_reason": run.get("failure_reason")})
    return result


def receipt_times(store, stages, output):
    ids = {s["run_id"]: k for k, s in stages.items()}
    receipts = {k: [] for k in stages}
    for kind in ("transfer", "archive"):
        for path in sorted((store / "receipts" / kind).glob("*.json")):
            r = load(path)
            role = ids.get(r.get("run_id"))
            if role and r.get("manifest_sha256") == stages[role]["manifest_sha256"]:
                receipts[role].append({"path": relative(path, output), "sha256": sha(path), "kind": kind, "status": r["status"],
                                       "elapsed_seconds": r.get("elapsed_seconds"), "bytes": r.get("bytes"),
                                       "container_sha256": r.get("container_sha256")})
    return receipts


def fraction(n, d):
    return f"{n:,}/{d:,} ({n / d:.2%})" if d else f"{n:,}/{d:,} (undefined)"


def gib(n):
    if n is None:
        return "unavailable"
    return "<0.001" if 0 < n < 2**30 / 1000 else f"{n / 2**30:.3f}"


def markdown(report):
    data = report["metrics"]["family_sum"]
    l = [f"# P2 engineering trial {report['trial_id']}", "",
         f"One B0 family across six planned target cameras. RGB MAE is {data['sum_rgb_mae'] / data['E']:.5f} on E." if data["E"] else "One B0 family across six cameras. RGB MAE is undefined because E is empty.",
         f"RGB error >0.05 covers {fraction(data['Y'], data['E'])}. Output failure covers {fraction(data['invalid_output'], data['V'])}.", "",
         "## Evidence", "", " · ".join(link(k, v) for k, v in report["links"].items()), "",
         "## Pixel counts", "", "V counts reference-valid RGB. E additionally requires valid predicted RGB. H, L and U partition E into high, low and unknown risk.", "",
         "| Camera | V | E | RGB errors | L0 H | L0 L | L0 U | Errors in U |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for view, row in [*report["metrics"]["by_view"].items(), ("All six", data)]:
        r = row["methods"]["L0"]
        l.append(f"| {view} | {row['V']:,} | {row['E']:,} | {row['Y']:,} | {r['H']:,} | {r['L']:,} | {r['U']:,} | {r['Y_in_U']:,} |")
    l += ["", "| Method | Recall | Warning precision | Low-risk error | Overall low-risk coverage | Unknown fraction |",
          "|---|---:|---:|---:|---:|---:|"]
    for name, r in data["methods"].items():
        ratios = ((r["Y_in_H"], data["Y"]), (r["Y_in_H"], r["H"]), (r["Y_in_L"], r["L"]),
                  (r["L"], data["V"]), (r["U"], data["E"]))
        l.append("| " + " | ".join([name, *(fraction(a, b) for a, b in ratios)]) + " |")
    l += ["", "## Selected stages", "", "| Stage and run | Elapsed s | Sampled tree RSS GiB | Output artifacts GiB | Entire sealed run GiB | Stored ZIP GiB |",
          "|---|---:|---:|---:|---:|---:|"]
    for k, s in report["stages"].items():
        a = s["archive"]
        z = link(gib(a["bytes"]), a["paths"][0]) if a["paths"] else "unavailable"
        l.append(f"| {k} {link(s['run_id'], s['bundle_path'] + '/run.json')} | {s['elapsed_seconds']:.3f} | {gib(s['sampled_peak_tree_rss_bytes'])} | {gib(s['bytes_by_role']['payload'])} | {gib(s['logical_bytes'])} | {z} |")
    l += ["", "RSS is the sampled sum for the worker and its descendants. Shared pages can repeat and short peaks can fall between samples. Windows RSS uses the process working set. CUDA allocated and reserved peaks are framework counters. Driver process VRAM and device-wide readings have separate fields in the JSON.", ""]
    for k in ("generation", "render"):
        s = report["stages"][k]
        timings = ", ".join(f"{n} {v['seconds']:.3f} s" for n, v in s["worker_phases"].items() if v.get("seconds") is not None)
        l.append(f"- **{k}** on {s['worker_runtime'].get('gpu', 'an unreported GPU')}. {timings or 'Worker phase timings unavailable'}. Peak CUDA allocated/reserved memory was {gib(s['peak_cuda_allocated_bytes'])}/{gib(s['peak_cuda_reserved_bytes'])} GiB. Driver process VRAM peak was {gib(s['driver_process_gpu_peak_bytes'])} GiB.")
    timings = report["stages"]["score"]["view_timings"]
    if len(timings) == len(VIEWS) and all("features_seconds" in t and "scoring_seconds" in t for t in timings.values()):
        f_seconds = sum(t["features_seconds"] for t in timings.values())
        s_seconds = sum(t["scoring_seconds"] for t in timings.values())
        l.append(f"- The six targets used {f_seconds:.3f} s for feature extraction and {s_seconds:.3f} s for numerical scoring. The recorded stage elapsed time also includes loading, hashing and compression.")
    l += ["", "Packing and receiving times are recorded separately in the JSON under each stage receipt. Each receipt measures that local operation."]
    b = report["storage_projection"]
    l += ["", "## Storage for 32 planned conditions", "", "The projection reuses 24 references and counts 28 generator calls, four Q1 derivations and 32 seven-view renders. It assumes one independent reference readback per family and one fixed-view metric check per condition.", "",
          "| Stage | Copies | Unit logical GiB | Projected logical GiB | Projected ZIP GiB | Projected ZIP GiB with 50% margin |", "|---|---:|---:|---:|---:|---:|"]
    for k, r in b["stages"].items():
        l.append(f"| {k} | {r['count']} | {gib(r['unit_logical_bytes'])} | {gib(r['projected_logical_bytes'])} | {gib(r['projected_zip_bytes'])} | {gib(r['projected_zip_bytes_with_margin'])} |")
    l += ["", f"Count the shared installed checkpoint once per independent store ({gib(b['shared_checkpoint']['bytes'])} GiB). Q1 uses a storage proxy equal to the B0 generation bundle plus one additional PLY for the embedded B0 input.", ""]
    for label, totals in b["totals"].items():
        if totals["missing_stages"]:
            l.append(f"- The {label} subtotal is {gib(totals['known_subtotal_bytes_including_shared'])} GiB. The full estimate awaits {', '.join(totals['missing_stages'])}.")
        else:
            l.append(f"- The {label} estimate is {gib(totals['all_stages_estimate_bytes'])} GiB, or {gib(totals['with_50pct_margin_bytes'])} GiB with 50% working margin.")
    l += ["", "Logical and packed totals describe separate storage layouts. Coexisting originals, transfers and restoration copies need additional space. The estimate uses one B0 family. Re-estimate each stage with larger observed sizes after 3–5 representative development conditions. L1 training, alternate evaluations, failures, timing repeats and cloud caches have additional costs.", "",
          "## Selection and exceptions", "", report["history_scope"] + " The selection follows " + report["selection_rule"] + ".", "",
          f"{report['failed_started_stage_count']} launched stage attempts in this inventory ended unsuccessfully. {len(report['preflights'])} externally documented handoff preflights are listed separately.", ""]
    for a in report["attempts"]:
        desc = f" ({a['failure_reason']})" if a["failure_reason"] else ""
        l.append(f"- {a['run_id']} · {a['kind']} · {a['status']} · worker {'started' if a['worker_started'] else 'did not start'}{desc}")
    for excluded in report["earlier_ineligible_chains"]:
        l.append(f"- Earlier chain {excluded['recompute_run_id']} failed selection checks. {excluded['reason']}")
    if report["preflights"]:
        for p in report["preflights"]:
            l.append(f"- {link('Handoff preflight evidence', p['path'])} · {p['record']['status']} · {p['record']['reason']}. Recorded before scientific execution.")
    else:
        l.append("- External handoff preflight history was not supplied.")
    return "\n".join(l) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selection", type=Path)
    group.add_argument("--trial", help="read STORE/trials/TRIAL/trial_selection.json")
    group.add_argument("--run", action="append", help="explicit role=RUN_ID; repeat for all seven")
    parser.add_argument("--containers", action="append", required=True, type=Path, help="directory with original RUN_ID.zip containers")
    parser.add_argument("--preflight-evidence", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new private report directory")
    args = parser.parse_args(argv)
    store, out = args.store.resolve(), args.output.resolve()
    require(not out.exists() and not args.output.is_symlink() and not out.is_relative_to(store / "runs"), "choose a fresh output outside sealed runs")
    ids, checks, records, selection, source = selected(args, store)
    stage = {k: stage_info(k, checks[k], store, args.containers, out) for k in STAGES}
    path = {k: store / "runs" / rid / "payload" for k, rid in ids.items()}
    validation = load(path["recompute"] / "independent-view-check.json")
    require(validation["passed"] is True and validation["view"] == "v01", "fixed-view independent recomputation did not pass")
    files = {"Input PNG": path["reference"] / "input.png", "v01 reference array": path["reference"] / "reference/v01.npz",
             "v01 prediction array": path["render"] / "v01.npz",
             "v01 prediction": path["render"] / "v01.png", "Fixed board": path["evaluation"] / "board_v01.png",
             "RGB error": path["evaluation"] / "v01_rgb_error.png", "L0 state": path["evaluation"] / "v01_l0_state.png",
             "FP32 labels": path["evaluation"] / "labels/v01.npz", "Raw L0": path["score"] / "v01.npz",
             "Evaluation metrics CSV": path["evaluation"] / "metrics.csv", "Evaluation metrics Parquet": path["evaluation"] / "metrics.parquet",
             "Independent recomputation": path["recompute"] / "independent-view-check.json"}
    ref_png = path["reference"] / "reference/v01.png"
    if ref_png.is_file():
        files["v01 reference PNG"] = ref_png
    board_manifest = load(path["evaluation"] / "board_v01.json")
    board_fallback = []
    for label in ("RGB error", "L0 state"):
        name = files[label].name
        listed = board_manifest.get("native_preview_sha256", {})
        if name in listed:
            artifact = checks["evaluation"]["artifacts"].get("payload/" + name)
            require(artifact is not None and artifact["sha256"] == listed[name], f"board and sealed artifact hashes disagree for {name}")
        else:
            files[label] = files["Fixed board"]
            board_fallback.append(label)
    require(all(p.is_file() for p in files.values()), "required trial artifact is absent")
    preflights = []
    for p in args.preflight_evidence:
        record = load(p)
        require(record.get("scientific_attempt") is False and record.get("stage") == "handoff_preflight"
                and record.get("status") and record.get("reason"), "expected explicitly non-scientific handoff preflight evidence")
        fields = {k: record[k] for k in ("stage", "status", "reason", "scientific_attempt")}
        for key in ("failed_handoff", "replacement"):
            if key in record:
                fields[key] = {k: record[key][k] for k in ("name", "sha256", "bytes", "import_probe") if k in record[key]}
        preflights.append({"path": relative(p, out), "sha256": sha(p), "record": fields})
    meta = checks["evaluation"]["run"]
    history = attempts_info(records, store)
    report = {"schema": "el-p2-trial-report-1", "trial_id": meta["trial_id"], "scene_id": meta["scene_id"],
              "scene_family_id": meta["scene_family_id"], "condition_id": meta["condition_id"], "stages": stage,
              "selection_rule": str(selection["rule_version"]) if selection else "explicit supplied run IDs",
              "selection_file": {"path": relative(source, out), "sha256": sha(source)} if source else None,
              "history_scope": "Attempts recorded in the pinned selection." if selection else "Only the seven supplied stage IDs are represented.",
              "earlier_ineligible_chains": selection.get("earlier_ineligible_chains", []) if selection else [],
              "attempts": history, "failed_started_stage_count": sum(a["worker_started"] and a["status"] != "succeeded" for a in history),
              "preflights": preflights, "stage_receipts": receipt_times(store, stage, out),
              "links": {k: relative(p, out) for k, p in files.items()}, "preview_links_using_board": board_fallback,
              "independent_view_check": validation,
              "evaluation_metric_sources": {name: checks["evaluation"]["artifacts"]["payload/" + name]["sha256"]
                                            for name in ("metrics.csv", "metrics.parquet")},
              "metrics": result_counts(path["evaluation"] / "metrics.csv", ids["evaluation"], path["evaluation"] / "metrics.parquet"),
              "storage_projection": budget(stage, checks, store)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".p2-report-", dir=out.parent) as tmp:
        dst = Path(tmp) / "report"
        dst.mkdir(mode=0o700)
        (dst / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
        (dst / "report.md").write_text(markdown(report), encoding="utf8")
        dst.rename(out)
    print(out / "report.md")


if __name__ == "__main__":
    main()
