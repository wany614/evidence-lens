"""Export the two fixed cohorts' aggregate numerical results and recorded costs for distribution."""

import csv
import json
import math
from collections import Counter
from statistics import mean

from evidence_lens.formal_inputs import compact_pin, require, signature
from evidence_lens.scoring_io import new_output, write_json
from evidence_lens.stage2_parity import CATALOG_SHA
from vision_confirmation import COHORTS, CONDITIONS, PRIOR_REPORT, authenticated, catalog_grid, successful
from vision_evaluate import RATES
from vision_sources import METHODS, find_run, member, signed
from vision_pipeline import ORIGINAL_CATALOG_SHA

RANK_DOMAINS = ("own", "five_common", "seven_common")


def table(path, rows):
    require(rows and all(set(r) == set(rows[0]) for r in rows), "empty or inconsistent public rows")
    with path.open("w", encoding="utf-8", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_output(run, evaluation, name):
    path, _ = member(run, "payload/" + name, evaluation["files"][name])
    return json.loads(path.read_bytes())


def flatten(cohort, grid, families, status, tau):
    counts = {}
    statuses = {}
    for v in status:
        key = v["family"], v["condition"]
        require(
            key in grid and v["view"] not in statuses.setdefault(key, {}), "unexpected/duplicated public view"
        )
        statuses[key][v["view"]] = v
    rows = []
    for family in families:
        key = family["family"], family["condition"]
        require(
            key in grid and key not in counts and set(family["methods"]) == set(METHODS),
            "wrong family metric set",
        )
        vv = statuses[key]
        require(set(vv) == {f"v{i:02}" for i in range(1, 7)}, "six scheduled targets required")
        counts[key] = len(vv)
        for m in METHODS:
            d = family["methods"][m]
            n = d["counts"]
            view_status = Counter(r["methods"][m]["status"] for r in vv.values())
            render_ok = sum(r["render"] == "completed" for r in vv.values())
            count = sum(r["V"] for r in vv.values())
            e = sum(r["E"] for r in vv.values())
            require(
                n["V"] == count
                and n["E"] == e
                and n["E"] == n["L"] + n["U"] + n["H"]
                and n["Y"] == n["HY"] + n["LY"] + n["UY"]
                and 0 <= n["S7"] <= n["S5"] <= n["E"] <= n["V"],
                "inconsistent denominators",
            )
            outcome = (
                "completed"
                if view_status["scored"] == 6
                else "partial"
                if view_status["scored"]
                else "render_failed"
                if not render_ok
                else "unsupported"
            )
            r = {
                "cohort": cohort,
                "scene_family_id": key[0],
                "condition_id": key[1],
                "method": m,
                "status": outcome,
                "tau": tau[m],
                "views_planned": 6,
                "views_rendered": render_ok,
                "views_scored": view_status["scored"],
                "views_unsupported": view_status["unsupported"],
                "views_render_failed": 6 - render_ok,
                "terminal_upstream_views": sum(
                    x["scientific_disposition"] == "terminal_upstream_failure" for x in vv.values()
                ),
                **n,
            }
            for domain in RANK_DOMAINS:
                a = d["ranking"][domain]
                r[domain + "_ap"] = a["ap"] if a else None
                r[domain + "_defined_views"] = a["defined_views"] if a else None
                r[domain + "_undefined_views"] = 6 - a["defined_views"] if a else None
                r[domain + "_status"] = (
                    ("defined" if a["ap"] is not None else "no_two_class_view")
                    if a
                    else "outside_method_domain"
                )
            r.update({name: d["rates"][name]["value"] for name in RATES})
            rows.append(r)
    require(set(counts) == set(grid), "a scheduled family-condition is absent")
    return rows


def resource(run):
    return {
        **compact_pin(run.pin),
        "run_kind": run.run["run_kind"],
        "status": run.run["execution_status"],
        "elapsed_seconds_monotonic": run.run.get("elapsed_seconds_monotonic"),
        "sampled_peak_tree_rss_bytes": run.run.get("sampled_peak_tree_rss_bytes"),
        "manifest_logical_bytes": sum(z["bytes"] for z in run.files.values()),
        "payload_logical_bytes": sum(p["bytes"] for n, p in run.files.items() if n.startswith("payload/")),
    }


def engineering_readback(run):
    """Publish bounded numerical facts and named dispositions from explicitly supplied records."""
    from evidence_lens._bundle import HASH, RUN_ID, load

    exp = run.run["experiment_id"]
    data = resource(run)
    data["exit_code"] = run.run.get("exit_code")
    data["scope"] = "engineering"
    data["description"] = "Recorded engineering activity"
    if exp == "stage2-backbone-verification":
        data.update(description="Official encoder verification", scope="model_verification")
        if (
            run.run["execution_status"] == "failed"
            and b"official Python source inventory differs" in run.open("stderr.log").read_bytes()
        ):
            data["finding"] = (
                "The initial official-source inventory check exited before the verification completed."
            )
        if run.run["execution_status"] == "succeeded" and "payload/base-verification.json" in run.files:
            original, spec = member(run, "payload/base-verification.json")
            actual = load(original)
            require(
                actual["schema"] == "el-vision-base-check-1"
                and actual["strict_equal_loaded_tensors"] is True,
                "different encoder recovery evidence",
            )
            data.update(
                receipt_sha256=spec["sha256"],
                strict_equal_loaded_tensors=True,
                checked_tensors=actual["tensors"],
                checked_parameters=actual["parameters"],
                max_abs_difference=actual["max_abs_difference"],
            )
    elif exp == "stage2-preopening-acquisition-cost":
        original, spec = member(run, "payload/acquisition.json")
        ledger = load(original)
        require(
            ledger["schema"] == "el-vision-preopening-acquisition-cost-1"
            and ledger["run_id"] == run.pin["run_id"],
            "different learning acquisition ledger",
        )
        expected = {
            ("v11_new", "train"): 12,
            ("v11_new", "development"): 4,
            ("original_learning", "train"): 16,
            ("original_learning", "development"): 4,
        }
        require(
            {(r["origin"], r["split"]): r["families"] for r in ledger["summary"]} == expected
            and len(ledger["summary"]) == 4
            and len(ledger["cells"]) == sum(expected.values()),
            "wrong preopening learning roster",
        )

        def numbers(row, names):
            out = {k: row[k] for k in names}
            require(
                all(
                    (val is None and "rss_bytes" in key)
                    or (type(val) in (float, int) and math.isfinite(val) and val >= 0)
                    for key, val in out.items()
                ),
                "learning worker measurement is not numeric",
            )
            return out

        fields = (
            "selected_unique_runs",
            "attempts_including_failures",
            "unsuccessful_attempts",
            "sum_attempt_wall_seconds",
            "sum_selected_wall_seconds",
            "selected_payload_logical_bytes",
            "max_sampled_tree_rss_bytes",
        )
        summaries = []
        for r in ledger["summary"]:
            require(r["prediction_views"] == 7 * r["families"], "different selected learning views")
            summaries.append(
                {
                    **{k: r[k] for k in ("origin", "split", "families", "prediction_views")},
                    "workers": {k: numbers(r["workers"][k], fields) for k in ("generation", "render")},
                }
            )
        public = []
        for cell in ledger["cells"]:
            require(
                cell["condition"] == "B0"
                and cell["verified_prediction_views"] == 7
                and (cell["origin"], cell["split"]) in expected
                and isinstance(cell["scene_family_id"], str)
                and cell["scene_family_id"].isidentifier()
                and HASH.fullmatch(cell["selection_sha256"]),
                "unexpected learning condition",
            )
            worker_rows = {}
            for role in ("generation", "render"):
                pin = compact_pin(cell["phases"][role]["pin"])
                rows = [a for a in cell["attempts"] if a["run_id"] == pin["run_id"] and a["selected"]]
                require(
                    len(rows) == 1
                    and rows[0]["kind"] == role
                    and rows[0]["manifest_sha256"] == pin["manifest_sha256"],
                    "missing original selected Windows worker",
                )
                worker_rows[role] = pin | numbers(
                    rows[0], ("wall_seconds", "tree_rss_bytes", "payload_logical_bytes")
                )
            public.append(
                {
                    "family": cell["scene_family_id"],
                    "origin": cell["origin"],
                    "split": cell["split"],
                    "selection_sha256": cell["selection_sha256"],
                    "workers": worker_rows,
                }
            )
        require(
            Counter((r["origin"], r["split"]) for r in public) == expected
            and len({r["family"] for r in public}) == len(public),
            "duplicate or missing learning family",
        )
        for group in summaries:
            cells = [
                c for c in ledger["cells"] if (c["origin"], c["split"]) == (group["origin"], group["split"])
            ]
            for role, stats in group["workers"].items():
                source_attempts = {a["run_id"]: a for c in cells for a in c["attempts"] if a["kind"] == role}
                measured = [
                    a["wall_seconds"] for a in source_attempts.values() if a["wall_seconds"] is not None
                ]
                require(
                    len(source_attempts) == stats["attempts_including_failures"]
                    and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in measured)
                    and math.isclose(
                        sum(measured), stats["sum_attempt_wall_seconds"], abs_tol=1e-9, rel_tol=1e-12
                    ),
                    "pre-opening attempt timing differs from original receipts",
                )
                stats["attempts_with_wall_reading"] = len(measured)
        data.update(
            description="Pre-opening B0 acquisition costs",
            scope="preopening_resources",
            receipt_sha256=spec["sha256"],
            learning_summary=summaries,
            learning_cells=public,
        )
    elif exp == "stage2-pilot-budget":
        doc, spec = member(run, "payload/budget.json")
        budget = load(doc)
        require(budget["decision"] in ("proceed", "pause", "stop"), "unrecognized pilot budget decision")
        data.update(
            description="Acquisition pilot and budget",
            scope="engineering_pilot",
            pilot_family_count=len(budget["pilot_families"]),
            decision=budget["decision"],
            receipt_sha256=spec["sha256"],
        )
    elif exp == "stage2-vision-pilot":
        doc, spec = member(run, "payload/training.json")
        tr = load(doc)
        require(
            tr["phase"] == "pilot" and type(tr["optimizer_steps"]) is int and tr["optimizer_steps"] > 0,
            "incorrect GPU engineering pilot",
        )
        data.update(
            description="GPU engineering pilot",
            scope="engineering_pilot",
            optimizer_steps=tr["optimizer_steps"],
            receipt_sha256=spec["sha256"],
        )
    elif exp == "stage2-dev-mac-cost-readback":
        doc, spec = member(run, "payload/readback.json")
        qa = load(doc)
        require(
            qa["schema"] == "el-stage2-mac-primary-cost-readback-1" and qa["run_id"] == run.run["run_id"],
            "incorrect CPU/CUDA readback",
        )
        data.update(
            description="Development CPU and CUDA numerical readback",
            scope="device_validation",
            native_targets=qa["native_targets"],
            warm_repeats=qa["warm_passes"],
            receipt_sha256=spec["sha256"],
        )
    elif exp == "stage2-engineering-incident-readback":
        doc, spec = member(run, "payload/engineering-events.json")
        events = load(doc)
        require(
            events["schema"] == "el-stage2-engineering-events-1" and events["run_id"] == run.run["run_id"],
            "incorrect engineering readback",
        )
        registered = {
            "windows_transfer_deferred": (
                "acquisition_transport",
                "resumed_same_identity",
                "A Windows disk limit deferred outgoing packaging for one development trial. The Mac subsequently received the original generation and render.",
                {"control", "observation", "resolution", "selection"},
            ),
            "isolated_mac_bytecode_cache": (
                "engineering_pilot",
                "invalid_source_snapshot",
                "The isolated Mac scorer exited successfully. Python bytecode files changed the source inventory and its recorder package failed verification.",
                {"orphan_manifest", "orphan_events", "orphan_snapshot"},
            ),
            "release_example_input_lineage": (
                "release_example",
                "rejected_then_repaired",
                "The public example scoring request failed its input lineage check before its scorer process started. A separate selection derivation supplied the verified input for a successful repeat.",
                {"reject_manifest", "reject_events"},
            ),
        }
        gathered = []
        for r in events["events"]:
            name = r["event"]
            require(name in registered, "unrecognized public engineering observation")
            scope, outcome, note, roles = registered[name]
            digests = r["secondary_source_sha256"]
            require(
                r["scope"] == scope
                and r["outcome"] == outcome
                and set(digests) == roles
                and all(isinstance(v, str) and HASH.fullmatch(v) for v in digests.values()),
                "different engineering observation identity",
            )
            row = {
                "event": name,
                "description": note,
                "scope": scope,
                "outcome": outcome,
                "secondary_source_sha256": digests,
                "elapsed_seconds": r["elapsed_seconds"],
            }
            if "run_id" in r:
                require(RUN_ID.fullmatch(r["run_id"]), "incorrect event ID")
                row["run_id"] = r["run_id"]
            if "linked_runs" in r:
                row["linked_runs"] = [compact_pin(j) for j in r["linked_runs"]]
            if name == "windows_transfer_deferred":
                row["scientific_generation_attempts"] = 1
            if name == "isolated_mac_bytecode_cache":
                row.update(
                    unplanned_bytecode_files=int(r["unplanned_bytecode_files"]),
                    child_exit_code=int(r["child_exit_code"]),
                )
            if name == "release_example_input_lineage":
                row["child_started"] = r["child_started"]
            gathered.append(row)
        require(
            len(gathered) == len(registered) and {r["event"] for r in gathered} == set(registered),
            "incomplete engineering observation inventory",
        )
        data.update(
            description="Readback of three engineering and release events",
            events=gathered,
            receipt_sha256=spec["sha256"],
        )
    return data


def acquisition_accounting(
    roots, engineering, acquisitions, attempts, cost_rows, *, allow_fixture_without_learning=False
):
    """Count distinct workers and actual selected view files across new learning and confirmation."""
    inputs = [r for r in engineering if r.get("scope") == "preopening_resources"]
    if not inputs:
        require(allow_fixture_without_learning, "recorded pre-opening acquisition resources are required")
        return None
    require(len(inputs) == 1, "duplicate learning acquisition ledger")
    learning = inputs[0]["learning_summary"]
    new = [r for r in learning if r["origin"] == "v11_new"]
    reused = [r for r in learning if r["origin"] == "original_learning"]
    held = [c for c in acquisitions if c["cohort"] == "new_confirmation"]
    require(
        len({(c["family"], c["condition"]) for c in held}) == len(held),
        "duplicate new confirmation family and condition",
    )
    chosen = {}
    actual_views = 0
    target_names = ("source", *(f"v{i:02}" for i in range(1, 7)))
    for c in held:
        if c["status"] != "selected":
            continue
        for role in ("generation", "prediction"):
            if role in c["roles"]:
                info = cost_rows[c["roles"][role]["run_id"]]
                if info["run_kind"] in ("generation", "render"):
                    chosen[info["run_id"]] = info
        pred = find_run(roots, c["roles"]["prediction"])
        p, spec = member(pred, "payload/prediction.json")
        doc = signed(p, spec["sha256"])
        for v in target_names:
            claim = doc["views"].get(v, {})
            if not claim.get("sha256") and "bytes" not in claim:
                continue
            require(
                type(claim.get("bytes")) is int
                and claim["bytes"] > 0
                and isinstance(claim.get("sha256"), str),
                "selected view must have original byte count and SHA-256",
            )
            name = "payload/" + v + ".npz"
            item = pred.files.get(name)
            require(
                item and item["sha256"] == claim["sha256"] and item["bytes"] == claim["bytes"],
                "selected native view receipt differs",
            )
            member(pred, name, {"bytes": claim["bytes"], "sha256": claim["sha256"]})
            actual_views += 1
    workers = {}
    for stage in ("generation", "render"):
        seen = {
            a["run_id"]: cost_rows[a["run_id"]]
            for a in attempts
            if a["cohort"] == "new_confirmation" and a["stage"] == stage
        }
        selected = {rid: r for rid, r in chosen.items() if r["run_kind"] == stage}
        seen.update(selected)
        times = [
            r["elapsed_seconds_monotonic"]
            for r in seen.values()
            if r["elapsed_seconds_monotonic"] is not None
        ]
        require(
            all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in times),
            "confirmation attempt timing is invalid",
        )
        pre_total = sum(z["workers"][stage]["attempts_including_failures"] for z in new)
        pre_measured = sum(z["workers"][stage]["attempts_with_wall_reading"] for z in new)
        total = pre_total + len(seen)
        measured = pre_measured + len(times)
        workers[stage] = {
            "preopening_attempts_with_wall_reading": pre_measured,
            "new_acquisition_attempts_with_wall_reading": measured,
            "new_acquisition_attempts_missing_wall_reading": total - measured,
            "preopening_selected_runs": sum(z["workers"][stage]["selected_unique_runs"] for z in new),
            "preopening_attempts": pre_total,
            "confirmation_selected_runs": len(selected),
            "confirmation_attempts": len(seen),
            "confirmation_unsuccessful_attempts": sum(r["status"] != "succeeded" for r in seen.values()),
            "confirmation_attempts_with_wall_reading": len(times),
            "new_acquisition_attempt_wall_seconds": sum(
                z["workers"][stage]["sum_attempt_wall_seconds"] for z in new
            )
            + sum(times),
            "new_acquisition_attempts": total,
        }
    return {
        "preopening_new_B0": sum(x["families"] for x in new),
        "preopening_reused_B0": sum(x["families"] for x in reused),
        "confirmation_scheduled_conditions": len(held),
        "new_acquisition_conditions": sum(x["families"] for x in new) + len(held),
        "selected_native_prediction_views": sum(x["prediction_views"] for x in new) + actual_views,
        "workers": workers,
        "preopening_evidence_sha256": inputs[0]["receipt_sha256"],
        "wall_time_scope": "Sum of available recorder wall readings for distinct attempts. A missing reading contributes no elapsed duration to this sum.",
    }


def scientific_text(path, summary, paired, cpu, training, attempts, engineering, acquisition=None):
    """A standalone reading copy beside the portable numeric inventory."""

    def fmt(v, pct=False):
        suffix = ""
        if isinstance(v, dict):
            defined, total = v.get("defined_families"), v.get("total_families")
            if defined is not None and total is not None and defined < total:
                suffix = f" ({defined}/{total} families)"
            v = v["value"]
        if v is None:
            return "N/A" + suffix
        return (f"{v * 100:.2f}%" if pct else f"{v:.4f}") + suffix

    def fv(v):
        return "N/A" if v is None else f"{v:.3e}"

    rows = [
        "# Stage B confirmation results",
        "",
        f"Recorded report `{summary['run_id']}`. Method freeze `{summary['method_sha256']}`. The primary visual models used the preregistered seed 1729.",
        "",
        "A reference pixel is valid (V) when all three display-RGB channels are finite and within [0, 1]. The evaluable set E is the subset of V whose predicted RGB is also valid. An evaluable pixel is a confirmed error when its channel-mean absolute RGB difference strictly exceeds 0.05.",
        "",
        "Low risk requires a finite method score strictly below its frozen threshold inside E. Scores at or above that threshold are high. Nonfinite scores inside E are unknown. V outside E contributes to the invalid-output count. Low coverage is low pixels divided by V. Error in low is confirmed errors among the low pixels.",
        "",
        "Seven-common AP ranks E pixels with finite scores from all seven methods. Equal scores remain together. AP averages target views containing both classes within each family. Operating rates pool the six target counts within each family. Each mean gives equal weight to families with a defined value. Cells with missing family estimates show the contributing count over the number scheduled.",
        "",
    ]
    for cohort, title in zip(COHORTS, ("New eight-family confirmation", "Historical four-family regression")):
        section = summary["cohorts"][cohort]
        macro = {r["condition"]: r["methods"] for r in section["family_equal"] if r["cohort"] == "all"}
        rows += [
            "## " + title,
            "",
            f"{len(section['families'])} families, {section['planned_family_conditions']} family–condition pairs and {section['planned_views']} planned target views. Render dispositions were "
            + ", ".join(f"{v} {k}" for k, v in sorted(section["render_status"].items()))
            + ".",
            "",
            "| Condition | Method | Own AP | Five-common AP | Seven-common AP | Families with seven-common AP | Seven-common coverage | High error on seven-common | Low coverage | Error in low |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for c in CONDITIONS:
            for m in METHODS:
                d = macro[c][m]
                rows.append(
                    "| "
                    + " | ".join(
                        [
                            c,
                            m,
                            fmt(d["own_ap"]),
                            fmt(d["five_common_ap"]),
                            fmt(d["seven_common_ap"]),
                            str(d["seven_common_ap"]["defined_families"]),
                            fmt(d["seven_common_coverage"], True),
                            fmt(d["seven_common_prevalence"], True),
                            fmt(d["low_coverage"], True),
                            fmt(d["low_risk_error"], True),
                        ]
                    )
                    + " |"
                )
        rows += [
            "",
            "| Changed input | Mean paired target RGB absolute change | Largest channel change | Families with paired RGB |",
            "| --- | ---: | ---: | ---: |",
        ]
        for c in ("R1", "Q1"):
            members = [
                r["paired"]["rgb"]
                for r in section["paired_numerical_response"]["families"]
                if r["condition"] == c
            ]
            values = [
                m["mean_absolute_difference"] for m in members if m["mean_absolute_difference"] is not None
            ]
            maximum = max(
                (m["max_absolute_difference"] for m in members if m["max_absolute_difference"] is not None),
                default=None,
            )
            rows.append(
                f"| {c} versus B0 | {fv(mean(values) if values else None)} | {fv(maximum)} | {len(values)} |"
            )
        rows += [
            "",
            "| Condition | Mean paired fine-tuned AP minus frozen encoder AP | Mean paired fine-tuned AP minus L1-full AP |",
            "| --- | ---: | ---: |",
        ]
        for c in CONDITIONS:
            pairs = [
                r
                for r in paired
                if r["cohort"] == cohort
                and r["condition_id"] == c
                and r["scope"] == "family_equal"
                and r["comparison"] == "same_condition"
                and r["metric"] == "ap"
                and r["left"] == "L2-vision-ft"
            ]
            result = {r["right"]: r["difference"] for r in pairs}
            rows.append(f"| {c} | {fmt(result.get('L2-frozen'))} | {fmt(result.get('L1-full'))} |")
        rows.append("")
    rows += [
        "## Development selection and resource measurements",
        "",
        f"The development rule selected **{summary['default_method']}** as the deployment default before opening the new families. The eight candidate fits contributed {sum(r['optimizer_steps'] for r in training):,} recorded optimizer steps and {sum(r['wall_seconds'] for r in training) / 3600:.2f} worker-hours.",
        "",
        "| Seed-A model | Cold process to first output | Warm p50 | Warm p95 | Peak process-tree RSS |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for m, c in cpu.items():
        rows.append(
            f"| {m} | {c['cold_seconds']:.2f} s | {c['hot_p50_seconds']:.2f} s | {c['hot_p95_seconds']:.2f} s | {c['rss_bytes'] / 2**30:.2f} GiB |"
        )
    rows.extend(
        [
            "",
            "Cold latency spans process creation, interpreter and library imports, model loading, source and prediction decoding, feature extraction, inference and the first complete output file. Each warm percentile uses 18 completed outputs across six views.",
            "",
            "The memory column uses the larger of two sampled process-tree RSS peaks. Both peaks and both cold clock readings are recorded in the numerical cost file. The two sampling intervals are shown below.",
            "",
            "| Model | External observer | Native scorer |",
            "| --- | ---: | ---: |",
        ]
    )
    for m, c in cpu.items():
        rows.append(
            f"| {m} | {1000 * c['external_sample_interval_seconds']:.1f} ms | "
            f"{1000 * c['native_rss_sample_interval_seconds']:.1f} ms |"
        )
    failed = len({r["run_id"] for r in attempts if r["status"] != "succeeded"})
    rows += [
        "",
        f"The scientific selection histories list {failed} distinct unsuccessful attempts. The cost inventory records measured recovery and engineering entries separately. CUDA allocated and reserved bytes describe the PyTorch allocator. The CPU memory figure is sampled process-tree RSS.",
        "",
        "These results describe the registered static opaque scenes and nearby calibrated views. Risk scores rank RGB error under that protocol. The eight new families include two preregistered groups of four. Complete group estimates and native family risk–coverage points are available in the study summary.",
        "",
        "[Study summary and group results](public-numerical/study-summary.json) · [Per-family denominators](public-numerical/family-metrics.csv) · [Matched comparisons](public-numerical/paired-changes.csv) · [Measured costs](public-numerical/execution-costs.json) · [Development decision](public-numerical/selected-method.json) · [File hashes](public-numerical/manifest.json)",
        "",
    ]
    if acquisition:
        g, r = (acquisition["workers"][k] for k in ("generation", "render"))
        rows.extend(
            [
                "## New acquisition and reused records",
                "",
                f"The {acquisition['new_acquisition_conditions']} new scheduled conditions comprised {acquisition['preopening_new_B0']} pre-opening B0 conditions and {acquisition['confirmation_scheduled_conditions']} confirmation conditions. The learning datasets also reused {acquisition['preopening_reused_B0']} earlier B0 acquisitions. Selected new renders supplied {acquisition['selected_native_prediction_views']} source and target view files.",
                "",
                f"The Windows ledger records {g['new_acquisition_attempts']} distinct generation attempts and {r['new_acquisition_attempts']} distinct render attempts. Generation wall times sum to {g['new_acquisition_attempt_wall_seconds'] / 60:.2f} minutes across {g['new_acquisition_attempts_with_wall_reading']}/{g['new_acquisition_attempts']} timed attempts. Render wall times sum to {r['new_acquisition_attempt_wall_seconds'] / 60:.2f} minutes across {r['new_acquisition_attempts_with_wall_reading']}/{r['new_acquisition_attempts']} timed attempts. Missing clock readings contribute no duration to the sum. A shared generation counts once across condition references.",
                "",
            ]
        )
    if engineering:
        rows.extend(
            [
                "## Engineering and release evidence",
                "",
                "These records describe engineering or release work outside the two scientific cohorts.",
                "",
                "| Activity | Recorder outcome | Wall time |",
                "| --- | --- | ---: |",
            ]
        )
        for r in engineering:
            elapsed = r["elapsed_seconds_monotonic"]
            rows.append(
                f"| {r['description']} | {r['status']} | {elapsed:.2f} s |"
                if elapsed is not None
                else f"| {r['description']} | {r['status']} | N/A |"
            )
        rows.append("")
        for r in engineering:
            if r.get("finding"):
                rows.append(r["finding"])
            if "pilot_family_count" in r:
                rows.append(
                    f"The {r['pilot_family_count']}-family acquisition pilot recorded a decision to {r['decision']}."
                )
            if "optimizer_steps" in r:
                rows.append(
                    f"The GPU engineering pilot completed {r['optimizer_steps']} optimizer steps. The formal fit roster contains {len(training)} separate runs."
                )
            for ev in r.get("events", []):
                rows.append(ev["description"])
        rows.append("")
    path.write_text("\n".join(rows), encoding="utf-8")


def publish_report(roots, spec, new, old, output, run_id):
    require(
        isinstance(run_id, str) and run_id and set(spec["cohorts"]) == set(COHORTS),
        "report requires both predeclared case series",
    )
    marker, method, mp, method_sha = authenticated(roots, new, spec["catalog_sha256"])
    require(
        compact_pin(spec["method"]) == mp and spec["method_sha256"] == method_sha,
        "aggregate uses another method freeze",
    )
    tau = {m: d["tau"] for m, d in method["thresholds"].items()}
    grids = {
        COHORTS[0]: catalog_grid(signed(new, spec["catalog_sha256"])),
        COHORTS[1]: catalog_grid(signed(old, ORIGINAL_CATALOG_SHA)),
    }
    require(
        len(grids[COHORTS[0]]) == 24
        and len(grids[COHORTS[1]]) == 12
        and not ({f for f, c in grids[COHORTS[0]]} & {f for f, c in grids[COHORTS[1]]}),
        "overlapping or incomplete cohorts",
    )
    meta = {}
    family_rows = []
    paired_rows = []
    costs = {}
    private = {}
    attempts = []
    acquisitions = []

    def cost(pin):
        if pin["run_id"] not in costs:
            run = find_run(roots, compact_pin(pin))
            costs[pin["run_id"]] = resource(run)
        return compact_pin(pin)

    for cohort, roles in spec["cohorts"].items():
        ev = find_run(roots, compact_pin(roles["evaluation"]))
        ep, es = successful(roots, roles["evaluation"], "evaluation.json")
        e = signed(ep, es)
        expected = "el-vision-comparison-1" if cohort == COHORTS[0] else "el-vision-historical-regression-1"
        require(
            e["schema"] == expected
            and e["thresholds"] == tau
            and e["test_opening_sha256"] == marker
            and e["methods"] == list(METHODS),
            "wrong scientific evaluation",
        )
        mat, msha = successful(roots, roles["materials"], "materials.json")
        materials = signed(mat, msha)
        require(
            materials["test_unlock_sha256"] == marker
            and len(materials["cells"]) == len(grids[cohort])
            and {(c["family"], c["condition"]) for c in materials["cells"]} == set(grids[cohort]),
            "materials refer to another opening or cohort",
        )
        board, bs = successful(roots, roles["boards"], "boards.json")
        bd = signed(board, bs)
        require(
            bd["cohort"] == cohort
            and bd["method_sha256"] == method_sha
            and bd["count"] == len(grids[cohort])
            and bd["dataset_sha256"] == e["dataset_sha256"]
            and {(c["family"], c["condition"], c["view"]) for c in bd["cases"]}
            == {(f, c, "v01") for f, c in grids[cohort]},
            "private fixed case registry incomplete",
        )
        for tag, pin in roles["scores"].items():
            sp, ss = successful(roots, pin, "scores.json")
            s = signed(sp, ss)
            name = "L1-full" if tag == "original-five" else tag
            require(
                s["method_freeze_sha256"] == method_sha
                and e["inputs"][tag]
                == {"sha256": ss, "model_sha256": method["models"][name]["model_sha256"]},
                "evaluation inputs differ from original model and score",
            )
            cost(pin)
        fam = read_output(ev, e, "families.json")
        status = read_output(ev, e, "status.json")
        gs = grids[cohort]
        rows = flatten(cohort, gs, fam, status, tau)
        family_rows += rows
        pr = read_output(ev, e, "paired.json")
        for kind, raws in pr.items():
            for x in raws:
                per = kind == "family_differences"
                paired_rows.append(
                    {
                        "cohort": cohort,
                        "scope": "within_family" if per else "family_equal",
                        "scene_family_id": x["family"] if per else None,
                        "condition_id": x["condition"],
                        "left": x["left"],
                        "right": x["right"],
                        "comparison": x["comparison"],
                        "metric": x["metric"],
                        "left_value": x.get("left_value"),
                        "right_value": x.get("right_value"),
                        "difference": x["difference"] if per else x["mean_paired_difference"],
                        "defined_families": int(x["difference"] is not None)
                        if per
                        else len(x["defined_families"]),
                        "scheduled_families": 1 if per else x["scheduled_families"],
                    }
                )
        macro = read_output(ev, e, "macro.json")
        require(
            {r["condition"] for r in macro if r["cohort"] == "all"} == set(CONDITIONS),
            "missing family-equal estimate",
        )
        risk_curves = {}
        for fr in fam:
            key = fr["family"] + "/" + fr["condition"]
            curve = read_output(ev, e, key + "/curves.json")
            require(
                set(curve) == set(METHODS)
                and all(
                    curve[m]["frozen_L"] == fr["methods"][m]["counts"]["L"]
                    and curve[m]["frozen_LY"] == fr["methods"][m]["counts"]["LY"]
                    and curve[m]["frozen_tau"] == tau[m]
                    for m in METHODS
                ),
                "curve endpoint differs from frozen operating state",
            )
            risk_curves[key] = curve
        br = find_run(roots, compact_pin(roles["boards"]))
        rsp, proof = member(br, "payload/paired-response.json", bd["paired_response"])
        sensitivity = signed(rsp, proof["sha256"])
        require(
            sensitivity["cohort"] == cohort
            and sensitivity["method_sha256"] == method_sha
            and sensitivity["dataset_sha256"] == e["dataset_sha256"],
            "paired source evidence differs",
        )
        meta[cohort] = {
            "families": sorted({f for f, c in gs}),
            "conditions": list(CONDITIONS),
            "planned_family_conditions": len(gs),
            "planned_views": len(gs) * 6,
            "scientific_view_dispositions": dict(Counter(r["scientific_disposition"] for r in status)),
            "render_status": dict(Counter(r["render"] for r in status)),
            "family_equal": macro,
            "family_risk_coverage": risk_curves,
            "paired_numerical_response": sensitivity,
            "evaluation": {**cost(roles["evaluation"]), "sha256": es},
            "materials": {**cost(roles["materials"]), "sha256": msha},
        }
        for cell in materials["cells"]:
            q = signed(
                roots[0] / "trials" / cell["trial_id"] / "trial_selection.json", cell["selection_sha256"]
            )
            chosen_ids = {r["run_id"] for r in q.get("selected", {}).values()}
            acquisitions.append(
                {
                    "cohort": cohort,
                    "family": cell["family"],
                    "condition": cell["condition"],
                    "selection_sha256": cell["selection_sha256"],
                    "status": q["selection_status"],
                    "roles": {role: cost(pin) for role, pin in cell["parents"].items()},
                }
            )
            for a in q.get("attempts", []):
                ar = find_run(roots, compact_pin(a))
                require(
                    ar.run["execution_status"] == a["execution_status"], "condition attempt receipt differs"
                )
                attempts.append(
                    {
                        "cohort": cohort,
                        "family": cell["family"],
                        "condition": cell["condition"],
                        **cost(a),
                        "status": ar.run["execution_status"],
                        "stage": ar.run["run_kind"],
                        "selected": a["run_id"] in chosen_ids,
                        "retry_of_run_id": ar.run.get("retry_of_run_id"),
                        "requested_retry_of_run_id": ar.run.get("requested_retry_of_run_id"),
                        "exit_code": ar.run.get("exit_code"),
                    }
                )
            if q["selection_status"] == "failed" and not q.get("attempts"):
                pin = compact_pin(q["failure"])
                ar = find_run(roots, pin)
                attempts.append(
                    {
                        "cohort": cohort,
                        "family": cell["family"],
                        "condition": cell["condition"],
                        **cost(pin),
                        "status": ar.run["execution_status"],
                        "stage": ar.run["run_kind"],
                        "selected": True,
                        "retry_of_run_id": ar.run.get("retry_of_run_id"),
                        "requested_retry_of_run_id": ar.run.get("requested_retry_of_run_id"),
                        "exit_code": ar.run.get("exit_code"),
                    }
                )
        for role in ("dataset", "boards"):
            cost(roles[role])
        private[cohort] = {**compact_pin(roles["boards"]), "boards_sha256": bs}
    training = []
    for t in method["training_receipts"]:
        p, sha = successful(roots, t, "training.json")
        require(sha == t["training_json_sha256"], "training source differs from sealed model")
        tr = signed(p, sha)
        training.append(
            {
                **cost(t),
                "model_sha256": t["model_sha256"],
                "selected_primary": t["run_id"] == method["models"][t["method"]]["run_id"],
                "method": t["method"],
                **{
                    k: tr[k]
                    for k in (
                        "seed",
                        "head_lr",
                        "backbone_lr",
                        "epochs",
                        "optimizer_steps",
                        "best_epoch",
                        "best_development_family_equal_ap",
                        "wall_seconds",
                        "peak_cuda_allocated",
                        "peak_cuda_reserved",
                    )
                },
            }
        )
    engineering = []
    for pin in spec.get("engineering", []):
        trusted = cost(pin)
        engineering.append(engineering_readback(find_run(roots, trusted)))
    fixture_optional = spec.get("fixture_allow_missing_preopening", False)
    require(
        type(fixture_optional) is bool and (not fixture_optional or spec["catalog_sha256"] != CATALOG_SHA),
        "the formal catalog requires recorded pre-opening acquisition resources",
    )
    acquisition = acquisition_accounting(
        roots, engineering, acquisitions, attempts, costs, allow_fixture_without_learning=fixture_optional
    )
    for upstream in method["models"].values():
        cost(upstream)
    for c in method["cpu_costs"].values():
        cost(c["pin"])
    mr = find_run(roots, mp)
    dev, dr = member(mr, "payload/development-replay.json", method["development_replay_sha256"])
    replay = json.loads(dev.read_bytes())
    public_docs = (
        "study-summary.json",
        "family-metrics.csv",
        "paired-changes.csv",
        "execution-costs.json",
        "selected-method.json",
    )
    with new_output(output) as out:
        pr = out / "public-numerical"
        pr.mkdir()
        summary = {
            "schema": "el-vision-public-study-1",
            "run_id": run_id,
            "method_freeze": mp | {"sha256": method_sha},
            "method_sha256": method_sha,
            "catalog_sha256": spec["catalog_sha256"],
            "study_sha256": method["study_sha256"],
            "test_opening_sha256": marker,
            "methods": list(METHODS),
            "cohorts": meta,
            "units": "Families are independent units. AP averages two-class target views within each family. Operating rates use per-family pooled counts. Each mean gives equal weight to families with a defined value.",
            "rgb_event": "E is a subset of reference-valid V with valid predicted RGB. The event is channel-mean absolute display-RGB difference strictly greater than 0.05 in E.",
            "operating_rule": "Low is E with a finite method score strictly below its frozen threshold. Finite scores at or above the threshold are high.",
            "rank_domains": {
                "own": "finite method scores in E",
                "five_common": "five original methods finite in E, reported for the five originals",
                "seven_common": "all seven finite in E",
            },
            "unknown": "Nonfinite method scores inside E are unknown. V minus E counts invalid outputs.",
            "display_rule": "v01",
            "private_boards": private,
            "default_method": method["deployment_rule"]["default"],
            "historical_scope": "Previously reported four families scored using the frozen Stage B packages. This series does not enter the new eight-family confirmation.",
            "original_report": PRIOR_REPORT,
        }
        write_json(pr / "study-summary.json", summary)
        table(pr / "family-metrics.csv", family_rows)
        table(pr / "paired-changes.csv", paired_rows)
        cpu = {
            k: {a: v for a, v in c.items() if a != "canonical_win_to_mac"}
            for k, c in method["cpu_costs"].items()
        }
        write_json(
            pr / "execution-costs.json",
            {
                "schema": "el-vision-public-costs-1",
                "model_training": training,
                "acquisitions": acquisitions,
                "new_acquisition_budget": acquisition,
                "scientific_selection_attempts": attempts,
                "independently_pinned_engineering": engineering,
                "native_primary_cpu": cpu,
                "observed_runs": sorted(costs.values(), key=lambda d: d["run_id"]),
                "memory_units": "Bytes. CUDA allocated and reserved describe the PyTorch allocator. Process RSS is a separate operating-system measurement.",
                "storage_units": "Logical bytes. Each parent shared across conditions appears once. Hard links and archive compression affect occupied disk space.",
            },
        )
        write_json(
            pr / "selected-method.json",
            {
                "schema": "el-vision-public-method-1",
                "method_sha256": method_sha,
                "default_method": method["deployment_rule"]["default"],
                "deployment_rule": method["deployment_rule"],
                "thresholds": method["thresholds"],
                "models": {
                    m: {**compact_pin(src), "model_sha256": src["model_sha256"]}
                    for m, src in method["models"].items()
                },
                "development_replay_sha256": dr["sha256"],
                "baseline_development_qa": method["baseline_development_qa"],
                "development_seed_comparisons": replay["paired"],
                "development_primary": {
                    m: {k: v["macro"] if v else None for k, v in z.items()}
                    for m, z in replay["primary_domains"].items()
                },
                "inference_parity_protocol_sha256": method["inference_parity_protocol_sha256"],
                "native_inference_parity": method["native_inference_parity"],
                "development_support": {
                    "all_three_paired_seed_views": replay["matched_48_three_seed_domains"],
                    "candidate_fits": len(replay["candidates"]),
                },
                "parameter_scope": {
                    m: next(
                        t["trainable_parameters"]
                        for t in method["training_receipts"]
                        if t["run_id"] == method["models"][m]["run_id"]
                    )
                    for m in ("L2-frozen", "L2-vision-ft")
                },
                "deployment_parity": {k: v["canonical_win_to_mac"] for k, v in method["cpu_costs"].items()},
            },
        )
        scientific_text(
            out / "study.md", summary, paired_rows, cpu, training, attempts, engineering, acquisition
        )
        write_json(
            pr / "manifest.json",
            {
                "schema": "el-vision-public-numerical-1",
                "report_run_id": run_id,
                "method_sha256": method_sha,
                "files": {n: signature(pr / n) for n in public_docs},
            },
        )
