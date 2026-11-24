"""Seal a general-stage completion matrix from native evidence and the public files at its cutoff."""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from evidence_lens._bundle import load, save, sha, utc
from evidence_lens._verify import require


STATES = {"done", "done_negative", "failed", "blocked", "not_started", "out_of_scope"}
PUBLIC = "docs/general-closeout/"
HEAD_DOC = PUBLIC + "README.md"
PERF_DOC = PUBLIC + "performance/README.md"
LOG = PUBLIC + "research-log.md"
READY = PUBLIC + "specialization-readiness.md"
CARD = "docs/releases/v0.4.0/MODEL_CARD.md"
RIGHTS = "docs/releases/v0.4.0/RIGHTS.md"
FORMAT = "docs/tooling/assets/README.md"
BRIDGE = "docs/tooling/numerical-bridge.md"
T1 = "docs/results/t1/"
RELEASE = "v0.3.0-rc.1"

HIST = {
    "t0_opening": ("20261124T013603310781Z-d1ba70c0ba3c4034", "payload/opening.json"),
    "t0_evaluation": ("20261124T013936088026Z-3e1df3bb0a484836", "payload/public/evaluation.json"),
    "c6_calibration": ("20261124T021235608840Z-f0d12afab15c4b3e", "payload/calibration-results.json"),
    "c6_recount": ("20261124T021259461920Z-0f16697bc8ca4700", "payload/calibration-audit.json"),
    "format_parity": ("20261124T021118010901Z-e7bca336730e47df", "payload/parity.json"),
    "format_portable": ("20261124T021735523259Z-8ee25029bf9d4422", "payload/verification.json"),
    "format_recount": ("20261124T030422988018Z-770cf828f0f945dd", "payload/windows-cpu.json"),
}

CLOSED = (
    "g0-entry",
    "g0-platform-inventory",
    "g0-windows-intake",
    "t1-terminals",
    "t1-closed-matrix",
    "t1-opening",
    "t1-fixed-evaluation",
    "t1-fixed-recount",
    "t1-decisions-set118",
    "t1-decisions-set76",
    "t1-nbv-evaluation",
    "t1-nbv-independent",
    "g0-original-public-models",
    "g2-original-four-models-final",
    "g2-original-release-archive",
    "g2-legacy-control-final",
    "g2-t1-frozen-four-models",
    "g2-t1-selected8-preregistration",
    "g2-t1-selected8-supplement",
    "g4-original-trajectory-intake",
    "g4-public-original-trajectories",
    "t1-resources",
    "g6-complete-clean-handler-control",
)
ALIASES = {
    "mac_g3": "g3-official-mac",
    "mac_g4": "g4-official-mac",
    "windows_g3": "g3-windows-intake",
    "windows_g4": "g4-windows-intake",
    "tables": "g3-g4-final-measurement-tables",
    "build": "g6-source-build",
    "manifest": "g6-final-release-inventory",
    "download": "g6-public-anonymous-all",
    "mac_acceptance": "g6-mac-clean-replay",
    "windows_acceptance": "g6-windows-public-intake",
    "linux_acceptance": "g6-linux-clean-replay",
    "cloud_log": "g5-docs-020-live",
    "github": "g6-github-public",
    "verification": "g6-post-public-verification",
    "regressions": "g6-immutable-regressions",
    "windows_first_attempt": "g6-windows-native-tls-failure",
    "chronology": "g6-post-public-chronology",
}
STATIC = (
    (
        "g0_history_platforms",
        "done",
        ["g0-entry", "g0-platform-inventory", "g0-windows-intake"],
        [PUBLIC + "status.json"],
        "Windows is native x86-64. Linux is an aarch64 container on the Mac.",
    ),
    (
        "t0_confirmation",
        "done",
        ["t0_opening", "t0_evaluation"],
        ["docs/results/t0/confirmation/README.md", "docs/results/t0/confirmation/evaluation.json"],
        "All 288 registered earlier-cohort scores were available. The six physical subjects are distinct from T1.",
    ),
    (
        "c6_threshold_adaptation",
        "done_negative",
        ["c6_calibration", "c6_recount"],
        [
            "docs/results/batch-two/calibration/calibration-results.json",
            "docs/results/batch-two/calibration/calibration-audit.json",
        ],
        "Both registered operating points have null tau. Their finite support and exploratory ordering remain separate from the frozen threshold.",
    ),
    (
        "t1_fixed_risk_and_opening",
        "done",
        ["t1-terminals", "t1-closed-matrix", "t1-opening", "t1-fixed-evaluation", "t1-fixed-recount"],
        [T1 + "README.md", T1 + "fixed/origins.json"],
        "42 logical dispositions and 288 fixed score slots. Six M2 and four M4 subjects produced the 240 available method–view scores.",
    ),
    (
        "set118_set76_original_probe_and_downstream_m4",
        "failed",
        ["t1-decisions-set118", "t1-decisions-set76", "t1-closed-matrix"],
        [T1 + "fixed/sealed-matrix.json", T1 + "resources.md"],
        "M2 produced 16,251 and 17,199 Gaussians. The original 20,000-point probe condition prevented twelve M4 fits across the two subjects. Both had zero candidate photograph acquisitions.",
    ),
    (
        "t1_active_view_endpoint",
        "done",
        ["t1-nbv-evaluation", "t1-nbv-independent"],
        [T1 + "nbv.md", T1 + "independent-nbv.json"],
        "Four of six subjects paired. The risk-minus-uniform object-equal MAE difference was -0.0066173, with two improvements, one adverse outcome and one exact tie.",
    ),
    (
        "four_original_models",
        "done",
        ["g0-original-public-models", "g2-original-four-models-final", "g2-original-release-archive"],
        [CARD, "docs/tooling/original-archive.md"],
        "The optional neural parameters come from the original CPU archive. The default wheel embeds MS-L1.",
    ),
    (
        "resident_array_api_and_exact_cache",
        "done",
        ["g2-original-four-models-final", "g2-t1-frozen-four-models"],
        ["docs/tooling/resident.md", "tests/test_resident.py"],
        "The byte limit applies to retained cache payload. Each frame recomputes target-dependent features and scores.",
    ),
    (
        "four_thread_scientific_numerical_bridge",
        "done",
        ["g2-original-four-models-final", "g2-legacy-control-final", "g2-t1-frozen-four-models"],
        [PUBLIC + "t1-numerical-bridge.json", BRIDGE],
        "All four arrays matched for 240 available method–views on Mac at four neural threads and 64 MiB. The old archive contains three arrays. Frozen features and validity predicates recover the fourth reason channel.",
    ),
    (
        "selected_thread_numerical_bridge",
        "done",
        ["g2-t1-selected8-preregistration", "g2-t1-selected8-supplement"],
        [PUBLIC + "t1-threads8-preregistration.json", PUBLIC + "t1-threads8-numerical-bridge.json"],
        "At eight threads all three engineering paths matched 240/240. Four versus eight threads matched 204/240, or 219/240 when only MS-ft changed. Up to 1.1921e-7 absolute risk difference; all cutoff states and attainable quota memberships were identical.",
    ),
    (
        "original_static_renderer_trajectory",
        "done",
        ["g4-original-trajectory-intake", "g4-public-original-trajectories"],
        [
            PUBLIC + "performance/g4-trajectory-verification.json",
            PUBLIC + "performance/g4-upstream-frames.csv",
        ],
        "Four 120-pose trajectories and two revision sequences contain 548 original CUDA source and target renders. Scene content was static.",
    ),
    (
        "format_support",
        "done",
        ["format_parity", "format_portable", "format_recount"],
        [FORMAT, "docs/tooling/assets/render-parity.csv", "docs/results/batch-two/README.md"],
        "Historical COLMAP 3.10 text/binary, Gaussian PLY and native NPZ controls used the original Windows CUDA renderer. Prepared bundle scoring runs on CPU.",
    ),
    (
        "optional_metadata_and_specialization_contract",
        "done",
        ["g4-public-original-trajectories"],
        [READY, "src/evidence_lens_bundle/replay_sidecar.py"],
        "Capture time, covariance, dynamic-mask references and acquisition policy are recorded as declared. Effects of these optional fields on risk remain unevaluated.",
    ),
    (
        "repository_research_report_and_log",
        "done",
        ["t1-fixed-recount", "t1-nbv-independent", "g2-t1-selected8-supplement"],
        [HEAD_DOC, LOG],
        "The repository records the experimental results, Wany's decisions and the model and agent contributions. The companion-document verification has its own row.",
    ),
    (
        "rights_inventory",
        "done",
        ["g0-original-public-models", "g2-original-release-archive", "g4-public-original-trajectories"],
        [RIGHTS, "configs/public-source.json", "licenses/multisource-assets.json"],
        "The legal inventory distinguishes original scenes, source code, third-party parameters and restricted DTU imagery. Published artifacts receive separate checks.",
    ),
)


def named(store, name):
    f = store / "work/controller" / name / "selected.json"
    return load(f) if f.is_file() else None


def run_folder(store, rid):
    require(re.fullmatch(r"[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}", rid) is not None, "invalid run identity")
    for parent in (
        store / "runs",
        store / "history/runs",
        store.parent / "batch-one/runs",
        store.parent / "batch-two/runs",
    ):
        folder = parent / rid
        if (folder / "completion.json").is_file():
            return folder
    raise ValueError("sealed run unavailable: " + rid)


def checked(store, pin):
    require(set(pin) == {"run_id", "manifest_sha256", "member", "sha256"}, "incomplete evidence pin")
    folder = run_folder(store, pin["run_id"])
    manifest = load(folder / "manifest.json")
    c = load(folder / "completion.json")
    require(
        c["execution_status"] == "succeeded"
        and c["run_id"] == manifest["run_id"] == pin["run_id"]
        and c["manifest_sha256"] == sha(folder / "manifest.json") == pin["manifest_sha256"],
        "different native seal",
    )
    member = pin["member"]
    require(
        member.startswith("payload/") and not Path(member).is_absolute() and ".." not in Path(member).parts,
        "non-payload member",
    )
    records = [s for s in manifest["files"] if s["path"] == member]
    f = folder / member
    require(
        len(records) == 1
        and f.stat().st_size == records[0]["bytes"]
        and sha(f) == records[0]["sha256"] == pin["sha256"],
        "native member differs: " + member,
    )
    origin = load(folder / "run.json")
    return load(f), {
        **pin,
        "recorded_code_commit": origin["code_commit"],
        "recorded_source_snapshot_sha256": origin["code_snapshot_sha256"],
    }


def historical(store, rid, member):
    folder = run_folder(store, rid)
    return {
        "run_id": rid,
        "manifest_sha256": load(folder / "completion.json")["manifest_sha256"],
        "member": member,
        "sha256": sha(folder / member),
    }


def collect(store, project, phase, updates):
    allowed = {"pins", "public_evidence", "tag", "code_commit", "notes"}
    require(set(updates) <= allowed, "unknown matrix binding")
    require(set(updates.get("pins", {})) <= set(ALIASES), "unknown optional evidence role")
    pins = {k: named(store, k) for k in (*CLOSED, "g6-storage-" + phase)}
    require(all(pins.values()), "a fixed scientific or resource seal is absent")
    pins.update({k: historical(store, *v) for k, v in HIST.items()})
    if phase == "final":
        previous = named(store, "g5-docs-020-live")
        if previous:
            pins["cloud_previous"] = previous
    for role, alias in ALIASES.items():
        if role == "cloud_log" and phase == "final":
            alias = "g6-docs-final-live"
        override = updates.get("pins", {}).get(role, alias)
        p = named(store, override) if isinstance(override, str) else override
        if p and role in ("windows_g3", "windows_g4", "windows_acceptance"):
            record, _ = checked(store, p)
            if "native" in record:
                pins[role + "_intake"] = p
                p = record["native"]
        if p:
            pins[role] = p
    files = {p for _, _, _, xs, _ in STATIC for p in xs}
    files.update(
        {
            PERF_DOC,
            LOG,
            RIGHTS,
            READY,
            BRIDGE,
            T1 + "resources.md",
            PUBLIC + "resources/" + phase + "/resources.json",
            PUBLIC + "performance/tables.json",
            "docs/releases/v0.4.0/README.md",
        }
    )
    additional = {k: list(v) for k, v in updates.get("public_evidence", {}).items()}
    if phase == "final":
        projections = {
            "cloud_log": [PUBLIC + "research-log-readback-final.json"],
            "verification": ["docs/releases/v0.4.0/verification.json", PUBLIC + "release-verification.json"],
            "manifest": ["docs/releases/v0.4.0/release-manifest.json"],
            "tables": [PUBLIC + "performance/public-archive.json"],
        }
        for role, paths in projections.items():
            for path in paths:
                if role in pins and (project / path).is_file() and sha(project / path) == pins[role]["sha256"]:
                    additional[role] = list(dict.fromkeys([*additional.get(role, []), path]))
    require(set(additional) <= set(ALIASES), "unknown public evidence role")
    for paths in additional.values():
        files.update(paths)
    for key in files:
        path = Path(key)
        require(
            not path.is_absolute()
            and ".." not in path.parts
            and path.parts[0] in {"docs", "src", "tests", "licenses", "configs"},
            "evidence path outside the repository projection",
        )
    mandatory = {p for _, _, _, xs, _ in STATIC for p in xs} | {
        PERF_DOC,
        T1 + "resources.md",
        PUBLIC + "resources/" + phase + "/resources.json",
    }
    proofs = {p: sha(project / p) for p in sorted(files) if (project / p).is_file()}
    require(
        mandatory <= set(proofs),
        "referenced repository evidence absent: " + ",".join(sorted(mandatory - set(proofs))),
    )
    git = subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip()
    commit = updates.get("code_commit") or (
        checked(store, pins["manifest"])[0]["code_commit"] if "manifest" in pins else git
    )
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "expected a full code commit")
    return {
        "schema": "el-general-matrix-inputs-1",
        "phase": phase,
        "tool_version": updates.get("tag", "v0.4.0"),
        "code_commit": commit,
        "repository_head": git,
        "cutoff_utc": utc(),
        "pins": pins,
        "public_evidence": additional,
        "repository_files_sha256": proofs,
        "notes": updates.get("notes", {}),
    }


def build(store, packet):
    pins = packet["pins"]
    data, identities = {}, {}
    for name, pin in pins.items():
        data[name], identities[name] = checked(store, pin)
    docs = packet["repository_files_sha256"]
    rows = []

    def add(item, status, roles, paths, limit, details=None):
        require(status in STATES and not any(r["item"] == item for r in rows), "invalid or duplicate item")
        pointers = [p for role in roles for p in packet["public_evidence"].get(role, [])]
        evidence = list(dict.fromkeys([*paths, *pointers]))
        require(all(p in docs for p in evidence), "public evidence missing at cutoff")
        provenance = [identities[role] for role in roles]
        code = (
            ("sha256:" + provenance[-1]["recorded_source_snapshot_sha256"])
            if provenance
            else packet["code_commit"]
        )
        rows.append(
            {
                "item": item,
                "status": status,
                "evidence": evidence,
                "code_version": code,
                "model_version": RELEASE,
                "runs": list(dict.fromkeys(p["run_id"] for p in provenance)),
                "evidence_roles": list(roles),
                "limitation": limit,
                **({"observations": details} if details is not None else {}),
            }
        )

    for item, status, roles, paths, limit in STATIC:
        add(item, status, roles, paths, limit)
    agree = {
        "t0_evaluation": "docs/results/t0/confirmation/evaluation.json",
        "c6_calibration": "docs/results/batch-two/calibration/calibration-results.json",
        "c6_recount": "docs/results/batch-two/calibration/calibration-audit.json",
        "g2-t1-frozen-four-models": PUBLIC + "t1-numerical-bridge.json",
        "g2-t1-selected8-supplement": PUBLIC + "t1-threads8-numerical-bridge.json",
        "g4-public-original-trajectories": PUBLIC + "performance/g4-trajectory-verification.json",
    }
    for role, path in agree.items():
        require(
            pins[role]["sha256"] == docs[path], "public projection differs from the native original: " + path
        )
    require(
        data["t1-closed-matrix"]["schema"] == "el-t1-closed-matrix-1"
        and len(data["t1-closed-matrix"]["fixed_score_terminals"]) == 288
        and data["g2-t1-frozen-four-models"]["summary"]["compared_method_views"] == 240
        and data["g2-t1-selected8-supplement"]["summary"]["comparisons"]["archived4_to_file8"][
            "bit_exact_method_views"
        ]
        == 204,
        "fixed denominators or bridge receipts differ",
    )
    status = {}

    def pending(item, role, paths, reason, default="blocked"):
        note = packet["notes"].get(role, {})
        state = note.get("status", default)
        require(state in {"blocked", "not_started", "failed"}, "pending state cannot assert completion")
        add(item, state, [], paths, note.get("limitation", reason))
        return False

    for host in ("mac", "windows"):
        for phase in ("g3", "g4"):
            role = host + "_" + phase
            if role not in data:
                status[role] = pending(
                    role + "_measurements",
                    role,
                    [PERF_DOC],
                    "Native result verification awaits the dispatched platform return.",
                )
                continue
            rec = data[role]
            observations = None
            native = run_folder(store, pins[role]["run_id"])
            env = load(native / "environment.json")
            files = load(native / "manifest.json")["files"]
            require(
                any(
                    f["path"] == "environment.json" and f["sha256"] == sha(native / "environment.json")
                    for f in files
                ),
                "native performance environment differs from its seal",
            )
            expected_os = "macOS-" if host == "mac" else "Windows-"
            require(
                env["platform"].startswith(expected_os)
                and rec["engineering_config_sha256"]
                == data["g2-t1-frozen-four-models"]["configuration_sha256"],
                "wrong performance host or engineering freeze",
            )
            require(
                rec["schema"]
                == ("el-original-cpu-performance-1" if phase == "g3" else "el-static-replay-tests-1"),
                "performance role points to an intake instead of its native result",
            )
            if phase == "g3":
                require(
                    rec["release"] == data["g2-t1-frozen-four-models"]["model_release_sha256"],
                    "performance scorer used different released weights",
                )
                configs = {(c["scene"], c["method"], c["path"]): c for c in rec["configurations"]}
                required = {
                    (s, m, p)
                    for s in ("bell-M4", "bell-M2", "beads-M4", "beads-M2")
                    for m in ("MS-L1", "MS-ft")
                    for p in ("file", "resident-cache")
                }
                complete = (
                    rec["status"] == "done"
                    and not rec["failures"]
                    and all(
                        key in configs and configs[key]["status"] == "done" and configs[key]["N"] >= 200
                        for key in required
                    )
                )
                desc = f"{len(configs)} measured configurations; {sum(c['N'] for c in configs.values())} timed calls and {len(rec['failures'])} registered failures. Elapsed G3 time was {rec['wall_seconds']:.3f} s within a {rec['budget_seconds']} s budget."
            else:
                require(
                    rec["platform"].startswith(expected_os)
                    and rec["labels_accessed"] is False
                    and rec["new_risk_training_updates"] == 0,
                    "wrong replay provenance",
                )
                traces = rec["loads"]
                arms = {"MS-L1", "MS-ft"} if host == "mac" else {"MS-L1"}
                expected = {(m, h) for m in arms for h in (10, 30)}
                observed = {(r["method"], int(r["id"].split("-")[-1].removesuffix("Hz"))) for r in traces}
                complete = (
                    rec["status"] == "done"
                    and observed == expected
                    and set(rec["parity"]) >= arms
                    and all(rec["parity"][m]["passed"] for m in arms)
                    and all(
                        r["inputs"] == sum(r["status"].values()) and r["unique_offered"] == 120 for r in traces
                    )
                    and rec["version_switch_status"]["finished"] > 0
                )
                desc = f"{len(traces)} sustained loads. {sum(x['inputs'] for x in traces)} input slots; {sum(x['status']['finished'] for x in traces)} finished. Every generated output is reconciled against the static render source. The revision check recorded {rec['version_switch_status']['finished']} completions."
            if phase == "g4" and packet["phase"] == "final":
                consume = ("finished", "dropped", "expired", "failed")
                terminal_rows = [
                    {
                        "workload": t["id"],
                        "method": t["method"],
                        "offered": t["inputs"],
                        "distinct_target_views": t["unique_offered"],
                        **{k: t["status"][k] for k in consume},
                    }
                    for t in traces
                ]
                switch = {k: rec["version_switch_status"][k] for k in consume}
                native_run = load(native / "run.json")
                observations = {
                    "sustained_loads": terminal_rows,
                    "version_switch": {**switch, "offered": sum(switch.values())},
                    "native_recorder_exit_code": native_run["exit_code"],
                    "native_worker_exit_code": native_run["worker_exit_code"],
                    "sustained_frame_execution_failures": sum(x["failed"] for x in terminal_rows),
                    "switch_frame_execution_failures": switch["failed"],
                }
                desc += " The finished/dropped/expired/failed counts per load were " + ", ".join(
                    f"{t['workload']} {t['finished']}/{t['dropped']}/{t['expired']}/{t['failed']}"
                    for t in terminal_rows
                )
                desc += f". Revision finished/dropped/expired/failed counts were {switch['finished']}/{switch['dropped']}/{switch['expired']}/{switch['failed']}. Native worker exit code {native_run['worker_exit_code']}."
            add(
                role + "_measurements",
                "done" if complete else "failed",
                [role],
                [PERF_DOC],
                desc,
                observations,
            )
            status[role] = complete
            if phase == "g4":
                if packet["phase"] == "final":
                    ten = [r for r in traces if r["id"].endswith("10Hz")]
                    offered = sum(r["inputs"] for r in ten)
                    outcomes = {
                        k: sum(r["status"][k] for r in ten)
                        for k in ("finished", "dropped", "expired", "failed")
                    }
                    add(
                        host + "_10hz_full_frame_delivery",
                        "done" if outcomes["finished"] == offered else "done_negative",
                        [role],
                        [PERF_DOC],
                        f"{outcomes['finished']} of {offered} offers finished; {outcomes['dropped']} were dropped, {outcomes['expired']} expired and {outcomes['failed']} failed.",
                    )
                thirty = [r for r in traces if r["id"].endswith("30Hz")]
                dropped = sum(r["status"]["dropped"] for r in thirty)
                offered = sum(r["inputs"] for r in thirty)
                add(
                    host + "_30hz_full_frame_delivery",
                    "done_negative" if dropped else "done",
                    [role],
                    [PERF_DOC],
                    f"{dropped} of {offered} offered inputs were dropped under 30 Hz load. This load characterizes bounded service on prepared static evidence.",
                )
    tables_ok = False
    if "tables" in data:
        archive = data["tables"]
        tables_path = "payload/tables.json"
        tpin = historical(store, pins["tables"]["run_id"], tables_path)
        tables, table_identity = checked(store, tpin)
        require(
            tables["schema"] == "el-static-engineering-measurements-1" and bool(archive),
            "missing hash-sealed performance tables",
        )
        for host in ("mac", "windows"):
            for stage_name in ("g3", "g4"):
                role = host + "_" + stage_name
                source = tables["sources"].get(role) or tables["sources"].get(host, {}).get(stage_name)
                require(
                    role in pins
                    and source
                    and source["run_id"] == pins[role]["run_id"]
                    and source["receipt_sha256"] == pins[role]["sha256"]
                    and source["manifest_sha256"] == pins[role]["manifest_sha256"],
                    "table/native platform identity differs",
                )
        identities["table_index"] = table_identity
        table_root = run_folder(store, pins["tables"]["run_id"]) / "payload"
        manifest_members = {f["path"]: f for f in load(table_root.parent / "manifest.json")["files"]}
        for filename in ("g3-configurations.csv", "g4-loads.csv", "g4-version-checks.csv"):
            entry = tables["tables"][filename]
            f = table_root / filename
            require(
                entry["sha256"] == sha(f) == manifest_members["payload/" + filename]["sha256"]
                and f.stat().st_size == entry["bytes"],
                "published performance projection bytes differ",
            )
            with f.open(encoding="utf-8", newline="") as inp:
                rows_in_table = list(csv.DictReader(inp))
            require(len(rows_in_table) == entry["rows"], "performance projection row count differs")
            if filename == "g3-configurations.csv":
                requires = {
                    (h, scene, m, route)
                    for h in ("mac", "windows")
                    for scene in ("beads-M2", "beads-M4", "bell-M2", "bell-M4")
                    for m in ("MS-L1", "MS-ft")
                    for route in ("file", "resident-cache", "resident-no-cache")
                }
                actual = {(r["platform"], r["condition"], r["method"], r["path"]): r for r in rows_in_table}
                require(
                    set(actual) == requires
                    and len(actual) == len(rows_in_table)
                    and all(int(r["N"]) == 200 and int(r["cold_N"]) == 3 for r in rows_in_table),
                    "published real baseline, resident paths or cold-process coverage differs",
                )
        tables_ok = True
        add(
            "published_performance_tables",
            "done",
            ["tables", "table_index"],
            [PERF_DOC],
            "The tables identify each native platform receipt. Replay uses pre-rendered static evidence.",
        )
    else:
        pending(
            "published_performance_tables",
            "tables",
            [PERF_DOC],
            "The cross-platform public projection requires both native measurement returns.",
        )
    cloud_ok = False
    if "cloud_log" in data:
        rec = data["cloud_log"]
        checks = rec.get("entry_checks", [])
        names = [x["entry"] for x in checks]
        require(
            rec["schema"] == "el-general-private-log-projection-1"
            and bool(rec["revision_sha256"])
            and bool(rec["document_text_sha256"])
            and (packet["phase"] == "final" or {"RL-017", "RL-018", "RL-019", "RL-020"} <= set(names))
            and len(names) == len(set(names))
            and all(x["count"] == 1 and x["paragraph_text_sha256"] for x in checks),
            "recorded companion-document readback is incomplete",
        )
        if packet["phase"] == "final":
            prev = data.get("cloud_previous", {})
            old_entries = prev.get("entry_checks", [])
            require(
                {"RL-017", "RL-018", "RL-019", "RL-020"}
                <= set(names)
                | {x["entry"] for x in old_entries if x["count"] == 1 and x["paragraph_text_sha256"]},
                "earlier log sections lack a verified companion readback",
            )
            require(
                "RL-022" in names,
                "final companion log lacks the post-public RL-022 readback",
            )
        cloud_ok = True
        add(
            "companion_research_log",
            "done",
            [k for k in ("cloud_previous", "cloud_log") if k in data],
            [LOG],
            "Each verified section occurred once: "
            + ", ".join(names)
            + ". The revision digest is "
            + rec["revision_sha256"]
            + ".",
        )
    else:
        pending(
            "companion_research_log",
            "cloud_log",
            [LOG],
            "The repository log is available. This cutoff lacks a companion-document readback.",
        )
    phase = packet["phase"]
    rd = data["g6-storage-" + phase]
    require(
        rd["schema"] == "el-general-size-ledger-2"
        and docs[PUBLIC + "resources/" + phase + "/resources.json"] == pins["g6-storage-" + phase]["sha256"],
        "published storage ledger differs",
    )
    add(
        "resource_accounting",
        "done",
        ["t1-resources", "g6-storage-" + phase],
        [T1 + "resources.md", PUBLIC + "resources/" + phase + "/resources.json"],
        f"{rd['sealed_runs']} sealed native runs at {rd['start_utc']}. Full local-store logical bytes are {rd['local_store']['logical_bytes']}; external linked history and the active census are outside that scan.",
    )
    pub_ready = release_rows(packet, data, pins, add, pending, status, store)
    add(
        "additional_risk_training_t2_specialization_embodied_control",
        "out_of_scope",
        [],
        [READY],
        "Zero new risk-model fits. Field data collection, specialized risk models, geometry and semantics training and vehicle or robot action await a separate study.",
    )
    return {
        "schema": "el-general-completion-matrix-2",
        "run_id": os.getenv("EL_RUN_ID"),
        "phase": phase,
        "evidence_cutoff_utc": packet["cutoff_utc"],
        "tool_version": packet["tool_version"],
        "model_version": RELEASE,
        "code_commit_at_cutoff": packet["code_commit"],
        **({"repository_head_at_cutoff": packet["repository_head"]} if "repository_head" in packet else {}),
        **(
            {
                "public_release_url": data["github"]["url"]
                if "github" in data and data["github"].get("draft") is False
                else None
            }
            if phase == "final"
            else {}
        ),
        "software_release_ready": pub_ready,
        "general_phase_complete": pub_ready
        and status.get("linux_acceptance", False)
        and tables_ok
        and cloud_ok
        and all(status.get(h + "_" + s, False) for h in ("mac", "windows") for s in ("g3", "g4")),
        "new_risk_training_count": 0,
        "T1_unique_scene_optimizations": 29,
        "T1_recalibration_count": 0,
        "items": rows,
        "recorder_evidence": identities,
        "repository_files_sha256": docs,
    }


def release_rows(packet, data, pins, add, pending, status, store=None):
    evidence = ["docs/releases/v0.4.0/README.md"]
    tag = packet["tool_version"]
    doc = data.get("manifest")
    if doc:
        require(
            doc["schema"] == "el-general-release-1"
            and doc["tag"] == tag
            and doc["code_commit"] == packet["code_commit"],
            "release tag or commit differs",
        )
    public = data.get("download")
    published = False
    if public and doc:
        require(
            public["schema"] == "el-general-public-download-1"
            and public["tag"] == tag
            and public["code_commit"] == doc["code_commit"],
            "download is for a different release",
        )
        requests = {r["name"]: r for r in public["anonymous_requests"]}
        entries = doc["assets"]
        expect = {name: item["sha256"] for name, item in entries.items()}
        expect["release-manifest.json"] = pins["manifest"]["sha256"]
        require(
            all(
                name in requests
                and requests[name]["sha256"] == digest
                and requests[name]["http_status"] == 200
                and requests[name]["anonymous"] is True
                for name, digest in expect.items()
            ),
            "public readback did not verify all artifacts",
        )
        published = True
        add(
            "anonymous_release_download",
            "done",
            ["manifest", "download"],
            evidence,
            "All declared current-release assets were downloaded and checked against the published manifest.",
        )
    else:
        pending(
            "anonymous_release_download",
            "download",
            evidence,
            "Verification of the anonymous current-release assets awaits public publication.",
            "not_started",
        )
    primary = False
    for host, platform in (("mac", "Darwin"), ("windows", "Windows"), ("linux", "Linux")):
        role = host + "_acceptance"
        if role not in data or not published:
            status[role] = pending(
                role,
                role,
                evidence,
                "Clean replay against the current public wheel awaits an authenticated platform receipt.",
                "not_started",
            )
            continue
        result = data[role]
        if result.get("schema") == "el-general-linux-clean-acceptance-1" or "clean_acceptance" in result:
            result = result["clean_acceptance"]
        require(
            result["schema"] == "el-general-clean-acceptance-1"
            and result["tag"] == tag
            and result["code_commit"] == doc["code_commit"]
            and result["platform"]["os"] == platform,
            "clean replay used a different platform, commit or tag",
        )
        required = {"default"} if host == "linux" else {"default", "vision"}
        assets = {a["role"]: a["sha256"] for a in doc["assets"].values()}
        require(
            result["wheel_sha256"] == assets["wheel"]
            and result["archive_sha256"] == assets["original_replays"],
            "clean replay used a different wheel or original archive",
        )
        groups = result["groups"]
        ok = (
            required <= set(groups)
            and all(
                g["replay_completed"] and g["distributions"]["evidence-lens"] == tag[1:]
                for g in groups.values()
            )
            and groups["default"]["method"] == "MS-L1"
            and (host == "linux" or groups["vision"]["method"] in ("MS-ft", "all"))
        )
        status[role] = ok
        primary = primary or bool(ok and groups.get("vision", {}).get("method") == "all")
        roles = [role]
        desc = (
            result["platform"]["arch"]
            + " "
            + platform
            + ", clean groups "
            + ", ".join(n + "=" + g["method"] + " (" + g["status"] + ")" for n, g in groups.items())
            + "."
        )
        observations = None
        if packet["phase"] == "final" and "verification" in data:
            v = data["verification"]["original_replays"][host]
            observations = {
                "authored_scene": "offset bell cage",
                "layout": "M4",
                "target_views_per_method": 6,
                "numerical_results_by_group": v["groups"],
            }
            if host == "windows":
                first = data.get("windows_first_attempt")
                require(
                    first
                    and first["terminal_status"] == "failed"
                    and first["clean_replay_started"] is False
                    and first["release_manifest_sha256"] == pins["manifest"]["sha256"],
                    "original Windows failure provenance differs",
                )
                rows_path = (
                    run_folder(store, pins["verification"]["run_id"])
                    / "payload/windows/numerical-differences.csv"
                )
                require(
                    sha(rows_path) == v["per_view_quota_table_sha256"], "verified Windows quota table differs"
                )
                with rows_path.open(encoding="utf-8", newline="") as inp:
                    changed = [
                        {
                            "method": r["method"],
                            "target_view": r["view"],
                            "quota": r["anchor"],
                            "added_pixels": int(r["added_pixels"]),
                            "removed_pixels": int(r["removed_pixels"]),
                        }
                        for r in csv.DictReader(inp)
                        if r["old_attainable"] == "True"
                        and r["new_attainable"] == "True"
                        and (int(r["added_pixels"]) or int(r["removed_pixels"]))
                    ]
                default = v["groups"]["default"]["MS-L1"]
                ft = v["groups"]["vision"]["MS-ft"]
                require(
                    sum(x["method"] == "MS-ft" for x in changed) == ft["quota_membership_changed"],
                    "Windows changed quota membership recount differs",
                )
                observations["changed_attainable_quotas"] = changed
                observations["first_failed_native"] = first["native_run_id"]
                desc += f" The first native attempt stopped at TLS issuer-chain validation before scoring. The recovered public download completed using certificate verification. All {default['bit_exact_views']}/{default['views']} default target views matched. MS-ft matched {ft['bit_exact_views']}/{ft['views']} target views bit for bit. The {ft['changed_pixels']['risk']:,} changed risk values had a maximum absolute difference of {ft['max_abs_risk_difference']:.17g}. Valid/state/reason differences were {ft['changed_pixels']['valid']}/{ft['changed_pixels']['state']}/{ft['changed_pixels']['unknown_reason']}, with {ft['threshold_crossings']} threshold crossings."
                desc += (
                    f" {ft['quota_membership_changed']} of {ft['attainable_quotas']} attainable MS-ft view–quotas changed. The affected entries were "
                    + ", ".join(
                        f"{x['target_view']} at {x['quota']} (+{x['added_pixels']}, -{x['removed_pixels']})"
                        for x in changed
                        if x["method"] == "MS-ft"
                    )
                    + "."
                )
                roles.extend(
                    [
                        n
                        for n in (
                            "windows_first_attempt",
                            "windows_acceptance_intake",
                            "verification",
                            "chronology",
                        )
                        if n in data
                    ]
                )
        add(role, "done" if ok else "failed", roles, evidence, desc, observations)
    final_audited = False
    chronology_ok = False
    if packet["phase"] == "final":
        if "chronology" in data and doc:
            clock = data["chronology"]
            chain = {
                "previous_failure": "windows_first_attempt",
                "windows_grade": "windows_acceptance",
                "windows_intake": "windows_acceptance_intake",
                "publication": "github",
                "mac_download": "download",
                "mac_grade": "mac_acceptance",
                "linux_grade": "linux_acceptance",
            }
            require(
                clock["schema"] == "el-public-replay-chronology-1"
                and clock["tag"] == tag
                and clock["code_commit"] == doc["code_commit"]
                and clock["manifest_sha256"] == pins["manifest"]["sha256"]
                and all(k in pins and clock["source_receipts"][ck] == pins[k] for ck, k in chain.items())
                and clock["windows_trust"]["certificate_verification"] == "CERT_REQUIRED"
                and clock["windows_trust"]["hostname_verification"] is True
                and bool(clock["windows_trust"]["ssl_cert_file_sha256"]),
                "post-public native timing and Windows trust chronology differs",
            )
            paths = packet["public_evidence"].get("chronology", [])
            require(
                any(packet["repository_files_sha256"].get(p) == pins["chronology"]["sha256"] for p in paths),
                "public chronology projection differs from its sealed original",
            )
            chronology_ok = True
            add(
                "post_public_utc_and_windows_trust",
                "done",
                ["chronology"],
                evidence,
                f"The release became public at {clock['published_at']}. The strict chronology binds the later anonymous fetch and clean replay runs, the network-off Linux input and the first failed Windows download. The successful Windows recovery required the CA chain and verified the hostname.",
            )
        else:
            pending(
                "post_public_utc_and_windows_trust",
                "chronology",
                evidence,
                "The independently checked public timing and trust chronology is pending.",
            )
        if "build" in data and doc:
            b = data["build"]
            require(
                b["schema"] == "el-general-build-1"
                and b["code_commit"] == doc["code_commit"]
                and b["risk_weights_release"] == RELEASE
                and b["version"] == tag[1:]
                and {n: s for n, s in b["assets"].items()}
                == {
                    n: {k: a[k] for k in ("bytes", "sha256")}
                    for n, a in doc["assets"].items()
                    if a["role"] in ("sources", "wheel")
                },
                "recorded wheel or source archive differs from the published inventory",
            )
            add(
                "current_source_and_wheel",
                "done",
                ["build", "manifest"],
                evidence,
                "The committed source export and wheel match the release manifest by length and SHA-256.",
            )
        else:
            pending(
                "current_source_and_wheel",
                "build",
                evidence,
                "Awaiting the recorded build and its matching release inventory.",
            )
        if "github" in data and doc:
            gh = data["github"]
            require(
                gh["schema"] == "el-general-release-upload-1"
                and gh["repo"] == "wany614/evidence-lens"
                and gh["operation"] == "public"
                and gh["draft"] is False
                and gh["tag"] == tag
                and gh["code_commit"] == doc["code_commit"]
                and gh["manifest_sha256"] == pins["manifest"]["sha256"],
                "GitHub publication identity differs",
            )
            add(
                "tagged_github_assets",
                "done",
                ["github", "manifest"],
                evidence,
                f"GitHub release {gh['release_id']} resolved to the declared commit and package inventory at {gh['published_at']}.",
            )
        else:
            pending(
                "tagged_github_assets", "github", evidence, "The verified public GitHub release is pending."
            )
        if "regressions" in data and doc:
            tests = data["regressions"]
            require(
                tests["source_commit"] == doc["code_commit"]
                and tests["tests_passed"] > 0
                and tests["skipped"] == 0,
                "immutable source regression results differ",
            )
            add(
                "current_release_regressions",
                "done",
                ["regressions"],
                evidence,
                f"{tests['tests_passed']} source-pinned regression tests passed, including the deployed interfaces.",
            )
        else:
            pending(
                "current_release_regressions",
                "regressions",
                evidence,
                "Awaiting source-pinned regression results.",
            )
        if (
            "verification" in data
            and published
            and all(
                x in data
                for x in (
                    "build",
                    "github",
                    "regressions",
                    "mac_acceptance",
                    "windows_acceptance",
                    "linux_acceptance",
                )
            )
        ):
            verified = data["verification"]
            require(
                verified["schema"] == "el-general-public-verification-1"
                and verified["tag"] == tag
                and verified["code_commit"] == doc["code_commit"]
                and set(verified["original_replays"]) == {"mac", "windows", "linux"}
                and verified["compiled_tests"] == data["regressions"]
                and verified["manifest_sha256"] == pins["manifest"]["sha256"]
                and verified["release"] == data["github"]
                and verified["anonymous_mac"] == data["download"],
                "public verification lineage differs",
            )
            for host in ("mac", "windows", "linux"):
                expected = (
                    data["linux_acceptance"]["acceptance_sha256"]
                    if host == "linux"
                    else pins[host + "_acceptance"]["sha256"]
                )
                require(
                    verified["original_replays"][host]["acceptance_sha256"] == expected,
                    "public platform summary differs",
                )
            final_audited = True
            add(
                "independent_public_release_reconciliation",
                "done",
                ["verification"],
                evidence,
                "Public digest lists, Mac and Windows anonymous downloads, native per-view grades, pinned Linux container and source regressions were reconciled.",
            )
        else:
            pending(
                "independent_public_release_reconciliation",
                "verification",
                evidence,
                "Post-public reconciliation awaits complete sealed native replay inputs.",
            )
    if published:
        add(
            "public_release",
            "done",
            ["manifest", "download"],
            evidence,
            "The tagged assets and anonymous verification are recorded; platform replay coverage is itemized independently.",
        )
    else:
        pending(
            "public_release",
            "manifest",
            evidence,
            "The declared release has no anonymous publication receipt at this cutoff.",
            "not_started",
        )
    return (
        published
        and (packet["phase"] != "final" or (final_audited and chronology_ok))
        and primary
        and status.get("mac_acceptance", False)
        and status.get("windows_acceptance", False)
    )


def record(args):
    from general_records import resolve, stage

    store = args.store.resolve()
    project = Path(__file__).resolve().parents[1]
    phase = args.phase
    name = "g6-completion-" + phase
    controller = store / "work/controller" / name
    controller.mkdir(exist_ok=True, parents=True)
    frozen = controller / "matrix-inputs.json"
    updates = load(args.facts) if args.facts else {}
    if not frozen.exists():
        save(frozen, collect(store, project, phase, updates))
    packet = load(frozen)
    parents = {
        role: packet["pins"][role]
        for role in ("g6-storage-" + phase, "g2-t1-selected8-supplement", "tables", "download")
        if role in packet["pins"]
    }
    pin = stage(
        store,
        store / "history",
        name,
        "derivation",
        parents,
        {"phase": phase, "fixed_cutoff_utc": packet["cutoff_utc"]},
        [
            sys.executable,
            "scripts/general_closeout_matrix.py",
            "worker",
            "--store",
            str(store),
            "--packet",
            "{input:matrix-inputs}",
            "--out",
            "{out}/completion.json",
        ],
        "payload/completion.json",
        450000,
        extra={"matrix-inputs": frozen},
        retry=args.retry,
    )
    target = args.public_root / ("completion-" + phase + ".json")
    raw = resolve((store, store / "history"), pin).read_bytes()
    require(
        not target.exists() or target.read_bytes() == raw,
        "matrix for the phase already exists with other bytes",
    )
    if not target.exists():
        with target.open("xb") as f:
            f.write(raw)
    print(json.dumps(pin, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("record")
    run.add_argument("--phase", required=True, choices=("candidate", "final"))
    run.add_argument("--facts", type=Path)
    run.add_argument("--retry")
    run.add_argument("--public-root", type=Path, default=Path("docs/general-closeout"))
    w = sub.add_parser("worker")
    w.add_argument("--packet", type=Path, required=True)
    w.add_argument("--out", type=Path, required=True)
    for x in (run, w):
        x.add_argument("--store", required=True, type=Path)
    a = parser.parse_args()
    if a.action == "record":
        record(a)
    else:
        save(a.out, build(a.store, load(a.packet)))
