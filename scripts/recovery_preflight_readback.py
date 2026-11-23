"""Independently qualify the sealed synthetic Windows capacity, resume, camera, and cutoff preflight."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import zipfile
import hashlib

import numpy as np

from evidence_lens._bundle import encoded, load, sha
from evidence_lens._verify import require
from recovery_inputs import copy_script, located
from recovery_outputs import verify_code

EXPECTED = {
    "plan",
    "data",
    "topology",
    "projection",
    "continuous",
    "first-segment",
    "continuation",
    "render-all",
    "render-selected",
    "cutoff",
    "parity",
}


def read_original(store, pin, name):
    root = located(store, pin)
    manifest = load(root / "manifest.json")
    row = next((m for m in manifest["files"] if m["path"] == name), None)
    p = root / name
    require(
        row
        and p.is_file()
        and not p.is_symlink()
        and p.stat().st_size == row["bytes"]
        and sha(p) == row["sha256"],
        "Different engineering member " + name,
    )
    return p


def sample_resources(recs):
    result = {}
    for role, r in recs.items():
        data = r["resources"]
        result[role] = {
            k: data.get(k)
            for k in (
                "recorder_wall_seconds",
                "numeric_process_wall",
                "torch_allocated_peak_bytes",
                "torch_reserved_peak_bytes",
                "sampled_device_memory_peak_MiB",
                "sampled_process_memory_peak_bytes",
                "device_memory_samples",
                "process_memory_samples",
                "process_counter_missing_reasons",
            )
        }
    return result


def verify_probe(store, summary, source, environment, earlier):
    d = load(summary)
    recs = d["runs"]
    require(
        d["status"] == "completed"
        and d["scope"] == "synthetic-only"
        and set(recs) == EXPECTED
        and len({r["run"] for r in recs.values()}) == len(EXPECTED),
        "Incomplete synthetic probe roster",
    )
    pins = {k: {"run_id": r["run"], "manifest_sha256": r["manifest_sha256"]} for k, r in recs.items()}

    def asset(k, n):
        return read_original(store, pins[k], "payload/" + n)

    def info(k, n):
        return load(asset(k, n))

    snapshots = {}
    for role, row in recs.items():
        t = load(store / "recovery/returns" / row["run"] / "transport.json")
        require(
            t["pin"] == pins[role]
            and t["worker_receipt"] == row
            and t["zip_sha256"] == row["sha256"]
            and t["zip_bytes"] == row["bytes"],
            "Unverified original provider delivery: " + role,
        )
        root = located(store, pins[role])
        r = load(read_original(store, pins[role], "run.json"))
        require(
            r["execution_status"] == row["status"]
            and r["run_kind"] == row["run_kind"]
            and r["experiment_id"] == "r-recovery-engineering"
            and ((row["status"] == "failed") == (role == "cutoff")),
            "Unexpected engineering outcome",
        )
        snapshot = load(read_original(store, pins[role], "code_snapshot.json"))["snapshot_sha256"]
        snapshots[role] = snapshot
        if role in ("plan", "data"):
            # These earlier source-only preparatory records precede the repaired CUDA probe.
            # Compare their actual executable and Python recorder modules to the final source ZIP.
            script = "tests/engineering_registration.py" if role == "plan" else "tests/make_fixture.py"
            argv = load(read_original(store, pins[role], "command.json"))["argv_template"]
            require("{source}/" + script in argv, "Different plan or synthetic fixture preparer")
            with zipfile.ZipFile(source) as z:
                names = [
                    script,
                    *[n for n in z.namelist() if n.startswith("src/evidence_lens/") and n.endswith(".py")],
                ]
                require(
                    all(
                        sha(read_original(store, pins[role], "source/" + n))
                        == hashlib.sha256(z.read(n)).hexdigest()
                        for n in names
                    ),
                    "Earlier plan/fixture preparer changed",
                )
        else:
            verify_code(source, root, {"source_sha256": snapshot, "gsplat": "1.5.3"})
    reg = info("plan", "registration.json")
    fix = info("data", "fixture.json")
    require(
        reg["recipe_candidate_sha256"] == d["candidate_recipe_sha256"]
        and reg["point_cap"] == 20000
        and reg["recipe_horizon"] == 5000
        and reg["resource_probe_resolution"] == [256, 256]
        and fix["reference_target_images"] == 0,
        "Wrong synthetic probe registration",
    )
    topo = info("topology", "forced-topology.json")
    steps = topo["steps"]
    cap = reg["point_cap"]
    require(
        topo["passed"]
        and [r["step"] for r in steps] == [1, 2, 3, 4, 5]
        and [r["after"] for r in steps] == [cap, cap, cap - 4, cap, cap]
        and all(
            r["after"] <= cap
            and r["visible"] > 0
            and r["delta_l2"]["means"] > 0
            and r["delta_l2"]["colors"] > 0
            for r in steps
        )
        and not steps[-1]["events"],
        "Capacity or fixed-tail control failed",
    )
    first = next(e for e in steps[0]["events"] if e["operation"] == "growth_budget")
    ops = {
        op: sum(e.get("selected", 0) for r in steps for e in r["events"] if e["operation"] == op)
        for op in ("prune", "clone", "split", "reset")
    }
    require(
        first["blocked"] == 8
        and first["admitted"] == 0
        and all(ops.values())
        and all(e["adam_moments_checked"] for r in steps for e in r["events"] if e["operation"] in ops),
        "Did not exercise source optimizer cap and four topology branches",
    )
    proj = info("projection", "ed-probe.json")
    opacity = 0.4 + (1 - 0.4) * 0.65
    expected_z = (0.4 * 0.41 + (1 - 0.4) * 0.65 * 0.91) / opacity
    mix = opacity * np.array([0.2, 0.45, 0.62])
    display = np.where(mix <= 0.0031308, 12.92 * mix, 1.055 * mix ** (1 / 2.4) - 0.055)
    require(
        proj["passed"]
        and abs(proj["actual_z_m"] - expected_z) < 1e-4
        and abs(proj["actual_alpha"] - opacity) < 5e-4
        and np.max(np.abs(np.array(proj["actual_display_rgb"]) - display)) < 5e-4
        and np.max(np.abs(np.asarray(proj["native_edge_means2d"]) - [22.5, 12.5])) < 1e-4,
        "Independent two-surface projection/colour/depth check failed",
    )
    parity = info("parity", "resume-parity.json")
    require(
        parity["passed"]
        and not parity["interruption_boundary"]["mismatches"]
        and not parity["endpoint"]["mismatches"]
        and parity["same_trace_schedule_and_topology"]
        and parity["same_selected_checkpoint_views"]
        and parity["max_source_loss_difference"] <= fix["absolute_display_loss_tolerance"]
        and parity["all_checkpoint_steps"] == reg["resume_fixture_steps"],
        "Continuation or arbitrary checkpoint rendering failed",
    )
    full = info("continuous", "reconstruction.json")
    head = info("first-segment", "reconstruction.json")
    tail = info("continuation", "reconstruction.json")
    require(
        full["last_completed_step"] == tail["last_completed_step"] == fix["end_step"]
        and head["last_completed_step"] == tail["first_completed_step"] == fix["resume_at"]
        and tail["resume_origin"]["run_id"] == pins["first-segment"]["run_id"]
        and tail["resume_origin"]["checkpoint_sha256"]
        == head["checkpoints"][str(fix["resume_at"])]["checkpoint_sha256"],
        "Resume does not use the issued interruption state",
    )
    allout = info("render-all", "outputs.json")
    single = info("render-selected", "outputs.json")
    subset = str(fix["resume_at"])
    require(
        single["checkpoints"] == {subset: allout["checkpoints"][subset]}
        and allout["reconstruction_run_id"] == pins["continuation"]["run_id"],
        "Selected checkpoint disagrees with all-checkpoint original",
    )
    same = 0
    for view, item in allout["checkpoints"][subset]["views"].items():
        require(
            sha(asset("render-all", item["path"]))
            == sha(asset("render-selected", item["path"]))
            == item["sha256"],
            "Published checkpoint image differs: " + view,
        )
        same += 1
    cut = info("cutoff", "process-wall.json")
    require(
        cut["resource_cutoff"] is True
        and cut["exit_code"] == recs["cutoff"]["exit_code"] == 124
        and cut["budget_seconds"] == 1
        and cut["cumulative_seconds"] >= cut["budget_seconds"]
        and d["expected_cutoff_run"] == pins["cutoff"]["run_id"],
        "Watchdog did not produce the expected original failed attempt",
    )
    runtime = topo["runtime"]
    system = load(environment)
    device = dict(zip(system["device_fields"], system["nvidia_sample"], strict=True))
    device_bytes = int(device["total_MiB"]) * 2**20
    resources = sample_resources(recs)
    cap_memory = resources["topology"]
    require(
        0
        < cap_memory["torch_allocated_peak_bytes"]
        <= cap_memory["torch_reserved_peak_bytes"]
        <= device_bytes
        and device["GPU"] == runtime["device"],
        "Actual forced cap allocator exceeds recorded matched GPU capacity",
    )
    failures = []
    for pin in earlier:
        erun = load(read_original(store, pin, "run.json"))
        require(
            erun["execution_status"] == "failed" and erun["experiment_id"] == "r-recovery-engineering",
            "Not a recorded engineering failure",
        )
        stderr = read_original(store, pin, "stderr.log")
        failures.append(
            {
                **pin,
                "trial_id": erun["trial_id"],
                "exit_code": erun["exit_code"],
                "failure_reason": erun["failure_reason"],
                "stderr_sha256": sha(stderr),
                "last_error": stderr.read_text(encoding="utf8", errors="replace").splitlines()[-1],
                "elapsed_seconds": erun["elapsed_seconds_monotonic"],
            }
        )
    return {
        "schema": "el-recovery-resource-probe-1",
        "run_id": os.environ["EL_RUN_ID"],
        "passed": True,
        "series": d["series"],
        "scope": "synthetic engineering only",
        "candidate_recipe_sha256": d["candidate_recipe_sha256"],
        "worker_tree_sha256": verify_code(
            source,
            located(store, pins["topology"]),
            {"source_sha256": snapshots["topology"], "gsplat": "1.5.3"},
        ),
        "worker_zip_sha256": sha(source),
        "original_summary_sha256": sha(summary),
        "raw_runs": pins,
        "actual_worker_snapshots": snapshots,
        "geometry": {
            "frame": "metres, OpenCV integer-centre calibrations; measured off-axis GPU means use pixel-edge coordinates",
            "two_layer_expected_z_m": expected_z,
            "alpha": opacity,
            "expected_display_rgb": display.tolist(),
            "actual": proj,
        },
        "density": {
            "cap": cap,
            "point_counts": [r["after"] for r in steps],
            "scale": [256, 256],
            "forced_selected_totals": ops,
            "growth_at_20000": first,
            "peak_projected_radius_px": max(r["max_radius_px"] for r in steps),
            "all_active_and_fixed_geometry_color_updated": True,
            "adam_moment_controls_passed": True,
            "full": topo,
        },
        "continuation": {
            "full": pins["continuous"],
            "head": pins["first-segment"],
            "tail": pins["continuation"],
            "issued_resume_step": fix["resume_at"],
            "endpoint": fix["end_step"],
            "selected_native_views_reproduced": same,
            "published_numeric_comparison": parity,
        },
        "resource_cutoff": {"run": pins["cutoff"], "recorded_status": "failed", "evidence": cut},
        "resources": resources,
        "cap_probe_device_capacity_bytes": device_bytes,
        "environment_source_sha256": sha(environment),
        "gpu_reserved_fraction_at_20k": cap_memory["torch_reserved_peak_bytes"] / device_bytes,
        "earlier_engineering_failures": failures,
        "generation_wall_limit_seconds": 1800,
        "measured_processor_scope": "Measured synthetic 20K and short real optimizer path; no 5000-step wall-time or memory projection.",
        "torch_device_runtime": runtime,
        "disposition": "20,000 points were rasterized and updated, the forced-cap and continuation controls passed, and the one-second watchdog produced its registered cutoff. The timings and memory values below describe these synthetic probes.",
    }


def record(a):
    from stage3_reference import perform
    from make_handoff import publish

    store, project = a.store.resolve(), a.project.resolve()
    require(
        sha(a.summary) == a.summary_sha256 and sha(a.environment) == a.environment_sha256,
        "Different engineering completion or hardware receipt",
    )
    earlier = load(a.history)["failed_engineering"] if a.history else []
    summary = load(a.summary)
    source_pin = load(a.worker_pin)
    source = read_original(store, source_pin, source_pin["member"])
    require(sha(source) == source_pin["sha256"], "Different frozen Windows source package")
    base = store / "recovery/returns" / "preflight"
    base.mkdir(exist_ok=True, parents=True)
    dst = base / (a.summary_sha256 + "-" + source_pin["sha256"] + ".json")
    if dst.exists():
        p = load(dst)
        require(sha(read_original(store, p, p["member"])) == p["sha256"], "Changed preflight selection")
        return p
    # The original failed watchdog is embedded by seal and explicitly referenced in its manifest.
    # The recorder accepts successful scientific parents only.
    pins = {"source": {k: source_pin[k] for k in ("run_id", "manifest_sha256")}}
    for name, row in summary["runs"].items():
        if row["status"] == "succeeded":
            pins[name] = {"run_id": row["run"], "manifest_sha256": row["manifest_sha256"]}
    cfg = {
        "experiment_id": "recovery-windows-resource-readback-v1",
        "trial_id": "recovery-resource-" + a.summary_sha256[:14],
        "summary_sha256": a.summary_sha256,
        "worker_zip_sha256": source_pin["sha256"],
        "worker_pin": source_pin,
        "environment_sha256": a.environment_sha256,
        "earlier_engineering": earlier,
        "engineering_failed_original": {
            "run_id": summary["runs"]["cutoff"]["run"],
            "manifest_sha256": summary["runs"]["cutoff"]["manifest_sha256"],
        },
        "recording": {"expected_output_bytes": 200000},
    }
    inputs = {
        "summary": (a.summary.resolve(), None),
        "worker": (source, "source"),
        "environment": (a.environment.resolve(), None),
    }
    scratch = store.parent / "recovery-private-scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as t:
        engine = Path(t) / "engine"
        for name in (
            "recovery_preflight_readback.py",
            "recovery_inputs.py",
            "recovery_outputs.py",
            "recovery_receive.py",
            "stage3_reference.py",
            "make_handoff.py",
            "formal_lineage.py",
        ):
            copy_script(project, engine, name)
        pin = perform(
            store,
            project,
            engine,
            cfg,
            "archive_verify",
            inputs,
            pins,
            [
                "{python}",
                "{source}/scripts/recovery_preflight_readback.py",
                "worker",
                "--store",
                str(store),
                "--summary",
                "{input:summary}",
                "--source",
                "{input:worker}",
                "--environment",
                "{input:environment}",
                "--config",
                "{config}",
                "--out",
                "{out}",
            ],
        )
    m = "payload/resource-probe.json"
    out = {**pin, "member": m, "sha256": sha(read_original(store, pin, m))}
    publish(dst, encoded(out))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="action", required=True)
    q = subs.add_parser("record")
    for n in ("store", "project", "summary", "worker-pin", "environment"):
        q.add_argument("--" + n, type=Path, required=True)
    q.add_argument("--summary-sha256", required=True)
    q.add_argument("--environment-sha256", required=True)
    q.add_argument("--history", type=Path)
    q = subs.add_parser("worker")
    for n in ("store", "summary", "source", "config", "out", "environment"):
        q.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    if a.action == "record":
        print(json.dumps(record(a)))
    else:
        c = load(a.config)
        require(
            sha(a.summary) == c["summary_sha256"]
            and sha(a.source) == c["worker_zip_sha256"]
            and sha(a.environment) == c["environment_sha256"],
            "Changed staged engineering inputs",
        )
        result = verify_probe(a.store, a.summary, a.source, a.environment, c["earlier_engineering"])
        (a.out / "resource-probe.json").write_bytes(encoded(result))
