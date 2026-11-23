"""Dispatch or receive Stage B R1/Q1 from the opened family's exact sealed B0 selection."""

import argparse
import fcntl
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

from evidence_lens._bundle import HASH, LABEL, RUN_ID, encoded, load, local_path, sha
from evidence_lens._verify import require
from evidence_lens.archive import receive

EXPERIMENT = "stage2-heldout-conditions"
CONDITIONS = ("R1", "Q1")
TARGETS = tuple(f"v{i:02d}" for i in range(1, 7))
HERE = Path(__file__).resolve().parent


def study_context(project, store, registry, expected, family, condition):
    from select_trial import chain_ids, scientific_gates, transfer_evidence
    from stage2_process import context, handoff

    require(condition in CONDITIONS, "Unknown Stage B condition")
    # context reads only the catalog until the one cohort opening authenticates.
    ctx = context(project, store, registry, expected, family, project / ".venv-cpu/bin/python")
    row = ctx["row"]
    require(
        row["split"] == "test" and condition in row["trials"] and ctx["opening"] is not None,
        "Both Stage B stress conditions require an opened registered test family",
    )
    refmeta, b0_dispatch = handoff(ctx)
    opening = ctx["opening"]
    attestation = {
        k: opening[k] for k in ("catalog_sha256", "freeze_run_id", "freeze_manifest_sha256", "method_sha256")
    }
    attestation["opening_sha256"] = sha(ctx["opening_path"])
    method_file = store / "runs" / opening["freeze_run_id"] / opening["method_member"]
    require(sha(method_file) == opening["method_sha256"], "Opened method changed")
    trial = row["trials"]["B0"]
    choice = store / "trials" / trial / "trial_selection.json"
    b0 = load(choice)
    require(
        b0["schema"] == "el-p2-trial-selection-1"
        and b0["selection_status"] == "selected"
        and b0["experiment_id"] == "stage2-single-image"
        and b0["condition_id"] == "B0"
        and all(b0[k] == row[k] for k in ("trial_id", "scene_id", "scene_family_id")),
        "Wrong B0 baseline",
    )
    runs = ctx["verified"]
    ids = {k: p["run_id"] for k, p in b0["selected"].items()}
    records = {rid: runs.get(rid) for rid in ids.values()}
    require(
        chain_ids(records, ids["recompute"]) == ids
        and all(records[r]["manifest_sha256"] == b0["selected"][k]["manifest_sha256"] for k, r in ids.items())
        and all(ids[k] == b0_dispatch[k]["run_id"] for k in ("reference", "readback")),
        "Wrong B0 native reference or direct scientific graph",
    )
    identity = {**row, "dataset_hash": expected, "dataset_id": ctx["catalog"]["dataset_id"]}
    for k, rid in ids.items():
        meta = records[rid]["run"]
        require(
            all(
                meta[f] == identity[f]
                for f in (
                    "dataset_id",
                    "dataset_hash",
                    "split",
                    "scene_family_id",
                    "scene_id",
                    "scene_version",
                    "trial_id",
                )
            )
            and meta["experiment_id"]
            == ("stage2-reference" if k in ("reference", "readback") else "stage2-single-image"),
            "Baseline run was selected from another experiment",
        )
    for role in ("score", "evaluation", "recompute"):
        run = records[ids[role]]
        cfg = load(store / "runs" / ids[role] / "config.resolved.yaml")
        require(
            cfg["method_freeze_sha256"] == opening["method_sha256"]
            and cfg["test_opening_sha256"] == attestation["opening_sha256"]
            and any(
                f["sha256"] == attestation["opening_sha256"]
                for i in run["inputs"]["inputs"]
                for f in i["files"]
            ),
            "A held-out B0 stage did not bind the exact cohort opening",
        )
    evidence = scientific_gates(store, records, ids, transfer_evidence(store, records), b0["experiment_id"])
    require(
        all(b0["evidence"][k] == value for k, value in evidence.items() if k != "transfer_receipts")
        and refmeta["blend_sha256"] == evidence["scene_sha256"],
        "B0 scientific evidence differs",
    )
    ctx.update(
        family=identity,
        condition=condition,
        quality_experiment=EXPERIMENT,
        quality_trial=row["trials"][condition],
        baseline=b0,
        b0_ids=ids,
        b0_selection_sha256=sha(choice),
        b0_dispatch=b0_dispatch,
        worker_source=worker_source,
        q1_render_response=q1_render_response,
        opening_evidence=attestation,
        method_file=method_file,
        release={
            "method_freeze_sha256": opening["method_sha256"],
            "test_opening_sha256": attestation["opening_sha256"],
            "stage2_method_freeze_run_id": opening["freeze_run_id"],
        },
    )
    return ctx


def worker_source(ctx):
    """Recover exact B0 generator/renderer bytes from the sent B0 ZIP. Add only the stress worker."""
    from make_handoff import publish, sha_bytes, zip_bytes

    dispatch = ctx["b0_dispatch"]
    family = ctx["family"]["scene_family_id"]
    directory = ctx["store"] / "stage2" / ctx["catalog_hash"] / "dispatches"
    candidate = next(
        (p for p in sorted(directory.glob("*/" + family + ".zip")) if sha(p) == dispatch["container_sha256"]),
        None,
    )
    require(candidate is not None, "Dispatched B0 source ZIP for this family is unavailable")
    with zipfile.ZipFile(candidate) as archive:
        manifest = archive.read("handoff.json")
        require(sha_bytes(manifest) == dispatch["handoff_sha256"], "B0 source activation differs")
        job = json.loads(manifest)
        contents = {local_path(k[5:]): archive.read(k) for k in job["files"] if k.startswith("code/")}
        require(
            all(sha_bytes(val) == job["files"]["code/" + key] for key, val in contents.items())
            and sha_bytes(zip_bytes(contents)) == dispatch["worker_source_sha256"],
            "Baseline worker code has changed",
        )
    for name in ("condition_stage.py", "inspect_sharp_ply.py", "stress_condition.py"):
        path = (HERE / name) if name == "condition_stage.py" else ctx["project"] / "scripts" / name
        contents["scripts/" + name] = path.read_bytes()
    raw = zip_bytes(contents)
    root = ctx["store"] / "formal" / ctx["catalog_hash"] / "conditions"
    publish(root / "worker-source.zip", raw)
    publish(
        root / "worker-source.json",
        encoded(
            {
                "schema": "el-stage2-condition-worker-1",
                "sha256": sha_bytes(raw),
                "b0_worker_sha256": dispatch["worker_source_sha256"],
                "member_sha256": {k: sha_bytes(v) for k, v in contents.items()},
            }
        ),
    )
    return {"code/" + k: v for k, v in contents.items()}, sha_bytes(raw)


def start_dispatch(args):
    from condition_jobs import make_job
    from condition_mac import prepare
    from make_handoff import publish
    from run_windows_batch import deliver

    require(
        LABEL.fullmatch(args.batch)
        and len(set(args.families)) == len(args.families)
        and len(set(args.conditions)) == len(args.conditions),
        "Invalid batch, duplicate family or duplicate condition",
    )
    root, store = args.project, args.store
    # Resolve every selection before creating a batch or performing a derivation.
    contexts = [
        study_context(root, store, args.catalog, args.catalog_sha256, f, c)
        for f in args.families
        for c in args.conditions
    ]
    folder = store / "stage2" / args.catalog_sha256 / "condition-batches" / args.batch
    folder.mkdir(parents=True, exist_ok=True)
    jobs = []
    for ctx in contexts:
        prepared = prepare(ctx, args.retry_of).get("run_id") if ctx["condition"] == "R1" else None
        jobs.append(make_job(ctx, prepared, None, folder / (ctx["quality_trial"] + ".zip")))
    runner = root / "scripts/run_conditions_batch.py"
    transport = root / "scripts/run_windows_batch.py"
    batch = {
        "schema": "el-windows-condition-batch-1",
        "batch_id": args.batch,
        "runner_sha256": sha(runner),
        "transport_sha256": sha(transport),
        "catalog_sha256": args.catalog_sha256,
        "opening": contexts[0]["opening_evidence"],
        "jobs": jobs,
    }
    meta = folder / "batch.json"
    publish(meta, encoded(batch))
    dest = args.exchange / "Jobs" / args.batch
    (args.exchange / "Returns" / args.batch).mkdir(parents=True, exist_ok=True)
    for job in jobs:
        deliver(folder / job["name"], dest / job["name"])
    for file in (runner, transport, meta):
        deliver(file, dest / file.name)
    return {
        "batch": args.batch,
        "batch_sha256": sha(meta),
        "runner_sha256": sha(runner),
        "transport_sha256": sha(transport),
        "trials": [j["trial_id"] for j in jobs],
    }


def verify_dispatch(ctx):
    from formal_lineage import run_pin
    from make_handoff import sha_bytes

    path = (
        ctx["store"] / "formal" / ctx["catalog_hash"] / "conditions/dispatch" / (ctx["quality_trial"] + ".json")
    )
    dispatch = load(path)
    job = dispatch["job"]
    require(
        job["schema"] == "el-stage2-condition-handoff-1"
        and job["opening"] == ctx["opening_evidence"]
        and sha_bytes(encoded(job)) == dispatch["handoff_sha256"]
        and job["method_freeze_sha256"] == ctx["release"]["method_freeze_sha256"]
        and job["b0_selection_sha256"] == ctx["b0_selection_sha256"]
        and job["baseline_trial_id"] == ctx["family"]["trial_id"]
        and job["condition"] == ctx["condition"]
        and job["experiment_id"] == EXPERIMENT
        and job["trial_id"] == ctx["quality_trial"]
        and job["timing_repeat"] is None
        and job["warmup"] is None,
        "Stage B condition dispatch or gate differs",
    )
    for old, key in (
        ("reference", "reference"),
        ("readback", "readback"),
        ("generation", "baseline_generation"),
    ):
        require(run_pin(ctx["verified"].get(ctx["b0_ids"][old])) == job[key], "Dispatched B0 parent differs")
    require(
        all(
            job["identity"][k] == ctx["family"][k]
            for k in ("scene_family_id", "scene_id", "scene_version", "dataset_id", "dataset_hash", "split")
        )
        and job["identity"]["condition_id"] == ctx["condition"]
        and job["identity"]["trial_id"] == ctx["quality_trial"],
        "Condition job identity differs from the registered family",
    )
    if ctx["condition"] == "R1":
        from condition_jobs import validate_r1

        require(
            run_pin(validate_r1(ctx, job["source"]["run_id"])) == job["source"], "Dispatched R1 source differs"
        )
    else:
        require(job["source"] == job["reference"], "Q1 source must be the original B0 input")
    return dispatch


def q1_render_response(ctx, ids):
    """Compare all seven real rerenders, preserving tiny effects and any geometry deviations."""
    import numpy as np
    from evidence_lens.features import extract_features
    from evidence_lens.formal_evaluation import contribution_changes, field_change
    from evidence_lens.l1 import columns
    from evidence_lens.risk_model import score_features
    from evidence_lens.scoring_io import ReadAudit, load_cameras, load_source_png, rgb_valid

    base, current = ctx["b0_ids"], ids
    root = ctx["store"] / "runs"
    ref = root / current["reference"] / "payload"
    audit = ReadAudit()
    cameras = load_cameras(ref / "cameras.json", audit)
    input_rgb = load_source_png(ref / "input.png", cameras["source"], audit)
    geometry = (
        "l0",
        "l0_reason",
        "l0_valid",
        "alpha_score",
        "alpha_score_valid",
        "scoreable",
        "frame_relation",
        *columns("L1-geometry")[1],
    )
    result = {}
    for view in ("source", *TARGETS):

        def data(chain):
            with np.load(root / chain["render"] / "payload" / (view + ".npz"), allow_pickle=False) as f:
                prediction = {k: f[k] for k in ("rgb", "z", "alpha")}
            if view == "source":
                features = extract_features(
                    input_rgb, prediction, prediction, cameras[view], cameras[view], True
                )
                rules = {**features, **score_features(features, True)}
            else:
                with np.load(root / chain["score"] / "payload" / (view + ".npz"), allow_pickle=False) as f:
                    rules = {k: f[k] for k in geometry}
            return prediction, rules

        before, b_rules = data(base)
        after, a_rules = data(current)
        good = rgb_valid(before["rgb"]) & rgb_valid(after["rgb"])
        delta = after["rgb"][good].astype(np.float64) - before["rgb"][good].astype(np.float64)
        mae = abs(delta).mean(axis=-1)
        fields = {k: field_change(before[k], after[k]) for k in ("z", "alpha")}
        fields.update(contribution_changes(before["alpha"], after["alpha"]))
        fields.update({k: field_change(b_rules[k], a_rules[k]) for k in geometry})
        result[view] = {
            "joint_valid_rgb": int(len(mae)),
            "rgb_mae_sum": float(mae.sum()),
            "max_pixel_rgb_mae": float(mae.max()) if len(mae) else None,
            "pixels_rgb_mae_strictly_positive": int(np.count_nonzero(mae > 0)),
            "pixels_rgb_mae_strictly_over_0_05": int(np.count_nonzero(mae > 0.05)),
            "geometry_bitwise_equal": all(x["bitwise_different"] == 0 for x in fields.values()),
            "geometry": fields,
        }
    return {
        "schema": "el-stage2-q1-render-response-1",
        "baseline_render": base["render"],
        "condition_render": current["render"],
        "baseline_score": base["score"],
        "condition_score": current["score"],
        "scale": "native floating-point display sRGB [0,1]; per-pixel channel MAE",
        "source_rule_origin": "Public rules applied source-to-itself; separate from target quality metrics",
        "views": result,
    }


def worker_envelope(ctx, record, dispatch):
    from formal_lineage import run_pin

    meta = record["run"]
    job = dispatch["job"]
    kind = meta["run_kind"]
    require(
        kind in (("generation", "render") if ctx["condition"] == "R1" else ("derivation", "render"))
        and meta["experiment_id"] == EXPERIMENT
        and all(meta.get(k) == v for k, v in job["identity"].items()),
        "Wrong worker family, condition or split",
    )
    cfg = load(ctx["store"] / "runs" / meta["run_id"] / "config.resolved.yaml")
    require(
        cfg.get("stage2_opening") == ctx["opening_evidence"]
        and cfg["formal_handoff_sha256"] == dispatch["handoff_sha256"]
        and cfg["b0_selection_sha256"] == ctx["b0_selection_sha256"]
        and cfg["method_freeze_sha256"] == ctx["release"]["method_freeze_sha256"]
        and cfg["worker_source_sha256"] == job["worker_source_sha256"]
        and cfg["timing_repeat"] is None
        and all(cfg.get(k) == v for k, v in job["identity"].items()),
        "Worker settings differ from the opened handoff",
    )
    parents = record["inputs"]["parents"]
    if kind == "render":
        candidates = [p for p in parents if p != job["source"]]
        require(len(parents) == 2 and job["source"] in parents and len(candidates) == 1, "Wrong render parents")
        pin = candidates[0]
        actual = ctx["verified"].get(pin["run_id"], "generation" if ctx["condition"] == "R1" else "derivation")
        require(run_pin(actual) == pin, "Render consumed another producer manifest")
        worker_envelope(ctx, actual, dispatch)
    else:
        require(
            parents == [job["source"] if kind == "generation" else job["baseline_generation"]],
            "Derivation/generation did not consume the selected exact source",
        )


def returned(ctx, folder, dispatch):
    from select_trial import transfer_evidence
    from formal_lineage import run_pin

    records = {}
    envelopes = []
    for p in sorted(folder.glob("*.receipt.json")):
        receipt = load(p)
        if receipt.get("trial_id") != ctx["quality_trial"]:
            continue
        rid = receipt["run"]
        require(
            RUN_ID.fullmatch(rid)
            and p.name == rid + ".receipt.json"
            and receipt["package"] == rid + ".zip"
            and receipt["schema"] == "el-worker-return-1"
            and receipt["handoff_sha256"] == dispatch["handoff_sha256"]
            and receipt["run_kind"]
            in (("generation", "render") if ctx["condition"] == "R1" else ("derivation", "render"))
            and HASH.fullmatch(receipt["manifest_sha256"])
            and HASH.fullmatch(receipt["sha256"]),
            "Malformed condition return",
        )
        envelopes.append(receipt)
    for r in sorted(envelopes, key=lambda r: (r["run_kind"] == "render", r["run"])):
        rid = r["run"]
        z = folder / r["package"]
        require(
            z.is_file() and not z.is_symlink() and z.stat().st_size == r["bytes"] and sha(z) == r["sha256"],
            "Return ZIP changed",
        )
        pin = {"run_id": rid, "manifest_sha256": r["manifest_sha256"]}
        if not (ctx["store"] / "runs" / rid).is_dir() or not transfer_evidence(ctx["store"], {rid: pin}).get(
            rid
        ):
            result = receive(z, ctx["store"], r["sha256"], ctx["project"] / "configs/storage.json")
            require(
                result["status"] in {"accepted", "identical"}
                and result["manifest_sha256"] == pin["manifest_sha256"]
                and result["run_id"] == rid,
                "Transfer rejected: " + str(result.get("reason")),
            )
        record = ctx["verified"].get(rid)
        require(
            run_pin(record) == pin
            and all(
                record["run"][k] == r[v]
                for k, v in (
                    ("run_kind", "run_kind"),
                    ("execution_status", "status"),
                    ("exit_code", "exit_code"),
                )
            ),
            "Return envelope contradicts its original recorder package",
        )
        worker_envelope(ctx, record, dispatch)
        records[rid] = record
    return records


def terminal_disposition(ctx, dispatch, pin):
    from evidence_lens.recorder import worker_started
    from formal_lineage import run_pin
    from make_handoff import publish
    from select_trial import transfer_evidence

    checked = ctx["verified"].get(pin["run_id"])
    run = checked["run"]
    worker_envelope(ctx, checked, dispatch)
    receipts = transfer_evidence(ctx["store"], {pin["run_id"]: checked})
    require(
        run_pin(checked) == pin
        and receipts.get(pin["run_id"])
        and run["execution_status"] in {"failed", "cancelled", "interrupted"}
        and run["failure_reason"]
        and worker_started(ctx["store"] / "runs" / pin["run_id"], run),
        "Terminal condition requires a genuinely failed, received scientific execution",
    )
    for p in (ctx["store"] / "runs").glob("*/run.json"):
        other = load(p)
        if (other["experiment_id"], other["trial_id"], other["run_kind"]) == (
            EXPERIMENT,
            ctx["quality_trial"],
            run["run_kind"],
        ):
            require(
                other["execution_status"] != "succeeded"
                and not (other["retry_of_run_id"] == pin["run_id"] and worker_started(p.parent, other)),
                "A successful attempt or later linked scientific retry exists",
            )
    out = {
        "schema": "el-vision-failed-trial-1",
        "selection_status": "failed",
        "planned_targets": list(TARGETS),
        "trial_id": ctx["quality_trial"],
        "scene_id": ctx["family"]["scene_id"],
        "scene_family_id": ctx["family"]["scene_family_id"],
        "experiment_id": EXPERIMENT,
        "condition_id": ctx["condition"],
        "split": "test",
        "catalog_sha256": ctx["catalog_hash"],
        "reference": dispatch["job"]["reference"],
        "readback": dispatch["job"]["readback"],
        "failure": pin,
        "reason": run["failure_reason"],
        "opening": ctx["opening_evidence"],
        "b0_selection_sha256": ctx["b0_selection_sha256"],
        "method_freeze_sha256": ctx["release"]["method_freeze_sha256"],
        "handoff_sha256": dispatch["handoff_sha256"],
    }
    dest = ctx["store"] / "trials" / ctx["quality_trial"] / "trial_selection.json"
    publish(dest, encoded(out))
    return {"trial_id": ctx["quality_trial"], "selection": str(dest), "sha256": sha(dest), "status": "failed"}


def process(args):
    from condition_mac import run_stage
    from formal_lineage import run_pin
    from select_trial import transfer_evidence

    ctx = study_context(
        args.project, args.store, args.catalog, args.catalog_sha256, args.family, args.condition
    )
    dispatch = verify_dispatch(ctx)
    lock = (
        ctx["store"]
        / "stage2"
        / ctx["catalog_hash"]
        / "condition-processing"
        / (ctx["quality_trial"] + ".lock")
    )
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        records = returned(ctx, args.returns_dir, dispatch) if args.returns_dir else {}
        if args.terminal_failure:
            return terminal_disposition(ctx, dispatch, args.terminal_failure)
        require((args.producer is None) == (args.render is None), "Supply both explicit worker pins")
        if args.producer:
            records = {}
            for p in (args.producer, args.render):
                r = ctx["verified"].get(p["run_id"])
                require(run_pin(r) == p, "Explicit worker pin differs")
                worker_envelope(ctx, r, dispatch)
                records[p["run_id"]] = r
        live = sorted(
            (r for r in records.values() if r["run"]["execution_status"] == "succeeded"),
            key=lambda r: (r["run"]["started_at_utc"], r["run"]["ended_at_utc"], r["run"]["run_id"]),
        )
        p = next((r for r in live if r["run"]["run_kind"] != "render"), None)
        require(p is not None, "Successful condition producer has not returned")
        producer = p["run"]["run_id"]
        r = next(
            (r for r in live if r["run"]["run_kind"] == "render" and producer in r["run"]["parent_run_ids"]),
            None,
        )
        require(r is not None, "Successful condition render with that producer has not returned")
        render = r["run"]["run_id"]
        require(
            set(transfer_evidence(ctx["store"], {producer: p, render: r})) == {producer, render},
            "Mac transfer receipts are missing",
        )
        a = SimpleNamespace(
            producer_run=producer,
            render_run=render,
            preparation_run=dispatch["job"]["source"]["run_id"] if ctx["condition"] == "R1" else None,
            score_run=None,
            evaluation_run=None,
            recompute_run=None,
            retry_of=None,
            cpu_python=args.cpu_python or args.project / ".venv-cpu/bin/python",
        )
        dest = ctx["store"] / "trials" / ctx["quality_trial"] / "trial_selection.json"
        if dest.exists():
            require(
                load(dest)["selection_status"] == "selected", "An immutable failure selection already exists"
            )
        for stage, field in (
            ("score", "score_run"),
            ("evaluate", "evaluation_run"),
            ("verify", "recompute_run"),
            ("select", None),
        ):
            a.stage = stage
            a.retry_of = args.retry_of if stage == args.retry_stage else None
            result = run_stage(ctx, a)
            if field is not None:
                setattr(a, field, result["run_id"])
        return result


def main(argv=None):
    from stage2_process import supplied_pin

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("dispatch", "process"))
    for f in ("project", "store", "catalog"):
        p.add_argument("--" + f, type=Path, required=True)
    p.add_argument("--catalog-sha256", required=True)
    p.add_argument("--families", nargs="+")
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument("--batch")
    p.add_argument("--exchange", type=Path)
    p.add_argument("--family")
    p.add_argument("--condition", choices=CONDITIONS)
    p.add_argument("--returns-dir", type=Path)
    p.add_argument("--producer", type=supplied_pin)
    p.add_argument("--render", type=supplied_pin)
    p.add_argument("--terminal-failure", type=supplied_pin)
    p.add_argument("--cpu-python", type=Path)
    p.add_argument("--retry-stage", choices=("prepare", "score", "evaluate", "verify"))
    p.add_argument("--retry-of")
    a = p.parse_args(argv)
    a.project = a.project.resolve()
    a.store = a.store.resolve()
    # The overlay and the promoted script use the same concrete source paths.
    sys.path.insert(0, str(HERE))
    sys.path.insert(1, str(a.project / "scripts"))
    require(
        bool(a.retry_of) == bool(a.retry_stage) and (not a.retry_of or RUN_ID.fullmatch(a.retry_of)),
        "A Mac retry identifies both the failed recorder run and its exact stage",
    )
    if a.action == "dispatch":
        require(
            not a.retry_of
            or (a.retry_stage == "prepare" and a.conditions == ["R1"] and a.families and len(a.families) == 1),
            "Retry one named R1 source preparation per dispatch",
        )
        require(a.families and a.batch and a.exchange, "Dispatch requires --families, --batch and --exchange")
        result = start_dispatch(a)
    else:
        require(
            a.family and a.condition and (a.returns_dir or a.producer or a.terminal_failure),
            "Processing requires family, condition and explicit worker evidence",
        )
        require(a.retry_stage != "prepare", "A preparation retry belongs to its R1 dispatch")
        require(
            not (a.terminal_failure and (a.producer or a.render)),
            "Select a terminal failure or a successful chain",
        )
        result = process(a)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
