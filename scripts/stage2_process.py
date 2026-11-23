"""Receive one Stage B baseline trial, verify its direct scientific parents, score and independently recompute v01."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from evidence_lens._bundle import HASH, RUN_ID, encoded, load, sha
from evidence_lens._verify import require
from evidence_lens.archive import receive

EXP = "stage2-single-image"
CAMERAS = ("source", *(f"v{i:02d}" for i in range(1, 7)))
TARGETS = CAMERAS[1:]
CPU_SETTINGS = "7f6c4b2fe22d8e8c6fa643fb8b77432320b985e55d3d14dff80ca3a77b7de2de"


def supplied_pin(raw):
    rid, sep, hash_ = raw.partition("=")
    require(bool(sep and RUN_ID.fullmatch(rid) and HASH.fullmatch(hash_)), "Expected run_id=manifest_sha256")
    return {"run_id": rid, "manifest_sha256": hash_}


def handoff(ctx):
    from formal_lineage import reference_identity, run_pin

    row, store = ctx["row"], ctx["store"]
    reference = load(store / "formal-references" / ctx["catalog_hash"] / (row["scene_family_id"] + ".json"))
    require(
        reference["schema"] == "el-formal-reference-selection-1"
        and reference["dataset_hash"] == ctx["catalog_hash"]
        and all(
            reference[k] == row[k]
            for k in (
                "trial_id",
                "scene_family_id",
                "split",
                "cv_fold",
                "recipe_sha256",
                "pair_structure_sha256",
            )
        ),
        "Reference selection conflicts with Stage B roster",
    )
    meta, ref, rb = reference_identity(
        ctx["verified"], reference["reference_run_id"], reference["readback_run_id"]
    )
    for stage, prefix in ((ref, "reference"), (rb, "readback")):
        run = stage["run"]
        require(
            run["experiment_id"] == "stage2-reference"
            and run["condition_id"] == "B0"
            and run["dataset_hash"] == ctx["catalog_hash"]
            and all(run[k] == row[k] for k in ("trial_id", "scene_family_id", "scene_id", "split"))
            and run["dataset_id"] == ctx["catalog"]["dataset_id"]
            and stage["manifest_sha256"] == reference[prefix + "_manifest_sha256"],
            "Unexpected B reference or readback identity",
        )
    require(
        meta["formal"]["recipe_sha256"] == row["recipe_sha256"]
        and meta["camera_sha256"] == row["camera_sha256"]
        and meta["formal"]["structural_fingerprint"] == row["pair_structure_sha256"],
        "Incorrect native geometry or cameras",
    )
    dispatch = load(store / "formal" / ctx["catalog_hash"] / "dispatch" / (row["trial_id"] + ".json"))
    require(
        dispatch["schema"] == "el-formal-dispatch-1"
        and dispatch["reference"] == run_pin(ref)
        and dispatch["readback"] == run_pin(rb)
        and all(
            dispatch["identity"][k] == row[k]
            for k in ("trial_id", "scene_family_id", "scene_id", "scene_version", "split")
        )
        and dispatch["identity"]["scene_hash"] == meta["blend_sha256"]
        and dispatch["identity"]["camera_set_hash"] == meta["camera_sha256"]
        and dispatch["identity"]["dataset_hash"] == ctx["catalog_hash"]
        and dispatch["identity"]["condition_id"] == "B0",
        "Windows handoff differs from registered selection",
    )
    return meta, dispatch


def collect_returns(ctx, folder):
    """Authenticate every returned attempt matching this trial, including failures; accept dependency order."""
    from select_trial import transfer_evidence

    dispatch = ctx["dispatch"]
    trial = ctx["row"]["trial_id"]
    store = ctx["store"]
    receipts = []
    for path in sorted(folder.glob("*.receipt.json")):
        item = load(path)
        if item.get("trial_id") != trial:
            continue
        rid = item["run"]
        require(
            RUN_ID.fullmatch(rid)
            and path.name == rid + ".receipt.json"
            and item["schema"] == "el-worker-return-1"
            and item["handoff_sha256"] == dispatch["handoff_sha256"]
            and item["package"] == rid + ".zip"
            and HASH.fullmatch(item["sha256"])
            and HASH.fullmatch(item["manifest_sha256"])
            and item["run_kind"] in {"generation", "render"},
            "Invalid worker return receipt",
        )
        receipts.append(item)
    checked = {}
    for kind in ("generation", "render"):
        # Recorder run IDs begin with the UTC timestamp. Retry parents of a given kind precede their children.
        for item in sorted((x for x in receipts if x["run_kind"] == kind), key=lambda x: x["run"]):
            rid = item["run"]
            bundle = folder / item["package"]
            require(
                bundle.is_file()
                and not bundle.is_symlink()
                and bundle.stat().st_size == item["bytes"]
                and sha(bundle) == item["sha256"],
                "Worker return container has a wrong length or digest",
            )
            existing = store / "runs" / rid
            if not existing.is_dir() or not transfer_evidence(
                store, {rid: {"manifest_sha256": item["manifest_sha256"]}}
            ).get(rid):
                received = receive(bundle, store, item["sha256"], ctx["project"] / "configs/storage.json")
                require(
                    received["status"] in {"accepted", "identical"}
                    and received["run_id"] == rid
                    and received["manifest_sha256"] == item["manifest_sha256"],
                    "Mac receiver disposition: " + str(received.get("reason") or received["status"]),
                )
            record = ctx["verified"].get(rid)
            require(
                record["manifest_sha256"] == item["manifest_sha256"]
                and record["run"]["execution_status"] == item["status"]
                and record["run"]["run_kind"] == kind
                and record["run"]["exit_code"] == item["exit_code"],
                "Returned recorder seal differs",
            )
            ref = dispatch["reference"]
            generation = None
            if kind == "render":
                gen_parents = [p for p in record["inputs"]["parents"] if p != ref]
                require(len(gen_parents) == 1, "Returned render lacks its sole PLY producer")
                generation = gen_parents[0]
                gen_record = ctx["verified"].get(generation["run_id"], "generation")
                validate_stage(ctx, gen_record, "generation", ref)
            validate_stage(ctx, record, kind, ref, generation)
            checked[rid] = record
    return checked


def validate_stage(ctx, run, kind, ref, gen=None):
    dispatch = ctx["dispatch"]
    row = ctx["row"]
    meta = run["run"]
    store = ctx["store"]
    require(
        meta["run_kind"] == kind
        and meta["experiment_id"] == EXP
        and all(meta.get(k) == v for k, v in dispatch["identity"].items()),
        "Returned stage belongs to another family, split, or condition",
    )
    settings = load(store / "runs" / meta["run_id"] / "config.resolved.yaml")
    require(
        settings["formal_handoff_sha256"] == dispatch["handoff_sha256"]
        and settings["worker_source_sha256"] == dispatch["worker_source_sha256"]
        and settings.get("upstream_reference") == ref
        and settings.get("independent_readback") == dispatch["readback"]
        and all(settings.get(k) == v for k, v in dispatch["identity"].items()),
        "Worker executed another handoff",
    )
    parents = {p["run_id"]: p["manifest_sha256"] for p in run["inputs"]["parents"]}
    require(
        parents == {p["run_id"]: p["manifest_sha256"] for p in (ref, *([gen] if kind == "render" else []))},
        "Worker lineage must bind exactly the selected source and PLY producer",
    )

    def required_inputs(owner, hashes):
        actual = {
            f["sha256"]
            for asset in run["inputs"]["inputs"]
            if asset["parent_run_id"] == owner["run_id"]
            for f in asset["files"]
        }
        require(set(hashes) <= actual, "A directly parented, exact worker input is missing")

    if meta["execution_status"] == "succeeded":
        required_inputs(ref, [ctx["refmeta"]["camera_sha256"], ctx["refmeta"]["input_sha256"]])
        if kind == "render":
            g = ctx["verified"].get(gen["run_id"], "generation")
            required_inputs(
                gen, [g["artifacts"]["payload/" + n]["sha256"] for n in ("generation.json", "scene.ply")]
            )
    require(meta["trial_id"] == row["trial_id"], "Incorrect trial ID")


def windows(ctx, rows, requested=None):
    from formal_lineage import run_pin
    from select_trial import transfer_evidence

    ref = ctx["dispatch"]["reference"]
    ready = {}
    if requested:
        rows = {}
        for kind, p in requested.items():
            rec = ctx["verified"].get(p["run_id"], kind)
            require(run_pin(rec) == p, "Requested scientific pin differs")
            rows[p["run_id"]] = rec
    transfer = transfer_evidence(ctx["store"], rows)
    ordered = sorted(
        rows.values(), key=lambda r: (r["run"]["started_at_utc"], r["run"]["ended_at_utc"], r["run"]["run_id"])
    )
    for kind in ("generation", "render"):
        options = [
            r for r in ordered if r["run"]["run_kind"] == kind and r["run"]["execution_status"] == "succeeded"
        ]
        if kind == "render":
            options = [r for r in options if ready["generation"]["run_id"] in r["run"]["parent_run_ids"]]
        require(
            bool(options),
            f"No successful {kind} for this trial has returned. Its sealed attempts are retained.",
        )
        ck = options[0]
        rid = ck["run"]["run_id"]
        require(
            bool(transfer.get(rid)), "Original successful Windows bundle needs a Mac accepted/identical receipt"
        )
        validate_stage(ctx, ck, kind, ref, ready.get("generation"))
        ready[kind] = run_pin(ck)
    pred = load(ctx["store"] / "runs" / ready["render"]["run_id"] / "payload/prediction.json")
    gen = load(ctx["store"] / "runs" / ready["generation"]["run_id"] / "payload/generation.json")
    pr = rows[ready["render"]["run_id"]]
    gr = rows[ready["generation"]["run_id"]]
    calibr = load(ctx["store"] / "runs" / ready["render"]["run_id"] / "payload/depth-probe.json")
    require(
        gen["schema"] == "el-generation-1"
        and gen["producer_run_id"] == ready["generation"]["run_id"]
        and gen["gaussian_count"] > 0
        and gen["checkpoint_sha256"]
        == load(ctx["project"] / "configs/windows-b0.json")["generator"]["checkpoint_sha256"]
        and gen["checkpoint_sha256"] in {s["sha256"] for s in gr["shared"]},
        "Wrong checkpoint or output generation",
    )
    require(
        pred["producer_run_id"] == ready["render"]["run_id"]
        and pred["generation_run_id"] == ready["generation"]["run_id"]
        and pred["ply_sha256"] == gen["ply_sha256"] == gr["artifacts"]["payload/scene.ply"]["sha256"]
        and all(pred[k] == gen[k] == ctx["refmeta"][k] for k in ("input_sha256", "camera_sha256"))
        and set(pred["views"]) == set(CAMERAS)
        and not pred.get("failed_views")
        and pred["depth_validation"] == calibr
        and pred["depth_validation_receipt_sha256"] == pr["artifacts"]["payload/depth-probe.json"]["sha256"]
        and all(
            calibr[k] is True
            for k in (
                "off_axis_plane_passed",
                "alpha_normalization_passed",
                "translation_passed",
                "linear_blend_passed",
            )
        )
        and calibr["empty_pixel_alpha"] == 0
        and all(
            pred["views"][v]["sha256"] == pr["artifacts"]["payload/" + v + ".npz"]["sha256"] for v in CAMERAS
        ),
        "Incomplete measured seven-camera products or failed frozen calibration",
    )
    return ready


def trial_stages(ctx, kind, parents):
    """Inspect sealed attempt envelopes of this trial. Pick by recorded start, never by output."""
    store = ctx["store"]
    rows = []
    for f in (store / "runs").glob("*/run.json"):
        meta = load(f)
        if (
            meta["trial_id"] == ctx["row"]["trial_id"]
            and meta["experiment_id"] == EXP
            and meta["run_kind"] == kind
            and set(meta["parent_run_ids"]) == set(parents)
        ):
            candidate = ctx["verified"].get(meta["run_id"])
            settings = load(f.parent / "config.resolved.yaml")
            require(
                all(settings.get(k) == v for k, v in ctx["settings"].items() if k != "recording"),
                "Matching scientific parents have conflicting Stage B settings",
            )
            rows.append(candidate)
    return sorted(
        rows, key=lambda x: (x["run"]["started_at_utc"], x["run"]["ended_at_utc"], x["run"]["run_id"])
    )


def run_cpu_stage(ctx, cfg, stages, role, temp, retry):
    from mac_trial import recompute_view, select_files

    store, project, row = ctx["store"], ctx["project"], ctx["row"]
    dirs = {k: store / "runs" / p["run_id"] for k, p in stages.items()}
    ref, pred = dirs["reference"] / "payload", dirs["render"] / "payload"
    opening = ctx["opening_path"]
    if role == "recompute":
        return recompute_view(
            store,
            project,
            cfg,
            dirs["reference"],
            dirs["readback"],
            dirs["render"],
            dirs["score"],
            dirs["evaluation"],
            row["trial_id"],
            EXP,
            retry,
            opening,
        )

    names = ["prediction.json", *[v + ".npz" for v in CAMERAS]]
    selected_predictions = select_files(pred, temp / "predictions", names)
    upstream = ["reference", "readback", "render"]
    inputs = [
        "--input",
        f"rgb={ref / 'input.png'}",
        "--input",
        f"cameras={ref / 'cameras.json'}",
        "--input",
        f"predictions={selected_predictions}",
        "--input-parent",
        "predictions=" + dirs["render"].name,
    ]
    arguments = [
        "--input",
        "{input:rgb}",
        "--cameras",
        "{input:cameras}",
        "--predictions",
        "{input:predictions}",
        "--output",
        "{out}",
        "--run-id",
        "{run_id}",
    ]
    if role == "score":
        kind = "model_score"
        worker = [str(ctx["cpu"]), "-m", "evidence_lens.risk_model", *arguments]
    else:
        kind = "evaluation"
        upstream.append("score")
        truth = select_files(
            ref,
            temp / "reference",
            ["reference.json", "validation/results.json", *[f"reference/{v}.npz" for v in TARGETS]],
        )
        scores = select_files(
            dirs["score"] / "payload",
            temp / "scores",
            ["score.json", "feature_schema.json", *[v + ".npz" for v in TARGETS]],
        )
        inputs.extend(
            [
                "--input",
                f"reference={truth}",
                "--input-parent",
                "reference=" + dirs["reference"].name,
                "--input",
                f"scores={scores}",
                "--input-parent",
                "scores=" + dirs["score"].name,
            ]
        )
        worker = [
            "{python}",
            "-m",
            "evidence_lens.evaluation",
            *arguments,
            "--references",
            "{input:reference}",
            "--scores",
            "{input:scores}",
            "--trial-id",
            row["trial_id"],
            "--scene-id",
            row["scene_id"],
            "--scene-family-id",
            row["scene_family_id"],
            "--condition-id",
            "B0",
            "--parquet",
        ]

    cmd = [
        sys.executable,
        "-m",
        "evidence_lens.recorder",
        "--root",
        str(store),
        "--source-root",
        str(project),
        "--config",
        str(cfg),
        "--experiment",
        EXP,
        "--trial",
        row["trial_id"],
        "--kind",
        kind,
    ]
    for role_name in upstream:
        pin = stages[role_name]
        cmd.extend(["--parent", pin["run_id"] + "=" + pin["manifest_sha256"]])
    if retry:
        cmd.extend(["--retry-of", retry])
    if opening:
        cmd.extend(["--input", "test-opening=" + str(opening)])
    return subprocess.run([*cmd, *inputs, "--", *worker], cwd=project).returncode


def cpu_stages(ctx, stages):
    from evidence_lens.recorder import worker_started
    from formal_lineage import run_pin

    store, project = ctx["store"], ctx["project"]
    original = project / "configs/p2-cpu.json"
    require(
        sha(original) == CPU_SETTINGS, "CPU scoring or label configuration differs from the registered source"
    )
    dispatch = ctx["dispatch"]
    settings = {**load(original), **dispatch["identity"]}
    settings.update(
        experiment_id=EXP,
        config_version="stage2-b0-mac-1",
        formal_handoff_sha256=dispatch["handoff_sha256"],
        worker_source_sha256=dispatch["worker_source_sha256"],
        catalog_sha256=ctx["catalog_hash"],
        family_recipe_sha256=ctx["row"]["recipe_sha256"],
        stage2_processor_sha256=sha(Path(__file__)),
        comparison_camera="v01",
    )
    if ctx["opening_path"]:
        settings.update(
            test_opening_sha256=sha(ctx["opening_path"]), method_freeze_sha256=ctx["opening"]["method_sha256"]
        )
    ctx["settings"] = settings
    methods = [("score", "model_score"), ("evaluation", "evaluation"), ("recompute", "archive_verify")]
    parent_roles = ["reference", "readback", "render"]
    for role, kind in methods:
        parents = {stages[k]["run_id"] for k in parent_roles}

        def attempted():
            return trial_stages(ctx, kind, parents)

        attempts = attempted()
        accepted = [record for record in attempts if record["run"]["execution_status"] == "succeeded"]
        if not accepted:
            started = [
                rec["run"]
                for rec in attempts
                if worker_started(store / "runs" / rec["run"]["run_id"], rec["run"])
            ]
            require(
                len(started) <= 1,
                f"Both scientific {role} attempts already failed. The original bundles are retained.",
            )
            retry = started[0]["run_id"] if started else None
            with tempfile.TemporaryDirectory(dir=store, prefix="stage2-mac-") as tmp:
                temp = Path(tmp)
                cfg = temp / "trial.json"
                cfg.write_bytes(encoded(settings))
                exit_code = run_cpu_stage(ctx, cfg, stages, role, temp, retry)
            attempts = attempted()
            accepted = [record for record in attempts if record["run"]["execution_status"] == "succeeded"]
            require(
                exit_code == 0 and accepted,
                f"Mac {role} stopped with exit code {exit_code}; inspect its sealed attempt before retry.",
            )
        stages[role] = run_pin(accepted[0])
        parent_roles.append(role)
    return stages


def terminal(ctx, pin):
    from evidence_lens.recorder import worker_started
    from formal_lineage import run_pin
    from make_handoff import publish
    from select_trial import transfer_evidence

    store, row, dispatch = ctx["store"], ctx["row"], ctx["dispatch"]
    record = ctx["verified"].get(pin["run_id"])
    run = record["run"]
    require(
        run_pin(record) == pin
        and run["execution_status"] in {"failed", "cancelled", "interrupted"}
        and isinstance(run["failure_reason"], str)
        and run["failure_reason"].strip()
        and worker_started(store / "runs" / pin["run_id"], run),
        "A terminal failure must be a sealed failed scientific attempt",
    )
    require(
        run["run_kind"] in {"generation", "render"}, "Select the terminal Windows generation or render attempt"
    )
    receipts = transfer_evidence(store, {pin["run_id"]: record})
    require(receipts.get(pin["run_id"]), "Terminal return needs an accepted Mac transport receipt")
    gen = None
    if run["run_kind"] == "render":
        originals = [p for p in record["inputs"]["parents"] if p["run_id"] != dispatch["reference"]["run_id"]]
        require(len(originals) == 1, "Terminal render must identify its actual producer")
        gen = originals[0]
        genrec = ctx["verified"].get(gen["run_id"], "generation")
        require(
            run_pin(genrec) == gen and transfer_evidence(store, {gen["run_id"]: genrec}).get(gen["run_id"]),
            "Terminal render needs the accepted exact Windows generation",
        )
        validate_stage(ctx, genrec, "generation", dispatch["reference"])
    validate_stage(ctx, record, run["run_kind"], dispatch["reference"], gen)
    for path in (store / "runs").glob("*/run.json"):
        other = load(path)
        if (
            other["trial_id"] == run["trial_id"]
            and other["experiment_id"] == EXP
            and other["run_kind"] == run["run_kind"]
            and other["run_id"] != run["run_id"]
        ):
            if other["execution_status"] == "succeeded":
                raise ValueError("A successful trial attempt exists; a full failure disposition would hide it")
            require(
                other.get("retry_of_run_id") != run["run_id"] or not worker_started(path.parent, other),
                "A later scientific retry belongs to this failed attempt; select its terminal outcome",
            )
    result = {
        "schema": "el-vision-failed-trial-1",
        "selection_status": "failed",
        "planned_targets": list(TARGETS),
        "trial_id": row["trial_id"],
        "condition_id": "B0",
        "scene_family_id": row["scene_family_id"],
        "scene_id": row["scene_id"],
        "split": row["split"],
        "experiment_id": EXP,
        "catalog_sha256": ctx["catalog_hash"],
        "reference": dispatch["reference"],
        "readback": dispatch["readback"],
        "failure": pin,
        "reason": run["failure_reason"],
        "failed_kind": run["run_kind"],
        "handoff_sha256": dispatch["handoff_sha256"],
    }
    if gen:
        result["generation"] = gen
    if ctx["opening_path"]:
        result["opening_sha256"] = sha(ctx["opening_path"])
    dest = store / "trials" / row["trial_id"] / "trial_selection.json"
    publish(dest, encoded(result))
    return dest


def context(project, store, catalog_path, expected, family, cpu):
    from formal_lineage import VerifiedRuns
    from stage2_opening import verify_opening

    require(HASH.fullmatch(expected) and sha(catalog_path) == expected, "Registry digest mismatch")
    cat = load(catalog_path)
    require(
        cat["schema"] == "el-vision-family-roster-1" and len(cat["families"]) == 24, "Invalid Stage B roster"
    )
    matching = [r for r in cat["families"] if r["scene_family_id"] == family]
    require(len(matching) == 1, "Unknown or duplicate family")
    row = matching[0]
    require(
        row["split"] in {"train", "development", "test"}
        and row["trial_id"] == row["trials"]["B0"]
        and row["example_view"] == "v01",
        "Unexpected Stage B trial or fixed comparison view",
    )
    ctx = {
        "catalog_hash": expected,
        "catalog": cat,
        "row": row,
        "project": project,
        "store": store,
        "opening_path": None,
        "opening": None,
        "cpu": cpu,
        "verified": VerifiedRuns(store),
    }
    if row["split"] == "test":
        ctx["opening"] = verify_opening(store, cat, expected)
        ctx["opening_path"] = store / "protocol/stage2" / expected / "test-opening.json"
    return ctx


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("project", "store", "catalog"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--catalog-sha256", required=True)
    p.add_argument(
        "--returns-dir", type=Path, help="Folder containing the original RID.receipt.json and RID.zip pairs"
    )
    p.add_argument(
        "--generation", type=supplied_pin, help="Already Mac-received generation RUN_ID=MANIFEST_SHA256"
    )
    p.add_argument("--render", type=supplied_pin, help="Already Mac-received render RUN_ID=MANIFEST_SHA256")
    p.add_argument(
        "--terminal-failure",
        type=supplied_pin,
        help="Explicitly close an unsuccessful scientific Windows trial",
    )
    p.add_argument(
        "--cpu-python", type=Path, help="Base-only score interpreter; defaults to PROJECT/.venv-cpu/bin/python"
    )
    args = p.parse_args(argv)
    project, store = args.project.resolve(), args.store.resolve()
    sys.path.insert(0, str(project / "scripts"))
    require(bool(args.generation) == bool(args.render), "Provide both already-received Windows pins together")
    require(
        not (args.terminal_failure and args.generation)
        and (args.returns_dir is not None or args.generation or args.terminal_failure),
        "Choose original returns, exact already-received pins, or a terminal failed pin",
    )
    cpu = (args.cpu_python or project / ".venv-cpu/bin/python").absolute()
    ctx = context(project, store, args.catalog, args.catalog_sha256, args.family, cpu)
    from select_trial import freeze

    lock = store / "stage2" / args.catalog_sha256 / "mac-processing" / (args.family + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        prior = store / "trials" / ctx["row"]["trial_id"] / "trial_selection.json"
        if prior.exists():
            require(
                (load(prior)["selection_status"] == "failed") == bool(args.terminal_failure),
                "An immutable, differently disposed selection already occupies this trial",
            )
        ctx["refmeta"], ctx["dispatch"] = handoff(ctx)
        rows = collect_returns(ctx, args.returns_dir) if args.returns_dir else {}
        if args.terminal_failure:
            result = terminal(ctx, args.terminal_failure)
        else:
            require(cpu.is_file(), "Base-only scoring interpreter not found")
            direct = {"generation": args.generation, "render": args.render} if args.generation else None
            selected = windows(ctx, rows, direct)
            stages = {k: ctx["dispatch"][k] for k in ("reference", "readback")}
            picked = cpu_stages(ctx, {**stages, **selected})
            result, _ = freeze(store, ctx["row"]["trial_id"], {k: v["run_id"] for k, v in picked.items()})
            selection = load(result)
            require(
                all(
                    selection["selected"][k]["manifest_sha256"] == pin["manifest_sha256"]
                    for k, pin in picked.items()
                ),
                "Canonical scientific selection disagrees with the validated input pins",
            )
    print(
        json.dumps(
            {
                "trial_id": ctx["row"]["trial_id"],
                "selection_path": str(result),
                "selection_sha256": sha(result),
                "selection_status": load(result)["selection_status"],
            },
            sort_keys=True,
        )
    )
    return result


if __name__ == "__main__":
    main()
