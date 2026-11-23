"""Record R1 source preparation, score/evaluate/check an explicit held-out render, and freeze its quality selection."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from evidence_lens._bundle import encoded, load, save
from evidence_lens._index import writer_gate
from evidence_lens._verify import require
from evidence_lens.recorder import worker_started
from condition_jobs import QUALITY, ROOT, VIEWS, cli_context, common, validate_r1
from formal_lineage import run_pin
from mac_trial import select_files
from stress_condition import half_source


def payload(ctx, rid):
    return ctx["store"] / "runs" / rid / "payload"


def stage(ctx, role, kind, parents, inputs, command, camera_sha, retry=None, interpreter=None, dispatch=None):
    family, store, trial = ctx["family"], ctx["store"], ctx["quality_trial"]
    parent_ids = list(dict.fromkeys(parents))
    config = load(ROOT / "configs/p2-cpu.json")
    config.update({k:family[k] for k in ("scene_family_id","scene_id","dataset_id","dataset_hash","split","scene_version")},
                  trial_id=trial, condition_id=ctx["condition"], scene_hash=ctx["baseline"]["evidence"]["scene_sha256"], camera_set_hash=camera_sha,
                  config_version="p6-heldout-cpu-1", condition_stage_role=role, **ctx["release"], b0_selection_sha256=ctx["b0_selection_sha256"])
    if dispatch:
        config["formal_handoff_sha256"] = dispatch["handoff_sha256"]
    lock = store / "formal" / family["dataset_hash"] / "conditions" / "local" / trial / role
    with writer_gate(lock):
        matches = []
        for directory in ("runs", "work"):
            for p in sorted((store / directory).glob("*/run.json")):
                meta = load(p)
                if (meta["experiment_id"], meta["trial_id"], meta["run_kind"]) != (ctx.get("quality_experiment", QUALITY), trial, kind):
                    continue
                previous = load(p.parent / "config.resolved.yaml")
                if previous.get("condition_stage_role") != role:
                    continue
                require(directory != "work", "condition stage already has an unsealed local recorder attempt")
                require(all(previous.get(key) == value for key, value in config.items()) and set(meta["parent_run_ids"]) == set(parent_ids),
                        "existing condition stage has different execution settings or parents")
                matches.append((meta["started_at_utc"], p.parent, meta))
        for _, path, meta in sorted(matches):
            if meta["execution_status"] == "succeeded":
                checked = ctx["verified"].get(meta["run_id"], kind)
                return run_pin(checked)
        scientific = [m for _, p, m in matches if worker_started(p, m)]
        if scientific:
            require(len(scientific) == 1 and retry == scientific[0]["run_id"], "a failed condition stage needs its one explicit linked retry")
        else:
            require(not retry, "this condition stage has no scientific failure to retry")
        with tempfile.TemporaryDirectory(dir=store, prefix="condition-config-") as tmp:
            cfg = Path(tmp) / "stage.json"
            save(cfg, config)
            cmd = [sys.executable, "-m", "evidence_lens.recorder", "--root", str(store), "--source-root", str(ROOT), "--config", str(cfg),
                   "--experiment", ctx.get("quality_experiment", QUALITY), "--trial", trial, "--kind", kind, "--input", f'method-freeze={ctx["method_file"]}']
            if "opening_path" in ctx:
                cmd.extend(["--input", "stage2-opening=" + str(ctx["opening_path"])])
            for rid in parent_ids:
                cmd.extend(["--parent", rid + "=" + ctx["verified"].get(rid)["manifest_sha256"]])
            for name, (path, owner) in inputs.items():
                cmd.extend(["--input", f"{name}={path}"])
                if owner:
                    cmd.extend(["--input-parent", name + "=" + owner])
            if retry:
                cmd.extend(["--retry-of", retry])
            done = subprocess.run([*cmd, "--", interpreter or "{python}", *command], cwd=ROOT, text=True, stdout=subprocess.PIPE)
            require(done.returncode == 0, f"recorded condition {role} failed: {done.stdout.strip()}")
            return run_pin(ctx["verified"].get(json.loads(done.stdout)["run_id"], kind))


def prepare(ctx, retry=None):
    require(ctx["condition"] == "R1", "only R1 prepares a new source image on the Mac")
    rid, rb = (ctx["b0_ids"][k] for k in ("reference", "readback"))
    source = payload(ctx, rid)
    with Image.open(source / "input.png") as image:
        _, cameras = half_source(np.asarray(image), load(source / "cameras.json"))
    expected = hashlib.sha256((json.dumps(cameras, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
    inputs = {k:(source / n, rid) for k, n in (("rgb","input.png"), ("cameras","cameras.json"), ("reference","reference.json"))}
    worker = ["{source}/scripts/stress_condition.py", "R1", "--input", "{input:rgb}", "--cameras", "{input:cameras}",
              "--reference", "{input:reference}", "--out", "{out}"]
    result = stage(ctx, "preparation", "derivation", [rid, rb], inputs, worker, expected, retry)
    validate_r1(ctx, result["run_id"])
    return result


def condition_chain(ctx, args):
    from condition_selection import validate_windows

    require(ctx["condition"] in {"R1", "Q1"}, "B0 uses the formal controller; timing is never evaluated or selected")
    ref, rb, bgen = (ctx["b0_ids"][k] for k in ("reference", "readback", "generation"))
    require(args.producer_run and args.render_run, "supply explicit --producer-run and --render-run")
    cond = ctx["condition"]
    require((cond == "R1") == bool(args.preparation_run), "only R1 requires an explicit Mac source preparation")
    prep = args.preparation_run if cond == "R1" else args.producer_run
    if cond == "R1":
        validate_r1(ctx, prep)
    dispatch_file = ctx["store"] / "formal" / ctx["family"]["dataset_hash"] / "conditions/dispatch" / (ctx["quality_trial"] + ".json")
    dispatch = load(dispatch_file)
    require(hashlib.sha256(encoded(dispatch["job"])).hexdigest() == dispatch["handoff_sha256"], "condition dispatch content hash differs")
    ids = {"reference":ref,"readback":rb,"preparation":prep,"generation":args.producer_run if cond == "R1" else bgen,
           "producer":args.producer_run,"render":args.render_run}
    source_id = prep if cond == "R1" else ref
    require(dispatch["job"]["source"] == run_pin(ctx["verified"].get(source_id)) and dispatch["job"]["baseline_generation"] == run_pin(ctx["verified"].get(bgen))
            and dispatch["job"]["experiment_id"] == ctx.get("quality_experiment", QUALITY) and dispatch["job"]["timing_repeat"] is None and
            dispatch["job"]["b0_selection_sha256"] == ctx["b0_selection_sha256"] and dispatch["job"]["method_freeze_sha256"] == ctx["release"]["method_freeze_sha256"],
            "held-out render requires this condition's actual quality dispatch")
    validate_windows(ctx, ids, dispatch)
    return ids, source_id, dispatch


def run_stage(ctx, args):
    from condition_selection import accepted_local, select

    if args.stage == "timing":
        from condition_timing import explicit_timing
        return explicit_timing(ctx, args.timing_repeat, args.preparation_run, args.producer_run, args.render_run)
    if args.stage == "prepare":
        return prepare(ctx, args.retry_of)
    ids, src_id, dispatch = condition_chain(ctx, args)
    ref, rb, prep, render = (ids[k] for k in ("reference", "readback", "preparation", "render"))
    src = payload(ctx, src_id)
    rgb, cams = src / "input.png", src / "cameras.json"
    camera_hash = dispatch["job"]["identity"]["camera_set_hash"]
    base_parents = list(dict.fromkeys([ref, rb, prep, render]))
    targets = tuple(v for v in VIEWS if v != "source")
    if args.stage in ("evaluate", "verify", "select"):
        ids["score"] = args.score_run
        accepted_local(ctx, ids["score"], "model_score", "score", base_parents, dispatch)
    if args.stage in ("verify", "select"):
        ids["evaluation"] = args.evaluation_run
        accepted_local(ctx, ids["evaluation"], "evaluation", "evaluation", [*base_parents, ids["score"]], dispatch)
    if args.stage == "select":
        ids["recompute"] = args.recompute_run
        accepted_local(ctx, ids["recompute"], "archive_verify", "recompute", [*base_parents, ids["score"], ids["evaluation"]], dispatch)
        return select(ctx, ids, src_id, dispatch)
    if args.stage == "verify":
        items = {"cameras":(cams, src_id), "metadata":(payload(ctx, render) / "prediction.json", render),
                 "source-prediction":(payload(ctx, render)/"source.npz", render), "prediction":(payload(ctx, render)/"v01.npz", render),
                 "reference":(payload(ctx, ref)/"reference/v01.npz", ref), "score":(payload(ctx, ids["score"])/"v01.npz", ids["score"]),
                 "labels":(payload(ctx, ids["evaluation"])/"labels/v01.npz", ids["evaluation"]),
                 "metrics":(payload(ctx, ids["evaluation"])/"metrics.csv", ids["evaluation"])}
        command = ["{source}/scripts/recompute_view.py", "--view", "v01", "--out", "{out}"]
        for key in items:
            command.extend(["--" + key, "{input:" + key + "}"])
        return stage(ctx, "recompute", "archive_verify", [*base_parents, ids["score"], ids["evaluation"]], items, command,
                     camera_hash, args.retry_of, dispatch=dispatch)
    with tempfile.TemporaryDirectory(dir=ctx["store"], prefix="condition-bounded-") as tmp:
        temp = Path(tmp)
        preds = select_files(payload(ctx, render), temp / "predictions", ["prediction.json", *(v + ".npz" for v in VIEWS)])
        items = {"rgb":(rgb, src_id), "cameras":(cams, src_id), "predictions":(preds, render)}
        inputs = ["--input", "{input:rgb}", "--cameras", "{input:cameras}", "--predictions", "{input:predictions}", "--output", "{out}"]
        if args.stage == "score":
            return stage(ctx, "score", "model_score", base_parents, items, ["-m", "evidence_lens.risk_model", *inputs, "--run-id", "{run_id}"],
                         camera_hash, args.retry_of, str(args.cpu_python.absolute()) if args.cpu_python else None, dispatch)
        targets_dir = select_files(payload(ctx, ref), temp / "target-reference", ["validation/results.json", *(f"reference/{v}.npz" for v in targets)])
        score = ids["score"]
        score_dir = select_files(payload(ctx, score), temp / "scores", ["score.json", "feature_schema.json", *(v + ".npz" for v in targets)])
        items.update({"target-reference":(targets_dir, ref), "reference-metadata":(src/"reference.json",src_id), "scores":(score_dir,score)})
        command = ["{source}/scripts/condition_evaluate.py", *inputs, "--target-reference", "{input:target-reference}",
                   "--reference-metadata", "{input:reference-metadata}", "--scores", "{input:scores}", "--trial-id", ctx["quality_trial"],
                   "--scene-id", ctx["family"]["scene_id"], "--scene-family-id", ctx["family"]["scene_family_id"], "--condition-id",ctx["condition"]]
        return stage(ctx, "evaluation", "evaluation", [*base_parents, score], items, command, camera_hash, args.retry_of, dispatch=dispatch)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "score", "evaluate", "verify", "select", "timing"))
    common(p)
    for name in ("producer-run", "render-run", "score-run", "evaluation-run", "recompute-run", "retry-of"):
        p.add_argument("--" + name)
    p.add_argument("--timing-repeat", type=int, choices=(1, 2, 3))
    p.add_argument("--cpu-python", type=Path)
    a = p.parse_args()
    print(json.dumps(run_stage(cli_context(a), a)))
