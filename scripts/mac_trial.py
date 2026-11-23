"""Score, evaluate or independently verify a received B0 trial against its explicit, independently checked reference."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from evidence_lens._bundle import RUN_ID, load, sha
from evidence_lens._verify import require
from formal_lineage import EXPERIMENT, VIEWS, VerifiedRuns, family_record, heldout_method, reference_identity, run_pin


def select_files(directory, destination, filenames):
    """Assemble a bounded input directory for the recorder from exact asset names."""
    destination.mkdir()
    for name in filenames:
        source = directory/name
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
    return destination


def recompute_view(store, project, config, ref, readback, render, score, evaluation, trial_id, experiment, retry=None, method_file=None):
    view = "v01"
    files = {
        "cameras": ref / "payload/cameras.json",
        "metadata": render / "payload/prediction.json",
        "source-prediction": render / "payload/source.npz",
        "prediction": render / "payload" / f"{view}.npz",
        "reference": ref / "payload/reference" / f"{view}.npz",
        "score": score / "payload" / f"{view}.npz",
        "labels": evaluation / "payload/labels" / f"{view}.npz",
        "metrics": evaluation / "payload/metrics.csv",
    }
    cmd = [sys.executable, "-m", "evidence_lens.recorder", "--root", str(store),
           "--source-root", str(project), "--config", str(config), "--experiment", experiment,
           "--trial", trial_id, "--kind", "archive_verify"]
    if method_file:
        cmd.extend(["--input", f"method-freeze={method_file}"])
    if retry:
        cmd.extend(["--retry-of", retry])
    for upstream in (ref, readback, render, score, evaluation):
        cmd.extend(["--parent", upstream.name])
    worker = ["{python}", "{source}/scripts/recompute_view.py", "--view", view, "--out", "{out}"]
    for role, path in files.items():
        cmd.extend(["--input", f"{role}={path}"])
        worker.extend([f"--{role}", "{input:" + role + "}"])
    return subprocess.run([*cmd, "--", *worker], cwd=project).returncode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("score", "evaluate", "verify"))
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--reference-run", required=True)
    p.add_argument("--render-run", required=True)
    p.add_argument("--readback-run", required=True, help="independent accepted Blender reference readback")
    p.add_argument("--generation-run", help="pinned successful Windows generation; required for a formal family")
    p.add_argument("--families", type=Path, help="frozen family catalog for formal B0")
    p.add_argument("--freeze", type=Path, help="formal freeze JSON; defaults to configs/formal-freeze-v1.json")
    p.add_argument("--method-freeze", type=Path, help="P5 frozen method for held-out evaluation and independent readback")
    p.add_argument("--retry-of", help="failed first scientific stage being retried")
    p.add_argument("--score-run")
    p.add_argument("--evaluation-run")
    p.add_argument("--cpu-python", type=Path, help="separate base-only interpreter for the scoring subprocess")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    store = args.store.resolve()
    verified = VerifiedRuns(store)

    def accepted_run(run_id, kind):
        record = verified.get(run_id, kind)
        return store / "runs" / run_id, record["run"]

    ref, ref_run = accepted_run(args.reference_run, "scene_render")
    check, _ = accepted_run(args.readback_run, "archive_verify")
    family, binding, method_file, release = None, None, None, None
    if args.families:
        require(args.generation_run, "formal Mac stage requires --generation-run")
        family = family_record(args.families, args.freeze or root / "configs/formal-freeze-v1.json", ref_run["scene_family_id"])
        if family["split"] == "test" and args.stage in ("evaluate", "verify"):
            method_file = args.method_freeze or store / "protocol/method-freeze-v1.json"
            release = heldout_method(verified, family, method_file, args.families.parent / "study-v1.json", args.freeze or root / "configs/formal-freeze-v1.json")
    elif ref_run.get("dataset_hash") is not None:
        raise ValueError("formal Mac stage requires the frozen --families catalog")
    ref_meta, checked_ref, checked_rb = reference_identity(verified, ref.name, check.name, family)
    experiment = EXPERIMENT if family else "p2-single-image"
    pred, pred_run = accepted_run(args.render_run, "render")
    if family:
        binding = load(store / "formal" / family["dataset_hash"] / "dispatch" / (family["trial_id"] + ".json"))
        require(binding["reference"] == run_pin(checked_ref) and binding["readback"] == run_pin(checked_rb),
                "formal reference or readback differs from the dispatched handoff")
        for rid, kind in ((args.generation_run, "generation"), (pred.name, "render")):
            checked = verified.get(rid, kind)
            settings = load(store / "runs" / rid / "config.resolved.yaml")
            require(checked["run"]["experiment_id"] == experiment and settings.get("formal_handoff_sha256") == binding["handoff_sha256"]
                    and all(settings.get(k) == v for k, v in binding["identity"].items())
                    and settings.get("worker_source_sha256") == binding["worker_source_sha256"], "worker belongs to a different dispatch")
        require(verified.get(args.generation_run)["inputs"]["parents"] == [run_pin(checked_ref)] and
                set(pred_run["parent_run_ids"]) == {ref.name, args.generation_run}, "wrong formal generation/render parents")
    ref_src, pred_src = ref/"payload", pred/"payload"
    pred_meta = load(pred_src/"prediction.json")
    if not ref_meta["validation_passed"] or args.reference_run not in pred_run["parent_run_ids"]:
        raise ValueError("prediction does not descend from this validated scene")
    if (pred_meta["producer_run_id"] != args.render_run or pred_run["trial_id"] != ref_run["trial_id"]
            or pred_meta["generation_run_id"] not in pred_run["parent_run_ids"]):
        raise ValueError("prediction and recorder provenance differ")
    if (pred_meta["camera_sha256"] != ref_meta["camera_sha256"] or pred_meta["input_sha256"] != ref_meta["input_sha256"]
            or pred_meta["depth_validation_receipt_sha256"] != sha(pred_src/"depth-probe.json")
            or pred_meta["depth_validation"] != load(pred_src/"depth-probe.json")):
        raise ValueError("camera, source PNG, or calibration provenance differs")
    if args.generation_run and pred_meta["generation_run_id"] != args.generation_run:
        raise ValueError("prediction identifies a different producer")
    if family:
        probe = load(pred_src / "depth-probe.json")
        require(set(pred_meta["views"]) == set(VIEWS) and not pred_meta.get("failed_views") and
                all(probe.get(key) is True for key in ("off_axis_plane_passed", "alpha_normalization_passed", "translation_passed", "linear_blend_passed"))
                and probe.get("empty_pixel_alpha") == 0 and all(verified.get(pred.name)["artifacts"].get("payload/"+view+ext, {}).get("bytes", 0) > 0
                    for view in VIEWS for ext in (".npz", ".png")), "formal render is missing seven products or measured calibration")
    if pred_run["condition_id"] != "B0":
        raise ValueError("this helper is for B0 packages")
    score = None
    if args.stage in ("evaluate", "verify"):
        if not args.score_run:
            p.error("--score-run is required for this stage")
        score, score_run = accepted_run(args.score_run, "model_score")
        if (set(score_run["parent_run_ids"]) != {ref.name, check.name, pred.name} or score_run["trial_id"] != ref_run["trial_id"]
                or score_run["experiment_id"] != experiment or (family and load(score / "config.resolved.yaml").get("formal_handoff_sha256") != binding["handoff_sha256"])):
            raise ValueError("scores do not descend from this prediction and trial")
    targets = [f"v{i:02}" for i in range(1, 7)]
    config = load(root/"configs/p2-cpu.json")
    config.update(scene_id=ref_run["scene_id"], scene_family_id=ref_run["scene_family_id"],
                  camera_set_hash=ref_meta["camera_sha256"], scene_hash=ref_meta["blend_sha256"])
    if family:
        config.update(binding["identity"], experiment_id=experiment, config_version="formal-b0-cpu-1",
                      formal_handoff_sha256=binding["handoff_sha256"], worker_source_sha256=binding["worker_source_sha256"])
    if release:
        config.update(release)
    with tempfile.TemporaryDirectory(dir=store, prefix="mac-stage-") as tmp:
        temp = Path(tmp)
        cfg = temp/"trial.json"
        cfg.write_text(json.dumps(config, indent=2)+"\n", encoding="utf8")
        if args.stage == "verify":
            if not args.evaluation_run:
                p.error("--evaluation-run is required for verify")
            evaluation, eval_run = accepted_run(args.evaluation_run, "evaluation")
            if (set(eval_run["parent_run_ids"]) != {ref.name, check.name, pred.name, score.name}
                    or eval_run["trial_id"] != ref_run["trial_id"] or eval_run["experiment_id"] != experiment
                    or (family and load(evaluation / "config.resolved.yaml").get("formal_handoff_sha256") != binding["handoff_sha256"])):
                raise ValueError("evaluation belongs to different inputs")
            if release:
                require(load(evaluation / "config.resolved.yaml").get("method_freeze_sha256") == release["method_freeze_sha256"],
                        "evaluation used a different held-out method freeze")
            raise SystemExit(recompute_view(store, root, cfg, ref, check, pred, score,
                                            evaluation, ref_run["trial_id"], experiment, args.retry_of, method_file))
        prediction = select_files(pred_src, temp/"predictions", ["prediction.json"]+
                                  [n+".npz" for n in ["source", *targets] if n not in pred_meta.get("failed_views", {})])
        bindings = ["--parent", ref.name, "--parent", check.name, "--parent", pred.name, "--input", f"rgb={ref_src/'input.png'}",
                    "--input", f"cameras={ref_src/'cameras.json'}", "--input", f"predictions={prediction}",
                    "--input-parent", "predictions="+pred.name]
        common = ["--input", "{input:rgb}", "--cameras", "{input:cameras}", "--predictions", "{input:predictions}",
                  "--output", "{out}", "--run-id", "{run_id}"]
        if args.stage == "score":
            kind = "model_score"
            interpreter = str(args.cpu_python.absolute()) if args.cpu_python else "{python}"
            worker = [interpreter, "-m", "evidence_lens.risk_model", *common]
        else:
            truth = select_files(ref_src, temp/"truth", ["reference.json","validation/results.json"]+
                                [f"reference/{v}.npz" for v in targets])
            scores = select_files(score/"payload", temp/"scores", ["score.json","feature_schema.json"]+
                                  [v+".npz" for v in targets if v not in pred_meta.get("failed_views", {})])
            kind = "evaluation"
            bindings += ["--parent", score.name, "--input", f"reference={truth}", "--input-parent", "reference="+ref.name,
                         "--input", f"scores={scores}", "--input-parent","scores="+score.name]
            worker = ["{python}", "-m", "evidence_lens.evaluation", *common, "--references", "{input:reference}",
                      "--scores", "{input:scores}", "--trial-id", ref_run["trial_id"], "--scene-id", ref_run["scene_id"],
                      "--scene-family-id", ref_run["scene_family_id"], "--condition-id", "B0", "--parquet"]
        if method_file:
            bindings.extend(["--input", f"method-freeze={method_file}"])
        if args.retry_of:
            bindings.extend(["--retry-of", args.retry_of])
        result = subprocess.run([sys.executable,"-m","evidence_lens.recorder","--root",str(store),
                                "--source-root",str(root),"--config",str(cfg),"--experiment",experiment,
                                "--trial",ref_run["trial_id"],"--kind",kind,*bindings,"--",*worker], cwd=root)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
