"""Score, evaluate or independently verify a received P2 B0 trial."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from evidence_lens._bundle import RUN_ID, load, sha
from evidence_lens._verify import verify_bundle, verify_dependencies


def accepted_run(store, run_id, kind):
    if not RUN_ID.fullmatch(run_id):
        raise ValueError(f"invalid {kind} ID")
    bundle = store/"runs"/run_id
    checked = verify_bundle(bundle)
    verify_dependencies(checked, store)
    meta = checked["run"]
    if meta["run_id"] != run_id or meta["run_kind"] != kind or meta["execution_status"] != "succeeded":
        raise ValueError(f"{kind} run unavailable or unsuccessful")
    return bundle, meta


def select_files(directory, destination, filenames):
    """Assemble a bounded input directory for the recorder from exact asset names."""
    destination.mkdir()
    for name in filenames:
        source = directory/name
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return destination


def recompute_view(store, project, config, ref, readback, render, score, evaluation, trial_id):
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
           "--source-root", str(project), "--config", str(config), "--experiment", "p2-single-image",
           "--trial", trial_id, "--kind", "archive_verify"]
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
    p.add_argument("--score-run")
    p.add_argument("--evaluation-run")
    p.add_argument("--cpu-python", type=Path, help="separate base-only interpreter for the scoring subprocess")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    store = args.store.resolve()
    ref, ref_run = accepted_run(store, args.reference_run, "scene_render")
    check, check_run = accepted_run(store, args.readback_run, "archive_verify")
    confirmation = load(check/"payload/reference_readback.json")
    if (ref.name not in check_run["parent_run_ids"] or check_run["trial_id"] != ref_run["trial_id"]
            or confirmation["reference_json_sha256"] != sha(ref/"payload/reference.json")
            or confirmation["scientific_acceptance"] is not True or confirmation["integrity_checked"] is not True):
        raise ValueError("reference readback is incomplete or belongs to a different reference")
    pred, pred_run = accepted_run(store, args.render_run, "render")
    ref_src, pred_src = ref/"payload", pred/"payload"
    ref_meta, pred_meta = load(ref_src/"reference.json"), load(pred_src/"prediction.json")
    if not ref_meta["validation_passed"] or args.reference_run not in pred_run["parent_run_ids"]:
        raise ValueError("prediction does not descend from this validated scene")
    if (pred_meta["producer_run_id"] != args.render_run or pred_run["trial_id"] != ref_run["trial_id"]
            or pred_meta["generation_run_id"] not in pred_run["parent_run_ids"]):
        raise ValueError("prediction and recorder provenance differ")
    if (pred_meta["camera_sha256"] != ref_meta["camera_sha256"] or pred_meta["input_sha256"] != ref_meta["input_sha256"]
            or pred_meta["depth_validation_receipt_sha256"] != sha(pred_src/"depth-probe.json")
            or pred_meta["depth_validation"] != load(pred_src/"depth-probe.json")):
        raise ValueError("camera, source PNG, or calibration provenance differs")
    # Only the P2 B0 engineering trial uses this transfer helper.
    if pred_run["condition_id"] != "B0":
        raise ValueError("this helper is for B0 packages")
    score = None
    if args.stage in ("evaluate", "verify"):
        if not args.score_run:
            p.error("--score-run is required for this stage")
        score, score_run = accepted_run(store, args.score_run, "model_score")
        if pred.name not in score_run["parent_run_ids"] or score_run["trial_id"] != ref_run["trial_id"]:
            raise ValueError("scores do not descend from this prediction and trial")
    targets = [f"v{i:02}" for i in range(1, 7)]
    config = load(root/"configs/p2-cpu.json")
    config.update(scene_id=ref_run["scene_id"], scene_family_id=ref_run["scene_family_id"],
                  camera_set_hash=ref_meta["camera_sha256"], scene_hash=ref_meta["blend_sha256"])
    with tempfile.TemporaryDirectory(dir=store, prefix="mac-stage-") as tmp:
        temp = Path(tmp)
        cfg = temp/"trial.json"
        cfg.write_text(json.dumps(config, indent=2)+"\n", encoding="utf8")
        if args.stage == "verify":
            if not args.evaluation_run:
                p.error("--evaluation-run is required for verify")
            evaluation, eval_run = accepted_run(store, args.evaluation_run, "evaluation")
            if (not {ref.name, check.name, pred.name, score.name}.issubset(eval_run["parent_run_ids"])
                    or eval_run["trial_id"] != ref_run["trial_id"]):
                raise ValueError("evaluation belongs to different inputs")
            raise SystemExit(recompute_view(store, root, cfg, ref, check, pred, score,
                                            evaluation, ref_run["trial_id"]))
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
        result = subprocess.run([sys.executable,"-m","evidence_lens.recorder","--root",str(store),
                                "--source-root",str(root),"--config",str(cfg),"--experiment","p2-single-image",
                                "--trial",ref_run["trial_id"],"--kind",kind,*bindings,"--",*worker], cwd=root)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
