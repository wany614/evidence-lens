"""Dispatch one isolated Windows P6 generation, colour derivation or calibrated render."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evidence_lens._bundle import HASH, LABEL, RUN_ID, files, load, local_path, sha
from evidence_lens._verify import require, verify_bundle
from worker_jobs import dispatch, frozen_json


def read_job(base):
    job = load(base / "handoff.json")
    cond, identity = job["condition"], job["identity"]
    require(job["schema"] in {"el-condition-handoff-1","el-stage2-condition-handoff-1"} and cond in {"R1", "Q1", "B0"} and
            identity["split"] == "test" and identity["condition_id"] == cond and identity["trial_id"] == job["trial_id"] and
            LABEL.fullmatch(job["trial_id"]) and all(HASH.fullmatch(job[k]) for k in ("method_freeze_sha256", "b0_selection_sha256", "worker_source_sha256")),
            "invalid held-out condition job")
    if identity.get("dataset_id") == "evidence-lens-vision-families-v1":
        require(job["schema"] == "el-stage2-condition-handoff-1", "Stage B cannot use a legacy condition gate")
    timed = job["timing_repeat"] is not None
    if job["schema"] == "el-stage2-condition-handoff-1":
        opening = job["opening"]
        require(not timed and cond in {"R1", "Q1"} and job["warmup"] is None
                and job["experiment_id"] == "stage2-heldout-conditions" and LABEL.fullmatch(job["baseline_trial_id"])
                and opening["catalog_sha256"] == identity["dataset_hash"]
                and opening["method_sha256"] == job["method_freeze_sha256"]
                and RUN_ID.fullmatch(opening["freeze_run_id"])
                and all(HASH.fullmatch(opening[k]) for k in ("catalog_sha256", "opening_sha256", "freeze_manifest_sha256", "method_sha256")),
                "Stage-two condition does not bind its Mac-opened cohort and method")
    else:
        require((timed and type(job["timing_repeat"]) is int and job["timing_repeat"] in (1, 2, 3) and job["warmup"] and job["experiment_id"] == "p6-timing-repeats")
                or (not timed and cond != "B0" and job["warmup"] is None and job["experiment_id"] == "p6-heldout-conditions"), "invalid timing dispatch")
    for name in ("reference", "readback", "source", "baseline_generation"):
        pin = job[name]
        require(RUN_ID.fullmatch(pin["run_id"]) and HASH.fullmatch(pin["manifest_sha256"]), "invalid external parent pin")
    require(cond == "R1" or job["source"] == job["reference"], "condition must use its registered source owner")
    manifest = files(base)
    require(set(manifest) == {"handoff.json", *job["files"]} and
            set(n for n in job["files"] if not n.startswith("code/")) == {"input.png", "cameras.json"} and
            all(sha(manifest[local_path(n)]) == digest for n, digest in job["files"].items()) and
            identity["camera_set_hash"] == job["files"]["cameras.json"], "changed or unexpected condition handoff member")
    return job, sha(base / "handoff.json")


def checked_producer(store, job, digest, rid=None):
    baseline = rid is None
    pin = job["baseline_generation"] if baseline else None
    rid = pin["run_id"] if baseline else rid
    require(isinstance(rid, str) and RUN_ID.fullmatch(rid), "provide the explicit producer run ID")
    bundle = store / "runs" / rid
    result = verify_bundle(bundle)
    rec, kind = result["run"], ("generation" if baseline or job["condition"] != "Q1" else "derivation")
    require(rec["run_id"] == rid and rec["execution_status"] == "succeeded" and rec["run_kind"] == kind and
            all(rec.get(k) == job["identity"][k] for k in ("scene_family_id", "scene_id", "dataset_id", "dataset_hash", "split")),
            "producer is from another held-out family")
    generation = load(bundle / "payload/generation.json")
    require(generation["producer_run_id"] == rid and result["artifacts"]["payload/scene.ply"]["sha256"] == generation["ply_sha256"],
            "producer metadata differs from the actual PLY")
    if baseline:
        if job["schema"] == "el-stage2-condition-handoff-1":
            require(rec["experiment_id"] == "stage2-single-image" and rec["trial_id"] == job["baseline_trial_id"],
                    "Q1 baseline is not the registered Stage B B0 experimental trial")
        require(result["manifest_sha256"] == pin["manifest_sha256"] and rec["condition_id"] == "B0" and
                generation["ply_sha256"] == job["baseline_ply_sha256"] and result["inputs"]["parents"] == [job["reference"]],
                "Q1 requires the exact selected baseline generation")
    else:
        require(rec["trial_id"] == job["trial_id"] and rec["experiment_id"] == job["experiment_id"] and
                rec["condition_id"] == job["condition"] and load(bundle / "config.resolved.yaml")["formal_handoff_sha256"] == digest and
                result["inputs"]["parents"] == [job["baseline_generation"] if kind == "derivation" else job["source"]] and
                generation["camera_sha256"] == job["files"]["cameras.json"] and generation["input_sha256"] == job["files"]["input.png"],
                "render producer was produced for a different condition dispatch")
    return {"run_id":rid, "manifest_sha256":result["manifest_sha256"]}


def parent(pin):
    return ["--parent", pin["run_id"] + "=" + pin["manifest_sha256"]]


def run(args):
    base, store, outgoing = (p.resolve() for p in (args.handoff, args.store, args.return_dir))
    job, digest = read_job(base)
    require(Path(__file__).resolve() == base / "code/scripts/condition_stage.py", "use the handoff's frozen Windows entry point")
    paths = [store, outgoing, Path(sys.executable), *([args.weights] if args.weights else [])]
    require(all(p.resolve().drive.casefold() == base.drive.casefold() for p in paths), "all worker storage must use its research volume")
    store.mkdir(parents=True, exist_ok=True)
    outgoing.mkdir(parents=True, exist_ok=True)
    for key, sub in {"TORCH_EXTENSIONS_DIR":"torch-extensions", "TORCH_HOME":"torch", "HF_HOME":"models", "CUDA_CACHE_PATH":"cuda",
                     "TMP":"temp", "TEMP":"temp"}.items():
        dest = store / "cache" / sub
        dest.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(dest)
    code = base / "code"
    os.environ.update(PYTHONPATH=str(code / "src"), PYTHONDONTWRITEBYTECODE="1")
    settings = load(code / "configs/windows-b0.json") | job["identity"]
    settings.update(config_version="p6-condition-1", formal_handoff_sha256=digest, worker_source_sha256=job["worker_source_sha256"],
                    method_freeze_sha256=job["method_freeze_sha256"], b0_selection_sha256=job["b0_selection_sha256"],
                    timing_repeat=job["timing_repeat"], warmup=job["warmup"], recording={"expected_output_bytes":120_000_000})
    if "opening" in job:
        settings.update(config_version="stage2-condition-1", stage2_opening=job["opening"])
    cfg = store / "formal" / "configs" / (digest + ".json")
    cfg.parent.mkdir(parents=True, exist_ok=True)
    frozen_json(cfg, settings)
    source = job["source"]
    source_args = [*parent(source), "--input", f"rgb={base/'input.png'}", "--input-parent", "rgb=" + source["run_id"],
                   "--input", f"cameras={base/'cameras.json'}", "--input-parent", "cameras=" + source["run_id"]]
    worker_args = ["--input", "{input:rgb}", "--cameras", "{input:cameras}", "--config", "{config}", "--out", "{out}"]
    producer = None
    if args.stage == "generate":
        require(job["condition"] != "Q1" and args.weights is not None and args.producer_run is None, "only B0 timing or R1 can generate")
        kind = "generation"
        weights = args.weights.resolve()
        pin = settings["generator"]["checkpoint_sha256"]
        cas = store / "assets/sha256" / pin[:2] / pin
        if cas.is_file() and weights != cas:
            require(weights.is_file() and weights.stat().st_size == settings["generator"]["checkpoint_bytes"] and sha(weights) == pin,
                    "requested weights differ from the frozen model")
            weights = cas
        bindings = [*source_args, "--shared-input", "weights=" + str(weights)]
        worker = ["{python}", "-m", "evidence_lens.generator_adapter", "--weights", "{input:weights}", *worker_args]
    elif args.stage == "derive":
        require(job["condition"] == "Q1" and args.weights is None and args.producer_run is None, "derive is the registered Q1 colour operation")
        kind = "derivation"
        producer = checked_producer(store, job, digest)
        prev = store / "runs" / producer["run_id"] / "payload"
        bindings = [*parent(producer), "--input", f"ply={prev/'scene.ply'}", "--input", f"generation={prev/'generation.json'}"]
        worker = ["{python}", "{source}/scripts/stress_condition.py", "Q1", "--ply", "{input:ply}", "--generation", "{input:generation}", "--out", "{out}"]
    else:
        require(args.producer_run and args.weights is None, "render needs its exact producer and no checkpoint input")
        kind = "render"
        producer = checked_producer(store, job, digest, args.producer_run)
        prev = store / "runs" / producer["run_id"] / "payload"
        bindings = [*source_args, *parent(producer), "--input", f"ply={prev/'scene.ply'}", "--input", f"generation={prev/'generation.json'}"]
        worker = ["{python}", "-m", "evidence_lens.renderer_adapter", "--ply", "{input:ply}", "--generation", "{input:generation}", *worker_args]
    if args.retry_of:
        bindings.extend(["--retry-of", args.retry_of])
    command = [sys.executable, "-m", "evidence_lens.recorder", "--root", str(store), "--source-root", str(code), "--config", str(cfg),
               "--kind", kind, "--experiment", job["experiment_id"], "--trial", job["trial_id"], *bindings, "--", *worker]
    def launch():
        result = subprocess.run(command, cwd=code, text=True, stdout=subprocess.PIPE)
        require(result.stdout.strip(), f"recorder returned no sealed run (exit {result.returncode})")
        return json.loads(result.stdout)
    return dispatch(store, outgoing, job, digest, kind, producer, args.retry_of, args.resume_preflight, launch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "derive", "render"))
    for name in ("handoff", "store", "return-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--producer-run")
    parser.add_argument("--retry-of")
    parser.add_argument("--resume-preflight")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("the registered Windows worker is required")
    response = run(args)
    print(json.dumps(response), flush=True)
    raise SystemExit(response["exit_code"])
