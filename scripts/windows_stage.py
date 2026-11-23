"""Run or resume one recorder-backed Windows generation or seven-view render."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# This entry point also runs in the extracted, deliberately minimal handoff.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evidence_lens._bundle import HASH, LABEL, RUN_ID, files, load, local_path, sha
from evidence_lens._verify import require
from worker_jobs import check_generation, dispatch, frozen_json, outgoing_receipt


def read_handoff(base):
    job = load(base / "handoff.json")
    require(job["schema"] in {"el-worker-handoff-1", "el-worker-handoff-2"}, "unsupported handoff schema")
    actual = files(base)
    require(set(actual) == {"handoff.json", *job["files"]}, "handoff contains missing or undeclared files")
    for name, digest in job["files"].items():
        require(HASH.fullmatch(digest) and sha(actual[local_path(name)]) == digest, f"changed handoff member: {name}")
    require(set(n for n in job["files"] if not n.startswith("code/")) == {"input.png", "cameras.json"},
            "handoff must contain exactly the one source PNG and cameras outside its worker code")
    if job["schema"] == "el-worker-handoff-2":
        identity = job["identity"]
        require(job["condition"] == identity["condition_id"] == "B0" and job["trial_id"] == identity["trial_id"] and
                LABEL.fullmatch(job["trial_id"]) and LABEL.fullmatch(job["experiment_id"]) and
                identity["camera_set_hash"] == job["files"]["cameras.json"] and identity["split"] in {"train", "development", "test"}
                and all(HASH.fullmatch(identity[k]) for k in ("scene_hash", "dataset_hash")) and
                all(RUN_ID.fullmatch(job[k]["run_id"]) and HASH.fullmatch(job[k]["manifest_sha256"]) for k in ("reference", "readback")),
                "formal family, readback or camera pins are incomplete")
    return job, sha(base / "handoff.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["generate", "render"])
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--return-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--generation-run")
    parser.add_argument("--retry-of")
    parser.add_argument("--resume-preflight", help="explicit last run that failed before its scientific worker started")
    args = parser.parse_args(argv)
    if os.name != "nt":
        parser.error("the registered Windows worker is required")
    base, store, outgoing = (p.resolve() for p in (args.handoff, args.store, args.return_dir))
    job, digest = read_handoff(base)
    formal = job["schema"] == "el-worker-handoff-2"
    require(not formal or Path(__file__).resolve() == (base / "code/scripts/windows_stage.py").resolve(),
            "execute the worker entry point shipped in this frozen handoff")
    if args.stage == "generate" and not args.weights:
        parser.error("--weights is required for generation")
    if args.stage == "render" and not (args.generation_run and RUN_ID.fullmatch(args.generation_run)):
        parser.error("--generation-run must name the exact producer run")
    paths = [store, outgoing, Path(sys.executable), *([args.weights] if args.weights else [])]
    require(all(path.resolve().drive.casefold() == base.drive.casefold() for path in paths),
            "the handoff, store, environment, weights and return packs must use the research volume")
    store.mkdir(parents=True, exist_ok=True)
    outgoing.mkdir(parents=True, exist_ok=True)
    code = base / "code"
    for key, sub in {"TORCH_EXTENSIONS_DIR": "torch-extensions", "TORCH_HOME": "torch", "HF_HOME": "models",
                     "CUDA_CACHE_PATH": "cuda", "TMP": "temp", "TEMP": "temp"}.items():
        dest = store / "cache" / sub
        dest.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(dest)
    os.environ.update(PYTHONPATH=str(code / "src"), PYTHONDONTWRITEBYTECODE="1")
    cfg = code / "configs/windows-b0.json"
    if formal:
        settings = load(cfg) | job["identity"]
        settings.update(config_version="formal-b0-1", formal_handoff_sha256=digest,
                        upstream_reference=job["reference"], independent_readback=job["readback"],
                        worker_source_sha256=job["worker_source_sha256"], experiment_id=job["experiment_id"],
                        recording={"expected_output_bytes": 120_000_000})
        cfg = store / "formal" / "configs" / (digest + ".json")
        cfg.parent.mkdir(parents=True, exist_ok=True)
        frozen_json(cfg, settings)
    ref = job["reference"]
    bound = ["--parent", f'{ref["run_id"]}={ref["manifest_sha256"]}', "--input", f"rgb={base/'input.png'}",
             "--input-parent", "rgb=" + ref["run_id"], "--input", f"cameras={base/'cameras.json'}", "--input-parent", "cameras=" + ref["run_id"]]
    common = ["--input", "{input:rgb}", "--cameras", "{input:cameras}", "--config", "{config}", "--out", "{out}"]
    generation = None
    if args.stage == "generate":
        kind = "generation"
        weights = args.weights.resolve()
        if formal:
            expected = settings["generator"]["checkpoint_sha256"]
            canonical = store / "assets/sha256" / expected[:2] / expected
            if canonical.is_file() and weights != canonical:
                require(weights.is_file() and weights.stat().st_size == settings["generator"]["checkpoint_bytes"] and sha(weights) == expected,
                        "requested weights do not match the frozen checkpoint")
                weights = canonical
        bound.extend(["--shared-input", "weights=" + str(weights)])
        worker = ["{python}", "-m", "evidence_lens.generator_adapter", "--weights", "{input:weights}", *common]
    else:
        kind = "render"
        rid = args.generation_run
        prev = store / "runs" / rid
        if formal:
            generation = check_generation(store, rid, job, digest)
        bound.extend(["--parent", rid + ("=" + generation["manifest_sha256"] if generation else ""),
                      "--input", f"ply={prev/'payload/scene.ply'}", "--input", f"generation={prev/'payload/generation.json'}"])
        worker = ["{python}", "-m", "evidence_lens.renderer_adapter", "--ply", "{input:ply}", "--generation", "{input:generation}", *common]
    if args.retry_of:
        bound.extend(["--retry-of", args.retry_of])
    command = [sys.executable, "-m", "evidence_lens.recorder", "--root", str(store), "--source-root", str(code),
               "--kind", kind, "--config", str(cfg), "--experiment", job["experiment_id"], "--trial", job["trial_id"], *bound, "--", *worker]

    def launch():
        result = subprocess.run(command, cwd=code, stdout=subprocess.PIPE, text=True)
        require(result.stdout.strip(), f"recorder returned no sealed run (exit {result.returncode})")
        return json.loads(result.stdout)

    if formal:
        receipt = dispatch(store, outgoing, job, digest, kind, generation, args.retry_of, args.resume_preflight, launch)
    else:
        outcome = launch()
        receipt = outgoing_receipt(store, outgoing, Path(outcome["path"]), digest)
    print(json.dumps(receipt), flush=True)
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
