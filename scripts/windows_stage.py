"""Run one recorded P2 worker stage after verifying the private input handoff."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["generate", "render"])
    p.add_argument("--handoff", type=Path, required=True, help="newly extracted input handoff")
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--return-dir", type=Path, required=True, help="local non-sync staging")
    p.add_argument("--weights", type=Path)
    p.add_argument("--generation-run")
    p.add_argument("--retry-of")
    args = p.parse_args()
    if os.name != "nt":
        p.error("the P2 Windows worker is required")
    base = args.handoff.resolve()
    store = args.store.resolve()
    outgoing = args.return_dir.resolve()
    job = json.loads((base/"handoff.json").read_text(encoding="utf8"))
    for name, digest in job["files"].items():
        path = (base/name).resolve()
        if not path.is_relative_to(base) or sha(path) != digest:
            raise ValueError(f"changed handoff file: {name}")
    if args.stage == "generate" and not args.weights:
        p.error("--weights is required for generation")
    if args.stage == "render" and not args.generation_run:
        p.error("--generation-run is required to render an exact producer run")
    # Keep large data and compilation caches on the same project volume as the handoff.
    paths = [store, outgoing, sys.executable, args.weights] if args.weights else [store, outgoing, sys.executable]
    if any(Path(path).resolve().drive.casefold() != base.drive.casefold() for path in paths):
        p.error("the handoff, research store, Python environment, weights and return packs need the project volume")
    store.mkdir(parents=True, exist_ok=True)
    outgoing.mkdir(parents=True, exist_ok=True)
    minimum = json.loads((base/"code/configs/storage.json").read_text(encoding="utf8"))["local_free_floor_bytes"]
    if shutil.disk_usage(store).free < minimum:
        raise OSError("project disk free-space floor reached")
    for key, suffix in {"TORCH_EXTENSIONS_DIR":"torch-extensions", "TORCH_HOME":"torch", "HF_HOME":"models",
                        "CUDA_CACHE_PATH":"cuda", "TMP":"temp", "TEMP":"temp"}.items():
        path = store/"cache"/suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    code = base/"code"
    os.environ["PYTHONPATH"] = str(code/"src")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    cfg = code/"configs/windows-b0.json"
    ref = job["reference"]
    bound = ["--parent", f'{ref["run_id"]}={ref["manifest_sha256"]}',
             "--input", f"rgb={base/'input.png'}", "--input-parent", "rgb="+ref["run_id"],
             "--input", f"cameras={base/'cameras.json'}", "--input-parent", "cameras="+ref["run_id"]]
    common_worker = ["--input", "{input:rgb}", "--cameras", "{input:cameras}", "--config", "{config}", "--out", "{out}"]
    if args.stage == "generate":
        kind = "generation"
        bound += ["--shared-input", "weights="+str(args.weights.resolve())]
        worker = ["{python}", "-m", "evidence_lens.generator_adapter", "--weights", "{input:weights}", *common_worker]
    else:
        kind = "render"
        prev = store/"runs"/args.generation_run
        prev_id = json.loads((prev/"run.json").read_text(encoding="utf8"))["run_id"]
        if prev_id != args.generation_run or prev.resolve().parent != (store/"runs").resolve():
            p.error("generation run must identify a run in this store")
        bound += ["--parent", prev_id, "--input", f"ply={prev/'payload/scene.ply'}",
                  "--input", f"generation={prev/'payload/generation.json'}"]
        worker = ["{python}", "-m", "evidence_lens.renderer_adapter", "--ply", "{input:ply}",
                  "--generation", "{input:generation}", *common_worker]
    if args.retry_of:
        bound += ["--retry-of", args.retry_of]
    command = [sys.executable,"-m","evidence_lens.recorder", "--root",str(store),"--source-root",str(code),
               "--kind",kind,"--config",str(cfg),"--experiment",job["experiment_id"],"--trial",job["trial_id"],
               *bound, "--", *worker]
    result = subprocess.run(command, cwd=code, stdout=subprocess.PIPE, text=True)
    print(result.stdout, flush=True)
    if not result.stdout.strip():
        raise SystemExit(result.returncode or 1)
    run = json.loads(result.stdout.strip())
    package = outgoing/(run["run_id"] + ".zip")
    subprocess.run([sys.executable, "-m", "evidence_lens.archive", "pack", run["path"], "--root",
                    str(store), "--output", str(package)], cwd=code, check=True)
    print(json.dumps({"host":platform.node(),"run":run["run_id"], "package":package.name,"sha256":sha(package),
                      "bytes":package.stat().st_size,"status":run["execution_status"]}), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
