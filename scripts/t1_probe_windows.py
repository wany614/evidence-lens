"""Record the frozen source-only CUDA probe from each original T1 M2 generation on Windows."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require, verify_bundle
from evidence_lens.archive import pack


def probe(index, job, store, returns):
    obj = job["object_id"]
    target = returns / "probes" / obj
    target.mkdir(parents=True, exist_ok=True)
    fit = load(returns / job["trial_id"] / "completed.json")
    require(job["condition"] == "M2-fixed" and fit["trial_id"] == job["trial_id"]
            and fit["job_sha256"] == job["job_sha256"] and fit["status"] == "succeeded", "Wrong original M2 fit")
    gen, rend = fit["runs"]
    g = store / "runs" / gen["run"]
    proof = verify_bundle(g)
    require(proof["manifest_sha256"] == gen["manifest_sha256"] and proof["run"]["trial_id"] == job["trial_id"]
            and gen["run_kind"] == "generation" and rend["run_kind"] == "render", "Unverified generation")
    checkpoint = g / "payload/checkpoints/step-5000.pt"
    rec = load(g / "payload/issued.json")
    ck = rec["checkpoints"]["5000"]
    kit = load(ROOT / "probe-authority.json")
    require(ck["producer_run_id"] == gen["run"] and ck["checkpoint_sha256"] == sha(checkpoint)
            and rec["identity"]["object_id"] == obj and rec["identity"]["condition"] == "M2"
            and rec["identity"]["recipe_sha256"] == kit["recipe_sha256"]
            and sha(ROOT / "cameras.json") == kit["cameras"]["sha256"]
            and kit["entry"] == index["entry"], "Wrong probe camera/checkpoint authority")
    require(all(sha(ROOT / k) == v for k, v in kit["source_sha256"].items()), "Changed probe source")
    cfg = dict(experiment_id="general-t1-confirmation", trial_id="t1-probe-" + obj, object_id=obj,
               recording={"expected_output_bytes": 38000000}, pool=kit["pool"],
               checkpoint_sha256=ck["checkpoint_sha256"], checkpoint_producer=gen["run"],
               cameras_sha256=kit["cameras"]["sha256"], recipe_sha256=kit["recipe_sha256"],
               entry=kit["entry"], frozen_selector=kit["freeze"],
               selection_freeze=kit["selector"], registration_sha256=job["job_sha256"],
               input_roles=["checkpoint", "cameras"])
    work = store / "work/probe-requests" / obj
    work.mkdir(parents=True, exist_ok=True)
    path = work / "config.json"
    if path.exists():
        require(load(path) == cfg, "Conflicting reserved probe")
    else:
        save(path, cfg)
    done = target / "completed.json"
    if done.exists():
        receipt = load(done)
        require(receipt["config_sha256"] == sha(path) and sha(target / receipt["zip"]) == receipt["zip_sha256"],
                "Changed previous probe result")
        return receipt
    log = work / "recorder.json"
    if not log.exists():
        # A prior launch with no recorder response needs investigation before any retry.
        marker = work / "launched"
        with marker.open("x") as stream:
            stream.write(sha(path))
        cmd = [sys.executable, "-B", "-m", "evidence_lens.recorder", "--source-root", str(ROOT),
               "--root", str(store), "--kind", "render", "--config", str(path)]
        for role, pin, asset in (("cameras", kit["cameras"], ROOT / "cameras.json"),
                                 ("checkpoint", {"run_id": gen["run"], "manifest_sha256": gen["manifest_sha256"]}, checkpoint)):
            cmd += ["--parent", pin["run_id"] + "=" + pin["manifest_sha256"],
                    "--input", role + "=" + str(asset), "--input-parent", role + "=" + pin["run_id"]]
        cmd += ["--", "{python}", "-B", "{source}/scripts/b2_probe_worker.py", "compute", "--config", "{config}",
                "--checkpoint", "{input:checkpoint}", "--cameras", "{input:cameras}", "--out", "{out}"]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
        (work / "recorder.stderr").write_text(proc.stderr, encoding="utf-8")
        require(bool(proc.stdout.strip()), "Recorder returned no result; inspect recorded launch")
        save(log, json.loads(proc.stdout.splitlines()[-1]))
    run = load(log)
    rid = run["run_id"]
    root = store / "runs" / rid
    v = verify_bundle(root)
    dest = target / (rid + ".zip")
    packed = pack(root, dest, store)
    receipt = dict(schema="el-t1-probe-return-1", run_id=rid, manifest_sha256=v["manifest_sha256"],
                   zip=dest.name, zip_sha256=packed["container_sha256"], config_sha256=sha(path),
                   status=v["run"]["execution_status"], checkpoint_producer=gen["run"],
                   checkpoint_sha256=ck["checkpoint_sha256"], cameras=kit["cameras"])
    save(done, receipt)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("registry", "store", "returns"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--registry-sha", required=True)
    p.add_argument("--object")
    a = p.parse_args()
    require(os.name == "nt" and sha(a.registry) == a.registry_sha, "Use the verified registry on Windows")
    receipt = load(a.registry)
    for ticket in receipt["jobs"]:
        if a.object is None or ticket["object_id"] == a.object:
            print(json.dumps(probe(receipt, ticket, a.store.resolve(), a.returns.resolve())), flush=True)
