"""Record the synthetic-only resource, resume, cutoff, rendering and topology probes."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens._index import writer_gate
from evidence_lens._verify import require, verify_bundle
from recovery.contract import field_names
from recovery.transport import deliver

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("store", "returns", "work", "gpu-slot"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--series", required=True)
    a = p.parse_args()
    store, returns, work = (x.resolve() for x in (a.store, a.returns, a.work))
    for folder in (store, returns, work):
        folder.mkdir(parents=True, exist_ok=True)
    candidate = CODE / "configs/candidate-recipe.json"
    env = {**os.environ, "PYTHONPATH": str(CODE / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    ledger = {}
    def save_result(label, receipt, expected="succeeded"):
        ledger[label] = receipt
        save(work / "progress.json", {"series": a.series, "runs": ledger})
        require(receipt["status"] == expected, label + " produced " + receipt["status"] + " / " + receipt["run"])
        return receipt
    def raw(label, kind, script, roles, parents=()):
        """Each role holds its concrete file, optional producer receipt, and corresponding worker CLI flag."""
        trial = a.series + "-" + label
        old = [f.parent for f in (store / "runs").glob("*/run.json") if load(f)["trial_id"] == trial]
        if old:
            require(len(old) == 1, "Multiple previous probes of this identity")
            return save_result(label, deliver(store, returns, old[0].name))
        fcfg = work / (label + "-config.json")
        origins = {}
        for role, (path, producer, flag) in roles.items():
            if producer and path.resolve().is_relative_to(store / "runs" / producer["run"] / "payload"):
                origins[role] = {"run_id": producer["run"], "manifest_sha256": producer["manifest_sha256"],
                                 "member": path.relative_to(store / "runs" / producer["run"]).as_posix()}
        cfg = {"experiment_id": "r-recovery-engineering", "trial_id": trial, "scope": "synthetic-only",
               "artifact_origins": origins, "recording": {"expected_output_bytes": 60*1024**2}}
        cfg["request_sha256"] = hashlib.sha256(encoded(cfg)).hexdigest()
        save(fcfg, cfg)
        command = [sys.executable, "-B", "-m", "evidence_lens.recorder", "--root", str(store), "--source-root", str(CODE),
                   "--config", str(fcfg), "--kind", kind]
        pins = {r["run"]:r["manifest_sha256"] for r in parents}
        for _, producer, _ in roles.values():
            if producer:
                pins[producer["run"]] = producer["manifest_sha256"]
        for rid, digest in pins.items():
            command += ["--parent", rid + "=" + digest]
        child = ["{python}", "-B", "{source}/tests/" + script, "--out", "{out}"]
        for key, (path, producer, flag) in roles.items():
            command += ["--shared-input", key + "=" + str(path)]
            if producer:
                command += ["--input-parent", key + "=" + producer["run"]]
            child += [flag, "{input:" + key + "}"]
        with writer_gate(a.gpu_slot.resolve()):
            run = subprocess.run(command + ["--"] + child, env=env, capture_output=True, text=True)
        (work / (label + "-controller.log")).write_text(run.stdout + "\n" + run.stderr, encoding="utf8")
        require(run.stdout.strip(), "Recorder did not seal " + label + ": " + run.stderr[-700:])
        rid = json.loads(run.stdout.splitlines()[-1])["run_id"]
        return save_result(label, deliver(store, returns, rid))
    reg = raw("plan", "archive_verify", "engineering_registration.py", {"candidate":(candidate, None, "--candidate")})
    data = raw("data", "dataset_build", "make_fixture.py", {"candidate":(candidate, reg, "--candidate")}, [reg])
    raw("topology", "archive_verify", "forced_topology.py", {"candidate":(candidate, reg, "--recipe")}, [reg])
    raw("projection", "archive_verify", "ed_probe.py", {}, [reg])
    data_dir = store / "runs" / data["run"] / "payload"
    def cell(label, recipe_file):
        folder = work / ("cell-" + label)
        folder.mkdir(exist_ok=True)
        files = {}
        for key in sorted(field_names("M2")):
            src = data_dir / key
            dst = folder / key
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copyfile(src, dst)
            require(sha(dst) == sha(src), "Fixture copy differs")
            files[key] = {"bytes": dst.stat().st_size, "sha256": sha(dst)}
        job = {"schema":"el-recovery-cell-1", "purpose":"engineering", "object_id":"generated-fixture", "condition":"M2",
               "experiment_id":"r-recovery-engineering", "trial_id":a.series + "-" + label, "stop_step":12,
               "parents":[{"run_id":data["run"], "manifest_sha256":data["manifest_sha256"]}],
               "recipe_sha256":sha(data_dir / recipe_file), "files":files,
               "input_parents":{k:data["run"] for k in [*files, "recipe"]}}
        path = folder / "job.json"
        if path.exists():
            require(load(path) == job, "Engineering ticket changed")
        else:
            save(path, job)
        return folder
    def action(label, cmd, folder, extra, expected="succeeded"):
        command = [sys.executable, "-B", str(CODE/"scripts/recovery_windows.py"), cmd,
                   "--cell", str(folder), "--job-sha256", sha(folder/"job.json"), "--store", str(store),
                   "--returns", str(returns), "--gpu-slot", str(a.gpu_slot), *list(map(str, extra))]
        run = subprocess.run(command, env=env, capture_output=True, text=True)
        (work / (label + "-controller.log")).write_text(run.stdout + "\n" + run.stderr, encoding="utf8")
        require(run.stdout.strip(), label + " controller stopped before receipt: " + run.stderr[-900:])
        return save_result(label, json.loads(run.stdout.splitlines()[-1]), expected)
    full = cell("full", "fixture-recipe.json")
    segmented = cell("segments", "fixture-recipe.json")
    short = cell("timeout", "timeout-recipe.json")
    recipe_args = ["--recipe", data_dir / "fixture-recipe.json"]
    ref = lambda r: r["run"] + "=" + r["manifest_sha256"]
    complete = action("continuous", "optimize", full, recipe_args)
    head = action("first-segment", "optimize", segmented, [*recipe_args, "--segment-end", "4"])
    tail = action("continuation", "optimize", segmented, [*recipe_args, "--resume", ref(head), "--resume-step", "4"])
    all_render = action("render-all", "render", segmented, ["--generation", ref(tail)])
    single = action("render-selected", "render", segmented, ["--generation", ref(tail), "--steps", "4"])
    cutoff = action("cutoff", "optimize", short, ["--recipe", data_dir / "timeout-recipe.json"], "failed")
    cutdoc = load(store / "runs" / cutoff["run"] / "payload/process-wall.json")
    require(cutdoc["resource_cutoff"] and cutoff["exit_code"] == 124, "Unexpected reason for resource cutoff")
    def role(receipt, filename, flag):
        return (store / "runs" / receipt["run"] / "payload" / filename, receipt, flag)
    raw("parity", "archive_verify", "compare_resume.py",
        {"fullmid":role(complete,"checkpoints/step-0004.pt","--full-mid"),
         "segmid":role(head,"checkpoints/step-0004.pt","--seg-mid"),
         "fullend":role(complete,"checkpoints/step-0012.pt","--full-end"),
         "segend":role(tail,"checkpoints/step-0012.pt","--seg-end"),
         "fulltrace":role(complete,"steps.jsonl","--full-log"),"headtrace":role(head,"steps.jsonl","--head-log"),
         "tailtrace":role(tail,"steps.jsonl","--tail-log"),"allviews":role(all_render,"outputs.json","--render-all"),
         "oneview":role(single,"outputs.json","--render-one"),"fixture":role(data,"fixture.json","--fixture")})
    for receipt in ledger.values():
        checked = verify_bundle(store / "runs" / receipt["run"])
        require(checked["manifest_sha256"] == receipt["manifest_sha256"], "Final engineering bundle readback")
    save(returns / (a.series + "-summary.json"), {"scope":"synthetic-only", "series":a.series, "status":"completed",
         "candidate_recipe_sha256":sha(candidate), "expected_cutoff_run":cutoff["run"], "runs":ledger})
    print(json.dumps({k:(v["run"],v["status"]) for k,v in ledger.items()}))

if __name__ == "__main__":
    main()
