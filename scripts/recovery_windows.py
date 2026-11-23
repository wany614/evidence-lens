"""Run a pinned source-only recovery cell through the original Evidence Lens recorder."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from evidence_lens._bundle import encoded, entry, load, save, sha
from evidence_lens._index import writer_gate
from evidence_lens._snapshot import SOURCE_FILES, SOURCE_ROOTS, commit_shared, source_name
from evidence_lens._verify import require, verify_bundle
from recovery.contract import HASH, RUN, SOURCES, cell_manifest, checked_member, pin, validate_recipe
from recovery.transport import deliver, restore


def worker_identity():
    paths = [CODE / name for name in SOURCE_FILES if (CODE / name).is_file()]
    for directory in SOURCE_ROOTS:
        base = CODE / directory
        if base.is_dir():
            paths.extend(p for p in base.rglob("*") if p.is_file() and source_name(p.relative_to(CODE).as_posix()))
    rows = [entry(p, p.relative_to(CODE).as_posix()) for p in sorted(paths, key=lambda p: p.relative_to(CODE).as_posix())]
    dist = importlib.metadata.distribution("gsplat")
    native = Path(dist.locate_file("gsplat/csrc.pyd"))
    require(dist.version == "1.5.3" and native.is_file(), "Use the registered native gsplat 1.5.3 extension")
    return {"source_sha256": hashlib.sha256(encoded(rows)).hexdigest(), "gsplat": dist.version,
            "extension_sha256": sha(native), "torch": importlib.metadata.version("torch"),
            "numpy": importlib.metadata.version("numpy"), "python": sys.version.split()[0]}


def load_run(store, arg):
    rid, sep, digest = arg.partition("=")
    require(sep and RUN.fullmatch(rid) and HASH.fullmatch(digest), "Use the exact RUN=MANIFEST_SHA256")
    root = store / "runs" / rid
    v = verify_bundle(root)
    require(v["manifest_sha256"] == digest, "Sealed run manifest differs")
    return root, v["run"], {"run_id": rid, "manifest_sha256": digest}


def issued(store, parentref, science):
    root, meta, ref = load_run(store, parentref)
    cfg = load(root / "config.resolved.yaml")
    require(meta["run_kind"] == "generation" and cfg["science"] == science, "Checkpoint parent identity")
    issue = load(root / "payload/issued.json")
    require(issue["run_id"] == ref["run_id"] and issue["identity"] == science, "Issued checkpoint index")
    return root, meta, ref, cfg, issue["checkpoints"]


def point_input(item, bundle, meta, ref, store, role, bindings, origins, parents, claims):
    step = item["step"]
    expected = f"checkpoints/step-{step:04d}.pt"
    require(item["path"] == expected and item["producer_run_id"] == ref["run_id"], "Noncanonical issued checkpoint")
    path = bundle / "payload" / expected
    digest = item["checkpoint_sha256"]
    require(path.stat().st_size == item["bytes"] and sha(path) == digest, "Issued checkpoint content differs")
    origins[role] = {**ref, "member": "payload/" + expected, "execution_status": meta["execution_status"]}
    if meta["execution_status"] == "succeeded":
        parents[ref["run_id"]] = ref["manifest_sha256"]
        claims[role] = ref["run_id"]
    else:
        # A sealed failure remains a failure. Record its original ID, manifest, status and artifact
        # explicitly. The legacy recorder accepts successful scientific parents only.
        path = commit_shared(path, store, digest)
    bindings[role] = path


def scientific_request(a):
    cell, store = a.cell.resolve(), a.store.resolve()
    j = cell_manifest(cell, a.job_sha256)
    src = SOURCES[j["condition"]]
    files = j["files"]
    science = {"worker": worker_identity(), "object_id": j["object_id"], "condition": j["condition"], "recipe_sha256": j["recipe_sha256"],
               "generator_source_ids": list(src), "source_rgb_sha256": {s: files["sources/" + s + ".png"]["sha256"] for s in src},
               "source_camera_sha256": files["source_cameras.json"]["sha256"], "init_sha256": files["init.npz"]["sha256"],
               "init_json_sha256": files["init.json"]["sha256"]}
    parents = {p["run_id"]: p["manifest_sha256"] for p in j["parents"]}
    cfg = {"identity": j.get("identity", {}), "experiment_id": j["experiment_id"], "trial_id": j["trial_id"],
           "cell_sha256": a.job_sha256, "science": science, "consumed_wall_seconds": 0.0,
           "recording": {"expected_output_bytes": 256 * 1024**2}}
    require(not ({"experiment_id", "trial_id"} & set(cfg["identity"])), "Use the top-level experiment and trial")
    bindings, claims, origins = {}, {}, {}
    def add(role, name):
        bindings[role] = checked_member(cell, j, name)
        if name in j.get("input_parents", {}):
            claims[role] = j["input_parents"][name]
    if a.action == "optimize":
        kind = "generation"
        require(sha(a.recipe) == j["recipe_sha256"], "Recipe digest differs")
        r = validate_recipe(load(a.recipe))
        bindings["recipe"] = a.recipe.resolve()
        if "recipe" in j.get("input_parents", {}):
            claims["recipe"] = j["input_parents"]["recipe"]
        for role, name in [("init", "init.npz"), ("initmeta", "init.json"), ("cameras", "source_cameras.json")]:
            add(role, name)
        for sid in src:
            add(sid, "sources/" + sid + ".png")
        planned = j.get("stop_step", r["stop_step"])
        require(type(planned) is int and 0 < planned <= r["stop_step"] and planned in r["checkpoints"],
                "The SHA-pinned job must select a positive issued endpoint")
        end = planned if a.segment_end is None else a.segment_end
        require(end in r["checkpoints"] and 0 < end <= planned and (end == planned or j.get("purpose") == "engineering"),
                "Only a registered engineering segment may pause before the job endpoint")
        cfg.update(end_step=end, planned_stop_step=planned, wall_seconds=r["wall_seconds"])
        if a.resume:
            root, meta, ref, oldcfg, points = issued(store, a.resume, science)
            require(oldcfg["cell_sha256"] == a.job_sha256 and a.resume_step is not None and str(a.resume_step) in points
                    and a.resume_step < end, "Resume must name an issued step in this cell")
            item = points[str(a.resume_step)]
            point_input(item, root, meta, ref, store, "resume", bindings, origins, parents, claims)
            guard = root / "payload/process-wall.json"
            used = (load(guard)["cumulative_seconds"] if guard.is_file() else
                    oldcfg["consumed_wall_seconds"] + meta["elapsed_seconds_monotonic"])
            cfg.update(resume_origin={**ref, "step": a.resume_step, "status": meta["execution_status"],
                                      "checkpoint_sha256": item["checkpoint_sha256"]}, consumed_wall_seconds=used)
        else:
            require(a.resume_step is None, "Resume step needs a producer pin")
    else:
        kind = "render"
        add("cameras", "render_cameras.json")
        root, meta, first_ref, gcfg, pts = issued(store, a.generation, science)
        require(meta["execution_status"] == "succeeded" and gcfg["cell_sha256"] == a.job_sha256, "Selected reconstruction")
        parents[first_ref["run_id"]] = first_ref["manifest_sha256"]
        cfg.update(generation=first_ref, render_camera_sha256=files["render_cameras.json"]["sha256"],
                   wall_seconds=gcfg["wall_seconds"])
        wanted = None if a.steps == "all" else set(map(int, a.steps.split(",")))
        selected, seen = {}, set()
        ref = first_ref
        bound = gcfg["end_step"]
        while True:
            require(ref["run_id"] not in seen, "Cyclic checkpoint lineage")
            seen.add(ref["run_id"])
            for key, item in pts.items():
                k = int(key)
                if k > bound or key in selected or (wanted is not None and k not in wanted):
                    continue
                point_input(item, root, meta, ref, store, "c" + key, bindings, origins, parents, claims)
                selected[key] = item
            older = gcfg.get("resume_origin")
            if not older:
                break
            bound = min(bound, older["step"])
            root, meta, ref, gcfg, pts = issued(store, older["run_id"] + "=" + older["manifest_sha256"], science)
            require(gcfg["cell_sha256"] == a.job_sha256, "Changed cell in the checkpoint lineage")
        require(selected and (wanted is None or {int(k) for k in selected} == wanted), "Requested checkpoint was never issued")
        cfg["selected"] = selected
    cfg["artifact_origins"] = origins
    cfg["role_hashes"] = {k: sha(p) for k, p in bindings.items()}
    key = hashlib.sha256(encoded({"stage": kind, "cfg": cfg, "parents": parents})).hexdigest()
    cfg["request_sha256"] = key
    record = store / "recovery-requests" / key
    record.mkdir(parents=True, exist_ok=True)
    for f in (store / "runs").glob("*/config.resolved.yaml"):
        if load(f).get("request_sha256") == key:
            return deliver(store, a.returns, f.parent.name)
    stamp = record / "reserved.json"
    require(not stamp.exists(), "Recorded request already reserved. Inspect original recorder work before any new attempt.")
    save(record / "config.json", cfg)
    save(stamp, {"request_sha256": key, "kind": kind, "cell_sha256": a.job_sha256})
    args = [sys.executable, "-B", "-m", "evidence_lens.recorder", "--root", str(store), "--source-root", str(CODE),
            "--storage-policy", str(CODE / "configs/storage.json"), "--kind", kind, "--config", str(record / "config.json")]
    for rid, digest in parents.items():
        args += ["--parent", rid + "=" + digest]
    internal = ["{python}", "-B", "{source}/scripts/recovery_windows.py", "_guard", "--kind", kind,
                "--config", "{config}", "--out", "{out}"]
    for role, path in bindings.items():
        args += ["--shared-input", role + "=" + str(path)]
        if role in claims:
            args += ["--input-parent", role + "=" + claims[role]]
        internal += ["--member", role, "{input:" + role + "}"]
    env = {**os.environ, "PYTHONPATH": str(CODE / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([*args, "--", *internal], env=env, capture_output=True, text=True)
    (record / "recorder.stdout").write_text(result.stdout, encoding="utf8")
    (record / "recorder.stderr").write_text(result.stderr, encoding="utf8")
    require(result.stdout.strip(), "Recorder returned no sealed run. Inspect " + str(record))
    rid = json.loads(result.stdout.splitlines()[-1])["run_id"]
    return deliver(store, a.returns, rid)


def guarded(a):
    require(RUN.fullmatch(os.environ.get("EL_RUN_ID", "")) and os.environ.get("EL_RUN_KIND") == a.kind, "Recorder context")
    cfg = load(a.config)
    started = time.perf_counter()
    prior = cfg["consumed_wall_seconds"]
    remaining = cfg["wall_seconds"] - prior
    code, cutoff = 124, True
    if remaining > 0:
        cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "_compute", "--kind", a.kind,
               "--out", str(a.out), "--config", str(a.config)]
        for k, v in a.member:
            cmd += ["--member", k, v]
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
        try:
            code = proc.wait(timeout=max(0, remaining - (time.perf_counter() - started)))
            cutoff = time.perf_counter() - started >= remaining
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    elapsed = time.perf_counter() - started
    save(a.out / "process-wall.json", {"process_seconds": elapsed, "prior_seconds": prior, "cumulative_seconds": prior + elapsed,
         "budget_seconds": cfg["wall_seconds"], "resource_cutoff": cutoff, "exit_code": 124 if cutoff else code})
    return 124 if cutoff else code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="action", required=True)
    for name in ("optimize", "render"):
        c = subs.add_parser(name)
        for k in ("cell", "store", "returns", "gpu-slot"):
            c.add_argument("--" + k, type=Path, required=True)
        c.add_argument("--job-sha256", required=True)
        if name == "optimize":
            c.add_argument("--recipe", type=Path, required=True)
            c.add_argument("--resume")
            c.add_argument("--resume-step", type=int)
            c.add_argument("--segment-end", type=int)
        else:
            c.add_argument("--generation", required=True)
            c.add_argument("--steps", default="all")
    t = subs.add_parser("restore-assets")
    t.add_argument("--receipt", type=Path, required=True)
    t.add_argument("--store", type=Path, required=True)
    for n in ("_guard", "_compute"):
        c = subs.add_parser(n)
        c.add_argument("--kind", choices=("generation", "render"), required=True)
        c.add_argument("--config", type=Path, required=True)
        c.add_argument("--out", type=Path, required=True)
        c.add_argument("--member", nargs=2, action="append", required=True)
    a = p.parse_args()
    if a.action == "restore-assets":
        print(json.dumps(restore(a.receipt, a.store)))
        return 0
    if a.action == "_guard":
        return guarded(a)
    if a.action == "_compute":
        require(RUN.fullmatch(os.environ.get("EL_RUN_ID", "")) and os.environ.get("EL_RUN_KIND") == a.kind,
                "Numeric stage must run through the recorder")
        from recovery.engine import generation, render
        members = {k: Path(v) for k, v in a.member}
        require(len(members) == len(a.member), "Duplicate role")
        (generation if a.kind == "generation" else render)(a.config, members, a.out)
        return 0
    require(os.name == "nt" and a.store.resolve().drive.casefold() == Path(sys.executable).drive.casefold(),
            "Use the project CUDA interpreter and store on the project volume")
    with writer_gate(a.gpu_slot.resolve()):
        receipt = scientific_request(a)
    print(json.dumps(receipt))
    return 0 if receipt["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
