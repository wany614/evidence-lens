"""Prepare and receive frozen, source-only T0 reconstruction jobs in a new recorder store."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile
import sys
import tempfile

from evidence_lens._bundle import encoded, load, sha
from evidence_lens._verify import require, verify_bundle
from make_handoff import publish
from stage3_reference import perform

PROJECT = Path(__file__).resolve().parents[1]
INIT = "024a4a08552e748afafc3ead27ba745d6f96e84757572e5fbf9aec387653d086"
RECIPE = "57730c706aaaa2f88c7f39ff179ba04ff971f31e599bf3409a3d545c61d72c98"
WORKER = "aa65bc5a412cdafdac2e9ffa5f64f675ba8253904100dafbc086284b5a9632c1"
GEOMETRY = "fbcb2b8b51ea7f7ede80c5b115e8581e7a4c4611fa83eecf56d4db88b2ef5840"
IDS = {"M2": ["s00", "s03"], "M4": ["s00", "s01", "s02", "s03"]}
PACK_SCRIPTS = ("t0_worker_pack.py", "t0_worker_run.py")


def short(p):
    return {k: p[k] for k in ("run_id", "manifest_sha256")}


def member(store, p, name=None):
    from recovery_inputs import member as checked

    f = checked(store, short(p), name if name else p["member"])
    if not name or name == p.get("member"):
        require(sha(f) == p["sha256"], "Selected member changed")
    return f


def pin(store, r, name):
    return {**short(r), "member": name, "sha256": sha(member(store, r, name))}


HISTORICAL_COMMIT = "bc8da49815751fa7a6e191d0dc17c9f987f9c825"


def source_file(name):
    if name.startswith("historical/"):
        relative = name.removeprefix("historical/")
        return relative, subprocess.check_output(
            ["git", "-C", str(PROJECT), "show", HISTORICAL_COMMIT + ":" + relative]
        )
    return "scripts/" + name, (PROJECT / "scripts" / name).read_bytes()


def staged(src, names):
    for n in names:
        relative, data = source_file(n)
        f = src / relative
        f.parent.mkdir(exist_ok=True, parents=True)
        f.write_bytes(data)


def job_record(store, kind, cfg, inputs, parents, source_files, command):
    # A reserved trial may be resumed from a sealed result after a controller interruption.
    from evidence_lens._bundle import LABEL

    t = cfg["trial_id"]
    require(bool(LABEL.fullmatch(t)), "Invalid trial id")
    directory = store / "t0" / "requests" / t
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "request.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reserved = directory / "requested.json"
        sel = directory / "run.json"
        identity = {
            "cfg": cfg,
            "parents": parents,
            "source": {s: hashlib.sha256(source_file(s)[1]).hexdigest() for s in source_files},
            "inputs": {
                k: sha(f) if f.is_file() else cfg["origin_tree_sha256"] for k, (f, _) in inputs.items()
            },
        }
        request = hashlib.sha256(encoded(identity)).hexdigest()
        if sel.exists():
            old = load(sel)
            require(
                old["request_sha256"] == request
                and verify_bundle(store / "runs" / old["run_id"])["manifest_sha256"]
                == old["manifest_sha256"],
                "Changed completed trial",
            )
            return short(old)
        if reserved.exists():
            require(load(reserved) == {"request_sha256": request}, "Conflicting reserved job")
            for p in (store / "runs").glob("*/config.resolved.yaml"):
                if load(p).get("t0_request_sha256") == request:
                    v = verify_bundle(p.parent)
                    require(
                        v["run"]["execution_status"] == "succeeded",
                        "Earlier recorded attempt failed: " + p.parent.name,
                    )
                    ans = {"run_id": p.parent.name, "manifest_sha256": v["manifest_sha256"]}
                    publish(sel, encoded({**ans, "request_sha256": request}))
                    return ans
            raise ValueError("Inspect existing reserved trial before another attempt: " + t)
        publish(reserved, encoded({"request_sha256": request}))
        cfg = {**cfg, "t0_request_sha256": request, "recording": {"expected_output_bytes": 3 * 1024**2}}
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            src = Path(tmp) / "source"
            staged(src, source_files)
            ans = perform(store, PROJECT, src, cfg, kind, inputs, parents, command)
        publish(sel, encoded({**ans, "request_sha256": request}))
        return ans


def packcmd(action, roles):
    cmd = [
        "{python}",
        "-B",
        "{source}/scripts/t0_worker_pack.py",
        action,
        "--config",
        "{config}",
        "--out",
        "{out}",
    ]
    for k in roles:
        cmd += ["--input", k, "{input:" + k + "}"]
    return cmd


def historic_bridge(store, old):
    require(store != old and not store.is_relative_to(old), "The historical store is read only")
    control = old / "work/r-control"
    h = load(control / "r2-scale-decay-02-set1-M2-ticket.json")
    norm = load(control / "r2-scale-decay-02-set1-M2-normalize.json")
    origins = {
        "old-freeze.json": load(control / "freeze.json"),
        "recipe.json": h["recipe"],
        "worker.zip": h["worker"],
        "joined.json": load(control / "all-scientific-returns-pin.json"),
        "d-natural-init.json": h["initialization"],
        "depth-probe.json": norm["depth_probe"],
        "resource-probe.json": norm["resource_probe"],
    }
    require(
        origins["worker.zip"]["sha256"] == WORKER
        and origins["recipe.json"]["sha256"] == RECIPE
        and origins["old-freeze.json"]["sha256"]
        == "2669469cc51eb7d0bcfc95d175b5f0b052373800e2a903da669ffcf3f3b8f07a"
        and origins["joined.json"]["sha256"]
        == "7c2c7b4e73211597889218a978a6b70256bcf38a63bcb61621bd4a6a4eb592d3",
        "Different historical study",
    )
    for cell in load(member(old, origins["joined.json"]))["cells"]:
        gp = cell["original_runs"]["generation"]
        origins["historical/" + gp["run_id"] + ".json"] = pin(old, gp, "payload/reconstruction.json")
    hand = load(member(old, h["handoff"]))
    require(
        hand["object_id"] == "set1"
        and hand["condition"] == "M2"
        and hand["generator_source_ids"] == IDS["M2"],
        "Wrong historical D cell",
    )
    for n in ["source_cameras.json", "render_cameras.json", "sources/s00.png", "sources/s03.png"]:
        e = pin(old, h["handoff"], "payload/handoff/" + n)
        require(e["sha256"] == hand["files"][n]["sha256"], "Different historical source")
        origins[n] = e
    with tempfile.TemporaryDirectory(dir=store / "work") as tmp:
        src = Path(tmp) / "origins"
        src.mkdir()
        for role, p in origins.items():
            f = member(old, p)
            target = src / "members" / role
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, target)
            root = f.parents[len(Path(p["member"]).parts) - 1]
            dst = src / "proofs" / p["run_id"]
            if not dst.exists():
                dst.mkdir(parents=True)
                for n in ("run.json", "completion.json", "manifest.json"):
                    shutil.copyfile(root / n, dst / n)
        digest = hashlib.sha256(
            encoded({p.relative_to(src).as_posix(): sha(p) for p in sorted(src.rglob("*")) if p.is_file()})
        ).hexdigest()
        cfg = {
            "experiment_id": "t0-historical-reconstruction-readback",
            "trial_id": "t0-historical-source-bridge-1",
            "origins": origins,
            "origin_tree_sha256": digest,
        }
        r = job_record(
            store,
            "archive_verify",
            cfg,
            {"origins": (src, None)},
            {},
            PACK_SCRIPTS,
            packcmd("bridge", ["origins"]),
        )
    result = pin(store, r, "payload/bridge.json")
    publish(store / "t0" / "origin.json", encoded(result))
    return result


def prepare(store, bridge, source, obj, cond, interpreter, freeze=None, qualification=None, forced=False):
    ids = IDS[cond]
    b = load(member(store, bridge))
    require(
        b["schema"] == "el-t0-recovery-bridge-1"
        and b["recipe_sha256"] == RECIPE
        and sha(PROJECT / "scripts/recovery_triangulate.py") == GEOMETRY
        and sha(PROJECT / "protocol/recovery-initialization-v1.json") == INIT,
        "Changed frozen source algorithm",
    )
    if forced:
        require(
            (obj, cond) == ("set1", "M2")
            and b["completed_final_recipe_fallback"] == 0
            and short(source) == short(bridge),
            "D engineering probe already covered or invalid",
        )
    else:
        fr = load(member(store, freeze))
        dq = load(member(store, qualification))
        require(
            fr["schema"] == "el-t0-freeze-1"
            and dq["schema"] == "el-t0-reconstruction-readback-1"
            and dq["forced_fallback_fixture"]
            and dq["verified"]
            and dq["completed_steps"] == 5000,
            "Missing frozen T0 authorization or D qualification",
        )
        require(
            obj in fr["object_ids"]
            and obj not in ("set1", "set6", "scene_057_offset_bell_cage")
            and cond in fr["conditions"]
            and fr["target_ids"] == [f"v{i:02d}" for i in range(1, 7)]
            and fr["recipe_sha256"] == RECIPE
            and fr["stop_step"] == 5000
            and fr["target_reference_permission"] == "closed"
            and freeze["member"] == "payload/freeze.json"
            and fr["run_id"] == freeze["run_id"],
            "Unregistered T0 object, method or endpoint",
        )
        registered_d = load(store / "t0/forced-fallback/selection.json")["readback"]
        require(
            qualification == registered_d
            and dq["object_id"] == "set1"
            and dq["condition"] == "M2"
            and not dq["scientific_sample"]
            and dq["actual_method"] == "endpoint_stratified_inverse_depth"
            and dq["initial_points"] == 2048
            and dq["origins"]["bridge"] == short(bridge),
            "Incorrect independent D qualification",
        )
        provenance = load(member(store, source))
        require(
            source["member"] == "payload/source-provenance.json"
            and provenance["object_id"] == obj
            and provenance["condition"] == cond
            and provenance["roster_sha256"] == fr["roster_sha256"]
            and provenance["freeze_sha256"] == freeze["sha256"]
            and provenance["schema"] == "el-t0-dtu-source-1"
            and provenance["run_id"] == source["run_id"]
            and provenance["reference_photo_reads"] == 0
            and provenance["source_ids"] == ids
            and set(provenance["sources"]) == set(ids)
            and provenance["physical_family_id"] == fr["families"][obj]
            and all(
                sha(PROJECT / "scripts" / f) == fr["new_scientific_files"]["scripts/" + f]
                for f in (*PACK_SCRIPTS, "t0_worker_init.py")
            ),
            "Source object/condition differs",
        )
    trial = ("t0-d-forced-fallback-" if forced else "t0-frozen-") + obj + "-" + cond
    sc = member(store, source, "payload/source_cameras.json")
    rc = member(store, source, "payload/render_cameras.json")
    bind = {
        "cameras": (sc, "source"),
        **{s: (member(store, source, "payload/sources/" + s + ".png"), "source") for s in ids},
    }
    if not forced:
        require(
            provenance["source_camera_sha256"] == sha(sc)
            and provenance["render_camera_sha256"] == sha(rc)
            and all(provenance["sources"][k]["rgb"]["sha256"] == sha(bind[k][0]) for k in ids),
            "Source provenance pixels or calibration differ",
        )
    pins = {"source": short(source)}
    if not forced:
        pins.update({"freeze": short(freeze), "qualification": short(qualification)})
    cfg = {
        "experiment_id": "t0-source-initialization",
        "trial_id": trial + "-init",
        "scene_family_id": obj,
        "condition_id": cond,
        "generator_source_ids": ids,
        "registration": short(freeze or bridge),
        "source_parent": short(source),
        "recipe_sha256": RECIPE if forced else INIT,
        **({} if forced else {"optimization_recipe_sha256": RECIPE}),
        "initialization_protocol_sha256": INIT,
        "source_camera_sha256": sha(sc),
        "source_rgb_sha256": {s: sha(bind[s][0]) for s in ids},
        "forced_fallback_fixture": forced,
    }
    if forced:
        bind["previous"] = (member(store, bridge, "payload/d-natural-init.json"), "source")
    bind["rules"] = (PROJECT / "protocol/recovery-initialization-v1.json", None)
    command = [
        str(interpreter.absolute()),
        "-B",
        "{source}/scripts/t0_worker_init.py",
        "--config",
        "{config}",
        "--protocol",
        "{input:rules}",
        "--cameras",
        "{input:cameras}",
        "--out",
        "{out}",
    ]
    if forced:
        command += ["--previous", "{input:previous}"]
    for s in ids:
        command += ["--source", s, "{input:" + s + "}"]
    init = job_record(
        store, "derivation", cfg, bind, pins, ("t0_worker_init.py", "recovery_triangulate.py"), command
    )
    ip = pin(store, init, "payload/init.json")
    inputs = {k: v for k, v in bind.items() if k in ("cameras", *ids)}
    inputs.update(
        {
            "rendercameras": (rc, "source"),
            "init": (member(store, ip, "payload/init.npz"), "init"),
            "initmeta": (member(store, ip), "init"),
        }
    )
    if short(bridge) == short(source):
        owner = "source"
    else:
        owner = "bridge"
        pins["bridge"] = short(bridge)
    inputs.update(
        {
            "recipe": (member(store, bridge, "payload/recipe.json"), owner),
            "worker": (member(store, bridge, "payload/worker.zip"), owner),
        }
    )
    pins["init"] = init
    cfg = {
        **cfg,
        "experiment_id": "t0-reconstruction-dispatch",
        "trial_id": trial + "-dispatch",
        "scientific_trial_id": trial,
        "windows_experiment": "t0-d-engineering-forced-fallback" if forced else "t0-frozen-reconstruction",
        "requested_render_steps": [5000],
        "inputs_origin": "source-only",
        "initialization": ip,
        "source": source,
        "freeze": freeze,
        "D_qualification": qualification,
        "worker_sha256": WORKER,
        "windows_parents": [pins[k] for k in sorted(pins)],
        "input_origin_ids": {k: pins[owner]["run_id"] for k, (_, owner) in inputs.items()},
    }
    d = job_record(store, "dataset_build", cfg, inputs, pins, PACK_SCRIPTS, packcmd("dispatch", inputs))
    ticket = pin(store, d, "payload/ticket.json")
    dst = store / "transport/Handoffs/t0-v1" / trial
    for n in ("dispatch.zip", "t0_worker_run.py", "ticket.json", "READY.json"):
        publish(dst / n, member(store, d, "payload/" + n).read_bytes())
    publish(dst / "dispatch-pin.json", encoded(ticket))
    return {"ticket": ticket, "path": str(dst), **load(member(store, ticket))}


READBACK_SOURCE = (
    ("t0_reconstruct_readback.py",)
    + tuple(
        "historical/scripts/" + x + ".py"
        for x in ("recovery_outputs", "recovery_inputs", "recovery_receive")
    )
    + tuple(
        "historical/src/evidence_lens/" + x + ".py"
        for x in (
            "__init__",
            "_bundle",
            "_verify",
            "_metrics",
            "_capacity",
            "_index",
            "_snapshot",
            "_dedup",
            "archive",
            "stage3_gaussians",
            "scoring_io",
            "features",
            "multisource",
            "multisource_io",
        )
    )
)


def authorize_history(store, history, registry, expected):
    from t0_reconstruct_history import missing_parents

    require(registry.resolve() == (store / "transport/Handoffs/t0-v1/frozen-12.json").resolve()
            and sha(registry) == expected, "Expected the independently frozen dispatch registry")
    doc = load(registry)
    tickets = [j["ticket"] for j in doc["jobs"]]
    parents = {f"d{i}": short(ticket) for i, ticket in enumerate(tickets)}
    cfg = {
        "experiment_id": "t0-cross-store-archive-verification",
        "trial_id": "t0-verify-historical-anchors-v1",
        "dispatch_registry_sha256": expected,
        "freeze": doc["freeze"],
        "historical_parents": missing_parents(store, tickets),
    }
    sources = ("t0_reconstruct_history.py",) + tuple(
        "historical/src/evidence_lens/" + n + ".py" for n in ("__init__", "_bundle", "_verify", "_metrics")
    )
    command = ["{python}", "{source}/scripts/t0_reconstruct_history.py", "--config", "{config}", "--registry",
               "{input:registry}", "--store", str(store), "--history", str(history), "--out", "{out}"]
    recorded = job_record(store, "archive_verify", cfg, {"registry": (registry, None)}, parents, sources, command)
    proof = pin(store, recorded, "payload/cross-store-audit.json")
    publish(store / "t0/historical-lineage.json", encoded(proof))
    return proof


def retain_original_return(t, trial, returns, summary_sha, retained_base):
    """Authenticate the Windows summary, side receipts, ZIPs and exact shared-input inventory."""
    from evidence_lens._bundle import HASH, RUN_ID, local_path
    from t0_worker_pack import spec

    require(
        returns.name == trial
        and t["trial_id"] == trial
        and HASH.fullmatch(summary_sha)
        and sha(returns / "completed.json") == summary_sha,
        "Wrong return or summary hash",
    )
    summary = load(returns / "completed.json")
    require(
        summary["schema"] == "el-t0-windows-return-1"
        and summary["trial_id"] == trial
        and summary["package_sha256"] == t["package"]["sha256"]
        and summary["job_sha256"] == t["job_sha256"],
        "Windows did not run the selected ticket",
    )
    legs = summary["runs"]
    require(
        [l["run_kind"] for l in legs] in (["generation"], ["generation", "render"])
        and len(set(l["run"] for l in legs)) == len(legs)
        and all(
            RUN_ID.fullmatch(l["run"])
            and HASH.fullmatch(l["sha256"])
            and HASH.fullmatch(l["manifest_sha256"])
            for l in legs
        ),
        "Incomplete or duplicate Windows stage identities",
    )
    successful = len(legs) == 2 and all(l["status"] == "succeeded" for l in legs)
    require(
        summary["status"] == ("succeeded" if successful else "failed")
        and (successful or legs[-1]["status"] != "succeeded"),
        "Different return terminal state",
    )
    allowed = set()
    staged = []
    for leg in legs:
        rid = leg["run"]
        side = rid + ".receipt.json"
        arch = rid + ".zip"
        allowed.update((side, arch))
        doc = load(returns / side)
        require(
            doc["schema"] == "el-recovery-return-1"
            and doc["run"] == rid
            and doc["trial_id"] == trial
            and doc["run_kind"] == leg["run_kind"]
            and doc["cell_sha256"] == t["job_sha256"]
            and doc["manifest_sha256"] == leg["manifest_sha256"]
            and doc["status"] == leg["status"]
            and doc["package"] == arch
            and doc["sha256"] == leg["sha256"],
            "Original receipt differs from independent summary",
        )
        for digest, desc in doc["shared_inputs"].items():
            if "file" in desc:
                name = f"assets/sha256/{digest[:2]}/{digest}"
                require(HASH.fullmatch(digest) and desc["file"] == name, "Unexpected shared member")
                allowed.add(name)
        staged.append((doc, returns / arch))
    require(set(summary["files"]) == allowed, "Missing or surplus scientific transfer members")
    for name, p in summary["files"].items():
        f = returns / local_path(name)
        require(
            not f.is_symlink() and f.resolve().is_relative_to(returns.resolve()) and spec(f) == p,
            "Changed return " + name,
        )
    retained = retained_base / trial
    for name in sorted(allowed | {"completed.json"}):
        original = returns / name
        if original.resolve() != (retained / name).resolve():
            publish(retained / name, original.read_bytes())
    returns = retained
    staged = [(doc, returns / archive.name) for doc, archive in staged]
    return summary, staged, retained


def accept(store, ticket, returns, summary_sha, interpreter, previous_audit=None, history=None):
    from evidence_lens._bundle import asset_path
    from evidence_lens._index import writer_gate
    from evidence_lens._verify import verify_shared
    from evidence_lens.archive import receive
    from recovery_receive import restore_shared
    from evidence_lens import _capacity

    t = load(member(store, ticket))
    ready = load(member(store, ticket, "payload/READY.json"))
    trial = ready["trial_id"]
    summary, staged, returns = retain_original_return(t, trial, returns, summary_sha, store / "transport/Returns/t0-v1")
    historical_receipt = None if ready["forced_fallback_fixture"] else load(store / "t0/historical-lineage.json")
    originals = {
        "dispatch": short(ticket),
        "bridge": short(load(store / "t0/origin.json")),
        "source": short(ready["source"]),
        "init": short(ready["initialization"]),
    }
    success_parents = {p["run_id"]: p for p in originals.values()}
    for doc, arch in staged:
        rid = doc["run"]
        rp = {"run_id": rid, "manifest_sha256": doc["manifest_sha256"]}
        originals[doc["run_kind"]] = rp
        with zipfile.ZipFile(arch) as z:
            inp = json.loads(z.read("inputs.json"))
            require(
                hashlib.sha256(z.read("manifest.json")).hexdigest() == rp["manifest_sha256"],
                "Returned manifest differs",
            )
            needed = sum(
                e["bytes"] for hx, e in doc["shared_inputs"].items() if not asset_path(store, hx).exists()
            )
        with writer_gate(store):
            cap = _capacity.check(
                store, needed, _capacity.policy(PROJECT / "configs/storage.json", PROJECT)
            )
            require(not cap["reason"], str(cap["reason"]))
            restore_shared(returns, store, doc, inp)
        policy = PROJECT / "configs/storage.json"
        if historical_receipt is None:
            receipt = receive(arch, store, doc["sha256"], policy)
        else:
            from t0_reconstruct_intake import receive_cross_store

            receipt = receive_cross_store(arch, store, doc["sha256"], policy, (history or PROJECT / ".research").resolve(),
                                          historical_receipt, ready["freeze"])
        require(
            receipt["status"] in ("accepted", "identical")
            and receipt["run_id"] == rid
            and receipt["manifest_sha256"] == rp["manifest_sha256"],
            "Original archive did not pass intake: " + str(receipt.get("reason")),
        )
        v = verify_bundle(store / "runs" / rid)
        verify_shared(v, store)
        require(v["run"]["execution_status"] == doc["status"], "Original terminal state differs")
        if doc["status"] == "succeeded":
            success_parents[rid] = rp
    if historical_receipt:
        success_parents[historical_receipt["run_id"]] = short(historical_receipt)
    parents = {
        f"p{i}": x for i, x in enumerate(sorted(success_parents.values(), key=lambda x: x["run_id"]))
    }
    cfg = {
        "experiment_id": "t0-reconstruction-qualification"
        if ready["forced_fallback_fixture"]
        else "t0-frozen-reconstruction-readback",
        "trial_id": trial + "-readback",
        "scene_family_id": ready["object_id"],
        "condition_id": ready["condition"],
        "job_sha256": t["job_sha256"],
        "return_sha256": summary_sha,
        "originals": originals,
    }
    if historical_receipt:
        cfg["historical_verification"] = historical_receipt
        cfg["intake_code_sha256"] = sha(PROJECT / "scripts/t0_reconstruct_intake.py")
    if previous_audit:
        failed_id, failed_hash = previous_audit.split("=", 1)
        previous = verify_bundle(store / "runs" / failed_id)
        old_conf = load(store / "runs" / failed_id / "config.resolved.yaml")
        require(
            previous["manifest_sha256"] == failed_hash
            and previous["run"]["execution_status"] == "failed"
            and previous["run"]["trial_id"] == cfg["trial_id"]
            and old_conf["originals"] == originals
            and old_conf["return_sha256"] == summary_sha,
            "Only a failed readback of this same exact return may be corrected",
        )
        cfg["previous_engineering_audit"] = {"run_id": failed_id, "manifest_sha256": failed_hash}
        cfg["trial_id"] += "-corrected"

    probe = subprocess.run(
        [str(interpreter.absolute()), "-B", "-c", "import torch; print(torch.__version__)"],
        capture_output=True,
        text=True,
    )
    require(
        probe.returncode == 0 and probe.stdout.strip().split("+")[0] == "2.8.0",
        "Use the recorded original Torch 2.8.0 CPU audit interpreter",
    )
    cfg["readback_torch_version"] = probe.stdout.strip()
    owner = next(k for k, v in parents.items() if v == short(ticket))
    inputs = {
        "job": (member(store, ticket, "payload/job.json"), owner),
        "summary": (returns / "completed.json", None),
    }
    r = job_record(
        store,
        "archive_verify",
        cfg,
        inputs,
        parents,
        READBACK_SOURCE + (("t0_reconstruct_intake.py",) if historical_receipt else ()),
        [
            str(interpreter.absolute()),
            "-B",
            "{source}/scripts/t0_reconstruct_readback.py",
            "--config",
            "{config}",
            "--store",
            str(store),
            "--out",
            "{out}",
        ],
    )
    readback = pin(store, r, "payload/reconstruction-readback.json")
    data = load(member(store, readback))
    selection = {
        "schema": "el-t0-reconstruction-selection-1",
        "object_id": ready["object_id"],
        "condition": ready["condition"],
        "trial_id": trial,
        "readback": readback,
        "status": data["status"],
        "case": pin(store, r, "payload/case/case.json") if data["verified"] else None,
        "ticket": ticket,
        "originals": originals,
        "return_sha256": summary_sha,
    }
    dest = (
        store
        / "t0"
        / (
            "forced-fallback"
            if ready["forced_fallback_fixture"]
            else "reconstruction/" + ready["object_id"] + "/" + ready["condition"]
        )
        / "selection.json"
    )
    publish(dest, encoded(selection))
    return selection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest="stage", required=True)
    for action in ("bridge", "force-D", "prepare", "verify-history", "receive"):
        q = sp.add_parser(action)
        q.add_argument("--store", type=Path, required=True)
        if action in ("bridge", "verify-history"):
            q.add_argument("--historical-store", type=Path, required=True)
            if action == "verify-history":
                q.add_argument("--registry", type=Path, required=True)
                q.add_argument("--registry-sha256", required=True)
        elif action == "receive":
            for flag in ("ticket", "returns", "audit-python"):
                q.add_argument("--" + flag, type=Path, required=True)
            q.add_argument("--summary-sha256", required=True)
            q.add_argument("--historical-store", type=Path, default=PROJECT / ".research")
            q.add_argument("--previous-audit", help="Prior failed local audit RUN=MANIFEST_SHA256")
        else:
            q.add_argument("--geometry-python", type=Path, required=True)
        if action == "prepare":
            for flag in ("freeze", "source", "qualification"):
                q.add_argument("--" + flag, type=Path, required=True)
            q.add_argument("--object", required=True)
            q.add_argument("--condition", choices=IDS, required=True)
    a = p.parse_args()
    s = a.store.resolve()
    (s / "work").mkdir(parents=True, exist_ok=True)
    if a.stage == "bridge":
        v = historic_bridge(s, a.historical_store.resolve())
    elif a.stage == "force-D":
        b = load(s / "t0/origin.json")
        v = prepare(s, b, b, "set1", "M2", a.geometry_python, forced=True)
    elif a.stage == "verify-history":
        v = authorize_history(s, a.historical_store.resolve(), a.registry.resolve(), a.registry_sha256)
    elif a.stage == "receive":
        v = accept(s, load(a.ticket), a.returns.resolve(), a.summary_sha256, a.audit_python,
                   a.previous_audit, a.historical_store)
    else:
        v = prepare(
            s,
            load(s / "t0/origin.json"),
            load(a.source),
            a.object,
            a.condition,
            a.geometry_python,
            load(a.freeze),
            (lambda q: q.get("readback", q))(load(a.qualification)),
        )
    print(json.dumps(v, sort_keys=True))


if __name__ == "__main__":
    main()
