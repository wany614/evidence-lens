"""Authenticate Windows recovery bundles, project source-only selector inputs and seal view choices."""

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

from b2_worker_queue import completed
from make_handoff import publish
from recovery_receive import restore_shared
from t0_reconstruct import retain_original_return, short
from t0_reconstruct_intake import receive_verified_zip
from t0_reconstruct_readback import finish
from t1_access import POOL, S0
from t1_study import (
    DEFAULT,
    PROJECT,
    RUN_PYTHON,
    authority,
    bind,
    locate,
    options,
    selected,
    stage,
)

from evidence_lens._bundle import load, save, sha
from evidence_lens._index import writer_gate
from evidence_lens._snapshot import commit_shared
from evidence_lens._verify import require, verify_bundle

NATIVE = ["s00", "s03", *[f"c{i:03}" for i in POOL], *[f"q{i:02}" for i in range(4)]]


@lru_cache(maxsize=80)
def external_anchor(history, rid, digest):
    directory = history / "runs" / rid
    proof = verify_bundle(directory.resolve())
    require(
        proof["manifest_sha256"] == digest and proof["run"]["run_id"] == rid, "Historical manifest mismatch"
    )
    return proof


def boundary(roots, parents):
    store, history = roots
    external, seen = {}, set()
    stack = list(parents)
    while stack:
        pin = stack.pop()
        rid = pin["run_id"]
        if rid in seen:
            continue
        seen.add(rid)
        local = store / "runs" / rid
        if local.is_dir():
            require(sha(local / "manifest.json") == pin["manifest_sha256"], "Changed current parent")
            d = load(local / "inputs.json")
            stack.extend(d["parents"])
            if d.get("retry_of"):
                stack.append(d["retry_of"])
        else:
            external[rid] = external_anchor(history, rid, pin["manifest_sha256"])
    return external


def import_zip(roots, path, digest, expected_id, expected_manifest, g0, run_inputs):
    anchors = boundary(roots, run_inputs["parents"])
    receipt = receive_verified_zip(path, roots[0], digest, PROJECT / "configs/storage.json", anchors, g0)
    require(
        receipt["status"] in ("accepted", "identical")
        and receipt["run_id"] == expected_id
        and receipt["manifest_sha256"] == expected_manifest,
        f"Original archive import: {receipt}",
    )


def readback(roots, job, row, returns, ledger):
    store = roots[0]
    obj, cond = job["object_id"], job["condition"]
    name = "t1-readback-" + obj + "-" + cond
    if (store / "work/controller" / name / "selected.json").is_file():
        existing = selected(store, name)
        require(
            load(locate(roots, existing))["return_summary_sha256"] == row["completed_sha256"],
            "Already-sealed return changed",
        )
        return existing
    g0, _, fr = authority(roots)
    tp = job["ticket"]
    ticket = load(locate(roots, tp))
    ready = load(locate(roots, bind(roots, tp, "payload/READY.json")))
    trial = job["trial_id"]
    require(
        job["job_sha256"] == row["job_sha256"] == ticket["job_sha256"] == ready["job_sha256"]
        and ready["object_id"] == obj
        and ready["condition"] == cond[:2]
        and ready["freeze"] == fr["sources"]["reconstruction"]
        and ready["target_reference_members"] == 0,
        "Returned scientific identity differs",
    )
    summary, staged, local = retain_original_return(
        ticket, trial, returns / trial, row["completed_sha256"], store / "transport/Returns/t1-v1"
    )
    require(completed(job, local) == row and summary["runs"] == row["runs"], "Cumulative return bytes differ")
    pins = {
        "dispatch": tp,
        "bridge": ready["freeze"],
        "source": ready["source"],
        "init": ready["initialization"],
    }
    origins = {k: short(v) for k, v in pins.items()}
    for doc, file in staged:
        with zipfile.ZipFile(file) as z:
            inputs = json.loads(z.read("inputs.json"))
        with writer_gate(store):
            restore_shared(local, store, doc, inputs)
        import_zip(roots, file, doc["sha256"], doc["run"], doc["manifest_sha256"], g0, inputs)
        shortp = {"run_id": doc["run"], "manifest_sha256": doc["manifest_sha256"]}
        origins[doc["run_kind"]] = shortp
        if doc["status"] == "succeeded":
            member = "reconstruction.json" if doc["run_kind"] == "generation" else "outputs.json"
            pins[doc["run_kind"]] = bind(roots, shortp, "payload/" + member)
    cmd = [
        str(RUN_PYTHON),
        "-B",
        "{source}/scripts/t1_receive.py",
        "readback-worker",
        "--config",
        "{config}",
        "--out",
        "{out}",
        "--store",
        str(store),
    ]
    cfg = {
        "originals": origins,
        "object_id": obj,
        "logical_condition": cond,
        "job_sha256": job["job_sha256"],
        "return_sha256": row["completed_sha256"],
        "ledger_sha256": sha(ledger),
    }
    return stage(
        roots,
        name,
        "archive_verify",
        pins,
        cfg,
        cmd,
        "reconstruction-readback.json",
        14 * 1024**2,
        {"windowsledger": ledger, "summary": local / "completed.json"},
    )


def readback_worker(store, cfg, out):
    # Original() looks up exactly store / "runs" / registered_id. Return real
    # original directories so the frozen verifier can enforce its no-symlink rule.
    class RunRoots:
        def __init__(self):
            self.exact = {}
            for pin in cfg["originals"].values():
                rid = pin["run_id"]
                locations = [
                    r / "runs" / rid for r in (store, store / "history") if (r / "runs" / rid).is_dir()
                ]
                require(
                    locations and all(sha(p / "manifest.json") == pin["manifest_sha256"] for p in locations),
                    "Original readback cannot identify a unique manifest",
                )
                self.exact[rid] = locations[0].resolve()

        def __truediv__(self, key):
            return self if key == "runs" else self.exact[key]

    finish(RunRoots(), cfg, out)
    p = out / "reconstruction-readback.json"
    doc = load(p)
    require(
        doc["object_id"] == cfg["object_id"] and doc["condition"] == cfg["logical_condition"][:2],
        "Native readback object/condition mismatch",
    )
    doc.update(schema="el-t1-reconstruction-readback-1", logical_condition=cfg["logical_condition"])
    save(p, doc)


def poll(roots, reg, expected, returns):
    require(sha(reg) == expected, "Unverified batch registry")
    index = load(reg)
    require(
        index["entry"] == authority(roots)[0] and index["experiment"] == "general-t1-confirmation",
        "Wrong batch",
    )
    previous = []
    done = []
    for n, job in enumerate(index["jobs"], 1):
        path = returns / f"{reg.stem}-completed-{n:02}.json"
        if not path.is_file():
            break
        m = load(path)
        require(
            m["batch_sha256"] == expected
            and m["completed"] == n
            and m["registered_jobs"] == len(index["jobs"])
            and m["jobs"][:-1] == previous
            and m["jobs"][-1]["trial_id"] == job["trial_id"],
            "Invalid prefix ledger",
        )
        mirror = roots[0] / "transport/Returns/t1-v1" / path.name
        publish(mirror, path.read_bytes())
        done.append(readback(roots, job, m["jobs"][-1], returns, mirror))
        previous = m["jobs"]
        print(job["trial_id"], done[-1]["run_id"], flush=True)
    return done


def receive_probe(roots, obj, returns):
    s = roots[0]
    gp = selected(s, "t1-readback-" + obj + "-M2-fixed")
    summary = load(locate(roots, gp))
    require(summary["status"] == "succeeded" and summary["verified"], "Probe requires the same successful M2")
    generation = summary["origins"]["generation"]
    checkpoint = bind(roots, generation, "payload/checkpoints/step-5000.pt")
    entry, _, fr = authority(roots)
    cameras = fr["sources"]["query_cameras"]
    directory = returns / "probes" / obj
    reply = load(directory / "completed.json")
    rid = reply["run_id"]
    require(
        reply["schema"] == "el-t1-probe-return-1"
        and reply["status"] == "succeeded"
        and reply["cameras"] == cameras
        and reply["checkpoint_producer"] == generation["run_id"]
        and reply["checkpoint_sha256"] == checkpoint["sha256"]
        and reply["zip"] == rid + ".zip",
        "Probe return binding mismatch",
    )
    original = directory / reply["zip"]
    require(sha(original) == reply["zip_sha256"], "Wrong probe carrier")
    mirror = s / "transport/Returns/t1-v1/probes" / obj
    for filename in (reply["zip"], "completed.json"):
        publish(mirror / filename, (directory / filename).read_bytes())
    with zipfile.ZipFile(original) as z:
        run_input = json.loads(z.read("inputs.json"))
        cfg = json.loads(z.read("config.resolved.yaml"))
        rendered = json.loads(z.read("payload/render.json"))
        require(
            cfg["object_id"] == rendered["object_id"] == obj
            and cfg["input_roles"] == ["checkpoint", "cameras"]
            and cfg["entry"] == entry
            and cfg["pool"] == POOL
            and cfg["trial_id"] == "t1-probe-" + obj
            and cfg["recipe_sha256"] == summary["recipe_sha256"]
            and cfg["checkpoint_sha256"] == rendered["checkpoint_sha256"] == checkpoint["sha256"]
            and cfg["cameras_sha256"] == rendered["cameras_sha256"] == cameras["sha256"]
            and set(rendered["products"]) == set(NATIVE),
            "Unregistered probe payload",
        )
        supplied = {"cameras": cameras, "checkpoint": checkpoint}
        require(
            {r["role"] for r in run_input["inputs"]} == set(supplied), "Selector probe input scope changed"
        )
        for row in run_input["inputs"]:
            pin = supplied[row["role"]]
            require(
                len(row["files"]) == 1
                and row["files"][0]["sha256"] == pin["sha256"]
                and row["parent_run_id"] == pin["run_id"],
                "Wrong recorded probe input",
            )
            if row["storage"] == "shared":
                commit_shared(locate(roots, pin), s, pin["sha256"])
    if not (s / "runs" / rid).is_dir():
        import_zip(
            roots, mirror / reply["zip"], reply["zip_sha256"], rid, reply["manifest_sha256"], entry, run_input
        )
    proof = verify_bundle(s / "runs" / rid)
    require(
        proof["manifest_sha256"] == reply["manifest_sha256"]
        and proof["run"]["execution_status"] == "succeeded",
        "Original probe verification failed",
    )
    return bind(roots, {"run_id": rid}, "payload/render.json")


def decide(roots, obj, probe):
    s = roots[0]
    _, _, frozen = authority(roots)
    m2 = selected(s, "t1-readback-" + obj + "-M2-fixed")
    original = load(locate(roots, m2))
    source = original["source"]
    pins = {"predictions": probe, "cameras": frozen["sources"]["query_cameras"]}
    for n in NATIVE:
        pins["pred-" + n] = bind(roots, probe, "payload/" + n + ".npz")
    for n in S0:
        slot = "s00" if n == 24 else "s03"
        pins[slot] = bind(roots, source, "payload/sources/" + slot + ".png")
        pins["original-" + slot] = bind(
            roots, original["origins"]["render"], "payload/step-5000/" + slot + ".npz"
        )
    released = frozen["sources"]["risk_release"]
    for n in ("model", "coefficients", "preprocessing"):
        pins[n] = bind(roots, released, "payload/cpu/models/MS-L1/" + n + ".json")
    require(
        pins["model"]["sha256"] == frozen["released_models"]["MS-L1"]["model_index_sha256"]
        and pins["coefficients"]["sha256"] == frozen["released_models"]["MS-L1"]["weights_sha256"],
        "Wrong released MS-L1",
    )
    cfg = {
        "object_id": obj,
        "study_role": "test",
        "registered_test_objects": frozen["T1_objects"],
        "source_view_ids": {"s00": 24, "s03": 22},
        "checkpoint_sha256": original["checkpoints"][-1]["checkpoint_sha256"],
        "model_index_sha256": pins["model"]["sha256"],
        "selector_source_sha256": frozen["selection_code"]["scripts/b2_nbv_select.py"],
    }
    cmd = [
        "{python}",
        "-B",
        "{source}/scripts/t1_receive.py",
        "select-worker",
        "--config",
        "{config}",
        "--out",
        "{out}",
        "--protocol",
        "{input:protocol}",
        *options(pins),
    ]
    return stage(
        roots,
        "t1-decisions-" + obj,
        "model_score",
        pins,
        cfg,
        cmd,
        "choices.json",
        12 * 1024**2,
        {"protocol": PROJECT / "protocol/b2-nbv-v1.json"},
    )


def select_worker(cfg, inputs, protocol, out):
    from b2_nbv_select import run

    require(
        sha(PROJECT / "scripts/b2_nbv_select.py") == cfg["selector_source_sha256"], "Frozen selector changed"
    )
    require(
        set(inputs)
        == {
            "predictions",
            "cameras",
            "model",
            "coefficients",
            "preprocessing",
            "s00",
            "s03",
            "original-s00",
            "original-s03",
            *["pred-" + n for n in NATIVE],
        },
        "Disallowed selector role",
    )
    for k, v in inputs.items():
        require(sha(v) == cfg["sources"][k]["sha256"], "Changed selector role " + k)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        p = Path(tmp)
        for folder in ("probe", "selector", "model"):
            (p / folder).mkdir()
        prod = load(inputs["predictions"])
        require(set(prod["products"]) == set(NATIVE), "Blocked selector input set")
        shutil.copyfile(inputs["predictions"], p / "probe/render.json")
        for n in NATIVE:
            require(prod["products"][n]["path"] == n + ".npz", "Bad native member")
            shutil.copyfile(inputs["pred-" + n], p / "probe" / (n + ".npz"))
        origin = {
            "object_id": cfg["object_id"],
            "source_view_ids": cfg["source_view_ids"],
            "checkpoint_sha256": cfg["checkpoint_sha256"],
            "files": {},
        }
        for name, key, rel in (
            ("s00", "s00", "s00.png"),
            ("s03", "s03", "s03.png"),
            ("r00", "original-s00", "r00.npz"),
            ("r03", "original-s03", "r03.npz"),
        ):
            shutil.copyfile(inputs[key], p / "selector" / rel)
            origin["files"][name] = {
                "path": rel,
                "sha256": sha(inputs[key]),
                "bytes": inputs[key].stat().st_size,
            }
        save(p / "origin.json", origin)
        for n in ("model", "coefficients", "preprocessing"):
            shutil.copyfile(inputs[n], p / "model" / (n + ".json"))
        run(
            cfg,
            p / "origin.json",
            p / "probe/render.json",
            inputs["cameras"],
            p / "model",
            load(protocol),
            out,
        )
    save(
        out / "selector-input-audit.json",
        {
            "run_id": os.environ["EL_RUN_ID"],
            "object_id": cfg["object_id"],
            "permitted_photo_ids": S0,
            "candidate_RGB_reads": 0,
            "evaluation_poses": 0,
            "evaluation_reference_reads": 0,
            "native_source_renders": ["s00", "s03"],
            "permitted_predicted_views": NATIVE,
            "input_hashes": cfg["sources"],
            "M2_native_render": "Frozen endpoint renderer emitted independent evaluation predictions with native self renders in its restricted stage before selection; only the two named source self renders entered this process.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["poll", "decide", "readback-worker", "select-worker"])
    p.add_argument("--store", type=Path, default=DEFAULT)
    for k in ("registry", "returns", "config", "protocol", "out"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--sha256")
    p.add_argument("--object")
    p.add_argument("--input", action="append", nargs=2, default=[])
    a = p.parse_args()
    roots = a.store.resolve(), a.store.resolve() / "history"
    if a.action == "poll":
        poll(roots, a.registry, a.sha256, a.returns)
    if a.action == "readback-worker":
        readback_worker(roots[0], load(a.config), a.out)
    if a.action == "select-worker":
        select_worker(load(a.config), {k: Path(v) for k, v in a.input}, a.protocol, a.out)
    if a.action == "decide":
        print(decide(roots, a.object, receive_probe(roots, a.object, a.returns)))
