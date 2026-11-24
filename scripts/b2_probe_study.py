"""Freeze, dispatch, receive and score the two source-only NBV probe experiments."""

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from b2_records import bind, selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require, verify_bundle
from t0_records import resolve
from t0_worker_run import extract

PROJECT = Path(__file__).resolve().parents[1]
OBJECTS = ("set1", "set6")
SCIENTIFIC = [
    "scripts/b2_probe_geometry.py",
    "scripts/b2_probe_worker.py",
    "scripts/b2_nbv_select.py",
    "src/recovery/scene.py",
    "src/recovery/engine.py",
    "src/evidence_lens/multisource.py",
    "src/evidence_lens/multisource_model.py",
    "src/evidence_lens/features.py",
]
RELEASE = "20261123T193505072501Z-6046afc1fce4480b"


def worker(args):
    cfg = load(args.config)
    refs = cfg["sources"]
    if args.kind == "freeze":
        for n, value in cfg["scientific_files"].items():
            require(sha(PROJECT / n) == value, "scientific selection code changed")
        doc = load(args.protocol)
        save(
            args.out / "freeze.json",
            {
                "schema": "el-b2-selection-freeze-1",
                "run_id": os.environ["EL_RUN_ID"],
                "sources": refs,
                "protocol": doc,
                "protocol_sha256": sha(args.protocol),
                "scientific_files": cfg["scientific_files"],
                "model_index_sha256": cfg["model_index_sha256"],
            },
        )
        return
    if args.kind == "pack":
        lock = load(args.freeze)
        for n, value in lock["scientific_files"].items():
            require(
                sha(PROJECT / n) == value,
                "selection/render code differs from the pre-dispatch lock",
            )
        with zipfile.ZipFile(
            args.out / "probe-jobs.zip", "x", zipfile.ZIP_DEFLATED, compresslevel=4
        ) as z:
            for base in ("src/evidence_lens", "src/recovery"):
                for p in sorted((PROJECT / base).rglob("*.py")):
                    z.write(p, p.relative_to(PROJECT).as_posix())
            z.write(
                PROJECT / "scripts/b2_probe_worker.py", "scripts/b2_probe_worker.py"
            )
            for obj, ck, ori in zip(
                OBJECTS, [args.ck1, args.ck6], [args.origin1, args.origin6], strict=True
            ):
                origin = load(ori)
                r = origin["files"]["checkpoint"]
                require(
                    sha(ck) == r["sha256"] == origin["checkpoint_sha256"]
                    and origin["object_id"] == obj,
                    "mismatched M2 checkpoint",
                )
                spec = {
                    "experiment_id": "batch-two-dtu-v1",
                    "trial_id": "b2-probe-" + obj,
                    "object_id": obj,
                    "recording": {"expected_output_bytes": 38000000},
                    "pool": lock["protocol"]["candidate_ids"],
                    "checkpoint_sha256": sha(ck),
                    "checkpoint_producer": r["source"]["run_id"],
                    "cameras_sha256": sha(args.cameras),
                    "recipe_sha256": origin["files"]["recipe"]["sha256"],
                    "selection_freeze": refs["freeze"],
                    "parents": {
                        "cameras": refs["cameras"],
                        "checkpoint": refs["ck" + obj[3:]],
                    },
                    "input_names": {
                        "checkpoint": "checkpoint.pt",
                        "cameras": "cameras.json",
                    },
                }
                z.writestr(
                    obj + "/config.json",
                    json.dumps(spec, indent=2, sort_keys=True) + "\n",
                )
                z.write(ck, obj + "/checkpoint.pt")
                z.write(args.cameras, obj + "/cameras.json")
        save(
            args.out / "ticket.json",
            {
                "schema": "el-b2-probe-ticket-1",
                "run_id": os.environ["EL_RUN_ID"],
                "file": "probe-jobs.zip",
                "sha256": sha(args.out / "probe-jobs.zip"),
                "bytes": (args.out / "probe-jobs.zip").stat().st_size,
                "sources": refs,
            },
        )
        return
    require(args.kind == "receive", "incorrect controller action")
    rec = load(args.reply)
    ticket = load(args.ticket)
    obj = cfg["object_id"]
    require(
        rec["schema"] == "el-b2-probe-return-1"
        and rec["status"] == "succeeded"
        and sha(args.carrier) == rec["zip_sha256"]
        and sha(args.dispatch) == ticket["sha256"],
        "incorrect GPU carrier",
    )
    with zipfile.ZipFile(args.dispatch) as package:
        issued = package.read(obj + "/config.json")
        expected = json.loads(issued)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        from evidence_lens._capacity import FALLBACK
        from evidence_lens.recorder import resolve_config

        issued_file = Path(tmp) / "issued.json"
        issued_file.write_bytes(issued)
        root = Path(tmp) / "original"
        root.mkdir()
        extract(args.carrier, root)
        v = verify_bundle(root)
        require(
            v["manifest_sha256"] == rec["manifest_sha256"]
            and v["run"]["run_id"] == rec["run_id"]
            and v["run"]["execution_status"] == "succeeded"
            and v["run"]["run_kind"] == "render"
            and load(root / "config.resolved.yaml") == resolve_config(issued_file, None, FALLBACK)
            and sha(issued_file) == rec["config_sha256"],
            "GPU job identity mismatch",
        )
        prod = load(root / "payload/render.json")
        require(
            prod["cameras_sha256"] == expected["cameras_sha256"]
            and prod["object_id"] == obj
            and prod["checkpoint_sha256"] == expected["checkpoint_sha256"]
            and prod["run_id"] == rec["run_id"],
            "incorrect predicted views",
        )
        for name, x in prod["products"].items():
            require(
                x["path"] == name + ".npz"
                and name
                in {
                    "s00",
                    "s03",
                    *[f"c{i:03}" for i in expected["pool"]],
                    *[f"q{i:02}" for i in range(4)],
                }
                and sha(root / "payload" / x["path"]) == x["sha256"],
                "unregistered view",
            )
            shutil.copyfile(root / "payload" / x["path"], args.out / x["path"])
        save(
            args.out / "render.json",
            {**prod, "gpu_parent": rec, "mac_readback_run": os.environ["EL_RUN_ID"]},
        )


def run(a):
    store, history = a.store.resolve(), a.history.resolve()
    roots = (store, history)
    spec = PROJECT / "protocol/b2-nbv-v1.json"
    rig = selected(store, "b2-dtu-cohort")
    origin = {o: selected(store, "b2-nbv-original-" + o) for o in OBJECTS}
    org = lambda obj, n: bind(store, origin[obj]["run_id"], "payload/selector/" + n)
    if a.action == "freeze":
        model = bind(history, RELEASE, "payload/cpu/models/MS-L1/model.json")
        return stage(
            store,
            history,
            "nbv-method",
            "derivation",
            {"model": model, "cohort": rig},
            {
                "scientific_files": {n: sha(PROJECT / n) for n in SCIENTIFIC},
                "model_index_sha256": model["sha256"],
            },
            [
                "{python}",
                "{source}/scripts/b2_probe_study.py",
                "worker",
                "freeze",
                "--config",
                "{config}",
                "--protocol",
                "{input:protocol}",
                "--out",
                "{out}",
            ],
            "payload/freeze.json",
            40000,
            {"protocol": spec},
        )
    fr = selected(store, "nbv-method")
    lock = load(resolve(roots, fr))
    require(
        lock["protocol_sha256"] == sha(spec),
        "selection rules differ from pre-dispatch freeze",
    )
    if a.action == "geometry":
        allowed = bind(store, rig["run_id"], "payload/candidate-cameras.json")
        return stage(
            store,
            history,
            "nbv-query-geometry",
            "derivation",
            {
                "freeze": fr,
                "rig": allowed,
                "source": org("set1", "source_cameras.json"),
            },
            {"role": "only ten permitted poses"},
            [
                "{python}",
                "{source}/scripts/b2_probe_geometry.py",
                "--rig",
                "{input:rig}",
                "--source",
                "{input:source}",
                "--protocol",
                "{input:protocol}",
                "--out",
                "{out}",
            ],
            "payload/cameras.json",
            45000,
            {"protocol": spec},
        )
    geom = selected(store, "nbv-query-geometry")
    if a.action == "pack":
        pins = {
            "freeze": fr,
            "cameras": geom,
            "origin1": origin["set1"],
            "origin6": origin["set6"],
            **{"ck" + o[3:]: org(o, "checkpoint.pt") for o in OBJECTS},
        }
        cmd = [
            "{python}",
            "{source}/scripts/b2_probe_study.py",
            "worker",
            "pack",
            "--config",
            "{config}",
            "--out",
            "{out}",
        ]
        for n in pins:
            cmd += ["--" + n, "{input:" + n + "}"]
        return stage(
            store,
            history,
            "nbv-probe-ticket",
            "derivation",
            pins,
            {
                "role": "two CUDA visibility/virtual query renders without candidate photographs"
            },
            cmd,
            "payload/ticket.json",
            10000000,
        )
    ticket = selected(store, "nbv-probe-ticket")
    obj = a.object
    if a.action == "receive":
        directory = a.returns.resolve() / obj
        meta = load(directory / "completed.json")
        require(meta["zip"] == meta["run_id"] + ".zip", "unbound package name")
        pins = {
            "ticket": ticket,
            "dispatch": bind(store, ticket["run_id"], "payload/probe-jobs.zip"),
            "origin": origin[obj],
        }
        cmd = [
            "{python}",
            "{source}/scripts/b2_probe_study.py",
            "worker",
            "receive",
            "--config",
            "{config}",
            "--out",
            "{out}",
        ]
        for n in ("ticket", "dispatch", "reply", "carrier"):
            cmd += ["--" + n, "{input:" + n + "}"]
        return stage(
            store,
            history,
            "nbv-probe-receive-" + obj,
            "archive_verify",
            pins,
            {"object_id": obj},
            cmd,
            "payload/render.json",
            70000000,
            {"reply": directory / "completed.json", "carrier": directory / meta["zip"]},
            a.retry_of,
        )
    predictions = selected(store, "nbv-probe-receive-" + obj)
    model = bind(history, RELEASE, "payload/cpu/models/MS-L1/model.json")
    require(
        model["sha256"] == lock["model_index_sha256"]
        and all(sha(PROJECT / n) == v for n, v in lock["scientific_files"].items()),
        "registered selection code or model changed",
    )
    p = resolve(roots, predictions)
    orient = resolve(roots, origin[obj])
    return stage(
        store,
        history,
        "nbv-choices-" + obj,
        "model_score",
        {
            "origin": origin[obj],
            "predictions": predictions,
            "freeze": fr,
            "cameras": geom,
            "model": model,
        },
        {
            "study_role": "development",
            "object_id": obj,
            "model_index_sha256": model["sha256"],
        },
        [
            "{python}",
            "{source}/scripts/b2_nbv_select.py",
            "--config",
            "{config}",
            "--origin",
            str(orient),
            "--predictions",
            str(p),
            "--cameras",
            "{input:cameras}",
            "--model",
            str(resolve(roots, model).parent),
            "--protocol",
            "{input:protocol}",
            "--out",
            "{out}",
        ],
        "payload/choices.json",
        7000000,
        {"protocol": spec},
        a.retry_of,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action", choices=("freeze", "geometry", "pack", "receive", "select", "worker")
    )
    p.add_argument("kind", nargs="?", choices=("freeze", "pack", "receive"))
    for name in (
        "store",
        "history",
        "config",
        "out",
        "protocol",
        "freeze",
        "origin1",
        "origin6",
        "ck1",
        "ck6",
        "cameras",
        "reply",
        "carrier",
        "ticket",
        "dispatch",
        "returns",
    ):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--object", choices=OBJECTS)
    p.add_argument("--retry-of")
    a = p.parse_args()
    if a.action == "worker":
        worker(a)
    else:
        print(json.dumps(run(a), sort_keys=True))
