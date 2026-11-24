"""Start native B data stages with independent authorization and positive per-member roles."""

import argparse
import json
import os
from pathlib import Path

from dual_records import resolve, selected, stage
from evidence_lens._bundle import load, sha
from evidence_lens._verify import require

ROOT = Path(__file__).resolve().parents[1]


def key(scene):
    return scene.replace("_", "-")


def arch(store, scene):
    record = selected(store, "b-eth-download-" + key(scene))
    data = load(resolve((store, store / "history"), record))
    path = store / "runs" / record["run_id"] / "payload" / data["file"]
    return record, path


def control(a):
    store = a.store.resolve()
    if a.action == "register":
        pins = {"baseline": selected(store, "a-baseline"), "public": selected(store, "a-anonymous-readback")}
        return stage(
            store,
            "b-data-registration",
            "archive_verify",
            pins,
            {
                "phase": "before first source or target decode",
                "scope": "six original family rules/seeds/cameras/roles and ETH3D three fixed roles; no pixel reads",
            },
            [
                "{python}",
                "-I",
                "-c",
                "import shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2])",
                "{input:protocol}",
                "{out}/registration.json",
            ],
            "payload/registration.json",
            16000,
            {"protocol": ROOT / "protocol/many-source-data-v1.json"},
        )
    pins = {"roles": selected(store, "b-data-registration"), "study": selected(store, "b-study-registration")}
    protocol = load(resolve((store, store / "history"), pins["roles"]))
    originals = {r["id"]: r for r in protocol["originals"]}
    externals = {r["id"]: r for r in protocol["external"]["roster"]}
    if a.scene not in originals.keys() | externals.keys():
        raise ValueError("unregistered scene")
    if a.action == "download":
        require(a.scene in externals, "registered external scene required")
        sizes = {"delivery_area": 600000000, "living_room": 850000000, "bridge": 1450000000}
        return stage(
            store,
            "b-eth-download-" + key(a.scene),
            "archive_verify",
            pins,
            {
                "body_cap_frozen": protocol["external"]["download_cap_bytes"],
                "opaque_collection": True,
                "phase": "first official acquisition",
                "scene_id": a.scene,
            },
            [
                "{python}",
                "-B",
                "{source}/scripts/many_data_download.py",
                "--scene",
                a.scene,
                "--registration",
                "{input:roles}",
                "--ledger",
                str(store / "work/eth-download-ledger.jsonl"),
                "--out",
                "{out}",
            ],
            "payload/download.json",
            sizes[a.scene],
        )
    cmd = [
        "{python}",
        "-B",
        "{source}/scripts/many_data_control.py",
        a.action + "-worker",
        "--scene",
        a.scene,
        "--registration",
        "{input:roles}",
        "--config",
        "{config}",
        "--out",
        "{out}",
    ]
    extra = {}
    if a.scene in externals:
        d, archive = arch(store, a.scene)
        pins["download"] = d
        cmd += ["--download", "{input:download}", "--archive", str(archive)]
        if a.action != "roster":
            p = selected(store, "b-roles-" + key(a.scene))
            pins["roster"] = p
            cmd += ["--roster", "{input:roster}"]
    else:
        extra["renderer"] = ROOT / "scripts/many_data_raster.cpp"
        cmd += ["--renderer", "{input:renderer}"]
    if a.action == "roster":
        require(a.scene in externals, "only ETH3D needs original camera roster extraction")
        return stage(
            store,
            "b-roles-" + key(a.scene),
            "derivation",
            pins,
            {"scene_id": a.scene, "role": "camera records only"},
            cmd,
            "payload/roles.json",
            600000,
            extra,
        )
    if a.action == "source":
        estimate = archive.stat().st_size + 60000000 if a.scene in externals else 200000000
        return stage(
            store,
            "b-source-" + key(a.scene),
            "derivation",
            pins,
            {"scene_id": a.scene, "role": "R-only preparation; H camera entries only"},
            cmd,
            "payload/source.json",
            estimate,
            extra,
        )
    p = selected(store, "b-source-" + key(a.scene))
    pins["source"] = p
    cmd += ["--source-index", "{input:source}"]
    split = (originals | externals)[a.scene]["split"]
    if split == "confirmation":
        require(a.opening_label == "b-confirmation-h-opening", "one common confirmation opening is required")
    else:
        require(
            a.opening_label == "b-train-dev-h-opening",
            "training and development openings are separate from confirmation",
        )
    pins["opening"] = selected(store, a.opening_label)
    if split == "confirmation":
        from many_data_prepare import confirmation_terminals

        gatepath = resolve((store, store / "history"), pins["opening"])
        native = load(gatepath.parents[1] / "manifest.json")
        parents = {(p["run_id"], p["manifest_sha256"]) for p in native["parents"]}
        for p in confirmation_terminals(load(gatepath), protocol):
            require(
                (p["run_id"], p["manifest_sha256"]) in parents,
                "opening references a receipt outside its sealed recorder parents",
            )
            resolve((store, store / "history"), p)
    cmd += ["--opening", "{input:opening}"]
    return stage(
        store,
        "b-truth-" + key(a.scene),
        "evaluation",
        pins,
        {"scene_id": a.scene, "split": split, "authorization": a.opening_label},
        cmd,
        "payload/truth.json",
        120000000,
        extra,
    )


def worker(a):
    from many_data_eth import roster

    require(os.environ.get("EL_RUN_ID"), "run data processing under the recorder")
    protocol = load(a.registration)
    if a.archive:
        download = load(a.download)
        require(
            download["scene_id"] == a.scene
            and download["bytes"] == a.archive.stat().st_size
            and sha(a.archive) == download["sha256"],
            "opaque archive differs from sealed network intake",
        )
    if a.action == "roster-worker":
        roster(a.archive, protocol, a.scene, a.out / "roles.json")
        return
    from many_data_prepare import prepare

    prepare(a, protocol)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=(
            "register",
            "download",
            "roster",
            "roster-worker",
            "source",
            "source-worker",
            "truth",
            "truth-worker",
        ),
    )
    p.add_argument("--scene")
    for n in (
        "store",
        "registration",
        "out",
        "config",
        "roster",
        "renderer",
        "download",
        "archive",
        "source-index",
        "opening",
    ):
        p.add_argument("--" + n, type=Path)
    p.add_argument("--opening-label")
    a = p.parse_args()
    if a.action.endswith("-worker"):
        worker(a)
    else:
        print(json.dumps(control(a)), flush=True)
