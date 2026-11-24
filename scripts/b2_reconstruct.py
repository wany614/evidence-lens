"""Register and prepare source-only batch-two reconstructions for the original Windows gsplat worker."""

import argparse
import json
from pathlib import Path

from evidence_lens._bundle import encoded, load, sha
from evidence_lens._verify import require
from make_handoff import publish
from t0_records import bind, record, resolve

PROJECT = Path(__file__).resolve().parents[1]
GEOMETRY = (
    Path.home() / "Library/Application Support/EvidenceLens/recovery-runtime/triangulation-v1/bin/python"
)
STUDY = PROJECT / "protocol/b2-reconstruction-v1.json"
IDS = {"M2": ("s00", "s03"), "M4": ("s00", "s01", "s02", "s03")}


def input_args(roles):
    return [item for role in roles for item in ("--input", role, "{input:" + role + "}")]


def register(store, old, seed):
    protocol = load(STUDY)
    require(seed in (43171, 43172, 43173), "Unregistered seed")
    original = load(old / "work/r-control/r2-scale-decay-02-set1-M2-normalize.json")
    pins = {
        "recipe": original["recipe"],
        "worker": original["worker"],
        "depth": original["depth_probe"],
        "resource": original["resource_probe"],
        "historical": bind(old, original["generation"]["run_id"], "payload/reconstruction.json"),
    }
    raw = {"study": STUDY, "initrules": PROJECT / "protocol/recovery-initialization-v1.json"}
    require(
        sha(PROJECT / "scripts/recovery_triangulate.py") == protocol["geometry_code_sha256"]
        and sha(raw["initrules"]) == protocol["source_initialization_sha256"],
        "Numerical source changed",
    )
    cmd = [
        "{python}",
        "{source}/scripts/b2_worker_register.py",
        "--config",
        "{config}",
        "--out",
        "{out}",
        *input_args([*pins, *raw]),
    ]
    return record(
        store,
        old,
        f"b2-reconstruction-seed-{seed}",
        "archive_verify",
        pins,
        {
            "experiment_id": "b2-reconstruction-registration",
            "trial_id": f"b2-reconstruction-seed-{seed}",
            "optimizer_seed": seed,
        },
        cmd,
        "payload/bridge.json",
        2 * 1024**2,
        raw,
    )


def prepare(store, old, authority, source, obj, cond, layout, geometry=GEOMETRY, cohort=None, selection=None):
    roots = (store, old)
    bridge = load(resolve(roots, authority))
    seed = bridge["optimizer_seed"]
    study = load(STUDY)
    ids = IDS[cond]
    require(
        bridge["schema"] == "el-b2-reconstruction-origin-1"
        and bridge["study_sha256"] == sha(STUDY)
        and sha(PROJECT / "scripts/recovery_triangulate.py") == study["geometry_code_sha256"],
        "Unregistered authority",
    )
    if cohort:
        fr = load(resolve(roots, cohort))
        require(
            obj in fr["splits"]["C6"]
            and obj not in fr["splits"]["T1"]
            and layout == cond + "-fixed"
            and seed == 43171
            and selection is None,
            "Only authorized fixed C6 sources",
        )
        split = "calibration"
    else:
        require(
            obj in study["development_objects"] and cond == "M4" and layout in ("M4-fixed", "M4-selected"),
            "Unregistered development cell",
        )
        require(
            (layout == "M4-fixed" and seed in (43172, 43173) and selection is None)
            or (layout == "M4-selected" and seed == 43171 and selection is not None),
            "Seed/branch budget",
        )
        split = "development"
    selected = None
    if selection:
        decision = load(resolve(roots, selection))
        selected = decision["selected_official_ids"]
        require(
            decision["schema"] == "el-b2-nbv-selection-1"
            and decision["object_id"] == obj
            and decision["strategy_id"] in study["development_selection"]["strategies"]
            and decision["pool"] == study["development_selection"]["candidate_ids"]
            and len(set(selected)) == 2
            and set(selected) <= set(decision["pool"]),
            "Wrong source selection",
        )
    src = load(resolve(roots, source))
    expected_schema = "el-b2-dtu-source-1" if cohort else "el-b2-development-source-1"
    require(
        src["schema"] == expected_schema
        and src["object_id"] == obj
        and src["condition"] == cond
        and src.get("reference_photo_reads", 0) == 0
        and (not cohort or src.get("cohort") == cohort)
        and (not selection or src.get("selection") == selection),
        "Wrong source authority",
    )
    official = {k: src["sources"][k]["official_view_id"] for k in ids}
    expected = (
        study["calibration"].get("fixed_" + cond)
        if not selected
        else dict(zip(ids, [24, *sorted(selected), 22], strict=True))
    )
    require(official == expected and set(src["sources"]) == set(ids), "Wrong slot order or source count")
    suffix = ("sel-" + str(official["s01"]) + "-" + str(official["s02"])) if selected else "fixed"
    trial = f"b2-{obj}-{cond.lower()}-{suffix}-{seed}"
    source_id = source["run_id"]
    auth_id = authority["run_id"]
    sp = {k: bind(store, source_id, "payload/sources/" + k + ".png") for k in ids}
    cam = bind(store, source_id, "payload/source_cameras.json")
    rc = bind(store, source_id, "payload/render_cameras.json")
    require(
        cam["sha256"] == src["source_camera_sha256"]
        and rc["sha256"] == src["render_camera_sha256"]
        and all(sp[k]["sha256"] == src["sources"][k]["rgb"]["sha256"] for k in ids),
        "Sealed source hashes differ",
    )
    cfg = {
        "scene_family_id": obj,
        "scene_id": obj,
        "condition_id": cond,
        "split": split,
        "layout": layout,
        "scientific_trial_id": trial,
        "source_parent": source,
        "registration": authority,
        "authority": authority,
        "source": source,
        "cohort": cohort,
        "selection": selection,
        "selected_official_ids": selected,
        "official_by_slot": official,
        "generator_source_ids": list(ids),
        "optimizer_seed": seed,
        "initialization_protocol_sha256": study["source_initialization_sha256"],
        "recipe_sha256": bridge["recipe_sha256"],
        "source_camera_sha256": cam["sha256"],
        "source_rgb_sha256": {k: sp[k]["sha256"] for k in ids},
    }
    pins = {"cameras": cam, "rules": bind(store, auth_id, "payload/init-protocol.json"), **sp}
    cmd = [
        str(geometry.absolute()),
        "-B",
        "{source}/scripts/recovery_triangulate.py",
        "--config",
        "{config}",
        "--protocol",
        "{input:rules}",
        "--cameras",
        "{input:cameras}",
        "--out",
        "{out}",
    ]
    for k in ids:
        cmd += ["--source", k, "{input:" + k + "}"]
    init = record(
        store,
        old,
        trial + "-init",
        "derivation",
        pins,
        {**cfg, "experiment_id": "b2-source-initialization", "trial_id": trial + "-init"},
        cmd,
        "payload/init.json",
        2 * 1024**2,
    )
    pins = {
        "authority": authority,
        "study": bind(store, auth_id, "payload/protocol.json"),
        "provenance": source,
        "cameras": cam,
        "rendercameras": rc,
        "recipe": bind(store, auth_id, "payload/recipe.json"),
        "worker": bind(store, auth_id, "payload/worker.zip"),
        "init": bind(store, init["run_id"], "payload/init.npz"),
        "initmeta": init,
        **sp,
    }
    if cohort:
        pins["cohort"] = cohort
    if selection:
        pins["selection"] = selection
    cmd = [
        "{python}",
        "{source}/scripts/b2_worker_pack.py",
        "--config",
        "{config}",
        "--out",
        "{out}",
        *input_args(pins),
    ]
    ticket = record(
        store,
        old,
        trial + "-dispatch",
        "dataset_build",
        pins,
        {
            **cfg,
            "experiment_id": "b2-reconstruction-dispatch",
            "trial_id": trial + "-dispatch",
            "initialization": init,
        },
        cmd,
        "payload/ticket.json",
        2 * 1024**2,
    )
    dest = store / "transport/Handoffs/b2-v1" / trial
    for name in ("ticket.json", "dispatch.zip", "READY.json", "t0_worker_run.py"):
        publish(dest / name, resolve(roots, bind(store, ticket["run_id"], "payload/" + name)).read_bytes())
    publish(dest / "dispatch-pin.json", encoded(ticket))
    return {"trial_id": trial, "ticket": ticket, "path": str(dest), "job": load(dest / "ticket.json")}


def development_source(store, old, authority, obj, selection=None):
    require(obj in ("set1", "set6"), "D sources are restricted to the exposed subjects")
    study = load(STUDY)
    condition = "M2" if selection else "M4"
    names = IDS[condition]
    h = "r2-scale-decay-02-" + obj + "-" + condition
    original = load(old / "work/r-control" / (h + "-normalize.json"))
    ticket = load(old / "work/r-control" / (h + "-ticket.json"))
    src = ticket["handoff"]["run_id"]
    pins = {
        "authority": authority,
        "study": bind(store, authority["run_id"], "payload/protocol.json"),
        "prior": original["original_provenance"],
        "cameras": bind(old, src, "payload/handoff/source_cameras.json"),
        "rendercameras": bind(old, src, "payload/handoff/render_cameras.json"),
        **{k: bind(old, src, "payload/handoff/sources/" + k + ".png") for k in names},
    }
    layout = "M4-fixed"
    strategy = None
    if selection:
        decision = load(resolve((store, old), selection))
        start = load(store / "work/controller" / ("b2-nbv-original-" + obj) / "selected.json")
        ids = decision["selected_official_ids"]
        strategy = decision["strategy_id"]
        layout = "M4-selected"
        require(
            decision["schema"] == "el-b2-nbv-selection-1"
            and decision["object_id"] == obj
            and strategy in study["development_selection"]["strategies"]
            and decision["pool"] == study["development_selection"]["candidate_ids"]
            and len(ids) == 2
            and all(type(n) is int for n in ids)
            and len(set(ids)) == 2
            and set(ids) <= set(decision["pool"])
            and decision["m2_checkpoint_sha256"] == load(resolve((store, old), start))["checkpoint_sha256"],
            "Unsealed or unauthorized candidate",
        )
        rig = "20261123T193804778622Z-622e5e9ca6ef434b"
        archive = "20261123T193512888729Z-8f881159624847fc"
        pins.update(
            decision=selection,
            start=start,
            directory=bind(old, archive, "payload/directory.json"),
            archive=bind(old, archive, "payload/archive.json"),
            crop=bind(old, rig, "payload/crop.json"),
            metric=bind(old, rig, "payload/metric-frame.json"),
        )
        for sid, n in zip(("s01", "s02"), sorted(ids), strict=True):
            pins["projection" + sid] = bind(old, rig, f"inputs/matrices/cal18/pos_{n:03d}.txt")
    name = "b2-D-source-" + obj + "-" + (strategy or "fixed")
    cmd = [
        "{python}",
        "{source}/scripts/b2_worker_development.py",
        "--config",
        "{config}",
        "--out",
        "{out}",
        *input_args(pins),
    ]
    return record(
        store,
        old,
        name,
        "dataset_build",
        pins,
        {
            "experiment_id": "b2-development-source-access",
            "trial_id": name,
            "layout": layout,
            "object_id": obj,
            "strategy_id": strategy,
        },
        cmd,
        "payload/source-provenance.json",
        24 * 1024**2,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--historical-store", type=Path, default=PROJECT / ".research")
    sub = p.add_subparsers(dest="op", required=True)
    reg = sub.add_parser("register")
    reg.add_argument("--seed", type=int, choices=(43171, 43172, 43173), required=True)
    src = sub.add_parser("development-source")
    src.add_argument("--authority", type=Path, required=True)
    src.add_argument("--object", required=True)
    src.add_argument("--selection", type=Path)
    prep = sub.add_parser("prepare")
    for key in ("authority", "source", "cohort", "selection"):
        prep.add_argument("--" + key, type=Path, required=key in ("source", "authority"))
    prep.add_argument("--object", required=True)
    prep.add_argument("--condition", choices=IDS, required=True)
    prep.add_argument("--layout", choices=("M2-fixed", "M4-fixed", "M4-selected"), required=True)
    prep.add_argument("--geometry-python", type=Path, default=GEOMETRY)
    a = p.parse_args()
    s = a.store.resolve()
    h = a.historical_store.resolve()
    if a.op == "register":
        out = register(s, h, a.seed)
    elif a.op == "development-source":
        out = development_source(
            s, h, load(a.authority), a.object, load(a.selection) if a.selection else None
        )
    else:
        out = prepare(
            s,
            h,
            load(a.authority),
            load(a.source),
            a.object,
            a.condition,
            a.layout,
            a.geometry_python,
            load(a.cohort) if a.cohort else None,
            load(a.selection) if a.selection else None,
        )
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
