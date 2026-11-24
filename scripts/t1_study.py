"""Run the registered T1 source, initialization and immutable Windows dispatch stages."""

import argparse
import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

from b2_intake_record import locate, sealed_member
from make_handoff import publish
from t0_records import record
from t1_access import CONDITIONS, FREEZE_SHA, OBJECTS, SLOTS, STRATEGIES, registered

from evidence_lens._bundle import encoded, load, sha
from evidence_lens._verify import require

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT = Path(os.environ.get("EVIDENCE_LENS_STUDY_ROOT", ".research/general-v1"))
GEOMETRY = Path(
    os.environ.get(
        "EVIDENCE_LENS_GEOMETRY_PYTHON",
        Path.home() / "Library/Application Support/EvidenceLens/recovery-runtime/triangulation-v1/bin/python",
    )
)
RUN_PYTHON = Path(os.environ.get("EVIDENCE_LENS_READBACK_PYTHON", sys.executable))


def selected(store, name):
    return load(store / "work/controller" / name / "selected.json")


def named(store, name):
    return (store / "work/controller" / name / "selected.json").is_file()


def bind(roots, pin, member):
    return sealed_member(roots, pin["run_id"], member)


def options(roles):
    return [s for key in roles for s in ("--input", key, "{input:" + key + "}")]


def stage(roots, name, kind, pins, details, cmd, filename, size, extra=None, retry=None):
    preopening = (
        name.startswith(("t1-source-", "t1-decisions-", "t1-probe-terminal-"))
        or name.endswith(("-init", "-dispatch"))
        or name in ("t1-terminals", "t1-probe-kit")
    )
    if preopening and named(roots[0], "t1-opening"):
        require(
            named(roots[0], name),
            "The joint opening prevents new scientific predictions, selections, or source fits",
        )
    return record(
        *roots,
        name,
        kind,
        pins,
        dict(experiment_id="general-t1-confirmation", trial_id=name, **details),
        cmd,
        "payload/" + filename,
        size,
        extra,
        retry,
    )


@lru_cache(maxsize=1)
def authority(roots):
    entry = selected(roots[0], "g0-entry")
    start = load(locate(roots, entry))
    fp = start["freeze"]
    frozen = load(locate(roots, fp))
    registered(start, frozen, fp)
    require(
        sha(PROJECT / "docs/results/batch-two/holdout/t1-freeze.json") == FREEZE_SHA
        and all(
            sha(PROJECT / name) == digest
            for name, digest in {**frozen["scientific_code_sha256"], **frozen["selection_code"]}.items()
        ),
        "A frozen scientific source identity changed",
    )
    return entry, fp, frozen


def common(roots):
    entry, frozen_pin, frozen = authority(roots)
    cohort = frozen["cohort"]
    roster = load(locate(roots, cohort))["upstream_roster"]
    return {
        "entry": entry,
        "freeze": frozen_pin,
        "cohort": cohort,
        "crop": bind(roots, roster, "payload/crop.json"),
    }, roster


def acquire(roots, obj, condition, opening=None, retry=None):
    s = roots[0]
    pins, roster = common(roots)
    require(obj in OBJECTS, "Unknown object")
    if opening:
        require(
            condition == "references" and opening == selected(s, "t1-opening"),
            "Only the selected collective opening may authorize target access",
        )
        pins["opening"] = opening
        pins["terminals"] = selected(s, "t1-terminals")
    else:
        require(condition in CONDITIONS, "Unknown condition")
        cond = condition[:2]
        pins.update(
            candidateRig=bind(roots, pins["cohort"], "payload/candidate-cameras.json"),
            sourceRig=bind(roots, roster, "payload/condition/" + cond + "/source_cameras.json"),
            renderRig=bind(roots, roster, "payload/condition/" + cond + "/render_cameras.json"),
        )
        if cond == "M4":
            choices = selected(s, "t1-decisions-" + obj)
            pins["choices"] = choices
            for strategy in STRATEGIES:
                pins["choice-" + strategy] = bind(roots, choices, "payload/" + strategy + ".json")
            numbers = (
                [24, 23, 17, 22]
                if condition == "M4-fixed"
                else [24, *load(locate(roots, choices))["results"][condition[3:]], 22]
            )
            # Read only prior access receipts to locate physically cached bytes separately
            # authorized on this trajectory. Never pass another trajectory's unselected file.
            cache = {}
            for prior in CONDITIONS[: CONDITIONS.index(condition)]:
                pname = "t1-source-" + obj + "-" + prior
                if named(s, pname):
                    orig = selected(s, pname)
                    for row in load(locate(roots, orig))["sources"].values():
                        cache.setdefault(row["official_view_id"], (orig, "payload/" + row["original_path"]))
            for slot, cam in zip(SLOTS[cond], numbers, strict=True):
                if cam in cache:
                    orig, member = cache[cam]
                    pins["cached-" + slot] = bind(roots, orig, member)
    kind = "t1-reference-" + obj if opening else "t1-source-" + obj + "-" + condition
    cmd = [
        "{python}",
        "-B",
        "{source}/scripts/t1_access.py",
        "--config",
        "{config}",
        "--out",
        "{out}",
        *options(pins),
    ]
    return stage(
        roots,
        kind,
        "dataset_build",
        pins,
        {"object_id": obj, "logical_condition": condition},
        cmd,
        "references.json" if opening else "source-provenance.json",
        32 * 1024**2,
        retry=retry,
    )


def science_input(roots, source, frozen):
    src = load(locate(roots, source))
    names = SLOTS[src["condition"]]
    base = frozen["sources"]["reconstruction"]
    roles = {
        k: bind(roots, base, "payload/" + n)
        for k, n in (("recipe", "recipe.json"), ("worker", "worker.zip"), ("rules", "init-protocol.json"))
    }
    roles["cameras"] = bind(roots, source, "payload/source_cameras.json")
    roles["rendercameras"] = bind(roots, source, "payload/render_cameras.json")
    roles.update({n: bind(roots, source, "payload/" + src["sources"][n]["rgb"]["path"]) for n in names})
    recipe = load(locate(roots, roles["recipe"]))
    master = load(locate(roots, base))
    protocol = load(locate(roots, bind(roots, base, "payload/protocol.json")))
    require(
        master["optimizer_seed"] == recipe["seed"] == 43171
        and recipe["stop_step"] == 5000
        and roles["recipe"]["sha256"] == master["recipe_sha256"]
        and roles["worker"]["sha256"] == protocol["worker_zip_sha256"]
        and sha(PROJECT / "scripts/recovery_triangulate.py") == protocol["geometry_code_sha256"]
        and roles["rules"]["sha256"] == protocol["source_initialization_sha256"],
        "Wrong optimizer registration",
    )
    key = {
        "object_id": src["object_id"],
        "condition": src["condition"],
        "ordered_ids": [src["official_by_slot"][n] for n in names],
        "ordered_original_sha256": [src["sources"][n]["original_sha256"] for n in names],
        "input_bytes": {n: r["sha256"] for n, r in roles.items()},
        "seed": 43171,
        "init_code": protocol["geometry_code_sha256"],
        "trajectories": "cold start; 5000 updates with the original resource guard",
    }
    return roles, key


def fit(roots, obj, which):
    s = roots[0]
    _, _, frozen = authority(roots)
    src = selected(s, "t1-source-" + obj + "-" + which)
    roles, key = science_input(roots, src, frozen)
    signature = hashlib.sha256(encoded(key)).hexdigest()
    name = "t1-" + obj + "-" + which
    policy = s / "work/fit-identities" / obj / (signature + ".json")
    if policy.exists():
        prior = load(policy)
        require(prior["source_key"] == key, "Source key collision")
        init = prior["initialization"]
        ticket = prior["ticket"]
        for k, expected in (("init.json", prior["init_json_sha256"]), ("init.npz", prior["init_npz_sha256"])):
            require(bind(roots, init, "payload/" + k)["sha256"] == expected, "Aliased initializer changed")
        require(
            prior["scientific_job_sha256"] == load(locate(roots, ticket))["job_sha256"],
            "Aliased complete optimizer/configuration bytes changed",
        )
        entry = dict(prior, source=src, alias_of=prior["owner"] if prior["owner"] != which else None)
        publish(s / "work/branches" / obj / (which + ".json"), encoded(entry))
        return entry
    reg = frozen["sources"]["reconstruction"]
    cond = which[:2]
    ids = SLOTS[cond]
    config = {
        "scene_family_id": obj,
        "scene_id": obj,
        "condition_id": cond,
        "registration": reg,
        "source_parent": src,
        "recipe_sha256": roles["recipe"]["sha256"],
        "initialization_protocol_sha256": roles["rules"]["sha256"],
        "source_camera_sha256": roles["cameras"]["sha256"],
        "source_rgb_sha256": {n: roles[n]["sha256"] for n in ids},
        "generator_source_ids": ids,
    }
    initpins = {k: roles[k] for k in ["cameras", "rules", *ids]}
    cmd = [
        str(GEOMETRY),
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
    for n in ids:
        cmd += ["--source", n, "{input:" + n + "}"]
    init = stage(roots, name + "-init", "derivation", initpins, config, cmd, "init.json", 2 * 1024**2)
    pins = {k: roles[k] for k in ("cameras", "rendercameras", "recipe", "worker", *ids)}
    pins.update(init=bind(roots, init, "payload/init.npz"), initmeta=init)
    parents = {v["run_id"]: v["manifest_sha256"] for v in [*pins.values(), src, reg]}
    cfg = dict(
        **config,
        scientific_trial_id=name,
        forced_fallback_fixture=False,
        requested_render_steps=[5000],
        worker_sha256=roles["worker"]["sha256"],
        initialization=init,
        source=src,
        freeze=reg,
        input_origin_ids={k: v["run_id"] for k, v in pins.items()},
        windows_experiment="general-t1-confirmation",
        windows_parents=[{"run_id": k, "manifest_sha256": parents[k]} for k in sorted(parents)],
    )
    cmd = [
        "{python}",
        "-B",
        "{source}/scripts/t0_worker_pack.py",
        "dispatch",
        "--config",
        "{config}",
        "--out",
        "{out}",
        *options(pins),
    ]
    ticket = stage(roots, name + "-dispatch", "dataset_build", pins, cfg, cmd, "ticket.json", 3 * 1024**2)
    entry = {
        "owner": which,
        "alias_of": None,
        "source": src,
        "initialization": init,
        "ticket": ticket,
        "source_key": key,
        "scientific_job_sha256": load(locate(roots, ticket))["job_sha256"],
        "init_json_sha256": init["sha256"],
        "init_npz_sha256": pins["init"]["sha256"],
    }
    publish(policy, encoded(entry))
    publish(s / "work/branches" / obj / (which + ".json"), encoded(entry))
    return entry


def publish_registry(roots, name, jobs, exchange):
    store = roots[0]
    rows = []
    for item in jobs:
        if item["alias_of"]:
            continue
        pin = item["ticket"]
        job = load(locate(roots, pin))
        trial = job["trial_id"]
        for member in ("dispatch.zip", "t0_worker_run.py", "ticket.json", "READY.json"):
            data = locate(roots, bind(roots, pin, "payload/" + member)).read_bytes()
            for folder in (store / "transport/Handoffs/t1-v1", exchange):
                publish(folder / trial / member, data)
        rows.append(
            dict(
                trial_id=trial,
                object_id=item["source_key"]["object_id"],
                condition=item["owner"],
                ticket=pin,
                **{k: job[k] for k in ("job_sha256", "package", "launcher")},
            )
        )
    require(len(rows) > 0 and len({j["trial_id"] for j in rows}) == len(rows), "Empty or duplicate registry")
    reg = {
        "schema": "el-b2-windows-batch-1",
        "batch": name,
        "experiment": "general-t1-confirmation",
        "entry": authority(roots)[0],
        "jobs": rows,
    }
    for folder in (store / "transport/Handoffs/t1-v1", exchange):
        publish(folder / "b2_worker_queue.py", (PROJECT / "scripts/b2_worker_queue.py").read_bytes())
        publish(folder / (name + ".json"), encoded(reg))
    return sha(exchange / (name + ".json"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["m2", "m4", "references"])
    p.add_argument("--store", type=Path, default=DEFAULT)
    p.add_argument("--exchange", type=Path)
    p.add_argument("--object", choices=OBJECTS)
    p.add_argument("--opening", type=Path)
    a = p.parse_args()
    roots = a.store.resolve(), a.store.resolve() / "history"
    objs = [a.object] if a.object else OBJECTS
    rows = []
    for o in objs:
        if a.action == "references":
            require(a.opening is not None, "Sealed joint target opening required")
            print(json.dumps(acquire(roots, o, "references", load(a.opening))), flush=True)
            continue
        for c in ["M2-fixed"] if a.action == "m2" else CONDITIONS[1:]:
            src = acquire(roots, o, c)
            item = fit(roots, o, c)
            rows.append(item)
            print(o, c, src["run_id"], item["ticket"]["run_id"], item["alias_of"], flush=True)
    if rows:
        require(a.exchange is not None, "Specify the registered private Handoffs path")
        name = "T1-" + a.action + "-" + (a.object or "all")
        print(name, publish_registry(roots, name, rows, a.exchange.resolve()), flush=True)


if __name__ == "__main__":
    main()
