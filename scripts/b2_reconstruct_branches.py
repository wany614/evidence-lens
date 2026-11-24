"""Expand two sealed development decisions into registered, unique source-only Windows jobs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from b2_reconstruct import (
    GEOMETRY,
    STUDY,
    bind,
    development_source,
    encoded,
    load,
    prepare,
    publish,
    require,
    resolve,
    sha,
)


def dispatch(store, history, authority, decisions, exchange, geometry=GEOMETRY):
    study = load(STUDY)
    strategies = study["development_selection"]["strategies"]
    require(set(decisions) == set(study["development_objects"]), "Both development objects are required")
    pending = []
    for obj in study["development_objects"]:
        root = decisions[obj]
        choice = load(resolve((store, history), root))
        require(
            choice["schema"] == "el-b2-joint-nbv-choices-1"
            and choice["object_id"] == obj
            and set(choice["selections"]) == set(strategies),
            "Wrong sealed development selection",
        )
        for strategy in strategies:
            pin = bind(store, root["run_id"], "payload/" + strategy + ".json")
            dec = load(resolve((store, history), pin))
            require(
                pin["manifest_sha256"] == root["manifest_sha256"]
                and pin["sha256"] == choice["selections"][strategy]
                and dec["object_id"] == obj
                and dec["strategy_id"] == strategy
                and dec["selected_official_ids"] == choice["results"][strategy],
                "Selection does not match joint decision",
            )
            pending.append((obj, strategy, pin, tuple(sorted(dec["selected_official_ids"]))))

    def acquire(item):
        obj, strategy, pin, pair = item
        source = development_source(store, history, authority, obj, pin)
        doc = load(resolve((store, history), source))
        require(
            doc["selection"] == pin and [doc["official_by_slot"][s] for s in ("s01", "s02")] == list(pair),
            "Authorized source does not implement the selected pair",
        )
        return dict(object_id=obj, branch=strategy, selection=pin, source=source, pair=list(pair)), doc

    with ThreadPoolExecutor(max_workers=4) as pool:
        opened = list(pool.map(acquire, pending))
    groups = {}
    observations = []
    for observation, source_doc in opened:
        obj, pair = observation["object_id"], tuple(observation["pair"])
        key = (obj, pair)
        fingerprint = (
            source_doc["source_camera_sha256"],
            source_doc["render_camera_sha256"],
            tuple(
                (s, source_doc["sources"][s]["official_view_id"], source_doc["sources"][s]["rgb"]["sha256"])
                for s in ("s00", "s01", "s02", "s03")
            ),
        )
        if key in groups:
            owner, previous = groups[key]
            require(previous == fingerprint, "Same pair has different slot, source bytes or calibration")
            observation["reuse_of"] = owner["branch"]
        else:
            groups[key] = observation, fingerprint
            observation["reuse_of"] = None
        observations.append(observation)

    def package(entry):
        row, _ = entry
        result = prepare(
            store,
            history,
            authority,
            row["source"],
            row["object_id"],
            "M4",
            "M4-selected",
            geometry=geometry,
            selection=row["selection"],
        )
        path = Path(result["path"])
        for name in ("dispatch.zip", "ticket.json", "READY.json", "dispatch-pin.json", "t0_worker_run.py"):
            publish(exchange / path.name / name, (path / name).read_bytes())
        j = result["job"]
        return dict(
            object_id=row["object_id"],
            condition="M4",
            layout="M4-selected",
            optimizer_seed=43171,
            strategy_id=row["branch"],
            trial_id=result["trial_id"],
            ticket=result["ticket"],
            job_sha256=j["job_sha256"],
            package=j["package"],
            launcher=j["launcher"],
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = list(pool.map(package, groups.values()))
    by_pair = dict(zip(groups, jobs, strict=True))
    for row in observations:
        job = by_pair[row["object_id"], tuple(row["pair"])]
        row.update(ticket=job["ticket"], trial_id=job["trial_id"])
    manifest = dict(schema="el-b2-windows-batch-1", batch="D-selected-unique", authority=authority, jobs=jobs)
    base = store / "transport/Handoffs/b2-v1"
    for folder in (base, exchange):
        publish(folder / "D-selected-unique.json", encoded(manifest))
    doc = dict(
        schema="el-b2-development-sources-and-branches-1",
        decisions=decisions,
        authority=authority,
        sources=observations,
        unique_jobs=len(jobs),
        observations=len(observations),
        dispatch_registry_sha256=sha(base / "D-selected-unique.json"),
    )
    publish(store / "work/controller/d-authorized-branches.json", encoded(doc))
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("store", "history", "authority", "exchange"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--geometry-python", type=Path, default=GEOMETRY)
    for obj in ("set1", "set6"):
        p.add_argument("--" + obj, required=True, help="run ID = manifest SHA without spaces")
    a = p.parse_args()
    decisions = {}
    for obj in ("set1", "set6"):
        rid, digest = getattr(a, obj).split("=")
        pin = bind(a.store, rid, "payload/choices.json")
        require(pin["manifest_sha256"] == digest, "Wrong joint choice manifest")
        decisions[obj] = pin
    print(
        dispatch(a.store, a.history, load(a.authority), decisions, a.exchange, a.geometry_python)[
            "dispatch_registry_sha256"
        ]
    )
