"""Prepare the two fixed batches or expose just an original two-photo M2 selector input packet."""

import argparse
from pathlib import Path
from b2_reconstruct import (
    GEOMETRY,
    PROJECT,
    bind,
    development_source,
    encoded,
    input_args,
    load,
    prepare,
    publish,
    record,
    require,
    resolve,
    sha,
)


def selector_packet(store, history, obj, interpreter):
    require(obj in ("set1", "set6"), "Only previously exposed D starts are available")
    prefix = history / "work/r-control" / ("r2-scale-decay-02-" + obj + "-M2")
    norm = load(prefix.with_name(prefix.name + "-normalize.json"))
    ticket = load(prefix.with_name(prefix.name + "-ticket.json"))
    source, generation, renderer = (
        ticket["handoff"]["run_id"],
        norm["generation"]["run_id"],
        norm["render"]["run_id"],
    )
    pins = {
        "checkpoint": bind(history, generation, "payload/checkpoints/step-5000.pt"),
        "recipe": ticket["recipe"],
        "cameras": bind(history, source, "payload/handoff/source_cameras.json"),
    }
    require(
        pins["checkpoint"]["manifest_sha256"] == norm["generation"]["manifest_sha256"],
        "Different historical optimizer",
    )
    for index in ("00", "03"):
        pins["s" + index] = bind(history, source, "payload/handoff/sources/s" + index + ".png")
        pins["r" + index] = bind(history, renderer, "payload/step-5000/s" + index + ".npz")
        require(
            pins["r" + index]["manifest_sha256"] == norm["render"]["manifest_sha256"],
            "Different historical renderer",
        )
    cmd = [
        str(interpreter.absolute()),
        "-B",
        "{source}/scripts/b2_worker_origin.py",
        "selector",
        "--config",
        "{config}",
        "--out",
        "{out}",
        *input_args(pins),
    ]
    return record(
        store,
        history,
        "b2-nbv-original-" + obj,
        "archive_verify",
        pins,
        {
            "experiment_id": "b2-nbv-selector-original-readback",
            "trial_id": "b2-" + obj + "-source-only-m2",
            "object_id": obj,
        },
        cmd,
        "payload/selector-input.json",
        8 * 1024**2,
    )


def prepare_fixed(store, history, batch, exchange, geometry):
    control = store / "work/controller"

    def authority(seed):
        return load(control / f"b2-reconstruction-seed-{seed}/selected.json")

    jobs = []
    primary = authority(43171)
    if batch == "C6-fixed-12":
        cohort = load(control / "b2-dtu-cohort/selected.json")
        roster = load(resolve((store, history), cohort))
        require(
            len(roster["splits"]["C6"]) == len(set(roster["splits"]["C6"])) == 6
            and set(roster["splits"]["T1"]).isdisjoint(roster["splits"]["C6"]),
            "Wrong calibration roster",
        )
        cells = [(o, c, 43171) for o in roster["splits"]["C6"] for c in ("M2", "M4")]
        doc = dict(
            schema="el-b2-windows-batch-1", batch="C6-fixed", authority=primary, cohort=cohort, jobs=jobs
        )
    else:
        require(batch == "D-sensitivity-4", "Unregistered batch")
        cohort = None
        cells = [(o, "M4", seed) for o in ("set1", "set6") for seed in (43172, 43173)]
        doc = dict(
            schema="el-b2-windows-batch-1",
            batch="D-fixed-M4-sensitivity",
            authorities={seed: authority(seed) for seed in (43171, 43172, 43173)},
            jobs=jobs,
        )
    for obj, cond, seed in cells:
        src = (
            load(control / f"b2-C6-source-{obj}-{cond}/selected.json")
            if cohort
            else development_source(store, history, primary, obj)
        )
        output = prepare(
            store, history, authority(seed), src, obj, cond, cond + "-fixed", geometry=geometry, cohort=cohort
        )
        path = Path(output["path"])
        if not cohort:
            first = load(history / "work/r-control" / f"r2-scale-decay-02-{obj}-M4-ticket.json")[
                "initialization"
            ]
            new = load(resolve((store, history), load(path / "READY.json")["initialization"]))
            old = load(resolve((store, history), first))
            require(
                all(
                    new[k] == old[k]
                    for k in (
                        "parameter_sha256",
                        "source_camera_sha256",
                        "source_rgb_sha256",
                        "scene_scale_m",
                        "N",
                        "eligibility",
                        "init_npz_sha256",
                    )
                ),
                "Fixed sensitivity starting point changed",
            )
        for name in ("dispatch.zip", "ticket.json", "READY.json", "dispatch-pin.json", "t0_worker_run.py"):
            publish(exchange / path.name / name, (path / name).read_bytes())
        row = {
            "object_id": obj,
            "condition": cond,
            "layout": cond + "-fixed",
            "ticket": output["ticket"],
            "trial_id": path.name,
            **{k: output["job"][k] for k in ("job_sha256", "package", "launcher")},
        }
        if not cohort:
            row["optimizer_seed"] = seed
        jobs.append(row)
    for folder in (store / "transport/Handoffs/b2-v1", exchange):
        publish(folder / (batch + ".json"), encoded(doc))
    return sha(exchange / (batch + ".json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=["C6-fixed-12", "D-sensitivity-4", "selector-packet"])
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--exchange", type=Path)
    p.add_argument("--object", choices=["set1", "set6"])
    p.add_argument("--python", type=Path, default=PROJECT / ".venv-vision-cpu/bin/python")
    p.add_argument("--geometry-python", type=Path, default=GEOMETRY)
    a = p.parse_args()
    store = a.store.resolve()
    history = a.history.resolve()
    if a.operation == "selector-packet":
        print(selector_packet(store, history, a.object, a.python))
    else:
        require(a.exchange is not None, "Supply an explicit project Handoffs path")
        print(prepare_fixed(store, history, a.operation, a.exchange, a.geometry_python))
