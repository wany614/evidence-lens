"""Score independently accepted native endpoints and assemble the closed T0 matrix without reading target photographs."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from evidence_lens._bundle import load, save
from evidence_lens._verify import require
from make_handoff import publish
from t0_records import resolve
from t0_study import study


def collect(store, history, cpu_python, wait):
    store = store.resolve()
    history = history.resolve()
    state = store / "work/controller"
    fp = load(state / "freeze/selected.json")
    fr = load(resolve((store, history), fp))
    cells = {}
    deadline = time.monotonic() + wait
    while True:
        for obj in fr["object_ids"]:
            for cond in fr["conditions"]:
                key = obj + "-" + cond
                if key in cells:
                    continue
                terminal = state / ("disposition-" + key) / "selected.json"
                if terminal.exists():
                    failure = load(terminal)
                    d = load(resolve((store, history), failure))
                    cells[key] = {"status": d["status"], "reason": d["reason"], "disposition": failure}
                    continue
                selection = store / "t0/reconstruction" / obj / cond / "selection.json"
                if not selection.is_file():
                    continue
                leaf = load(selection)
                require(
                    (leaf["object_id"], leaf["condition"]) == (obj, cond) and leaf["status"] == "succeeded",
                    "non-success needs a recorded terminal disposition",
                )
                proof_dir = state / "readbacks"
                proof_dir.mkdir(exist_ok=True)
                readback = proof_dir / (key + ".json")
                save(readback, leaf["readback"])
                score = study(
                    SimpleNamespace(
                        store=store,
                        history=history,
                        action="score",
                        object=obj,
                        condition=cond,
                        readback=readback,
                        cpu_python=cpu_python,
                        retry_of=None,
                    )
                )
                cells[key] = {
                    "status": "scored",
                    "case": leaf["case"],
                    "readback": leaf["readback"],
                    "scores": score,
                }
                print(f"{key} {score['run_id']}", flush=True)
        if len(cells) == fr["expected_fits"]:
            from evidence_lens._bundle import encoded

            path = state / "closed-matrix.json"
            publish(path, encoded({"freeze": fp, "cells": cells}))
            return path
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"{len(cells)}/{fr['expected_fits']} terminal registrations, target references remain closed"
            )
        time.sleep(3)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("store", "history", "cpu-python"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--wait-seconds", type=float, default=0)
    a = p.parse_args()
    print(collect(a.store, a.history, a.cpu_python, a.wait_seconds))
