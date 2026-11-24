"""Receive one sealed native result per issued fit and produce its development diagnostics when applicable."""

import argparse
import json
import time
from pathlib import Path

from dual_records import resolve, selected
from evidence_lens._bundle import load
from evidence_lens._verify import require
from many_upstream_bank import control as assemble
from many_upstream_diagnostics import DEV, control as diagnose


def exists(store, label):
    return (store / "work" / "controller" / label / "selected.json").is_file()


def completion_ready(folder):
    side = folder / "completed.json"
    if not side.is_file():
        return None
    try:
        receipt = load(side)
        name = receipt["package"]
        require(Path(name).name == name, "unexpected returned ZIP path")
        archive = side.parent / name
        if archive.is_file() and archive.stat().st_size == receipt["bytes"]:
            return side
    except (FileNotFoundError, json.JSONDecodeError):
        # The transfer can expose an incomplete sidecar before its final bytes arrive.
        pass
    return None


def receive(store, exchange, queue_label, wait):
    queue = selected(store, queue_label)
    plan = load(resolve((store, store / "history"), queue))
    allowed = {"b-training-development-fit-queue": "el-many-development-fit-queue-1",
               "b-confirmation-fit-queue": "el-many-confirmation-fit-queue-1"}
    require(plan["schema"] == allowed[queue_label], "unregistered queue type")
    pending = {row["tag"]: row for row in plan["fits"]}
    while pending:
        completed = []
        for tag, row in pending.items():
            require(selected(store, "b-fit-ticket-" + tag) == row["pin"], "fit ticket changed after dispatch")
            bank = "b-bank-" + tag
            if not exists(store, bank) and not exists(store, "b-terminal-failure-" + tag):
                side = completion_ready(exchange / row["trial"])
                if side is None:
                    continue
                result = assemble(store, tag, side)
                print(json.dumps({"tag": tag, "result": result}), flush=True)
            if tag in DEV and exists(store, bank) and not exists(store, "b-upstream-dev-" + tag):
                print(json.dumps({"development": tag, "result": diagnose(store, "evaluate", tag)}), flush=True)
            completed.append(tag)
        for tag in completed:
            del pending[tag]
        if not wait or not pending:
            break
        time.sleep(15)
    if all(exists(store, "b-upstream-dev-" + key) for key in DEV) and not exists(store, "b-development-diagnostics"):
        print(json.dumps({"development_summary": diagnose(store, "summary")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--exchange", type=Path, required=True)
    parser.add_argument("--queue", choices=("b-training-development-fit-queue", "b-confirmation-fit-queue"), required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    receive(args.store.resolve(), args.exchange, args.queue, args.wait)
