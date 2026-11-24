"""Advance T1 source-only execution from immutable Windows cumulative receipts, without opening targets."""

import argparse
import time
from pathlib import Path

from t1_access import CONDITIONS, OBJECTS
from t1_probe_terminal import confirm
from t1_receive import decide, poll, receive_probe
from t1_study import DEFAULT, acquire, fit, locate, named, publish_registry, selected

from evidence_lens._bundle import load, sha


def ready_returns(returns, registry):
    # Ignore provider-directory timing until every file in a prefix exists at its
    # advertised size. The original receiver separately verifies each full SHA-256.
    index = load(registry)
    for n, j in enumerate(index["jobs"], 1):
        acc = returns / (registry.stem + f"-completed-{n:02}.json")
        root = returns / j["trial_id"]
        summary = root / "completed.json"
        if not acc.is_file():
            break
        if not summary.is_file():
            return False
        rec = load(summary)
        if any(
            not (root / k).is_file() or (root / k).stat().st_size != spec["bytes"]
            for k, spec in rec["files"].items()
        ):
            return False
    return True


def poll_available(roots, registry, expected, returns):
    rows = load(registry)["jobs"]
    fresh = any(
        (returns / (registry.stem + f"-completed-{i:02}.json")).is_file()
        and not named(roots[0], "t1-readback-" + r["object_id"] + "-" + r["condition"])
        for i, r in enumerate(rows, 1)
    )
    if fresh and ready_returns(returns, registry):
        poll(roots, registry, expected, returns)


def advance(roots, exchange, returns, expected):
    s = roots[0]
    issued = s / "transport/Handoffs/t1-v1"
    m2 = issued / "T1-m2-all.json"
    poll_available(roots, m2, expected, returns)
    finished = 0
    for obj in OBJECTS:
        name = "T1-m4-" + obj
        reg = issued / (name + ".json")
        if reg.is_file():
            poll_available(roots, reg, sha(reg), returns)
            done = all(named(s, "t1-readback-" + obj + "-" + r["condition"]) for r in load(reg)["jobs"])
            finished += int(done)
            continue
        decision = "t1-decisions-" + obj
        if not named(s, decision):
            if not named(s, "t1-readback-" + obj + "-M2-fixed"):
                break
            reply = returns / "probes" / obj / "completed.json"
            if not reply.is_file():
                break
            rec = load(reply)
            if not (reply.parent / rec["zip"]).is_file():
                break
            selection = (
                confirm(roots, obj, returns)
                if rec["status"] == "failed"
                else decide(roots, obj, receive_probe(roots, obj, returns))
            )
            print("SELECTION TERMINAL", obj, selection, flush=True)
        if load(locate(roots, selected(s, decision)))["schema"] == "el-t1-probe-terminal-1":
            finished += 1
            continue
        work = []
        for c in CONDITIONS[1:]:
            acquire(roots, obj, c)
            work.append(fit(roots, obj, c))
        print("REGISTRY", name, publish_registry(roots, name, work, exchange), flush=True)
    return finished == len(OBJECTS)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, default=DEFAULT)
    p.add_argument("--exchange", type=Path, required=True)
    p.add_argument("--returns", type=Path, required=True)
    p.add_argument("--m2-sha", required=True, help="Independently issued SHA of the S0 registry")
    p.add_argument("--watch", action="store_true")
    a = p.parse_args()
    roots = a.store.resolve(), a.store.resolve() / "history"
    while True:
        done = advance(roots, a.exchange.resolve(), a.returns.resolve(), a.m2_sha)
        if done or not a.watch:
            break
        time.sleep(8)
