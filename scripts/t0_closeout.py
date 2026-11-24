"""Build the public numeric T0 report and record the closed batch's resource census."""

import argparse
import json
from pathlib import Path

from evidence_lens._bundle import RUN_ID, load, save
from evidence_lens._verify import require
from t0_records import record, resolve


def close(store, history, action):
    state = store / "work/controller"
    evaluation = load(state / "evaluate/selected.json")
    intake = load(state / "dtu-numerical-intake-complete/selected.json")
    roster = load(state / "roster-pin.json")
    if action == "report":
        tables = resolve((store, history), evaluation).parent
        return record(store, history, "results-report", "report",
                      {"evaluation": evaluation, "roster": roster, "intake": intake},
                      {"scope": "Object-equal report of the single sealed T0 evaluation"},
                      ["{python}", "{source}/scripts/t0_results.py", "--tables", str(tables),
                       "--roster", "{input:roster}", "--out", "{out}"], "payload/README.md", 30000)
    report = load(state / "results-report/selected.json")
    path = state / "closed-resource-inventory.json"
    if not path.exists():
        runs = {}
        for parent in (store / "runs", store / "quarantine"):
            for bundle in parent.iterdir():
                if not RUN_ID.fullmatch(bundle.name) or not (bundle / "completion.json").is_file():
                    continue
                seal = load(bundle / "completion.json")
                require(bundle.name == seal["run_id"] and
                        runs.setdefault(bundle.name, seal["manifest_sha256"]) == seal["manifest_sha256"],
                        "ambiguous source for a closed run")
        save(path, {"runs": [{"run_id": rid, "manifest_sha256": sha} for rid, sha in sorted(runs.items())]})
    return record(store, history, "batch-resource-census", "report",
                  {"evaluation": evaluation, "intake": intake, "report": report},
                  {"scope": "All sealed execution originals directly present in this batch store at inventory close"},
                  ["{python}", "{source}/scripts/t0_resource_ledger.py", "--store", str(store),
                   "--inventory", "{input:inventory}", "--out", "{out}"], "payload/resources.json",
                  300000, {"inventory": path})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("report", "resources"))
    for name in ("store", "history"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(close(args.store.resolve(), args.history.resolve(), args.action), sort_keys=True))
