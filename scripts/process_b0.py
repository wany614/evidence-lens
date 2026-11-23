"""Complete Mac diagnostics for one nominated reference and received B0 generation/render pair."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evidence_lens._bundle import load, sha
from evidence_lens._index import writer_gate
from evidence_lens._verify import require
from formal_lineage import family_record
from select_trial import freeze

ROOT = Path(__file__).resolve().parents[1]


def decision(store, trial, experiment, kind, parents):
    """Resume by trial, experiment, stage and exact parents before reading outputs."""
    matches = []
    for tree in ("work", "runs"):
        for path in (store / tree).glob("*/run.json"):
            row = load(path)
            if (row["trial_id"] == trial and row["experiment_id"] == experiment and row["run_kind"] == kind
                    and set(row["parent_run_ids"]) == set(parents)):
                require(tree != "work", f"matching {kind} worker is still active")
                matches.append(row)
    matches.sort(key=lambda r: (r["started_at_utc"], r["run_id"]))
    success = [r for r in matches if r["execution_status"] == "succeeded"]
    require(len(success) <= 1, f"multiple successful {kind} runs for one parent set")
    if success:
        return success[0]["run_id"]
    require(not matches, f"inspect existing failed {kind} attempt before retrying: {[r['run_id'] for r in matches]}")
    return None


def invoke(command):
    result = subprocess.run(command, check=True, cwd=ROOT, stdout=subprocess.PIPE, text=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def complete(store, args):
    catalog = ROOT / "protocol/families-v1.json"
    row = family_record(catalog, ROOT / "configs/formal-freeze-v1.json", args.family)
    trial = row["trial_id"]
    reference = store / "formal-references" / sha(catalog) / (args.family + ".json")
    ref = load(reference)
    invoke([sys.executable, "scripts/verify_reference_config.py", "--store", str(store), "--selection", str(reference)])
    ids = {"generation": args.generation_run, "render": args.render_run,
           "reference": ref["reference_run_id"], "readback": ref["readback_run_id"]}
    exp = load(store / "runs" / ids["generation"] / "run.json")["experiment_id"]
    base = ["--store", str(store), "--families", str(catalog), "--reference-run", ids["reference"], "--readback-run",
            ids["readback"], "--generation-run", ids["generation"], "--render-run", ids["render"]]
    for stage, role, kind, parent_roles in (
        ("score", "score", "model_score", ("reference", "readback", "render")),
        ("evaluate", "evaluation", "evaluation", ("reference", "readback", "render", "score")),
        ("verify", "recompute", "archive_verify", ("reference", "readback", "render", "score", "evaluation")),
    ):
        rid = decision(store, trial, exp, kind, [ids[r] for r in parent_roles])
        if not rid:
            flags = []
            if stage == "score":
                flags += ["--cpu-python", str(args.cpu_python.absolute())]
            if "score" in parent_roles:
                flags += ["--score-run", ids["score"]]
            if "evaluation" in parent_roles:
                flags += ["--evaluation-run", ids["evaluation"]]
            receipt = invoke([sys.executable, "scripts/mac_trial.py", stage, *base, *flags])
            rid = receipt["run_id"]
        ids[role] = rid
    path, status = freeze(store, trial, ids)
    output = {"scene_family_id": args.family, "trial_id": trial, "selection_sha256": sha(path), "sel_status": status, "runs": ids}
    if row["split"] in ("train", "development"):
        rec = invoke([str(args.train_python.absolute()), "scripts/formal_batch.py", "prepare", "--store", str(store), "--family", args.family])
        output["extract"] = {k: rec[k] for k in ("run_id", "manifest_sha256", "extract_manifest_sha256")}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("family", "generation-run", "render-run"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--cpu-python", type=Path, required=True)
    p.add_argument("--train-python", type=Path, required=True)
    args = p.parse_args()
    store = args.store.resolve()
    row = family_record(ROOT / "protocol/families-v1.json", ROOT / "configs/formal-freeze-v1.json", args.family)
    with writer_gate(store / "formal" / row["dataset_hash"] / ".mac-locks" / args.family):
        print(json.dumps(complete(store, args)), flush=True)


if __name__ == "__main__":
    main()
