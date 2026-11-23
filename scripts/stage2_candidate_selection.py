"""Record the preregistered four-fit learning-rate choice or eight-fit primary choice on the Mac."""
import argparse
import json
from pathlib import Path

from evidence_lens._bundle import load, sha
from evidence_lens.formal_inputs import require
from vision_record import Recorder, parse_pin, publish


def record(args):
    project, store = args.project.resolve(), args.store.resolve()
    catalog = project / "protocol/stage2-families-v1.json"
    study = project / "protocol/stage2-training-v1.json"
    require(sha(catalog) == load(study)["catalog_sha256"], "training protocol refers to a different roster")
    state = store / "stage2" / sha(catalog) / "candidate-selection"
    fits = sorted((parse_pin(f) for f in args.fit), key=lambda pin: pin["run_id"])
    expected = 4 if args.stage == "learning-rate" else 8
    require(len(fits) == len({p["run_id"] for p in fits}) == expected, "incomplete or duplicate original fits")
    recorder = Recorder(project, store, args.shelf, state, sha(catalog), sha(study))
    inputs = {}
    for index, pin in enumerate(fits, 1):
        training_path, _ = recorder.asset(pin, "training.json")
        model_path, _ = recorder.asset(pin, "package/model.json")
        inputs["training_" + str(index)] = (training_path, pin["run_id"])
        inputs["model_" + str(index)] = (model_path, pin["run_id"])
    roster = state / (args.stage + "-roster.json")
    publish(roster, {"schema": "el-vision-candidate-roster-1", "stage": args.stage, "runs": fits})
    inputs["roster"] = (roster, None)
    return recorder.stage(args.stage, "threshold_select", fits, inputs, ["vision_worker.py"],
                          ["vision_worker.py", "select", "--stage", args.stage,
                           "--roster", "{input:roster}", "--store", str(store), "--output", "{out}"],
                          "selection.json", 300_000, {"registered_candidates": expected})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("learning-rate", "primary"), required=True)
    parser.add_argument("--fit", action="append", required=True, help="A successful formal RUN_ID=MANIFEST_SHA256")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--shelf", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    print(json.dumps(record(parser.parse_args()), sort_keys=True))
