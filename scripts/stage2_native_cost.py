"""Measure the released neural scorer on six native views of the first registered new development family."""
import argparse
import json
import os
import sys
from pathlib import Path

from evidence_lens._bundle import load, sha
from evidence_lens.formal_inputs import PinnedRun, require


def record(args):
    from vision_record import Recorder, parse_pin
    project = args.project.resolve()
    store = args.store.resolve()
    roster_path = project / "protocol/stage2-families-v1.json"
    protocol = project / "protocol/stage2-training-v1.json"
    catalog = load(roster_path)
    digest = sha(roster_path)
    require(load(protocol)["catalog_sha256"] == digest, "training protocol and population differ")
    eligible = sorted(r["scene_family_id"] for r in catalog["families"] if r["split"] == "development")
    require(len(eligible) == 4, "expected the four registered new development families")
    family = eligible[0]
    choice = load(store / "trials" / (family + "-B0") / "trial_selection.json")
    require(choice["selection_status"] == "selected" and choice["scene_family_id"] == family
            and choice["condition_id"] == "B0", "missing selected development trial")
    ref, pred, trained = [PinnedRun(store, pin) for pin in
                          (choice["selected"]["reference"], choice["selected"]["render"], parse_pin(args.model))]
    model = trained.open("payload/package/model.json")
    identity = load(model)
    require(trained.run["run_kind"] == "model_fit" and identity["phase"] == "train"
            and identity["branch"] in {"L2-frozen", "L2-vision-ft"} and identity["seed"] == 1729
            and identity["model_id"] == trained.run["run_id"] and identity["threshold"] is not None,
            "benchmark requires a formal seed-A model with its frozen development threshold")
    require(ref.run["split"] == pred.run["split"] == "development" and pred.run["condition_id"] == "B0"
            and pred.run["scene_family_id"] == family and ref.run["scene_family_id"] == family,
            "crossed development reference or prediction")
    runtime = trained.open("payload/package/score-native.py")
    selected_pin = parse_pin(args.selection)
    selected_run = PinnedRun(store, selected_pin)
    selection = load(selected_run.open("payload/selection.json"))
    choice_pin = selection["chosen"][identity["branch"]]
    require(selection["schema"] in {"el-vision-lr-selection-1", "el-vision-primary-selection-1"}
            and choice_pin["run_id"] == identity["model_id"]
            and choice_pin["manifest_sha256"] == parse_pin(args.model)["manifest_sha256"]
            and choice_pin["model_sha256"] == sha(model), "timing candidate is not the registered seed-A choice")
    prediction = pred.open("payload/prediction.json").parent
    camera = ref.open("payload/cameras.json")
    source = ref.open("payload/input.png")
    state = store / "stage2" / digest / "costs"
    recorder = Recorder(project, store, args.shelf, state, digest, sha(protocol))
    # Only the selected source image and calibration cross the reference boundary. The prediction
    # and package are original immutable recorder members, checked and opened again by their own readers.
    inputs = {"model": (model, trained.run["run_id"]), "runtime": (runtime, trained.run["run_id"]),
              "input_image": (source, ref.run["run_id"]), "cameras": (camera, ref.run["run_id"]),
              "prediction": (prediction / "prediction.json", pred.run["run_id"])}
    pins = [parse_pin(args.model), selected_pin, choice["selected"]["reference"], choice["selected"]["render"]]
    result = recorder.stage("native-" + trained.run["run_id"], "model_score", pins, inputs, ["stage2_native_cost.py"],
        ["stage2_native_cost.py", "launch", str(args.python.absolute()), str(model.parent), sha(runtime),
         "--model-sha256", sha(model),
         "--input", "{input:input_image}", "--cameras", "{input:cameras}", "--predictions", str(prediction),
         "--view", "all", "--warm-repeats", "3", "--output", "{out}"], "native-cost.json", 50_000_000,
        {"family": family, "branch": identity["branch"], "model_sha256": sha(model), "pipeline":
         "Isolated CPU process reads actual trained package, source, cameras and native predictions through NPZ writing."})
    cost = load(store / "runs" / result["run_id"] / "payload/native-cost.json")
    require(cost["device"] == "cpu" and cost["model_sha256"] == sha(model) and cost["model_id"] == identity["model_id"]
            and cost["warm_views"] == 18 and len(cost["views"]) == 24 and cost["all_measured_views_512"],
            "native benchmark omitted a model, view, or repetition")
    return result


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "launch":
        interpreter, package_dir, runner_sha = sys.argv[2:5]
        entry = Path(package_dir) / "score-native.py"
        require(sha(entry) == runner_sha, "portable inference entry differs from its scientific seal")
        os.execv(interpreter, [interpreter, "-I", str(entry), *sys.argv[5:]])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--shelf", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True, help="clean CPU environment containing the optional vision dependencies")
    parser.add_argument("--model", required=True, help="selected formal training RUN_ID=MANIFEST_SHA256")
    parser.add_argument("--selection", required=True, help="learning-rate or primary selection RUN_ID=MANIFEST_SHA256")
    print(json.dumps(record(parser.parse_args())))
