"""Score the registered native plane using the deployment inputs and three frozen CPU packages."""
import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_public_smoke import METHODS, compute, control_inputs, control_result, put, read, require, sha


def score(args):
    selection = read(args.selection)
    require(sha(args.selection) == args.selection_sha256 and
            selection["schema"] == "el-public-control-deployment-selection-1",
            "the deployment selection differs from its external pin")
    models = {"L1": args.lr, "L2-frozen": args.frozen, "L2-vision-ft": args.adapted}
    pins = selection["models"]
    thresholds = selection["thresholds"]
    require(set(pins) == set(models) and set(thresholds) == set(METHODS) and
            all(math.isfinite(t["tau"]) and 0 <= t["tau"] <= 1 for t in thresholds.values()),
            "incomplete frozen deployment method")
    # Verify the bytes of every shipped inference module before importing that package.
    for name, folder in models.items():
        index = folder / "model.json"
        require(sha(index) == pins[name], "unrecognized model index for " + name)
        members = read(index)["files"]
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() or p.is_symlink()}
        require(actual == set(members) | {"model.json"}, "undeclared model members for " + name)
        for rel, meta in members.items():
            candidate = folder / rel
            digest = meta if name == "L1" else meta["sha256"]
            require(candidate.resolve().is_relative_to(folder.resolve()) and not candidate.is_symlink() and
                    candidate.is_file() and (name == "L1" or candidate.stat().st_size == meta["bytes"])
                    and sha(candidate) == digest,
                    "unexpected model member for " + name + "/" + rel)
    out = args.output
    out.mkdir(exist_ok=True, parents=True)
    require(not any(out.iterdir()), "use an empty score destination")
    with tempfile.TemporaryDirectory() as work:
        isolated = Path(work)
        case = control_inputs(args.case, args.case_sha256, isolated, read(args.registration))
        runtime = compute(isolated, models, pins, out / "scores", require_support=False)
    summary = control_result(out / "scores", out, thresholds)
    summary.update(schema="el-vision-public-control-scores-1", run_id=os.environ["EL_RUN_ID"],
                   status="scored", case_sha256=args.case_sha256, registration_sha256=sha(args.registration),
                   family=case["family"], condition=case["condition"], generator=case["generator"],
                   inputs=case["files"], models=pins, method=selection["method"], runtime=runtime)
    put(out / "control-scores.json", summary)
    print(json.dumps({"run_id": summary["run_id"], "views": len(summary["views"]),
                      "maps": len(summary["map_files"]), "scores_sha256": sha(out / "control-scores.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "registration", "selection", "lr", "frozen", "adapted", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("case-sha256", "selection-sha256"):
        parser.add_argument("--" + name, required=True)
    score(parser.parse_args())
