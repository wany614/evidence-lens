"""Evaluate a held-out condition using separately parented target arrays and source-bound reference metadata."""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from evidence_lens.evaluation import run_evaluation
from evidence_lens.scoring_io import TARGETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "cameras", "predictions", "target-reference", "reference-metadata", "scores", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("trial-id", "scene-id", "scene-family-id", "condition-id"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    # Each mounted logical input was independently frozen and parent-bound by the recorder.
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix="condition-reference-") as tmp:
        directory = Path(tmp)
        sources = {"reference.json":args.reference_metadata, "validation/results.json":args.target_reference / "validation/results.json"}
        sources.update({f"reference/{v}.npz":args.target_reference / f"reference/{v}.npz" for v in TARGETS})
        for name, source in sources.items():
            dest = directory / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, dest)
            except OSError:
                shutil.copyfile(source, dest)
        identity = {k:getattr(args,k) for k in ("trial_id", "scene_id", "scene_family_id", "condition_id")}
        identity["run_id"] = os.environ["EL_RUN_ID"]
        run_evaluation(args.input, args.cameras, directory, args.predictions, args.scores, args.output, identity, parquet=True)


if __name__ == "__main__":
    main()
