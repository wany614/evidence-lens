"""Recorded workers for preparing P4 samples, grouped model fitting and P5 threshold selection."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import tempfile


@contextmanager
def recorded_dataset(args):
    """Reunite independently parent-bound recorder inputs in the split reader's named layout."""
    from evidence_lens.l1_data import FAMILY, check
    from evidence_lens.scoring_io import TARGETS
    if args.dataset is not None:
        check(not args.index and not args.catalog and not args.extract, "use one dataset layout")
        yield args.dataset
        return
    check(args.index and args.catalog and args.extract, "a split dataset or all recorded split inputs are required")
    names = [entry["scene_family_id"] for entry in json.loads(args.index.read_bytes())["families"]]
    pairs = [entry.split("=", 1) for entry in args.extract]
    check(all(len(pair) == 2 and FAMILY.fullmatch(pair[0]) for pair in pairs)
          and len(pairs) == len(names) == len(set(names)) and {k for k, _ in pairs} == set(names), "invalid recorder family bindings")
    with tempfile.TemporaryDirectory(prefix="el-learning-inputs-") as tmp:
        stage = Path(tmp)
        links = {stage / "dataset.json": args.index, stage / "catalog.json": args.catalog}
        for name, source in pairs:
            dest = stage / name
            dest.mkdir()
            for filename in ("family.json", *(v + ".npz" for v in TARGETS)):
                links[dest / filename] = Path(source) / filename
        for dest, src in links.items():
            check(not src.is_symlink() and src.is_file(), f"missing or symlinked split input {src.name}")
            try:
                os.link(src, dest)
            except OSError:
                shutil.copyfile(src, dest)
        yield stage


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    prepare = sub.add_parser("prepare", help="prepare one train or development family from pinned score and label payloads")
    for key in ("family", "catalog", "scores", "evaluation", "output"):
        prepare.add_argument("--" + key, required=True, type=Path)
    for stage in ("fit", "freeze"):
        cmd = sub.add_parser(stage)
        layouts = cmd.add_mutually_exclusive_group(required=True)
        layouts.add_argument("--dataset", type=Path)
        layouts.add_argument("--index", type=Path, help="recorded split index with separately bound family payloads")
        cmd.add_argument("--catalog", type=Path)
        cmd.add_argument("--extract", action="append", metavar="FAMILY=PAYLOAD")
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--run-id", default=os.environ.get("EL_RUN_ID"))
        if stage == "freeze":
            cmd.add_argument("--fit", type=Path, required=True, help="fitted model directory")
    a = p.parse_args(argv)
    if a.stage == "prepare":
        from evidence_lens.l1_data import prepare as prepare_family
        prepare_family(a.family, a.scores, a.evaluation, a.output, a.catalog)
        return 0
    with recorded_dataset(a) as dataset:
        if a.stage == "fit":
            from evidence_lens.l1_fit import fit
            result = fit(dataset, a.output, a.run_id)
        else:
            from evidence_lens.l1_freeze import freeze
            result = freeze(dataset, a.fit, a.output, a.run_id)
    return 2 if result["status"] == "insufficient_data" else 0


if __name__ == "__main__":
    raise SystemExit(main())
