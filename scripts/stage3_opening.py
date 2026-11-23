"""Open the four registered multisource test families after recording the selected method."""

import argparse
import json
from pathlib import Path

from evidence_lens.stage3_gate import open_cohort, require_opening


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "open"))
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--shelf", type=Path)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--freeze-run")
    parser.add_argument("--freeze-manifest")
    args = parser.parse_args()
    if args.action == "open":
        if not (args.freeze_run and args.freeze_manifest):
            parser.error("Opening requires the successful freeze run and its SHA-256")
        outcome = open_cohort(args.store, args.catalog_sha256, args.freeze_run, args.freeze_manifest, args.shelf)
    else:
        outcome = require_opening(args.store, args.catalog_sha256, shelf=args.shelf)
    print(json.dumps(outcome, sort_keys=True))
