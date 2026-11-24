"""Isolated reconstruction of v0.3.0-rc.1 feature reasons from the original CPU distribution."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

RELEASE = "150b04e702f5386b5808ac17545cd073e3d4d837ea789e6ae167168ac086084d"


def run(root, case_dir, case_sha, out):
    import numpy as np

    def digest(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    release = root / "release.json"
    entry = root / "cpu.py"
    if (
        digest(release) != RELEASE
        or digest(entry) != json.loads(release.read_bytes())["files"]["cpu.py"]["sha256"]
    ):
        raise ValueError("unverified archived CPU runtime")
    if any(k.startswith("evidence_lens") for k in sys.modules):
        raise ValueError("frozen reason worker needs a separate interpreter")
    loader = importlib.util.spec_from_file_location("original_cpu_release", entry)
    api = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(api)
    api.Runtime(
        root, RELEASE, 1
    )  # authenticates old scientific import tree, including the full feature kernel
    from evidence_lens.multisource_io import Case
    from evidence_lens.scoring_io import alpha_valid, depth_valid, rgb_valid

    case = Case(case_dir, case_sha)
    out.mkdir()
    for view in case.meta["target_ids"]:
        frame = case.frame(view)
        if frame is None:
            b = ms = np.full(case.cameras[view].shape, 7, np.uint8)
        else:
            b = np.select(
                [
                    np.full(case.cameras[view].shape, not case.pred["depth_verified"]),
                    ~rgb_valid(frame["rgb"]),
                    ~alpha_valid(frame["alpha"]),
                    ~depth_valid(frame["z"]),
                ],
                [1, 2, 3, 4],
                default=0,
            ).astype(np.uint8)
            _, features = case.features(view, "all")
            ms = features["unknown_reason"]
        np.savez(out / (view + ".npz"), primary=b, multisource=ms)
    package = (root / "models/MS-frozen/runtime").resolve()
    origins = {}
    for name, module in sorted(sys.modules.items()):
        if name.startswith("evidence_lens") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(package):
                raise ValueError("historical reason module was imported from another installation")
            origins[name] = digest(path)
    (out / "origin.json").write_text(json.dumps({"release_sha256": RELEASE, "modules": origins}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "case", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case-sha256", required=True)
    args = parser.parse_args()
    run(args.models.resolve(), args.case, args.case_sha256, args.out)
