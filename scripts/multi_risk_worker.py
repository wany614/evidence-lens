"""Train, validate or run the registered stage C multi-source CPU model."""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    cmd = p.add_subparsers(dest="action", required=True)

    def paths(q, *names):
        for n in names:
            q.add_argument("--" + n, type=Path, required=True)

    def strings(q, *names):
        for n in names:
            q.add_argument("--" + n, required=True)

    a = cmd.add_parser("inspect-init")
    paths(a, "B-index", "B-weights", "feature-schema", "output")
    a.add_argument("--seed", type=int, default=1729)
    a = cmd.add_parser("train")
    paths(
        a,
        "B-index",
        "B-weights",
        "feature-schema",
        "protocol",
        "catalog",
        "authority",
        "config",
        "train",
        "development",
        "preprocessing",
        "output",
    )
    strings(a, "config-sha256", "run-id")
    a.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a.add_argument("--threads", type=int, default=4)
    a.add_argument("--resume", type=Path)
    a.add_argument("--resume-sha256")
    a = cmd.add_parser("verify-updates")
    paths(a, "B-index", "B-weights", "training", "output")
    strings(a, "training-sha256")
    for key in ("score-native", "_warm", "cost"):
        a = cmd.add_parser(key)
        paths(a, "model", "case", "output")
        strings(a, "model-sha256", "case-sha256")
        a.add_argument("--threads", type=int, default=4)
        if key == "score-native":
            a.add_argument("--view", required=True, choices=[f"v{i:02d}" for i in range(1, 7)])
            a.add_argument("--scope", required=True, choices=("one", "two", "all"))
        else:
            a.add_argument(
                "--scopes", nargs="+", choices=("one", "two", "all"), default=["one", "two", "all"]
            )
            a.add_argument("--repetitions", type=int, default=1)
    a = cmd.add_parser("infer-dataset")
    paths(a, "dataset", "catalog", "model", "output")
    strings(a, "dataset-sha256", "model-sha256")
    a.add_argument("--split", choices=("development", "test"), required=True)
    a.add_argument("--method", choices=("MS-frozen", "MS-ft"), required=True)
    a.add_argument("--store", type=Path)
    a.add_argument("--shelf", type=Path)
    a.add_argument("--opening-sha256")
    a.add_argument("--threads", type=int, default=4)
    return p


def main(a):
    from evidence_lens_vision.contract import output_dir, put, require

    if hasattr(a, "threads"):
        require(1 <= a.threads <= 32, "threads must be in [1,32]")
    if a.action == "train":
        from evidence_lens_multirisk.train import run_fit

        return run_fit(a)
    if a.action == "inspect-init":
        import torch
        from evidence_lens_vision.train import hashes
        from evidence_lens_multirisk.data import signed
        from evidence_lens_multirisk.train import SCHEMA
        from evidence_lens_multirisk.model import METHODS, initialize

        torch.set_num_threads(4)
        cols = signed(a.feature_schema, SCHEMA)["columns"]
        out = output_dir(a.output)
        result = {}
        initial = None
        for method in METHODS:
            m, receipt = initialize(a.B_index, a.B_weights, cols, a.seed, method)
            before = hashes(m)
            require(initial is None or before == initial, "paired branches start with different parameters")
            initial = before
            result[method] = receipt
        record = {
            "schema": "el-paired-multisource-initialization-1",
            "seed": a.seed,
            "all_tensor_hashes_equal": True,
            "initial_tensors": initial,
            "methods": result,
        }
        put(out / "paired-initialization.json", record)
        return {"paired_initialization": True, "seed": a.seed, "tensors": len(initial)}
    if a.action == "verify-updates":
        from evidence_lens_multirisk.proof import verify

        return verify(a.B_index, a.B_weights, a.training, a.output, a.training_sha256)
    if a.action == "score-native":
        from evidence_lens_multirisk.deploy import score_native

        return score_native(
            a.model, a.model_sha256, a.case, a.case_sha256, a.view, a.scope, a.output, a.threads
        )
    from evidence_lens_multirisk.inference import batch, costs, warm

    if a.action == "infer-dataset":
        return batch(a)
    if a.action == "_warm":
        return warm(a)
    return costs(a, HERE)


if __name__ == "__main__":
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11+ is required for multi-source risk scoring.")
    print(json.dumps(main(parser().parse_args()), sort_keys=True, allow_nan=False))
