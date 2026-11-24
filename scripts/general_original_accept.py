"""Grade a fresh installed-wheel CLI replay against the four-method original development archive."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from evidence_lens.scoring_io import new_output
from evidence_lens_bundle import cli
from evidence_lens_bundle.bridge import environment
from evidence_lens_bundle.common import member, read_file, require, sha, write
from evidence_lens_bundle.numerics import METHODS, compare_arrays, matched_sets
from evidence_lens_bundle.report import read_arrays


def command(*args):
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-m", "evidence_lens_bundle.cli", *map(str, args)],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"installed CLI {args[0]} exited {result.returncode}: {result.stderr}")
    return json.loads(result.stdout)


def get_arrays(folder, receipt, view_index):
    view = receipt["views"][view_index]
    return read_arrays(
        member(folder, view["arrays"], expected=f"views/{view_index:03d}/scores.npz"),
        receipt["model"]["threshold"],
    )


def accept(root, sequence, wanted, output, models, threads, expected_sha=None):
    installed = importlib.metadata.distribution("evidence-lens")
    require(
        Path(installed.locate_file("evidence_lens_bundle/cli.py")).resolve() == Path(cli.__file__).resolve(),
        "the installed evidence-lens wheel must supply the CLI",
    )
    source = read_file(root, "expected.json")
    if expected_sha is not None:
        require(sha(source) == expected_sha, "expected.json digest differs from the release")
    index = json.loads(source)
    require(
        index["schema"] == "el-original-replay-gold-1" and sequence in index["sequences"],
        "unregistered original replay sequence",
    )
    intent = tuple(METHODS) if wanted == "all" else (wanted,)
    require(all(m in METHODS for m in intent), "unregistered risk method")
    require(models is not None or intent == ("MS-L1",), "supply --models for the external frozen weights")
    gold = index["sequences"][sequence]
    base = root / "inputs" / sequence
    records = {}
    expected_dirs = {}
    for m in METHODS:
        expected_dirs[m] = root / "expected" / sequence / m / "scores"
        data = read_file(expected_dirs[m], "scores.json")
        require(sha(data) == gold["methods"][m]["scores_sha256"], "expected method receipt differs: " + m)
        records[m] = json.loads(data)
        require(
            records[m]["method"] == m
            and records[m]["target_order"] == gold["target_order"]
            and records[m]["bundle_sha256"] == gold["bundle_sha256"],
            "gold identity mismatch",
        )
    observations = {}
    with new_output(output) as dest:
        measured = {}
        for m in intent:
            folder = dest / m
            folder.mkdir()
            extra = ["--models", models] if models is not None else []
            start = perf_counter()
            command(
                "score",
                "--bundle",
                base,
                "--bundle-sha256",
                gold["bundle_sha256"],
                "--model",
                m,
                "--device",
                "cpu",
                "--threads",
                threads,
                *extra,
                "--out",
                folder / "scores",
            )
            score_wall = perf_counter() - start
            raw = read_file(folder / "scores", "scores.json")
            r = measured[m] = json.loads(raw)
            require(
                r["method"] == m
                and r["model"] == records[m]["model"]
                and r["bundle_sha256"] == gold["bundle_sha256"]
                and r["target_order"] == gold["target_order"]
                and r["scorer_source_ids"] == records[m]["scorer_source_ids"],
                "installed score used a different evidence, source scope or model",
            )
            start = perf_counter()
            command(
                "report",
                "--scores",
                folder / "scores",
                "--scores-sha256",
                sha(raw),
                "--out",
                folder / "report.html",
            )
            html = (folder / "report.html").read_bytes()
            require(
                b"data:image/png;base64," in html and b"CUDA" in html and b"CPU" in html,
                "installed HTML is missing its panels or rendering boundary",
            )
            observations[m] = {
                "scores_sha256": sha(raw),
                "html_sha256": sha(html),
                "html": m + "/report.html",
                "counts": r["counts"],
                "score_process_seconds": score_wall,
                "report_process_seconds": perf_counter() - start,
                "views": {},
            }
        for i, view in enumerate(gold["target_order"]):
            expected = {m: get_arrays(expected_dirs[m], records[m], i) for m in METHODS}
            actual = {
                m: get_arrays(dest / m / "scores", measured[m], i) if m in intent else expected[m]
                for m in METHODS
            }
            sets = matched_sets(
                expected,
                actual,
                gold["object_id"],
                gold["condition"],
                view,
                records["MS-L1"]["model"]["threshold"],
            )
            for m in intent:
                observations[m]["views"][view] = {
                    "difference": compare_arrays(expected[m], actual[m], records[m]["model"]["threshold"]),
                    "matched_coverage": {
                        **{k: v for k, v in sets.items() if k != "methods"},
                        "anchors": sets["methods"][m],
                    },
                }
        exact = all(v["difference"]["bit_exact"] for o in observations.values() for v in o["views"].values())
        result = {
            "schema": "el-original-replay-grade-1",
            "sequence": sequence,
            "recorded_methods": list(intent),
            "common_domain_companions_from_gold": [m for m in METHODS if m not in intent],
            "status": "bit_exact" if exact else "numerical_difference",
            "bit_exact": exact,
            "gold_run": index["bridge_run_id"],
            "gold_index_sha256": sha(source),
            "bundle_sha256": gold["bundle_sha256"],
            "installed_distribution": installed.version,
            "installed_cli_sha256": hashlib.sha256(Path(cli.__file__).read_bytes()).hexdigest(),
            "observed_environment": environment(),
            "threads_argument": threads,
            "output": observations,
        }
        write(dest / "acceptance.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--sequence", default="bell-M4", choices=("bell-M2", "bell-M4", "beads-M2", "beads-M4"))
    p.add_argument("--method", default="MS-L1", choices=(*METHODS, "all"))
    p.add_argument("--models", type=Path)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-sha256")
    p.add_argument("--require-bit-exact", action="store_true")
    a = p.parse_args()
    r = accept(a.root.resolve(), a.sequence, a.method, a.out, a.models, a.threads, a.expected_sha256)
    print(
        json.dumps(
            {
                "status": r["status"],
                "methods": r["recorded_methods"],
                "receipt": str(a.out / "acceptance.json"),
            }
        )
    )
    sys.exit(2 if a.require_bit_exact and not r["bit_exact"] else 0)
