"""Hash the comparison protocol, stage explicit inputs or assemble the four-family report."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from evidence_lens.formal_inputs import (
    PinnedRun,
    protocol,
    selection_from_quality,
    stage,
)
from evidence_lens.formal_report import assemble, build
from evidence_lens.formal_roster import roster
from evidence_lens.scoring_io import write_json

ROOT = Path(__file__).resolve().parents[1]


def record_trial(args):
    """Give each direct parent a recorder pin and bind selected receipt bytes to that parent."""
    store = args.store.resolve()
    with tempfile.TemporaryDirectory(prefix="stage-comparison-") as temp:
        tmp = Path(temp)
        bundle = tmp / "comparison"
        info = stage(store, args.selection, bundle, ROOT)
        cfg = {
            k: info[k]
            for k in (
                "scene_id",
                "scene_family_id",
                "condition_id",
                "trial_id",
                "dataset_id",
                "dataset_hash",
                "method_freeze_sha256",
                "comparison_protocol_sha256",
                "model_id",
            )
        }
        cfg.update(
            split="test",
            metric_version="el-formal-comparison-1",
            recording={"expected_output_bytes": 200_000_000},
        )
        write_json(tmp / "config.json", cfg)
        cmd = [
            sys.executable,
            "-m",
            "evidence_lens.recorder",
            "--root",
            str(store),
            "--source-root",
            str(ROOT),
            "--experiment",
            "p6-heldout-comparison",
            "--trial",
            info["trial_id"],
            "--kind",
            "evaluation",
            "--config",
            str(tmp / "config.json"),
            "--input",
            f"comparison={bundle}",
        ]
        if args.retry_of:
            cmd += ["--retry-of", args.retry_of]
        if args.storage_policy:
            cmd += ["--storage-policy", str(args.storage_policy.resolve())]
        used = set()
        for role, pin in info["parents"].items():
            if pin["run_id"] not in used:
                cmd += ["--parent", pin["run_id"] + "=" + pin["manifest_sha256"]]
                used.add(pin["run_id"])
            receipts = [
                (rel, entry)
                for rel, entry in info["files"].items()
                if entry.get("parent_role") == role
                and entry.get("parent_file", "").startswith("payload/")
                and rel.endswith(".json")
            ]
            if receipts:
                # The worker reads the complete staged comparison; the extra tiny receipt explicitly
                # identifies which sealed parent in the recorder supplied each role.
                rel, _ = min(receipts, key=lambda pair: pair[1]["bytes"])
                cmd += [
                    "--input",
                    f"receipt-{role}={bundle / rel}",
                    "--input-parent",
                    "receipt-" + role + "=" + pin["run_id"],
                ]
            elif role in ("generation", "readback"):
                native = (
                    "payload/generation.json" if role == "generation" else "payload/reference_readback.json"
                )
                run = PinnedRun(store, pin)
                cmd += [
                    "--input",
                    f"receipt-{role}={run.open(native)}",
                    "--input-parent",
                    "receipt-" + role + "=" + pin["run_id"],
                ]
        cmd += [
            "--",
            "{python}",
            "-m",
            "evidence_lens.formal_evaluation",
            "--bundle",
            "{input:comparison}",
            "--output",
            "{out}",
            "--run-id",
            "{run_id}",
        ]
        return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_subparsers(dest="command", required=True)
    actions.add_parser("protocol", help="Print the versioned comparison source map and its canonical SHA-256")
    for name, help_text in (
        ("stage", "Stage one explicitly selected test family and condition"),
        ("assemble", "Stage 12 explicitly pinned P6 evaluation runs and selected engineering evidence"),
    ):
        cmd = actions.add_parser(name, help=help_text)
        cmd.add_argument("--store", type=Path, required=True)
        cmd.add_argument("--selection", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
    cmd = actions.add_parser(
        "select",
        help="Create the comparison selection from a fixed canonical quality selection and the deployed L1 run",
    )
    cmd.add_argument("--quality", type=Path, required=True)
    cmd.add_argument("--learned", required=True, help="Selected L1 run_id=manifest_sha256")
    cmd.add_argument("--method-freeze", type=Path, required=True)
    cmd.add_argument("--catalog", type=Path, default=ROOT / "protocol/families-v1.json")
    cmd.add_argument("--study", type=Path, default=ROOT / "protocol/study-v1.json")
    cmd.add_argument("--paired-quality", type=Path)
    cmd.add_argument("--paired-learned")
    cmd.add_argument("--out", type=Path, required=True)
    cmd = actions.add_parser(
        "record", help="Stage and run one P6 comparison using the recorder, with explicit named parents"
    )
    cmd.add_argument("--store", type=Path, required=True)
    cmd.add_argument("--selection", type=Path, required=True)
    cmd.add_argument("--retry-of")
    cmd.add_argument("--storage-policy", type=Path)
    cmd = actions.add_parser("build", help="Construct the final private report from a staged manifest")
    cmd.add_argument("--bundle", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True)
    cmd.add_argument("--run-id", required=True)
    cmd = actions.add_parser(
        "roster", help="Expand explicit scientific, timing and evidence pins into a full report selection"
    )
    cmd.add_argument("--store", type=Path, required=True)
    cmd.add_argument("--roster", type=Path, required=True)
    cmd.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.command == "roster":
        value = roster(a.store, a.roster, a.out)
        print(
            json.dumps(
                {"trials": len(value["trials"]), "runs": len(value["evidence"]), "proofs": len(value["proofs"])}
            )
        )
        return
    if a.command == "protocol":
        print(json.dumps(protocol(ROOT), indent=2))
        return
    if a.command == "select":
        selection_from_quality(
            a.quality, a.learned, a.method_freeze, a.catalog, a.study, a.out, a.paired_quality, a.paired_learned
        )
        return
    if a.command == "record":
        raise SystemExit(record_trial(a))
    if a.command == "stage":
        r = stage(a.store, a.selection, a.out, ROOT)
        print(
            json.dumps(
                {
                    "trial_id": r["trial_id"],
                    "parents": r["parents"],
                    "comparison_protocol_sha256": r["comparison_protocol_sha256"],
                }
            )
        )
    elif a.command == "assemble":
        r = assemble(a.store, a.selection, a.out, ROOT)
        print(json.dumps({"quality_trials": len(r["trials"]), "inventory_runs": len(r["evidence"])}))
    else:
        r = build(a.bundle, a.output, a.run_id)
        print(
            json.dumps({k: r[k] for k in ("schema", "run_id", "kind", "quality_trials", "completed_targets")})
        )


if __name__ == "__main__":
    main()
