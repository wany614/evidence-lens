"""Run a pinned list of quality or timing condition jobs in the existing Windows worker."""

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from run_windows_batch import RUN_ID, checked, deliver, digest, extract


def stage(args, job, handoff, mode, producer=None):
    flags = []
    if mode == "generate":
        flags = ["--weights", str(args.weights)]
    elif mode == "render":
        flags = ["--producer-run", producer]
    cmd = [
        sys.executable,
        str(handoff / "code/scripts/condition_stage.py"),
        mode,
        "--handoff",
        str(handoff),
        "--store",
        str(args.store),
        "--return-dir",
        str(args.outgoing),
        *flags,
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(os.environ, PYTHONUTF8="1"),
    )
    try:
        r = json.loads(res.stdout.strip().splitlines()[-1])
        kind = {"generate": "generation", "derive": "derivation", "render": "render"}[
            mode
        ]
        if (
            r["schema"] != "el-worker-return-1"
            or r["trial_id"] != job["trial_id"]
            or r["run_kind"] != kind
            or r["handoff_sha256"] != job["handoff_sha256"]
            or not RUN_ID.fullmatch(r["run"])
            or r["package"] != r["run"] + ".zip"
            or res.returncode != r["exit_code"]
        ):
            raise ValueError("condition worker receipt differs from dispatch")
        if (
            producer
            and producer
            not in json.loads(
                (args.store / "runs" / r["run"] / "run.json").read_text(encoding="utf8")
            )["parent_run_ids"]
        ):
            raise ValueError("condition render belongs to a different producer")
        receipt_file = args.outgoing / (r["run"] + ".receipt.json")
        if json.loads(receipt_file.read_text(encoding="utf-8")) != r:
            raise ValueError("durable return metadata changed")
        product = args.outgoing / r["package"]
        checked(product, r["bytes"], r["sha256"])
        for path in (product, receipt_file):
            deliver(path, args.publish / path.name)
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in (
                        "trial_id",
                        "run_kind",
                        "run",
                        "status",
                        "bytes",
                        "sha256",
                    )
                }
            ),
            flush=True,
        )
        return r
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        args.outgoing.mkdir(parents=True, exist_ok=True)
        path = args.outgoing / ("controller-" + uuid.uuid4().hex + ".json")
        with path.open("x", encoding="utf8") as f:
            json.dump(
                {
                    "trial": job["trial_id"],
                    "handoff": job["handoff_sha256"],
                    "stage": mode,
                    "exit_code": res.returncode,
                    "message": str(error),
                    "stdout_tail": res.stdout[-3000:],
                    "stderr_tail": res.stderr[-3000:],
                },
                f,
            )
        deliver(path, args.publish / path.name)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("batch", "staged", "store", "outgoing", "weights", "publish"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--sha256", required=True)
    a = p.parse_args()
    if os.name != "nt" or digest(a.batch) != a.sha256:
        p.error("the Windows worker and exact batch digest are required")
    if (
        len(
            {
                x.resolve().drive.casefold()
                for x in (
                    a.staged,
                    a.store,
                    a.outgoing,
                    a.weights,
                    Path(sys.executable),
                )
            }
        )
        != 1
    ):
        p.error("worker files and interpreter must be on the project volume")
    batch = json.loads(a.batch.read_text(encoding="utf8"))
    if (
        batch["schema"] != "el-windows-condition-batch-1"
        or batch["runner_sha256"] != digest(Path(__file__))
        or batch["transport_sha256"]
        != digest(Path(__file__).with_name("run_windows_batch.py"))
    ):
        raise ValueError("condition runner version differs")
    if len({j["trial_id"] for j in batch["jobs"]}) != len(batch["jobs"]):
        raise ValueError("duplicate condition jobs")
    failed = []
    for job in batch["jobs"]:
        h = extract(job, a.batch.parent, a.staged)
        spec = json.loads((h / "handoff.json").read_text(encoding="utf8"))
        primary = "derive" if spec["condition"] == "Q1" else "generate"
        if job["primary"] != primary or job["timing_repeat"] != spec["timing_repeat"]:
            raise ValueError("condition operation differs from the scientific handoff")
        r = stage(a, job, h, primary)
        if r["exit_code"] == 0:
            r = stage(a, job, h, "render", r["run"])
        if r["exit_code"] != 0:
            failed.append(r["run"])
    if failed:
        raise SystemExit("sealed failed stages: " + ", ".join(failed))


if __name__ == "__main__":
    main()
