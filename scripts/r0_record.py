"""Run an explicitly pinned R0 CPU readback or numerical report under the existing recorder."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from evidence_lens._bundle import load
from evidence_lens._verify import require


def main(a):
    project = Path(__file__).resolve().parents[1]
    cfg = load(a.config)
    if cfg.get("schema") == "el-r0-cpu-readback-config-1":
        mode = "cpu"
        parents = [p for arm in cfg["arms"] for p in arm["parents"].values()]
        require(
            [(x["object"], x["condition"]) for x in cfg["arms"]]
            == [(o, c) for o in ("set1", "set6") for c in ("M2", "M4")],
            "Incorrect D1 arm roster",
        )
    elif cfg.get("schema") == "el-r0-evidence-report-config-1":
        mode = "report"
        parents = [cfg["cpu"], cfg["centres"], *cfg["gpu"].values(), *cfg["evaluations"].values()]
    elif cfg.get("schema") == "el-r0-roundoff-readback-config-1":
        mode = "roundoff"
        parents = [cfg["report"], cfg["cpu"], cfg["centres"]]
    else:
        raise ValueError("Expected a pinned R0 configuration")
    unique = {}
    for p in parents:
        require(
            unique.setdefault(p["run_id"], p["manifest_sha256"]) == p["manifest_sha256"],
            "Conflicting run identity",
        )
    store = a.store.resolve()
    shelf = a.shelf.resolve() if a.shelf else store.parent / "cold-shelf"
    cmd = [
        sys.executable,
        "-m",
        "evidence_lens.recorder",
        "--root",
        str(store),
        "--source-root",
        str(project),
        "--kind",
        "report" if mode == "report" else "archive_verify",
        "--config",
        str(a.config.resolve()),
    ]
    for rid, digest in unique.items():
        cmd += ["--parent", rid + "=" + digest]
    cmd += [
        "--",
        "{python}",
        "-B",
        "{source}/scripts/r0_" + mode + ".py",
        "--config",
        "{config}",
        "--store",
        str(store),
        "--shelf",
        str(shelf),
        "--output",
        "{out}",
    ]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--shelf", type=Path)
    sys.exit(main(p.parse_args()))
