"""Record available Mac and local Linux runtimes plus storage on the selected T0 volume."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import psutil


def probe(argv):
    start = time.monotonic()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=12)
        return {
            "exit": p.returncode,
            "output": (p.stdout + p.stderr).strip()[-1600:],
            "wall_s": time.monotonic() - start,
        }
    except (OSError, subprocess.TimeoutExpired) as ex:
        return {"exit": None, "output": str(ex), "wall_s": time.monotonic() - start}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    args = parser.parse_args()
    cmd = shutil.which("docker")
    image = (
        probe([cmd, "image", "ls", "--format", "{{.Repository}}:{{.Tag}} {{.ID}}"])
        if cmd
        else {"exit": None, "output": "docker executable unavailable on PATH"}
    )
    result = {
        "schema": "el-t0-platform-1",
        "utc_run": os.environ["EL_RUN_ID"],
        "mac": platform.mac_ver()[0],
        "machine": platform.machine(),
        "cpu_logical": psutil.cpu_count(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "RAM_bytes": psutil.virtual_memory().total,
        "python": platform.python_version(),
        "external_free_bytes": shutil.disk_usage(args.out).free,
        "internal_free_bytes": shutil.disk_usage(args.checkout).free,
        "linux_image_inventory": image,
        "linux_numerical_acceptance": "separate CLI run required",
    }
    (args.out / "platform.json").write_text(json.dumps(result, indent=2) + "\n")
