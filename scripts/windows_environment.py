"""Capture P0 Windows capabilities before introducing model or toolchain dependencies."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    if os.name != "nt" or not args.project.resolve().drive:
        raise SystemExit("Run in the target Windows project.")
    args.out.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ["torch", "torchvision", "gsplat", "sharp", "psutil", "pyarrow", "numpy", "pillow"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    report = {"platform": platform.platform(), "python": sys.version, "host": platform.node(),
              "drive": args.project.resolve().drive, "disk_free_bytes": shutil.disk_usage(args.project).free,
              "packages": packages, "commands": {}}
    for name, command in {"nvidia-smi": ["nvidia-smi", "-q"], "nvcc": ["nvcc", "--version"],
                          "cl": ["cl.exe"], "ninja": ["ninja", "--version"], "git": ["git", "--version"]}.items():
        binary = shutil.which(command[0])
        if binary:
            result = subprocess.run([binary, *command[1:]], capture_output=True, timeout=60)
            (args.out / f"{name}.txt").write_bytes(result.stdout + result.stderr)
            report["commands"][name] = {"available": True, "exit_code": result.returncode}
        else:
            report["commands"][name] = {"available": False}
    (args.out/"environment.windows.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
