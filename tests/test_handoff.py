"""Catch missing internal imports in the actual restricted Windows source list."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_worker_sources_isolated(tmp_path):
    root = Path(__file__).resolve().parents[1]
    names = json.loads((root/"configs/windows-worker-files.json").read_text())["files"]
    assert not any("blender_reference" in n or n == "configs/reference.json" for n in names)
    for name in names:
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, target)
    for name in ("recorder", "archive", "generator_adapter", "renderer_adapter"):
        subprocess.run([sys.executable, "-B", "-m", "evidence_lens."+name, "--help"], cwd=tmp_path,
                       env={**os.environ,"PYTHONPATH":str(tmp_path/"src")}, check=True, capture_output=True)
