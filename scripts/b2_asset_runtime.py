"""Run the research-only scope recorder with the exact CPU wheel's deployed modules and model data."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wheel", type=Path, required=True)
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    with tempfile.TemporaryDirectory() as td, ZipFile(a.wheel) as z:
        for member in z.infolist():
            parts = Path(member.filename).parts
            if len(parts) > 1 and parts[0].startswith("evidence_lens") and ".dist-info" not in parts[0]:
                if Path(member.filename).is_absolute() or ".." in parts or "\\" in member.filename:
                    raise ValueError("unsafe wheel member")
                z.extract(member, td)
        args = a.args[1:] if a.args and a.args[0] == "--" else a.args
        env = {**os.environ, "PYTHONPATH": td, "PYTHONDONTWRITEBYTECODE": "1"}
        proc = subprocess.run([sys.executable, "-B", str(a.script), *args], env=env, check=False)
        raise SystemExit(proc.returncode)
