"""Run a recorded source snapshot with the four separately frozen wheel JSON assets."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    args = p.parse_args()
    snapshot = Path(__file__).resolve().parents[1]
    files = {"profile.json", "MS-L1/model.json", "MS-L1/preprocessing.json", "MS-L1/coefficients.json"}
    if {f.relative_to(args.assets).as_posix() for f in args.assets.rglob("*") if f.is_file()} != files:
        raise ValueError("expected the four original wheel JSON assets")
    with tempfile.TemporaryDirectory() as work:
        src = Path(work) / "src"
        shutil.copytree(snapshot / "src", src)
        assets = src / "evidence_lens_bundle/assets"
        assets.parent.chmod(0o755)
        shutil.copytree(args.assets, assets, dirs_exist_ok=True)
        env = {**os.environ, "PYTHONPATH": str(src), "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run(
            [args.python, "-B", "-m", "evidence_lens_bundle.bridge", *args.arguments], env=env
        )
        sys.exit(result.returncode)
