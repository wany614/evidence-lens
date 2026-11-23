"""Fingerprint the installed CUDA wheel, API and license without starting an experiment."""
import argparse
import ast
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import platform

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def inventory():
    dist = md.distribution("gsplat")
    if dist.version != "1.5.3":
        raise ValueError("Expected the pinned gsplat 1.5.3 distribution")
    root = Path(dist.locate_file("gsplat"))
    names = ["strategy/__init__.py", "strategy/base.py", "strategy/default.py", "strategy/ops.py", "rendering.py",
             "cuda/_wrapper.py", "csrc.pyd"]
    out = {n: {"bytes": (root / n).stat().st_size, "sha256": digest(root / n)} for n in names}
    lic = next(Path(dist.locate_file(p)) for p in dist.files if str(p).replace("\\", "/").endswith(".dist-info/licenses/LICENSE"))
    api = {}
    for n in ("strategy/ops.py", "strategy/default.py", "rendering.py"):
        for fn in ast.walk(ast.parse((root / n).read_bytes())):
            if isinstance(fn, ast.FunctionDef) and fn.name in {"rasterization", "duplicate", "remove", "split",
                    "reset_opa", "initialize_state", "step_pre_backward", "_update_state"}:
                api[fn.name] = {"module": n, "line": fn.lineno, "arguments": [a.arg for a in fn.args.args]}
    required = {"rasterization", "duplicate", "remove", "split", "reset_opa", "initialize_state", "step_pre_backward", "_update_state"}
    if set(api) != required:
        raise ValueError("Installed projection or strategy API differs")
    return {"schema": "el-recovery-installed-api-1", "python": platform.python_version(), "torch": md.version("torch"),
            "numpy": md.version("numpy"), "gsplat": dist.version, "installed_files": out, "api": api,
            "license": {"upstream": "https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/LICENSE",
                        "type": "Apache-2.0", "bytes": lic.stat().st_size, "sha256": digest(lic)}}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    body = json.dumps(inventory(), indent=2, sort_keys=True) + "\n"
    if a.output:
        with a.output.open("x", encoding="utf8", newline="\n") as f:
            f.write(body)
    else:
        print(body)
