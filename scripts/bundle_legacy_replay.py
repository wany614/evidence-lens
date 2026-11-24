"""Independent process running the unmodified numerical modules from the archived CPU release."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def replay(cpu, cases, out):
    index = "150b04e702f5386b5808ac17545cd073e3d4d837ea789e6ae167168ac086084d"
    entry = "79c0f2436f08e943cbef9d81b6bf1bad83ddaacb8cfaceb776fab11faa8b3de6"
    code = cpu / "cpu.py"
    if hashlib.sha256(code.read_bytes()).hexdigest() != entry:
        raise ValueError("wrong old CLI")
    spec = importlib.util.spec_from_file_location("archived_evidence_lens_cpu", code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = module.Runtime(cpu, index, 4)
    from evidence_lens.multisource_io import Case

    for c in ("M2", "M4"):
        case = Case(cases / c, hashlib.sha256((cases / c / "case.json").read_bytes()).hexdigest())
        for method in runtime.specs:
            scope = "one" if method == "B-L1-transfer" else "all"
            for t in case.pred["target_ids"]:
                data = runtime.score(method, case, t, scope)
                where = out / c / method / (t + ".npz")
                where.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(where, **data)
    sources = {}
    for name, m in sys.modules.items():
        if name.startswith(
            ("evidence_lens.", "evidence_lens_multirisk.", "evidence_lens_vision.")
        ) and getattr(m, "__file__", None):
            p = Path(m.__file__).resolve()
            if not p.is_relative_to(cpu / "models/MS-frozen/runtime"):
                raise ValueError("mixed installed/released numerical module: " + name)
            sources[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out / "runtime.json").write_text(
        json.dumps({"entry_sha256": entry, "module_sha256": sources}, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("cpu", "cases", "out"):
        p.add_argument("--" + n, required=True, type=Path)
    args = p.parse_args()
    replay(args.cpu.resolve(), args.cases, args.out)
