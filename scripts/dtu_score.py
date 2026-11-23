"""Score the six declared targets of one source-only case with the released v0.3.0-rc.1 CPU API."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter

RELEASE_SHA = "150b04e702f5386b5808ac17545cd073e3d4d837ea789e6ae167168ac086084d"
METHODS = ("B-L1-transfer", "MS-L1", "MS-frozen", "MS-ft")
SCOPES = {m: "one" if m == "B-L1-transfer" else "all" for m in METHODS}
VIEWS = tuple(f"v{n:02d}" for n in range(1, 7))


def require(test, msg):
    if not test:
        raise ValueError(msg)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publication(folder, expected):
    """Authenticate the distributed entry-point bytes before loading any package implementation."""
    require(
        expected == RELEASE_SHA and digest(folder / "release.json") == expected, "incorrect release index"
    )
    meta = json.loads((folder / "release.json").read_bytes())
    entry = folder / "cpu.py"
    spec = meta["files"]["cpu.py"]
    require(
        entry.is_file()
        and not entry.is_symlink()
        and digest(entry) == spec["sha256"]
        and entry.stat().st_size == spec["bytes"],
        "published CPU entry-point bytes differ",
    )
    require(
        not any(
            n == "evidence_lens"
            or n.startswith(("evidence_lens.", "evidence_lens_multirisk", "evidence_lens_vision"))
            for n in sys.modules
        ),
        "run the scorer in a fresh process before importing research code",
    )
    loader = importlib.util.spec_from_file_location("dtu_original_release_cpu", entry)
    api = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(api)
    return api, meta


def score(a):
    import numpy as np

    started = perf_counter()
    folder, case_dir, out = a.cpu_dir.resolve(), a.case.resolve(), a.output.absolute()
    api, meta = publication(folder, a.release_sha256)
    runtime = api.Runtime(folder, a.release_sha256, a.threads)
    from evidence_lens.multisource_io import Case

    case = Case(case_dir, a.case_sha256)
    require(case.condition in ("M2", "M4") and set(case.meta["target_ids"]) == set(VIEWS), "unexpected case")
    require(case.pred["depth_verified"] is True, "the renderer depth probe has not been validated")
    require(
        not out.exists() or (out.is_dir() and not out.is_symlink() and not any(out.iterdir())),
        "output must be new or an empty recorder payload",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".dtu-scores-", dir=out.parent))
    try:
        for m in METHODS:
            runtime.load(m)
        ready_time = perf_counter() - started
        obs = {}
        for v in VIEWS:
            if v in case.failed:
                obs[v] = {"status": "render_failed", "reason": case.failed[v], "methods": {}}
                continue
            rows = {}
            for m in METHODS:
                scope = SCOPES[m]
                tick = perf_counter()
                arrays = runtime.score(m, case, v, scope)
                target = work / v / m / "scores.npz"
                target.parent.mkdir(parents=True)
                np.savez_compressed(target, **arrays)
                arrays = api.arrays(target, runtime.specs[m]["threshold"])
                valid = arrays["valid"]
                require(
                    np.all((arrays["risk"][valid] >= 0) & (arrays["risk"][valid] <= 1)), "out-of-range risk"
                )
                rows[m] = {
                    "scope": scope,
                    "source_ids": list(("s00",) if scope == "one" else case.generator),
                    "target_prediction_sha256": case.meta["renders"][v]["sha256"],
                    "file": {
                        "path": target.relative_to(work).as_posix(),
                        "sha256": digest(target),
                        "bytes": target.stat().st_size,
                    },
                    "finite": int(valid.sum()),
                    "unknown_grid": int(valid.size - valid.sum()),
                    "score_write_readback_seconds": perf_counter() - tick,
                }
            obs[v] = {"status": "ready", "methods": rows}
        # Any imported numerical study module must originate in the fingerprinted distribution.
        rooted = (folder / "models/MS-frozen/runtime").resolve()
        scientific_sources = {}
        for name, module in sorted(sys.modules.items()):
            if (
                name == "evidence_lens"
                or name.startswith(("evidence_lens.", "evidence_lens_multirisk", "evidence_lens_vision"))
            ) and getattr(module, "__file__", None):
                original = Path(module.__file__).resolve()
                require(
                    original.is_relative_to(rooted), "unexpected non-distributed scientific module: " + name
                )
                scientific_sources[name] = digest(original)
        result = {
            "schema": "el-dtu-frozen-scores-1",
            "run_id": os.environ.get("EL_RUN_ID"),
            "release": "v0.3.0-rc.1",
            "release_sha256": a.release_sha256,
            "release_entry_sha256": meta["files"]["cpu.py"]["sha256"],
            "method_sha256": meta["method_sha256"],
            "models": runtime.specs,
            "method_scopes": SCOPES,
            "case_sha256": case.sha256,
            "condition": case.condition,
            "generator_source_ids": list(case.generator),
            "checkpoint_sha256": case.meta["checkpoint_sha256"],
            "reconstruction_run_id": case.meta["reconstruction_run_id"],
            "source_camera_sha256": case.meta["source_cameras"]["sha256"],
            "render_camera_sha256": case.meta["render_cameras"]["sha256"],
            "threads": a.threads,
            "device": "cpu",
            "loaded_model_seconds": ready_time,
            "worker_wall_seconds": perf_counter() - started,
            "imported_distribution_modules": scientific_sources,
            "views": obs,
        }
        api.write(work / "scores.json", result)
        if out.exists():
            out.rmdir()
        work.rename(out)
        return result
    finally:
        if work.exists():
            shutil.rmtree(work)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("cpu-dir", "case", "output"):
        parser.add_argument("--" + n, type=Path, required=True)
    for n in ("release-sha256", "case-sha256"):
        parser.add_argument("--" + n, required=True)
    parser.add_argument("--threads", type=int, default=4)
    a = parser.parse_args()
    r = score(a)
    print(
        json.dumps(
            {
                "condition": r["condition"],
                "release_sha256": r["release_sha256"],
                "ready": sum(v["status"] == "ready" for v in r["views"].values()),
            }
        )
    )
