"""Replay the public A prepared bundles from an installed wheel. Reads deployment evidence only."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


CASES = {"square": (256, 256), "landscape": (448, 252), "portrait": (252, 448),
         "photo": (378, 252), "odd": (375, 251), "legacy-letterbox": (256, 256)}
METHODS = ("B-L1-transfer", "MS-L1", "MS-frozen", "MS-ft")
VIEWS = [f"v{i:02d}" for i in range(1, 7)]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, document):
    path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def peak_rss():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("peak_working_set", "working_set", "peak_paged", "paged",
                                               "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
        val = MemoryCounters()
        val.cb = ctypes.sizeof(val)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(-1), ctypes.byref(val), val.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(val.peak_working_set)
    import resource
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def cell(a):
    import numpy as np
    from evidence_lens_bundle import cli
    from evidence_lens_bundle.bindings import binding_from_bundle, frame_meta_from_bundle
    from evidence_lens_bundle.bundle import Bundle
    from evidence_lens_bundle.numerics import CHANNELS, compare_arrays
    from evidence_lens_bundle.report import render
    from evidence_lens_bundle.resident import load_scorer
    from evidence_lens_bundle.scoring import score

    dist = importlib.metadata.distribution("evidence-lens")
    if Path(dist.locate_file("evidence_lens_bundle/cli.py")).resolve() != Path(cli.__file__).resolve():
        raise ValueError("use the installed wheel for profile acceptance")
    b = Bundle(a.bundle, a.bundle_sha)
    w, h = CASES[a.tag]
    expected_profile = None if a.tag == "legacy-letterbox" else f"analysis-{w}x{h}-v1"
    if b.analysis_profile != expected_profile or list(b.targets) != VIEWS:
        raise ValueError("different registered profile or target set")
    b.validate()
    started = time.perf_counter()
    cpu = time.process_time()
    r = score(a.bundle, a.out / "scores", a.method, cpu_dir=a.models,
              expected_sha256=b.sha256, threads=4)
    score_wall, score_cpu = time.perf_counter() - started, time.process_time() - cpu
    file_peak = peak_rss()
    if r["counts"]["canvas"] != w*h*len(VIEWS):
        raise ValueError("wrong analysis denominator")
    html = render(a.out / "scores", a.out / "report.html", digest(a.out / "scores/scores.json"))
    source = {s:b.source(s) for s in b.order}
    binding = binding_from_bundle(b)
    rows = {}
    for cap in (0, 64 << 20):
        model = load_scorer(a.method, models_dir=a.models, threads=4, cache_bytes=cap)
        ctx = model.prepare_sources(source, version_binding=binding)
        for i, view in enumerate(b.targets):
            spec = r["views"][i]["arrays"]
            path = a.out / "scores" / spec["path"]
            if digest(path) != spec["sha256"]:
                raise ValueError("score receipt does not bind an actual output")
            with np.load(path, allow_pickle=False) as f:
                reference = {key:f[key] for key in CHANNELS}
            meta = frame_meta_from_bundle(b, view, binding=binding, sequence_id=a.tag)
            tick = time.perf_counter()
            out = model.score_frame(ctx, b.frame(view), b.cameras[view], frame_meta=meta)
            seconds = time.perf_counter()-tick
            diff = compare_arrays(reference, out, r["model"]["threshold"], grid_shape=(h,w))
            if not diff["bit_exact"]:
                raise ValueError("same host file and array scoring differ: " + a.method + " " + view)
            item = rows.setdefault(view, {"arrays_sha256":spec["sha256"], "counts":r["views"][i]["counts"]})
            item[str(cap)] = {"exact":diff["bit_exact"], "seconds_including_target_read":seconds,
                              "threshold_crossings":diff["threshold_crossings_common_finite"],
                              "array_digests":diff["new_sha256"]}
    write(a.out / "grade.json", {"tag":a.tag, "shape_wh":[w,h], "profile":expected_profile,
          "bundle_sha256":b.sha256, "method":a.method,"model":r["model"],"views":rows,
          "counts":r["counts"],"cost":{"file_score_wall_s":score_wall,"file_score_cpu_s":score_cpu,
            "peak_rss_bytes_at_file_completion":file_peak, "peak_rss_bytes_process_lifetime":peak_rss()}, "html":html,
          "scores_sha256":digest(a.out/"scores/scores.json"),"wheel_version":dist.version})


def main(a):
    if a.action == "worker":
        cell(a)
        return
    # Determine the exact allowlisted roster. Each input directory must be independently identifiable.
    folders = [p.parent for p in a.bundles.rglob("bundle.json")]
    roster = {}
    for folder in folders:
        name = folder.name
        if name not in CASES or name in roster:
            raise ValueError("unexpected or duplicate prepared case: "+name)
        roster[name] = folder
    if set(roster) != set(CASES):
        raise ValueError("registered six-case archive is incomplete")
    a.out.mkdir(exist_ok=True, parents=True)
    rows = []
    for tag, folder in roster.items():
        for method in a.methods:
            dest = a.out / tag / method
            dest.mkdir(parents=True,exist_ok=False)
            args = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "worker",
                    "--bundle",str(folder), "--bundle-sha",digest(folder/"bundle.json"),
                    "--method",method,"--tag",tag,"--out",str(dest)]
            if a.models:
                args += ["--models",str(a.models)]
            subprocess.run(args,check=True)
            rows.append(json.loads((dest/"grade.json").read_bytes()))
    write(a.out/"acceptance.json", {"schema":"el-v05-installed-grid-replay-1","run_id":os.getenv("EL_RUN_ID"),
          "platform":platform.platform(),"arch":platform.machine(),"python":platform.python_version(),
          "wheel_version":importlib.metadata.version("evidence-lens"),"methods":list(a.methods),
          "views_per_cell":len(VIEWS),"cells":rows,
          "packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}})


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("run","worker"))
    p.add_argument("--bundle",type=Path)
    p.add_argument("--bundles",type=Path)
    p.add_argument("--bundle-sha")
    p.add_argument("--tag",choices=CASES)
    p.add_argument("--method",choices=METHODS)
    p.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    p.add_argument("--models",type=Path)
    p.add_argument("--out",type=Path,required=True)
    main(p.parse_args())
