"""Pack frozen inference sources and original G3 workloads; launch the same ticket on Mac and Windows."""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from evidence_lens_bundle.common import encoded, parts, row, sha, write

FILES = {
    "evidence_lens": (
        "__init__",
        "features",
        "scoring_io",
        "multisource_io",
        "multisource",
        "multisource_model",
        "l1",
        "risk_model",
    ),
    "evidence_lens_bundle": (
        "__init__",
        "bindings",
        "bundle",
        "calibration",
        "common",
        "report",
        "resident",
        "scoring",
        "replay",
        "replay_sidecar",
    ),
    "evidence_lens_multirisk": ("__init__", "data", "deploy", "model"),
    "evidence_lens_vision": ("__init__", "model", "contract"),
}
DRIVER = """import argparse, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
from zipfile import ZipFile
base = Path(__file__).resolve().parent
ticket = json.loads((base/"ticket.json").read_bytes())
for spec in ticket["files"]:
    p = base/spec["path"]
    if p.stat().st_size != spec["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError("modified CPU ticket " + spec["path"])
if not os.environ.get("EL_RUN_ID"): raise ValueError("start the recorder before the worker")
p=argparse.ArgumentParser()
p.add_argument("--operation",choices=("g3","g4"),required=True)
p.add_argument("--platform",choices=("mac","windows"),required=True)
p.add_argument("--models",type=Path,required=True)
p.add_argument("--out",type=Path,required=True)
p.add_argument("--trajectories",type=Path)
p.add_argument("--selection",type=Path)
a=p.parse_args()
if a.out.exists() and any(a.out.iterdir()): raise ValueError("expected an empty, fresh recorder payload")
a.out.mkdir(parents=True,exist_ok=True)
runtime=base/"runtime"
env={**os.environ,"PYTHONPATH":str(runtime),"PYTHONDONTWRITEBYTECODE":"1"}
args=["--models",str(a.models.resolve()),"--out",str(a.out.resolve())]
with tempfile.TemporaryDirectory() as raw:
    if a.operation=="g3":
        with ZipFile(base/"workloads.zip") as z:
            for name in z.namelist():
                f=Path(name)
                if f.is_absolute() or ".." in f.parts: raise ValueError("invalid file path in frozen workload")
                z.extract(name,raw)
        cmd=[sys.executable,str(runtime/"g3_perf.py"),"run","--trajectories",raw,"--budget","1800","--n","200",*args]
    else:
        if not a.selection or not a.trajectories: raise ValueError("G4 needs recorded platform selection and GPU trajectory directory")
        cmd=[sys.executable,str(runtime/"g4_replay_eval.py"),"--platform",a.platform,"--selection",str(a.selection.resolve()),"--trajectories",str(a.trajectories.resolve()),*args]
    result=subprocess.run(cmd,env=env,check=False)
raise SystemExit(result.returncode)
"""


def make(a):
    payload = {}
    for pkg, mods in FILES.items():
        for name in mods:
            f = f"{pkg}/{name}.py"
            payload["runtime/" + f] = (a.project / "src" / f).read_bytes()
    for name in ("g3_perf", "g3_perf_resources", "g4_replay_eval"):
        f = name + ".py"
        payload["runtime/" + f] = (a.project / "scripts" / f).read_bytes()
    with ZipFile(a.assets) as archive:
        for k in ("profile.json", "MS-L1/model.json", "MS-L1/coefficients.json", "MS-L1/preprocessing.json"):
            payload["runtime/evidence_lens_bundle/assets/" + k] = archive.read("assets/" + k)
    payload.update(
        {
            "run_cpu.py": DRIVER.encode(),
            "protocol.json": (a.project / "protocol/general-static-engineering-v1.json").read_bytes(),
            "workloads.zip": a.workloads.read_bytes(),
        }
    )
    with ZipFile(a.legal) as archive:
        for f in ("LICENSE", "NOTICE", "CC-BY-4.0.txt", "ORIGINAL-EXAMPLE-NOTICE.txt"):
            payload[f] = archive.read(f)
    files = [row(k, v) for k, v in sorted(payload.items())]
    ticket = {
        "schema": "el-g3-g4-cpu-ticket-1",
        "run_id": os.environ["EL_RUN_ID"],
        "files": files,
        "cpu_models_index_sha256": "150b04e702f5386b5808ac17545cd073e3d4d837ea789e6ae167168ac086084d",
        "weight_source": "unchanged public v0.3.0-rc.1 CPU bundle",
        "selection": "originals only, engineering code before label-blind holdout bridge",
    }
    payload["ticket.json"] = encoded(ticket)
    file = a.out / "original-static-cpu.zip"
    with ZipFile(file, "x", compression=ZIP_DEFLATED) as z:
        for n, raw in sorted(payload.items()):
            z.writestr(n, raw)
    write(
        a.out / "dispatch.json",
        {
            "run_id": os.environ["EL_RUN_ID"],
            "artifact": row(file.name, file.read_bytes()),
            "ticket_sha256": sha(payload["ticket.json"]),
            "protocol_sha256": sha(payload["protocol.json"]),
        },
    )


def launch(a):
    # Recorder input can carry a verified ZIP atop an immutable code snapshot.
    with tempfile.TemporaryDirectory() as tmp, ZipFile(a.ticket) as z:
        root = Path(tmp)
        for n in z.namelist():
            parts(n)
            z.extract(n, root)
        cmd = [
            sys.executable,
            "-I",
            "-B",
            str(root / "run_cpu.py"),
            "--operation",
            a.operation,
            "--platform",
            a.platform,
            "--models",
            str(a.models),
            "--out",
            str(a.out),
        ]
        if a.operation == "g4":
            cmd += ["--trajectories", str(a.trajectories), "--selection", str(a.selection)]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("make", "launch"))
    for n in (
        "project",
        "assets",
        "workloads",
        "legal",
        "out",
        "models",
        "trajectories",
        "selection",
        "ticket",
    ):
        p.add_argument("--" + n, type=Path)
    p.add_argument("--operation", choices=("g3", "g4"))
    p.add_argument("--platform", choices=("mac", "windows"))
    args = p.parse_args()
    (make if args.action == "make" else launch)(args)
