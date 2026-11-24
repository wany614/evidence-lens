"""Render only registered virtual/eligible camera predictions from one recorded M2 checkpoint."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require


def compute(cfg, checkpoint, cam_file, out):
    import numpy as np
    import torch
    from recovery.engine import valid_params
    from recovery.scene import cameras, draw, runtime, setup

    require(
        sha(checkpoint) == cfg["checkpoint_sha256"]
        and sha(cam_file) == cfg["cameras_sha256"],
        "probe inputs differ",
    )
    device = setup()
    doc = load(cam_file)
    names = [
        "s00",
        "s03",
        *[f"c{i:03}" for i in cfg["pool"]],
        *[f"q{i:02}" for i in range(4)],
    ]
    views, scale, calib, frame = cameras(cam_file, names, device)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(
        ck["schema"] == "el-recovery-checkpoint-1"
        and ck["step"] == 5000
        and ck["producer_run_id"] == cfg["checkpoint_producer"]
        and ck["source_calibration"] == {k: calib[k] for k in ("s00", "s03")}
        and ck["endpoint_frame_sha256"] == frame
        and abs(ck["scene_scale_m"] - scale) < 1e-7
        and ck["identity"]["recipe_sha256"] == cfg["recipe_sha256"],
        "wrong source-only checkpoint, roles, scale or physical frame",
    )
    require(len(ck["params"]["means"]) == 20000, "checkpoint point count differs")
    valid_params(ck["params"], 20000)
    params = {k: t.to(device) for k, t in ck["params"].items()}
    started = time.perf_counter()
    products = {}
    with torch.no_grad():
        for key in names:
            rgb, alpha, z, _ = draw(
                params, views[key], ck["recipe"]["eps2d"], depth=True
            )
            rgb, alpha, z = (
                t.cpu().numpy().astype(np.float32, copy=True) for t in (rgb, alpha, z)
            )
            require(
                np.isfinite(rgb).all()
                and np.isfinite(alpha).all()
                and ((alpha >= 0) & (alpha <= 1)).all(),
                "invalid predicted colour/alpha",
            )
            z[(alpha <= 0) | ~np.isfinite(z) | (z <= 0)] = np.nan
            arrays = {"alpha": alpha, "z": z}
            if not key.startswith("c"):
                arrays["rgb"] = rgb
            path = out / (key + ".npz")
            np.savez_compressed(path, **arrays)
            products[key] = {
                "path": path.name,
                "sha256": sha(path),
                "bytes": path.stat().st_size,
                "fields": sorted(arrays),
            }
    save(
        out / "render.json",
        {
            "schema": "el-b2-probe-render-1",
            "run_id": os.environ["EL_RUN_ID"],
            "object_id": cfg["object_id"],
            "cameras_sha256": sha(cam_file),
            "checkpoint_sha256": sha(checkpoint),
            "scene_scale_m": scale,
            "renderer": "gsplat RGB+ED; native recovered model photometric conversion; camera-Z in metres",
            "pool": cfg["pool"],
            "products": products,
            "worker_wall_seconds": time.perf_counter() - started,
            "resources": runtime(),
        },
    )


def launch(job, store, returns):
    from evidence_lens._verify import verify_bundle
    from evidence_lens.archive import pack

    cfg = load(job / "config.json")
    require(
        os.name == "nt" and cfg["experiment_id"] == "batch-two-dtu-v1",
        "Windows CUDA job required",
    )
    store.mkdir(parents=True, exist_ok=True)
    target = returns / cfg["object_id"]
    target.mkdir(parents=True, exist_ok=True)
    if (target / "completed.json").exists():
        rec = load(target / "completed.json")
        require(
            rec["config_sha256"] == sha(job / "config.json")
            and sha(target / rec["zip"]) == rec["zip_sha256"],
            "wrong previous return",
        )
        return
    cmd = [
        sys.executable,
        "-B",
        "-m",
        "evidence_lens.recorder",
        "--source-root",
        str(ROOT),
        "--root",
        str(store),
        "--kind",
        "render",
        "--config",
        str(job / "config.json"),
    ]
    for role, metadata in cfg["parents"].items():
        cmd += [
            "--parent",
            metadata["run_id"] + "=" + metadata["manifest_sha256"],
            "--input",
            role + "=" + str(job / cfg["input_names"][role]),
            "--input-parent",
            role + "=" + metadata["run_id"],
        ]
    cmd += [
        "--",
        "{python}",
        "-B",
        "{source}/scripts/b2_probe_worker.py",
        "compute",
        "--config",
        "{config}",
        "--checkpoint",
        "{input:checkpoint}",
        "--cameras",
        "{input:cameras}",
        "--out",
        "{out}",
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    if not proc.stdout.strip():
        raise ValueError(proc.stderr or "No recorder response")
    result = json.loads(proc.stdout.splitlines()[-1])
    rid = result["run_id"]
    folder = store / "runs" / rid
    v = verify_bundle(folder)
    dest = target / (rid + ".zip")
    packed = pack(folder, dest, store)
    save(
        target / "completed.json",
        {
            "schema": "el-b2-probe-return-1",
            "run_id": rid,
            "manifest_sha256": v["manifest_sha256"],
            "zip": dest.name,
            "zip_sha256": packed["container_sha256"],
            "config_sha256": sha(job / "config.json"),
            "status": v["run"]["execution_status"],
        },
    )
    require(proc.returncode == 0, "recorded probe renderer failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for command, fields in (
        ("launch", ("job", "store", "returns")),
        ("compute", ("config", "checkpoint", "cameras", "out")),
    ):
        a = sub.add_parser(command)
        for n in fields:
            a.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    if a.action == "launch":
        launch(a.job.resolve(), a.store.resolve(), a.returns.resolve())
    else:
        compute(load(a.config), a.checkpoint, a.cameras, a.out)
