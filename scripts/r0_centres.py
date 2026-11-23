"""Direct gsplat 1.5.3 readback of supported projected centres and camera-space depth."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from evidence_lens._bundle import files, load, sha
from evidence_lens._verify import require
from worker_jobs import dispatch, frozen_json


def probe(a):
    import numpy as np
    import torch
    from gsplat.cuda._wrapper import fully_fused_projection
    from evidence_lens import stage3_gaussians as g

    device = g.check_gpu()
    cfg = load(a.config)
    inputs = {n: Path(v) for n, v in a.member}
    require(
        set(inputs) == set(cfg["roles"])
        and all(sha(f) == cfg["roles"][k]["sha256"] for k, f in inputs.items()),
        "Camera or original checkpoint SHA differs",
    )
    results = {}
    with torch.no_grad():
        for cell in cfg["cells"]:
            cameras = load(inputs[cell["camera_role"]])["views"]
            for phase, path in cell["phase_roles"].items():
                with np.load(inputs[path], allow_pickle=False) as z:
                    p = {k: torch.as_tensor(z[k], device=device) for k in g.PARAMS}
                means, scales, quats, opacity, _ = g.native_parameters(p)
                for v, cam in cameras.items():
                    k, pose, w, h, clip = g.view_tensor(cam, device)
                    radii, xy, z, _, _ = fully_fused_projection(
                        means,
                        None,
                        quats,
                        scales,
                        pose,
                        k,
                        w,
                        h,
                        eps2d=cfg["eps2d"],
                        near_plane=clip[0],
                        far_plane=clip[1],
                        packed=False,
                        calc_compensations=False,
                        opacities=opacity,
                        radius_clip=0.0,
                    )
                    r = radii.detach().cpu().numpy()[0]
                    valid = np.all(r > 0, axis=-1)
                    ids = np.flatnonzero(valid).astype("<i4")
                    uv = xy.cpu().numpy()[0, valid]
                    depth = z.cpu().numpy()[0, valid]
                    require(
                        np.isfinite(uv).all() and np.isfinite(depth).all(), "Invalid backend support values"
                    )
                    name = cell["cell"] + "-" + phase + "-" + v + ".npz"
                    f = a.output / name
                    np.savez_compressed(f, index=ids, radii=r[valid], means2d_edge=uv, camera_z_m=depth)
                    results[name] = {"sha256": sha(f), "bytes": f.stat().st_size, "rows": int(valid.sum())}
    g.write(
        a.output / "centres.json",
        {
            "schema": "el-r0-native-centres-1",
            "run_id": os.environ["EL_RUN_ID"],
            "files": results,
            "originals": cfg["originals"],
            "photographs": 0,
            "optimization_steps": 0,
            "runtime": g.runtime(),
            "job_sha256": cfg["formal_handoff_sha256"],
            "convention": "means2d uses gsplat edge-coordinate K with +0.5 offset on the integer-centre calibration",
        },
    )


def windows(a):
    require(os.name == "nt" and sha(CODE.parent / "job.json") == a.sha256, "Wrong host or job")
    j = load(CODE.parent / "job.json")
    cfg = j["config"]
    store = a.store.resolve()
    out = a.returns.resolve()
    require(
        j["schema"] == "el-r0-centres-job-1"
        and {k: sha(v) for k, v in files(CODE).items()} == j["worker_files"]
        and set(files(CODE.parent / "data")) == {r["path"] for r in cfg["roles"].values()},
        "Unexpected source or input",
    )
    require(
        all(sha(CODE.parent / "data" / r["path"]) == r["sha256"] for r in cfg["roles"].values()),
        "Input SHA mismatch",
    )
    record = {
        **cfg,
        **j["identity"],
        "experiment_id": j["experiment_id"],
        "trial_id": j["trial_id"],
        "formal_handoff_sha256": a.sha256,
        "recording": {"expected_output_bytes": 4 * 1024**2},
    }
    cfgfile = store / "r0-configs" / j["trial_id"] / "config.json"
    cfgfile.parent.mkdir(parents=True, exist_ok=True)
    frozen_json(cfgfile, record)
    out.mkdir(parents=True, exist_ok=True)

    def execute():
        args = [
            sys.executable,
            "-m",
            "evidence_lens.recorder",
            "--root",
            str(store),
            "--source-root",
            str(CODE),
            "--kind",
            "archive_verify",
            "--config",
            str(cfgfile),
        ]
        for pin in cfg["originals"].values():
            args += ["--parent", pin["run_id"] + "=" + pin["manifest_sha256"]]
        cmd = [
            "{python}",
            "-B",
            "{source}/scripts/r0_centres.py",
            "probe",
            "--config",
            "{config}",
            "--output",
            "{out}",
        ]
        for role, r in cfg["roles"].items():
            args += [
                "--input",
                role + "=" + str(CODE.parent / "data" / r["path"]),
                "--input-parent",
                role + "=" + r["parent"]["run_id"],
            ]
            cmd += ["--member", role, "{input:" + role + "}"]
        q = subprocess.run(
            args + ["--"] + cmd,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(CODE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if q.stderr:
            print(q.stderr, file=sys.stderr)
        return json.loads(q.stdout.strip().splitlines()[-1])

    ret = dispatch(store, out, j, a.sha256, "archive_verify", None, None, None, execute)
    print(json.dumps(ret))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(required=True, dest="act")
    w = sub.add_parser("windows")
    w.add_argument("--sha256", required=True)
    for k in ("store", "returns"):
        w.add_argument("--" + k, required=True, type=Path)
    c = sub.add_parser("probe")
    c.add_argument("--config", required=True, type=Path)
    c.add_argument("--output", required=True, type=Path)
    c.add_argument("--member", nargs=2, action="append", required=True)
    a = p.parse_args()
    (probe if a.act == "probe" else windows)(a)
