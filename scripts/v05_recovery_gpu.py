"""Compare the five native optimizer tensors, then render one registered source camera with the release code."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(root, out):
    ticket = json.loads((root / "ticket.json").read_bytes())
    if ticket["schema"] != "el-v05-recovery-render-ticket-1" or ticket["source_view"] != "s00":
        raise ValueError("incorrect single-source engineering ticket")
    for name, expected in ticket["files"].items():
        if file_sha(root / name) != expected:
            raise ValueError("changed packet member " + name)
    sys.path.insert(0, str(root / "runtime"))
    from evidence_lens_asset.gaussian import RECOVERY, load
    from evidence_lens_asset.render import device, layer
    from evidence_lens_bundle.common import encoded, require

    torch = device()
    require(torch.cuda.device_count() == 1, "control requires one visible CUDA device")
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    checkpoint = root / "checkpoint.pt"
    camera = json.loads((root / "camera.json").read_bytes())
    metadata = {}
    arrays = load(checkpoint, RECOVERY, 1.0, metadata=metadata)
    native = torch.load(checkpoint, map_location="cpu", weights_only=True)
    calibration = hashlib.sha256(encoded(camera)).hexdigest()
    require(native["source_calibration"]["s00"] == calibration, "camera differs from original checkpoint calibration")
    require(len(arrays["means"]) == ticket["gaussians"] < native["recipe"]["topology"]["cap"] == ticket["cap"],
            "incorrect number of Gaussian primitives or native budget")
    # Names and activations come from recovery.scene.draw in the registered original producer.
    activations = {
        "means": ("means", lambda t: t),
        "log_scales": ("scales", torch.exp),
        "quats": ("quats", lambda t: torch.nn.functional.normalize(t, dim=-1)),
        "opacity_logits": ("opacities", torch.sigmoid),
        "color_logits": ("colors", torch.sigmoid),
    }
    comparison = {}
    with torch.no_grad():
        for key, (name, activate) in activations.items():
            expect = native["params"][name].contiguous()
            observed = np.ascontiguousarray(arrays[key])
            raw_expected = expect.numpy().tobytes()
            raw_observed = observed.tobytes()
            first = activate(expect.cuda()).contiguous().cpu().numpy().tobytes()
            second = activate(torch.from_numpy(observed).cuda()).contiguous().cpu().numpy().tobytes()
            comparison[key] = {"optimizer_field": name, "shape": list(observed.shape),
                               "raw_exact": raw_expected == raw_observed, "activation_exact": first == second,
                               "raw_sha256": hashlib.sha256(raw_observed).hexdigest(),
                               "activated_sha256": hashlib.sha256(second).hexdigest()}
        require(all(x["raw_exact"] and x["activation_exact"] for x in comparison.values()), "changed optimizer parameter")
        prediction = layer(torch, arrays, camera, recovery={**metadata, "clip_m": camera["clip_m"]})
        torch.cuda.synchronize()
    path = out / "s00.npz"
    np.savez_compressed(path, **prediction)
    data = {"schema": "el-v05-recovery-render-1", "run_id": os.environ["EL_RUN_ID"], "ticket_run": ticket["run_id"],
            "checkpoint_sha256": file_sha(checkpoint), "source_camera_sha256": calibration, "source_view": "s00",
            "model": metadata, "gaussians": len(arrays["means"]), "clip_m": camera["clip_m"], "parameters": comparison,
            "prediction": {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path)},
            "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "gsplat": importlib.metadata.version("gsplat"),
                        "gpu": torch.cuda.get_device_name(0), "python": platform.python_version()},
            "source_photo_reads": 0, "target_reference_reads": 0, "new_optimization_steps": 0,
            "previous_prediction_available_to_gpu": False}
    (out / "recovery.json").write_bytes(encoded(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(Path(__file__).resolve().parent, parser.parse_args().out)
