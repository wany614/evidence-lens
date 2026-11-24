"""Inspect one Windows CUDA runtime for the external gsplat trainer. The packet carries no study data."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys


def run(out):
    import torch
    from gsplat.strategy import MCMCStrategy
    import gsplat.strategy.mcmc as module

    ticket = json.loads((Path(__file__).parent / "ticket.json").read_bytes())
    if ticket["schema"] != "el-many-gsplat-inventory-ticket-1" or ticket["allowed_datasets"]:
        raise ValueError("unexpected inventory scope")
    packages = {}
    for name in ("gsplat", "torch", "numpy", "Pillow", "tqdm", "PyYAML", "scikit-learn", "scipy",
                 "tensorboard", "pycolmap", "opencv-python", "fused-ssim", "torchmetrics", "viser", "tyro"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    if packages["gsplat"] != "1.5.3" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("pinned gsplat 1.5.3 and one CUDA device required")
    strategy_file = Path(module.__file__)
    strategy_sha = hashlib.sha256(strategy_file.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    sample = MCMCStrategy(cap_max=60000, refine_start_iter=500, refine_stop_iter=4500, refine_every=100)
    receipt = {"schema": "el-many-gsplat-inventory-1", "run_id": os.environ["EL_RUN_ID"],
               "ticket": ticket["run_id"], "commit": ticket["commit"], "python": platform.python_version(),
               "device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
               "packages": packages, "fused_ssim_importable": importlib.util.find_spec("fused_ssim") is not None,
               "cap": sample.cap_max, "mcmc_module_canonical_lf_sha256": strategy_sha,
               "matches_official_mcmc": strategy_sha == ticket["mcmc_canonical_lf_sha256"],
               "training_steps": 0, "fits": 0, "photo_reads": 0}
    (out / "inventory.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args().out)
