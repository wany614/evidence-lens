"""Common Windows worker operations. All hardware readings are stage-local."""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


def read(path):
    with Path(path).open(encoding="utf8") as stream:
        return json.load(stream)


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_id():
    value = os.environ.get("EL_RUN_ID")
    if not value:
        raise RuntimeError("Start this worker through evidence_lens.recorder.")
    return value


def require_sharp(commit):
    import sharp

    root = Path(sharp.__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or git("rev-parse", "HEAD") != commit:
        raise RuntimeError("The editable SHARP checkout must match the configured commit.")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("SHARP checkout contains local changes; use a clean pinned checkout.")
    return commit


def require_cuda(cfg):
    import torch

    if platform.system() != "Windows" or not torch.cuda.is_available():
        raise RuntimeError("The registered Windows CUDA worker is required.")
    if torch.__version__.split("+")[0] != cfg["generator"]["torch_version"]:
        raise RuntimeError(f"Unexpected PyTorch build: {torch.__version__}")
    if md.version("gsplat") != cfg["renderer"]["gsplat_version"]:
        raise RuntimeError("Unexpected gsplat version.")
    torch.manual_seed(cfg["seed"])
    torch.set_grad_enabled(False)
    return torch.device("cuda:0")


@contextmanager
def measured(name, stats, device):
    import psutil
    import torch

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    free, total = torch.cuda.mem_get_info(device)
    start = time.perf_counter()
    data = {"cuda_free_start_bytes": free, "cuda_total_bytes": total,
            "system_available_start_bytes": psutil.virtual_memory().available}
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        data.update(seconds=time.perf_counter() - start,
                    cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(device),
                    cuda_reserved_peak_bytes=torch.cuda.max_memory_reserved(device),
                    cuda_free_end_bytes=torch.cuda.mem_get_info(device)[0],
                    system_available_end_bytes=psutil.virtual_memory().available)
        stats[name] = data


def runtime_info(device):
    import torch

    return {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "gsplat": md.version("gsplat"), "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "nvidia_smi_present": shutil.which("nvidia-smi") is not None}
