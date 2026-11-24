"""Experimental old-head Q4 adapter. Stage A weights and numerical kernels are unchanged."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from evidence_lens_bundle.common import encoded, require, sha
from evidence_lens_bundle.scoring import calculate_frame, models

from .schema import BASELINE_ID, PROFILE, SIZE


@lru_cache(maxsize=3)
def _released(method, folder, threads):
    # Each scorer instance owns the exact released model admitted by the Stage A hash checks.
    return models(method, Path(folder) if folder else None, threads)


def legacy_q4(bank, target_id, queried_source_ids, method, models_dir=None, threads=4):
    ids = tuple(queried_source_ids)
    require(
        method in ("MS-L1", "MS-ft")
        and len(ids) == len(set(ids)) == 4
        and set(ids) <= set(bank.ids) & bank.granted
        and target_id in bank.target_ids
        and bank.meta["reconstruction_provenance"] == "verified",
        "legacy transfer needs the same four unique registered authorized source IDs",
    )
    model, spec = _released(method, str(Path(models_dir).resolve()) if models_dir else None, threads)
    if method == "MS-ft":
        import torch

        require(torch.get_num_threads() == threads, "legacy neural process thread count changed")
    sources = {name: bank.source(name) for name in ids}
    frame = bank.target(target_id)
    missing = [n for n, s in sources.items() if s is None]
    if missing:
        risk = np.full(SIZE[::-1], np.nan, np.float32)
        arrays = {
            "risk": risk,
            "valid": np.zeros(risk.shape, bool),
            "state": np.full(risk.shape, 255, np.uint8),
            "unknown_reason": np.full(risk.shape, 254, np.uint8),
        }
    else:
        arrays = calculate_frame(
            frame,
            bank.cameras[target_id],
            sources,
            ids[0],
            bank.meta["depth_verified"],
            method,
            model,
            spec["threshold"],
            rectangular=True,
        )
    return {
        "rank": arrays["risk"],
        "finite": arrays["valid"],
        "state": arrays["state"],
        "reason": arrays["unknown_reason"],
        "profile": {
            "schema": "el-b-legacy-q4-transfer-1",
            "baseline_id": BASELINE_ID,
            "analysis_profile": PROFILE,
            "method": method,
            "model": spec,
            "queried_source_ids": list(ids),
            "metric_anchor_ids": bank.meta["metric_anchor_ids"],
            "frame_receipt_sha256": sha(encoded(bank.meta["frame_receipt"])),
            "bank_sha256": bank.sha256,
            "target_id": target_id,
            "missing_sources": missing,
            "unavailable_source_reason_code": 254,
            "threshold_status": "legacy_threshold_external_calibration_unestablished",
        },
    }
