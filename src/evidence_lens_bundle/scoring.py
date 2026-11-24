"""Score prepared predictions on CPU. No independent reference is accepted."""

from __future__ import annotations

import io
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, __version__ as pillow_version

from evidence_lens.l1 import risk_state
from evidence_lens.multisource import REASONS, extract_ordered
from evidence_lens.multisource_model import light_risk, load_light, load_transfer, transfer_risk
from evidence_lens.scoring_io import alpha_valid, depth_valid, new_output, rgb_valid

from .bundle import Bundle
from .common import ASSETS, PROFILE, RELEASE_INDEX_SHA256, Unsupported, profile, require, row, write

REASON_NAMES = (*REASONS, "render_failed", "visual_input_unavailable")


def models(method, cpu_dir, threads):
    registry = profile()["models"]
    require(method in registry, "unknown released model", Unsupported)
    spec = registry[method]
    if cpu_dir is None:
        require(
            method == "MS-L1",
            "this method needs --models pointing to the extracted published CPU release",
            Unsupported,
        )
        pkg = ASSETS / "MS-L1"
    else:
        from .common import read_file, sha

        require(
            sha(read_file(cpu_dir, "release.json")) == RELEASE_INDEX_SHA256, "wrong CPU release index SHA-256"
        )
        pkg = Path(cpu_dir) / "models" / method
        require(
            pkg.resolve().is_relative_to(Path(cpu_dir).resolve()) and not pkg.is_symlink(),
            "model path leaves the supplied release",
        )
    index = spec["model_index_sha256"]
    if method == "MS-L1":
        m, coeff, prep = load_light(pkg, index)
        require(
            m["threshold"] == spec["threshold"]
            and m["model_id"] == spec["model_id"]
            and m["weights_sha256"] == spec["weights_sha256"],
            "linear model operating point differs",
        )
        return (coeff, prep), spec
    if method == "B-L1-transfer":
        return load_transfer(pkg, index, spec["weights_sha256"], spec["threshold"]), spec
    try:
        import torch
        from evidence_lens_multirisk.deploy import load_model
    except ImportError as e:
        raise Unsupported(
            "MS-frozen / MS-ft require the optional 'vision' dependencies including CPU PyTorch and safetensors"
        ) from e
    torch.set_num_threads(threads)
    model = load_model(pkg, index, "cpu")
    require(
        model[2]["method"] == method
        and model[2]["model_id"] == spec["model_id"]
        and model[2]["threshold"] == spec["threshold"]
        and model[2]["files"]["weights.safetensors"]["sha256"] == spec["weights_sha256"],
        "visual model identity differs",
    )
    return model, spec


def calculate(bundle, view, frame, sources, method, m, tau):
    target_cam = bundle.cameras[view]
    if frame is None:
        risk = np.full(target_cam.shape, np.nan, np.float32)
        why = np.full(target_cam.shape, REASON_NAMES.index("render_failed"), np.uint8)
    elif method == "B-L1-transfer":
        verified = bundle.producer["depth_verified"]
        risk = transfer_risk(frame, target_cam, sources[bundle.primary], m, verified)
        why = np.select(
            [
                np.full(target_cam.shape, not verified),
                ~rgb_valid(frame["rgb"]),
                ~alpha_valid(frame["alpha"]),
                ~depth_valid(frame["z"]),
            ],
            [1, 2, 3, 4],
            default=0,
        ).astype(np.uint8)
    else:
        feat = extract_ordered(
            frame, target_cam, tuple(sources.items()), depth_verified=bundle.producer["depth_verified"]
        )
        why = feat["unknown_reason"].copy()
        if method == "MS-L1":
            risk = light_risk(feat, *m)
        else:
            from evidence_lens_multirisk.deploy import score_grid

            original = score_grid(m, frame["rgb"], feat["X"], feat["scoreable"], "cpu")
            risk = original["risk"]
            require(
                np.array_equal(original["state"], risk_state(risk, tau))
                and np.array_equal(original["valid"], np.isfinite(risk)),
                "neural support/state mismatch",
            )
            why[(why == 0) & ~np.isfinite(risk)] = REASON_NAMES.index("visual_input_unavailable")
    valid = np.isfinite(risk)
    require(
        np.array_equal(why == 0, valid) and np.all((risk[valid] >= 0) & (risk[valid] <= 1)),
        "invalid risk or reason partition",
    )
    return {"risk": risk, "valid": valid, "state": risk_state(risk, tau), "unknown_reason": why}


def counts(a):
    state = a["state"]
    return {
        "canvas": state.size,
        "scoreable": int(a["valid"].sum()),
        "low": int(np.count_nonzero(state == 0)),
        "high": int(np.count_nonzero(state == 1)),
        "unknown": int(np.count_nonzero(state == 255)),
    }


def reasons(a):
    return {name: int(np.count_nonzero(a["unknown_reason"] == i)) for i, name in enumerate(REASON_NAMES) if i}


def rgb_preview(rgb):
    valid = rgb_valid(rgb)
    image = np.rint(np.clip(np.nan_to_num(rgb, nan=0), 0, 1) * 255).astype(np.uint8)
    image[~valid] = [114, 117, 126]
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return stream.getvalue()


def score(
    root,
    output,
    method="MS-L1",
    *,
    cpu_dir=None,
    expected_sha256=None,
    threads=4,
    calibration_profile=None,
    calibration_sha256=None,
    scope_proof=None,
    scope_sha256=None,
):
    from .calibration import select, states, state_counts

    require(type(threads) is int and 1 <= threads <= 256, "CPU thread count must be 1–256")
    started = perf_counter()
    bundle = Bundle(root, expected_sha256)
    selected = select(bundle, method, calibration_profile, calibration_sha256, scope_proof, scope_sha256)
    m, spec = models(method, cpu_dir, threads)
    after_load = perf_counter()
    source_ids = (bundle.primary,) if method == "B-L1-transfer" else bundle.order
    sources = {s: bundle.source(s) for s in source_ids}
    observations = []
    with new_output(output) as stage:

        def save(relative, data):
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return row(relative, data)

        photos = []
        for i, s in enumerate(bundle.order):
            data = bundle.photo_bytes(s)
            # Decode even unscored authorized source photos. Render displayed RGB with no original metadata.
            from evidence_lens.multisource_io import decode_png

            photo = decode_png(data, bundle.cameras[s])
            photos.append(
                {
                    "id": s,
                    "original_sha256": bundle.meta["sources"][s]["sha256"],
                    "display": save(f"sources/{i:03d}.png", rgb_preview(photo)),
                    "self_render_status": bundle.producer["frames"][s]["status"],
                }
            )
        for i, t in enumerate(bundle.targets):
            tick = perf_counter()
            frame = bundle.frame(t)
            a = calculate(bundle, t, frame, sources, method, m, spec["threshold"])
            raw = io.BytesIO()
            np.savez_compressed(raw, **a)
            folder = f"views/{i:03d}/"
            observations.append(
                {
                    "id": t,
                    "render": bundle.producer["frames"][t],
                    "width": bundle.cameras[t].width,
                    "height": bundle.cameras[t].height,
                    "counts": counts(a),
                    "unknown_reasons": reasons(a),
                    "arrays": save(folder + "scores.npz", raw.getvalue()),
                    "prediction": None
                    if frame is None
                    else save(folder + "prediction.png", rgb_preview(frame["rgb"])),
                    "read_score_write_seconds": perf_counter() - tick,
                }
            )
            if selected and selected["applied"]:
                state = states(a["risk"], selected["threshold"])
                raw_state = io.BytesIO()
                np.save(raw_state, state, allow_pickle=False)
                observations[-1]["calibrated"] = {
                    "counts": state_counts(state),
                    "state": save(folder + "calibrated-state.npy", raw_state.getvalue()),
                }
        summary = {k: sum(x["counts"][k] for x in observations) for k in observations[0]["counts"]}
        if selected and selected["applied"]:
            selected["counts"] = {
                key: sum(v["calibrated"]["counts"][key] for v in observations)
                for key in ("canvas", "low", "high", "unknown")
            }
        endpoint = [bundle.cameras[s].c2w[:3, 3] for s in bundle.endpoints]
        receipt = {
            "schema": "el-bundle-scores-1",
            "run_id": os.getenv("EL_RUN_ID"),
            "bundle_sha256": bundle.sha256,
            "creation": bundle.meta["creation"],
            "generator": bundle.producer["generator"],
            "renderer": bundle.producer["renderer"],
            "checkpoint_sha256": bundle.producer["checkpoint_sha256"],
            "camera_sha256": bundle.meta["camera"]["sha256"],
            "condition": bundle.condition,
            "source_order": list(bundle.order),
            "primary_source_id": bundle.primary,
            "endpoint_source_ids": list(bundle.endpoints),
            "endpoint_distance_m": float(np.linalg.norm(endpoint[0] - endpoint[1])),
            "target_order": list(bundle.targets),
            "scorer_source_ids": list(source_ids),
            "sources": photos,
            "input_domain": bundle.meta["input_domain"],
            "tested_domain": "Native 256 x 256, M2/M4; original synthetic data and reported sparse-view photo experiments.",
            "geometry": bundle.meta["geometry"],
            "depth_verified": bundle.producer["depth_verified"],
            "profile_id": PROFILE,
            "release_index_sha256": RELEASE_INDEX_SHA256,
            "method": method,
            "model": spec,
            "calibration_profile": selected,
            "reason_codes": dict(enumerate(REASON_NAMES)),
            "counts": summary,
            "views": observations,
            "cost": {
                "device": "cpu",
                "configured_neural_threads": threads if method.startswith(("MS-ft", "MS-frozen")) else None,
                "bundle_and_model_load_seconds": after_load - started,
                "worker_to_receipt_seconds": perf_counter() - started,
                "process_environment": {
                    "python": sys.version.split()[0],
                    "numpy": np.__version__,
                    "pillow": pillow_version,
                    "platform": platform.system(),
                    "machine": platform.machine(),
                },
            },
        }
        write(stage / "scores.json", receipt)
    return receipt
