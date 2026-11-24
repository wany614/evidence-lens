"""A shared native-pixel head on stage B's verified DINOv2 visual backbone."""

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from evidence_lens_vision.contract import digest, read, require
from evidence_lens_vision.model import (
    ARCH, RiskModel, image_tensor, rectangular_image_tensor, rectangular_project, rectangular_sample,
)

B_INDEX = "08af4cba3537f6f677b24d045c4c374d2d46eedc63a20abc874cfcad3f7ddac5"
B_WEIGHTS = "f372bd34faa1b345ae2486fe6c96aea00d8d9ac5749b0b2c237f9ed08d50f228"
B_RUN = "20261123T133146588950Z-c1d19dcb78c74cf0"
B_RELEASE = "fb2fbdc232cf2be706ca88e4ad9ddfe884013cc3"
METHODS = ("MS-frozen", "MS-ft")
SEEDS = (1729, 2718)
HEAD = ("projection.", "head.")
TAIL = ("backbone.blocks.11.", "backbone.norm.")
WITNESSES = (
    "projection.weight",
    "head.2.weight",
    "backbone.blocks.11.attn.qkv.weight",
    "backbone.norm.weight",
)


def groups(method):
    require(method in METHODS, "unknown registered multi-source neural method")
    return HEAD if method == METHODS[0] else (*HEAD, *TAIL)


def architecture(columns):
    return {
        **ARCH,
        "schema": "el-multisource-dino-architecture-1",
        "legal_features": len(columns),
        "columns": list(columns),
        "feature_fusion": "target patch vector plus registered masked source aggregates",
        "feature_pyramid": False,
        "trained_scoring_scope": "all",
        "reconstruction_conditions": ["M2", "M4"],
    }


def initialize(index, weights, columns, seed, method):
    index, weights = Path(index), Path(weights)
    require(
        digest(index) == B_INDEX and digest(weights) == B_WEIGHTS,
        "expected the published stage B FT index and FP32 weights",
    )
    old_index = read(index)
    spec = old_index["files"]["weights.safetensors"]
    require(
        old_index["model_id"] == B_RUN
        and old_index["branch"] == "L2-vision-ft"
        and spec["sha256"] == B_WEIGHTS
        and spec["bytes"] == weights.stat().st_size,
        "the selected upstream model index and safetensors disagree",
    )
    before = load_file(str(weights), device="cpu")
    return initialize_tensors(before, columns, seed, method)


def initialize_tensors(old, columns, seed, method):
    """The production entry calls this only after both registered public artifacts verify."""
    require(
        seed in SEEDS and len(columns) == len(set(columns)) and len(columns) > 0,
        "invalid seed or feature schema",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RiskModel(features=len(columns)).cpu()
    expected = model.state_dict()
    require(set(expected) == set(old), "stage B tensor names disagree with the registered superclass")
    retained, replaced = {}, {}
    with torch.no_grad():
        for name, tensor in old.items():
            target = expected[name]
            require(
                tensor.dtype == target.dtype == torch.float32 and torch.isfinite(tensor).all().item(),
                "nonfinite or non-FP32 B tensor " + name,
            )
            if name.startswith("backbone."):
                require(target.shape == tensor.shape, "incompatible pretrained tensor " + name)
                target.copy_(tensor)
                retained[name] = {"shape": list(tensor.shape), "scalars": tensor.numel()}
            else:
                require(name.startswith(HEAD), "an unregistered B tensor would have been discarded: " + name)
                replaced[name] = {
                    "B_shape": list(tensor.shape),
                    "C_shape": list(target.shape),
                    "B_scalars": tensor.numel(),
                    "C_scalars": target.numel(),
                    "policy": "reinitialized_same_shape"
                    if tensor.shape == target.shape
                    else "reinitialized_shape_changed",
                }
    require(
        len(retained) == 175
        and len(replaced) == 6
        and sum(t["scalars"] for t in retained.values()) == 22056576,
        "incomplete pretrained backbone or stage B projection/head",
    )
    roles = groups(method)
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(roles))
    model.eval()
    report = {
        "schema": "el-multisource-initialization-1",
        "B_release_commit": B_RELEASE,
        "B_model_sha256": B_INDEX,
        "B_weights_sha256": B_WEIGHTS,
        "B_model_run": B_RUN,
        "seed": seed,
        "method": method,
        "retained": retained,
        "reinitialized": replaced,
        "unexplained_names": [],
        "retained_scalars": sum(t["scalars"] for t in retained.values()),
        "reinitialized_scalars": sum(t["C_scalars"] for t in replaced.values()),
        "trainable_scalars": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_names": [name for name, p in model.named_parameters() if p.requires_grad],
    }
    return model, report


def transform(raw, pre):
    from evidence_lens.multisource_model import transform as shared_transform

    result = shared_transform(raw, pre, dtype=np.float32)
    require(np.isfinite(result).all(), "multi-source preprocessing produced a nonfinite feature")
    return result


def native_score(model, rgb, x, scoreable, pre, device, tau=None, *, rectangular=False):
    h, w = rgb.shape[:2]
    require(
        x.dtype == np.float32
        and x.shape == (h, w, len(pre["columns"]))
        and scoreable.dtype == np.bool_
        and scoreable.shape == (h, w),
        "invalid native multi-source grid",
    )
    risk = np.full(h * w, np.nan, np.float32)
    with torch.inference_mode():
        if rectangular:
            image, valid_patches, frame, observed = rectangular_image_tensor(rgb, device)
            features = rectangular_project(model, image)
        else:
            image, valid_patches, frame, observed = image_tensor(rgb, device)
            features = model.project(image)
        idx = np.flatnonzero((scoreable & observed).ravel())
        for begin in range(0, len(idx), 16384):
            subset = idx[begin : begin + 16384]
            numeric = torch.from_numpy(transform(x.reshape(h * w, -1)[subset], pre)).to(device)
            if rectangular:
                visual, fraction = rectangular_sample(features, valid_patches, frame, subset)
                logits = model.head(torch.cat((visual, numeric), 1)).flatten()
                supported = fraction > 0
            else:
                logits, supported = model.at(features, valid_patches, frame, subset, numeric)
            require(
                torch.isfinite(logits[supported]).all().item(), "nonfinite neural score on eligible evidence"
            )
            yes = supported.cpu().numpy()
            risk[subset[yes]] = torch.sigmoid(logits[supported]).cpu().numpy()
    risk = risk.reshape(h, w)
    valid = np.isfinite(risk)
    require(
        np.array_equal(valid, scoreable & observed),
        "neural support differs from registered finite native support",
    )
    result = {"risk": risk, "valid": valid}
    if tau is not None:
        state = np.full(risk.shape, 255, np.uint8)
        state[valid] = (risk[valid].astype(np.float64) >= tau).astype(np.uint8)
        result["state"] = state
    return result
