"""Compare the two released FP32 scorers with the pinned official DINOv2 implementation on Stage A grids."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import tarfile
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from evidence_lens_multirisk.deploy import load_model, score_grid
from evidence_lens_vision.contract import digest, put, read, require
from evidence_lens_vision.model import rectangular_image_tensor, rectangular_sample


OFFICIAL = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
SOURCE_TREE = "b61b5cfcb9344ada215d4ff7c147587fbb3f711199e809388f42715ffa91cf2e"
REGISTRATION = "6bc9ca2ac09e3da1780bbae0df06aa24a13b99b5df1af458a6329b56d1f42df6"
CPU_RELEASE = "150b04e702f5386b5808ac17545cd073e3d4d837ea789e6ae167168ac086084d"
MODELS = {
    "MS-frozen": (
        "56991821dcbc3d05e7c7e17466f21c8af764cb772cc6116b42f53b99a91eeb4a",
        "00be5de5abd30bf3e7eb1686e458a96e3ca55dffd6b892af862c98913c85c0b6",
    ),
    "MS-ft": (
        "c2b465507c0caee5470af927be670bcaaa89396eecb6c262d778feb19b431318",
        "99dfef7c82b808939895ffbbd0327513011920e1f47a5d0556d0eb6c6c6d0fd3",
    ),
}
HAND_PATCH_IDS = {
    (256, 256): [18, 19, 360],
    (448, 252): [31, 32, 575],
    (252, 448): [17, 18, 575],
    (378, 252): [26, 27, 485],
    (375, 251): [26, 27, 485],
}


def original_member(name):
    return name in ("LICENSE", "MODEL_CARD.md") or name.startswith("dinov2/") and name.endswith(".py")


@contextmanager
def official_implementation(source):
    """Only execute the 157 inventoried members of the published Apache-2.0 Git tree."""
    blobs = {}
    if source.is_dir():
        files = [source / "LICENSE", source / "MODEL_CARD.md", *(source / "dinov2").rglob("*.py")]
        for path in files:
            require(
                path.resolve().is_relative_to(source.resolve()) and not path.is_symlink(),
                "external source member",
            )
            blobs[path.relative_to(source).as_posix()] = path.read_bytes()
    else:
        with tarfile.open(source) as archive:
            for member in archive:
                _, has_prefix, name = member.name.partition("/")
                if has_prefix and original_member(name):
                    require(
                        member.isfile() and ".." not in PurePosixPath(name).parts and name not in blobs,
                        "invalid official source archive member",
                    )
                    blobs[name] = archive.extractfile(member).read()
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in blobs.items()}
    canonical = json.dumps(hashes, separators=(",", ":"), sort_keys=True).encode()
    require(
        len(blobs) == 157 and hashlib.sha256(canonical).hexdigest() == SOURCE_TREE,
        "upstream source does not match the registered DINOv2 commit",
    )
    require(
        not any(k == "dinov2" or k.startswith("dinov2.") for k in sys.modules),
        "start a fresh reference interpreter",
    )
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for name, raw in blobs.items():
            dst = root / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(raw)
        sys.path.insert(0, str(root))
        try:
            with patch.dict(os.environ, {"XFORMERS_DISABLED": "1"}):
                factory = importlib.import_module("dinov2.hub.backbones").dinov2_vits14
                require(
                    all(
                        Path(module.__file__).resolve().is_relative_to(root.resolve())
                        for key, module in sys.modules.items()
                        if (key == "dinov2" or key.startswith("dinov2."))
                        and getattr(module, "__file__", None)
                    ),
                    "official verifier imported a foreign implementation",
                )
                yield (
                    factory,
                    {
                        "repository": "https://github.com/facebookresearch/dinov2",
                        "commit": OFFICIAL,
                        "source_member_sha256_index": SOURCE_TREE,
                        "source_members": len(blobs),
                        "vision_transformer_sha256": hashes["dinov2/models/vision_transformer.py"],
                        "hub_backbones_sha256": hashes["dinov2/hub/backbones.py"],
                        "archive_sha256": None if source.is_dir() else digest(source),
                    },
                )
        finally:
            sys.path.pop(0)


def compared(actual, reference, atol=0.0, rtol=0.0):
    a = np.asarray(actual.detach().cpu() if isinstance(actual, torch.Tensor) else actual)
    b = np.asarray(reference.detach().cpu() if isinstance(reference, torch.Tensor) else reference)
    require(a.shape == b.shape and a.size > 0, "different numerical probe shapes or empty probe")
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "shape": list(a.shape),
        "scalars": a.size,
        "atol": atol,
        "rtol": rtol,
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "outside_tolerance": int(np.count_nonzero(~np.isfinite(delta) | (delta > atol + rtol * np.abs(b)))),
        "actual_sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest(),
        "reference_sha256": hashlib.sha256(b.tobytes(order="C")).hexdigest(),
    }


def probe_inputs(h, w, pre):
    yy, xx = np.indices((h, w), dtype=np.int32)
    rgb = np.stack(
        ((13 * xx + 3 * yy) % 239, (xx + 19 * yy) % 237, (5 * xx + yy * yy) % 241), axis=-1
    ).astype(np.float32) / np.float32(241)
    rgb[14:42, 14:42, 1] = np.nan
    rgb[-2, -3, 0] = np.nan
    raw = np.zeros((h, w, len(pre["columns"])), np.float32)
    raw[..., : pre["numeric_count"]] = pre["means"]
    raw[..., 0] += ((xx % 11) / 11).astype(np.float32)
    return rgb, raw, (xx + 7 * yy) % 31 != 0


def reference_input(rgb, rectangular):
    h, w = rgb.shape[:2]
    valid = np.isfinite(rgb).all(axis=2)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    if rectangular:
        eh, ew = 14 * ((h + 13) // 14), 14 * ((w + 13) // 14)
        hwc = np.zeros((eh, ew, 3), np.float32)
        hwc[:h, :w][valid] = (rgb[valid] - mean) / std
        observed = np.zeros((eh, ew), bool)
        observed[:h, :w] = valid
        encoded = torch.from_numpy(hwc).permute(2, 0, 1)[None]
        support = observed.reshape(eh // 14, 14, ew // 14, 14).sum(axis=(1, 3)) / 196.0
    else:
        require((h, w) == (256, 256), "legacy comparison uses the registered 256 square")
        plain = torch.from_numpy(np.where(valid[:, :, None], rgb, np.float32(0))).permute(2, 0, 1)[None]
        resized = F.interpolate(plain, size=(252, 252), mode="bilinear", align_corners=False, antialias=True)
        encoded = (resized - torch.from_numpy(mean)[None, :, None, None]) / torch.from_numpy(std)[
            None, :, None, None
        ]
        support = np.ones((18, 18), np.float64)
    return encoded, support, valid


def reference_sample(feature_map, support, indices, width, height, rectangular):
    """Double-precision grid_sample oracle, with pixel centres mapped directly into the normalized encoder."""
    ph, pw = support.shape
    x = (indices % width).astype(np.float64) + 0.5
    y = (indices // width).astype(np.float64) + 0.5
    divisor_x, divisor_y = (pw * 14, ph * 14) if rectangular else (width, height)
    grid = torch.from_numpy(np.stack((2 * x / divisor_x - 1, 2 * y / divisor_y - 1), axis=-1))[
        None, :, None, :
    ]
    mask = torch.from_numpy(support)[None, None]
    fraction = F.grid_sample(mask, grid, mode="bilinear", padding_mode="border", align_corners=False)[
        0, 0, :, 0
    ]
    numerator = F.grid_sample(
        feature_map.double() * mask, grid, mode="bilinear", padding_mode="border", align_corners=False
    )[0, :, :, 0].T
    return (numerator / fraction.clamp_min(1e-8)[:, None]).float(), fraction


def instrumented_score(info, rgb, raw, ok, rectangular):
    model = info[0]
    captured = {}
    original = F.interpolate

    def interpolate(x, *args, **kwargs):
        out = original(x, *args, **kwargs)
        if tuple(x.shape) == (1, 384, 37, 37):
            captured["position"] = torch.cat(
                (model.backbone.pos_embed[:, :1], out.flatten(2).transpose(1, 2)), 1
            )
        return out

    def take_input(key):
        def hook(module, args):
            captured[key] = args[0]

        return hook

    def take_output(key):
        def hook(module, args, value):
            captured[key] = value

        return hook

    with ExitStack() as stack:
        stack.enter_context(patch.object(F, "interpolate", interpolate))
        stack.enter_context(model.backbone.patch_embed.register_forward_pre_hook(take_input("image")))
        stack.enter_context(model.backbone.blocks[0].register_forward_pre_hook(take_input("input_tokens")))
        stack.enter_context(model.backbone.norm.register_forward_hook(take_output("all_tokens")))
        stack.enter_context(model.projection.register_forward_hook(take_output("projection")))
        scored = score_grid(info, rgb, raw, ok, "cpu", rectangular=rectangular)
    return scored, captured


def verify_case(info, reference_model, state, width, height, rectangular, tolerances):
    model, pre, idx = info
    rgb, raw, scoreable = probe_inputs(height, width, pre)
    encoded, support, finite_rgb = reference_input(rgb, rectangular)
    ph, pw = support.shape
    scored, observed = instrumented_score(info, rgb, raw, scoreable, rectangular)
    # Validate preprocessing separately, then give both backbones identical input storage and pixel values.
    encoded_check = compared(observed["image"], encoded)
    image = observed["image"]
    reference_tokens = reference_model.prepare_tokens_with_masks(image)
    position = reference_model.interpolate_pos_encoding(reference_tokens, 14 * ph, 14 * pw)
    features = reference_model.forward_features(image)
    patches = features["x_norm_patchtokens"]
    require(
        tuple(features["x_norm_regtokens"].shape) == (1, 0, 384) and patches.shape[1] == ph * pw,
        "expected one CLS followed by the rectangular patch lattice, with zero registers",
    )
    ref_projection = F.linear(patches, state["projection.weight"], state["projection.bias"])
    fmap = ref_projection.reshape(1, ph, pw, 32).permute(0, 3, 1, 2)
    tok_tol = (tolerances["backbone_atol"], tolerances["backbone_rtol"])
    out_tol = (
        tolerances["projected_sampling_and_final_risk_atol"],
        tolerances["projected_sampling_and_final_risk_rtol"],
    )
    geo_tol = tolerances["geometric_coordinates_and_support_atol"]
    checks = {
        "encoder_pixels": encoded_check,
        "positions": compared(observed["position"], position),
        "input_tokens_including_cls": compared(observed["input_tokens"], reference_tokens),
        "normalized_cls": compared(observed["all_tokens"][:, 0], features["x_norm_clstoken"], *tok_tol),
        "normalized_patches": compared(observed["all_tokens"][:, 1:], patches, *tok_tol),
        "projected_official_tokens": compared(observed["projection"], ref_projection, *tok_tol),
    }
    if rectangular:
        _, actual_support, frame, actual_observed = rectangular_image_tensor(rgb, "cpu")
        checks["patch_support"] = compared(actual_support[0, 0], support, geo_tol)
        require(np.array_equal(actual_observed, finite_rgb), "native RGB eligibility differs")
        coordinates = {
            (0, 0),
            (width - 1, 0),
            (0, height - 1),
            (width - 1, height - 1),
            (6, 7),
            (13, 14),
            (14, 13),
            (20, 20),
            (27, 27),
            (28, 27),
            (width - 2, height - 2),
            (width // 2, height // 2),
        }
        xy = np.asarray(sorted(coordinates), np.int64)
        probes = xy[:, 1] * width + xy[:, 0]
        x_actual, y_actual = frame.patch_coordinates(probes, torch.device("cpu"))
        checks["native_centres_to_xy_patch_centres"] = compared(
            torch.stack((x_actual, y_actual), 1), (xy + 0.5) / 14 - 0.5, geo_tol
        )
        anchors = np.array([(width - 1, 0), (0, 14), (width - 1, height - 1)], np.int64)
        ax, ay = frame.patch_coordinates(anchors[:, 1] * width + anchors[:, 0], torch.device("cpu"))
        actual_ids = torch.floor(ay + 0.5).long() * pw + torch.floor(ax + 0.5).long()
        expected_ids = HAND_PATCH_IDS[width, height]
        checks["hand_checked_flat_patch_indices"] = compared(actual_ids, expected_ids)
        transposed_ids = torch.floor(ax + 0.5).long() * ph + torch.floor(ay + 0.5).long()
        half_pixel_off = xy / 14 - 0.5
        negative = {
            "transposed_patch_indices_disagree": int(
                np.count_nonzero(transposed_ids.numpy() != expected_ids)
            ),
            "missing_half_pixel_disagreements": int(
                np.count_nonzero(
                    np.abs(half_pixel_off - torch.stack((x_actual, y_actual), 1).numpy()) > geo_tol
                )
            ),
        }
        require(
            all(v > 0 for v in negative.values()),
            "probes failed to detect erroneous axes or centre convention",
        )
        # Hold the map fixed. A transpose, wrong row stride or wrong partially supported neighbour changes this test.
        v, f = rectangular_sample(fmap, actual_support, frame, probes)
        r, rf = reference_sample(fmap, support, probes, width, height, True)
        checks["sampled_edge_vectors"] = compared(v, r, *out_tol)
        checks["sampled_edge_support"] = compared(f, rf, geo_tol)
        edge_receipt = [{"pixel_xy": [int(x), int(y)], "support": float(fr)} for (x, y), fr in zip(xy, rf)]
    else:
        edge_receipt = []
        expected_ids, negative = [], {}
    eligible = scoreable & finite_rgb
    ids = np.flatnonzero(eligible.ravel())
    expected_risk = np.empty(len(ids), np.float32)
    for start in range(0, len(ids), 8192):
        sub = ids[start : start + 8192]
        visual, fraction = reference_sample(fmap, support, sub, width, height, rectangular)
        require(torch.all(fraction > 0).item(), "independent valid native pixel lacks supported patches")
        numeric = raw.reshape(height * width, -1)[sub].astype(np.float64)
        n = pre["numeric_count"]
        numeric[:, :n] = (numeric[:, :n] - pre["means"]) / pre["scales"]
        joined = torch.cat((visual, torch.from_numpy(numeric.astype(np.float32))), 1)
        hidden = F.gelu(F.linear(joined, state["head.0.weight"], state["head.0.bias"]))
        logits = F.linear(hidden, state["head.2.weight"], state["head.2.bias"]).flatten()
        expected_risk[start : start + len(sub)] = torch.sigmoid(logits).numpy()
    actual_risk = scored["risk"].ravel()[ids]
    checks["native_probabilities"] = compared(actual_risk, expected_risk, *out_tol)
    expected_state = np.full((height, width), 255, np.uint8)
    expected_state.ravel()[ids] = expected_risk.astype(np.float64) >= idx["threshold"]
    mask_changes = int(np.count_nonzero(scored["valid"] != eligible))
    state_changes = int(np.count_nonzero(scored["state"] != expected_state))
    require(np.array_equal(np.isfinite(scored["risk"]), scored["valid"]), "risk and finite mask differ")
    return {
        "analysis_wh": [width, height],
        "mode": "native-rectangular" if rectangular else "legacy-letterbox",
        "encoder_wh": [pw * 14, ph * 14],
        "patch_grid_wh": [pw, ph],
        "cls_tokens": 1,
        "register_tokens": 0,
        "right_bottom_padding": [pw * 14 - width, ph * 14 - height] if rectangular else [0, 0],
        "canvas": height * width,
        "eligible": len(ids),
        "unknown": height * width - len(ids),
        "finite_mask_changes": mask_changes,
        "threshold_crossings": state_changes,
        "threshold": idx["threshold"],
        "edge_probes": edge_receipt,
        "checks": checks,
        "hand_flat_ids_northeast_next_row_southeast": expected_ids,
        "counterfactual_differences": negative,
        "passed": mask_changes == 0
        and state_changes == 0
        and all(q["outside_tolerance"] == 0 for q in checks.values()),
    }


def attention_operator_witness(reference, image):
    block = reference.blocks[0]
    tokens = block.norm1(reference.prepare_tokens_with_masks(image))
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            block.attn(tokens)
        return {
            "attention_operators": sorted(
                event.key for event in profiler.key_averages() if "attention" in event.key
            )
        }
    except (RuntimeError, NotImplementedError) as error:
        return {"profiler_unavailable": str(error)}


def run(args):
    require(os.environ.get("EL_RUN_ID"), "start the recorder before the neural acceptance stage")
    require(
        digest(args.protocol) == REGISTRATION and digest(args.models / "release.json") == CPU_RELEASE,
        "wrong registered Stage A protocol or CPU model distribution",
    )
    contract = read(args.protocol)
    tolerances = contract["neural"]
    args.out.mkdir(parents=True, exist_ok=True)
    require(not any(args.out.iterdir()), "neural acceptance needs an empty output directory")
    torch.set_num_threads(args.threads)
    result = {
        "schema": "el-v05-official-rectangular-neural-2",
        "run_id": os.environ["EL_RUN_ID"],
        "protocol_sha256": REGISTRATION,
        "cpu_release_sha256": CPU_RELEASE,
        "device": "cpu",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "methods": {},
        "risk_updates": 0,
        "reference_sampling": "Double-precision grid_sample on support-weighted official projection with border padding and align_corners=False; released head loaded independently.",
        "reference_dispatch": {
            "cases": "Meta reference and released scorer both use PyTorch SDPA MATH, the explicit backend of the released scorer.",
            "official_default_dispatch_cases": "Meta reference chooses its default CPU SDPA backend. The released scorer retains its explicit MATH backend.",
            "acceptance": "All five explicit v2 grids must pass with both reference dispatches. The v1 control is checked with matched MATH, and its full default-dispatch diagnostics are reported independently.",
        },
        "probe": "Deterministic asymmetric RGB and 39 numeric columns; finite mask excludes a known block and one partial-channel edge, separate eligibility mask.",
    }
    shapes = [(False, (256, 256)), *((True, wh) for wh in contract["raw_and_analysis"]["analysis_wh"])]
    with official_implementation(args.official_source) as (factory, source), torch.inference_mode():
        result["official_source"] = source
        for method, (model_sha, weights_sha) in MODELS.items():
            folder = args.models / "models" / method
            info = load_model(folder, model_sha)
            require(digest(folder / "weights.safetensors") == weights_sha, "wrong released FP32 tensors")
            state = load_file(str(folder / "weights.safetensors"), device="cpu")
            reference = factory(pretrained=False).cpu().eval()
            require(
                reference.num_register_tokens == 0
                and reference.register_tokens is None
                and reference.patch_size == 14
                and reference.interpolate_offset == 0.1
                and reference.interpolate_antialias is False,
                "unexpected official model settings",
            )
            backbone = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
            reference.load_state_dict(backbone, strict=True)
            require(
                all(torch.equal(backbone[k], t) for k, t in reference.state_dict().items()),
                "official backbone did not receive the original tensors",
            )

            def compare_shapes():
                return [
                    verify_case(info, reference, state, w, h, rect, tolerances) for rect, (w, h) in shapes
                ]

            dispatch_image = reference_input(probe_inputs(256, 256, info[1])[0], rectangular=False)[0]
            default_witness = attention_operator_witness(reference, dispatch_image)
            default_cases = compare_shapes()
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                math_witness = attention_operator_witness(reference, dispatch_image)
                cases = compare_shapes()
            twins = [r["checks"]["positions"] for r in cases if r["analysis_wh"] in ([375, 251], [378, 252])]
            same = (
                len(twins) == 2
                and twins[0]["actual_sha256"]
                == twins[1]["actual_sha256"]
                == twins[0]["reference_sha256"]
                == twins[1]["reference_sha256"]
            )
            result["methods"][method] = {
                "model_index_sha256": model_sha,
                "weights_sha256": weights_sha,
                "backbone_tensors": len(backbone),
                "padded_and_3x2_positions_identical": same,
                "cases": cases,
                "official_default_dispatch_cases": default_cases,
                "official_attention_witness": {"default": default_witness, "math": math_witness},
            }
    methods = list(result["methods"].values())
    result["default_dispatch_passed_on_all_cases"] = all(
        c["passed"] for m in methods for c in m["official_default_dispatch_cases"]
    )
    result["default_dispatch_passed_on_all_v2"] = all(
        c["passed"]
        for m in methods
        for c in m["official_default_dispatch_cases"]
        if c["mode"] == "native-rectangular"
    )
    result["matched_dispatch_passed_on_all_cases"] = all(c["passed"] for m in methods for c in m["cases"])
    result["passed"] = (
        result["matched_dispatch_passed_on_all_cases"]
        and result["default_dispatch_passed_on_all_v2"]
        and all(m["padded_and_3x2_positions_identical"] for m in methods)
    )
    put(args.out / "neural.json", result)
    require(result["passed"], "official reference comparison exceeded a registered tolerance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-source", type=Path, required=True, help="Pinned DINOv2 directory or commit tarball"
    )
    parser.add_argument(
        "--models", type=Path, required=True, help="Extracted original CPU distribution with release.json"
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
