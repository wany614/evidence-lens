"""GPU collection or independent CPU grading of the immutable phase-one known-geometry probe."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from phase1_geometry_oracle import centroid, original_fixture_rgb, sha
from PIL import Image


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_bytes())


def numpy(x):
    return x.detach().cpu().numpy()


def ply_image_metadata(meta):
    width, height = meta.resolution_px
    return {
        "resolution_wh": [int(width), int(height)],
        "f_px": float(meta.focal_length_px),
        "color_space": str(meta.color_space),
    }


def torch_gaussians(means, scales, wxyz, colors, opacity, device):
    import torch
    from sharp.utils.gaussians import Gaussians3D

    return Gaussians3D(
        *(
            torch.tensor(a, device=device, dtype=torch.float32).unsqueeze(0)
            for a in (means, scales, wxyz, colors, opacity)
        )
    )


def collect(spec, config, out):
    import gsplat
    import sharp
    import torch
    from plyfile import PlyData
    from sharp.cli.predict import predict_image
    from sharp.utils.gaussians import load_ply, save_ply

    import evidence_lens
    from evidence_lens import renderer_adapter
    from evidence_lens._worker import require_cuda, require_sharp

    cfg = read(config)
    device = require_cuda(cfg)
    assert require_sharp(cfg["generator"]["commit"]) == spec["sharp_commit"]
    seen = {}
    for group, root in (("sharp", sharp), ("gsplat", gsplat), ("evidence_lens", evidence_lens)):
        folder = Path(root.__file__).resolve().parent
        seen[group] = {p: sha(folder / p, True) for p in spec["normalized_source_sha256"][group]}
        if seen[group] != spec["normalized_source_sha256"][group]:
            raise ValueError(f"{group} executed source differs from the fixed source registry")
    runtime = {
        "sharp_commit": spec["sharp_commit"],
        "gsplat": importlib.metadata.version("gsplat"),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "recorded_run_id": os.environ.get("EL_RUN_ID"),
        "normalized_source_sha256": seen,
        "learned_checkpoint_forward_calls": 0,
    }
    write(out / "runtime.json", runtime)
    pts = spec["means_m"]
    physical = torch_gaussians(
        pts, spec["scales_m"], spec["wxyz"], spec["linear_rgb"], spec["opacity"], device
    )
    saved = {}
    angles = {}
    for w in spec["official_predict_image_witness"]:
        rgb = original_fixture_rgb(w["width"], w["height"])
        samples = {}

        def witness(image, disparity, profile=w, measurement=samples):
            measurement["shape"] = list(image.shape)
            measurement["f_over_w"] = float(disparity.item())
            measurement["sample_rgb"] = [
                numpy(image[0, :, s["internal_xy"][1], s["internal_xy"][0]]).tolist()
                for s in profile["samples"]
            ]
            return torch_gaussians(
                profile["ndc_means"],
                spec["scales_m"],
                spec["wxyz"],
                spec["linear_rgb"],
                spec["opacity"],
                device,
            )

        actual = predict_image(witness, rgb, w["f_px"], device)
        samples["means_m"] = numpy(actual.mean_vectors[0]).tolist()
        samples["source_rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
        angles[w["name"]] = samples
    write(out / "predict_image_witness.json", angles)

    def roundtrip(name, gs, cam):
        ply = out / (name + ".ply")
        save_ply(gs, float(cam["K"][0][0]), (cam["height"], cam["width"]), ply)
        restored, meta = load_ply(ply)
        raw = PlyData.read(ply)
        saved[name] = {
            "ply_sha256": sha(ply),
            **ply_image_metadata(meta),
            "header_K": raw["intrinsic"]["intrinsic"].reshape(3, 3).tolist(),
            "header_extrinsic": raw["extrinsic"]["extrinsic"].reshape(4, 4).tolist(),
            "xyz": numpy(restored.mean_vectors[0]).tolist(),
            "scales": numpy(restored.singular_values[0]).tolist(),
            "wxyz": numpy(restored.quaternions[0]).tolist(),
            "rgb": numpy(restored.colors[0]).tolist(),
            "alpha": numpy(restored.opacities.reshape(-1)).tolist(),
        }
        return restored.to(device)

    arrays = out / "samples"
    arrays.mkdir()

    def collect_view(gs, view, sc2w, name, d_mode=False):
        colour, alpha = renderer_adapter.render_geometry(
            gs, view, sc2w, "RGB+ED", cfg["renderer"]["eps2d"], device
        )
        record = {"alpha": numpy(alpha), "ed": numpy(colour[..., 3]), "linear_rgb": numpy(colour[..., :3])}
        if d_mode:
            accum, _ = renderer_adapter.render_geometry(
                gs, view, sc2w, "RGB+D", cfg["renderer"]["eps2d"], device
            )
            record["D"] = numpy(accum[..., 3])
        np.savez_compressed(arrays / (name + ".npz"), **record)

    for scene in spec["scenes"]:
        name = scene["name"]
        cams = scene["cameras"]
        source = cams["source"]
        s2w = np.asarray(source["c2w"])
        gs = roundtrip(name, physical, source)
        for i in range(len(pts)):
            singleton = type(gs)(*(arr[:, i : i + 1] for arr in gs))
            for view, c in cams.items():
                collect_view(singleton, c, s2w, f"{name}-{view}-{i}")
            if name == "asymmetric":
                wrong = copy.deepcopy(cams["oblique"])
                wrong["K"][0][2] -= 0.5
                wrong["K"][1][2] -= 0.5
                collect_view(singleton, wrong, s2w, f"missing-half-{i}")
    lay = spec["layered"]
    c = lay["camera"]
    k = c["K"]
    u, v = lay["uv"]
    centres = [[(u - k[0][2]) * z / k[0][0], (v - k[1][2]) * z / k[1][1], z] for z in lay["z_m"]]
    layered = torch_gaussians(
        centres, [[0.045] * 3, [0.083] * 3], [[1, 0, 0, 0]] * 2, lay["linear_rgb"], lay["opacity"], device
    )
    gs = roundtrip("layers", layered, c)
    collect_view(gs, c, np.array(c["c2w"]), "layers", True)
    write(out / "ply_roundtrip.json", saved)

    # Execute the actual production entry point and its exact postprocessing. Seven B0 poses share one true metric PLY.
    official = spec["scenes"][0]["cameras"]
    source = official["source"]
    cams = out / "cameras.json"
    png = out / "input.png"
    ply = out / "B0.ply"
    generation = out / "known-geometry.json"
    write(
        cams,
        {
            "schema": "el-camera-1",
            "axis_convention": "OpenCV x right, y down, z forward",
            "unit": "m",
            "pixel_center": "integer",
            "views": official,
        },
    )
    Image.fromarray(original_fixture_rgb(source["width"], source["height"])).save(png)
    write(
        generation,
        {
            "producer_run_id": "predeclared-known-metric-geometry",
            "camera_sha256": sha(cams),
            "input_sha256": sha(png),
            "ply_sha256": sha(ply),
        },
    )
    renderer_adapter.render(config, png, cams, ply, generation, out / "native")
    existing = read(out / "native/depth-probe.json")
    forced = dict(existing, off_axis_plane_passed=False)
    with patch.object(renderer_adapter, "depth_probe", return_value=forced):
        renderer_adapter.render(config, png, cams, ply, generation, out / "forced-unverified")


def grade(spec, folder):
    errors = []
    details = []
    eps = spec["tolerance"]

    def check(label, actual, expected, atol=0):
        a, b = np.asarray(actual), np.asarray(expected)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            gap = None
        else:
            gap = float(np.max(np.abs(a.astype(float) - b.astype(float)), initial=0))
        ok = gap is not None and gap <= atol
        if not ok:
            errors.append(label)
        details.append({"test": label, "max_abs_error": gap, "absolute_tolerance": atol, "passed": ok})

    def flag(label, yes):
        check(label, bool(yes), True)

    samples = folder / "samples"
    read_ply = read(folder / "ply_roundtrip.json")
    witness = read(folder / "predict_image_witness.json")
    runt = read(folder / "runtime.json")
    flag("pinned-source", runt["normalized_source_sha256"] == spec["normalized_source_sha256"])
    plys = {}
    for w in spec["official_predict_image_witness"]:
        n = w["name"]
        hit = witness[n]
        flag(
            n + "/declared-grid/image",
            hit["shape"] == [1, 3, 1536, 1536] and hit["source_rgb_sha256"] == w["native_rgb_sha256"],
        )
        check(n + "/original-f-over-W", hit["f_over_w"], w["disparity_factor"], eps["f_over_width"])
        check(
            n + "/endpoint-aligned-resize",
            hit["sample_rgb"],
            [a["rgb"] for a in w["samples"]],
            eps["preprocess"],
        )
        check(n + "/official-NDC-to-metric", hit["means_m"], spec["means_m"], eps["metric_m"])
    measured = []
    for scene in spec["scenes"]:
        n = scene["name"]
        c = scene["cameras"]["source"]
        p = read_ply[n]
        plys[n] = p["ply_sha256"]
        flag(n + "/actually-saved-ply", sha(folder / (n + ".ply")) == p["ply_sha256"])
        flag(
            n + "/metadata",
            p["color_space"] == "linearRGB" and p["resolution_wh"] == [c["width"], c["height"]],
        )
        check(n + "/fx", p["f_px"], c["K"][0][0], eps["metric_m"])
        edge = np.array(c["K"])
        edge[:2, 2] += 0.5
        check(n + "/header-K-edge", p["header_K"], edge, eps["metric_m"])
        check(n + "/header-pose", p["header_extrinsic"], np.eye(4))
        for prop, truth in (
            ("xyz", "means_m"),
            ("scales", "scales_m"),
            ("wxyz", "wxyz"),
            ("rgb", "linear_rgb"),
            ("alpha", "opacity"),
        ):
            check(n + "/roundtrip-" + prop, p[prop], spec[truth], eps["ply_field"])
        for view, expected in scene["expected"].items():
            for i, point in enumerate(expected):
                tag = f"{n}-{view}-{i}"
                with np.load(samples / (tag + ".npz"), allow_pickle=False) as data:
                    alpha, ed, linear = data["alpha"], data["ed"], data["linear_rgb"]
                    uv = centroid(alpha)
                    yy, xx = np.unravel_index(np.argmax(alpha), alpha.shape)
                    mask = alpha > 0
                    rgb = spec["linear_rgb"][i]
                    measured.append(
                        {
                            "case": tag,
                            "uv": uv.tolist(),
                            "expected_uv": point["uv"],
                            "peak_ED_m": float(ed[yy, xx]),
                            "expected_Z_m": point["z_m"],
                            "range_m": point["range_m"],
                            "projected_residual_xy": (uv - point["uv"]).tolist(),
                            "peak_alpha": float(alpha[yy, xx]),
                        }
                    )
                    check(tag + "/centre_px", uv, point["uv"], eps["centroid_px"])
                    check(tag + "/depth_m", ed[mask], np.full(mask.sum(), point["z_m"]), eps["metric_m"])
                    check(
                        tag + "/linear_premultiplied",
                        linear[mask],
                        alpha[mask, None] * rgb,
                        eps["linear_rgb"],
                    )
                    flag(
                        tag + "/finite-opacity",
                        np.all(np.isfinite(alpha)) and np.all((alpha >= 0) & (alpha <= 1)),
                    )
                if n == "asymmetric" and view == "oblique":
                    with np.load(samples / f"missing-half-{i}.npz", allow_pickle=False) as changed:
                        check(
                            tag + "/detect_missing_half",
                            centroid(changed["alpha"]) - point["uv"],
                            [-0.5, -0.5],
                            eps["centroid_px"],
                        )
    lay = spec["layered"]
    u, v = lay["uv"]
    exp = lay["expected"]
    with np.load(samples / "layers.npz", allow_pickle=False) as z:
        for prop, etol in (
            ("alpha", "opacity"),
            ("D", "metric_m"),
            ("ed", "metric_m"),
            ("linear_rgb", "linear_rgb"),
        ):
            check("layers/" + prop, z[prop][v, u], exp["ED" if prop == "ed" else prop], eps[etol])
        check("layers/empty-alpha", z["alpha"][0, 0], 0)
        observed = {k: z[k][v, u].tolist() for k in ("alpha", "D", "ed", "linear_rgb")}
    native = folder / "native"
    forced = folder / "forced-unverified"
    index = read(native / "prediction.json")
    old = read(native / "depth-probe.json")
    flag(
        "legacy/booleans",
        all(
            old[k] is True
            for k in (
                "off_axis_plane_passed",
                "alpha_normalization_passed",
                "translation_passed",
                "linear_blend_passed",
            )
        ),
    )
    check("legacy/empty_alpha", old["empty_pixel_alpha"], 0)
    for n, s in enumerate(old["samples"]):
        check(f"legacy/{n}/ED", s["ED"], 2.0, eps["metric_m"])
        check(f"legacy/{n}/alpha", s["alpha"], [0.4, 0.75][n], eps["opacity"])
        check(f"legacy/{n}/D", s["D"], 2 * s["alpha"], eps["metric_m"])
        check(f"legacy/{n}/linear", s["red"], 0.25 * s["alpha"], eps["linear_rgb"])
    check("legacy/moved-alpha", old["translated_probe_alpha"], 0.4, eps["opacity"])
    for view in spec["scenes"][0]["cameras"]:
        nf = native / (view + ".npz")
        ff = forced / (view + ".npz")
        flag(view + "/native-view-sha", sha(nf) == index["views"][view]["sha256"])
        with np.load(nf, allow_pickle=False) as z, np.load(ff, allow_pickle=False) as f:
            a, rgb, ed = z["alpha"], z["rgb"], z["z"]
            covered = a > 0
            flag(view + "/covered+background", covered.any() and (~covered).any())
            flag(
                view + "/native-depth-availability",
                np.isfinite(ed[covered]).all() and np.isnan(ed[~covered]).all(),
            )
            flag(
                view + "/forced-gate",
                np.isnan(f["z"]).all() and np.array_equal(f["rgb"], rgb) and np.array_equal(f["alpha"], a),
            )
            flag(view + "/black-background", np.all(rgb[~covered] == 0))
            for i, pt in enumerate(spec["scenes"][0]["expected"][view]):
                x, y = np.rint(pt["uv"]).astype(int)
                check(f"{view}/{i}/native-ED", ed[y, x], pt["z_m"], eps["metric_m"])
                raw = a[y, x] * np.array(spec["linear_rgb"][i])
                display = np.where(raw <= 0.0031308, 12.92 * raw, 1.055 * raw ** (1 / 2.4) - 0.055)
                check(f"{view}/{i}/display-sRGB", rgb[y, x], display, eps["colour_srgb"])
    return {
        "schema": "el-phase1-geometry-readback-1",
        "passed": not errors,
        "failed_tests": errors,
        "checks": details,
        "centres_and_depth": measured,
        "layered": observed,
        "ply_sha256": plys,
        "neural_checkpoint_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gpu", "grade"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--collected", type=Path)
    a = parser.parse_args()
    if sha(a.protocol) != a.protocol_sha256:
        raise ValueError("incorrect pre-GPU protocol digest")
    spec = read(a.protocol)
    if a.mode == "gpu":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        if not os.environ.get("EL_RUN_ID"):
            raise ValueError("run the GPU worker under the existing recorder")
        if not a.config or sha(a.config) != spec["baseline_config_sha256"]:
            raise ValueError("baseline config digest differs")
        a.out.mkdir(parents=True, exist_ok=True)
        if any(a.out.iterdir()):
            raise FileExistsError("choose a new or empty recorder payload")
        collect(spec, a.config, a.out)
        result = grade(spec, a.out)
        dest = a.out / "gpu-readback.json"
    else:
        if not a.collected or a.out.exists():
            raise ValueError("supply a collected folder and a new readback JSON path")
        result = grade(spec, a.collected)
        dest = a.out
    result["protocol_sha256"] = a.protocol_sha256
    write(dest, result)
    print(
        json.dumps(
            {"passed": result["passed"], "tests": len(result["checks"]), "failed": result["failed_tests"]}
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
