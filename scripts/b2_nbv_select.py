"""Select two camera IDs from eight permitted poses and a released M2 model's own predictions."""

import argparse
import hashlib
import itertools
import os
import time
from pathlib import Path

import numpy as np
from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens._verify import require
from evidence_lens.multisource import extract
from evidence_lens.multisource_io import decode_png
from evidence_lens.multisource_model import light_risk, load_light
from evidence_lens.scoring_io import Camera


def checked(root, entry):
    path = root / entry["path"]
    require(
        path.is_file()
        and path.resolve().is_relative_to(root.resolve())
        and path.stat().st_size == entry["bytes"]
        and sha(path) == entry["sha256"],
        "changed permitted view member",
    )
    return path


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def camera(spec):
    return Camera(
        spec["width"], spec["height"], np.asarray(spec["K"]), np.asarray(spec["c2w"])
    )


def visible(xyz, cam, frame, scale, rules):
    local = (xyz - cam.c2w[:3, 3]) @ cam.c2w[:3, :3]
    z = local[:, 2]
    safe = np.where(z > 0, z, 1)
    uv = (local @ cam.K.T)[:, :2] / safe[:, None]
    xy = np.floor(uv + 0.5).astype(np.int64)
    valid = (xy >= 0).all(axis=1) & (xy[:, 0] < cam.width) & (xy[:, 1] < cam.height)
    x, y = np.clip(xy[:, 0], 0, cam.width - 1), np.clip(xy[:, 1], 0, cam.height - 1)
    ed, a = frame["z"][y, x], frame["alpha"][y, x]
    tol = np.maximum(
        rules["absolute_depth_tolerance_in_baselines"] * scale,
        rules["relative_depth_tolerance"] * z,
    )
    return (
        valid
        & (z >= rules["clip"][0])
        & (z <= rules["clip"][1])
        & np.isfinite(ed)
        & (ed > 0)
        & np.isfinite(a)
        & (a >= rules["minimum_candidate_alpha"])
        & (np.abs(ed - z) <= tol)
    )


def select_pair(utilities, tolerance=1e-12):
    best = max(utilities.values())
    return min(
        k for k, v in utilities.items() if best - v <= tolerance * max(1.0, abs(best))
    )


def flat(values):
    return max(values) - min(values) <= 1e-12 * max(1.0, max(abs(v) for v in values))


def choices(pool, points, weights, vis, axes, object_id, seeds):
    pairs = list(itertools.combinations(sorted(pool), 2))
    uniform, risk, separation = {}, {}, {}
    for pair in pairs:
        a, b = pair
        union = vis[a] | vis[b]
        uniform[pair] = int(union.sum())
        risk[pair] = float(weights[union].sum(dtype=np.float64))
        separation[pair] = float(np.arccos(np.clip(axes[a] @ axes[b], -1, 1)))
    reason = None
    if len(points) == 0:
        reason = "empty_support"
    elif float(weights.sum(dtype=np.float64)) <= 1e-12:
        reason = "zero_weight"
    elif flat(list(uniform.values())) or flat(list(risk.values())):
        reason = "constant_utilities"
    shared = select_pair(separation)
    result = {
        "uniform": shared if reason else select_pair(uniform),
        "risk": shared if reason else select_pair(risk),
    }
    for seed in seeds:
        result["random-" + str(seed)] = min(
            pairs,
            key=lambda p: (
                hashlib.sha256(
                    f"random|{seed}|{object_id}|{p[0]}|{p[1]}".encode()
                ).digest(),
                p,
            ),
        )
    rows = [
        {
            "pair": list(p),
            "uniform": uniform[p],
            "risk": risk[p],
            "axis_angle_rad": separation[p],
        }
        for p in pairs
    ]
    return result, reason, rows


def run(cfg, origin_path, product_path, allowed_cameras, model_dir, protocol, out):
    start = time.perf_counter()
    origin, rendered, camdoc = (
        load(origin_path),
        load(product_path),
        load(allowed_cameras),
    )
    assert_set = (
        protocol["development_objects"]
        if cfg["study_role"] == "development"
        else cfg["registered_test_objects"]
    )
    require(
        origin["object_id"] == rendered["object_id"] == cfg["object_id"]
        and origin["object_id"] in assert_set
        and origin["source_view_ids"] == {"s00": 24, "s03": 22}
        and origin["checkpoint_sha256"] == rendered["checkpoint_sha256"]
        and sha(allowed_cameras) == rendered["cameras_sha256"],
        "different reconstruction, source pair or query geometry",
    )
    root = origin_path.parent / "selector"
    pool = protocol["candidate_ids"]
    expected = {
        "s00",
        "s03",
        *[f"c{i:03}" for i in pool],
        *[f"q{i:02}" for i in range(4)],
    }
    require(
        expected == set(rendered["products"]) == set(camdoc["views"]),
        "unexpected input pose",
    )
    cams = {k: camera(v) for k, v in camdoc["views"].items()}
    sources = {}
    source_errors = {}
    for key, rkey in (("s00", "r00"), ("s03", "r03")):
        predicted = arrays(checked(product_path.parent, rendered["products"][key]))
        original = arrays(checked(root, origin["files"][rkey]))
        e = {}
        for k, x in original.items():
            require(
                np.array_equal(np.isfinite(x), np.isfinite(predicted[k])),
                "source native support mismatch",
            )
            finite = np.isfinite(x)
            e[k] = float(np.max(np.abs(x[finite] - predicted[k][finite]), initial=0))
            require(e[k] <= 1e-5, "re-render differs from the original M2 source frame")
        source_errors[key] = e
        sources[key] = (
            decode_png(checked(root, origin["files"][key]).read_bytes(), cams[key]),
            original,
            cams[key],
        )
    _, coeff, prep = load_light(model_dir, cfg["model_index_sha256"])
    grid = protocol["grid"]
    all_xyz, all_w, counts = [], [], {}
    features_started = time.perf_counter()
    for i in range(4):
        key = f"q{i:02}"
        cam = cams[key]
        f = arrays(checked(product_path.parent, rendered["products"][key]))
        risk = light_risk(extract(f, cam, sources, depth_verified=True), coeff, prep)
        z, a = f["z"], f["alpha"]
        y, x = np.mgrid[
            grid["offset"] : cam.height : grid["stride"],
            grid["offset"] : cam.width : grid["stride"],
        ]
        y, x = y.ravel(), x.ravel()
        good = (
            np.isfinite(risk[y, x])
            & np.isfinite(z[y, x])
            & (z[y, x] > 0)
            & (a[y, x] >= grid["minimum_query_alpha"])
        )
        counts[key] = {
            "canvas": int(risk.size),
            "unknown_full": int((~np.isfinite(risk)).sum()),
            "lattice": len(x),
            "unknown_lattice": int((~np.isfinite(risk[y, x])).sum()),
            "eligible_lattice": int(good.sum()),
        }
        y, x = y[good], x[good]
        local = (
            np.linalg.solve(cam.K, np.stack([x, y, np.ones(len(x))])).T * z[y, x, None]
        )
        all_xyz.append(local @ cam.c2w[:3, :3].T + cam.c2w[:3, 3])
        all_w.append(risk[y, x])
        np.savez_compressed(out / (key + "-risk.npz"), risk=risk)
    feature_seconds = time.perf_counter() - features_started
    xyz, weights = np.concatenate(all_xyz), np.concatenate(all_w)
    scale = rendered["scene_scale_m"]
    cells = np.floor(xyz / (scale * grid["voxel_size_in_endpoint_baselines"])).astype(
        np.int64
    )
    _, idx = np.unique(cells, axis=0, return_index=True)
    idx.sort()
    xyz, weights = xyz[idx], weights[idx]
    vis, axes = {}, {}
    for i in pool:
        key = f"c{i:03}"
        cam = cams[key]
        frame = arrays(checked(product_path.parent, rendered["products"][key]))
        require(
            set(frame) == {"alpha", "z"},
            "candidate observation is restricted to predicted visibility",
        )
        vis[i] = visible(
            xyz,
            cam,
            frame,
            scale,
            {**protocol["visibility"], "clip": camdoc["views"][key]["clip_m"]},
        )
        axes[i] = cam.c2w[:3, 2]
    result, reason, utilities = choices(
        pool,
        xyz,
        weights,
        vis,
        axes,
        origin["object_id"],
        protocol["random_selector_seeds"],
    )
    np.savez_compressed(
        out / "support.npz",
        xyz_m=xyz,
        weights=weights,
        **{f"c{i:03}": v for i, v in vis.items()},
    )
    common = {
        "run_id": os.environ["EL_RUN_ID"],
        "object_id": origin["object_id"],
        "pool": pool,
        "m2_checkpoint_sha256": origin["checkpoint_sha256"],
        "cameras_sha256": sha(allowed_cameras),
        "support_sha256": sha(out / "support.npz"),
        "protocol_fingerprint": hashlib.sha256(encoded(protocol)).hexdigest(),
        "assessed_query_points": len(xyz),
        "per_query": counts,
        "legal_photo_ids_before_selection": [24, 22],
        "model_index_sha256": cfg["model_index_sha256"],
    }
    for k, pair in result.items():
        save(
            out / (k + ".json"),
            {
                "schema": "el-b2-nbv-selection-1",
                **common,
                "strategy_id": k,
                "selected_official_ids": list(pair),
                "fallback_reason": reason if k in ("uniform", "risk") else None,
                "allocation": "both camera IDs sealed together before either new photo",
            },
        )
    save(
        out / "choices.json",
        {
            "schema": "el-b2-joint-nbv-choices-1",
            **common,
            "results": {k: list(v) for k, v in result.items()},
            "source_replay_max_abs": source_errors,
            "fallback_reason": reason,
            "pair_utilities": utilities,
            "point_count_before_dedup": len(cells),
            "feature_and_score_s": feature_seconds,
            "selection_stage_s": time.perf_counter() - start,
            "selections": {k: sha(out / (k + ".json")) for k in result},
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("config", "origin", "predictions", "cameras", "model", "protocol", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    run(
        load(a.config),
        a.origin,
        a.predictions,
        a.cameras,
        a.model,
        load(a.protocol),
        a.out,
    )
