"""Evaluate the two registered bell frusta after all 48 deployment scores have been sealed."""

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evidence_lens._bundle import load, save, sha
from evidence_lens.multisource_io import decode_prediction
from evidence_lens_bundle.bindings import array_sha
from evidence_lens_bundle.bundle import Bundle
from evidence_lens_bundle.common import json_bytes, parts, profile, require, sha as sha_bytes
from evidence_lens_bundle.grids import crop_transform
from evidence_lens_bundle.numerics import METHODS, checked_arrays
from evidence_lens_bundle.scoring import REASON_NAMES
from v05_bell_common import CHECKPOINT, OLD, SOURCES, TARGETS

ROUTES = ("landscape", OLD)
COMMON = (256, 144)
CANVASES = {"landscape": (448, 252), OLD: (256, 256)}
QUOTAS = (10, 20)
PROGRAM = "a-bell-evaluation-program"
RESULT = "a-bell-landscape-comparison"
PROCEDURE = {
    "object": "scene_057_offset_bell_cage",
    "status": "descriptive engineering comparison on one previously exposed authored object, six fixed viewpoints",
    "pixel_label": "mean of three absolute FP32 display-sRGB prediction-minus-reference differences > FP32(0.05)",
    "photometry": "use original 896x504 FP32 display-sRGB target measurements; separable area average on pixel edges in float64, then cast to FP32; all input RGB in a reduced footprint must be finite",
    "independent_labels": "form an error label from each route's own rendered RGB against the same independent measurement; labels can differ between routes",
    "native_content": "all 448x252 native pixels; rows 56 through 199 inclusive from the 256x256 legacy score and prediction",
    "aligned_content": "same-frustum 256x144; area-reduce native prediction and measured image directly; use the cropped legacy prediction without resampling",
    "score_sampling": "same-frustum risk, finite/state/reason use one nearest source centre per axis, floor((output_index+0.5)*input_size/output_size); no averaging of categories or scores",
    "eligible": "per-method finite & labelled; also within-route four-method intersection; aligned panel adds per-method two-route and full eight-path intersections with both labels available",
    "quota": "per view floor(q * eligible / 100), q=10 or 20; ascending risk then flattened row-major pixel index; summation over the six separate selections",
    "resources": "one four-thread file scoring invocation per route and model over six views; CPU time, wall time and process high-water memory recorded by the installed replay worker",
    "example": "first registered view v01 and MS-ft, aligned 256x144; highlight 20% of the joint eight-path finite labelled domain",
}


def _area_axis(array, length, axis):
    """Integrate a piecewise-constant pixel signal over each exact output footprint."""
    a = np.moveaxis(array, axis, 0)
    size = a.shape[0]
    require(0 < length <= size, "BOX evaluation only admits a downsample")
    if size == length:
        return array
    bounds = np.arange(length + 1, dtype=np.int64) * size
    index, remainder = np.divmod(bounds, length)
    zeros = np.zeros((1, *a.shape[1:]), dtype=np.float64)
    integral = np.concatenate((zeros, np.cumsum(a, axis=0, dtype=np.float64)))
    fraction = (remainder / length).reshape((-1,) + (1,) * (a.ndim - 1))
    cumulative = integral[index] + a[np.minimum(index, size - 1)] * fraction
    return np.moveaxis(np.diff(cumulative, axis=0) * (length / size), 0, axis)


def area_rgb(image, size):
    """Box area reduction in display sRGB. Any nonfinite RGB input invalidates its footprint."""
    w, h = size
    require(image.dtype == np.float32 and image.ndim == 3 and image.shape[-1] == 3, "expected FP32 RGB")
    if image.shape[:2] == (h, w):
        return image
    good = np.isfinite(image).all(axis=-1)
    data = np.where(good[..., None], image, 0).astype(np.float64)
    result = _area_axis(_area_axis(data, w, 1), h, 0).astype(np.float32)
    if not np.all(good):
        missing = _area_axis(_area_axis((~good).astype(np.float64), w, 1), h, 0)
        result[missing > 1e-12] = np.nan
    return result


def nearest(array, size):
    w, h = size
    ih, iw = array.shape[:2]
    x = ((2 * np.arange(w) + 1) * iw) // (2 * w)
    y = ((2 * np.arange(h) + 1) * ih) // (2 * h)
    return array[y[:, None], x]


def labels(predicted, measured):
    require(
        predicted.shape == measured.shape and predicted.dtype == measured.dtype == np.float32,
        "pixel labels require equal FP32 RGB grids",
    )
    error = np.mean(np.abs(predicted - measured), axis=-1, dtype=np.float32)
    return error, np.isfinite(error), error > np.float32(0.05)


def ranking(risk, eligible):
    ids = np.flatnonzero(eligible)
    return ids[np.lexsort((ids, risk.ravel()[ids]))]


def rate(n, d):
    return n / d if d else None


def population(error, valid_label, high, arrays):
    valid = arrays["valid"]
    n = int(valid_label.sum())
    positives = int(high[valid_label].sum())
    error_sum = float(error[valid_label].sum(dtype=np.float64))
    return {
        "canvas_pixels": valid.size,
        "reference_pixels": valid.size,
        "labelled_pixels": n,
        "prediction_rgb_unavailable": valid.size - n,
        "pixel_error_sum": error_sum,
        "high_error_pixels": positives,
        "mean_pixel_error": rate(error_sum, n),
        "error_rate": rate(positives, n),
        "finite_risk": int(valid.sum()),
        "unknown": int((~valid).sum()),
        **{
            "reason_" + reason: int(np.count_nonzero(arrays["unknown_reason"] == i))
            for i, reason in enumerate(REASON_NAMES)
            if i
        },
    }


def quota_rows(risk, eligible, high, domain, *, example=False):
    ordered = ranking(risk, eligible)
    total = len(ordered)
    positives = int(high.ravel()[ordered].sum())
    result, marked = [], None
    for q in QUOTAS:
        k = q * total // 100
        chosen = ordered[:k]
        nbad = int(high.ravel()[chosen].sum())
        result.append(
            {
                "domain": domain,
                "quota_percent": q,
                "canvas_pixels": eligible.size,
                "eligible_pixels": total,
                "eligible_errors": positives,
                "eligible_error_rate": rate(positives, total),
                "selected_pixels": k,
                "selected_errors": nbad,
                "selected_error_rate": rate(nbad, k),
                "selected_share_of_canvas": rate(k, eligible.size),
                "selected_share_of_eligible": rate(k, total),
                "selected_row_major_indices_sha256": array_sha(np.sort(chosen).astype("<i8")),
            }
        )
        if example and q == 20:
            marked = np.zeros(eligible.shape, bool)
            marked.ravel()[chosen] = True
    return result, marked


def combined(rows, group, additive, fractions):
    totals = {}
    for row in rows:
        key = tuple(row[k] for k in group)
        accumulator = totals.setdefault(key, {k: 0 for k in additive})
        for name in additive:
            accumulator[name] += row[name]
    return [
        {
            **dict(zip(group, key)),
            "views": len(TARGETS),
            **counts,
            **{name: rate(counts[a], counts[b]) for name, (a, b) in fractions.items()},
        }
        for key, counts in totals.items()
    ]


def write_csv(path, rows):
    require(bool(rows), "empty registered output table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        table = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        table.writeheader()
        table.writerows(rows)


def unique_suffix(z, suffix):
    matches = [
        i.filename
        for i in z.infolist()
        if not i.is_dir() and (i.filename == suffix or i.filename.endswith("/" + suffix))
    ]
    require(len(matches) == 1, "missing or duplicate registered archive member " + suffix)
    parts(matches[0])
    return matches[0]


def read_zipped(z, base, ref):
    parts(ref["path"])
    name = (PurePosixPath(base) / ref["path"]).as_posix()
    data = z.read(name)
    require(len(data) == ref["bytes"] and sha_bytes(data) == ref["sha256"], "changed archive member " + name)
    return data


def load_bundle(z, scratch, tag):
    name = unique_suffix(z, tag + "/bundle.json")
    base = PurePosixPath(name).parent
    folder = scratch / tag
    folder.mkdir()
    raw = z.read(name)
    (folder / "bundle.json").write_bytes(raw)
    doc = json_bytes(raw)
    for role in ("camera", "producer"):
        ref = doc[role]
        require(
            ref["path"] == ("cameras.json" if role == "camera" else "producer.json"),
            "unexpected Bell bundle metadata member",
        )
        (folder / ref["path"]).write_bytes(read_zipped(z, base, ref))
    bundle = Bundle(folder, sha_bytes(raw))
    require(
        list(bundle.order) == SOURCES
        and list(bundle.targets) == TARGETS
        and bundle.condition == "M4"
        and bundle.analysis_size == CANVASES[tag]
        and bundle.producer["checkpoint_sha256"] == CHECKPOINT
        and bundle.analysis_profile == ("analysis-448x252-v1" if tag == "landscape" else None),
        "wrong Bell score input or profile",
    )
    return bundle, base


def load_scores(z, accepted, bundle, tag, method):
    (grade,) = [g for g in accepted["cells"] if (g["method"], g["tag"]) == (method, tag)]
    name = unique_suffix(z, f"{tag}/{method}/scores/scores.json")
    raw = z.read(name)
    scores = json_bytes(raw)
    require(
        grade["scores_sha256"] == sha_bytes(raw)
        and grade["bundle_sha256"] == bundle.sha256
        and scores["bundle_sha256"] == bundle.sha256
        and scores["method"] == method
        and scores["model"] == profile()["models"][method] == grade["model"]
        and scores["checkpoint_sha256"] == CHECKPOINT
        and scores["source_order"] == SOURCES
        and scores["target_order"] == TARGETS == [v["id"] for v in scores["views"]]
        and scores["camera_sha256"] == bundle.meta["camera"]["sha256"]
        and scores["condition"] == "M4"
        and not (scores["calibration_profile"] or {}).get("applied"),
        "deployment scores, model, camera or input bundle differ",
    )
    if bundle.versioned_grid:
        require(
            scores["grids"] == bundle.grids and scores["grids_sha256"] == bundle.producer["grids_sha256"],
            "different native grid provenance",
        )
    require(
        scores["schema"] == ("el-bundle-scores-2" if bundle.versioned_grid else "el-bundle-scores-1")
        and grade["views"].keys() == set(TARGETS),
        "different score schema or registered views",
    )
    return scores, grade, PurePosixPath(name).parent


def verify_frustum(camera, bundles, view):
    require(
        (camera["width"], camera["height"]) == (896, 504)
        and camera["projection"] == "pinhole"
        and camera["distortion"] == "none",
        "wrong independent target original camera",
    )
    expected = crop_transform((0, 0, 896, 504), COMMON) @ np.array(camera["K"])
    origin = bundles["landscape"][0].grids["views"][view]
    require(
        origin["original_size"] == [896, 504]
        and origin["crop_xywh"] == [0, 0, 896, 504]
        and np.array_equal(origin["original_K"], camera["K"]),
        "native analysis has a different original camera or crop",
    )
    errors = {}
    for tag, crop in (("landscape", [0, 0, 448, 252]), (OLD, [0, 56, 256, 144])):
        cam = bundles[tag][0].cameras[view]
        mapped = crop_transform(crop, COMMON) @ cam.K
        errors[tag] = float(np.abs(expected - mapped).max())
        require(
            np.array_equal(cam.c2w, camera["c2w"]) and errors[tag] <= 1e-6,
            "comparison cameras do not describe the same physical frustum",
        )
    return {"common_K": expected.tolist(), "max_K_residual_pixels": errors, "c2w": camera["c2w"]}


def score_view(z, cell, view_i, prediction, size):
    scores, grade, base = cell
    row = scores["views"][view_i]
    view = TARGETS[view_i]
    require(
        row["render"] == prediction
        and (row["width"], row["height"]) == size
        and grade["views"][view]["arrays_sha256"] == row["arrays"]["sha256"],
        "wrong registered scored view",
    )
    with np.load(io.BytesIO(read_zipped(z, base, row["arrays"])), allow_pickle=False) as archive:
        require(len(archive.files) == 4, "registered score channel count differs")
        arrays = checked_arrays(dict(archive), scores["model"]["threshold"], grid_shape=size[::-1])
    expect = grade["views"][view]
    digests = {k: array_sha(val) for k, val in arrays.items()}
    require(
        all(
            expect[str(cap)]["exact"] is True and expect[str(cap)]["array_digests"] == digests
            for cap in (0, 64 << 20)
        ),
        "replay worker four-channel parity receipt differs",
    )
    return arrays


def rgb8(data):
    return np.rint(np.clip(np.nan_to_num(data, nan=0), 0, 1) * 255).astype(np.uint8)


def panel(out, reference, example):
    width, height = COMMON
    x0, gap = 126, 12
    sheet = Image.new("RGB", (x0 + 4 * (width + gap), 451), "#f7f6f2")
    draw = ImageDraw.Draw(sheet)
    heading, body = ImageFont.load_default(size=22), ImageFont.load_default(size=15)
    draw.text(
        (12, 10),
        "Bell cage   |   Fixed target v01   |   Shared 256 x 144 field of view",
        fill="#202724",
        font=heading,
    )
    for j, title in enumerate(
        ("Independent reference", "Gaussian prediction", "Per-pixel error > 0.05", "Selected by MS-ft at 20%")
    ):
        draw.text((x0 + j * (width + gap), 51), title, font=body, fill="#202724")
    for i, tag in enumerate(ROUTES):
        y = 79 + i * (height + 15)
        pred, high, valid, finite_risk, selected = example[tag]
        measured = np.full((*high.shape, 3), (225, 229, 222), dtype=np.uint8)
        measured[high] = (167, 61, 37)
        measured[~valid] = (112, 112, 112)
        marked = rgb8(pred)
        marked[~selected] = np.rint(marked[~selected].astype(np.float32) * 0.24).astype(np.uint8)
        marked[~finite_risk] = (105, 126, 145)
        draw.text((12, y + 57), "Native 448" if i == 0 else "Legacy 256", fill="#202724", font=body)
        for j, pixels in enumerate((rgb8(reference), rgb8(pred), measured, marked)):
            sheet.paste(Image.fromarray(pixels), (x0 + j * (width + gap), y))
    draw.text(
        (12, 405),
        "Rust = RGB error above cutoff. Gray = unavailable RGB. Bright = selected. Blue-gray = unknown risk.",
        fill="#343c37",
        font=body,
    )
    draw.text(
        (12, 428),
        "Original authored scene and derived photography, CC BY 4.0. One of six registered views.",
        fill="#343c37",
        font=body,
    )
    sheet.save(out / "bell-comparison.png")


def read_reference(path, spec):
    require(
        path.stat().st_size == spec["bytes"] and sha(path) == spec["sha256"],
        "independent FP32 reference identity differs",
    )
    with np.load(path, allow_pickle=False) as file:
        require(file.files == ["rgb"], "unrecognized reference array")
        rgb = file["rgb"]
    require(
        rgb.dtype == np.float32
        and rgb.shape == (504, 896, 3)
        and np.isfinite(rgb).all()
        and np.min(rgb) >= 0
        and np.max(rgb) <= 1,
        "expected a finite original display-sRGB reference",
    )
    return rgb


def route_products(predictions, scores, bundles, cells, tag, i, measured):
    view = TARGETS[i]
    bundle, base = bundles[tag]
    terminal = bundle.producer["frames"][view]
    shape = bundle.cameras[view].shape
    if terminal["status"] == "ready":
        prediction = decode_prediction(
            read_zipped(predictions, base, terminal["file"]), bundle.cameras[view]
        )["rgb"]
    else:
        prediction = np.full((*shape, 3), np.nan, np.float32)
    scores_full = {m: score_view(scores, cells[tag, m], i, terminal, bundle.analysis_size) for m in METHODS}
    active = slice(56, 200) if tag == OLD else slice(None)
    rgb = prediction[active]
    content_scores = {m: {k: a[active] for k, a in channels.items()} for m, channels in scores_full.items()}
    ref_own = area_rgb(measured, (rgb.shape[1], rgb.shape[0]))
    ref_shared = area_rgb(measured, COMMON)
    native = (rgb, labels(rgb, ref_own), content_scores)
    small = area_rgb(rgb, COMMON)
    sampled = {m: {k: nearest(a, COMMON) for k, a in values.items()} for m, values in content_scores.items()}
    aligned = (small, labels(small, ref_shared), sampled)
    evidence = {
        "bundle_sha256": bundle.sha256,
        "camera": bundle.meta["camera"],
        "producer": bundle.meta["producer"],
        "checkpoint_sha256": CHECKPOINT,
        "terminal": terminal,
        "analysis_wh": bundle.analysis_size,
        "original_prediction_array_sha256": array_sha(prediction),
        "content_rgb_sha256": array_sha(rgb),
        "aligned_rgb_sha256": array_sha(small),
        "methods": {
            m: {
                "scores_sha256": cells[tag, m][1]["scores_sha256"],
                "arrays": cells[tag, m][0]["views"][i]["arrays"],
                "entire_analysis_canvas": scores_full[m]["valid"].size,
                "entire_analysis_unknown": int((~scores_full[m]["valid"]).sum()),
                "view_score_timing": cells[tag, m][0]["views"][i]["timing"],
            }
            for m in METHODS
        },
    }
    return {"own_content": native, "aligned_256x144": aligned}, evidence


def analyse_view(view, products, base_rows, quota_table):
    example = {}
    for plane in ("own_content", "aligned_256x144"):
        paths = {tag: products[tag][plane] for tag in ROUTES}
        within = {
            tag: np.logical_and.reduce([paths[tag][2][m]["valid"] for m in METHODS]) & paths[tag][1][1]
            for tag in ROUTES
        }
        joint = np.logical_and(within[ROUTES[0]], within[ROUTES[1]]) if plane == "aligned_256x144" else None
        for tag in ROUTES:
            pred, (error, can_label, high), scores = paths[tag]
            for method, channels in scores.items():
                keys = {"view": view, "plane": plane, "route": tag, "method": method}
                base_rows.append({**keys, **population(error, can_label, high, channels)})
                own = channels["valid"] & can_label
                domains = {"method_finite": own, "four_method_finite": within[tag]}
                if joint is not None:
                    other = paths[OLD if tag == "landscape" else "landscape"]
                    domains.update(
                        {
                            "paired_method_finite": own & other[2][method]["valid"] & other[1][1],
                            "joint_eight_finite": joint,
                        }
                    )
                for name, eligible in domains.items():
                    show = view == TARGETS[0] and name == "joint_eight_finite" and method == "MS-ft"
                    rows, marked = quota_rows(channels["risk"], eligible, high, name, example=show)
                    quota_table.extend([{**keys, **r} for r in rows])
                    if show:
                        example[tag] = (pred, high, can_label, channels["valid"], marked)
    return example


def resource_rows(accepted, cells):
    result = []
    for (tag, method), (doc, grade, _) in cells.items():
        cost = grade["cost"]
        require(
            all(
                isinstance(cost.get(key), (int, float)) and math.isfinite(cost[key]) and cost[key] > 0
                for key in (
                    "file_score_wall_s",
                    "file_score_cpu_s",
                    "peak_rss_bytes_at_file_completion",
                    "peak_rss_bytes_process_lifetime",
                )
            ),
            "measured wall, CPU and RSS required",
        )
        require(
            cost["peak_rss_bytes_process_lifetime"] >= cost["peak_rss_bytes_at_file_completion"],
            "process RSS high-water moved backwards",
        )
        result.append(
            {
                "route": tag,
                "method": method,
                "views": len(TARGETS),
                "host": accepted["platform"],
                "configured_neural_threads": doc["cost"]["configured_neural_threads"],
                **cost,
                "file_mean_wall_s_per_view_including_model_load": cost["file_score_wall_s"] / len(TARGETS),
                "load_seconds": doc["cost"]["bundle_and_model_load_seconds"],
                "source_decode_seconds": doc["cost"]["source_decode_seconds"],
            }
        )
    return result


def opening_gate(op, registration):
    require(
        op.get("schema") == "el-v05-bell-reference-opening-1"
        and op.get("registered_scores_sealed") is True
        and op.get("terminal_score_count") == 48
        and op.get("targets") == TARGETS
        and op.get("conditions") == list(ROUTES)
        and op.get("registration") == registration,
        "complete registered two-route score opening required",
    )
    for role in ("bundles_archive", "scores_archive"):
        pin = op.get(role)
        require(
            isinstance(pin, dict)
            and set(pin) == {"run_id", "manifest_sha256", "member", "sha256"}
            and pin["member"].endswith(".zip"),
            "opening does not bind the " + role,
        )


def program_identity():
    from evidence_lens_bundle import grids, numerics

    return {
        p.name: sha(p)
        for p in (
            Path(__file__),
            Path(__file__).with_name("v05_bell_common.py"),
            Path(grids.__file__),
            Path(numerics.__file__),
        )
    }


def run(inputs, cfg, out):
    pins = cfg["sources"]
    for key in ("program", "registration", "opening", "bundles_archive", "scores_archive"):
        require(sha(inputs[key]) == pins[key]["sha256"], "changed evaluator parent " + key)
    opening = load(inputs["opening"])
    opening_gate(opening, pins["registration"])
    require(
        all(opening[k] == pins[k] for k in ("bundles_archive", "scores_archive")),
        "sealed score or prediction archive differs from opening",
    )
    program = load(inputs["program"])
    require(
        program["operation"] == PROCEDURE
        and program["code_sha256"] == program_identity()
        and program["registration"] == pins["registration"],
        "frozen evaluator or registration differs",
    )
    require(sha(inputs["references"]) == pins["references"]["sha256"], "changed references receipt")
    refs = load(inputs["references"])
    require(
        refs["schema"] == "el-v05-bell-photos-1"
        and refs["role"] == "references"
        and refs["opening"] == pins["opening"]
        and refs["registration"] == pins["registration"]
        and refs["source_renders"] == 0
        and refs["target_reference_renders"] == len(TARGETS)
        and set(refs["products"]) == {"landscape"}
        and set(refs["products"]["landscape"]) == set(TARGETS),
        "source of independent measurements differs",
    )
    provenance, rates, coverage = [], [], []
    with (
        TemporaryDirectory() as tmp,
        ZipFile(inputs["bundles_archive"]) as deployment,
        ZipFile(inputs["scores_archive"]) as score_zip,
    ):
        for file in (deployment, score_zip):
            require(len(file.namelist()) == len(set(file.namelist())), "duplicate ZIP member")
        bundles = {tag: load_bundle(deployment, Path(tmp), tag) for tag in ROUTES}
        accepted_raw = score_zip.read(unique_suffix(score_zip, "acceptance.json"))
        accepted = json_bytes(accepted_raw)
        require(
            accepted["schema"] == "el-v05-installed-grid-replay-1"
            and set(accepted["methods"]) == set(METHODS)
            and accepted["views_per_cell"] == len(TARGETS),
            "different acceptance or method roster",
        )
        cells = {
            (tag, method): load_scores(score_zip, accepted, bundles[tag][0], tag, method)
            for tag in ROUTES
            for method in METHODS
        }
        resources = resource_rows(accepted, cells)
        for i, view in enumerate(TARGETS):
            ref = refs["products"]["landscape"][view]
            require(
                ref["view"] == view
                and ref["display_fp32"]["path"] == f"landscape/{view}-display-fp32.npz"
                and pins["reference_" + view]["sha256"] == ref["display_fp32"]["sha256"]
                and ref["license"] == "CC-BY-4.0",
                "different independent target identity or rights",
            )
            geometry = verify_frustum(ref["camera"], bundles, view)
            measured = read_reference(inputs["reference_" + view], ref["display_fp32"])
            products, evidence = {}, {}
            for tag in ROUTES:
                products[tag], evidence[tag] = route_products(
                    deployment, score_zip, bundles, cells, tag, i, measured
                )
            example = analyse_view(view, products, rates, coverage)
            if example:
                panel(out, area_rgb(measured, COMMON), example)
            provenance.append(
                {
                    "view": view,
                    "reference": ref,
                    "reference_array_sha256": array_sha(measured),
                    "frustum": geometry,
                    "deployment": evidence,
                }
            )

    dimensions = ("plane", "route", "method")
    counts = (
        "canvas_pixels",
        "reference_pixels",
        "labelled_pixels",
        "prediction_rgb_unavailable",
        "pixel_error_sum",
        "high_error_pixels",
        "finite_risk",
        "unknown",
        *("reason_" + item for item in REASON_NAMES[1:]),
    )
    aggregate = combined(
        rates,
        dimensions,
        counts,
        {
            "error_rate": ("high_error_pixels", "labelled_pixels"),
            "mean_pixel_error": ("pixel_error_sum", "labelled_pixels"),
        },
    )
    grouped = combined(
        coverage,
        (*dimensions, "domain", "quota_percent"),
        ("canvas_pixels", "eligible_pixels", "eligible_errors", "selected_pixels", "selected_errors"),
        {
            "eligible_error_rate": ("eligible_errors", "eligible_pixels"),
            "selected_error_rate": ("selected_errors", "selected_pixels"),
            "selected_share_of_canvas": ("selected_pixels", "canvas_pixels"),
            "selected_share_of_eligible": ("selected_pixels", "eligible_pixels"),
        },
    )
    tables = {
        "per-view-populations.csv": rates,
        "per-view-coverage.csv": coverage,
        "populations.csv": aggregate,
        "coverage.csv": grouped,
        "scoring-resources.csv": resources,
    }
    for name, values in tables.items():
        write_csv(out / name, values)
    public = [{"member": name, "sha256": sha(out / name), "rows": len(rows)} for name, rows in tables.items()]
    public.append({"member": "bell-comparison.png", "sha256": sha(out / "bell-comparison.png")})
    save(
        out / "bell-comparison.json",
        {
            "schema": "el-v05-bell-landscape-comparison-1",
            "run_id": os.environ["EL_RUN_ID"],
            "procedure": PROCEDURE,
            "parents": pins,
            "reference_world": refs["world"],
            "reference_camera_origin": refs["camera_origin"],
            "independent_measurement": {
                k: refs[k] for k in ("bpy", "cycles_device", "cycles_samples", "seed")
            },
            "installed_acceptance_sha256": sha_bytes(accepted_raw),
            "host": accepted["platform"],
            "wheel_version": accepted["wheel_version"],
            "resource_scope": "RSS at file completion is the process high-water through validation, model load, decode and file scoring; lifetime also includes HTML and both resident cache modes",
            "labelled_baselines": aggregate,
            "registered_coverage": grouped,
            "scoring_cost": resources,
            "views": provenance,
            "files": public,
        },
    )


def control(store, action, retry):
    from dual_records import selected, stage
    from t0_records import bind, resolve

    pins = {"registration": selected(store, "a-bell-register-six")}
    if action == "freeze":
        pins["grid_registration"] = selected(store, "a-grid-registration")
        name, kind, member, estimate = PROGRAM, "archive_verify", "program.json", 12_000
        details = {"code_sha256": program_identity(), "operation": PROCEDURE}
    else:
        pins.update(program=selected(store, PROGRAM), opening=selected(store, "a-bell-reference-opening"))
        roots = (store, store / "history")
        opening = load(resolve(roots, pins["opening"]))
        opening_gate(opening, pins["registration"])
        pins.update({k: opening[k] for k in ("bundles_archive", "scores_archive")})
        pins["references"] = selected(store, "a-bell-references")
        ref = load(resolve(roots, pins["references"]))
        require(
            ref["opening"] == pins["opening"] and ref["role"] == "references",
            "references were not opened by this gate",
        )
        for view in TARGETS:
            item = ref["products"]["landscape"][view]["display_fp32"]
            require(item["path"] == f"landscape/{view}-display-fp32.npz", "unregistered reference path")
            pin = bind(store, pins["references"]["run_id"], "payload/" + item["path"])
            require(pin["sha256"] == item["sha256"], "reference member receipt mismatch")
            pins["reference_" + view] = pin
        name, kind, member, estimate = RESULT, "evaluation", "bell-comparison.json", 3_000_000
        details = {"operation": PROCEDURE}
    command = [
        "{python}",
        "-B",
        "{source}/scripts/v05_bell_evaluate.py",
        action + "-worker",
        "--config",
        "{config}",
        "--out",
        "{out}",
    ]
    for key in pins:
        command += ["--input", key, "{input:" + key + "}"]
    return stage(store, name, kind, pins, details, command, "payload/" + member, estimate, retry=retry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "evaluate", "freeze-worker", "evaluate-worker"))
    parser.add_argument("--store", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", action="append", nargs=2, default=[])
    parser.add_argument("--retry")
    args = parser.parse_args()
    if args.action.endswith("-worker"):
        config = load(args.config)
        if args.action == "freeze-worker":
            require(config["code_sha256"] == program_identity(), "changed evaluator at freeze")
            save(
                args.out / "program.json",
                {
                    "schema": "el-v05-bell-evaluation-program-1",
                    "run_id": os.environ["EL_RUN_ID"],
                    "registration": config["sources"]["registration"],
                    "grid_registration": config["sources"]["grid_registration"],
                    "code_sha256": program_identity(),
                    "operation": PROCEDURE,
                },
            )
        else:
            run({k: Path(v) for k, v in args.input}, config, args.out)
    else:
        print(json.dumps(control(args.store.resolve(), args.action, args.retry)))
