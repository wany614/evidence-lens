"""Independently evaluate the redistributable 512-square plane predictions and optional CPU risk scores."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

VIEWS = tuple(f"v{i:02}" for i in range(1, 7))
METHODS = {"constant": "constant", "1-alpha": "alpha_score", "L0": "l0", "L1-full": "l1_full",
           "L1-geometry": "l1_geometry", "L2-frozen": "l2_frozen", "L2-vision-ft": "l2_vision_ft"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_index(path, expected, schema):
    require(digest(path) == expected, "index checksum differs: " + path.name)
    record = json.loads(path.read_bytes())
    require(record["schema"] == schema, "unrecognized index: " + path.name)
    return record


def checked(root, name, index):
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink() and
            path.is_file() and index[name] == {"bytes": path.stat().st_size, "sha256": digest(path)},
            "member checksum or location differs: " + name)
    return path


def score_counts(path, v, e, positive):
    result = {}
    with np.load(path, allow_pickle=False) as data:
        for method, key in METHODS.items():
            risk, valid, state = (data[key + suffix] for suffix in ("", "_valid", "_state"))
            require(v.shape == risk.shape == valid.shape == state.shape and risk.dtype == np.float32
                    and valid.dtype == np.bool_ and state.dtype == np.uint8 and np.array_equal(valid, np.isfinite(risk))
                    and np.array_equal(valid, state != 255) and np.isin(state, [0, 1, 255]).all(),
                    "score, finite-support and state grids disagree for " + method)
            low, high, unknown = e & (state == 0), e & (state == 1), e & (state == 255)
            result[method] = {name: int(mask.sum()) for name, mask in (
                ("low_valid", v & (state == 0)), ("low_evaluable", low), ("low_errors", low & positive),
                ("warnings", high), ("warning_errors", high & positive), ("unknown_evaluable", unknown),
                ("unknown_errors", unknown & positive))}
    return result


def previews(out, actual, predicted, error):
    for name, rgb in (("reference-v01", actual), ("prediction-v01", predicted)):
        img = np.rint(255 * np.clip(np.nan_to_num(rgb), 0, 1)).astype(np.uint8)
        Image.fromarray(img).save(out / (name + ".png"))
    stops = [0, .025, .05, .10, .20]
    colors = np.array([(24, 32, 49), (46, 91, 153), (193, 102, 46), (244, 169, 79), (255, 242, 196)])
    vals = np.minimum(np.nan_to_num(error), stops[-1])
    img = np.stack([np.interp(vals, stops, colors[:, i]) for i in range(3)], axis=-1).round().astype(np.uint8)
    img[~np.isfinite(error)] = (220, 35, 171)
    Image.fromarray(img).save(out / "rgb-error-v01.png")


def evaluate(args):
    case = read_index(args.case / "case.json", args.case_sha256, "el-vision-public-adapter-case-1")
    ref = read_index(args.reference / "reference.json", args.reference_sha256, "el-vision-public-control-reference-1")
    require(case["family"] == ref["family"] == "scene_037_bottle_carrier" and
            case["condition"] == "public-plane" and case["generator"] == "authored-two-metre-plane"
            and case["width"] == case["height"] == 512, "wrong public control")
    required = {"input.png", "cameras.json", "predictions/prediction.json",
                *("predictions/" + v + ".npz" for v in ("source", *VIEWS))}
    require(set(case["files"]) == required and set(ref["files"]) == {v + ".npz" for v in VIEWS},
            "incomplete control or reference")
    for name in required:
        checked(args.case, name, case["files"])
    pred_meta = json.loads((args.case / "predictions/prediction.json").read_bytes())
    require(pred_meta["adapter"] == "evidence-lens-textured-plane-1" and pred_meta["depth_validation"]["off_axis_plane_passed"]
            and pred_meta["depth_validation"]["alpha_normalization_passed"] and pred_meta["depth_validation"]["empty_depth_proved"],
            "missing plane calibration")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), "use an empty evaluation destination")
    rows = {}
    for view in VIEWS:
        orig = checked(args.reference, view + ".npz", ref["files"])
        target = args.case / "predictions" / (view + ".npz")
        with np.load(orig, allow_pickle=False) as truth, np.load(target, allow_pickle=False) as pred:
            a, z, surface = truth["rgb"], truth["z"], truth["surface"]
            b, depth, alpha = pred["rgb"], pred["z"], pred["alpha"]
            require(a.shape == b.shape == (512, 512, 3) and z.shape == depth.shape == alpha.shape == surface.shape == (512, 512)
                    and surface.dtype == np.bool_ and all(x.dtype == np.float32 for x in (a, z, b, depth, alpha)),
                    "wrong control dimensions or numerical precision")
            v = np.isfinite(a).all(axis=2)
            e = v & np.isfinite(b).all(axis=2)
            err = np.full(z.shape, np.nan, np.float32)
            err[e] = np.abs(b[e] - a[e]).mean(axis=-1)
            y = e & (err > np.float32(.05))
            z_ok = surface & np.isfinite(z) & (z > 0) & np.isfinite(depth) & (depth > 0) & (alpha >= .5) & (alpha <= 1)
            row = {"V": int(v.sum()), "E": int(e.sum()), "rgb_high_error": int(y.sum()),
                   "rgb_absolute_error_sum": float(err[e].sum(dtype=np.float64)), "depth_evaluable": int(z_ok.sum()),
                   "depth_absolute_error_sum_m": float(np.abs(depth[z_ok] - z[z_ok]).sum(dtype=np.float64))}
            if args.scores is not None:
                row["methods"] = score_counts(args.scores / (view + ".npz"), v, e, y)
            if view == "v01":
                previews(out, a, b, err)
            rows[view] = row
    totals = {k: sum(r[k] for r in rows.values()) for k in ("V", "E", "rgb_high_error", "rgb_absolute_error_sum",
                                                              "depth_evaluable", "depth_absolute_error_sum_m")}
    totals["display_srgb_mae"] = totals["rgb_absolute_error_sum"] / totals["E"] if totals["E"] else None
    totals["depth_mae_m"] = totals["depth_absolute_error_sum_m"] / totals["depth_evaluable"] if totals["depth_evaluable"] else None
    report = {"schema": "el-native-plane-control-evaluation-1", "family": case["family"], "condition": case["condition"],
              "case_sha256": args.case_sha256, "reference_sha256": args.reference_sha256, "views": rows,
              "six_view_totals": totals, "scores_included": args.scores is not None,
              "scope": "One licensed textured-plane reconstruction of a development family; six target views."}
    (out / "measurement.json").write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps(totals, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("case", "reference", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("case-sha256", "reference-sha256"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--scores", type=Path, help="directory with the six NPZ files from smoke.py --case")
    evaluate(parser.parse_args())
