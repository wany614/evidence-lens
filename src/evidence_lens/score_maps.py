"""Export fixed-scale PNG risk and decision maps from an Evidence Lens CPU score directory."""

import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .l1 import KEYS, risk_state
from .scoring_io import TARGETS, ReadAudit, file_hash, new_output, write_json

# The study figures use the same RGB anchors. This reader requires only the CPU inference package.
RAMP = ((34, 49, 79), (34, 109, 129), (117, 162, 129), (222, 183, 93), (233, 104, 49))
STOPS = (0.0, 0.25, 0.5, 0.75, 1.0)
STATE_COLORS = {0: (66, 153, 133), 1: (226, 117, 60), 255: (113, 118, 127)}
STATE_NAMES = {0: "low", 1: "high", 255: "unknown"}


def palette_info():
    return {
        "continuous": {
            "range": [0, 1],
            "stops": [{"risk": v, "rgb": list(c)} for v, c in zip(STOPS, RAMP, strict=True)],
            "finite_indices": "round(254 * risk), linearly interpolated RGB",
            "unknown_index": 255,
            "unknown_rgb": list(STATE_COLORS[255]),
        },
        "state": {str(k): {"label": STATE_NAMES[k], "rgb": list(v)} for k, v in STATE_COLORS.items()},
    }


def emit_png(path, indices, palette):
    image = Image.fromarray(indices)
    image.putpalette(palette.tobytes())
    image.save(path, format="PNG")
    return {"file": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size}


def decode_maps(raw, key, threshold, view, row):
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        names = (key, key + "_valid", key + "_state")
        if not all(name in arrays for name in names):
            raise ValueError(f"{view}: score, validity or state array missing")
        score, valid, state = (arrays[name] for name in names)
    if (
        score.ndim != 2
        or 0 in score.shape
        or score.shape != valid.shape
        or score.shape != state.shape
        or score.dtype != np.float32
        or valid.dtype != bool
        or state.dtype != np.uint8
    ):
        raise ValueError(f"{view}: score, validity or state shape/dtype mismatch")
    if (
        np.isinf(score).any()
        or not np.array_equal(np.isfinite(score), valid)
        or np.any((score[valid] < 0) | (score[valid] > 1))
    ):
        raise ValueError(f"{view}: score must be finite in [0,1] or NaN wherever validity is false")
    if not np.array_equal(state, risk_state(score, threshold)):
        raise ValueError(f"{view}: state codes differ from validity or the frozen threshold")
    if key == "l0" and not np.all((score[valid] == 0) | (score[valid] == 1)):
        raise ValueError(f"{view}: L0 scores must be binary")
    n = int(valid.sum())
    if (row["status"] in ("unknown", "render_failed") and n) or (
        key in ("l1_full", "l1_geometry")
        and (row.get("l1_scoreable") != n or (row["status"] == "unsupported_l1_depth" and n))
    ):
        raise ValueError(f"{view}: finite scores disagree with the scoring receipt")
    return score, valid, state


def export(scores, output, *, view=None, method="L1-full", receipt_sha256=None):
    scores, output = Path(scores), Path(output)
    selected = TARGETS if view is None else (view,)
    if method not in KEYS or any(v not in TARGETS for v in selected):
        raise ValueError("unrecognized method or target camera")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"choose a new output directory: {output}")
    audit = ReadAudit()
    receipt = json.loads(audit.read("l1_score.json", scores / "l1_score.json"))
    if receipt_sha256 is not None and receipt_sha256 != audit.sha("l1_score.json"):
        raise ValueError("l1_score.json differs from the supplied SHA-256")
    if (
        receipt.get("schema") != "el-l1-score-1"
        or receipt.get("methods") != KEYS
        or receipt.get("state_codes") != {str(k): v for k, v in STATE_NAMES.items()}
        or set(receipt.get("views", {})) != set(TARGETS)
        or receipt.get("status") not in ("scored", "unsupported_l1_depth", "unsupported")
        or set(receipt.get("thresholds", {})) != set(KEYS) - {"L0"}
    ):
        raise ValueError("incompatible L1 score receipt")
    threshold = 0.5 if method == "L0" else receipt["thresholds"][method]
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1.000001:
        raise ValueError("invalid frozen threshold")
    key = KEYS[method]
    legend = palette_info()
    continuous_palette = np.zeros((256, 3), dtype=np.uint8)
    continuous_palette[:255] = np.rint(
        np.stack([np.interp(np.arange(255) / 254, STOPS, x) for x in zip(*RAMP, strict=True)], axis=-1)
    ).astype(np.uint8)
    continuous_palette[255] = STATE_COLORS[255]
    state_palette = np.zeros((256, 3), dtype=np.uint8)
    for code, color in STATE_COLORS.items():
        state_palette[code] = color
    source = {
        "receipt": {"file": "l1_score.json", **audit.files["l1_score.json"]},
        "score_run_id": receipt.get("run_id"),
        "model_id": receipt["model_id"],
        "prediction_run_id": receipt.get("producer_run_id"),
        "model_index_sha256": receipt["model_files"]["model_index"]["sha256"],
    }
    source["input_sha256"] = receipt["inputs"]["input_png"]["sha256"]
    source["cameras_sha256"] = receipt["inputs"]["cameras"]["sha256"]
    results = {}
    with new_output(output) as out:
        for v in selected:
            row = receipt["views"][v]
            status = row["status"]
            if status not in ("scored", "unsupported_l1_depth", "render_failed", "unknown"):
                raise ValueError(f"{v}: unrecognized score status")
            if status not in (receipt["status"], "render_failed") and not (
                status == "unknown" and receipt["status"] == "unsupported"
            ):
                raise ValueError(f"{v}: per-target and global score status disagree")
            record = {
                "schema": "el-score-map-view-1",
                "camera_id": v,
                "method": method,
                "score_key": key,
                "threshold": threshold,
                "low_rule": "finite risk < threshold",
                "score_status": status,
                "source": dict(source),
                "colours": legend,
                "width": None,
                "height": None,
                "counts": None,
                "maps": {},
            }
            if "prediction/" + v in receipt["inputs"]:
                record["source"]["prediction_view_sha256"] = receipt["inputs"]["prediction/" + v]["sha256"]
            if "sha256" not in row:
                if (
                    status != "unknown"
                    or receipt["status"] != "unsupported"
                    or (scores / (v + ".npz")).exists()
                ):
                    raise ValueError(f"{v}: no matching score-array digest")
                record["map_status"] = "unavailable_dimensions"
            else:
                name = v + ".npz"
                raw = audit.read(name, scores / name)
                if audit.sha(name) != row["sha256"]:
                    raise ValueError(f"{v}: score arrays differ from the receipt SHA-256")
                record["source"]["scores"] = {"file": name, **audit.files[name]}
                score, valid, state = decode_maps(raw, key, threshold, v, row)
                indices = np.full(score.shape, 255, np.uint8)
                indices[valid] = np.rint(np.float64(254) * score[valid].astype(np.float64)).astype(np.uint8)
                record.update(
                    map_status="written",
                    width=score.shape[1],
                    height=score.shape[0],
                    counts={
                        "total": state.size,
                        "valid": int(valid.sum()),
                        **{label: int(np.count_nonzero(state == i)) for i, label in STATE_NAMES.items()},
                    },
                    maps={
                        "continuous": emit_png(out / (v + "-risk.png"), indices, continuous_palette),
                        "state": emit_png(out / (v + "-state.png"), state, state_palette),
                    },
                )
            metadata = out / (v + ".json")
            write_json(metadata, record)
            results[v] = {
                "file": metadata.name,
                "sha256": file_hash(metadata),
                "map_status": record["map_status"],
                "counts": record["counts"],
            }
        index = {"schema": "el-score-maps-1", "method": method, "source": source, "views": results}
        write_json(out / "index.json", index)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores", type=Path, required=True, help="Directory containing l1_score.json and target NPZ files"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for the PNGs and portable JSON"
    )
    parser.add_argument("--view", choices=TARGETS, help="One target camera; default is all six")
    parser.add_argument("--method", choices=KEYS, default="L1-full")
    parser.add_argument("--receipt-sha256", help="Optional independently supplied SHA-256 of l1_score.json")
    args = parser.parse_args(argv)
    index = export(
        args.scores, args.output, view=args.view, method=args.method, receipt_sha256=args.receipt_sha256
    )
    print(
        json.dumps(
            {
                "index": str(args.output / "index.json"),
                "method": index["method"],
                "views": list(index["views"]),
            }
        )
    )


if __name__ == "__main__":
    main()
