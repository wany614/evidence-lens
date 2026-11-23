"""Fixed native-grid six-target panels, using separate answers only during evaluation."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evidence_lens.formal_inputs import verified_file
from evidence_lens.l1 import risk_state
from evidence_lens.multisource import METHODS, SCOPES
from evidence_lens.multisource_io import Case
from stage3_risk_data import read_asset
from stage3_risk_sources import asset

INVALID = (127, 55, 150)
UNKNOWN = (113, 118, 127)
LOW = (75, 168, 184)
HIGH = (230, 99, 67)
RAMP = np.array([[23, 71, 96], [37, 138, 151], [137, 200, 162], [245, 205, 123], [207, 63, 68]], float)


def rgb(a):
    ok = np.isfinite(a).all(-1)
    x = np.rint(np.clip(np.nan_to_num(a), 0, 1) * 255).astype(np.uint8)
    x[~ok] = INVALID
    return x


def heat(a, maximum, e):
    ok = np.isfinite(a) & e
    frac = np.clip(np.where(ok, a / maximum, 0), 0, 1)
    x = np.stack([np.interp(frac, np.linspace(0, 1, len(RAMP)), RAMP[:, i]) for i in range(3)], axis=-1)
    x = np.rint(x).astype(np.uint8)
    x[~ok] = UNKNOWN
    x[~e] = INVALID
    return x


def state(a, tau, e):
    s = risk_state(a, tau)
    x = np.full((*s.shape, 3), UNKNOWN, np.uint8)
    x[s == 0] = LOW
    x[s == 1] = HIGH
    x[~e] = INVALID
    return x


def save_case(root, out, row, truth, scores, thresholds):
    f, c, v = row["family"], row["condition"], row["view"]
    h, w = row["height"], row["width"]
    # Real formal frames are 256 square. Fixtures use a smaller native input and a legible display tile.
    tile = max(150, w)
    font = ImageFont.load_default(size=14)
    sm = ImageFont.load_default(size=12)
    margin, gap, label = 12, 6, 22
    stride = tile + gap
    image = Image.new(
        "RGB", (2 * margin + 8 * stride - gap, 90 + 4 * (tile + label + gap) + 52), (248, 247, 242)
    )
    draw = ImageDraw.Draw(image)
    draw.text(
        (margin, 8), f"{f}  /  {c}  /  {v}     {w} × {h}     {row['status']}", font=font, fill=(24, 37, 45)
    )
    draw.text(
        (margin, 30),
        "Native target RGB and expected camera Z. All scoring modes use this same reconstruction.",
        font=sm,
        fill=(45, 58, 64),
    )
    draw.text(
        (margin, 49),
        "Continuous scales: RGB MAE 0–0.20 display sRGB, depth error 0–0.20 m, risk 0–1. Higher values are red.",
        font=sm,
        fill=(45, 58, 64),
    )
    draw.text(
        (margin, 67),
        "State colours: blue = low; red = high; grey = unknown; purple = outside the evaluable domain. B transfer always uses s00.",
        font=sm,
        fill=(45, 58, 64),
    )

    def place(col, r, title, pixels=None):
        x = margin + col * stride
        y = 90 + r * (tile + label + gap)
        draw.text((x, y), title, font=sm, fill=(30, 51, 59))
        if pixels is None:
            draw.rectangle((x, y + label, x + tile - 1, y + label + tile - 1), fill=UNKNOWN)
            return
        native = Image.fromarray(pixels)
        if native.size != (tile, tile):
            native = native.resize((tile, tile), Image.Resampling.NEAREST)
        image.paste(native, (x, y + label))

    case = (
        Case(verified_file(root, row["case"]["path"], row["case"]).parent, row["case"]["sha256"])
        if row.get("case")
        else None
    )
    source_images = case.sources("all") if case is not None else {}
    for i, s in enumerate(("s00", "s01", "s02", "s03")):
        place(
            i,
            0,
            s + " input" if s in source_images else s + " unavailable",
            rgb(source_images[s][0]) if s in source_images else None,
        )
    place(4, 0, "Target reference", rgb(read_asset(root, row["reference_rgb"])))
    place(5, 0, "Target prediction", rgb(read_asset(root, row["rgb"])) if row.get("rgb") else None)
    e = truth["rgb_evaluable"]
    place(6, 0, "RGB |error|", heat(truth["rgb_error"], 0.2, e))
    place(7, 0, "Depth |error| (m)", heat(truth["depth_error"], 0.2, truth["depth_evaluable"]))
    for k, scope in enumerate(SCOPES, 1):
        for j, m in enumerate(METHODS):
            a = scores[scope][m]
            note = scope if (c, scope) != ("M2", "all") else "all = two"
            place(2 * j, k, m + " / " + note + " risk", heat(a, 1.0, e))
            place(2 * j + 1, k, m + " / " + note + " state", state(a, thresholds[m], e))
    draw.text(
        (margin, image.height - 45),
        "Sources are exact allowed PNGs. The independent reference, RGB and auxiliary depth errors are evaluation-only.",
        font=sm,
        fill=(45, 58, 64),
    )
    draw.text(
        (margin, image.height - 25),
        f"Selection {row['selection_sha256'][:16]}   Target prediction {(row.get('prediction_member_sha256') or 'unavailable')[:16]}   Reference {row['reference_member_sha256'][:16]}",
        font=sm,
        fill=(45, 58, 64),
    )
    path = out / "cases" / f / c / (v + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return {
        "family": f,
        "condition": c,
        "view": v,
        "selection_sha256": row["selection_sha256"],
        "reference_member_sha256": row["reference_member_sha256"],
        "prediction_member_sha256": row["prediction_member_sha256"],
        "case_sha256": row["case"]["sha256"] if row.get("case") else None,
        "status": row["status"],
        "file": asset(out, path),
    }
