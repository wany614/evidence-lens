"""Write private, fixed-scale figures and aggregate-only tables for phase-one diagnostics."""

import csv

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from evidence_lens.comparison_board import (
    ERROR_COLORS,
    ERROR_LIMITS,
    INVALID_COLOR,
    RISK_COLORS,
    VIS_COLORS,
    _error_image,
    _risk_image,
)
from evidence_lens.formal_board import RISK_RAMP, RISK_STOPS, risk_panel
from evidence_lens.risk_model import rgb_preview
from evidence_lens.scoring_io import file_hash, write_json

REGIONS = (
    "all",
    "foreground",
    "floor",
    "back_wall",
    "no_mesh",
    "edge",
    "interior",
    "observed",
    "disoccluded",
    "outside_input_fov",
    "background_or_invalid",
)
COLORS = ((40, 88, 168), (243, 243, 235), (175, 52, 45))
INK = (28, 41, 56)
PAPER = (250, 249, 245)


def csv_out(path, records):
    if not records:
        raise ValueError(f"empty diagnostic table: {path.name}")
    with path.open("x", encoding="utf-8", newline="") as file:
        columns = dict.fromkeys(k for record in records for k in record)
        writer = csv.DictWriter(file, list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def depth_panel(values, valid, stops):
    rgb = np.full((*valid.shape, 3), INVALID_COLOR, np.uint8)
    if valid.any():
        rgb[valid] = np.rint(
            np.stack([np.interp(values[valid], stops, [c[ch] for c in COLORS]) for ch in range(3)], axis=-1)
        ).astype(np.uint8)
    return Image.fromarray(rgb)


def _text(draw, xy, msg, size=17, fill=INK):
    draw.text(xy, msg, fill=fill, font=ImageFont.load_default(size=size))


def _legend(draw, y, title, items):
    _text(draw, (23, y), title, 16)
    x = 160
    for label, color in items:
        draw.rectangle((x, y + 4, x + 12, y + 16), fill=tuple(map(int, color)))
        _text(draw, (x + 17, y), label, 15)
        x += 31 + round(draw.textlength(label, font=ImageFont.load_default(size=15)))


def private_board(out, family, rid, image, reference, pred, lab, score, state, tau, sources):
    out.mkdir(parents=True)
    e, z = lab["rgb_evaluable"], lab["depth_evaluable"]
    ratio = np.divide(pred["z"], reference["z"], out=np.full(z.shape, np.nan), where=z)
    panels = [
        ("input", "Original input", image),
        ("reference", "Blender target RGB", rgb_preview(reference["rgb"])),
        ("prediction", "SHARP target RGB", rgb_preview(pred["rgb"])),
        ("error", "Actual RGB error", _error_image(lab["rgb_error"])),
        ("risk", "L1 full risk score", risk_panel(score, e)),
        ("state", "L1 full operating state", _risk_image(np.where(state == 255, np.nan, state), e)),
        (
            "signed-z",
            "Predicted minus reference Z (m)",
            depth_panel(lab["signed_depth_error"], z, (-2, 0, 2)),
        ),
        ("z-ratio", "Predicted / reference Z", depth_panel(ratio, z, (0, 1, 2))),
        (
            "visibility",
            "Reference visibility",
            Image.fromarray(np.asarray(VIS_COLORS, np.uint8)[reference["visibility"]]),
        ),
    ]
    tile, margin, gap, top, row, footer = 344, 23, 18, 71, 376, 155
    canvas = Image.new("RGB", (2 * margin + 3 * tile + 2 * gap, top + 3 * row + footer), PAPER)
    draw = ImageDraw.Draw(canvas)
    _text(draw, (margin, 10), f"{family}   B0 / v01    |    Evidence Lens", 24)
    _text(draw, (margin, 42), f"Private formal case   {rid}    |    fixed first target", 16)
    signatures = {}
    for i, (key, title, pic) in enumerate(panels):
        x, y = margin + i % 3 * (tile + gap), top + i // 3 * row
        _text(draw, (x, y), title, 18)
        p = out / (key + ".png")
        pic.save(p)
        signatures[p.name] = file_hash(p)
        scaled = ImageOps.contain(pic.convert("RGB"), (tile, tile), Image.Resampling.NEAREST)
        canvas.paste(scaled, (x + (tile - scaled.width) // 2, y + 26 + (tile - scaled.height) // 2))
    y = top + 3 * row
    _legend(draw, y, "Actual RGB error", zip(["0", ".025", ".05", ".10", ".20+"], ERROR_COLORS))
    _legend(draw, y + 24, "L1 full risk", zip(["0", ".25", ".50", ".75", "1"], RISK_RAMP))
    _legend(
        draw,
        y + 48,
        "Operating state",
        [(k, RISK_COLORS[k]) for k in ("low", "high", "unknown")] + [("outside E / invalid", INVALID_COLOR)],
    )
    _legend(draw, y + 72, "Depth panels", zip(["-2 m / 0x", "0 m / 1x", "+2 m / 2x"], COLORS))
    _legend(
        draw,
        y + 96,
        "Visibility",
        zip(["background / invalid", "observed", "disoccluded", "outside source FOV"], VIS_COLORS),
    )
    _text(
        draw,
        (margin, y + 123),
        f"Low = finite score < {tau:.8f}. Values beyond the displayed depth range saturate. Exact numerical data remain in the sealed sources.",
        14,
    )
    path = out / "board.png"
    canvas.save(path)
    result = {
        "schema": "el-private-depth-case-board-1",
        "run_id": rid,
        "family": family,
        "condition": "B0",
        "view": "v01",
        "access": "owner_private_formal_SHARP_images",
        "sha256": file_hash(path),
        "panels": signatures,
        "sources": sources,
        "tau": tau,
        "scales": {
            "rgb_error": ERROR_LIMITS.tolist(),
            "risk": list(RISK_STOPS),
            "signed_z_m": [-2, 0, 2],
            "z_ratio": [0, 1, 2],
        },
        "counts": {
            "E": int(e.sum()),
            "L": int((e & (state == 0)).sum()),
            "H": int((e & (state == 1)).sum()),
            "U": int((e & (state == 255)).sum()),
            "outside_E": int((~e).sum()),
        },
    }
    write_json(out / "board.json", result)
    return result


def strata_plot(path, family, condition, rows, views):
    """Show exact per-view ratio percentiles with a common fixed 0..2 axis."""
    lookup = {
        (r["view"], r["region"]): r for r in rows if r["family"] == family and r["condition"] == condition
    }
    lab, cell, row, head = 153, 133, 38, 106
    paper = Image.new("RGB", (lab + cell * len(views) + 12, head + row * len(REGIONS) + 50), PAPER)
    draw = ImageDraw.Draw(paper)
    _text(draw, (14, 9), f"{family}  /  {condition}   Predicted to reference camera-Z ratio", 23)
    _text(
        draw,
        (14, 45),
        "Line = 5th to 95th percentile. Dot = median. Triangle = percentile beyond the fixed 0–2 range. Blank = no comparable depth.",
        14,
    )
    for col, view in enumerate(views):
        x = lab + col * cell
        _text(draw, (x + 40, 71), view, 15)
        low, high = x + 8, x + cell - 16
        mid = (low + high) // 2
        draw.line((mid, head - 4, mid, head + len(REGIONS) * row), fill=(203, 200, 190))
        for i, region in enumerate(REGIONS):
            y = head + i * row + 10
            if col == 0:
                _text(draw, (12, y - 6), region.replace("_", " "), 13)
            r = lookup.get((view, region), {})
            if r.get("ratio_q50") is None:
                continue
            left, centre, right = [r["ratio_q" + q] for q in ("05", "50", "95")]

            def xpos(v, low=low, high=high):
                return round(low + min(2, max(0, v)) * (high - low) / 2)

            a, m, b = map(xpos, (left, centre, right))
            draw.line((a, y, b, y), fill=(36, 109, 137), width=3)
            draw.ellipse((m - 4, y - 4, m + 4, y + 4), fill=(23, 56, 75))
            if left < 0:
                draw.polygon([(low - 3, y), (low + 4, y - 4), (low + 4, y + 4)], fill=(143, 62, 39))
            if right > 2:
                draw.polygon([(high + 3, y), (high - 4, y - 4), (high - 4, y + 4)], fill=(143, 62, 39))
        y = head + row * len(REGIONS) + 7
        for xaxis, label in ((low, "0"), (mid, "1"), (high, "2")):
            _text(draw, (xaxis - 3, y), label, 12)
    paper.save(path)
    return file_hash(path)
