"""Fixed P2 engineering board for the preselected first target camera."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .risk_model import rgb_preview
from .scoring_io import file_hash


ERROR_LIMITS = np.array([0, 0.025, 0.05, 0.10, 0.20])
ERROR_COLORS = np.array([(24, 32, 49), (46, 91, 153), (193, 102, 46), (244, 169, 79), (255, 242, 196)])
INVALID_COLOR = (220, 35, 171)
RISK_COLORS = {"low": (66, 153, 133), "high": (226, 117, 60), "unknown": (113, 118, 127)}
VIS_COLORS = [(91, 97, 110), (51, 137, 179), (229, 138, 58), (145, 103, 175)]


def _indexed_image(codes, palette):
    image = Image.fromarray(codes)
    image.putpalette(palette.astype(np.uint8).tobytes())
    return image


def _error_image(array):
    valid = np.isfinite(array)
    codes = np.full(array.shape, 255, np.uint8)
    codes[valid] = np.rint(np.clip(array[valid], 0, ERROR_LIMITS[-1]) * (254 / ERROR_LIMITS[-1])).astype(np.uint8)
    values = np.linspace(0, ERROR_LIMITS[-1], 255)
    palette = np.empty((256, 3), np.uint8)
    palette[:255] = np.stack([np.interp(values, ERROR_LIMITS, ERROR_COLORS[:, i]) for i in range(3)], axis=-1).round()
    palette[255] = INVALID_COLOR
    return _indexed_image(codes, palette)


def _risk_image(score, rgb_evaluable):
    codes = np.full(score.shape, 2, np.uint8)
    codes[score == 0], codes[score == 1] = 0, 1
    codes[~rgb_evaluable] = 255
    palette = np.zeros((256, 3), np.uint8)
    palette[[0, 1, 2, 255]] = [RISK_COLORS["low"], RISK_COLORS["high"], RISK_COLORS["unknown"], INVALID_COLOR]
    return _indexed_image(codes, palette)


def render_board(path: Path, input_rgb, ref, pred, labels, scores, run_id: str):
    """Use identical error limits and palettes for every run. Missing renders get labelled tiles."""
    error = _error_image(labels["rgb_error"]) if pred is not None else None
    risk = _risk_image(scores["l0"], labels["rgb_evaluable"]) if scores is not None else None
    previews = {}
    for name, panel in (("v01_rgb_error.png", error), ("v01_l0_state.png", risk)):
        if panel is not None:
            panel_path = path.parent / name
            panel.save(panel_path)
            previews[name] = file_hash(panel_path)
    display = [
        ("Exact source PNG", rgb_preview(input_rgb)),
        ("v01 reference", rgb_preview(ref["rgb"])),
        ("v01 prediction", rgb_preview(pred["rgb"]) if pred is not None else None),
        ("Actual RGB error (mean |difference|)", error),
        ("L0 risk on evaluable RGB", risk),
        ("Geometric visibility", Image.fromarray(np.array(VIS_COLORS, np.uint8)[ref["visibility"]])),
    ]
    tile_w, tile_h, gap = 512, 512, 16
    w, h = 3 * tile_w + 4 * gap, 1280
    paper = Image.new("RGB", (w, h), "#f7f7f4")
    draw = ImageDraw.Draw(paper)
    title_font, font, small = (ImageFont.load_default(size=s) for s in (30, 21, 17))
    ink = "#212733"
    draw.text((gap, 8), "Evidence Lens  /  P2 engineering comparison", fill=ink, font=title_font)
    draw.text((gap, 48), f"Fixed target v01   |   {run_id}", fill=ink, font=small)
    for i, (title, panel) in enumerate(display):
        x, y = gap + (i % 3) * (tile_w + gap), 84 + (i // 3) * (tile_h + 48)
        draw.text((x, y), title, fill=ink, font=font)
        y += 30
        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), fill="#e5e6e5")
        if panel is None:
            draw.text((x + tile_w // 2, y + tile_h // 2), "Render unavailable", fill=ink, font=font, anchor="mm")
        else:
            scaled = ImageOps.contain(panel, (tile_w, tile_h), method=Image.Resampling.NEAREST)
            paper.paste(scaled, (x + (tile_w - scaled.width) // 2, y + (tile_h - scaled.height) // 2))
    def legend(x, y, entries):
        for label, color in entries:
            draw.rectangle((x, y + 3, x + 14, y + 17), fill=tuple(int(c) for c in color))
            draw.text((x + 20, y), label, fill=ink, font=small)
            x += 31 + int(draw.textlength(label, font=small))
    legend(gap, 1202, zip(("0", ".025", ".05", ".10", ">=.20"), ERROR_COLORS))
    legend(2 * gap + tile_w, 1202, [*RISK_COLORS.items(), ("RGB unevaluable", INVALID_COLOR)])
    legend(3 * gap + 2 * tile_w, 1202, zip(("BG/invalid", "Observed", "Disoccluded", "Outside input"), VIS_COLORS))
    draw.text((gap, 1244), "Magenta marks unavailable RGB. Gray risk marks an unknown L0 decision where both RGB images are valid.", fill=ink, font=small)
    paper.save(path)
    return previews
