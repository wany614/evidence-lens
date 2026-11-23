"""Fixed P2 engineering board for the preselected first target camera."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .risk_model import rgb_preview


ERROR_LIMITS = np.array([0, 0.025, 0.05, 0.10, 0.20])
ERROR_COLORS = np.array([(24, 32, 49), (46, 91, 153), (193, 102, 46), (244, 169, 79), (255, 242, 196)])
INVALID_COLOR = (220, 35, 171)
RISK_COLORS = {"low": (66, 153, 133), "high": (226, 117, 60), "unknown": (113, 118, 127)}
VIS_COLORS = [(91, 97, 110), (51, 137, 179), (229, 138, 58), (145, 103, 175)]


def _error_image(array):
    values = np.nan_to_num(array, nan=0)
    colors = np.stack([np.interp(values, ERROR_LIMITS, ERROR_COLORS[:, i]) for i in range(3)], axis=-1).astype(np.uint8)
    colors[~np.isfinite(array)] = INVALID_COLOR
    return Image.fromarray(colors)


def _risk_image(score):
    colors = np.empty((*score.shape, 3), np.uint8)
    colors[:] = RISK_COLORS["unknown"]
    colors[score == 0] = RISK_COLORS["low"]
    colors[score == 1] = RISK_COLORS["high"]
    return Image.fromarray(colors)


def render_board(path: Path, input_rgb, ref, pred, labels, scores, run_id: str):
    """Use identical error limits and palettes for every run. Missing renders get labelled tiles."""
    display = [
        ("Exact source PNG", rgb_preview(input_rgb), False),
        ("v01 reference", rgb_preview(ref["rgb"]), False),
        ("v01 prediction", rgb_preview(pred["rgb"]) if pred is not None else None, False),
        ("Actual RGB error (mean |difference|)", _error_image(labels["rgb_error"]) if pred is not None else None, True),
        ("L0 risk", _risk_image(scores["l0"]) if scores is not None else None, True),
        ("Geometric visibility", Image.fromarray(np.array(VIS_COLORS, np.uint8)[ref["visibility"]]), True),
    ]
    tile_w, tile_h = 280, 245
    w, h = 3 * tile_w + 48, 690
    paper = Image.new("RGB", (w, h), "#f7f7f4")
    draw = ImageDraw.Draw(paper)
    title_font, font, small = (ImageFont.load_default(size=s) for s in (21, 15, 13))
    ink = "#212733"
    draw.text((12, 8), "Evidence Lens  /  P2 engineering comparison", fill=ink, font=title_font)
    draw.text((12, 37), f"Fixed target v01   |   {run_id}", fill=ink, font=small)
    for i, (title, panel, discrete) in enumerate(display):
        x, y = 12 + (i % 3) * (tile_w + 12), 66 + (i // 3) * (tile_h + 32)
        draw.text((x, y), title, fill=ink, font=font)
        y += 23
        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), fill="#e5e6e5")
        if panel is None:
            draw.text((x + tile_w // 2, y + tile_h // 2), "Render unavailable", fill=ink, font=font, anchor="mm")
        else:
            sampling = Image.Resampling.NEAREST if discrete else Image.Resampling.LANCZOS
            scaled = ImageOps.contain(panel, (tile_w, tile_h), method=sampling)
            paper.paste(scaled, (x + (tile_w - scaled.width) // 2, y + (tile_h - scaled.height) // 2))
    def legend(x, y, entries):
        for label, color in entries:
            draw.rectangle((x, y + 2, x + 11, y + 13), fill=tuple(int(c) for c in color))
            draw.text((x + 16, y), label, fill=ink, font=small)
            x += 23 + int(draw.textlength(label, font=small))
    legend(12, 622, list(zip(("0", ".025", ".05", ".10", ">=.20"), ERROR_COLORS)))
    legend(304, 622, RISK_COLORS.items())
    legend(596, 622, zip(("Other", "Observed", "Newly visible", "Outside"), VIS_COLORS))
    draw.text((12, 655), "Error colours interpolate between fixed values. Magenta marks invalid RGB. Visibility is evaluated from Blender geometry.", fill=ink, font=small)
    paper.save(path)
