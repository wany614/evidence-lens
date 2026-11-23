"""Write one private, fixed-scale v01 comparison per registered family and condition."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from evidence_lens.comparison_board import (
    ERROR_COLORS,
    INVALID_COLOR,
    RISK_COLORS,
    VIS_COLORS,
    _error_image,
    _risk_image,
)
from evidence_lens.formal_board import RISK_RAMP, risk_panel
from evidence_lens.l1 import risk_state
from evidence_lens.risk_model import rgb_preview
from evidence_lens.scoring_io import file_hash, write_json


METHODS = ("L1-full", "L2-frozen", "L2-vision-ft")
PANEL_KEYS = (
    "input",
    "reference",
    "prediction",
    "error",
    "L1-full",
    "L2-frozen",
    "L2-vision-ft",
    "visibility",
    "L1-full_state",
    "L2-frozen_state",
    "L2-vision-ft_state",
    "outcome",
)


def case_board(out, identity, source, ref, pred, truth, scores, thresholds, pins):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    e = truth["rgb_evaluable"]
    states = {}
    for m in METHODS:
        a = risk_state(scores[m], thresholds[m])
        states[m] = np.where(a == 255, np.float32(np.nan), a.astype(np.float32))
    items = [
        ("Condition input RGB", source),
        ("Blender target reference", rgb_preview(ref["rgb"])),
        ("Target prediction RGB", None if pred is None else rgb_preview(pred)),
        ("Actual display-sRGB error", _error_image(truth["rgb_error"])),
    ]
    items.extend((m + " risk", risk_panel(scores[m], e)) for m in METHODS)
    items.append(("Source visibility", Image.fromarray(np.asarray(VIS_COLORS, np.uint8)[truth["visibility"]])))
    items.extend((m + " decision", _risk_image(states[m], e)) for m in METHODS)
    items.append(("Actual error > 0.05", _risk_image(np.where(e, truth["rgb_high_error"], np.nan), e)))
    tile = 278
    gap = 16
    left = 18
    row_h = 311
    head = 66
    canvas = Image.new("RGB", (2 * left + 4 * tile + 3 * gap, head + 3 * row_h + 104), "#f7f7f4")
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(size=22)
    normal = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=12)
    draw.text(
        (left, 10),
        f"Evidence Lens    {identity['family']}    {identity['condition']}    v01",
        font=title,
        fill="#252c35",
    )
    draw.text(
        (left, 40),
        f"{identity['cohort']}    V {int(truth['reference_valid'].sum()):,}    E {int(e.sum()):,}    Y {int((truth['rgb_high_error'] == 1).sum()):,}",
        font=normal,
        fill="#252c35",
    )
    members = {}
    for i, ((label, panel), key) in enumerate(zip(items, PANEL_KEYS)):
        x = left + (i % 4) * (tile + gap)
        y = head + (i // 4) * row_h
        draw.text((x, y), label, font=normal, fill="#252c35")
        y += 24
        if panel is None:
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill="#e2e4e7")
            draw.text(
                (x + 12, y + tile // 2 - 12), "Unavailable for this condition", font=normal, fill="#252c35"
            )
        else:
            path = out / (key + ".png")
            panel.convert("RGB").save(path)
            members[path.name] = file_hash(path)
            shown = ImageOps.contain(panel.convert("RGB"), (tile, tile), method=Image.Resampling.NEAREST)
            canvas.paste(shown, (x + (tile - shown.width) // 2, y + (tile - shown.height) // 2))

    def legend(y, label, swatches):
        x = left
        draw.text((x, y), label, font=small, fill="#252c35")
        x += 160
        for text, color in swatches:
            draw.rectangle((x, y + 1, x + 12, y + 13), fill=tuple(color))
            draw.text((x + 16, y), text, font=small, fill="#252c35")
            x += int(draw.textlength(text, font=small)) + 42

    y = head + row_h * 3
    legend(y, "Mean absolute RGB error", zip(("0", ".025", ".05", ".10", ".20 and higher"), ERROR_COLORS))
    legend(y + 22, "Learned risk", zip(("0", ".25", ".5", ".75", "1"), RISK_RAMP))
    legend(
        y + 44,
        "Risk decision",
        (
            ("Low", RISK_COLORS["low"]),
            ("High", RISK_COLORS["high"]),
            ("Unknown", RISK_COLORS["unknown"]),
            ("Outside evaluable RGB", INVALID_COLOR),
        ),
    )
    legend(
        y + 66,
        "Source visibility",
        zip(("Background/invalid", "Observed", "Disoccluded", "Outside input FOV"), VIS_COLORS),
    )
    path = out / "board.png"
    canvas.save(path)
    members[path.name] = file_hash(path)
    write_json(
        out / "board.json",
        {
            "schema": "el-vision-private-case-1",
            "identity": identity,
            "view": "v01",
            "native_shape": list(e.shape),
            "thresholds": {m: thresholds[m] for m in METHODS},
            "sources": pins,
            "png_sha256": members,
            "fixed_scales": {
                "rgb_error": [0, 0.025, 0.05, 0.1, 0.2],
                "risk": [0, 0.25, 0.5, 0.75, 1],
                "state": "Within E, L means a finite score below tau, H means a finite score at or above tau, and U means a nonfinite score.",
            },
        },
    )
    return members
