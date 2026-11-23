"""Fixed full-size v01 panels for the private held-out report."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .comparison_board import (
    ERROR_COLORS,
    ERROR_LIMITS,
    INVALID_COLOR,
    RISK_COLORS,
    VIS_COLORS,
    _error_image,
    _indexed_image,
    _risk_image,
)
from .risk_model import rgb_preview
from .scoring_io import file_hash, write_json

RISK_STOPS = (0.0, 0.25, 0.5, 0.75, 1.0)
RISK_RAMP = ((34, 49, 79), (34, 109, 129), (117, 162, 129), (222, 183, 93), (233, 104, 49))


def risk_panel(arr, evaluable):
    # 0..253 hold uniformly spaced finite risk; 254 is unknown and 255 non-evaluable RGB.
    pixels = np.full(arr.shape, 254, np.uint8)
    finite = np.isfinite(arr)
    pixels[finite] = np.rint(253.0 * arr[finite]).astype(np.uint8)
    pixels[~evaluable] = 255
    vals = np.linspace(0.0, 1.0, 254)
    palette = np.zeros((256, 3), np.uint8)
    for c in range(3):
        palette[:254, c] = np.rint(np.interp(vals, RISK_STOPS, [x[c] for x in RISK_RAMP]))
    palette[254], palette[255] = RISK_COLORS["unknown"], INVALID_COLOR
    return _indexed_image(pixels, palette)


def board(out, meta, source, ref, pred, labels, scores, run_id, reason, sources):
    out.mkdir()
    e = labels["rgb_evaluable"]
    panels = [
        ("input", "Condition input", source),
        ("reference", "Blender reference", rgb_preview(ref["rgb"])),
        ("prediction", "Single-image prediction", rgb_preview(pred["rgb"]) if pred is not None else None),
        ("error", "Actual RGB error", _error_image(labels["rgb_error"])),
        ("L0", "L0 risk state", _risk_image(scores["L0"], e)),
        (
            "geometry",
            "True geometric visibility",
            Image.fromarray(np.asarray(VIS_COLORS, np.uint8)[labels["visibility"]]),
        ),
        ("L1-full", "L1-full risk score", risk_panel(scores["L1-full"], e)),
        ("L1-geometry", "L1-geometry risk score", risk_panel(scores["L1-geometry"], e)),
        ("outcome", "Thresholded RGB error", _risk_image(np.where(e, labels["rgb_high_error"], np.nan), e)),
    ]
    ink, margin, tile, gap, header, row_height = "#252c35", 24, 384, 20, 76, 420
    canvas = Image.new("RGB", (margin * 2 + tile * 3 + gap * 2, header + row_height * 3 + 132), "#f7f7f4")
    d = ImageDraw.Draw(canvas)
    title, normal, small = (ImageFont.load_default(size=s) for s in (23, 17, 14))
    d.text(
        (margin, 12),
        f"Evidence Lens  /  {meta['scene_family_id']}  /  {meta['condition_id']}  /  v01",
        font=title,
        fill=ink,
    )
    d.text((margin, 44), f"Fixed first target    {run_id}", font=small, fill=ink)
    paths = {}
    for i, (key, label, panel) in enumerate(panels):
        x, y = margin + (i % 3) * (tile + gap), header + (i // 3) * row_height
        d.text((x, y), label, fill=ink, font=normal)
        y += 26
        if panel is None:
            d.rectangle((x, y, x + tile - 1, y + tile - 1), fill="#e8e8e5")
            d.text((x + 10, y + tile // 2 - 14), "Prediction unavailable", fill=ink, font=normal)
            d.text((x + 10, y + tile // 2 + 10), str(reason)[:47], fill=ink, font=small)
        else:
            path = out / f"{key}.png"
            panel.save(path)
            paths[path.name] = file_hash(path)
            shown = ImageOps.contain(panel.convert("RGB"), (tile, tile), method=Image.Resampling.NEAREST)
            canvas.paste(shown, (x + (tile - shown.width) // 2, y + (tile - shown.height) // 2))

    def legend(y, title_text, entries):
        x = margin
        d.text((x, y), title_text, font=small, fill=ink)
        x += 139
        for name, color in entries:
            d.rectangle((x, y + 1, x + 12, y + 13), fill=tuple(map(int, color)))
            d.text((x + 17, y), name, font=small, fill=ink)
            x += 32 + round(d.textlength(name, font=small))

    y = header + row_height * 3
    legend(y, "Actual RGB error", zip(("0", ".025", ".05", ".10", ".20 or more"), ERROR_COLORS))
    legend(y + 23, "Learned risk", zip(("0", ".25", ".5", ".75", "1"), RISK_RAMP))
    legend(
        y + 46,
        "L0 / error state",
        [
            ("low / within tolerance", RISK_COLORS["low"]),
            ("high / exceeds tolerance", RISK_COLORS["high"]),
            ("unknown risk", RISK_COLORS["unknown"]),
            ("unevaluable RGB", INVALID_COLOR),
        ],
    )
    legend(
        y + 69,
        "True visibility",
        zip(("background or invalid", "observed", "disoccluded", "outside input FOV"), VIS_COLORS),
    )
    d.text(
        (margin, y + 99),
        "Every panel retains its own fixed units. Native pixels and the numeric source hashes accompany this board.",
        font=small,
        fill=ink,
    )
    result = out / "board_v01.png"
    canvas.save(result)
    desc = {
        "schema": "el-formal-board-1",
        "run_id": run_id,
        "scene_family_id": meta["scene_family_id"],
        "condition_id": meta["condition_id"],
        "camera_id": "v01",
        "selection": "preregistered first target",
        "render_failure_reason": reason,
        "source_files": sources,
        "panels": paths,
        "board_sha256": file_hash(result),
        "error_finite_scale": ERROR_LIMITS.tolist(),
        "risk_scale": list(RISK_STOPS),
        "risk_colors": RISK_RAMP,
        "risk_codes": {"0-253": "round(253*score)", "254": "unknown on E", "255": "outside E"},
    }
    write_json(out / "board_v01.json", desc)
    return desc
