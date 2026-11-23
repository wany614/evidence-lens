"""Native photographic evaluation panels for local research readback."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dtu_score import METHODS
from stage3_risk_boards import rgb, heat, state, UNKNOWN, INVALID, HIGH, LOW


def board(path, object_id, condition, view, source_images, target, ref, labels, pics, thresholds, alpha):
    v, e, y, err, aux = labels
    size = 256
    margin = 12
    stride = size + 6
    band = 284
    page = Image.new("RGB", (margin * 2 + 6 * stride - 6, 74 + 3 * band + 37), (248, 247, 242))
    d = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=13)
    d.text(
        (margin, 8),
        f"{object_id}   {condition}   {view}   |   256 x 256   |   private research evaluation",
        fill=(19, 45, 53),
        font=ImageFont.load_default(size=19),
    )
    xx = margin
    for t, c in [
        ("low", LOW),
        ("high", HIGH),
        ("unknown / missing", UNKNOWN),
        ("outside evaluation domain", INVALID),
    ]:
        d.rectangle((xx, 43, xx + 13, 56), fill=c)
        d.text((xx + 19, 41), t, fill=(19, 45, 53), font=font)
        xx += len(t) * 7 + 48

    def tile(col, row, title, a):
        x = margin + col * stride
        y0 = 74 + row * band
        d.text((x, y0), title, fill=(19, 45, 53), font=font)
        if a is None:
            d.rectangle((x, y0 + 20, x + size - 1, y0 + 20 + size - 1), fill=UNKNOWN)
        else:
            page.paste(Image.fromarray(a), (x, y0 + 20))

    for i, s in enumerate(("s00", "s01", "s02", "s03")):
        im = source_images.get(s)
        tile(
            i, 0, s + " source" if im is not None else s + " unavailable", rgb(im) if im is not None else None
        )
    tile(4, 0, "Target prediction", rgb(target) if target is not None else None)
    tile(5, 0, "Target reference", rgb(ref))
    tile(0, 1, "RGB MAE (0 to 0.20)", heat(err, 0.2, e))
    tile(1, 1, "3x3 minimum MAE (0 to 0.20)", heat(aux, 0.2, e))
    mask = np.full((*e.shape, 3), INVALID, np.uint8)
    mask[e & (y == 0)] = LOW
    mask[e & (y == 1)] = HIGH
    tile(0, 2, "Confirmed error (>0.05)", mask)
    tile(1, 2, "Predicted opacity (0 to 1)", heat(alpha, 1, np.ones_like(e)) if alpha is not None else None)
    for i, m in enumerate(METHODS):
        tile(i + 2, 1, m + " risk (0 to 1)", heat(pics[m], 1, e))
        tile(i + 2, 2, m + " frozen state", state(pics[m], thresholds[m], e))
    d.text(
        (margin, page.height - 27),
        "Scoring reads source evidence only. Independent target photographs and error maps belong to evaluation.",
        fill=(19, 45, 53),
        font=font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    page.save(path)
