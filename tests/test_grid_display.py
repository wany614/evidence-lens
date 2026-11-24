"""Exercise the exporter on a larger display without changing numerical results."""

import base64
import io
import re

import numpy as np
import pytest
from PIL import Image

from evidence_lens_bundle.cli import main
from evidence_lens_bundle.common import BundleError
from evidence_lens_bundle.report import STATE_COLORS, render
from evidence_lens_bundle.scoring import score
from test_bundle import fixture


def test_large_display_preserves_classes_and_analysis_counts(tmp_path):
    root = fixture(tmp_path / "bundle", analysis_profile="analysis-448x252-v1")
    scored = tmp_path / "scored"
    r = score(root, scored)
    html = tmp_path / "large.html"
    assert main(["report", "--scores", str(scored), "--display-size", "896", "504", "--out", str(html)]) == 0
    text = html.read_text()
    assert "448 × 252" in text and "896 × 504" in text
    assert r["counts"]["canvas"] == 2 * 448 * 252
    images = re.findall(r'src="data:image/png;base64,([^"]+)" alt="Threshold state [^"]+"', text)
    assert len(images) == 2
    palette = set(STATE_COLORS.values())
    for i, raw in enumerate(images):
        with Image.open(io.BytesIO(base64.b64decode(raw, validate=True))) as im:
            assert im.size == (896, 504)
            rgb = np.asarray(im)
            assert {tuple(row) for row in np.unique(rgb.reshape(-1, 3), axis=0)} <= palette
        with np.load(scored / r["views"][i]["arrays"]["path"]) as stored:
            state = stored["state"]
        for key, colour in STATE_COLORS.items():
            assert np.array_equal((rgb[::2, ::2] == colour).all(-1), state == key)
    with pytest.raises(BundleError):
        render(scored, tmp_path / "distorted.html", display_override=(1024, 1024))
