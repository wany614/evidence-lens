"""The README must use the same wheel, digests and six-view decisions as the release."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from release_model import check_guide_values


def test_guide_matches_all_commands_and_observed_state_counts():
    demo, model, wheel = "1" * 64, "2" * 64, "widget-1.0-py3-none-any.whl"
    guide = (f"pip install ./distribution/{wheel}\n"
             f"grade --manifest-sha256 {demo} --output grade\n"
             f"PowerShell --manifest-sha256 {demo} --output grade\n"
             f"score --model-sha256 {model} --output other\n"
             "1,234 low, 231 high and 50 unknown pixels\n")
    counts = (1234, 231, 50)
    check_guide_values(guide, demo, model, wheel, counts)
    for changed in (guide.replace(demo, model, 1), guide.replace(model, demo),
                    guide.replace(wheel, "different.whl"), guide.replace("50 unknown", "51 unknown")):
        with pytest.raises(ValueError, match="README"):
            check_guide_values(changed, demo, model, wheel, counts)
