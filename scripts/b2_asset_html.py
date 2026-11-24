"""Open the authored and scope-specific pages in an offline browser and exercise every target."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from bundle_inspect_html import CONTROL
from bundle_original import check, put, sha


def check_pages(a):
    policies = {
        "original": (3, 14, "Uncalibrated. The original model cutoff is shown."),
        "external": (3, 14, "outside the supplied calibration scope"),
        "fixed2": (6, 26, "no threshold satisfying the registered conditions"),
        "fixed4": (6, 28, "no threshold satisfying the registered conditions"),
    }
    rows = {}
    for name, (n, images, wording) in policies.items():
        page = getattr(a, name)
        text = page.read_text(encoding="utf-8")
        check(wording in text, "incorrect calibration status: " + name)
        with tempfile.TemporaryDirectory(dir=a.out.parent) as temp:
            screenshot = a.out / (name + "-offline.png")
            p = subprocess.run(
                [
                    shutil.which("node"),
                    "-e",
                    CONTROL,
                    str(a.chrome),
                    str(page.resolve()),
                    temp,
                    str(screenshot),
                ],
                capture_output=True,
                text=True,
                timeout=55,
            )
            check(p.returncode == 0, p.stderr[-500:])
            r = json.loads(p.stdout)
        check(
            r["request_schemes"] == ["data", "file"]
            and r["receipt"]
            and len(r["images"]) == images
            and all(x == {"width": 256, "height": 256} for x in r["images"]),
            "incorrect media resources " + name,
        )
        check(
            len(r["views"]) == n + 1
            and all(x["visible"] == [x["value"]] for x in r["views"][:-1])
            and len(r["views"][-1]["visible"]) == n,
            "view picker " + name,
        )
        rows[name] = {
            "html_sha256": sha(page.read_bytes()),
            "screenshot_sha256": sha(screenshot.read_bytes()),
            "image_count": images,
            "target_count": n,
            "picker_options": n + 1,
            "enabled_schemes": r["request_schemes"],
            "browser": r["product"],
            "receipt_download": True,
        }
    put(a.out / "offline-readback.json", {"schema": "el-b2-offline-browser-1", "passed": True, "pages": rows})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("chrome", "original", "external", "fixed2", "fixed4", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    check_pages(p.parse_args())
