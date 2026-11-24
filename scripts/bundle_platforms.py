"""Verify original per-platform records and compare the two wheels' numerical source and assets."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

from bundle_original import check, put, sha


def load(p):
    return json.loads(p.read_bytes())


def compare_wheels(first, second):
    with ZipFile(first) as a, ZipFile(second) as b:

        def members(z):
            return {n: z.read(n) for n in z.namelist() if ".dist-info/" not in n}

        x, y = members(a), members(b)
        changed = sorted(n for n in set(x) | set(y) if x.get(n) != y.get(n))
        check(
            changed
            == sorted("evidence_lens_bundle/" + n for n in ("bundle.py", "cli.py", "common.py", "legacy.py")),
            "unexpected wheel change",
        )
        assets = {n: v for n, v in x.items() if n.startswith("evidence_lens_bundle/assets/")}
        check(len(assets) == 4, "missing released assets")
        return changed, assets, len(x) - len(changed)


def check_checkout(attributes, assets, where):
    def git(*args):
        return subprocess.run(
            ["git", "-c", "core.autocrlf=true", *args], cwd=where, check=True, capture_output=True, text=True
        ).stdout

    git("init", "-q")
    shutil.copyfile(attributes, where / ".gitattributes")
    expected = {}
    for name, raw in assets.items():
        rel = "src/" + name
        dest = where / rel
        dest.parent.mkdir(exist_ok=True, parents=True)
        dest.write_bytes(raw)
        expected[rel] = sha(raw)
    git("add", ".gitattributes", *sorted(expected))
    for name in expected:
        (where / name).unlink()
    git("checkout-index", "--force", *sorted(expected))
    for name, identity in expected.items():
        check(
            sha((where / name).read_bytes()) == identity
            and git("check-attr", "text", "--", name).strip().endswith(": unset"),
            "checkout transformed " + name,
        )
    return expected


def main(a):
    request = load(a.request)
    paths = {k: Path(v) for k, v in a.role}
    check(set(paths) == set(request["roles"]), "incomplete primary records")
    for k, p in paths.items():
        check(sha(p.read_bytes()) == request["roles"][k]["sha256"], "mismatched original " + k)
    changed, assets, unchanged = compare_wheels(paths["wheel2"], paths["wheel3"])
    with tempfile.TemporaryDirectory(dir=a.out.parent) as d:
        restored = check_checkout(paths["attributes"], assets, Path(d))
    rows = []
    for v in ("2", "3"):
        grade = load(paths["mac" + v])
        l = load(paths["linux" + v])
        w = load(paths["win" + v])
        g = load(paths["wingrade" + v])
        e = load(paths["winenv" + v])
        original = load(paths["portable" + v])
        wheel = sha(paths["wheel" + v].read_bytes())
        check(
            w["grade"]["sha256"] == sha(paths["wingrade" + v].read_bytes())
            and w["environment_sha256"] == sha(paths["winenv" + v].read_bytes())
            and w["archive_identity"]["archive_sha256"] == original["archive"]["sha256"]
            and w["archive_identity"]["bundle_sha256"] == l["bundle_sha256"] == original["bundle_sha256"]
            and w["archive_identity"]["wheel_sha256"] == l["wheel_sha256"] == wheel,
            "different platform input",
        )
        check(
            w["status"] == "succeeded"
            and w["grade"]["passed"]
            and g["passed"]
            and grade["fresh_legacy_strict_passed"],
            "platform grade failed",
        )
        sources = [
            (
                "macOS",
                [
                    r["fresh_legacy"]
                    for r in grade["rows"]
                    if r["condition"] == "M4" and r["method"] == "MS-L1"
                ],
            ),
            ("Windows", list(g["views"].values())),
            (
                "Linux",
                [
                    {
                        "canvas": r["pixels"],
                        "finite_together": r["compared_finite"],
                        "max_absolute_difference": r["max_abs"],
                        "state_disagreements": r["mismatched_state"],
                        "finite_disagreements": r["mismatched_valid"],
                    }
                    for r in l["comparisons"]
                ],
            ),
        ]
        for system, views in sources:
            check(
                len(views) == 6
                and all(
                    not x["state_disagreements"]
                    and not x["finite_disagreements"]
                    and x["max_absolute_difference"] <= 1e-4
                    for x in views
                ),
                "incomplete overlap " + system,
            )
            rows.append(
                {
                    "version": "0.4.0a" + v,
                    "platform": system,
                    "method": "MS-L1",
                    "condition": "M4",
                    "views": 6,
                    "pixels": sum(x["canvas"] for x in views),
                    "finite_pairs": sum(x["finite_together"] for x in views),
                    "max_abs": max(x["max_absolute_difference"] for x in views),
                    "state_changes": sum(x["state_disagreements"] for x in views),
                    "support_changes": sum(x["finite_disagreements"] for x in views),
                    "finite_bits_different": sum(x["finite_float32_bits_differ"] for x in views)
                    if system != "Linux"
                    else None,
                }
            )
        check(e["installed"]["evidence-lens"] == "0.4.0a" + v, "installed wrong Windows version")
    old = load(paths["before"])
    new = load(paths["after"])
    check(
        old["results"].keys() == new["results"].keys()
        and all(
            q["status"] == "invalid" and q["exit"] == 2 and not q["traceback"] and not q["output_created"]
            for q in new["results"].values()
        ),
        "malformed boundary disagreement",
    )
    put(
        a.out / "platform-verification.json",
        {
            "schema": "el-bundle-platform-acceptance-1",
            "run_id": os.environ.get("EL_RUN_ID"),
            "engineering_only": True,
            "rows": rows,
            "unchanged_wheel_payload_members": unchanged,
            "changed_payload_members": changed,
            "asset_autocrlf_checkout": restored,
            "a2_original_diagnostics": old["results"],
            "a3_diagnostics": new["results"],
            "linux_grade": "maximum absolute FP32 difference and exact support/state; separate finite word comparison was not recorded by that worker",
            "origins": request["roles"],
            "engineering_failures": request["engineering_failures"],
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--role", action="append", nargs=2, required=True)
    main(p.parse_args())
