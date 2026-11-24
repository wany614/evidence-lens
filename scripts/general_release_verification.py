"""Reconcile the tagged assets and fresh public replays into portable per-view verification tables."""
import argparse
from collections import Counter
from pathlib import Path
import os

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from general_acceptance_tables import project

METHODS = {"B-L1-transfer", "MS-L1", "MS-frozen", "MS-ft"}


def public_files(path, manifest, manifest_sha, required_roles):
    report = load(path)
    require((report["code_commit"], report["tag"]) == (manifest["code_commit"], manifest["tag"]),
            "anonymous download refers to another release")
    assets = manifest["assets"]
    expected = {name for name, entry in assets.items() if entry["role"] in required_roles}
    expected |= {"release-manifest.json", "SHA256SUMS"}
    rows = report["anonymous_requests"]
    require(expected <= {row["name"] for row in rows} and len({row["name"] for row in rows}) == len(rows),
            "missing or repeated public download")
    for row in rows:
        name = row["name"]
        require(row["anonymous"] and row["http_status"] == 200 and row["environment_proxy"] is False
                and row["url"] == f'https://github.com/wany614/evidence-lens/releases/download/{manifest["tag"]}/{name}',
                "download boundary differs")
        if name == "release-manifest.json":
            require(row["sha256"] == manifest_sha, "different public manifest")
        elif name == "SHA256SUMS":
            import hashlib
            sums = {n: asset["sha256"] for n, asset in assets.items()}
            sums["release-manifest.json"] = manifest_sha
            lines = "".join(f"{sums[n]}  {n}\n" for n in sorted(sums)).encode("ascii")
            require(row["sha256"] == hashlib.sha256(lines).hexdigest() and row["bytes"] == len(lines),
                    "different public digest list")
        else:
            require(name in assets and row["sha256"] == assets[name]["sha256"]
                    and row["bytes"] == assets[name]["bytes"], "different downloaded asset")
    return report


def platform_tables(root, out, manifest, expected_os, coverage):
    grade = load(root / "acceptance.json")
    require(grade["tag"] == manifest["tag"] and grade["code_commit"] == manifest["code_commit"]
            and grade["platform"]["os"] == expected_os and grade["cuda_visible_devices"] == "-1",
            "clean replay platform or release differs")
    roles = {d["role"]: d["sha256"] for d in manifest["assets"].values()}
    require(grade["wheel_sha256"] == roles["wheel"] and grade["archive_sha256"] == roles["original_replays"]
            and set(grade["groups"]) == set(coverage), "different wheel, examples or groups")
    project(root, out)
    summaries = {}
    for group, methods in coverage.items():
        receipt = grade["groups"][group]
        replay = load(root / group / "acceptance.json")
        require(receipt["replay_completed"] and set(replay["output"]) == methods
                and replay["gold_index_sha256"] == manifest["original_gold_sha256"], "incomplete model replay")
        groups = {}
        for method in sorted(methods):
            views = replay["output"][method]["views"]
            require(len(views) == 6, "missing original four-source views")
            diffs = [v["difference"] for v in views.values()]
            changes = Counter()
            for d in diffs:
                changes.update(d["changed_pixels"])
            anchors = [q for v in views.values() for q in v["matched_coverage"]["anchors"]]
            reached = [q for q in anchors if q["old_attainable"] and q["new_attainable"]]
            groups[method] = {
                "views": len(views), "bit_exact_views": sum(d["bit_exact"] for d in diffs),
                "changed_pixels": dict(changes), "threshold_crossings": sum(d["threshold_crossings_common_finite"] for d in diffs),
                "max_abs_risk_difference": max((d["max_abs_difference_common_finite"] for d in diffs
                                                if d["max_abs_difference_common_finite"] is not None), default=None),
                "attainable_quotas": len(reached),
                "quota_membership_changed": sum(bool(q["added_pixels"] or q["removed_pixels"]) for q in reached),
            }
        summaries[group] = groups
    return {"platform": grade["platform"], "groups": summaries, "acceptance_sha256": sha(root / "acceptance.json"),
            "per_view_quota_table_sha256": sha(out / "numerical-differences.csv")}


def verify(a):
    spec = load(a.manifest)
    msha = sha(a.manifest)
    compared = {}
    for key, directory, os_name, methods in (
        ("mac", a.mac, "Darwin", {"default": {"MS-L1"}, "vision": METHODS}),
        ("windows", a.windows, "Windows", {"default": {"MS-L1"}, "vision": {"MS-ft"}}),
        ("linux", a.linux, "Linux", {"default": {"MS-L1"}}),
    ):
        compared[key] = platform_tables(directory, a.out / key, spec, os_name, methods)
    anon = public_files(a.mac_readback, spec, msha, {v["role"] for v in spec["assets"].values()})
    win = public_files(a.windows_readback, spec, msha, {"wheel", "original_replays"})
    tests = load(a.tests)
    require(tests["source_commit"] == spec["code_commit"] and tests["skipped"] == 0 and tests["tests_passed"] > 0,
            "different release tests")
    linux = load(a.linux_receipt)
    require(linux["acceptance_sha256"] == compared["linux"]["acceptance_sha256"]
            and linux["network"] == "none" and linux["container_architecture"] == "arm64",
            "different Linux readback")
    release = load(a.publication)
    require(release["draft"] is False and release["code_commit"] == spec["code_commit"]
            and release["manifest_sha256"] == msha, "unpublished or different release")
    config = load(a.config)
    save(a.out / "verification.json", {
        "schema": "el-general-public-verification-1", "run_id": os.environ["EL_RUN_ID"],
        "tag": spec["tag"], "code_commit": spec["code_commit"], "model_release": spec["model_release"],
        "manifest_sha256": msha, "release": release, "provenance": config["sources"],
        "anonymous_mac": anon, "anonymous_windows": win, "linux_image": linux["image"],
        "linux_download_transport": linux["distribution_boundary"],
        "compiled_tests": tests, "original_replays": compared,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "manifest", "mac", "windows", "linux", "mac-readback", "windows-readback",
                "linux-receipt", "tests", "publication", "out"):
        p.add_argument("--" + key, required=True, type=Path)
    verify(p.parse_args())
