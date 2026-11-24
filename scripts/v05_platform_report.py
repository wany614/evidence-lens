"""Project native Stage A installation and array-comparison receipts into distributable platform tables."""

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import zipfile

from dual_records import bind, resolve, selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from v05_platform_replay import CASES, MODELS, VIEWS


def check_array_bindings(installation, group, profile_path, index_path, archive_path):
    idx = load(index_path)
    require(sha(archive_path) == idx["archive_sha256"], "actual score archive differs from its index")
    matched = 0
    with zipfile.ZipFile(archive_path) as archived:
        for cell in load(profile_path)["cells"]:
            key = cell["tag"] + "/" + cell["method"]
            original = idx["cells"][key]
            require(
                original["scores_sha256"] == cell["scores_sha256"]
                and original["model"] == cell["model"]
                and original["bundle_sha256"] == cell["bundle_sha256"],
                "score set and installed profile differ",
            )
            for view, grade in cell["views"].items():
                raw = original["views"][view]
                require(
                    raw["sha256"] == grade["arrays_sha256"]
                    and hashlib.sha256(archived.read(raw["path"])).hexdigest() == raw["sha256"],
                    "actual cross-host comparison array differs from installed profile grade",
                )
                matched += 1
        for key, npy_digests in installation["groups"][group]["legacy_array_digests"].items():
            method, view = key.split("/")
            raw = idx["cells"]["historical-original/" + method]["views"][view]
            data = archived.read(raw["path"])
            require(hashlib.sha256(data).hexdigest() == raw["sha256"], "changed historical array")
            with zipfile.ZipFile(BytesIO(data)) as npz:
                require(
                    {name: hashlib.sha256(npz.read(name)).hexdigest() for name in npz.namelist()}
                    == npy_digests,
                    "historical numerical array differs from both installed versions' same-host grade",
                )
            matched += 1
        require(
            matched == len(archived.namelist()) == idx["target_arrays"],
            "partial physical profile/legacy readback",
        )
    return matched


def host_record(host, installation, files):
    groups = installation["groups"]
    require(
        set(groups) == ({"default"} if host == "Linux" else {"default", "vision"}),
        "incomplete installed groups",
    )
    rows = {}
    for group, r in groups.items():
        profile = files[group]
        require(sha(profile) == r["profiles_sha256"], "profile report differs from installation receipt")
        profiles = load(profile)
        registered = MODELS if group == "vision" else ("MS-L1",)
        expected = {(tag, m) for tag in CASES for m in registered}
        require(
            len(profiles["cells"]) == len(expected)
            and {(c["tag"], c["method"]) for c in profiles["cells"]} == expected,
            "incomplete native grid/method roster",
        )
        cells = {}
        for c in profiles["cells"]:
            require(
                set(c["views"]) == set(VIEWS) and tuple(c["shape_wh"]) == CASES[c["tag"]],
                "incorrect views or shape",
            )
            paths = 0
            for v in c["views"].values():
                for cap in ("0", str(64 << 20)):
                    require(
                        v[cap]["exact"] and v[cap]["threshold_crossings"] == 0,
                        "file and resident arrays differ",
                    )
                    paths += 1
            require(
                c["html"]["score_sha256"] == c["scores_sha256"]
                and c["html"]["views"] == 6
                and tuple(c["html"]["analysis_wh"]) == CASES[c["tag"]],
                "wrong actual HTML report",
            )
            cells[c["tag"] + "/" + c["method"]] = {
                "analysis_wh": c["shape_wh"],
                "profile": c["profile"],
                "counts": c["counts"],
                "file_views": len(c["views"]),
                "resident_cache_comparisons_exact": paths,
                "html_sha256": c["html"]["html_sha256"],
                "score_sha256": c["scores_sha256"],
                "native_resource": c["cost"],
            }
        n = 6 * len(registered)
        require(
            r["legacy_method_targets_exact"] == n
            and r["legacy_array_pairs_exact"] == 4 * n
            and r["profiles_method_targets"] == 6 * len(expected),
            "inconsistent same-host installation count",
        )
        rows[group] = {
            "methods": list(registered),
            "dependencies": r["versions"],
            "legacy_method_views_exact": n,
            "legacy_channels_exact": 4 * n,
            "new_profile_method_views": 6 * len(expected),
            "resident_comparisons_exact": 12 * len(expected),
            "html_reports": len(expected),
            "legacy_grades": r["legacy_grades"],
            "profiles_sha256": r["profiles_sha256"],
            "install_and_original_times": r["install_and_original_times"],
            "profile_wall_seconds": r["profiles_wall_seconds"],
            "cells": cells,
        }
    return {
        "platform": installation["platform"],
        "architecture": installation["arch"],
        "python": installation["python"],
        "run_id": installation["run_id"],
        "wheel_sha256": installation["wheel_sha256"],
        "old_wheel_sha256": installation["original_wheel_sha256"],
        "examples_sha256": installation["examples_sha256"],
        "original_examples_sha256": installation["original_examples_sha256"],
        "groups": rows,
    }


def neural_summary(doc):
    out = {}
    for method, entry in doc["methods"].items():
        dispatches = {}
        for key in ("cases", "official_default_dispatch_cases"):
            cases = entry[key]
            require(len(cases) == 6, "incomplete official shape roster")
            dispatches[key] = {
                "passing_cases": sum(c["passed"] for c in cases),
                "scheduled": len(cases),
                "passing_v2": sum(c["passed"] for c in cases if c["mode"] == "native-rectangular"),
                "finite_mask_changes": sum(c["finite_mask_changes"] for c in cases),
                "threshold_crossings": sum(c["threshold_crossings"] for c in cases),
                "max_abs_risk": max(c["checks"]["native_probabilities"]["max_abs"] for c in cases),
                "legacy_patch_scalars_outside_tolerance": cases[0]["checks"]["normalized_patches"][
                    "outside_tolerance"
                ],
            }
        out[method] = {
            "model_index_sha256": entry["model_index_sha256"],
            "weights_sha256": entry["weights_sha256"],
            "backbone_tensors": entry["backbone_tensors"],
            "padded_3x2_position_identity": entry["padded_and_3x2_positions_identical"],
            "attention_operators": entry["official_attention_witness"],
            "dispatch": dispatches,
        }
    return {"passed": doc["passed"], "official_commit": doc["official_source"]["commit"], "methods": out}


def render(public):
    rows = [
        "# Installed CPU platform verification",
        "",
        "Each host installed the audited 0.4.0 wheel, replayed the six original bell M4 views, installed 0.5.0 in the same environment, and repeated that replay. Risks, validity, state and unknown-reason NPY records agreed byte for byte across versions on each tested host. Each new installation also scored six prepared bundles, checked its file results against the resident API with its source cache off and on, and rendered an HTML report for each bundle and method. [Platform, dependency and grid receipts](platforms.json)",
        "",
        "| Native CPU and install | Methods | Original method–views exact · channels | Six-profile method–views | Resident checks | HTML reports |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for host, entry in public["platforms"].items():
        for group, g in entry["groups"].items():
            rows.append(
                f"| {host} {entry['architecture']} · {group} | {', '.join(g['methods'])} | {g['legacy_method_views_exact']} · {g['legacy_channels_exact']} | {g['new_profile_method_views']} | {g['resident_comparisons_exact']} | {g['html_reports']} |"
            )
    rows += [
        "",
        "The Linux arm64 container ran with a read-only root, its network disabled and loopback as its only visible interface. Its fresh default installation used NumPy and Pillow. Optional neural installation on Linux remains unmeasured. Mac and Windows installed the vision extra and loaded the complete original MS-frozen and MS-ft tensors. [Mac official-model checks](../../releases/v0.5.0/NEURAL_GRID.md) · [Windows official-model checks](neural-windows.json)",
        "",
        f"The Windows CPU recorder ran for {public['windows_run_window']['recorder_elapsed_seconds']:.3f} seconds. Its window overlapped a separate GPU control for {public['windows_run_window']['gpu_control_overlap']['wall_overlap_seconds_from_utc']:.3f} seconds. The platform table gives observed scoring costs for these installation checks. [Native timing and dependency evidence](platforms.json)",
        "",
        "## Direct comparison with Mac",
        "",
        "Each method contributes 42 paired views, comprising six historical controls and six views on each of the six prepared routes. The comparison reads the two platforms’ original risk, validity, state and unknown-reason arrays. Differences use host minus Mac on common finite support. The two quotas select the lowest 10% and 20% of eligible pixels separately in each view, breaking equal-risk ties by row-major pixel index. The downloadable per-view records also give each four-method common-support comparison where available. [Linux per-view differences](views-linux.json) · [Windows per-view differences](views-windows.json)",
        "",
        "| Host · method | Bit-exact views | Common finite pixels | Changed risk words | Max absolute risk difference | Valid / state / reason changes | Threshold crossings | Changed quotas / tests |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for host, data in public["cross_host"].items():
        for m, d in data["method_totals"].items():
            changed = d["changed_pixels"]
            rows.append(
                f"| {host} · {m} | {d['bit_exact_views']} / {d['views']} | {d['common_finite']:,} | {changed['risk']:,} | {d['max_abs_common']:.7g} | {changed['valid']} / {changed['state']} / {changed['unknown_reason']} | {d['threshold_crossings']} | {d['changed_method_finite_quota_tests']} / {d['method_finite_quota_tests']} |"
            )
    rows += [
        "",
        "The numerical archive records per-grid signed ranges, means, counts and both quota memberships. These comparisons use one authored bell and its fixed deployment evidence. They measure software and numerical reproducibility on the tested hosts. [All per-grid figures and recorder pins](platforms.json)",
        "",
    ]
    return "\n".join(rows)


def project(a):
    files = {k: Path(v) for k, v in a.input}
    cfg = load(a.config)
    hosts = {}
    for host in ("Mac", "Linux", "Windows"):
        h = host.lower()
        groups = ("default",) if host == "Linux" else ("default", "vision")
        install = load(files[h + "_install"])
        hosts[host] = host_record(host, install, {g: files[h + "_" + g] for g in groups})
        retained = groups[-1]
        hosts[host]["retained_arrays_verified_against_installed_grades"] = check_array_bindings(
            install, retained, files[h + "_" + retained], files[h + "_array_index"], files[h + "_arrays"]
        )
    require(
        all(
            len({h[k] for h in hosts.values()}) == 1
            for k in ["wheel_sha256", "old_wheel_sha256", "examples_sha256", "original_examples_sha256"]
        ),
        "hosts used different input bytes",
    )
    build = load(files["wheel_build"])
    require(
        build["assets"]["evidence_lens-0.5.0-py3-none-any.whl"]["sha256"] == hosts["Mac"]["wheel_sha256"],
        "installed wheel differs from the sealed build",
    )
    numbers = {}
    for host in ("Linux", "Windows"):
        h = host.lower()
        d = load(files[h + "_diff"])
        require(
            d["host"] == host and d["views_sha256"] == sha(files[h + "_views"]),
            "wrong native pair or signed per-view result",
        )
        numbers[host] = d
        view_data = load(files[h + "_views"])
        d["four_method_common_quota_totals"] = {}
        for method in d["method_totals"]:
            quotas = [
                q
                for cell in view_data["cells"].values()
                if cell["method"] == method
                for view in cell["views"].values()
                for q in view.get("quota_four_method_common_finite", [])
            ]
            if quotas:
                d["four_method_common_quota_totals"][method] = {
                    "quota_comparisons": len(quotas),
                    "changed_quota_comparisons": sum(bool(q["added"] or q["removed"]) for q in quotas),
                    "members_added": sum(q["added"] for q in quotas),
                    "members_removed": sum(q["removed"] for q in quotas),
                }
        shutil.copyfile(files[h + "_views"], a.out / ("views-" + h + ".json"))
    shutil.copyfile(files["windows_neural"], a.out / "neural-windows.json")
    linux = load(files["linux_native"])
    require(
        linux["network"] == "none"
        and linux["native"]["linux_visible_interfaces"] == ["lo"]
        and linux["readonly_root"],
        "different Linux execution boundary",
    )
    terminal = load(files["windows_terminal"])
    public = {
        "schema": "el-v05-public-platform-acceptance-1",
        "run_id": os.environ["EL_RUN_ID"],
        "provenance": cfg["sources"],
        "wheel_build_code_commit": build["code_commit"],
        "platforms": hosts,
        "windows_run_window": {
            k: terminal[k]
            for k in (
                "start_utc",
                "end_utc",
                "recorder_elapsed_seconds",
                "gpu_control_overlap",
                "dependency_installation",
            )
        },
        "cross_host": numbers,
        "official_neural": {
            h: neural_summary(load(files[h.lower() + "_neural"])) for h in ("Mac", "Windows")
        },
        "linux_container": {
            "image_id": linux["image_id"],
            "network": linux["network"],
            "visible_interfaces": ["lo"],
            "read_only_root": True,
            "offline_wheelhouse": linux["wheelhouse"],
        },
    }
    save(a.out / "platforms.json", public)
    (a.out / "README.md").write_text(render(public), encoding="utf-8")


def control(store):
    sel = lambda label: selected(store, label)
    roots = (store, store / "history")
    win_readback = sel("a-windows-installed-intake")
    win_id = load(resolve(roots, win_readback))["native_run_id"]
    mac = sel("a-mac-install")
    lin = sel("a-linux-installed-default")
    pins = {
        "mac_install": mac,
        "linux_native": lin,
        "windows_readback": win_readback,
        "windows_terminal": bind(store, win_readback["run_id"], "inputs/return_receipt/completed.json"),
        "ticket": sel("a-cpu-platform-ticket"),
        "wheel_build": sel("a-release-build"),
        "mac_neural": sel("a-neural-dispatch-mac"),
        "windows_neural": bind(store, win_id, "payload/neural.json"),
    }
    for h, rid, base in (
        ("mac", mac["run_id"], "payload/"),
        ("linux", lin["run_id"], "payload/accepted/"),
        ("windows", win_id, "payload/"),
    ):
        pins[h + "_install"] = bind(store, rid, base + "installation.json")
        for group in ("default",) if h == "linux" else ("default", "vision"):
            pins[h + "_" + group] = bind(store, rid, base + "profiles-" + group + ".json")
        arr_id = sel("a-mac-array-projection")["run_id"] if h == "mac" else rid
        pins[h + "_array_index"] = bind(store, arr_id, base + "array-index.json")
        pins[h + "_arrays"] = bind(store, arr_id, base + "score-arrays.zip")
    for h in ("linux", "windows"):
        p = sel("a-installed-differences-" + h)
        pins[h + "_diff"] = p
        pins[h + "_views"] = bind(store, p["run_id"], "payload/views.json")
    cmd = [
        "{python}",
        "-B",
        "{source}/scripts/v05_platform_report.py",
        "worker",
        "--config",
        "{config}",
        "--out",
        "{out}",
    ]
    for k in pins:
        cmd.extend(["--input", k, "{input:" + k + "}"])
    return stage(
        store,
        "a-platform-public-tables",
        "report",
        pins,
        {"scope": "native installed A workloads, direct numerical differences only"},
        cmd,
        "payload/platforms.json",
        1_500_000,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("worker", "record"))
    for k in ("store", "config", "out"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--input", nargs=2, action="append", default=[])
    a = p.parse_args()
    if a.action == "worker":
        project(a)
    else:
        print(json.dumps(control(a.store.resolve())), flush=True)
