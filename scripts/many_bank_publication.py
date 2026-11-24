"""Project the original bank deployment receipts and verify both offline witness panels."""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

CASES = {
    "prefit": "b-bank-performance-prefit",
    "fitted": "b-bank-performance-fitted",
    "scale": "b-bank-performance-scale",
    "external": "b-bank-external-acceptance",
    "example": "b-bank-public-example",
}
ACTIONS = ("score", "audit-bank")


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def write_csv(path, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


class Panel(HTMLParser):
    def __init__(self):
        super().__init__()
        self.layers = []

    def handle_starttag(self, tag, attrs):
        from PIL import Image

        args = dict(attrs)
        for name in ("src", "href", "action", "srcset"):
            if name in args:
                check(
                    tag == "img" and name == "src" and args[name].startswith("data:image/png;base64,"),
                    "external HTML resource",
                )
        if tag == "img":
            data = base64.b64decode(args["src"].split(",", 1)[1], validate=True)
            with Image.open(io.BytesIO(data)) as layer:
                check(layer.format == "PNG" and layer.size == (378, 252), "panel grid differs")
                layer.load()
            self.layers.append(args["alt"])


def verify_report(html, metadata, scores, native, dispatch):
    import numpy as np

    receipt = load(metadata)
    check(sha(metadata) == native["score_receipt_sha256"], "different per-mode score receipt")
    check(
        sha(scores) == receipt["arrays"]["sha256"] == native["scores_npz_sha256"],
        "different numeric archive",
    )
    check(sha(html) == native["html_sha256"], "different generated HTML")
    data = html.read_text(encoding="utf-8")
    panel = Panel()
    panel.feed(data)
    check(len(panel.layers) == len(set(panel.layers)) == 6, "expected six local maps")
    fields = re.search(r"const ids=(\[.*?\]),fields=(\[.*?\])\.map\(", data)
    check(fields is not None, "missing interactive source grids")
    ids = json.loads(fields.group(1))
    words = json.loads(fields.group(2))
    check(
        ids == receipt["canonical_source_ids"] and len(ids) == 16 and len(words) == 2,
        "HTML source identities differ",
    )
    check(
        receipt["queried_source_ids"] == dispatch["actual_q"]
        and receipt["model"]["model_sha256"] == dispatch["head_sha256"],
        "bank or head identity differs",
    )
    check(
        receipt["bank_sha256"] == dispatch["bank_sha256"] and receipt["target_id"] == dispatch["target"],
        "incorrect target bank",
    )
    with np.load(scores, allow_pickle=False) as arrays:
        rank, finite = arrays["rank"], arrays["finite"]
        check(
            rank.shape == finite.shape == (252, 378) and np.array_equal(np.isfinite(rank), finite),
            "unexpected finite rank support",
        )
        check(int(finite.sum()) == native["finite_pixels"], "finite count differs")
        state = arrays["scope_state"]
        check(
            {name: int(np.count_nonzero(state == i)) for i, name in enumerate(receipt["scope_names"])}
            == native["scope_counts"],
            "scope pixels differ",
        )
        for name, raw in zip(("first_witness", "gap_witness"), words, strict=True):
            values = np.frombuffer(base64.b64decode(raw, validate=True), dtype="<i4").reshape(252, 378)
            check(
                np.array_equal(values, arrays[name]) and (values >= -1).all() and (values < len(ids)).all(),
                "interactive source witness differs: " + name,
            )
        first, gap = arrays["first_witness"], arrays["gap_witness"]
        check(
            np.array_equal(first >= 0, arrays["comparable_count"] > 0),
            "finite selected witness is inconsistent with source count",
        )
        check(
            np.array_equal(gap >= 0, state == receipt["scope_names"].index("subset_comparability_gap")),
            "unwitnessed audit gap",
        )
        counts = {ids[int(i)]: int(n) for i, n in zip(*np.unique(gap[gap >= 0], return_counts=True))}
        check(counts == native["gap_witness_counts_by_source"], "different per-source audit gaps")
        check(
            not (set(counts) & set(dispatch["actual_q"])),
            "a queried source was listed as the missing witness",
        )
        indices = np.flatnonzero(gap >= 0)
        if not indices.size:
            indices = np.flatnonzero(first >= 0)
        y, x = divmod(int(indices[len(indices) // 2]) if indices.size else 0, 378)
        expected = [ids[int(a[y, x])] if a[y, x] >= 0 else "none" for a in (first, gap)]
        return {
            "mode": "audited" if receipt["audit"]["requested"] else "finite",
            "decoded_grid_images": panel.layers,
            "int32_witness_cells_verified_against_arrays": 2 * finite.size,
            "selected_witness_pixels": int((first >= 0).sum()),
            "gap_witness_pixels": int((gap >= 0).sum()),
            "finite_pixels": int(finite.sum()),
            "canvas_pixels": int(finite.size),
            "rank_word_sha256": hashlib.sha256(rank.astype("<f4", copy=False).tobytes()).hexdigest(),
            "gap_witness_counts_by_source": counts,
            "pointer_test": {"pixel": [x, y], "first": expected[0], "gap": expected[1]},
        }


def compare_html(mac, linux):
    from PIL import Image

    pattern = r"data:image/png;base64,[A-Za-z0-9+/=]+"
    m, l = (p.read_text(encoding="utf-8") for p in (mac, linux))
    mac_layers, linux_layers = (re.findall(pattern, p) for p in (m, l))
    check(len(mac_layers) == len(linux_layers) == 6, "different platform panel count")
    layers = []
    for a, b in zip(mac_layers, linux_layers, strict=True):
        images = []
        for encoded in (a, b):
            with Image.open(io.BytesIO(base64.b64decode(encoded.split(",", 1)[1], validate=True))) as im:
                check(im.mode == "RGB" and im.size == (378, 252), "different reported RGB pane")
                images.append(im.tobytes())
        layers.append({"same_png_bytes": a == b, "same_rgb_pixels": images[0] == images[1]})
    return {
        "identical_html_bytes": sha(mac) == sha(linux),
        "identical_non_png_html_including_witnesses": re.sub(pattern, "[PNG]", m)
        == re.sub(pattern, "[PNG]", l),
        "layers": layers,
    }


def performance_tables(pins, values):
    rows, samples = [], []
    fitted = values["fitted"]
    check([v["requested_thread_ceiling"] for v in fitted["runs"]] == [1, 2, 4], "missing fitted execution")
    check(
        fitted["source_bank"] == values["prefit"]["source_bank"],
        "the registered fitted comparison used a different bank",
    )
    for mode in ("prefit", "fitted"):
        doc = values[mode]
        for r in doc["runs"]:
            cold = r["cold"]
            setup = r["source_preparation_counters"]
            check(
                len(r["hot"]) == 7
                and setup["source_photos_decoded"] == 16
                and setup["source_renders_decoded"] == 16,
                "different prepared workload",
            )
            check(
                all(
                    s["rank_sha256"] == cold["rank_sha256"]
                    and s["scanning"]["cache_hits"] == 16
                    and s["scanning"]["source_bytes_read"] == 0
                    for s in r["hot"]
                ),
                "different warmed workload",
            )
            rows.append(
                {
                    "stage": mode,
                    "case": doc["actual_case"],
                    "model": r["model"],
                    "thread_ceiling": r["requested_thread_ceiling"],
                    "actual_q": len(cold["queried_source_ids"]),
                    "spawn_to_cold_s": r["spawn_to_cold_result_seconds"],
                    "bank_admission_s": r["initial_bank_admission"]["wall_seconds"],
                    "geometry_selection_s": r["geometry_selection"]["wall_seconds"],
                    "cold_scan_and_rank_s": r["cold_cost"]["wall_seconds"],
                    "cold_scan_and_rank_core_s": r["cold_cost"]["cpu_core_seconds"],
                    "cold_scan_s": cold["scanning"]["elapsed_through_snapshot_seconds"],
                    "cold_head_or_rule_s": cold["head_or_rule"]["wall_seconds"],
                    "first_source_preparation_s": r["first_source_preparation"]["wall_seconds"],
                    "first_source_preparation_core_s": r["first_source_preparation"]["cpu_core_seconds"],
                    "first_source_read_bytes": setup["source_bytes_read"],
                    "hot_sample_count": len(r["hot"]),
                    "hot_p50_s": r["hot_service_p50_s"],
                    "hot_p95_s": r["hot_service_p95_s"],
                    "hot_targets_per_s": r["hot_targets_per_second"],
                    "seven_hot_core_s": r["hot_cpu_core_seconds"],
                    "tested_cache_capacity_bytes": r["source_LRU_bytes"],
                    "source_cache_peak_bytes": r["cache"]["cache_peak_bytes"],
                    "native_peak_rss_bytes": r["native_peak_rss_bytes"],
                    "process_cpu_core_s": r["process_cpu_seconds"],
                    "finite_pixels": cold["finite"],
                    "rank_sha256": cold["rank_sha256"],
                    "source_run_id": pins[mode]["run_id"],
                }
            )
            for i, hot in enumerate(r["hot"], start=1):
                samples.append(
                    {
                        "stage": mode,
                        "thread_ceiling": r["requested_thread_ceiling"],
                        "hot_sample": i,
                        "wall_s": hot["wall_seconds"],
                        "cpu_core_s": hot["cpu_core_seconds"],
                        "cache_hits": hot["scanning"]["cache_hits"],
                        "source_decode_bytes": hot["scanning"]["source_bytes_read"],
                        "target_read_bytes": hot["scanning"]["target_bytes_read"],
                        "scan_s": hot["scanning"]["elapsed_through_snapshot_seconds"],
                        "head_or_rule_s": hot["head_or_rule"]["wall_seconds"],
                        "rss_bytes": hot["current_rss_bytes"],
                        "process_threads_after": hot["process_thread_count_after"],
                        "threads_with_observed_cpu_increment": hot[
                            "threads_with_observed_cpu_time_increment"
                        ],
                    }
                )
    scale = values["scale"]["runs"][0]
    scale_rows = []
    for s in scale["full_scan"]:
        cost = s["scan"]
        scale_rows.append(
            {
                "case": values["scale"]["actual_case"],
                "model": scale["model"],
                "actual_q": s["actual_k"],
                "source_photos_decoded_cumulative": cost["source_photos_decoded"],
                "source_renders_decoded_cumulative": cost["source_renders_decoded"],
                "source_read_bytes_cumulative": cost["source_bytes_read"],
                "target_read_bytes_cumulative": cost["target_bytes_read"],
                "source_geometry_projections_cumulative": cost["source_geometry_projections"],
                "source_photo_comparisons_cumulative": cost["source_photometric_comparisons"],
                "scan_through_snapshot_s": cost["elapsed_through_snapshot_seconds"],
                "head_or_rule_s": s["head_or_rule"]["wall_seconds"],
                "lru_peak_bytes": cost["cache_peak_bytes"],
                "native_peak_rss_bytes": s["native_peak_rss_bytes"],
                "rank_sha256": s["rank_sha256"],
            }
        )
    check(
        [x["actual_q"] for x in scale_rows] == [4, 16, 128]
        and scale_rows[-1]["source_photos_decoded_cumulative"]
        == scale_rows[-1]["source_renders_decoded_cumulative"]
        == 128,
        "incomplete nested R128 scan",
    )
    return rows, samples, scale_rows


def outputs_archive(out, source_files):
    archive = out / "evidence-lens-b04-r16-reports.zip"
    with ZipFile(archive, "x") as z:
        for key, path in sorted(source_files.items()):
            info = ZipInfo(key, (2000, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = ZIP_DEFLATED
            with z.open(info, "w") as target, path.open("rb") as source:
                shutil.copyfileobj(source, target)
    with ZipFile(archive) as z:
        check(set(z.namelist()) == set(source_files) and z.testzip() is None, "incomplete reports ZIP")
        check(
            all(hashlib.sha256(z.read(k)).hexdigest() == sha(path) for k, path in source_files.items()),
            "different report ZIP member",
        )
    return {"path": archive.name, "bytes": archive.stat().st_size, "sha256": sha(archive)}


def worker(args):
    cfg = load(args.config)
    pins = cfg["sources"]
    values = {role: load(getattr(args, role)) for role in CASES}
    for role, label in CASES.items():
        doc = values[role]
        check(
            sha(getattr(args, role)) == pins[role]["sha256"] and doc["run_id"] == pins[role]["run_id"],
            "different native receipt for " + label,
        )
        if role in ("prefit", "fitted", "scale"):
            check(doc["mode"] == role, "wrong measured mode")
    project = args.out
    host = cfg["measurement_hosts"]
    for mode, env in host.items():
        check(
            env["member"]["run_id"] == pins[mode]["run_id"]
            and env["member"]["manifest_sha256"] == pins[mode]["manifest_sha256"],
            "different measured host receipt",
        )
    rows, samples, scales = performance_tables(pins, values)
    write_csv(project / "cpu-performance.csv", rows)
    write_csv(project / "hot-samples.csv", samples)
    write_csv(project / "nested-scan.csv", scales)
    example = values["example"]
    panels, members = {}, {}
    for action in ACTIONS:
        index = 0 if action == "score" else 1
        table = {
            name: arg[index]
            for name, arg in (
                ("scores.json", args.metadata),
                ("scores.npz", args.arrays),
                ("report.html", args.html),
            )
        }
        panels[action] = verify_report(
            table["report.html"],
            table["scores.json"],
            table["scores.npz"],
            example["reports"][action],
            example,
        )
        probe = panels[action]["pointer_test"]
        x, y = probe["pixel"]
        result = subprocess.run(
            [args.node, args.panel_script, args.chrome, table["report.html"], str(x), str(y)],
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )
        check(result.returncode == 0, "offline pointer test: " + result.stderr[-900:])
        browser = json.loads(result.stdout)
        expected = f"Pixel ({x}, {y}) · first comparable source {probe['first']} · audit gap witness {probe['gap']}"
        check(browser["displayed_text"] == expected, "browser displayed different source IDs")
        panels[action]["offline_chrome"] = browser
        linux_receipt = load(args.linux_native)
        check(
            sha(args.linux_html[index])
            == linux_receipt["results"][action]["original_and_regenerated_html_sha256"],
            "different Linux report",
        )
        panels[action]["linux_mac_decoded_panel_comparison"] = compare_html(
            table["report.html"], args.linux_html[index]
        )
        members.update({action + "/" + name: path for name, path in table.items()})
    check(
        panels["score"]["rank_word_sha256"] == panels["audit-bank"]["rank_word_sha256"],
        "different finite and audited rank words",
    )
    archives = {"inputs": example["archive"], "outputs": outputs_archive(project, members)}
    ext = values["external"]
    check(
        ext["positive"]["verify_all_cost"]["source_photos_decoded"] == ext["source_count"] == 34
        and len(ext["positive"]["actual_q"]) == 4,
        "external membership differs",
    )
    check(
        ext["incomplete"]["full_complete"] is False
        and ext["bare_ply"]["score_exit"] == ext["incomplete"]["missing_depth_scoring_status"] == 2,
        "incomplete rejection was not captured",
    )
    for role in ("prefit", "fitted", "scale", "external", "example"):
        shutil.copyfile(getattr(args, role), project / (role + ".json"))
    for role in ("linux_native", "linux_comparison", "linux_installation", "linux_inspection"):
        source = getattr(args, role)
        check(sha(source) == pins[role]["sha256"], "different platform receipt")
        shutil.copyfile(source, project / (role.replace("_", "-") + ".json"))
    digests = {r["rank_sha256"] for r in rows if r["stage"] == "fitted"}
    receipt = {
        "schema": "el-b-bank-public-deployment-1",
        "run_id": os.environ["EL_RUN_ID"],
        "parents": pins,
        "measurement_hosts": host,
        "preceding_projection_attempts": cfg["preceding_projection_attempts"],
        "archives": archives,
        "html_panels": panels,
        "fitted_cold_rank_hashes_all_thread_ceilings_equal": len(digests) == 1,
        "all_tested_native_rss_within_2gib": all(
            v["all_rss_within_budget"] for v in (values["prefit"], values["fitted"], values["scale"])
        ),
        "all_tested_lru_peaks_within_512mib": all(
            v["all_source_caches_within_budget"]
            for v in (values["prefit"], values["fitted"], values["scale"])
        ),
        "tables": {
            name: {"rows": n, "sha256": sha(project / name)}
            for name, n in (
                ("cpu-performance.csv", len(rows)),
                ("hot-samples.csv", len(samples)),
                ("nested-scan.csv", len(scales)),
            )
        },
    }
    write_json(project / "deployment.json", receipt)


def run(args):
    from dtu_eval import read_pinned
    from dual_records import bind, resolve, selected, stage

    s = args.store.resolve()
    pins = {role: selected(s, label) for role, label in CASES.items()}
    pins["linux_native"] = selected(s, "b-bank-linux-clean-replay")
    pins["linux_comparison"] = selected(s, "b-bank-linux-mac-comparison")
    linux_run = pins["linux_native"]["run_id"]
    for suffix, name in (("installation", "installation.json"), ("inspection", "inspect.json")):
        pins["linux_" + suffix] = bind(s, linux_run, "payload/" + name)
    native_example = pins["example"]["run_id"]
    cmd = [
        "{python}",
        "-B",
        "{source}/scripts/many_bank_publication.py",
        "worker",
        "--config",
        "{config}",
        "--out",
        "{out}",
    ]
    for role in CASES:
        cmd += ["--" + role, "{input:" + role + "}"]
    for role in ("linux_native", "linux_comparison", "linux_installation", "linux_inspection"):
        cmd += ["--" + role.replace("_", "-"), "{input:" + role + "}"]
    host = {}
    for mode in ("prefit", "fitted", "scale"):
        env_pin = bind(s, pins[mode]["run_id"], "environment.json")
        env = load(resolve((s, s / "history"), env_pin))
        host[mode] = {
            "member": env_pin,
            **{
                k: env[k]
                for k in ("platform", "machine", "cpu_physical", "cpu_logical", "memory_total_bytes")
            },
        }
    predecessors = []
    for label, bucket, evidence, reason in (
        (
            "b-bank-deployment-tables",
            "quarantine",
            "input assets missing from declared parent: environment_prefit",
            "Recorder metadata was submitted as an exported payload member.",
        ),
        (
            "b-bank-deployment-tables-verified",
            "runs",
            "many_bank_panel_smoke.mjs",
            "The browser driver placed a fractional coordinate in the adjacent displayed pixel.",
        ),
    ):
        original = s / "work/controller" / label / "attempt-01.jsonl"
        if not original.is_file():
            continue
        failed = json.loads(original.read_text().splitlines()[-1])
        folder = s / bucket / failed["run_id"]
        check(
            failed["execution_status"] == "failed"
            and evidence in (failed["failure_reason"] + (folder / "stderr.log").read_text()),
            "unexpected previous publication-stage disposition",
        )
        prev = {
            "run_id": failed["run_id"],
            "manifest_sha256": failed["manifest_sha256"],
            "member": "run.json",
            "sha256": sha(folder / "run.json"),
        }
        read_pinned((s, s / "history"), prev, success=False)
        predecessors.append({"stage": label, "execution": prev, "detail": reason})
    panel_script = Path(__file__).with_name("many_bank_panel_smoke.mjs")
    node = shutil.which("node")
    check(node and args.chrome.is_file(), "local Chrome and Node are required for the offline witness test")
    cmd += ["--node", node, "--chrome", str(args.chrome), "--panel-script", "{input:panel_program}"]
    for i, action in enumerate(ACTIONS):
        role = "linux_html" + str(i)
        pins[role] = bind(s, linux_run, f"payload/{action}/report.html")
        cmd += ["--linux-html", "{input:" + role + "}"]
        for key, path in (("html", "report.html"), ("metadata", "scores.json"), ("arrays", "scores.npz")):
            role = key + str(i)
            pins[role] = bind(s, native_example, f"payload/{action}/{path}")
            cmd += ["--" + key, "{input:" + role + "}"]
    return stage(
        s,
        "b-bank-deployment-public",
        "report",
        pins,
        {
            "evidence": "native engineering results, authored example only",
            "worker_sha256": sha(Path(__file__)),
            "measurement_hosts": host,
            "preceding_projection_attempts": predecessors,
        },
        cmd,
        "payload/deployment.json",
        25_000_000,
        extra_files={"panel_program": panel_script},
        retry=args.retry,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["run", "worker"])
    for arg in (
        "store",
        "config",
        "out",
        "linux-native",
        "linux-comparison",
        "linux-installation",
        "linux-inspection",
        *CASES,
    ):
        p.add_argument("--" + arg, type=Path)
    for arg in ("html", "metadata", "arrays", "linux-html"):
        p.add_argument("--" + arg, type=Path, action="append")
    p.add_argument(
        "--chrome", type=Path, default=Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    )
    p.add_argument("--node", type=Path)
    p.add_argument("--panel-script", type=Path)
    p.add_argument("--retry")
    a = p.parse_args()
    if a.action == "run":
        print(json.dumps(run(a)))
    else:
        worker(a)
