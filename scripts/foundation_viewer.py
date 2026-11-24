"""Build and check the offline authored-case viewer from its positive native inventories."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from dual_records import bind, selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from foundation_case_data import verify_local
from foundation_case_io import ALLOWED_CASES, Native, decode_png

LABEL = "c-foundation-viewer"
METHODS = ("MS-L1", "MS-ft", "MS-frozen", "B-L1-transfer")
BUDGETS = (4, 8, 16, 32, 128)
SCREENSHOTS = ("picture", "assessment", "sources", "evaluation", "mobile")
OUTPUTS = ("index.html", "viewer-check.json", *("verification/" + x + ".png" for x in SCREENSHOTS))
HEX = re.compile(r"[0-9a-f]{64}")
FOLDER = Path(__file__).resolve().parent
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def check(value, message):
    require(value, "foundation viewer: " + message)


def content_record(path, name):
    return dict(path=name, bytes=path.stat().st_size, sha256=sha(path))


def parents(store):
    data = selected(store, "c-foundation-case-data")
    return dict(case_data=data, case_readback=selected(store, "c-foundation-case-readback"),
                case_files=bind(store, data["run_id"], "payload/case-files.json"),
                source_photos=bind(store, data["run_id"], "payload/source-photos.json"),
                registration=selected(store, "c-foundation-registration"), A=selected(store, "a-baseline"),
                B=selected(store, "b-research-release"))


def check_contract(data, archive, pins):
    check(data["schema"] == "el-foundation-case-index-1" and data["registration"] == pins["registration"]
          and data["B_research_release"] == pins["B"] and data["B_status"] == "completed_negative"
          and data["A_BASELINE_ID"] == "sha256:" + pins["A"]["sha256"], "case lineage")
    check(data["offline_inventory_sha256"] == pins["case_files"]["sha256"]
          and data["full_resolution_inventory_sha256"] == pins["source_photos"]["sha256"]
          and archive["archive"] == data["source_photo_archive"], "native inventories")
    check(tuple(c["id"] for c in data["cases"]) == ALLOWED_CASES, "registered case identifiers")
    entries = {r["member"]: r for r in archive["files"]}
    check(len(entries) == archive["archive"]["member_count"] == len(archive["files"]), "archive member roster")
    summary = []
    for case in data["cases"]:
        bell = case["id"] == ALLOWED_CASES[0]
        sources = case["source_ids"]
        n = 4 if bell else 128
        check(len(sources) == len(set(sources)) == n and sources == [r["id"] for r in case["sources"]],
              "source identities or order")
        check(case["rights"] == data["rights"] == archive["rights"] and
              case["rights"]["license"] == "CC-BY-4.0" and "synthetic" in case["rights"]["material"],
              "authored synthetic image grants")
        for s in case["sources"]:
            for rec in (s["analysis"], s["original"]):
                check(rec == entries[rec["member"]], "source archive identity")
        views = [f"v{i:02}" for i in range(1, 7)] if bell else ["h00"]
        check([v["id"] for v in case["views"]] == views, "registered targets")
        for v in case["views"]:
            prediction = v["target_prediction_sha256"]
            original = v["native_prediction"]
            check(original == entries[original["member"]] and original["sha256"] == prediction
                  and HEX.fullmatch(v["target_rgb_sha256"]), "native prediction and raw RGB identity")
            cs = v["conditions"]
            check(([q["method"] for q in cs] == list(METHODS) if bell else
                   [q["k"] for q in cs] == list(BUDGETS)), "method or budget roster")
            e = v["evaluation"]
            ec = e["counts"]
            w, h = v["target"]["size"]
            check(e["error_threshold"] == .05 and 0 <= ec["errors"] <= ec["labelled"] <= ec["reference"] <= w*h
                  and e["role"] == "independent evaluation only" and e["reference"]["size"] == e["error"]["size"]
                  == [w, h] and bool(e["origin"]), "independent measurement and denominator")
            prev, first, prev_comparable = [], None, None
            items = []
            for q in cs:
                ids, c = q["query_ids"], q["counts"]
                check(ids == q["actual_q"] and len(ids) == len(set(ids)) == q["k"] and set(ids) <= set(sources)
                      and q["available_count"] == q["reconstruction_count"] == n, "actual source scope")
                check(q["target_prediction_sha256"] == prediction and c["finite"] + c["unknown"] == c["canvas"]
                      == w*h and c["reference"] == ec["reference"], "fixed target or finite support")
                check(all(q[key]["size"] == [w, h] for key in ("rank", "state", "reason")), "assessment grids")
                for key in ("state", "reason"):
                    counts = c[key + "_counts"]
                    check(sum(counts.values()) == w*h and all(a >= 0 for a in counts.values())
                          and set(counts) == {r["label"] for r in q["legends"][key]}, "native category totals")
                item = dict(condition=q["id"], actual_query_ids=ids, method=q["method"], requested_k=q["requested_k"],
                            finite=c["finite"], unknown=c["unknown"], canvas=c["canvas"])
                if bell:
                    check(ids == (["s00"] if q["method"] == "B-L1-transfer" else sources)
                          and c["low"] + c["high"] == c["finite"] and q["threshold"] is not None,
                          "frozen bell scope")
                else:
                    comp = c["query_has_comparable"]
                    first = comp if first is None else first
                    previous = comp if prev_comparable is None else prev_comparable
                    check(q["method"] == "B-CPU-LR" and q["full_bank_audited"] and q["threshold"] is None
                          and ids[:len(prev)] == prev and ids[len(prev):] == q["new_query_ids"]
                          and c["new_comparable_since_q4"] == comp - first
                          and c["new_comparable_since_previous"] == comp - previous >= 0
                          and sum(q["gap_witness_source_counts"].values()) == c["selected_gap"]
                          and not (set(q["gap_witness_source_counts"]) & set(ids)), "nested audit or witness records")
                    prev, prev_comparable = ids, comp
                    item.update({key: c[key] for key in ("query_has_comparable", "selected_gap", "library_no_comparable",
                                            "new_comparable_since_q4", "new_comparable_since_previous")})
                    item["pixels_unchanged_in_comparability_from_q4"] = w*h - (comp-first)
                items.append(item)
            summary.append(dict(case=case["id"], view=v["id"], target_prediction_sha256=prediction,
                                target_rgb_sha256=v["target_rgb_sha256"], evaluation_counts=ec, conditions=items))
    return summary


def render(template, data):
    source = template.read_text()
    check(source.count("__CASE_DATA__") == 1, "one embedded case data block")
    inline = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    inline = inline.replace("<", r"\u003c").replace(">", r"\u003e").replace("&", r"\u0026")
    check(json.loads(inline) == data, "embedded data round trip")
    return source.replace("__CASE_DATA__", inline), inline


def build(store, pins, out, node, chrome, template, driver, run_id):
    native = Native(store, pins["case_data"])
    data = native.metadata
    inv = load(native.path("case-files.json", pins["case_files"]["sha256"]))
    archive = load(native.path("source-photos.json", pins["source_photos"]["sha256"]))
    reg = Native(store, pins["registration"]).metadata
    check(reg["parents"]["A"] == pins["A"] and reg["parents"]["B"] == pins["B"]
          and Native(store, pins["B"]).metadata["status"] == "completed_negative", "registered release sequence")
    readback = Native(store, pins["case_readback"]).metadata
    check(readback["parents"]["case"] == pins["case_data"] and readback["passed"] is True
          and readback["case_index_sha256"] == pins["case_data"]["sha256"], "independent native pixel readback")
    summary = check_contract(data, archive, pins)
    verify_local(native.path("case-index.json").parent, data, inv)
    html, inline = render(template, data)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verification").mkdir(exist_ok=True)
    (out / "index.html").write_text(html)
    with tempfile.TemporaryDirectory(prefix="foundation-exhibit-") as temp:
        root = Path(temp)
        shutil.copyfile(out / "index.html", root / "index.html")
        shutil.copyfile(native.path("case-index.json"), root / "case-index.json")
        for r in inv["files"]:
            src = native.path(r["path"], r)
            dest = root / r["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dest)
            except OSError:
                shutil.copyfile(src, dest)
        proc = subprocess.run([node, driver, chrome, root / "index.html", out / "verification"],
                              capture_output=True, text=True, check=False, timeout=175)
        check(proc.returncode == 0, "offline browser failed: " + proc.stderr[-4000:])
        browser = json.loads(proc.stdout)
    save(out / "viewer-check.json", browser)
    check(set(f.relative_to(out).as_posix() for f in out.rglob("*") if f.is_file()) == set(OUTPUTS),
          "unexpected or missing public output")
    digest = lambda x: hashlib.sha256(x.encode()).hexdigest()
    css, = re.findall(r"<style>([\s\S]*?)</style>", html)
    check(browser["external_requests"] == 0 and not browser["script_errors"] and
          set(browser["screenshots"]) == set(SCREENSHOTS), "browser verification scope")
    files = [content_record(out / name, name) for name in OUTPUTS]
    for row in files:
        name = row["path"]
        row["rights"] = "CC-BY-4.0, wany614, Evidence Lens. Authored measurements and generated exhibit materials."
        if name == "index.html":
            row["rights"] += " Embedded interface implementation under Apache-2.0."
        if name.endswith(".png"):
            shot = decode_png((out / name).read_bytes())
            spec = browser["screenshots"][Path(name).stem]
            row["pixel_sha256"] = hashlib.sha256(shot.tobytes()).hexdigest()
            row["size"] = list(shot.shape[1::-1])
            check(row["size"] == [spec["width"], spec["height"]], "RGB8 screenshot dimensions")
    save(out / "case-viewer.json", dict(schema="el-foundation-offline-viewer-1", run_id=run_id, parents=pins,
          files=files, local_case_image_count=len(inv["files"]), source_photo_archive=archive["archive"],
          rights=dict(generated_exhibit="CC-BY-4.0, wany614, Evidence Lens. The generated exhibit, text, numbers, readbacks and screenshots.",
                      implementation="Apache-2.0 for generator, source template, CSS and browser scripts, including embedded implementation.",
                      images=data["rights"], screenshots="CC-BY-4.0. Original offline browser captures of authored synthetic case material."),
          inputs=dict(case_index_sha256=pins["case_data"]["sha256"], positive_local_png_inventory=pins["case_files"],
                      positive_archived_native_inventory=pins["source_photos"], template_sha256=sha(template),
                      browser_driver_sha256=sha(driver), generator_sha256=sha(Path(__file__))),
          inline_css_sha256=digest(css), embedded_case_json_sha256=digest(inline),
          operating_scope="Offline synthetic camera scenes. Rank availability, geometry comparability and measured RGB error each use their own criterion.",
          independent_evaluation="Distinct opt-in action loads independent reference and measured per-pixel RGB-error image. Changing condition or view closes it.",
          summaries=summary, b_scientific_status=data["B_status"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("record", "preview", "worker", "copy-offline"))
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--template", type=Path, default=FOLDER / "foundation_viewer.html.tmpl")
    p.add_argument("--driver", type=Path, default=FOLDER / "foundation_viewer_smoke.mjs")
    p.add_argument("--node", default=shutil.which("node"))
    p.add_argument("--chrome", type=Path, default=CHROME)
    p.add_argument("--retry")
    a = p.parse_args()
    s = a.store.resolve()
    if a.mode == "record":
        cmd = ["{python}", "-B", "{source}/scripts/foundation_viewer.py", "worker", "--store", str(s),
               "--config", "{config}", "--out", "{out}", "--template", "{input:template}", "--driver",
               "{input:driver}", "--node", a.node, "--chrome", str(a.chrome)]
        config = dict(generator_sha256=sha(Path(__file__)), viewer="three progressive layers plus gated independent evaluation",
                      local_assets="sealed case-data positive RGB8 PNG inventory", cases=list(ALLOWED_CASES))
        stamp = stage(s, LABEL, "report", parents(s), config, cmd, "payload/case-viewer.json", 6_000_000,
                      extra_files={"template": a.template, "driver": a.driver}, retry=a.retry)
        print(json.dumps(stamp, indent=2))
    elif a.mode == "copy-offline":
        native = Native(s, selected(s, LABEL))
        check({r["path"] for r in native.metadata["files"]} == set(OUTPUTS), "published reader allowlist")
        check(a.out and a.out.is_dir(), "supply existing extracted case-data folder")
        for item in native.metadata["files"] + [dict(path="case-viewer.json", sha256=native.pin["sha256"])]:
            name = item["path"]
            src, dst = native.path(name, item), a.out / name
            dst.parent.mkdir(exist_ok=True, parents=True)
            if dst.exists():
                check(sha(dst) == item["sha256"], "existing checkout differs: " + name)
            else:
                shutil.copyfile(src, dst)
    else:
        check(a.out, "--out required")
        pins = load(a.config)["sources"] if a.mode == "worker" else parents(s)
        run_id = os.environ["EL_RUN_ID"] if a.mode == "worker" else "local-layout-preview"
        build(s, pins, a.out, a.node, a.chrome, a.template, a.driver, run_id)


if __name__ == "__main__":
    main()
