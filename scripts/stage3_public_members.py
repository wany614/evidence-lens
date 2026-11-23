"""Produce the two publication allowlists from positively named original Stage C scientific outputs."""

import argparse
import json
from pathlib import Path
import sys

CAT = "f14ca80777cf72e6f99075fb860e24cd38a6b4d9db342011173e9b216d3c66f5"
STUDY = "dd227c25648af4ab5e4e42d3fdd07702a8e46b3def36810ac9d65c55b07a7b23"
OPENING = "50b5cf30513ab739d60f44e404e914cb08681e2950bd91c058d6f328ddbd3026"
FAMILY = "scene_057_offset_bell_cage"
METHODS = ("B-L1-transfer", "MS-L1", "MS-frozen", "MS-ft")
CONDITIONS = ("M2", "M4")
VIEWS = tuple(f"v{i:02}" for i in range(1, 7))
GOLD = "CC-BY-4.0"


def imports(project):
    sys.path.insert(0, str(project / "scripts"))
    sys.path.insert(0, str(project / "src"))


def read(path):
    return json.loads(path.read_bytes())


def require(x, reason):
    if not x:
        raise ValueError(reason)


def pin(j):
    return {k: j[k] for k in ("run_id", "manifest_sha256")}


def work(args):
    from evidence_lens.scoring_io import new_output, write_json
    from evidence_lens.stage3_gate import get_frozen_method, require_opening
    from vision_sources import find_run, member, signed
    from vision_release import safe_name
    from release_model import safe_text

    req = signed(args.request, args.request_sha256)
    require(
        req["schema"] == "el-stage3-release-allow-request-1"
        and req["catalog"] == CAT
        and req["opening"] == OPENING
        and req["purpose"] in ("example", "numerical"),
        "wrong grant",
    )
    roots = (args.store.resolve(), args.shelf.resolve())
    mk = require_opening(roots[0], CAT, opening_sha256=OPENING, shelf=roots[1])
    method = get_frozen_method(roots[0], mk, roots[1])
    require(mk["method"] == req["method"], "wrong frozen method")
    expected_keys = (
        {"authored_case", "world_replay", "expected", *("native_" + m for m in METHODS)}
        if req["purpose"] == "example"
        else {"report"}
    )
    require(set(req["sources"]) == expected_keys, "missing or extraneous publication source")
    rows = {}

    def get(key):
        spec = req["sources"][key]
        r = find_run(roots, spec)
        require(r.run["execution_status"] == "succeeded", "no sealed publication original")
        path, info = member(r, spec["member"], spec["sha256"])
        return r, read(path), info

    def add(to, r, path, expect, license=GOLD, meta=None, duplicate_ok=False):
        require(safe_name(to) == to and license in (GOLD, "Apache-2.0"), "unsafe public member")
        p, g = member(r, path, expect)
        row = {
            "path": to,
            "license": license,
            "bytes": g["bytes"],
            "source": {**pin(r.pin), "member": path, "sha256": g["sha256"]},
        }
        if meta:
            row.update({k: meta[k] for k in ("original_source", "derived_from") if k in meta})
        if to in rows:
            require(duplicate_ok and rows[to] == row, "conflicting or duplicate public path " + to)
        if p.suffix in (".json", ".jsonl", ".csv", ".md", ".txt", ".toml"):
            safe_text(to, p.read_bytes())
        rows[to] = row

    def byref(to, r, base, obj):
        name = safe_name(obj["path"])
        add(to + name, r, base + name, obj, duplicate_ok=True)

    if req["purpose"] == "example":
        cr, ledger, _ = get("authored_case")
        allow = signed(args.authored_grant, req["authored_grant_sha256"])
        require(
            ledger["schema"] == "el-stage3-public-example-origins-1"
            and allow["family"] == ledger["science"]["family"] == FAMILY
            and ledger["science"]["catalog_sha256"] == CAT
            and set(ledger["files"]) == {x["path"] for x in allow["files"]}
            and len(allow["files"]) == len(ledger["files"]),
            "authored publication roster differs",
        )
        for x in allow["files"]:
            n = x["path"]
            src = x["source"]
            orig = ledger["files"][n]
            require(
                n.startswith(("CASE/", "REFERENCE/"))
                and src["member"] == "payload/" + n
                and pin(src) == pin(cr.pin)
                and x["license"] == orig["license"]
                and member(cr, src["member"], src["sha256"])[1]["bytes"] == x["bytes"],
                "original authored case provenance differs: " + n,
            )
            for k in ("original_source", "derived_from"):
                if k in x:
                    require(
                        x[k] == orig.get({"original_source": "original", "derived_from": "derived_from"}[k]),
                        "authored derivation mismatch",
                    )
            add(n, cr, src["member"], src["sha256"], x["license"], x)
        wr, world, _ = get("world_replay")
        require(
            world["schema"] == "el-stage3-standalone-public-world-replay-1"
            and world["subject"] == FAMILY
            and world["passed"]
            and pin(cr.pin) in read(wr.open("inputs.json"))["parents"]
            and set(world["comparisons"]) == {"s00", "s01", "s02", "s03", *VIEWS},
            "wrong original-world replay",
        )
        add(
            "REFERENCE/proofs/public-world-replay.json",
            wr,
            "payload/public-world-replay.json",
            req["sources"]["world_replay"]["sha256"],
        )
        ex, example, _ = get("expected")
        require(
            example["schema"] == "el-stage3-public-expected-example-1"
            and example["family"] == FAMILY
            and example["catalog_sha256"] == CAT
            and example["study_sha256"] == STUDY
            and set(example["scores"]) == set(METHODS)
            and [(z["condition"], z["view"]) for z in example["cases"]]
            == [(c, v) for c in CONDITIONS for v in VIEWS],
            "wrong scientific expected example",
        )
        require(
            ex.run["experiment_id"] == "stage3-public-expected-example" and "display_reissue" in example,
            "missing figure legend correction",
        )
        add(
            "EXPECTED/expected-example.json",
            ex,
            "payload/expected-example.json",
            req["sources"]["expected"]["sha256"],
        )
        byref("EXPECTED/", ex, "payload/", example["display_reissue"]["qa"])
        for z in example["cases"]:
            for k in ("board", "error"):
                byref("EXPECTED/", ex, "payload/", z[k])
        for m in METHODS:
            spec = example["scores"][m]
            byref("EXPECTED/", ex, "payload/", spec)
            gp, _ = member(ex, "payload/" + spec["path"], spec)
            sci = read(gp)
            nr, nd, _ = get("native_" + m)
            add(
                "EXPECTED/" + m + "/native-scores.json",
                nr,
                "payload/native-scores.json",
                req["sources"]["native_" + m]["sha256"],
            )
            for base, r, doc, sch in [
                ("EXPECTED/", ex, sci, "el-stage3-public-expected-scores-1"),
                ("EXPECTED/" + m + "/", nr, nd, "el-stage3-native-example-scores-1"),
            ]:
                require(
                    doc["schema"] == sch
                    and doc["family"] == FAMILY
                    and doc["method"] == m
                    and all(
                        doc[k] == method["models"][m][k]
                        for k in ("threshold", "model_id", "model_index_sha256", "weights_sha256")
                    )
                    and len(doc["views"]) == 12
                    and {(v["condition"], v["view"]) for v in doc["views"]}
                    == {(c, v) for c in CONDITIONS for v in VIEWS},
                    "different native or original scores",
                )
                for v in doc["views"]:
                    require(
                        set(v["modes"]) == {"one", "two", "all"} and v["status"] == "ready",
                        "missing native example scope",
                    )
                    for scope, val in v["modes"].items():
                        require(
                            val["alias_of"] == ("two" if (v["condition"], scope) == ("M2", "all") else None),
                            "wrong M2 alias",
                        )
                        byref(base, r, "payload/", val["file"])
                        if val["alias_of"] is not None:
                            counterpart = v["modes"][val["alias_of"]]
                            require(val["file"] == counterpart["file"], "M2 alias changed its original file")
                            if "receipt" in val:
                                require(
                                    val["receipt"] == counterpart.get("receipt"),
                                    "M2 alias changed its original receipt",
                                )
                        elif r is nr:
                            byref(base, r, "payload/", val["receipt"])
    else:
        rr, j, _ = get("report")
        require(
            j["schema"] == "el-stage3-joint-study-report-1"
            and j["method_sha256"] == mk["method_sha256"]
            and j["opening_sha256"] == OPENING,
            "wrong public study report",
        )
        spec = j["public_numerical"]
        require(
            rr.run["run_kind"] == "report"
            and rr.run["experiment_id"] == "stage3-joint-study-report"
            and spec["path"] == "public-numerical/manifest.json",
            "wrong numerical allowlist",
        )
        pp, _ = member(rr, "payload/" + spec["path"], spec)
        idx = read(pp)
        require(
            idx["schema"] == "el-stage3-public-joint-manifest-1"
            and idx["source_run_id"] == rr.pin["run_id"]
            and idx["catalog_sha256"] == CAT
            and idx["study_sha256"] == STUDY
            and idx["heldout"] == j["heldout"],
            "joint manifest differs",
        )
        add("manifest.json", rr, "payload/" + spec["path"], spec)
        for fs in idx["files"]:
            name = safe_name(fs["path"])
            require(
                "/" not in name and Path(name).suffix in (".json", ".csv", ".md"),
                "unexpected scientific report member",
            )
            add(name, rr, "payload/public-numerical/" + name, fs)
    result = {
        "schema": "el-stage3-public-members-1",
        "purpose": req["purpose"],
        "catalog_sha256": CAT,
        "study_sha256": STUDY,
        "method_sha256": mk["method_sha256"],
        "opening_sha256": OPENING,
        "sources": req["sources"],
        "files": [rows[n] for n in sorted(rows)],
    }
    if req["purpose"] == "example":
        result["family"] = FAMILY
    with new_output(args.output) as dest:
        write_json(dest / "public-members.json", result)
    return {"files": len(rows), "bytes": sum(x["bytes"] for x in rows.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "shelf", "request", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--authored-grant", type=Path)
    opts = parser.parse_args()
    imports(opts.project.resolve())
    print(json.dumps(work(opts), sort_keys=True))
