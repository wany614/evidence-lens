"""Bind a prerelease to its distribution scope, selected reports, model and recorded diagnostics."""

import csv
import re

from evidence_lens._bundle import HASH, RUN_ID, local_path, sha
from evidence_lens.formal_inputs import PinnedRun
from evidence_lens.scoring_io import TARGETS

NUMBER = r"(?:0|[1-9][0-9]*)"
VERSION = re.compile(rf"v({NUMBER}\.{NUMBER}\.{NUMBER})(?:-rc\.([1-9][0-9]*))?")
SCOPE = "RELEASE_SCOPE.json"
SUMMARY = "SCIENTIFIC_SCOPE.md"
DOCS = ("KNOWN_ISSUES.md", "docs/licenses.md")
CPU_SOURCE = tuple("source/evidence_lens/" + n + ".py" for n in
                   ("__init__", "features", "scoring_io", "risk_model", "l1", "l1_demo", "score_maps"))
PREVIEWS = {"technical_preview", "experimental_research_preview"}
EVIDENCE = {"original_frozen_study_rechecked", "historical_benchmark_regression", "original_study_under_review",
            "historical_results_withdrawn"}
REPOSITORY = "wany614/evidence-lens"
ATTRIBUTION = "wany614"
REPOSITORY_URL = "https://github.com/" + REPOSITORY


def publication(version):
    return {"publication_status":"public_prerelease", "repository":REPOSITORY, "attribution":ATTRIBUTION,
            "tag":version, "release_url":f"{REPOSITORY_URL}/releases/tag/{version}"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def python_version(tag):
    match = VERSION.fullmatch(tag)
    require(match, "use vX.Y.Z or vX.Y.Z-rc.N with a positive candidate number")
    return match[1] + ("rc" + match[2] if match[2] else "")


def graded_views(tag):
    return TARGETS if "-rc." in tag else ("v01",)

def archive_name(version, kind):
    require(kind in {"heldout_study", "synthetic_fixture"}, "unknown scientific package kind")
    prefix = "evidence-lens-synthetic-fixture" if kind == "synthetic_fixture" else "evidence-lens"
    return f"{prefix}-{version}-cpu.zip"


def scope_origin(digest):
    return {"kind": "selected_prerelease_scope", "sha256": digest}


def summary_origin(digest, table):
    return {"kind": "scope_and_recorded_counts", "scope_sha256": digest, "family_results_sha256": table}


def summary_bytes(scope, table, kind="heldout_study"):
    evidence = {"historical_benchmark_regression": "Historical benchmark regression following changes informed by this benchmark.",
                "original_frozen_study_rechecked": "Original frozen study rechecked against the selected diagnostic evidence.",
                "original_study_under_review": "The original formal measurements remain under review.",
                "historical_results_withdrawn": "The historical formal results are withdrawn as evidence for current performance."}
    preview = {"technical_preview": "Technical Preview", "experimental_research_preview": "Experimental Research Preview"}
    audience = (f"Public prerelease by {ATTRIBUTION}. [Release and assets]({publication(scope['version'])['release_url']})"
                if scope.get("publication_status")=="public_prerelease" and kind!="synthetic_fixture" else
                "**Synthetic software test.**" if kind=="synthetic_fixture" else "**Private candidate for owner evaluation.**")
    lines = [f"# {preview[scope['preview_class']]} · {scope['version']}", "", audience, "",
             evidence[scope["formal_evidence_status"]], "", scope["finding"], "", "## Remaining limits", "",
             *("- " + limit for limit in scope["limitations"]), "", "## Selected records", "",
             f"- Frozen model `{scope['model_id']}`, index SHA-256 `{scope['model_index_sha256']}`.",
             f"- Formal numerical report `{scope['report']['run_id']}`.",
             f"- Authored plane deployment score `{scope['example_scores']['run_id']}`.", ""]
    if kind == "synthetic_fixture":
        lines[0:0] = ["# Synthetic software fixture", "", "These fabricated records exercise the packaging tests.", ""]
    for row in scope["diagnostics"]:
        lines.append(f"- Diagnostic `{row['run_id']}`, `{row['path']}`, SHA-256 `{row['sha256']}`.")
    lines.extend(["", "## RGB error decisions in the selected formal record", "",
                  "Each row pools pixels across the four registered families for L1-full. V counts reference-valid pixels, "
                  "E counts RGB-evaluable pixels, H counts high-risk decisions within E, and Y counts confirmed RGB errors. "
                  "HY, LY and UY partition these errors by decision. LY + UY counts confirmed RGB errors outside high risk. "
                  "Depth-error measurements use their own surface and support denominators.", "",
                  "| Condition | V | E | High risk H | RGB errors Y | High HY | Low LY | Unknown UY | Outside high LY + UY |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    with table.open(encoding="utf-8", newline="") as source:
        rows = [r for r in csv.DictReader(source) if r["method"] == "L1-full" and r["region"] == "all"]
    keys = ("V", "E", "H", "Y", "HY", "LY", "UY")
    rates = []
    for c in ("B0", "R1", "Q1"):
        group = [r for r in rows if r["condition"] == c]
        require(len(group) == 4 and len({r["family"] for r in group}) == 4, "formal decision counts lack a unique held-out roster")
        if any(r[k] == "" for r in group for k in keys):
            cells = ["unavailable"] * 8
            ratio = ["unavailable"] * 4
        else:
            counts = [tuple(int(r[k]) for k in keys) for r in group]
            for v, e, h, y, hy, ly, uy in counts:
                require(0 <= h <= e <= v and 0 <= y <= e and 0 <= hy <= h and min(ly,uy) >= 0
                        and hy+ly+uy == y and ly+uy <= e-h, "formal RGB error decisions do not partition Y or its domains")
            v,e,h,y,hy,ly,uy = map(sum,zip(*counts))
            cells = [f"{n:,}" for n in (v,e,h,y,hy,ly,uy,ly+uy)]
            ratio = [f"{n/d:.2%}" if d else "undefined" for n,d in ((h,v),(h,e),(hy,h),(hy,y))]
        lines.append("| " + " | ".join([c, *cells]) + " |")
        rates.append("| " + " | ".join([c, *ratio]) + " |")
    lines.extend(["", "Rates use the pooled counts above. High-risk coverage can be expressed over V or E. "
                  "Precision uses H, and detection of confirmed RGB errors uses Y.", "",
                  "| Condition | High risk among reference-valid H/V | High risk among RGB-evaluable H/E | RGB-error precision HY/H | Confirmed errors detected HY/Y |",
                  "| --- | ---: | ---: | ---: | ---: |", *rates])
    lines.extend(["", "Source tables and per-family denominators are in [family results](results/family-results.csv).", ""])
    return "\n".join(lines).encode()


def check_scope(scope, version, method_sha, model_id, model_sha, sources, *, store=None):
    fields = {"schema", "version", "publication_status", "preview_class", "formal_evidence_status",
              "model_id", "model_index_sha256", "method_freeze_sha256", "example_scores", "report",
              "diagnostics", "finding", "limitations"}
    require(isinstance(scope,dict),"a prerelease scope object is required")
    public = scope.get("publication_status")=="public_prerelease"
    require(set(scope) == (fields | {"repository","attribution"} if public else fields) and scope["schema"] == "el-prerelease-scope-1"
            and scope["version"] == version and "-rc." in version and (public or scope["publication_status"]=="private_candidate"),
            "invalid prerelease scope or tag")
    if public:
        require(scope["repository"]==REPOSITORY and scope["attribution"]==ATTRIBUTION,
                "public scope must identify the authorized project and attribution")
    require(scope["preview_class"] in PREVIEWS and scope["formal_evidence_status"] in EVIDENCE and
            (scope["preview_class"] != "experimental_research_preview" or scope["formal_evidence_status"] in
             {"original_frozen_study_rechecked", "historical_benchmark_regression"}), "preview exceeds its scientific scope")
    require((scope["model_id"], scope["model_index_sha256"], scope["method_freeze_sha256"]) ==
            (model_id, model_sha, method_sha) and all(scope[k] == sources[k] for k in ("example_scores", "report")),
            "prerelease scope selects another model, method, example score or formal report")
    limitations = scope["limitations"]
    require(isinstance(limitations, list) and limitations and
            all(isinstance(s, str) and s.strip() == s and s for s in [scope["finding"], *limitations]),
            "a diagnostic finding and its limitations are required")
    rows = scope["diagnostics"]
    require(isinstance(rows, list) and rows, "pin at least one recorded diagnostic")
    selected = set()
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"run_id", "manifest_sha256", "path", "sha256"}
                and all(isinstance(v, str) for v in row.values()) and RUN_ID.fullmatch(row["run_id"])
                and HASH.fullmatch(row["manifest_sha256"]) and HASH.fullmatch(row["sha256"]),
                "invalid diagnostic run, manifest or payload digest")
        path = local_path(row["path"])
        key = row["run_id"], path
        require(path.startswith("payload/") and path.endswith(".json") and key not in selected,
                "diagnostics must name distinct recorded JSON results")
        selected.add(key)
        if store is not None:
            run = PinnedRun(store, {k: row[k] for k in ("run_id", "manifest_sha256")})
            require(run.run["execution_status"] == "succeeded" and sha(run.open(path)) == row["sha256"],
                    "diagnostic payload differs from its successful original")
