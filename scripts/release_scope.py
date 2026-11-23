"""Bind a private prerelease to its selected reports, model and recorded diagnostics."""

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


def require(ok, message):
    if not ok:
        raise ValueError(message)


def python_version(tag):
    match = VERSION.fullmatch(tag)
    require(match, "use vX.Y.Z or vX.Y.Z-rc.N with a positive candidate number")
    return match[1] + ("rc" + match[2] if match[2] else "")


def graded_views(tag):
    return TARGETS if "-rc." in tag else ("v01",)


def scope_origin(digest):
    return {"kind": "selected_prerelease_scope", "sha256": digest}


def summary_origin(digest, table):
    return {"kind": "scope_and_recorded_counts", "scope_sha256": digest, "family_results_sha256": table}


def summary_bytes(scope, table):
    evidence = {"historical_benchmark_regression": "Historical benchmark regression following changes informed by this benchmark.",
                "original_frozen_study_rechecked": "Original frozen study rechecked against the selected diagnostic evidence.",
                "original_study_under_review": "The original formal measurements remain under review.",
                "historical_results_withdrawn": "The historical formal results are withdrawn as evidence for current performance."}
    preview = {"technical_preview": "Technical Preview", "experimental_research_preview": "Experimental Research Preview"}
    lines = [f"# {preview[scope['preview_class']]} · {scope['version']}", "", "**Private candidate for owner evaluation.**", "",
             evidence[scope["formal_evidence_status"]], "", scope["finding"], "", "## Remaining limits", "",
             *("- " + limit for limit in scope["limitations"]), "", "## Selected records", "",
             f"- Frozen model `{scope['model_id']}`, index SHA-256 `{scope['model_index_sha256']}`.",
             f"- Formal numerical report `{scope['report']['run_id']}`.",
             f"- Authored plane deployment score `{scope['example_scores']['run_id']}`.", ""]
    for row in scope["diagnostics"]:
        lines.append(f"- Diagnostic `{row['run_id']}`, `{row['path']}`, SHA-256 `{row['sha256']}`.")
    lines.extend(["", "## Error decisions in the selected formal record", "",
                  "Each row sums the four registered families over the whole reference-valid region for L1-full. "
                  "Y counts confirmed RGB errors. HY, LY and UY count confirmed errors assigned high, low and unknown risk respectively. "
                  "LY + UY counts all confirmed errors outside the high-risk decision.", "",
                  "| Condition | Confirmed errors Y | High HY | Low LY | Unknown UY | Outside high LY + UY |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    with table.open(encoding="utf-8", newline="") as source:
        rows = [r for r in csv.DictReader(source) if r["method"] == "L1-full" and r["region"] == "all"]
    keys = ("Y", "HY", "LY", "UY")
    for c in ("B0", "R1", "Q1"):
        group = [r for r in rows if r["condition"] == c]
        require(len(group) == 4 and len({r["family"] for r in group}) == 4, "formal decision counts lack a unique held-out roster")
        if any(r[k] == "" for r in group for k in keys):
            cells = ["unavailable"] * 5
        else:
            counts = [tuple(int(r[k]) for k in keys) for r in group]
            require(all(min(ns) >= 0 and sum(ns[1:]) == ns[0] for ns in counts), "formal RGB error decisions do not partition Y")
            y, high, low, unknown = map(sum, zip(*counts))
            cells = [f"{n:,}" for n in (y, high, low, unknown, low + unknown)]
        lines.append("| " + " | ".join([c, *cells]) + " |")
    lines.extend(["", "Source tables and per-family denominators are in [family results](results/family-results.csv).", ""])
    return "\n".join(lines).encode()


def check_scope(scope, version, method_sha, model_id, model_sha, sources, *, store=None):
    fields = {"schema", "version", "publication_status", "preview_class", "formal_evidence_status",
              "model_id", "model_index_sha256", "method_freeze_sha256", "example_scores", "report",
              "diagnostics", "finding", "limitations"}
    require(isinstance(scope, dict) and set(scope) == fields and scope["schema"] == "el-prerelease-scope-1"
            and scope["version"] == version and "-rc." in version and scope["publication_status"] == "private_candidate",
            "invalid private prerelease scope or tag")
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
