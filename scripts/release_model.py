"""Assemble a versioned public CPU bundle from explicit successful run pins and committed original text."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from release_identity import (
    PAIR_ROLES,
    check_identity,
    check_q1_pairings,
    comparison_files_at_freeze,
    link_report,
    signature,
)
from release_media import check_example_media
from release_wheel import aggregate, read_wheel

from evidence_lens._bundle import HASH, RUN_ID, encoded, load, local_path, sha
from evidence_lens._bundle import files as regular_files
from evidence_lens.formal_inputs import PinnedRun
from evidence_lens.l1 import load_package, run as score_model
from evidence_lens.l1_demo import seal
from evidence_lens.scoring_io import TARGETS, VIEWS, new_output

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+")
PRIVATE = re.compile(
    r"(?i)(?:/(?:Users|home|Volumes|private/var|var/folders|mnt)/[^/\s]+|file://|\\\\[\w.-]+\\[^\s\\]+|(?<![A-Za-z0-9])[A-Z]:[\\/][^\s]+|(?:ghp_|github_pat_|sk-ant-|sk-proj-)[A-Za-z0-9_-]{8,}|-----BEGIN [^\n]*PRIVATE KEY|X-(?:Goog|Amz)-Signature=|access_token=)"
)
EMAIL = re.compile(r"(?i)(?<![\w.-])[\w.+%-]+@[\w.-]+\.[a-z]{2,}")
MODEL = (
    "model.json",
    "feature_schema.json",
    "L1-full.json",
    "L1-geometry.json",
    "thresholds.json",
)
TABLES = (
    "data-completion.csv",
    "family-results.csv",
    "family-equal-summary.csv",
    "paired-family-equal.csv",
    "Q1-numerical-response.csv",
    "paired-stage-times.csv",
    "stage-times.csv",
    "worker-phase-times.csv",
    "timing-repeats.csv",
    "measured-receipts.csv",
    "selection.json",
)
LICENSES = {
    "source": "LICENSE",
    "parameters": "licenses/MODEL-LICENSE.txt",
    "example": "licenses/EXAMPLE-LICENSE.txt",
}
LICENSE_SCOPE = "licenses/ASSET-SCOPE.json"


def license_files(scope):
    roles = scope["license_files"]
    require(
        scope["schema"] == "el-public-license-scope-1"
        and set(roles) == {*LICENSES, "research_records"}
        and all(roles[k] == v for k, v in LICENSES.items()),
        "missing owner license scope or existing material roles",
    )
    research = local_path(roles["research_records"])
    require(
        research in LICENSES.values()
        or research.startswith("licenses/")
        and research.count("/") == 1
        and research.endswith(".txt"),
        "choose a committed project license or a separate research notice",
    )
    return roles


def scope_members(names, terms, wheel):
    groups = {k: {"license_file": v, "members": []} for k, v in terms.items()}
    groups["notices"] = {"members": []}
    for name in sorted(names):
        if name == LICENSE_SCOPE or name in terms.values():
            role = "notices"
        elif name.startswith(("example/", "expected/")) or name == "demo.json":
            role = "example"
        elif name.startswith("model/") and name != "model/feature_schema.json":
            role = "parameters"
        elif name.startswith(("training/", "proofs/", "results/")) or name in (
            "protocol/method-freeze.json",
            "protocol/formal-freeze-v1.json",
            "release.json",
            "SHA256SUMS",
        ):
            role = "research_records"
        else:
            require(
                name == wheel
                or name.startswith("protocol/")
                or name
                in (
                    "README.md",
                    "MODEL_CARD.md",
                    "RELEASE_NOTES.md",
                    "distribution/pyproject.toml",
                    "model/feature_schema.json",
                ),
                "public member has no material class: " + name,
            )
            role = "source"
        groups[role]["members"].append(name)
    return groups


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ref(value):
    rid, sep, digest = value.partition("=")
    require(
        sep and RUN_ID.fullmatch(rid) and HASH.fullmatch(digest),
        "supply run_id=manifest_sha256",
    )
    return {"run_id": rid, "manifest_sha256": digest}


def blob(commit, relative):
    name = local_path(relative)
    git = ["git", "-C", str(ROOT)]
    entry = subprocess.check_output([*git, "ls-tree", "-z", commit, "--", name])
    require(entry.endswith(b"\0"), f"missing committed release file {name}")
    head, listed = entry[:-1].split(b"\t", 1)
    mode, kind, oid = head.split()
    require(
        listed == name.encode() and mode in (b"100644", b"100755") and kind == b"blob",
        f"release source must be a regular committed file: {name}",
    )
    return subprocess.check_output([*git, "cat-file", "blob", oid.decode()])


def safe_text(name, raw, notice_files=()):
    def screen(value):
        require(
            not PRIVATE.search(value), f"possible private machine path, credential or signed URL in {name}"
        )
        require(
            name in LICENSES.values() or name in notice_files or not EMAIL.search(value),
            f"possible personal email address in {name}",
        )

    string = raw.decode("utf-8")
    screen(string)
    if name.endswith(".json"):
        # Escaped slashes, backslashes and Unicode escapes must be checked as the recipient reads them.
        stack = [json.loads(string)]
        while stack:
            value = stack.pop()
            if isinstance(value, str):
                screen(value)
            elif isinstance(value, dict):
                stack.extend(value.keys())
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)


def pinned_inputs(run):
    data = load(run.open("inputs.json"))
    parents = data["parents"]
    ids = [p["run_id"] for p in parents]
    roles = {row["role"]: row for row in data["inputs"]}
    require(
        len(roles) == len(data["inputs"])
        and len(set(ids)) == len(ids)
        and sorted(ids) == sorted(run.run["parent_run_ids"])
        and all(RUN_ID.fullmatch(p["run_id"]) and HASH.fullmatch(p["manifest_sha256"]) for p in parents),
        "release source input inventory contains duplicate roles or inconsistent direct parent pins",
    )
    return parents, roles


def check_bound(row, parent, names):
    require(
        row["parent_run_id"] == parent["run_id"] and row["storage"] == "embedded",
        "release source has the wrong parent or input storage",
    )
    given = {Path(p["path"]).name: {k: p[k] for k in ("bytes", "sha256")} for p in row["files"]}
    require(
        len(given) == len(row["files"]) and given == names,
        "release source differs from its recorded input inventory",
    )


def original_reference_proof(run, sources, kind):
    path = run.open("payload/reference.json")
    meta = load(path)
    require(
        meta["input_sha256"] == sources["rgb"]["sha256"]
        and meta["camera_sha256"] == sources["cameras"]["sha256"],
        "authored source inputs differ from the original reference producer",
    )
    proof = {"schema": meta["schema"], "metadata_sha256": sha(path)}
    if kind == "synthetic_fixture" and meta["schema"] == "el-synthetic-plane-source-1":
        return proof
    require(
        meta["schema"] == "el-reference-1"
        and meta["validation_passed"] is True
        and set(meta["views"]) == set(VIEWS)
        and meta["views"]["source"]["png_sha256"] == meta["input_sha256"],
        "public example requires the accepted original Blender scene render",
    )
    validation_file = run.open("payload/validation/results.json")
    validation = load(validation_file)
    scene_path = local_path(meta["scene_path"])
    require(
        validation["schema"] == "el-reference-validation-1"
        and validation["passed"] is True
        and sha(validation_file) == meta["validation_sha256"]
        and scene_path.startswith("scene/")
        and scene_path.endswith(".blend"),
        "authored Blender source lacks its original validation or native scene",
    )
    native = run.open("payload/" + scene_path)
    require(
        sha(native) == meta["blend_sha256"], "authored source native scene differs from its recorded digest"
    )
    return dict(proof, native_blend_sha256=meta["blend_sha256"], validation_sha256=meta["validation_sha256"])


def pairing_proof(store, report, inputs, science, selection):
    manifests = [
        row
        for role in inputs.values()
        for row in role["files"]
        if Path(row["path"]).name == "report_inputs.json" and row["sha256"] == science["report_input_sha256"]
    ]
    require(len(manifests) == 1, "report did not bind a unique staged input manifest")
    staged = load(report.open(manifests[0]["path"]))
    rows = {}
    for trial in selection["comparison_trials"]:
        family, condition = trial["scene_family_id"], trial["condition_id"]
        if condition not in ("B0", "Q1"):
            continue
        pin = {k: trial[k] for k in ("run_id", "manifest_sha256")}
        parent = PinnedRun(store, pin)
        original = parent.open("payload/formal_evaluation.json")
        rec = load(original)
        rel = f"trials/{family}/{condition}/formal_evaluation.json"
        require(
            parent.run["run_kind"] == "evaluation"
            and parent.run["execution_status"] == "succeeded"
            and staged["files"][rel] == signature(original)
            and rec["schema"] == "el-formal-comparison-1"
            and (rec["run_id"], rec["scene_family_id"], rec["condition_id"])
            == (pin["run_id"], family, condition)
            and rec["model_id"] == selection["model_id"]
            and rec["method_freeze_sha256"] == selection["method_freeze_sha256"],
            "paired comparison differs from the selected sealed report input",
        )
        prefix = "paired_" if condition == "Q1" else ""
        rows.setdefault(family, {})[condition] = {
            "comparison_run": pin,
            "receipt_sha256": sha(original),
            "whole_render_failure": bool(rec["render_failure_reason"]),
            "score_parents": {
                k: rec["parents"][prefix + k] for k in PAIR_ROLES if prefix + k in rec["parents"]
            },
        }
    check_q1_pairings(rows, selection["comparison_trials"])
    return rows


def projection(parent, path, digest):
    return {
        "kind": "projection",
        "run_id": parent["run_id"],
        "manifest_sha256": parent["manifest_sha256"],
        "path": path,
        "source_sha256": digest,
    }


def build(a):
    method = load(a.method)
    method_sha = sha(a.method)
    require(not a.method_sha256 or a.method_sha256 == method_sha, "method differs from its supplied pin")
    require(
        method["schema"] == "el-method-freeze-1"
        and re.fullmatch(r"[a-f0-9]{40}", a.commit)
        and re.fullmatch(r"[a-f0-9]{40}", method["method_code_commit"])
        and VERSION.fullmatch(a.version),
        "invalid method or release identity",
    )
    scope_doc = blob(a.commit, LICENSE_SCOPE)
    terms = license_files(json.loads(scope_doc))
    subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            method["method_code_commit"],
            a.commit,
        ],
        check=True,
    )
    threshold = ref(method["threshold_select_run_id"] + "=" + method["threshold_select_manifest_sha256"])
    fit = ref(method["model_fit_run_id"] + "=" + method["model_fit_manifest_sha256"])
    pins = {
        "model": threshold,
        "fit": fit,
        "example": ref(a.example),
        "example_scores": ref(a.scores),
        "report": ref(a.report),
    }
    runs = {role: PinnedRun(a.store.resolve(), pin) for role, pin in pins.items()}
    require(
        all(r.run["execution_status"] == "succeeded" for r in runs.values()),
        "a release source run did not succeed",
    )
    mp = runs["model"].open("payload/model/model.json")
    pack = load_package(mp.parent, expected_index_sha256=method["model_index_sha256"])
    threshold_parents, _ = pinned_inputs(runs["model"])
    require(
        runs["fit"].run["run_kind"] == "model_fit"
        and runs["model"].run["run_kind"] == "threshold_select"
        and pack["index"]["model_id"] == method["model_id"]
        and fit in threshold_parents
        and fit["run_id"] == pack["index"]["fit_run_id"],
        "release fit/model identity differs from the method",
    )
    require(
        runs["report"].run["run_kind"] == "report"
        and runs["example_scores"].run["run_kind"] == "model_score"
        and {pins["example"]["run_id"], threshold["run_id"]}
        == set(runs["example_scores"].run["parent_run_ids"]),
        "example scores or report belong to unexpected sources",
    )
    science = load(runs["report"].open("payload/report.json"))
    demo = load(runs["example"].open("payload/example/provenance.json"))
    kind = "synthetic_fixture" if a.fixture else "heldout_study"
    require(
        runs["example"].run["run_kind"] == "derivation"
        and demo["adapter"] == "evidence-lens-textured-plane-1"
        and demo["run_id"] == pins["example"]["run_id"]
        and science["run_id"] == pins["report"]["run_id"]
        and science["kind"] == kind,
        "release needs the authored example and the completed held-out study",
    )
    train = runs["fit"].open("payload/training_dataset_manifest.json")
    manifest = load(train)
    require(
        sha(train) == pack["index"]["training_manifest_sha256"]
        and len(manifest["family_sources"]) == 16
        and manifest["positive_count"] > 0
        and manifest["row_count"] > manifest["positive_count"],
        "training population/manifest mismatch",
    )
    orig_parents, orig_inputs = pinned_inputs(runs["example"])
    require(
        len(orig_parents) == 1 and set(orig_inputs) == {"rgb", "cameras"},
        "authored plane must bind exactly the original PNG and camera roles",
    )
    original = PinnedRun(a.store.resolve(), orig_parents[0])
    if a.example_source:
        require(
            ref(a.example_source) == orig_parents[0], "authored original differs from its supplied run pin"
        )
    require(
        original.run["run_kind"] == "scene_render" and original.run["execution_status"] == "succeeded",
        "author source is not the original scene render",
    )
    source_signatures = {}
    for role, file, key in (
        ("rgb", "input.png", "source_png_sha256"),
        ("cameras", "cameras.json", "source_cameras_sha256"),
    ):
        sig = signature(original.open("payload/" + file))
        require(sig["sha256"] == demo[key], "example provenance differs from its original source")
        check_bound(orig_inputs[role], orig_parents[0], {file: sig})
        source_signatures[role] = sig
    pins["example_source"] = orig_parents[0]
    reference_proof = original_reference_proof(original, source_signatures, kind)
    score_parents, score_inputs = pinned_inputs(runs["example_scores"])
    require(
        sorted(score_parents, key=lambda r: r["run_id"])
        == sorted((threshold, pins["example"]), key=lambda r: r["run_id"])
        and set(score_inputs) == {"source", "cameras", "predictions", "model"},
        "example scoring input roles or parent manifest pins disagree",
    )
    for role, name in (("source", "input.png"), ("cameras", "cameras.json")):
        check_bound(
            score_inputs[role],
            pins["example"],
            {name: signature(runs["example"].open("payload/example/" + name))},
        )
    required_pred = ["prediction.json", *(v + ".npz" for v in VIEWS)]
    if any(Path(r["path"]).name == "depth-probe.json" for r in score_inputs["predictions"]["files"]):
        required_pred.append("depth-probe.json")
    check_bound(
        score_inputs["predictions"],
        pins["example"],
        {n: signature(runs["example"].open("payload/example/predictions/" + n)) for n in required_pred},
    )
    check_bound(
        score_inputs["model"],
        threshold,
        {n: signature(runs["model"].open("payload/model/" + n)) for n in MODEL},
    )
    for name in ("features", "scoring_io", "risk_model", "l1"):
        rel = "src/evidence_lens/" + name + ".py"
        require(
            hashlib.sha256(blob(a.commit, rel)).hexdigest() == method["method_source_sha256"][rel],
            "released inference code differs from the studied method: " + rel,
        )
    project_toml = blob(a.commit, "pyproject.toml")
    wheel_meta = read_wheel(a.wheel, project_toml, checkout=ROOT, commit=a.commit, text_check=safe_text)
    method_commit = method["method_code_commit"]
    historical_files = comparison_files_at_freeze(blob(method_commit, "src/evidence_lens/formal_inputs.py"))
    commit_protocol = {
        name: hashlib.sha256(blob(method_commit, name)).hexdigest() for name in historical_files
    }
    require(
        commit_protocol == method["comparison_files"]
        and aggregate(commit_protocol) == method["comparison_protocol_sha256"],
        "comparison protocol differs from original method commit",
    )
    report_selection = load(runs["report"].open("payload/public-numerical/selection.json"))
    signatures = {n: signature(runs["report"].open("payload/public-numerical/" + n)) for n in TABLES}
    families_bytes = blob(a.commit, "protocol/families-v1.json")
    link_report(science, report_selection, method, method_sha, json.loads(families_bytes), signatures, kind)
    report_parents, report_inputs = pinned_inputs(runs["report"])
    registered_parents = {(p["run_id"], p["manifest_sha256"]) for p in report_parents}
    require(
        all(
            (p["run_id"], p["manifest_sha256"]) in registered_parents
            for p in report_selection["comparison_trials"]
        ),
        "report did not bind all the selected comparison parents",
    )
    pairings = pairing_proof(a.store, runs["report"], report_inputs, science, report_selection)
    with new_output(a.output) as out:
        entries = {}

        def include(name, raw, origin):
            name = local_path(name)
            require(name not in entries, "duplicate release member")
            if Path(name).suffix in {".md", ".txt", ".csv", ".json"} or name == "LICENSE":
                safe_text(name, raw, terms.values())
            target = out / name
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_bytes(raw)
            entries[name] = {
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "origin": origin,
            }

        def recorded(role, name, source):
            include(
                name,
                runs[role].open(source).read_bytes(),
                {"run_id": pins[role]["run_id"], "path": source},
            )

        def committed(name, source):
            include(name, blob(a.commit, source), {"commit": a.commit, "path": source})

        for name in MODEL:
            recorded("model", "model/" + name, "payload/model/" + name)
        for name in (
            "input.png",
            "cameras.json",
            "provenance.json",
            "v01-prediction.png",
            "predictions/prediction.json",
            "predictions/depth-probe.json",
            *("predictions/" + v + ".npz" for v in VIEWS),
        ):
            recorded("example", "example/" + name, "payload/example/" + name)
        original_scores = runs["example_scores"].open("payload/l1_score.json")
        for view in TARGETS:
            runs["example_scores"].open("payload/" + view + ".npz")
        seal(out, original_scores.parent, sha(original_scores), ["v01"])
        demo_path = out / "demo.json"
        demo_path.write_bytes(demo_path.read_bytes().replace(b"\r\n", b"\n"))
        for name in ("demo.json", "expected/v01.npz"):
            raw = (out / name).read_bytes()
            entries[name] = {
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "origin": {
                    "score_run_id": pins["example_scores"]["run_id"],
                    "score_receipt_sha256": sha(original_scores),
                },
            }
        for name in ("training_dataset_manifest.json", "cv_results.json"):
            recorded("fit", "training/" + name, "payload/" + name)
        recorded("fit", "proofs/model-fit.json", "payload/fit.json")
        recorded("fit", "proofs/fitted-model-index.json", "payload/model/model.json")
        recorded("model", "proofs/development.json", "payload/development.json")
        recorded("example_scores", "proofs/example-score.json", "payload/l1_score.json")
        original_inputs_sha = sha(runs["example"].open("inputs.json"))
        include(
            "proofs/example-source.json",
            encoded(
                {
                    "schema": "el-example-source-projection-1",
                    "original_inputs_sha256": original_inputs_sha,
                    "parent": orig_parents[0],
                    "input_sha256": source_signatures["rgb"]["sha256"],
                    "camera_sha256": source_signatures["cameras"]["sha256"],
                    "reference": reference_proof,
                }
            ),
            projection(pins["example"], "inputs.json", original_inputs_sha),
        )
        report_public = {
            k: science[k]
            for k in (
                "run_id",
                "kind",
                "quality_trials",
                "planned_targets",
                "completed_targets",
                "model_id",
                "comparison_protocol_sha256",
            )
        }
        report_public.update(
            schema="el-public-report-projection-1",
            source_schema=science["schema"],
            q1_baseline_bindings=pairings,
            selected_trials=report_selection["comparison_trials"],
            files={"public-numerical/" + n: v for n, v in signatures.items()},
            original_report_sha256=sha(runs["report"].open("payload/report.json")),
        )
        include(
            "proofs/report-public.json",
            encoded(report_public),
            projection(pins["report"], "payload/report.json", report_public["original_report_sha256"]),
        )
        for name in TABLES:
            recorded("report", "results/" + name, "payload/public-numerical/" + name)
        for name in (
            "families-v1.json",
            "study-v1.json",
            "analysis-addendum-v1.json",
            "operating-points-v1.json",
        ):
            committed("protocol/" + name, "protocol/" + name)
        for name in ("MODEL_CARD.md", "RELEASE_NOTES.md", "README.md"):
            committed(name, "docs/releases/" + a.version + "/" + name)
        for name in sorted(set(terms.values())):
            committed(name, name)
        include(LICENSE_SCOPE, scope_doc, {"commit": a.commit, "path": LICENSE_SCOPE})
        committed("protocol/formal-freeze-v1.json", "configs/formal-freeze-v1.json")
        include("protocol/method-freeze.json", a.method.read_bytes(), {"method_freeze_sha256": method_sha})
        committed("distribution/pyproject.toml", "pyproject.toml")
        wheel_name = "distribution/" + a.wheel.name
        include(
            wheel_name,
            a.wheel.read_bytes(),
            {"commit": a.commit, "source_modules_sha256": wheel_meta["modules_sha256"]},
        )
        wheel_meta.update(path=wheel_name, **{k: entries[wheel_name][k] for k in ("bytes", "sha256")})
        study_protocol = load(out / "protocol/study-v1.json")
        index = {
            "schema": "el-public-model-release-1",
            "kind": kind,
            "version": a.version,
            "code_commit": a.commit,
            "method_code_commit": method["method_code_commit"],
            "model_id": method["model_id"],
            "model_index_sha256": method["model_index_sha256"],
            "thresholds_sha256": method["thresholds_sha256"],
            "fit_run_id": fit["run_id"],
            "training_manifest_sha256": sha(train),
            "feature_schema_sha256": sha(out / "model/feature_schema.json"),
            "original_report_receipt_sha256": sha(runs["report"].open("payload/report.json")),
            "original_example_inputs_sha256": original_inputs_sha,
            "fit_receipt_sha256": sha(out / "proofs/model-fit.json"),
            "example_score_receipt_sha256": sha(original_scores),
            "catalog_sha256": method["catalog_sha256"],
            "study_sha256": method["study_sha256"],
            "method_freeze_sha256": sha(a.method),
            "comparison_protocol_sha256": method["comparison_protocol_sha256"],
            "protocol_version": study_protocol["numerical_protocol_version"],
            "label_version": study_protocol["labels"]["version"],
            "feature_version": load(out / "model/feature_schema.json")["schema"],
            "license_files": terms,
            "license_scope": scope_members(set(entries) | {"release.json", "SHA256SUMS"}, terms, wheel_name),
            "wheel": wheel_meta,
            "sources": pins,
            "files": entries,
            "payload_members_sha256": aggregate(
                {n: {k: v[k] for k in ("bytes", "sha256")} for n, v in entries.items()}
            ),
        }
        (out / "release.json").write_bytes(encoded(index))
        members = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
        checks = [f"{sha(out / name)}  {name}" for name in members]
        (out / "SHA256SUMS").write_bytes(("\n".join(checks) + "\n").encode("utf-8"))
        check(out)
    return {
        "model_id": method["model_id"],
        "version": a.version,
        "commit": a.commit,
        "release_sha256": sha(a.output / "release.json"),
        "demo_sha256": sha(a.output / "demo.json"),
        "checksums_sha256": sha(a.output / "SHA256SUMS"),
        "wheel_sha256": wheel_meta["sha256"],
        "kind": kind,
    }


def asset_names(wheel, terms):
    return {
        "MODEL_CARD.md",
        "README.md",
        "RELEASE_NOTES.md",
        LICENSE_SCOPE,
        *terms.values(),
        "demo.json",
        "expected/v01.npz",
        wheel,
        "distribution/pyproject.toml",
        *("model/" + n for n in MODEL),
        *("results/" + n for n in TABLES),
        *(
            "proofs/" + n
            for n in (
                "model-fit.json",
                "fitted-model-index.json",
                "development.json",
                "report-public.json",
                "example-source.json",
                "example-score.json",
            )
        ),
        *("training/" + n for n in ("training_dataset_manifest.json", "cv_results.json")),
        *(
            "protocol/" + n
            for n in (
                "families-v1.json",
                "study-v1.json",
                "analysis-addendum-v1.json",
                "operating-points-v1.json",
                "formal-freeze-v1.json",
                "method-freeze.json",
            )
        ),
        *(
            "example/" + n
            for n in (
                "input.png",
                "cameras.json",
                "provenance.json",
                "v01-prediction.png",
                "predictions/prediction.json",
                "predictions/depth-probe.json",
            )
        ),
        *("example/predictions/" + v + ".npz" for v in VIEWS),
    }


def asset_origin(index, name):
    ids = {role: pin["run_id"] for role, pin in index["sources"].items()}
    if name in {"demo.json", "expected/v01.npz"}:
        return {
            "score_run_id": ids["example_scores"],
            "score_receipt_sha256": index["example_score_receipt_sha256"],
        }
    if name == "protocol/method-freeze.json":
        return {"method_freeze_sha256": index["method_freeze_sha256"]}
    if name == index["wheel"]["path"]:
        return {"commit": index["code_commit"], "source_modules_sha256": index["wheel"]["modules_sha256"]}
    if name == "proofs/report-public.json":
        return projection(
            index["sources"]["report"], "payload/report.json", index["original_report_receipt_sha256"]
        )
    if name == "proofs/example-source.json":
        return projection(index["sources"]["example"], "inputs.json", index["original_example_inputs_sha256"])
    roots = {
        "model": ("model", "payload/model/"),
        "training": ("fit", "payload/"),
        "results": ("report", "payload/public-numerical/"),
        "example": ("example", "payload/example/"),
    }
    folder, _, tail = name.partition("/")
    if folder in roots:
        role, parent_folder = roots[folder]
        return {"run_id": ids[role], "path": parent_folder + tail}
    proofs = {
        "model-fit.json": ("fit", "payload/fit.json"),
        "fitted-model-index.json": ("fit", "payload/model/model.json"),
        "development.json": ("model", "payload/development.json"),
        "example-score.json": ("example_scores", "payload/l1_score.json"),
    }
    if folder == "proofs":
        role, original = proofs[tail]
        return {"run_id": ids[role], "path": original}
    source = name
    if name in ("MODEL_CARD.md", "README.md", "RELEASE_NOTES.md"):
        source = "docs/releases/" + index["version"] + "/" + name
    elif name == "protocol/formal-freeze-v1.json":
        source = "configs/formal-freeze-v1.json"
    elif name == "distribution/pyproject.toml":
        source = "pyproject.toml"
    return {"commit": index["code_commit"], "path": source}


def check_guide_values(guide, demo_sha, model_sha, wheel_name, state_counts):
    """Check the commands and six-view outcome printed in the shipped guide."""
    for option, expected in (("--manifest-sha256", demo_sha), ("--model-sha256", model_sha)):
        values = re.findall(re.escape(option) + r"\s+(\S+)", guide)
        require(values and set(values) == {expected}, "README has a missing or stale " + option)
    wheels = re.findall(r"distribution[/\\]([\w.-]+\.whl)", guide)
    require(wheels and set(wheels) == {wheel_name}, "README wheel differs from the release")
    low, high, unknown = state_counts
    require(
        f"{low:,} low, {high:,} high and {unknown:,} unknown" in guide,
        "README six-view decision counts differ from CPU outputs",
    )


def check_real_example_guide(folder, release):
    model_sha = sha(folder / "model/model.json")
    counts = np.zeros(256, dtype=np.int64)
    with tempfile.TemporaryDirectory() as temp:
        scores = Path(temp) / "scores"
        example = folder / "example"
        score_model(example / "input.png", example / "cameras.json", example / "predictions",
                    folder / "model", scores, model_sha256=model_sha)
        for view in TARGETS:
            with np.load(scores / (view + ".npz"), allow_pickle=False) as data:
                counts += np.bincount(data["l1_full_state"].ravel(), minlength=256)
    check_guide_values((folder / "README.md").read_text(encoding="utf-8"), sha(folder / "demo.json"),
                       model_sha, release["wheel"]["name"], counts[[0, 1, 255]])


def check(folder, release_sha=None, sums_sha=None):
    actual = set(regular_files(folder))
    require("release.json" in actual and "SHA256SUMS" in actual, "missing release manifest or checksums")
    require(
        not release_sha or sha(folder / "release.json") == release_sha,
        "release manifest differs from the external pin",
    )
    require(
        not sums_sha or sha(folder / "SHA256SUMS") == sums_sha, "SHA256SUMS differs from the external pin"
    )
    release = load(folder / "release.json")
    require(LICENSE_SCOPE in actual, "missing committed asset class and license scope")
    terms = license_files(load(folder / LICENSE_SCOPE))
    require(
        release.get("schema") == "el-public-model-release-1"
        and release["kind"] in ("heldout_study", "synthetic_fixture")
        and VERSION.fullmatch(release["version"])
        and re.fullmatch(r"[a-f0-9]{40}", release["code_commit"])
        and release["license_files"] == terms,
        "invalid release identity or license map",
    )
    wheel = release["wheel"]
    require(
        wheel["path"] == "distribution/" + wheel["name"]
        and set(release["files"]) == asset_names(wheel["path"], terms)
        and actual == set(release["files"]) | {"release.json", "SHA256SUMS"},
        "unexpected or missing public asset",
    )
    require(
        release["license_scope"] == scope_members(actual, terms, wheel["path"]),
        "license asset coverage differs from the owner scope",
    )
    for name, row in release["files"].items():
        path = folder / local_path(name)
        require(
            not path.is_symlink()
            and path.stat().st_size == row["bytes"]
            and sha(path) == row["sha256"]
            and row["origin"] == asset_origin(release, name),
            "changed public member " + name,
        )
        if path.suffix in (".md", ".txt", ".csv", ".json", ".toml") or name == "LICENSE":
            safe_text(name, path.read_bytes(), terms.values())
    safe_text("release.json", (folder / "release.json").read_bytes())
    expected = [f"{sha(folder / n)}  {n}" for n in sorted(actual - {"SHA256SUMS"})]
    expected_bytes = ("\n".join(expected) + "\n").encode("utf-8")
    require((folder / "SHA256SUMS").read_bytes() == expected_bytes, "checksum list differs from exact release contents")
    require(
        release["payload_members_sha256"] == aggregate({n: signature(folder / n) for n in release["files"]}),
        "release payload aggregate differs",
    )
    measured_wheel = read_wheel(
        folder / wheel["path"], (folder / "distribution/pyproject.toml").read_bytes(), text_check=safe_text
    )
    require(
        wheel == dict(measured_wheel, path=wheel["path"], **signature(folder / wheel["path"]))
        and measured_wheel["source_license_sha256"] == sha(folder / LICENSES["source"]),
        "wheel or module aggregate differs",
    )
    check_example_media(folder)
    check_identity(folder, release, TABLES)
    if release["kind"] == "heldout_study":
        check_real_example_guide(folder, release)
    return sorted(actual)


def check_zip(folder, archive, names, expected=None):
    require(archive.is_file() and not archive.is_symlink(), "missing regular release ZIP")
    actual_sha = sha(archive)
    require(not expected or expected == actual_sha, "ZIP differs from its external pin")
    with zipfile.ZipFile(archive) as z:
        require(z.namelist() == names, "ZIP member inventory differs")
        for i in z.infolist():
            require(
                i.external_attr >> 16 == 0o100644
                and i.file_size == (folder / i.filename).stat().st_size
                and hashlib.sha256(z.read(i)).hexdigest() == sha(folder / i.filename),
                "ZIP member differs: " + i.filename,
            )
    return {
        "name": archive.name,
        "sha256": actual_sha,
        "bytes": archive.stat().st_size,
        "members": len(names),
    }


def package(folder, output, release_sha=None, sums_sha=None):
    release_sha = release_sha or sha(folder / "release.json")
    sums_sha = sums_sha or sha(folder / "SHA256SUMS")
    names = check(folder, release_sha, sums_sha)
    sidecar = output.with_name(output.name + ".sha256")
    require(
        not (output.parent.resolve() / output.name).is_relative_to(folder.resolve())
        and output.suffix == ".zip"
        and local_path(output.name),
        "choose a ZIP destination outside the release directory",
    )
    require(not output.is_symlink() and not sidecar.is_symlink(), "invalid ZIP or checksum destination")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        archive = Path(temp) / output.name
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for name in names:
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.create_system, info.external_attr = 3, 0o100644 << 16
                z.writestr(
                    info, (folder / name).read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6
                )
        result = check_zip(folder, archive, names)
        require(names == check(folder, release_sha, sums_sha), "release changed while creating the ZIP")
        checksum = Path(temp) / sidecar.name
        checksum.write_bytes((result["sha256"] + "  " + output.name + "\n").encode())
        for dst, src in ((output, archive), (sidecar, checksum)):
            require(
                not dst.exists() or dst.is_file() and sha(dst) == sha(src),
                "immutable ZIP/checksum target conflicts",
            )
        for dst, src in ((output, archive), (sidecar, checksum)):
            try:
                os.link(src, dst)
            except FileExistsError:
                require(not dst.is_symlink() and sha(src) == sha(dst), "conflicting published container")
    return dict(result, checksum_file=sidecar.name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cmds = p.add_subparsers(dest="op", required=True)
    b = cmds.add_parser("build")
    for key in ("store", "method", "output", "wheel"):
        b.add_argument("--" + key, type=Path, required=True)
    for key in ("version", "commit", "example", "scores", "report"):
        b.add_argument("--" + key, required=True)
    b.add_argument("--method-sha256", help="expected immutable method.json SHA-256")
    b.add_argument(
        "--example-source", help="independent run_id=manifest_sha256 pin for the original Blender reference"
    )
    b.add_argument(
        "--fixture",
        action="store_true",
        help="private synthetic integration only; requires a synthetic_fixture report",
    )
    for op in ("verify", "zip"):
        s = cmds.add_parser(op)
        s.add_argument("--folder", type=Path, required=True)
        s.add_argument("--release-sha256")
        s.add_argument("--checksums-sha256")
        if op == "zip":
            s.add_argument("--output", type=Path, required=True)
        else:
            s.add_argument("--archive", type=Path)
            s.add_argument("--archive-sha256")
    a = p.parse_args()
    if a.op == "build":
        result = build(a)
    elif a.op == "zip":
        result = package(a.folder, a.output, a.release_sha256, a.checksums_sha256)
    else:
        require(not a.archive_sha256 or a.archive, "an archive hash needs --archive")
        names = check(a.folder, a.release_sha256, a.checksums_sha256)
        result = {
            "verified_files": len(names),
            "release_sha256": sha(a.folder / "release.json"),
            "checksums_sha256": sha(a.folder / "SHA256SUMS"),
            "wheel_sha256": load(a.folder / "release.json")["wheel"]["sha256"],
            "externally_pinned": bool(a.release_sha256 or a.checksums_sha256),
        }
        if a.archive:
            result["archive"] = check_zip(a.folder, a.archive, names, a.archive_sha256)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
