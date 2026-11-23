"""Check the public provenance documents and numerical artifacts against one model and one study."""

import ast
import csv
import hashlib
import json

from evidence_lens._bundle import HASH, RUN_ID, load, local_path, sha
from evidence_lens.l1 import KEYS, MODELS, load_package
from evidence_lens.l1_demo import ABS_TOL, compare, public_inputs
from evidence_lens.scoring_io import TARGETS, VIEWS


def require(ok, message):
    if not ok:
        raise ValueError(message)


def signature(path):
    return {"bytes": path.stat().st_size, "sha256": sha(path)}


def comparison_files_at_freeze(source):
    """Read the historical protocol's static paths and path template without executing historical code."""
    entries = [
        n.value
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PROTOCOL_FILES" for t in n.targets)
    ]
    require(
        len(entries) == 1 and isinstance(entries[0], (ast.Tuple, ast.List)),
        "missing frozen comparison file list",
    )
    paths = []
    for entry in entries[0].elts:
        if isinstance(entry, ast.Constant):
            paths.append(local_path(entry.value))
            continue
        require(
            isinstance(entry, ast.Starred) and isinstance(entry.value, ast.GeneratorExp),
            "unsupported frozen comparison path declaration",
        )
        gen = entry.value
        require(
            isinstance(gen.elt, ast.JoinedStr) and len(gen.generators) == 1,
            "invalid frozen comparison path template",
        )
        loop = gen.generators[0]
        require(
            isinstance(loop.target, ast.Name)
            and not loop.ifs
            and not loop.is_async
            and isinstance(loop.iter, (ast.Tuple, ast.List)),
            "comparison path iteration must use frozen literals",
        )
        parts = gen.elt.values
        require(
            all(
                isinstance(p, ast.Constant)
                and isinstance(p.value, str)
                or isinstance(p, ast.FormattedValue)
                and isinstance(p.value, ast.Name)
                and p.value.id == loop.target.id
                and p.conversion == -1
                and p.format_spec is None
                for p in parts
            ),
            "invalid comparison template field",
        )
        for value in ast.literal_eval(loop.iter):
            require(isinstance(value, str), "invalid comparison path component")
            paths.append(
                local_path("".join(p.value if isinstance(p, ast.Constant) else value for p in parts))
            )
    require(paths and len(paths) == len(set(paths)), "empty or duplicate frozen comparison paths")
    return sorted(paths)


def link_report(report, selection, method, method_sha, catalog, table_files, kind):
    trials = selection["comparison_trials"]
    expected = {
        (f["scene_family_id"], c)
        for f in catalog["families"]
        if f["split"] == "test"
        for c in ("B0", "R1", "Q1")
    }
    require(
        (
            report["schema"] == "el-study-report-1"
            or report["schema"] == "el-public-report-projection-1"
            and report["source_schema"] == "el-study-report-1"
        )
        and selection["schema"] == "el-numerical-study-release-candidate-1"
        and report["kind"] == selection["kind"] == kind
        and report["model_id"] == selection["model_id"] == method["model_id"]
        and selection["model_index_sha256"] == method["model_index_sha256"]
        and selection["method_code_commit"] == method["method_code_commit"]
        and report["comparison_protocol_sha256"]
        == selection["comparison_protocol_sha256"]
        == method["comparison_protocol_sha256"]
        and selection["method_freeze_sha256"] == method_sha
        and selection["catalog_sha256"] == method["catalog_sha256"]
        and selection["study_sha256"] == method["study_sha256"],
        "study report belongs to a different model, protocol or population",
    )
    require(
        report["quality_trials"] == len(trials) == len(expected) == 12
        and report["planned_targets"] == 72
        and {(r["scene_family_id"], r["condition_id"]) for r in trials} == expected
        and len({r["run_id"] for r in trials}) == 12
        and all(RUN_ID.fullmatch(r["run_id"]) and HASH.fullmatch(r["manifest_sha256"]) for r in trials)
        and 0 <= report["completed_targets"] <= 72,
        "report selection is not the registered paired held-out roster",
    )
    chosen = [
        {k: r[k] for k in ("scene_family_id", "condition_id", "run_id", "manifest_sha256")}
        for r in report["selected_trials"]
    ]
    require(
        chosen == trials and set(selection["files"]) == set(table_files) - {"selection.json"},
        "report/table source inventory differs",
    )
    require(
        all(selection["files"][name] == sig for name, sig in table_files.items() if name != "selection.json"),
        "numerical table differs from its original selected report",
    )
    require(
        all(report["files"]["public-numerical/" + n] == sig for n, sig in table_files.items()),
        "numerical table differs from the original report manifest",
    )


def check_report_thresholds(path, thresholds, trials):
    frozen = dict(thresholds, L0=0.5)
    conditions = {(r["scene_family_id"], r["condition_id"]) for r in trials}
    groups, seen = set(), set()
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        require(
            rows.fieldnames and len(rows.fieldnames) == len(set(rows.fieldnames)),
            "invalid family-results header",
        )
        for row in rows:
            family, condition, region, method = (row[k] for k in ("family", "condition", "region", "method"))
            key = family, condition, region, method
            require(
                (family, condition) in conditions
                and region
                and method in frozen
                and key not in seen
                and float(row["tau"]) == frozen[method],
                "family-results row uses an incorrect frozen threshold or identity",
            )
            groups.add((family, condition, method))
            seen.add(key)
    require(
        groups == {(f, c, m) for f, c in conditions for m in frozen},
        "family-results omits a registered condition or method",
    )


PAIR_ROLES = ("learned", "prediction", "rules")


def check_q1_pairings(rows, trials):
    selected = {
        (r["scene_family_id"], r["condition_id"]): {k: r[k] for k in ("run_id", "manifest_sha256")}
        for r in trials
    }
    require(set(rows) == {f for f, c in selected if c == "Q1"}, "incomplete Q1 pairing projection")
    for family, pair in rows.items():
        require(set(pair) == {"B0", "Q1"}, "each Q1 pairing needs one baseline")
        for condition, row in pair.items():
            require(
                row["comparison_run"] == selected[family, condition]
                and HASH.fullmatch(row["receipt_sha256"])
                and type(row["whole_render_failure"]) is bool
                and set(row["score_parents"]) <= set(PAIR_ROLES),
                "Q1 pairing identifies a different original comparison",
            )
            for pin in row["score_parents"].values():
                require(
                    set(pin) == {"run_id", "manifest_sha256"}
                    and RUN_ID.fullmatch(pin["run_id"])
                    and HASH.fullmatch(pin["manifest_sha256"]),
                    "invalid Q1 score-parent pin",
                )
        baseline = pair["B0"]["score_parents"]
        paired = pair["Q1"]["score_parents"]
        require(
            "prediction" in paired
            and all(baseline.get(role) == pin for role, pin in paired.items())
            and (pair["Q1"]["whole_render_failure"] or set(paired) == set(PAIR_ROLES)),
            "Q1 paired learned, prediction or rules pin differs from its same-family B0",
        )


def check_cv(cv, families):
    counts = {
        f: (sum(v["rows"] for v in r["views"].values()), sum(v["positive_rows"] for v in r["views"].values()))
        for f, r in families.items()
    }
    trials = cv["fold_attempts"]
    require(
        trials
        and [r["k"] for r in trials] == [4, 3, 2][: len(trials)]
        and trials[-1]["usable_training_folds"]
        and not any(r["usable_training_folds"] for r in trials[:-1]),
        "missing valid grouped training folds",
    )
    for attempt in trials:
        valid_train, all_val = True, []
        for fold in attempt["folds"]:
            training, validation = set(fold["train_families"]), set(fold["validation_families"])
            require(
                not training & validation and training | validation == set(families) and validation,
                "CV family leakage or empty validation fold",
            )
            all_val.extend(validation)
            n, p = (sum(counts[f][i] for f in training) for i in range(2))
            m, q = (sum(counts[f][i] for f in validation) for i in range(2))
            require(
                (
                    fold["train_rows"],
                    fold["train_positive"],
                    fold["train_negative"],
                    fold["validation_rows"],
                    fold["validation_positive"],
                )
                == (n, p, n - p, m, q),
                "CV class counts disagree with selected family rows",
            )
            valid_train &= 0 < p < n
        require(
            len(attempt["folds"]) == attempt["k"]
            and sorted(all_val) == sorted(families)
            and attempt["usable_training_folds"] == valid_train,
            "CV fallback or coverage contradicts the recorded classes",
        )
    for method in MODELS:
        selected = cv["models"][method]
        require(
            {float(c) for c in selected["candidates"]} == {0.1, 1.0, 10.0}
            and selected["selected_C"] in (0.1, 1.0, 10.0)
            and all(len(c["folds"]) == trials[-1]["k"] for c in selected["candidates"].values()),
            "missing regularization candidate or completed CV fold",
        )


def check_identity(folder, index, tables):
    model = load_package(folder / "model", expected_index_sha256=index["model_index_sha256"])["index"]
    method = load(folder / "protocol/method-freeze.json")
    catalog = load(folder / "protocol/families-v1.json")
    fit = load(folder / "proofs/model-fit.json")
    dev = load(folder / "proofs/development.json")
    train = load(folder / "training/training_dataset_manifest.json")
    cv = load(folder / "training/cv_results.json")
    fit_index = load(folder / "proofs/fitted-model-index.json")
    study = load(folder / "protocol/study-v1.json")
    features = load(folder / "model/feature_schema.json")
    require(
        index["protocol_version"] == study["numerical_protocol_version"]
        and index["label_version"] == study["labels"]["version"]
        and index["feature_version"] == features["schema"],
        "release version labels differ from the registered protocol or exported features",
    )
    runs = index["sources"]
    require(
        set(runs) == {"model", "fit", "example", "example_source", "example_scores", "report"}
        and all(
            set(p) == {"run_id", "manifest_sha256"}
            and RUN_ID.fullmatch(p["run_id"])
            and HASH.fullmatch(p["manifest_sha256"])
            for p in runs.values()
        ),
        "missing or invalid public source pins",
    )
    require(
        model["model_id"]
        == model["threshold_run_id"]
        == method["model_id"]
        == method["threshold_select_run_id"]
        == index["model_id"]
        == runs["model"]["run_id"]
        == dev["run_id"]
        and model["fit_run_id"]
        == method["model_fit_run_id"]
        == runs["fit"]["run_id"]
        == dev["fit_run_id"]
        == fit["fit_run_id"]
        == train["fit_run_id"]
        == cv["fit_run_id"]
        == index["fit_run_id"]
        and method["threshold_select_manifest_sha256"] == runs["model"]["manifest_sha256"]
        and method["model_fit_manifest_sha256"] == runs["fit"]["manifest_sha256"]
        and index["method_code_commit"] == method["method_code_commit"]
        and index["model_index_sha256"] == dev["model_index_sha256"] == method["model_index_sha256"]
        and index["thresholds_sha256"] == model["files"]["thresholds.json"] == method["thresholds_sha256"]
        and index["feature_schema_sha256"] == model["files"]["feature_schema.json"],
        "public fit, development, model or threshold identities differ",
    )
    require(
        sha(folder / "protocol/method-freeze.json") == index["method_freeze_sha256"]
        and sha(folder / "protocol/families-v1.json")
        == index["catalog_sha256"]
        == method["catalog_sha256"]
        == model["split_manifest_sha256"]
        == train["split_manifest_sha256"]
        == dev["split_manifest_sha256"]
        and sha(folder / "protocol/study-v1.json") == method["study_sha256"] == index["study_sha256"]
        and sha(folder / "protocol/formal-freeze-v1.json") == method["formal_freeze_sha256"]
        and sha(folder / "training/training_dataset_manifest.json")
        == fit["training_manifest_sha256"]
        == model["training_manifest_sha256"]
        == index["training_manifest_sha256"]
        and sha(folder / "training/cv_results.json") == fit["cv_sha256"]
        and sha(folder / "proofs/development.json") == method["development_receipt_sha256"]
        and sha(folder / "proofs/model-fit.json") == index["fit_receipt_sha256"],
        "public scientific artifact digest differs from its source",
    )
    freeze = load(folder / "protocol/formal-freeze-v1.json")
    add = load(folder / "protocol/analysis-addendum-v1.json")
    op = load(folder / "protocol/operating-points-v1.json")
    require(
        freeze["catalog_sha256"] == index["catalog_sha256"]
        and freeze["preregistration_sha256"] == index["study_sha256"]
        and add["study_sha256"] == op["parent_study_sha256"] == index["study_sha256"]
        and op["parent_addendum_sha256"] == sha(folder / "protocol/analysis-addendum-v1.json")
        and all(
            sha(folder / local_path(k)) == v
            for k, v in method["comparison_files"].items()
            if k.startswith("protocol/")
        ),
        "public registration, amendments or operating points differ from the frozen study",
    )
    require(
        fit_index["status"] == "fit"
        and fit_index["model_id"] == index["fit_run_id"]
        and sha(folder / "proofs/fitted-model-index.json") == fit["model_index_sha256"]
        and fit_index["files"] == {n: h for n, h in model["files"].items() if n != "thresholds.json"}
        and all(
            fit_index.get(k) == model.get(k)
            for k in (
                "training_manifest_sha256",
                "split_manifest_sha256",
                "constant_risk",
                "training_sources",
            )
        ),
        "released coefficients, preprocessing or training identity differ from the original fit",
    )
    require(
        index["comparison_protocol_sha256"]
        == method["comparison_protocol_sha256"]
        == hashlib.sha256(
            json.dumps(method["comparison_files"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "comparison aggregate differs",
    )
    train_names = {r["scene_family_id"] for r in catalog["families"] if r["split"] == "train"}
    dev_names = {r["scene_family_id"] for r in catalog["families"] if r["split"] == "development"}
    require(
        fit["status"] == "fitted"
        and dev["status"] == "frozen"
        and len(train_names) == 16
        and len(dev_names) == 4
        and set(train["family_sources"]) == train_names == set(model["training_sources"])
        and set(dev["family_sources"]) == dev_names == set(model["development_sources"])
        and set(cv["models"]) == set(MODELS)
        and all(m["selected_C"] in (0.1, 1, 10) for m in cv["models"].values()),
        "release lacks the complete training and development populations",
    )
    positive = total = 0
    for family, record in train["family_sources"].items():
        require(
            set(record["views"]) == set(TARGETS)
            and record["source_input_sha256"] == model["training_sources"][family],
            "training view or source inventory mismatch",
        )
        for row in record["views"].values():
            require(
                0 <= row["positive_rows"] <= row["rows"] == min(row["eligible"], 4096)
                and row["eligible"] <= row["E"] <= row["V"],
                "invalid training population counts",
            )
            total += row["rows"]
            positive += row["positive_rows"]
    require(
        0 < positive < total
        and positive == train["positive_count"] == fit["positives"]
        and total == train["row_count"] == fit["rows"] == fit["parity_rows"]
        and all(fit["max_sklearn_numpy_error"][m] <= 1e-6 for m in MODELS),
        "release training counts or independent export check differ",
    )
    require(train["fold_attempts"] == cv["fold_attempts"], "training manifest and CV use different folds")
    check_cv(cv, train["family_sources"])
    report = load(folder / "proofs/report-public.json")
    table_files = {n: signature(folder / "results" / n) for n in tables}
    link_report(
        report,
        load(folder / "results/selection.json"),
        method,
        index["method_freeze_sha256"],
        catalog,
        table_files,
        index["kind"],
    )
    require(
        report["schema"] == "el-public-report-projection-1"
        and report["run_id"] == runs["report"]["run_id"]
        and HASH.fullmatch(report["original_report_sha256"])
        and report["original_report_sha256"] == index["original_report_receipt_sha256"],
        "public report proof identifies another run or receipt",
    )
    check_q1_pairings(report["q1_baseline_bindings"], report["selected_trials"])
    demo = load(folder / "demo.json")
    score = load(folder / "proofs/example-score.json")
    origin = load(folder / "example/provenance.json")
    pred = load(folder / "example/predictions/prediction.json")
    source = load(folder / "proofs/example-source.json")
    require(
        source["schema"] == "el-example-source-projection-1"
        and source["original_inputs_sha256"] == index["original_example_inputs_sha256"]
        and HASH.fullmatch(source["original_inputs_sha256"]),
        "incorrect example source projection digest",
    )
    original = source["reference"]
    require(HASH.fullmatch(original["metadata_sha256"]), "missing original reference producer digest")
    if original["schema"] == "el-synthetic-plane-source-1":
        require(
            index["kind"] == "synthetic_fixture", "synthetic source cannot identify a public study example"
        )
    else:
        require(
            original["schema"] == "el-reference-1"
            and HASH.fullmatch(original["native_blend_sha256"])
            and HASH.fullmatch(original["validation_sha256"]),
            "missing original Blender scene or validation digest",
        )
    package, roles, inputs = public_inputs(folder)
    check_report_thresholds(
        folder / "results/family-results.csv", package["thresholds"], report["selected_trials"]
    )
    require(
        demo["schema"] == "el-clean-demo-1"
        and score["schema"] == "el-l1-score-1"
        and score["status"] == "scored"
        and demo["methods"] == score["methods"] == KEYS
        and score["thresholds"] == package["thresholds"]
        and demo["maximum_absolute_score_difference"] == ABS_TOL
        and score["model_files"] == package["file_hashes"]
        and demo["model_id"] == score["model_id"] == index["model_id"]
        and demo["score_run_id"] == score["run_id"] == runs["example_scores"]["run_id"]
        and demo["producer_run_id"]
        == score["producer_run_id"]
        == origin["run_id"]
        == pred["producer_run_id"]
        == runs["example"]["run_id"]
        and demo["original_score_receipt_sha256"]
        == sha(folder / "proofs/example-score.json")
        == index["example_score_receipt_sha256"]
        and demo["input_files"] == inputs
        and score["inputs"] == {r: inputs[n] for r, n in roles.items()}
        and demo["model_index_sha256"]
        == index["model_index_sha256"]
        == score["model_files"]["model_index"]["sha256"]
        and demo["expected_views"] == {"v01": signature(folder / "expected/v01.npz")}
        and demo["expected_views"]["v01"]["sha256"] == score["views"]["v01"]["sha256"],
        "original example or expected scores differ",
    )
    expected = folder / "expected/v01.npz"
    values = compare(expected, expected)
    require(
        values["passed"]
        and all(
            values["methods"][m]["finite_expected"] == score["views"]["v01"]["l1_scoreable"] > 0
            for m in MODELS
        ),
        "authored example does not exercise a numerical prediction for both learned methods",
    )
    require(
        origin["schema"] == "el-original-example-1"
        and origin["adapter"] == pred["adapter"] == "evidence-lens-textured-plane-1"
        and source["parent"] == runs["example_source"]
        and source["input_sha256"] == origin["source_png_sha256"]
        and source["camera_sha256"] == origin["source_cameras_sha256"]
        and pred["input_sha256"] == inputs["example/input.png"]["sha256"]
        and pred["camera_sha256"] == inputs["example/cameras.json"]["sha256"]
        and pred["depth_validation_receipt_sha256"] == sha(folder / "example/predictions/depth-probe.json")
        and pred["depth_validation"] == load(folder / "example/predictions/depth-probe.json")
        and set(pred["views"]) == set(VIEWS)
        and all(pred["views"][v] == inputs[f"example/predictions/{v}.npz"] for v in VIEWS),
        "original image, cameras, calibration or rendered example does not match its producer",
    )
