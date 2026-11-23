"""Bind the single Stage C test opening to its recorded, pre-test method and roster."""

import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from ._bundle import HASH, RUN_ID, load, local_path, save, sha, utc
from ._index import writer_gate
from ._verify import matches, parent_directory, require, verify_bundle
from .formal_inputs import PinnedRun
from .stage3_decision import check_independent_readbacks, check_neural_fits, check_selection

MEMBER = "payload/method-freeze.json"
CONDITIONS = ["M2", "M4"]
METHODS = ("B-L1-transfer", "MS-L1", "MS-frozen", "MS-ft")
ROLES = {
    "catalog", "study", "feature_schema", "reconstruction_protocol", "learning_protocol",
    "training_dataset", "development_dataset", "lightweight_development", "neural_selection",
    "thresholds", "tensor_comparison", "cpu_validation", "production_budget",
}


def _pin(value):
    require(isinstance(value, dict) and matches(RUN_ID, value.get("run_id"))
            and matches(HASH, value.get("manifest_sha256")), "Invalid Stage C run pin")
    return {key: value[key] for key in ("run_id", "manifest_sha256")}


def _run(store, shelf, pin):
    pin = _pin(pin)
    path = parent_directory(store, pin, shelf)
    run = PinnedRun(path.parent.parent, pin)
    require(run.run["execution_status"] == "succeeded", "Stage C authority refers to an unsuccessful run")
    return run


def _path(store, digest):
    require(matches(HASH, digest), "Invalid Stage C roster SHA-256")
    return Path(store).resolve() / "protocol" / "stage3" / digest / "test-opening.json"


def _time(stamp):
    require(isinstance(stamp, str) and stamp.endswith("Z"), "Expected a recorded UTC timestamp")
    return datetime.fromisoformat(stamp[:-1] + "+00:00")


def _read_method(store, shelf, marker):
    freeze = _run(store, shelf, marker["method"])
    path = freeze.open(MEMBER)
    require(freeze.run["run_kind"] == "report" and freeze.run["experiment_id"] == "stage3-method-freeze"
            and sha(path) == marker["method_sha256"], "Opening and recorded method disagree")
    method = load(path)
    require(method.get("schema") == "el-stage3-method-freeze-1" and method.get("run_id") == freeze.pin["run_id"]
            and method.get("catalog_sha256") == marker["catalog_sha256"]
            and method.get("test_families") == marker["test_families"] and method.get("conditions") == CONDITIONS
            and set(method.get("models", {})) == set(METHODS), "Method does not cover the registered comparison")
    return freeze, method


def _check_marker(store, shelf, marker, catalog):
    require(marker.get("schema") == "el-stage3-test-opening-1" and marker.get("catalog_sha256") == catalog
            and matches(HASH, marker.get("method_sha256")) and marker.get("conditions") == CONDITIONS
            and isinstance(marker.get("test_families"), list) and len(set(marker["test_families"])) == 4
            and len(marker["test_families"]) == 4, "Opening does not match the four-family Stage C cohort")
    freeze, method = _read_method(store, shelf, marker)
    opened = _time(marker.get("opened_at_utc"))
    require(_time(freeze.run["ended_at_utc"]) <= opened <= datetime.now(timezone.utc),
            "Opening time does not follow the method freeze")
    return method


def require_opening(store, catalog_sha256, family=None, opening_sha256=None, shelf=None):
    """Validate the single committed marker and its original sealed method before accessing test data."""
    store = Path(store).resolve()
    shelf = Path(shelf).resolve() if shelf is not None else store.parent / "cold-shelf"
    p = _path(store, catalog_sha256)
    require(p.is_file() and not p.is_symlink(), "Stage C test families are still sealed")
    if opening_sha256 is not None:
        require(matches(HASH, opening_sha256) and sha(p) == opening_sha256, "Unexpected Stage C opening SHA-256")
    marker = load(p)
    _check_marker(store, shelf, marker, catalog_sha256)
    pin = _pin(marker["method"])
    _original_sources(store, shelf, pin["run_id"], pin["manifest_sha256"], marker["method_sha256"])
    require(family is None or family in marker["test_families"], "Family is outside the Stage C test opening")
    return marker


def get_frozen_method(store, marker, shelf=None):
    """Read a method only through its actual store marker."""
    store = Path(store).resolve()
    shelf = Path(shelf).resolve() if shelf is not None else store.parent / "cold-shelf"
    authoritative = require_opening(store, marker["catalog_sha256"], shelf=shelf)
    require(authoritative == marker, "The supplied opening differs from the stored record")
    return _read_method(store, shelf, authoritative)[1]


def _check_sources(store, shelf, freeze, method):
    direct = {_pin(r)["run_id"]: _pin(r)["manifest_sha256"] for r in load(freeze.open("inputs.json"))["parents"]}
    verify_method_sources(store, shelf, method, direct)


def verify_method_sources(store, shelf, method, direct):
    """Check the same original evidence during recording and at test opening."""
    cache = {}

    def original(run_id):
        require(run_id in direct, "Original fit missing from the method recorder parents")
        if run_id not in cache:
            cache[run_id] = _run(store, shelf, {"run_id": run_id, "manifest_sha256": direct[run_id]})
        return cache[run_id]

    def source(spec):
        p = _pin(spec)
        require(direct.get(p["run_id"]) == p["manifest_sha256"] and matches(HASH, spec.get("sha256")),
                "Missing direct scientific parent or member digest")
        run = original(p["run_id"])
        name = local_path(spec["member"])
        require(name in run.files and run.files[name]["sha256"] == spec["sha256"],
                "Scientific member differs from its recorder inventory")
        return run.open(name)

    components = method["components"]
    require(set(components) == ROLES, "Incomplete reconstruction and risk evidence")
    paths = {k: source(v) for k, v in components.items()}
    study, catalog = load(paths["study"]), load(paths["catalog"])
    require(study["schema"] == "el-stage3-study-1" and catalog["schema"] == "el-stage3-family-roster-1"
            and study["catalog_sha256"] == method["catalog_sha256"] == components["catalog"]["sha256"]
            and method["study_sha256"] == components["study"]["sha256"]
            and method["feature_schema_sha256"] == components["feature_schema"]["sha256"] == study["feature_schema_sha256"]
            and study["sources"] == catalog["source_sets"] and study["targets"] == method["targets"] == catalog["target_ids"],
            "Study or source, camera, feature or roster identity differs")
    learning, optimizer, budget = (load(paths[k]) for k in ("learning_protocol", "reconstruction_protocol", "production_budget"))
    require(learning["schema"] == "el-stage3-risk-learning-protocol-1" and learning["study_sha256"] == method["study_sha256"]
            and learning["catalog_sha256"] == method["catalog_sha256"] and learning["feature_schema_sha256"] == method["feature_schema_sha256"]
            and optimizer["schema"] == "el-stage3-joint-optimizer-1" and budget["schema"] == "el-stage3-production-budget-1"
            and budget["optimizer_sha256"] == components["reconstruction_protocol"]["sha256"]
            and budget["catalog_sha256"] == method["catalog_sha256"] and budget["scientific_production_restarts_from_initial"] is True
            and budget["condition_settings_identical"] is True and budget["pilot_samples_in_scientific_dataset"] == 0
            and budget["gaussian_count"] == optimizer["initialization"]["count"]
            and budget["optimization_steps"] in optimizer["optimization"]["production_candidate_updates"]
            and budget["pixels_per_condition"] == budget["optimization_steps"] * math.prod(study["resolution"]),
            "Reconstruction or learning configuration differs from registered evidence")
    families = {split: [r["scene_family_id"] for r in catalog["families"] if r["split"] == split]
                for split in ("train", "development", "test")}
    require([len(families[k]) for k in families] == [8, 4, 4] and len(set(sum(families.values(), []))) == 16
            and families["test"] == method["test_families"] == catalog["test_families"], "Family partition differs")
    for role, split in (("training_dataset", "train"), ("development_dataset", "development")):
        dataset = load(paths[role])
        keys = [(r["family"], r["condition"], r["view"]) for r in dataset["views"]]
        expected = {(f, c, v) for f in families[split] for c in CONDITIONS for v in study["targets"]}
        require(dataset["schema"] == "el-stage3-risk-dataset-1" and dataset["split"] == split
                and dataset["catalog_sha256"] == method["catalog_sha256"]
                and dataset["cohort"] == "formal" and dataset["families"] == families[split]
                and dataset["learning_sha256"] == components["learning_protocol"]["sha256"]
                and dataset["study_sha256"] == method["study_sha256"]
                and dataset["feature_schema_sha256"] == method["feature_schema_sha256"]
                and len(keys) == len(expected) and set(keys) == expected, "Training or development source roster differs")
    thresholds = method["thresholds"]
    require(set(thresholds) == set(METHODS) and method["scorer_scopes"] == ["one", "two", "all"]
            and method["default_method"] in ("MS-L1", "MS-ft") and method["primary_seed"] == study["neural"]["release_seed"],
            "Frozen methods, source scopes or deployment choice differ")
    for name, model in method["models"].items():
        require(isinstance(model.get("model_id"), str) and bool(model["model_id"]) and
                type(model.get("threshold")) in (int, float) and math.isfinite(model["threshold"])
                and model["threshold"] == thresholds[name] and model["index"]["sha256"] == model["model_index_sha256"]
                and model["weights"]["sha256"] == model["weights_sha256"]
                and _pin(model["index"]) == _pin(model["weights"]), "Invalid frozen model or operating point")
        source(model["index"])
        source(model["weights"])
    transfer = method["models"]["B-L1-transfer"]
    predecessor = study["stage_b"]
    require(transfer["model_index_sha256"] == predecessor["transfer_model_index_sha256"]
            and transfer["weights_sha256"] == predecessor["transfer_weights_sha256"]
            and transfer["threshold"] == predecessor["transfer_threshold"], "Single-source transfer was refitted")
    require(len({m["weights_sha256"] for m in method["models"].values()}) == len(METHODS),
        "The four methods must resolve to distinct real weights")

    selection, thresholds, lightweight = (load(paths[k]) for k in ("neural_selection", "thresholds", "lightweight_development"))
    check_selection(method, study, selection, thresholds, lightweight)
    roster = check_neural_fits(method, study, thresholds, original, store, shelf)
    check_independent_readbacks(method, study, roster, load(paths["tensor_comparison"]), load(paths["cpu_validation"]), original)


@lru_cache(maxsize=16)
def _original_sources(store, shelf, run_id, manifest_sha256, method_sha256):
    # The cache key includes both sealed digests. Each new process authenticates its own sources.
    freeze = _run(store, shelf, {"run_id": run_id, "manifest_sha256": manifest_sha256})
    source = freeze.open(MEMBER)
    require(sha(source) == method_sha256, "Opening refers to another sealed method member")
    _check_sources(store, shelf, freeze, load(source))


def open_cohort(store, catalog_sha256, run_id, manifest_sha256, shelf=None):
    """Authenticate full evidence then create the one shared, timestamped opening."""
    store = Path(store).resolve()
    shelf = Path(shelf).resolve() if shelf is not None else store.parent / "cold-shelf"
    path = _path(store, catalog_sha256)
    pin = _pin({"run_id": run_id, "manifest_sha256": manifest_sha256})
    freeze = _run(store, shelf, pin)
    require(verify_bundle(freeze.root)["manifest_sha256"] == manifest_sha256, "Invalid method seal")
    method_path = freeze.open(MEMBER)
    method = load(method_path)
    marker = {"schema": "el-stage3-test-opening-1", "catalog_sha256": catalog_sha256,
              "method": pin, "method_sha256": sha(method_path), "test_families": method["test_families"],
              "conditions": CONDITIONS, "opened_at_utc": utc()}
    _check_marker(store, shelf, marker, catalog_sha256)
    _original_sources(store, shelf, run_id, manifest_sha256, marker["method_sha256"])
    with writer_gate(store):
        if path.exists():
            old = require_opening(store, catalog_sha256, shelf=shelf)
            require(old["method"] == pin and old["method_sha256"] == marker["method_sha256"],
                    "This cohort has already been opened with a different freeze")
            return old
        path.parent.mkdir(parents=True, exist_ok=True)
        marker["opened_at_utc"] = utc()
        save(path, marker)
        path.chmod(0o444)
    return require_opening(store, catalog_sha256, shelf=shelf)
