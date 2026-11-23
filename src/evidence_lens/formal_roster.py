"""Expand an explicit study roster into reproducible private report inputs."""

import json
import math
from pathlib import Path

from ._bundle import LABEL, load
from .formal_inputs import CONDITIONS, PinnedRun, compact_pin, require
from .scoring_io import file_hash


def pinned_file(base, spec):
    path = (Path(base) / spec["path"]).resolve()
    require(
        path.is_file() and file_hash(path) == spec["sha256"], f"changed explicitly named file {path.name}"
    )
    return path


def measured_pointer(document, spec):
    """Resolve a JSON Pointer or its explicitly selected length; release numeric/bool measures only."""
    if isinstance(spec, str):
        spec = {"pointer": spec}
    expression = spec["pointer"]
    require(expression == "" or expression.startswith("/"), "invalid JSON Pointer")
    current = document
    for token in expression.split("/")[1:] if expression else ():
        token = token.replace("~1", "/").replace("~0", "~")
        current = current[int(token)] if isinstance(current, list) else current[token]
    op = spec.get("op", "value")
    if op == "length":
        require(isinstance(current, (list, dict)), "length requires a JSON object or array")
        current = len(current)
    elif op == "count_true":
        require(
            isinstance(current, list) and all(type(v) is bool for v in current),
            "count_true requires boolean values",
        )
        current = sum(current)
    else:
        require(op == "value", "unsupported proof measurement")
    require(
        current is None or type(current) in (bool, int, float) and math.isfinite(current),
        "public proof measures must be finite numbers, booleans or null",
    )
    return current


def roster(store, path, destination):
    path = Path(path).resolve()
    cfg = load(path)
    require(
        cfg["schema"] == "el-study-report-roster-1" and cfg["kind"] in ("synthetic_fixture", "heldout_study"),
        "unknown report roster",
    )
    pfiles = {
        k: {"path": str(pinned_file(path.parent, v)), "sha256": v["sha256"]}
        for k, v in cfg["protocol_files"].items()
    }
    catalog = load(Path(pfiles["catalog"]["path"]))
    method = load(Path(pfiles["method_freeze"]["path"]))
    held = [r for r in catalog["families"] if r["split"] == "test"]
    require(len(held) == 4, "frozen catalog needs four test families")
    family_ids = {r["scene_family_id"] for r in held}
    all_trials = {(r["scene_family_id"], c) for r in catalog["families"] for c in r.get("trials", {})}
    evidence, proofs, markers, quality_digests = {}, {}, {}, {}

    def add_run(entry, purpose, **more):
        pin = compact_pin(entry)
        rid = pin["run_id"]
        existing = evidence.get(rid)
        require(
            existing is None or existing["manifest_sha256"] == pin["manifest_sha256"],
            "conflicting roster run pins",
        )
        if existing:
            require(
                (purpose == "timing_repeat") == (existing["purpose"] == "timing_repeat"),
                "timing repeat also used scientifically",
            )
            if "timing_index" in more:
                require(
                    existing.get("timing_index", more["timing_index"]) == more["timing_index"],
                    "one run assigned to two repeats",
                )
            existing.update(more)
        else:
            evidence[rid] = dict(pin, purpose=purpose, **more)
        return pin

    def add_proof(id_, kind, source, sha, measures=None):
        require(
            isinstance(id_, str) and LABEL.fullmatch(id_) and LABEL.fullmatch(kind), "invalid receipt label"
        )
        spec = {"id": id_, "kind": kind, "path": str(source), "sha256": sha, "measures": measures or {}}
        if id_ in proofs:
            require(proofs[id_] == spec, "conflicting external proof identity")
        proofs[id_] = spec

    def transfer_markers(items):
        for rid, rows in items.items():
            for item in rows:
                identifier = item["receipt_id"]
                require(LABEL.fullmatch(identifier), "invalid transfer receipt name")
                p = pinned_file(
                    Path(store) / "receipts" / "transfer",
                    {"path": identifier + ".json", "sha256": item["sha256"]},
                )
                d = load(p)
                require(
                    d["run_id"] == rid and d["manifest_sha256"] == evidence[rid]["manifest_sha256"],
                    "transfer receipt belongs to another run",
                )
                measures = {
                    k: "/" + k
                    for k in ("elapsed_seconds", "bytes", "container_bytes", "logical_bytes", "file_count")
                    if type(d.get(k)) in (int, float)
                }
                add_proof("transfer-" + identifier, "transfer", p, item["sha256"], measures)

    def quality(spec):
        p = pinned_file(path.parent, spec)
        data = load(p)
        f, c = data["scene_family_id"], data["condition_id"]
        require(
            (f, c) in all_trials
            and data["selection_status"] == "selected"
            and data["schema"] in ("el-p2-trial-selection-1", "el-condition-selection-1"),
            "quality row outside the catalog",
        )
        require(
            (f, c) not in quality_digests or quality_digests[f, c] == spec["sha256"],
            "two different selected science chains",
        )
        quality_digests[f, c] = spec["sha256"]
        markers[f, c] = {
            "scene_family_id": f,
            "condition_id": c,
            "trial_id": data["trial_id"],
            "sha256": spec["sha256"],
            "selected": {k: compact_pin(v) for k, v in data["selected"].items()},
            "attempt_count": len(data["attempts"]),
        }
        for item in [*data["attempts"], *data["selected"].values()]:
            add_run(item, "scientific")
        add_proof("science-" + data["trial_id"], "scientific_marker", p, spec["sha256"])
        transfer_markers(data.get("transfer_receipts", {}))
        legacy = data.get("evidence", {}).get("transfer_receipts", {})
        if legacy:
            transfer_markers({data["selected"][role]["run_id"]: items for role, items in legacy.items()})
        return data

    def learned_or_fit(item, expected_kind, purpose):
        if "path" in item:
            p = pinned_file(path.parent, item)
            d = load(p)
            require(
                d.get("schema") == "el-scientific-stage-selection-1" and d.get("run_kind") == expected_kind,
                "incorrect learning-stage marker",
            )
            add_proof("stage-" + d["run_id"], "learning_marker", p, item["sha256"])
            item = d
        pin = add_run(item, purpose)
        run = PinnedRun(store, pin)
        require(
            run.run["run_kind"] == expected_kind and run.run["execution_status"] == "succeeded",
            "selected training-stage kind/status differs",
        )
        return pin

    fit = learned_or_fit(cfg["model_fit"], "model_fit", "training")
    frozen = learned_or_fit(cfg["threshold_select"], "threshold_select", "training")
    require(
        frozen
        == {
            "run_id": method["threshold_select_run_id"],
            "manifest_sha256": method["threshold_select_manifest_sha256"],
        },
        "wrong P5 threshold package",
    )
    if cfg["kind"] == "heldout_study":
        frozen_meta = load(PinnedRun(store, frozen).open("payload/model/model.json"))
        require(frozen_meta["fit_run_id"] == fit["run_id"], "P5 package derives from a different fit")
    trials = []
    for row in cfg["trials"]:
        f, c = row["scene_family_id"], row["condition_id"]
        pin = add_run(row["comparison"], "scientific")
        evaluated = load(PinnedRun(store, pin).open("payload/formal_evaluation.json"))
        require(
            (evaluated["scene_family_id"], evaluated["condition_id"]) == (f, c)
            and evaluated["kind"] == cfg["kind"]
            and evaluated["model_id"] == method["model_id"],
            "wrong comparison in roster",
        )
        if "quality" in row:
            q = quality(row["quality"])
            require(
                q["trial_id"] == evaluated["trial_id"]
                and all(
                    compact_pin(q["selected"][role]) == evaluated["parents"][parent]
                    for role, parent in (
                        ("reference", "reference"),
                        ("render", "prediction"),
                        ("score", "rules"),
                        ("evaluation", "labels"),
                    )
                ),
                "P6 comparison differs from the canonical quality chain",
            )
        for k, p in evaluated["parents"].items():
            add_run(p, "training" if k == "model" else "scientific")
        trials.append(dict(pin, scene_family_id=f, condition_id=c))
    require(
        len(trials) == 12
        and {(e["scene_family_id"], e["condition_id"]) for e in trials}
        == {(f, c) for f in family_ids for c in CONDITIONS},
        "roster requires each of the 12 planned P6 scientific rows",
    )
    for item in cfg.get("scientific_markers", []):
        quality(item)
    timed_renders = {}
    for spec in cfg.get("timing_markers", []):
        p = pinned_file(path.parent, spec)
        d = load(p)
        c, n, f = d["condition_id"], d["timing_repeat"], d["scene_family_id"]
        require(
            d["schema"] == "el-timing-repeat-1"
            and n in (1, 2, 3)
            and d["included_in_quality_or_training"] is False
            and f == held[0]["scene_family_id"]
            and d["catalog_sha256"] == pfiles["catalog"]["sha256"]
            and d["method_freeze_sha256"] == pfiles["method_freeze"]["sha256"]
            and d["warmup"]["selection_sha256"] == quality_digests[f, c],
            "timing marker does not match the registered family, warm-up or model",
        )
        require((f, c, n) not in timed_renders, "duplicate condition timing repeat")
        timed_renders[f, c, n] = (spec["sha256"], compact_pin(d["render"]))
        observed = {
            "timing_index": n,
            "timing_family": f,
            "timing_condition": c,
            "startup": "after selected quality warm-up",
        }
        for t in [*d["attempts"], d["producer"], d["render"]]:
            add_run(t, "timing_repeat", **observed)
        metrics = {
            k: "/" + k
            for k in ("producer_elapsed_seconds", "render_elapsed_seconds", "producer_seconds")
            if type(d.get(k)) in (int, float)
        }
        add_proof("timing-" + d["trial_id"], "windows_timing", p, spec["sha256"], metrics)
        transfer_markers(d.get("transfer_receipts", {}))
    for spec in cfg.get("cpu_timing", []):
        if "path" in spec:
            p = pinned_file(path.parent, spec)
            d = load(p)
            require(
                d.get("schema") == "el-scientific-stage-selection-1" and d["run_kind"] == "model_score",
                "wrong CPU timing marker",
            )
            add_proof("cpu-timing-" + d["run_id"], "cpu_timing", p, spec["sha256"])
        else:
            d = spec
        pin = compact_pin(d)
        tracked = PinnedRun(store, pin)
        rc = load(tracked.open("config.resolved.yaml"))
        n = rc["timing_repeat"]
        cond = tracked.run["condition_id"]
        f = tracked.run["scene_family_id"]
        require(
            n in (1, 2, 3)
            and f == held[0]["scene_family_id"]
            and cond in CONDITIONS
            and rc.get("quality_result") is False
            and tracked.run["run_kind"] == "model_score"
            and tracked.run["experiment_id"] == f"p6-timing-{n}-model-score",
            "CPU timing run also selected for quality or wrong repeat",
        )
        require(
            (f, cond, n) in timed_renders
            and (rc.get("timing_marker_sha256"), rc.get("render")) == timed_renders[f, cond, n],
            "CPU timing must pin the matching Windows repeat marker and independent timed render",
        )
        add_run(
            pin,
            "timing_repeat",
            timing_index=n,
            timing_family=f,
            timing_condition=cond,
            startup="after selected quality warm-up",
        )
    for spec in cfg.get("evidence", []):
        add_run(spec, spec["purpose"], **{k: v for k, v in spec.items() if k in ("documents", "startup")})
    for spec in cfg.get("proofs", []):
        p = pinned_file(path.parent, spec)
        add_proof(spec["id"], spec["kind"], p, spec["sha256"], spec.get("measures"))
    result = {
        "schema": "el-formal-report-selection-1",
        "kind": cfg["kind"],
        "protocol_files": pfiles,
        "trials": trials,
        "evidence": list(evidence.values()),
        "proofs": list(proofs.values()),
        "scientific_markers": list(markers.values()),
        "roster_sha256": file_hash(path),
        "inventory_scope": cfg.get(
            "inventory_scope", "Explicitly selected scientific, learning, timing and engineering records."
        ),
    }
    with Path(destination).open("x", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2, allow_nan=False)
        fp.write("\n")
    return result
