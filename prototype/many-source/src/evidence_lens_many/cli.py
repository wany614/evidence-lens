"""Command-line entry points for a prepared, registered static 3DGS source bank."""

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from evidence_lens_bundle.common import encoded, json_bytes, require

from .bank import Bank
from .legacy import legacy_q4
from .output import file_sha, report, write_result
from .schema import BASELINE_ID, FEATURE_SCHEMA_SHA256, RULES, feature_schema, score_lr, simple_ranks
from .select import select_sources
from .stream import audit_subset, stream_features


def _budget(value):
    if value == "all":
        return value
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("budget must be a positive integer or all") from exc
    if n < 1:
        raise argparse.ArgumentTypeError("budget must be positive")
    return n


def _inspect_ply(path):
    with path.open("rb") as f:
        head = f.read(65536).split(b"end_header", 1)
    require(len(head) == 2 and head[0].splitlines()[:1] == [b"ply"], "unrecognized PLY header")
    return {
        "status": "structural_inspection_only",
        "declared_vertex_count": next(
            (
                line.split()[-1].decode()
                for line in head[0].splitlines()
                if line.startswith(b"element vertex ")
            ),
            None,
        ),
        "missing_for_full_scoring": [
            "verified upstream photo membership",
            "source photo permissions and decoded inputs",
            "calibrated cameras and coordinate receipt",
            "supported Gaussian semantics",
            "native source and target RGB, expected camera-Z and opacity, depth probe",
        ],
    }


def _import(bank, dest):
    specs = {}
    for name in bank.accessible_ids:
        if name in bank.granted:
            for field in ("photo", "render"):
                ref = bank.meta["sources"][name][field]
                if ref is not None and bank._path(ref) is not None:
                    specs[ref["path"]] = ref
    for name in bank.target_ids:
        ref = bank.meta["targets"][name]["render"]
        if ref is not None and bank._path(ref) is not None:
            specs[ref["path"]] = ref
    for name, ref in specs.items():
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = bank._data(
            ref,
            "source"
            if name not in {s["render"]["path"] for s in bank.meta["targets"].values() if s["render"]}
            else "target",
        )
        with path.open("xb") as f:
            f.write(raw)
    (dest / "bank.json").write_bytes((bank.root / "bank.json").read_bytes())
    return Bank(dest, bank.sha256).inspect()


def _score(bank, a, dest):
    plan = select_sources(bank, a.target, strategy=a.strategy)
    query = a.budget
    if a.action == "features":
        slots = []
        for step in stream_features(bank, a.target, plan, budgets=a.budgets):
            path = dest / f"k{step.actual_k}.npz"
            np.savez_compressed(
                path,
                X=step.X,
                finite=step.finite,
                reason=step.reason,
                comparable_count=step.comparable_count,
                support_count=step.support_count,
                first_witness=step.first_witness,
            )
            slots.append(
                {
                    "actual_k": step.actual_k,
                    "nominal_budgets": step.nominal_budgets,
                    "queried_source_ids": step.queried_source_ids,
                    "file": path.name,
                    "sha256": file_sha(path),
                    "cost": step.cost,
                }
            )
        result = {
            "schema": "el-b-bank-feature-export-1",
            "baseline_id": BASELINE_ID,
            "bank_sha256": bank.sha256,
            "target": a.target,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "selection": plan.cost,
            "slots": slots,
        }
        (dest / "features.json").write_bytes(encoded(result))
        return result
    do_audit = a.action == "audit-bank"
    chosen_k = min(query, len(plan.ids)) if query != "all" else len(plan.ids)
    selected = full = None
    for step in stream_features(bank, a.target, plan, budgets=(query, "all") if do_audit else (query,)):
        if step.actual_k == chosen_k:
            selected = step
        if do_audit and step.actual_k == len(plan.ids):
            full = step
    scopes = audit_subset(selected, full)
    rules = simple_ranks(selected)
    if a.model:
        digest = file_sha(a.model)
        require(a.model_sha256 is None or a.model_sha256 == digest, "different ranking head SHA-256")
        model = json_bytes(a.model.read_bytes())
        rank = score_lr(selected, model)
        authority = {
            "name": "B-CPU-LR",
            "threshold_status": "uncalibrated_ranking",
            "model_sha256": digest,
            "tau": None,
        }
    else:
        require(a.model_sha256 is None, "a model digest requires a model")
        rank = rules[a.rule]
        authority = {"name": "rule_" + a.rule, "threshold_status": "uncalibrated_rule_ranking", "tau": None}
    arrays = {
        "rank": rank,
        "finite": selected.finite,
        "reason": selected.reason,
        "scope_state": scopes["state"],
        "gap_witness": scopes["gap_witness"],
        "selected_gap": scopes["selected_gap"],
        "query_count": selected.query_count,
        **{"rule_" + k: v for k, v in rules.items()},
    }
    for key in (
        "comparable_count",
        "in_frame_count",
        "support_count",
        "depth_conflict_count",
        "colour_conflict_count",
        "first_witness",
        "incomplete",
        "invalid_geometry",
    ):
        arrays[key] = getattr(selected, key)
    old_profiles = {}
    for name in a.legacy:
        old = legacy_q4(bank, a.target, selected.queried_source_ids, name, a.legacy_models, a.threads)
        old_profiles[name] = old["profile"]
        arrays.update(
            {"legacy_" + name + "_" + key: old[key] for key in ("rank", "finite", "state", "reason")}
        )
    authority["legacy_q4_transfers"] = old_profiles
    scope = {
        "requested": do_audit,
        "full_complete": scopes["full_complete"],
        "unique_library_size": len(bank.ids),
        "actual_queried": selected.actual_k,
        "full_scope_cost": full.cost if full is not None else None,
        "model_used_query_count": selected.actual_k,
        "first_witness_source_ids": list(bank.ids),
    }
    r = write_result(dest, bank, a.target, plan, selected, arrays, authority, scope)
    report(dest, dest / "report.html")
    return {
        "receipt": "scores.json",
        "bank_sha256": bank.sha256,
        "rank_eligible_pixels": r["rank_eligible_pixels"],
        "scope_counts": r["scope_counts"],
        "model": authority,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("schema", help="Print the numerical definitions, denominators and missing flags")
    for name in ("inspect", "import", "select", "features", "score", "audit-bank"):
        q = sub.add_parser(name)
        q.add_argument(
            "--bank",
            type=Path,
            required=True,
            help="prepared bank directory; inspect also accepts a bare PLY",
        )
        q.add_argument("--bank-sha256")
        q.add_argument("--cache-mib", type=int, default=128)
        if name == "inspect":
            q.add_argument(
                "--verify-members",
                action="store_true",
                help="read and verify all permitted source and target members",
            )
        if name in ("features", "score", "audit-bank", "import"):
            q.add_argument("--out", required=True, type=Path)
        if name in ("select", "features", "score", "audit-bank"):
            q.add_argument("--target", required=True)
            q.add_argument("--strategy", choices=("fixed", "nearest", "coverage"), default="coverage")
            q.add_argument("--budget", type=_budget, default=4)
        if name == "features":
            q.add_argument("--budgets", type=_budget, nargs="+", default=[4, 8, 16, 32, "all"])
        if name in ("score", "audit-bank"):
            model = q.add_mutually_exclusive_group(required=True)
            model.add_argument("--model", type=Path)
            model.add_argument("--rule", choices=RULES)
            q.add_argument("--model-sha256")
            q.add_argument("--legacy", nargs="*", choices=("MS-L1", "MS-ft"), default=[])
            q.add_argument("--legacy-models", type=Path)
            q.add_argument("--threads", type=int, default=4)
    q = sub.add_parser("report")
    q.add_argument("--scores", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q = sub.add_parser(
        "evaluate", help="Evaluator-only command for explicitly supplied independent reference measurements"
    )
    for n in ("prediction", "reference", "scores", "out"):
        q.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args(argv)
    try:
        if a.action == "schema":
            result = {"sha256": FEATURE_SCHEMA_SHA256, **feature_schema()}
        elif a.action == "evaluate":
            from .evaluation import evaluate_pair

            result = evaluate_pair(a.prediction, a.reference, a.scores, a.out)
        elif a.action == "report":
            result = report(a.scores, a.out)
        elif a.action == "inspect" and a.bank.suffix.lower() == ".ply" and a.bank.is_file():
            result = _inspect_ply(a.bank)
        else:
            bank = Bank(a.bank, a.bank_sha256, cache_bytes=a.cache_mib * (1 << 20))
            if a.action == "inspect":
                result = bank.inspect()
                if a.verify_members:
                    for n in bank.ids:
                        bank.source(n)
                    for n in bank.target_ids:
                        bank.target(n)
                    result["verified_member_access"] = bank.counters
            elif a.action == "select":
                plan = select_sources(bank, a.target, strategy=a.strategy)
                k = len(plan.ids) if a.budget == "all" else min(len(plan.ids), a.budget)
                result = {
                    "bank_sha256": bank.sha256,
                    "target": a.target,
                    "nominal_budget": a.budget,
                    "actual_k": k,
                    "queried_source_ids": plan.ids[:k],
                    "strategy": a.strategy,
                    "reasons": plan.reasons[:k],
                    "cost": plan.cost,
                }
            else:
                require(not a.out.exists(), "output already exists")
                a.out.parent.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(dir=a.out.parent) as tmp:
                    dest = Path(tmp) / "payload"
                    dest.mkdir()
                    result = _import(bank, dest) if a.action == "import" else _score(bank, a, dest)
                    require(not a.out.exists(), "output already exists")
                    os.rename(dest, a.out)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
