"""Seal the implementation used by the independent target reader and enforce it at opening and evaluation."""

import argparse
import os
from pathlib import Path

from dtu_eval import read_pinned
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from t0_authority import opening
from t0_evaluate import evaluate

# Original registration pins the scientific definitions; this file also pins their reader and implementation dependencies.
SUPPLEMENT = [
    "scripts/t0_study.py",
    "scripts/t0_score_matrix.py",
    "scripts/t0_opening_software.py",
    "scripts/t0_dtu_convert.py",
    "scripts/t0_dtu_record.py",
    "scripts/t0_reconstruct_readback.py",
    "scripts/t0_reconstruct.py",
    "scripts/dtu_sample.py",
    "scripts/dtu_eval.py",
    "scripts/dtu_eval_board.py",
    "scripts/recovery_eval_risk.py",
    "scripts/vision_evaluate.py",
    "scripts/vision_sources.py",
    "scripts/stage3_risk_boards.py",
    "src/evidence_lens/dtu_camera.py",
    "src/evidence_lens/multisource_io.py",
    "src/evidence_lens/features.py",
    "src/evidence_lens/multisource.py",
    "src/evidence_lens/scoring_io.py",
    "src/evidence_lens/formal_inputs.py",
]


def original(roots, pin):
    return load(read_pinned(roots, pin)[1])


def authenticate(project, docs):
    for name, digest in docs.items():
        require(sha(project / name) == digest, "changed pre-opening software: " + name)


def seal(config, roots, out, project):
    fp = config["sources"]["freeze"]
    fr = original(roots, fp)
    rehearsal = original(roots, config["sources"]["authored_fixture"])
    require(
        rehearsal["physical_T0_objects"] == rehearsal["target_reference_files_opened"] == 0
        and rehearsal["complete_maps_tested"] == rehearsal["deliberately_absent_maps_tested"] == 24,
        "missing evaluator software rehearsal",
    )
    require(
        rehearsal["evaluation_code_sha256"] == fr["new_scientific_files"]["scripts/t0_evaluate.py"],
        "rehearsal used a different evaluator",
    )
    frozen = {**fr["new_scientific_files"], **fr["protocol"]["source_sha256"]}
    authenticate(project, frozen)
    files = {**frozen, **{n: sha(project / n) for n in SUPPLEMENT}}
    revisions = {}
    if "prior_lock" in config["sources"]:
        prior = original(roots, config["sources"]["prior_lock"])
        audit = original(roots, config["sources"]["recovery_audit"])
        require(
            prior["sources"]["freeze"] == fp
            and audit["schema"] == "el-t0-cross-store-audit-1"
            and audit["new_target_reference_files_opened"] == 0
            and audit["verified_runs"] > 0,
            "missing evidence for cross-store intake repair",
        )
        files["scripts/t0_reconstruct_history.py"] = sha(project / "scripts/t0_reconstruct_history.py")
        revisions = {
            n: {"before": prior["files"].get(n), "after": digest}
            for n, digest in files.items()
            if prior["files"].get(n) != digest
        }
        allowed = {
            "scripts/t0_reconstruct.py",
            "scripts/t0_reconstruct_history.py",
            "scripts/t0_study.py",
            "scripts/t0_opening_software.py",
        }
        require(
            set(revisions) <= allowed and not set(revisions).intersection(frozen),
            "unexpected change to the evaluator, reader or frozen scientific files",
        )
    save(
        out / "software-lock.json",
        {
            "schema": "el-t0-opening-software-1",
            "run_id": os.environ["EL_RUN_ID"],
            "sources": config["sources"],
            "original_freeze_files_verified": len(frozen),
            "preopening_implementation_changes": revisions,
            "files": files,
            "preopening_addendum": "Adds software identity checks to the existing authority chain and tests missing predictions on authored material. No changes to registered scientific definitions, parameters or object roster.",
            "surrogate_references": "authored source used only to exercise evaluator output shapes; no T0 targets",
            "target_references_read": 0,
        },
    )


def main(a):
    cfg = load(a.config)
    roots = (a.store, a.history)
    project = Path(__file__).resolve().parents[1]
    if a.action == "lock":
        return seal(cfg, roots, a.out, project)
    lockpin = cfg["sources"]["software_lock"]
    lock = original(roots, lockpin)
    require(
        lock["schema"] == "el-t0-opening-software-1"
        and lock["sources"]["freeze"] == cfg["sources"]["freeze"],
        "software lock belongs to a different freeze",
    )
    authenticate(project, lock["files"])
    if a.action == "open":
        opening(cfg, roots, a.request, a.out)
        path = a.out / "opening.json"
        doc = load(path)
        doc["software_lock"] = lockpin
        save(path, doc)
    else:
        req = load(a.request)
        op = original(roots, req["opening"])
        require(
            op["software_lock"] == lockpin and op["freeze"] == lock["sources"]["freeze"],
            "reference opening is for different software",
        )
        evaluate(roots, req["opening"], req["references"], a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["lock", "open", "evaluate"])
    for k in ("config", "store", "history", "out"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--request", type=Path)
    main(p.parse_args())
