"""Record T0 freeze, published CPU scoring, collective target opening and independent evaluation."""

import argparse
import json
import sys
from pathlib import Path

from dtu_score import RELEASE_SHA
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from t0_records import bind, record, resolve

PROJECT = Path(__file__).resolve().parents[1]
RECOVERY = "20261123T223830955940Z-3256d8a180b341b3"
RELEASE = "20261123T193505072501Z-6046afc1fce4480b"


def study(a):
    store = a.store.resolve()
    history = a.history.resolve()
    roots = (store, history)
    state = store / "work/controller"
    state.mkdir(parents=True, exist_ok=True)
    release = bind(history, RELEASE, "payload/cpu/release.json")
    require(release["sha256"] == RELEASE_SHA, "release differs")
    if a.action == "freeze":
        keys = [
            "t0_authority.py",
            "t0_evaluate.py",
            "t0_records.py",
            "t0_dtu_roster.py",
            "t0_dtu_archive.py",
            "t0_worker_init.py",
            "t0_worker_pack.py",
            "t0_worker_run.py",
        ]
        files = {f"scripts/{n}": sha(PROJECT / "scripts" / n) for n in keys}
        spec = PROJECT / "protocol/t0-confirmation-v1.json"
        pins = {
            "roster": load(a.roster),
            "baseline": bind(history, RECOVERY, "payload/freeze.json"),
            "release": release,
        }
        worker = [
            "{python}",
            "{source}/scripts/t0_authority.py",
            "freeze",
            "--config",
            "{config}",
            "--protocol",
            "{input:protocol}",
            "--out",
            "{out}",
        ]
        for name in pins:
            worker += ["--" + name, "{input:" + name + "}"]
        return record(
            store,
            history,
            "freeze",
            "derivation",
            pins,
            {"new_scientific_files": files},
            worker,
            "payload/freeze.json",
            500000,
            {"protocol": spec},
        )
    freeze = load(state / "freeze/selected.json")
    fr = load(resolve(roots, freeze))
    if a.action == "score":
        require(a.object in fr["object_ids"] and a.condition in fr["conditions"], "unregistered cell")
        readback = load(a.readback)
        rp = resolve(roots, readback)
        receipt = load(rp)
        require(
            receipt["verified"]
            and receipt["object_id"] == a.object
            and receipt["condition"] == a.condition
            and receipt["freeze"] == freeze
            and not receipt["forced_fallback_fixture"]
            and receipt["target_reference_files_opened"] == 0
            and receipt["completed_steps"] == 5000,
            "unverified reconstruction or wrong exposure",
        )
        case = bind(store, readback["run_id"], "payload/case/case.json")
        cp = resolve(roots, case)
        require(case["sha256"] == receipt["case_sha256"], "case differs from its recovery verifier")
        release_path = resolve(roots, release)
        require(
            sha(PROJECT / "scripts/dtu_score.py") == fr["protocol"]["source_sha256"]["scripts/dtu_score.py"],
            "wrong scoring implementation",
        )
        worker = [
            str(a.cpu_python.absolute()),
            "-I",
            "-B",
            "{source}/scripts/dtu_score.py",
            "--case",
            str(cp.parent),
            "--case-sha256",
            case["sha256"],
            "--cpu-dir",
            str(release_path.parent),
            "--release-sha256",
            RELEASE_SHA,
            "--threads",
            "4",
            "--output",
            "{out}",
        ]
        return record(
            store,
            history,
            "score-" + a.object + "-" + a.condition,
            "model_score",
            {"case": case, "readback": readback, "freeze": freeze, "release": release},
            {
                "object_id": a.object,
                "condition": a.condition,
                "role": "frozen T0 scores; legal sources and prediction only",
            },
            worker,
            "payload/scores.json",
            12000000,
            retry=a.retry_of,
        )
    guarded = [
        "{python}",
        "{source}/scripts/t0_opening_software.py",
        a.action,
        "--config",
        "{config}",
        "--store",
        str(store),
        "--history",
        str(history),
        "--out",
        "{out}",
    ]
    if a.action == "lock":
        fixture = load(state / "evaluator-authored-fixture/selected.json")
        inputs = {"freeze": freeze, "authored_fixture": fixture}
        name = "opening-software"
        if a.recovery_audit:
            require(
                not (state / "collective-reference-opening/selected.json").exists(),
                "the reference gate has already opened",
            )
            inputs["prior_lock"] = load(state / "opening-software/selected.json")
            inputs["recovery_audit"] = load(a.recovery_audit)
            name = "opening-software-cross-store"
        return record(
            store,
            history,
            name,
            "archive_verify",
            inputs,
            {"role": "software closure before reference access"},
            guarded,
            "payload/software-lock.json",
            500000,
        )
    updated = state / "opening-software-cross-store/selected.json"
    software_lock = load(updated if updated.exists() else state / "opening-software/selected.json")
    if a.action == "open":
        req = load(a.matrix)
        require(req["freeze"] == freeze, "wrong matrix study")
        # Producer pins in the registered config make the authority part of the recorder lineage.
        pins = {"freeze": freeze, "software_lock": software_lock}
        causal = 0
        for i, cell in enumerate(req["cells"].values()):
            for n in ["case", "readback", "scores", "disposition"]:
                if n in cell:
                    pins[f"cell{i}-{n}"] = cell[n]
            if cell["status"] != "scored":
                d = load(resolve(roots, cell["disposition"]))
                pins["cause" + str(causal)] = d["causal_receipt"]
                causal += 1
        worker = guarded + ["--request", "{input:matrix}"]
        return record(
            store,
            history,
            "collective-reference-opening",
            "archive_verify",
            pins,
            {"role": "independent reference release"},
            worker,
            "payload/opening.json",
            400000,
            {"matrix": a.matrix},
            retry=a.retry_of,
        )
    opening = load(state / "collective-reference-opening/selected.json")
    req = load(a.references)
    require(
        req["opening"] == opening and set(req["references"]) == set(fr["object_ids"]),
        "not the complete reference list for the opened register",
    )
    pins = {"freeze": freeze, "software_lock": software_lock, "opening": opening, **req["references"]}
    worker = guarded + ["--request", "{input:references}"]
    return record(
        store,
        history,
        "evaluate",
        "evaluation",
        pins,
        {"role": "independent target evaluator"},
        worker,
        "payload/public/evaluation.json",
        100000000,
        {"references": a.references},
        retry=a.retry_of,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--history", type=Path, required=True)
    sub = p.add_subparsers(dest="action", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--roster", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--object", required=True)
    s.add_argument("--condition", choices=["M2", "M4"], required=True)
    s.add_argument("--readback", type=Path, required=True)
    s.add_argument("--cpu-python", type=Path, required=True)
    sub.add_parser("lock").add_argument("--recovery-audit", type=Path)
    o = sub.add_parser("open")
    o.add_argument("--matrix", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--references", type=Path, required=True)
    for action in [s, o, e]:
        action.add_argument("--retry-of")
    print(json.dumps(study(p.parse_args()), sort_keys=True))
