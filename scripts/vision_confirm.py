"""Record the opened new confirmation, separate historical regression, private case panels and numeric report."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from evidence_lens._bundle import load
from evidence_lens.formal_inputs import compact_pin, hash_json, require
from evidence_lens.scoring_io import file_hash
from vision_confirmation import COHORTS, PRIOR_REPORT, authenticated, catalog_grid, successful
from vision_evaluate import SCORE_SETS, TWO
from vision_pipeline import DEFAULT_STUDY_SHA
from vision_record import parse_pin, publish
from vision_sources import find_run, signed


def observed_failure_evidence(roots, pins, folder):
    """Embed original failed manifests as typed observations, keeping recorder success-parent semantics."""
    from hashlib import sha256

    records = {}
    for source in pins:
        pin = compact_pin(source)
        run = find_run(roots, pin)
        status = run.run["execution_status"]
        require(
            status in {"failed", "interrupted", "cancelled"}, "observation must identify an unsuccessful run"
        )
        raw = (run.root / "manifest.json").read_bytes()
        require(sha256(raw).hexdigest() == pin["manifest_sha256"], "observed original manifest differs")
        row = {
            **pin,
            "execution_status": status,
            "source_member": "manifest.json",
            "original_manifest_utf8": raw.decode("utf-8"),
        }
        previous = records.setdefault(pin["run_id"], row)
        require(previous == row, "conflicting observed failure identities")
    path = folder / "observed-failures.json"
    publish(
        path,
        {"schema": "el-vision-observed-failures-1", "observations": [records[k] for k in sorted(records)]},
    )
    return path


def report_sources(roots, pins):
    parents, observed = [], []
    for p in pins:
        verified = find_run(roots, p)
        (parents if verified.run["execution_status"] == "succeeded" else observed).append(compact_pin(p))
    return parents, observed


def record(
    project,
    roots,
    state,
    stage,
    kind,
    experiment,
    roster,
    parents,
    receipt,
    expected,
    interpreter,
    retries,
    registry_path=None,
):
    store, shelf = roots
    folder = state / stage
    folder.mkdir(parents=True, exist_ok=True)
    evidence = None
    if roster.get("observed_failures"):
        require(
            roster["action"] in ("materials", "report"),
            "unexpected failed observations on a computational stage",
        )
        evidence = observed_failure_evidence(roots, roster["observed_failures"], folder)
        roster = {**roster, "observed_failure_evidence_sha256": file_hash(evidence)}
    inputs = folder / "input-roster.json"
    publish(inputs, roster)
    request = {
        "schema": "el-vision-confirmation-stage-config-1",
        "experiment_id": experiment,
        "trial_id": "v11-" + stage,
        "dataset_hash": roster["catalog_sha256"],
        "input_roster_sha256": file_hash(inputs),
        "parents": sorted({p["run_id"]: compact_pin(p) for p in parents}.values(), key=lambda p: p["run_id"]),
        "recording": {"expected_output_bytes": expected},
    }
    label = receipt
    selected = folder / "selected.json"
    conf = folder / "config.json"
    publish(conf, request)

    def validate(pin):
        run = find_run(roots, compact_pin(pin))
        require(
            run.run["run_kind"] == kind
            and run.run["experiment_id"] == experiment
            and run.run["trial_id"] == request["trial_id"]
            and load(run.open("config.resolved.yaml"))["input_roster_sha256"]
            == request["input_roster_sha256"],
            "registered output belongs to another computation",
        )
        if evidence is not None:
            info = load(run.open("inputs.json"))
            files = [r for r in info["inputs"] if r["role"] == "observed_failures"]
            require(
                len(files) == 1
                and files[0]["kind"] == "file"
                and files[0]["storage"] == "embedded"
                and files[0]["parent_run_id"] is None
                and file_hash(run.open(files[0]["path"])) == roster["observed_failure_evidence_sha256"]
                and not {r["run_id"] for r in roster["observed_failures"]}.intersection(
                    p["run_id"] for p in info["parents"]
                ),
                "sealed failed observations or scientific parent roles differ",
            )
        p, sha = successful(roots, pin, label)
        return {
            **compact_pin(pin),
            "receipt": label,
            "receipt_sha256": sha,
            "request_sha256": hash_json(request),
        }

    with (folder / "lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if selected.is_file():
            previous = load(selected)
            require(previous == validate(previous), "existing result changed")
            return previous
        attempts = sorted(folder.glob("attempt-*.log"))
        prev = None
        if attempts:
            last = attempts[-1]
            try:
                prev = json.loads(last.read_text().strip().splitlines()[-1])
            except (ValueError, IndexError) as exc:
                raise ValueError("no recorder result in " + str(last)) from exc
            if prev["execution_status"] == "succeeded":
                result = validate(prev)
                publish(selected, result)
                return result
            require(
                retries.get(stage) == prev["run_id"],
                "sealed previous failure requires --retry " + stage + "=" + prev["run_id"],
            )
        cmd = [
            sys.executable,
            "-m",
            "evidence_lens.recorder",
            "--root",
            str(store),
            "--source-root",
            str(project),
            "--config",
            str(conf),
            "--kind",
            kind,
            "--input",
            "roster=" + str(inputs),
            "--input",
            "catalog=" + str(project / "protocol/stage2-families-v1.json"),
            "--input",
            "original_catalog=" + str(project / "protocol/families-v1.json"),
        ]
        if registry_path:
            cmd += ["--input", "approved_registry=" + str(registry_path)]
        if evidence is not None:
            cmd += ["--input", "observed_failures=" + str(evidence)]
        for p in request["parents"]:
            cmd += ["--parent", p["run_id"] + "=" + p["manifest_sha256"]]
        if prev:
            cmd += ["--retry-of", prev["run_id"]]
        cmd += [
            "--",
            str(interpreter) if interpreter else "{python}",
            "{source}/scripts/vision_confirmation.py",
            roster["action"],
            "--roster",
            "{input:roster}",
            "--sha256",
            request["input_roster_sha256"],
            "--catalog",
            "{input:catalog}",
            "--original-catalog",
            "{input:original_catalog}",
            "--store",
            str(store),
            "--shelf",
            str(shelf),
            "--run-id",
            "{run_id}",
            "--output",
            "{out}",
        ]
        if registry_path:
            cmd += ["--selection-registry", "{input:approved_registry}"]
        log = folder / f"attempt-{len(attempts) + 1:02d}.log"
        with log.open("xb") as stream:
            r = subprocess.run(
                cmd,
                cwd=project,
                env={**os.environ, "PYTHONPATH": str(project / "src")},
                stdout=stream,
                stderr=stream,
                pass_fds=(lock.fileno(),),
            )
        try:
            actual = json.loads(log.read_text().strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise ValueError("no recorder result in " + str(log)) from exc
        publish(log.with_suffix(".json"), actual)
        require(
            r.returncode == 0 and actual["execution_status"] == "succeeded",
            "recorded failed stage " + stage + " " + actual["run_id"],
        )
        result = validate(actual)
        publish(selected, result)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("new", "historical", "report"))
    p.add_argument("--project", type=Path, default=Path.cwd())
    p.add_argument("--store", type=Path, default=Path(".research"))
    p.add_argument("--shelf", type=Path)
    p.add_argument("--study-sha", default=DEFAULT_STUDY_SHA)
    p.add_argument("--method", required=True, help="frozen run=manifest")
    p.add_argument("--method-sha256", required=True)
    p.add_argument("--vision-python", type=Path)
    p.add_argument("--selection-registry", nargs=2, metavar=("APPROVED_REGISTRY", "SHA256"))
    p.add_argument("--retry", action="append", default=[], help="stage=failed run")
    p.add_argument(
        "--engineering",
        action="append",
        default=[],
        help="explicit additional engineering incident run=manifest for report",
    )
    a = p.parse_args()
    project = a.project.resolve()
    store = a.store.resolve()
    roots = (store, (a.shelf or store.parent / "cold-shelf").resolve())
    new = project / "protocol/stage2-families-v1.json"
    study = signed(project / "protocol/stage2-training-v1.json", a.study_sha)
    sha = study["catalog_sha256"]
    newcat = signed(new, sha)
    marker, method, mp, msha = authenticated(roots, new, sha)
    require(
        mp == parse_pin(a.method) and msha == a.method_sha256 and a.study_sha == method["study_sha256"],
        "joint freeze pin differs",
    )
    state = store / "stage2" / sha / "confirmation" / msha
    learning = store / "stage2" / sha / "learning"
    replay = load(learning / "postfit-dev-seven-thresholds/selected.json")
    require(
        successful(roots, replay, "thresholds.json")[1] == method["thresholds_sha256"],
        "missing original threshold run",
    )
    retries = dict(s.split("=", 1) for s in a.retry)
    common = {
        "schema": "el-vision-confirmation-request-1",
        "catalog_sha256": sha,
        "method": mp,
        "method_sha256": msha,
        "opening_sha256": marker,
    }

    def call(stage, kind, cohort, action, extra, parents, receipt, limit, vision=False, observations=()):
        spec = {**common, "cohort": cohort, "action": action, **extra}
        if observations:
            spec["observed_failures"] = observations
        it = None
        if vision:
            it = a.vision_python or project / ".venv-vision-cpu/bin/python"
            require(it.is_file(), "supply the CPU vision Python interpreter")
        return record(
            project,
            roots,
            state,
            stage,
            kind,
            "stage2-vision-confirmation-report" if action == "report" else "stage2-vision-confirmation",
            spec,
            [mp, *parents],
            receipt,
            limit,
            it,
            retries,
            Path(a.selection_registry[0]).resolve() if "acquisition_registry_sha256" in spec else None,
        )

    if a.action == "report":
        allroles = {}
        parents = []
        for c in COHORTS:

            def selected(s):
                return compact_pin(load(state / (c + "-" + s) / "selected.json"))

            rr = {n: selected(n) for n in ("materials", "dataset", "evaluation", "boards")}
            rr["scores"] = {tag: selected("score-" + tag) for tag in SCORE_SETS}
            parents.extend([*(p for k, p in rr.items() if k != "scores"), *rr["scores"].values()])
            allroles[c] = rr
        eng = [parse_pin(s) for s in a.engineering]
        engineering_parents, observed = report_sources(roots, eng)
        result = call(
            "joint-public-report",
            "report",
            "both",
            "report",
            {"cohorts": allroles, "engineering": eng},
            [*parents, *engineering_parents],
            "public-numerical/manifest.json",
            20_000_000,
            observations=observed,
        )
    else:
        c = COHORTS[0] if a.action == "new" else COHORTS[1]
        extra = {}
        parents = []
        observed = []
        if c == COHORTS[0]:
            grid = catalog_grid(newcat)
            trials = {}
            paths = [
                (cond, row, roots[0] / "trials" / row["trials"][cond] / "trial_selection.json")
                for (f, cond), row in grid.items()
            ]
            missing = [r["trials"][cond] for cond, r, path in paths if not path.is_file()]
            if missing:
                print(
                    json.dumps(
                        {
                            "status": "awaiting_terminal_trial_selections",
                            "cohort": c,
                            "missing_trials": missing,
                        },
                        indent=2,
                    )
                )
                return
            for cond, row, path in paths:
                sha_sel = file_hash(path)
                s = signed(path, sha_sel)
                require(
                    s["condition_id"] == cond
                    and s["scene_family_id"] == row["scene_family_id"]
                    and s["trial_id"] == row["trials"][cond]
                    and s["selection_status"] in ("selected", "failed"),
                    "incorrect condition index",
                )
                trials[s["trial_id"]] = {"selection_sha256": sha_sel, "status": s["selection_status"]}
                if s["selection_status"] == "failed":
                    parents.append(compact_pin(s["reference"]))
                    observed.append(compact_pin(s["failure"]))
                else:
                    parents.append(compact_pin(s["selected"]["evaluation"]))
            require(
                a.selection_registry,
                "opened acquisition requires the independently frozen 24-trial registry and SHA",
            )
            approved = signed(*a.selection_registry)
            require(
                approved["catalog_sha256"] == sha
                and approved["split"] == "test"
                and approved["schema"] == "el-vision-selected-trials-1"
                and set(approved["trials"]) == set(trials)
                and all(
                    rec["selection_sha256"] == approved["trials"][t]["selection_sha256"]
                    and rec["status"] == approved["trials"][t]["status"]
                    for t, rec in trials.items()
                ),
                "acquisition registry and original 24 terminal selections differ",
            )
            extra["trials"] = trials
            extra["acquisition_registry_sha256"] = a.selection_registry[1]
        else:
            parents = [PRIOR_REPORT]
        mat = call(
            c + "-materials",
            "dataset_build",
            c,
            "materials",
            extra,
            parents,
            "materials.json",
            6_000_000,
            observations=observed,
        )
        data = call(
            c + "-dataset",
            "dataset_build",
            c,
            "transport",
            {"materials": compact_pin(mat)},
            [mat],
            "dataset.json",
            (3_550_000_000 if c == COHORTS[0] else 1_850_000_000),
        )
        score = {}
        for tag in SCORE_SETS:
            mod = method["models"]["L1-full" if tag == "original-five" else tag]
            score[tag] = call(
                c + "-score-" + tag,
                "model_score",
                c,
                "score",
                {"dataset": compact_pin(data), "score_set": tag},
                [data, mod],
                "scores.json",
                450_000_000,
                tag in TWO,
            )
        source = {"dataset": compact_pin(data), "scores": {k: compact_pin(v) for k, v in score.items()}}
        ep = call(
            c + "-evaluation",
            "evaluation",
            c,
            "evaluate",
            source | {"thresholds": compact_pin(replay)},
            [data, replay, *score.values()],
            "evaluation.json",
            90_000_000,
        )
        bp = call(
            c + "-boards",
            "report",
            c,
            "boards",
            source | {"materials": compact_pin(mat)},
            [data, mat, *score.values(), ep],
            "boards.json",
            130_000_000,
        )
        result = {"cohort": c, "evaluation": ep, "boards": bp}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
