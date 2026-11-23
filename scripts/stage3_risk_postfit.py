"""Record native verification, paired development selection, original example and four-family confirmation."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from evidence_lens._bundle import load
from evidence_lens.formal_inputs import compact_pin, hash_json, require
from evidence_lens.multisource import METHODS
from evidence_lens.scoring_io import file_hash
from stage3_risk_control import chosen
from stage3_risk_sources import CATALOG, STUDY, TRANSFER
from vision_record import publish
from vision_sources import find_run, json_member, member, signed

SCIENTIFIC = {
    "selection": ("report", "selection.json", "stage3-development-paired-selection", 35_000_000),
    "development": ("evaluation", "evaluation.json", "stage3-risk-development-evaluation", 190_000_000),
    "example": ("derivation", "expected-example.json", "stage3-public-expected-example", 75_000_000),
    "parity": ("archive_verify", "parity.json", "stage3-native-risk-parity", 2_000_000),
    "pixel-replay": (
        "archive_verify",
        "pixel-replay.json",
        "stage3-independent-heldout-pixel-replay",
        2_000_000,
    ),
    "joint-report": ("report", "report.json", "stage3-joint-study-report", 25_000_000),
    "confirmation": ("evaluation", "evaluation.json", "stage3-risk-test-evaluation", 200_000_000),
}
for method in METHODS:
    SCIENTIFIC["native-" + method] = (
        "model_score",
        "native-scores.json",
        "stage3-native-example-" + method.lower(),
        35_000_000,
    )
for method in METHODS[2:]:
    SCIENTIFIC["test-" + method] = ("model_score", "scores.json", "stage3-test-" + method.lower(), 95_000_000)


def verified_selection(roots, state, name):
    raw = load(state / name / "selected.json")
    pin = compact_pin(raw)
    run = find_run(roots, pin)
    kind, n, exp, _ = SCIENTIFIC[name]
    require(
        run.run["execution_status"] == "succeeded"
        and run.run["run_kind"] == kind
        and run.run["experiment_id"] == exp,
        "unsuccessful or unrelated selected run",
    )
    path, _ = member(run, "payload/" + n, raw["receipt_sha256"])
    return pin, run, path


def formal_candidates(roots, path, sha, train_sha, dev_sha):
    from evidence_lens.stage3_decision import validate_fit_origin

    d = signed(path, sha)
    require(
        d["schema"] == "el-stage3-risk-candidate-roster-1"
        and d["catalog_sha256"] == CATALOG
        and len(d["candidates"]) == 4,
        "need all four signed formal candidates",
    )
    rows = {}
    for item in d["candidates"]:
        m, s = item["method"], item["seed"]
        key = m, s
        require(
            m in METHODS[2:] and s in (1729, 2718) and key not in rows,
            "unregistered or duplicate method/seed",
        )
        pin = compact_pin(item)
        r = find_run(roots, pin)
        j, _ = json_member(r, "payload/training.json")
        dev, _ = json_member(r, "payload/development.json")
        model, ms = json_member(r, "payload/package/model.json")
        score, ss = json_member(r, "payload/development-scores/scores.json")
        w = member(r, "payload/package/weights.safetensors", j["weights_sha256"])[1]
        validate_fit_origin(r, roots[0], roots[1])
        require(
            r.run["execution_status"] == "succeeded"
            and j["schema"] == "el-multisource-risk-training-1"
            and j["phase"] == "formal"
            and j["status"] == "completed"
            and j["method"] == m
            and j["seed"] == s
            and j["optimizer_steps"] > 0
            and j["run_id"] == model["model_id"] == score["model_id"] == pin["run_id"]
            and j["training_dataset_sha256"] == train_sha
            and j["development_dataset_sha256"] == dev_sha
            and j["weights_sha256"]
            == w["sha256"]
            == model["files"]["weights.safetensors"]["sha256"]
            == score["weights_sha256"]
            and j["model_index_sha256"] == ms["sha256"] == score["model_index_sha256"]
            and j["threshold"] == model["threshold"] == score["threshold"] == dev["threshold"]["tau"]
            and j["best_development_family_equal_ap"] == dev["family_equal_ap"],
            "scientific candidate source or output differs",
        )
        rows[key] = (
            pin,
            r,
            {
                "directory": str(r.root / "payload/development-scores"),
                "sha256": ss["sha256"],
                "method": m,
                "model_id": pin["run_id"],
                "model_index_sha256": ms["sha256"],
                "weights_sha256": w["sha256"],
                "threshold": model["threshold"],
                "training_run": pin,
                "reported_development": {
                    "path": str(r.root / "payload/development.json"),
                    "sha256": r.files["payload/development.json"]["sha256"],
                },
            },
        )
    return rows


class Post:
    def __init__(self, project, store, shelf, candidates, candidates_sha, python, *, baselines_only=False):
        self.project = project
        self.roots = (store, shelf)
        self.base = store / "stage3" / CATALOG / "risk"
        self.state = store / "stage3" / CATALOG / "risk-postfit"
        self.cpu = python
        self.sourcepin, self.dsrun, self.dev = chosen(self.roots, self.base, "data-development")
        _, _, self.train = chosen(self.roots, self.base, "data-train")
        self.basepin, self.br, self.brec = chosen(self.roots, self.base, "development")
        self.data_sha = file_hash(self.dev)
        self.baseline = load(self.brec)
        self.candidates = candidates
        self.candidate_sha = candidates_sha
        self.cand = (
            {}
            if baselines_only
            else formal_candidates(
                self.roots, candidates, candidates_sha, file_hash(self.train), self.data_sha
            )
        )
        self.rosters = {}
        for seed in (1729, 2718):
            ro = {}
            for m in METHODS:
                if m in METHODS[2:]:
                    if baselines_only:
                        continue
                    ro[m] = self.cand[m, seed][2]
                    continue
                spec = self.baseline["scores"][m]
                path, _ = member(self.br, "payload/" + spec["path"], spec)
                j = load(path)
                ro[m] = {
                    **{
                        k: j[k]
                        for k in ("method", "model_id", "model_index_sha256", "weights_sha256", "threshold")
                    },
                    "directory": str(path.parent),
                    "sha256": spec["sha256"],
                }
            self.rosters[str(seed)] = ro

    def package(self, m):
        if m == "B-L1-transfer":
            r = find_run(self.roots, TRANSFER)
            return TRANSFER, r.open("payload/model/model.json").parent
        if m == "MS-L1":
            return self.basepin, self.br.open("payload/package/model.json").parent
        pin, r, _ = self.cand[m, 1729]
        return pin, r.open("payload/package/model.json").parent

    def record(self, name, parents, roles, args, exe=None, retry=None):
        folder = self.state / name
        folder.mkdir(parents=True, exist_ok=True)
        if not exe:
            exe = "{python}"
        kind, receipt, experiment, est = SCIENTIFIC[name]
        inputs = {
            key: {
                "path_role_name": Path(path).name,
                "parent": parent["run_id"] if parent else None,
                "sha256": file_hash(path) if path.is_file() else None,
            }
            for key, (path, parent) in roles.items()
        }
        # Directory roles are sealed model or case directories; their index is a separately authenticated parent member.
        parent_map = {p["run_id"]: p["manifest_sha256"] for p in parents}
        request = hash_json(
            {"action": name, "inputs": inputs, "parents": parent_map, "argv": args, "executable": exe}
        )
        cfg = {
            "project_id": "evidence_lens",
            "dataset_hash": CATALOG,
            "study_sha256": STUDY,
            "experiment_id": experiment,
            "trial_id": experiment,
            "request_sha256": request,
            "roles": inputs,
            "recording": {"expected_output_bytes": est},
        }
        with (folder / "lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            publish(folder / "config.json", cfg)

            def resolve(result):
                p = compact_pin(result)
                r = find_run(self.roots, p)
                require(
                    r.run["execution_status"] == "succeeded"
                    and r.run["experiment_id"] == experiment
                    and load(r.open("config.resolved.yaml"))["request_sha256"] == request,
                    "wrong or failed post-fit stage",
                )
                rec = r.open("payload/" + receipt)
                selected = {
                    **p,
                    "receipt": "payload/" + receipt,
                    "receipt_sha256": file_hash(rec),
                    "request_sha256": request,
                }
                publish(folder / "selected.json", selected)
                return selected

            if (folder / "selected.json").exists():
                return resolve(load(folder / "selected.json"))
            logs = sorted(folder.glob("attempt-*.log"))
            if logs:
                try:
                    prev = json.loads(logs[-1].read_text().strip().splitlines()[-1])
                except (ValueError, IndexError) as e:
                    raise ValueError("unsealed previous attempt: " + str(logs[-1])) from e
                if prev["execution_status"] == "succeeded":
                    return resolve(prev)
                require(
                    retry == prev["run_id"], "explicit retry ID required for failed attempt " + prev["run_id"]
                )
            cmd = [
                sys.executable,
                "-m",
                "evidence_lens.recorder",
                "--root",
                str(self.roots[0]),
                "--source-root",
                str(self.project),
                "--storage-policy",
                str(self.project / "configs/storage.json"),
                "--kind",
                kind,
                "--config",
                str(folder / "config.json"),
            ]
            for rid, sha in sorted(parent_map.items()):
                cmd += ["--parent", rid + "=" + sha]
            for role, (p, parent) in roles.items():
                cmd += ["--input", role + "=" + str(p)]
                if parent:
                    cmd += ["--input-parent", role + "=" + parent["run_id"]]
            if retry:
                cmd += ["--retry-of", retry]
            logfile = folder / ("attempt-%02d.log" % (len(logs) + 1))
            with logfile.open("xb") as log:
                res = subprocess.run(
                    [*cmd, "--", exe, "{source}/scripts/" + args[0], *args[1:], "--output", "{out}"],
                    cwd=self.project,
                    env={
                        **os.environ,
                        "PYTHONPATH": str(self.project / "src"),
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                    stdout=log,
                    stderr=log,
                    pass_fds=(lock.fileno(),),
                )
            raw = logfile.read_text().strip()
            require(res.returncode == 0, "post-fit run failed: " + str(logfile) + " " + raw[-400:])
            return resolve(json.loads(raw.splitlines()[-1]))

    def joint_report(self, opening, evidence, evidence_sha, retry):
        from evidence_lens.stage3_gate import get_frozen_method, require_opening

        require(
            opening and evidence and evidence_sha,
            "joint report requires the opening and original extra evidence",
        )
        extra = signed(evidence, evidence_sha)
        require(
            extra["schema"] == "el-stage3-joint-report-extra-1"
            and extra["catalog_sha256"] == CATALOG
            and set(extra["sources"]) == {"input_qa", "acquisition_costs", "acquisition_quality"},
            "unexpected study supplements",
        )
        store, shelf = self.roots
        mk = require_opening(store, CATALOG, opening_sha256=opening, shelf=shelf)
        method = get_frozen_method(store, mk, shelf)
        sources, roles, parents = {}, {}, [mk["method"]]

        def add(key, pin, run, path):
            name = path.relative_to(run.root).as_posix()
            sources[key] = {**compact_pin(pin), "member": name, "sha256": file_hash(path)}
            roles[key] = (path, pin)
            parents.append(pin)

        for key, action in (
            ("train_dataset", "data-train"),
            ("development_dataset", "data-development"),
            ("test_dataset", "data-test"),
            ("fit", "fit"),
            ("baseline_development", "development"),
            ("test_baselines", "score-test"),
        ):
            add(key, *chosen(self.roots, self.base, action))
        for key in ("selection", "development", "example", "parity", "confirmation", "pixel-replay"):
            add(key, *verified_selection(self.roots, self.state, key))
        for m in METHODS:
            add("native_" + m, *verified_selection(self.roots, self.state, "native-" + m))
            if m in METHODS[2:]:
                add("test_" + m, *verified_selection(self.roots, self.state, "test-" + m))
        for (m, seed), (p, r, _) in self.cand.items():
            add("candidate_" + m + "_" + str(seed), p, r, r.open("payload/training.json"))
        for key, spec in extra["sources"].items():
            p = compact_pin(spec)
            r = find_run(self.roots, p)
            path, _ = member(r, spec["member"], spec["sha256"])
            add(key, p, r, path)
        for key in ("tensor_comparison", "cpu_validation", "production_budget"):
            sc = method["components"][key]
            p = compact_pin(sc)
            r = find_run(self.roots, p)
            roles[key] = (member(r, sc["member"], sc["sha256"])[0], p)
            parents.append(p)
        sources["freeze"] = {
            **mk["method"],
            "member": "payload/method-freeze.json",
            "sha256": mk["method_sha256"],
        }
        roles["freeze"] = (
            find_run(self.roots, mk["method"]).open("payload/method-freeze.json"),
            mk["method"],
        )
        request = {
            "schema": "el-stage3-joint-report-request-1",
            "catalog_sha256": CATALOG,
            "opening_sha256": opening,
            "additional_evidence_sha256": evidence_sha,
            "sources": sources,
        }
        register = self.state / "joint-report" / "originals.json"
        publish(register, request)
        roles["request"] = (register, None)
        args = [
            "stage3_risk_joint.py",
            "--store",
            str(store),
            "--shelf",
            str(shelf),
            "--opening-sha256",
            opening,
            "--request",
            "{input:request}",
            "--request-sha256",
            file_hash(register),
        ]
        return self.record("joint-report", parents, roles, args, retry=retry)

    def run(self, action, opening=None, retry=None, joint_evidence=None, joint_evidence_sha=None):
        require(
            self.cand or action in {"native-B-L1-transfer", "native-MS-L1"},
            "four original paired candidates required for this action",
        )
        store, shelf = self.roots
        folder = self.state / action
        folder.mkdir(parents=True, exist_ok=True)
        if action == "joint-report":
            return self.joint_report(opening, joint_evidence, joint_evidence_sha, retry)
        if action.startswith("native-"):
            m = action[len("native-") :]
            d = load(self.dev)
            f = d["families"][0]
            require(f.startswith("scene_057_"), "wrong native example")
            parent, package = self.package(m)
            r = self.rosters["1729"][m]
            roles = {"model": (package, parent)}
            args = [
                "stage3_risk_native.py",
                "--package",
                "{input:model}",
                "--python",
                str(self.cpu),
                "--method",
                m,
                "--model-id",
                r["model_id"],
                "--model-sha256",
                r["model_index_sha256"],
                "--threshold",
                str(r["threshold"]),
                "--weights-sha256",
                r["weights_sha256"],
                "--family",
                f,
            ]
            for c in ("M2", "M4"):
                cell = next(x for x in d["cells"] if x["family"] == f and x["condition"] == c)
                p = self.dsrun.open("payload/" + cell["case"]["path"])
                require(file_hash(p) == cell["case_sha256"], "wrong native case")
                roles["case_" + c] = (p.parent, self.sourcepin)
                args += [
                    "--case-" + c.lower(),
                    "{input:case_" + c + "}",
                    "--case-" + c.lower() + "-sha256",
                    cell["case_sha256"],
                ]
            return self.record(action, [parent, self.sourcepin], roles, args, retry=retry)
        if action.startswith("test-"):
            m = action[len("test-") :]
            from evidence_lens.stage3_gate import get_frozen_method, require_opening

            require(opening is not None, "explicit opening SHA required")
            mk = require_opening(store, CATALOG, opening_sha256=opening, shelf=shelf)
            method = get_frozen_method(store, mk, shelf)
            p, run, ds = chosen(self.roots, self.base, "data-test")
            modelp, package = self.package(m)
            require(
                method["models"][m]["model_index_sha256"] == self.rosters["1729"][m]["model_index_sha256"],
                "unexpected held-out model",
            )
            roles = {
                "model_index": (package / "model.json", modelp),
                "test_dataset": (ds, p),
                "catalog": (self.project / "protocol/stage3-families-v1.json", None),
            }
            args = [
                "multi_risk_worker.py",
                "infer-dataset",
                "--model",
                str(package),
                "--dataset",
                str(ds),
                "--dataset-sha256",
                file_hash(ds),
                "--model-sha256",
                file_hash(package / "model.json"),
                "--catalog",
                "{input:catalog}",
                "--method",
                m,
                "--store",
                str(store),
                "--shelf",
                str(shelf),
                "--split",
                "test",
                "--opening-sha256",
                opening,
                "--threads",
                "4",
            ]
            return self.record(action, [modelp, p, mk["method"]], roles, args, exe=str(self.cpu), retry=retry)
        if action == "pixel-replay":
            from evidence_lens.stage3_gate import require_opening

            require(opening, "explicit held-out opening digest required")
            marker = require_opening(store, CATALOG, opening_sha256=opening, shelf=shelf)
            ep, er, erec = verified_selection(self.roots, self.state, "confirmation")
            dp, dr, drec = chosen(self.roots, self.base, "data-test")
            bp, br, _ = chosen(self.roots, self.base, "score-test")

            def spec(p, r, path):
                return {
                    **compact_pin(p),
                    "member": path.relative_to(r.root).as_posix(),
                    "sha256": file_hash(path),
                }

            req = {
                "schema": "el-stage3-confirmation-replay-request-1",
                "opening_sha256": opening,
                "dataset": spec(dp, dr, drec),
                "evaluation": spec(ep, er, erec),
                "scores": {},
            }
            roles = {"dataset": (drec, dp), "evaluation": (erec, ep)}
            parents = [ep, dp, bp, marker["method"]]
            for m in METHODS:
                if m in METHODS[:2]:
                    p, r, path = bp, br, br.open("payload/" + m + "/scores.json")
                else:
                    p, r, path = verified_selection(self.roots, self.state, "test-" + m)
                    parents.append(p)
                roles["scores_" + m] = (path, p)
                req["scores"][m] = spec(p, r, path)
            request = folder / "pixel-originals.json"
            publish(request, req)
            roles["request"] = (request, None)
            argv = [
                "stage3_risk_replay.py",
                "--store",
                str(store),
                "--shelf",
                str(shelf),
                "--opening-sha256",
                opening,
                "--request",
                "{input:request}",
                "--request-sha256",
                file_hash(request),
            ]
            return self.record(action, parents, roles, argv, retry=retry)
        if action == "confirmation":
            from evidence_lens.stage3_gate import require_opening

            require(opening is not None, "explicit opening SHA required")
            mk = require_opening(store, CATALOG, opening_sha256=opening, shelf=shelf)
            dspin, _, ds = chosen(self.roots, self.base, "data-test")
            bpin, br, brec = chosen(self.roots, self.base, "score-test")
            rr = {}
            roles = {"dataset": (ds, dspin)}
            parents = [dspin, bpin, mk["method"]]
            for m in METHODS:
                if m in METHODS[:2]:
                    p = bpin
                    path = br.open("payload/" + m + "/scores.json")
                else:
                    p, _, path = verified_selection(self.roots, self.state, "test-" + m)
                    parents.append(p)
                j = load(path)
                rr[m] = {
                    **{k: j[k] for k in ("model_id", "model_index_sha256", "weights_sha256", "threshold")},
                    "sha256": file_hash(path),
                    "directory": str(path.parent),
                }
                roles["scores_" + m] = (path, p)
            register = folder / "test-scores.json"
            publish(register, rr)
            roles["roster"] = (register, None)
            args = [
                "stage3_risk_evaluate.py",
                "--dataset",
                str(ds.parent),
                "--dataset-sha256",
                file_hash(ds),
                "--roster",
                "{input:roster}",
                "--roster-sha256",
                file_hash(register),
                "--store",
                str(store),
                "--shelf",
                str(shelf),
                "--split",
                "test",
                "--opening-sha256",
                opening,
            ]
            return self.record(action, parents, roles, args, retry=retry)
        rr = self.rosters if action == "selection" else self.rosters["1729"]
        registry = folder / "original-scores.json"
        publish(registry, rr)
        parents = [self.sourcepin, self.basepin]
        roles = {"dataset": (self.dev, self.sourcepin), "roster": (registry, None)}
        for (m, s), (p, r, _) in self.cand.items():
            if action == "selection" or s == 1729:
                parents.append(p)
                roles["scores_" + m + "_" + str(s)] = (r.open("payload/development-scores/scores.json"), p)
        for m in METHODS[:2]:
            roles["scores_" + m] = (self.br.open("payload/" + m + "/scores.json"), self.basepin)
        baseargs = [
            "--dataset",
            str(self.dev.parent),
            "--dataset-sha256",
            self.data_sha,
            "--roster",
            "{input:roster}",
            "--roster-sha256",
            file_hash(registry),
        ]
        if action == "selection":
            roles["study"] = (self.project / "protocol/stage3-study-v1.json", None)
            args = ["stage3_risk_selection.py", *baseargs, "--study", "{input:study}"]
        elif action == "development":
            roles["paired_selection"] = (
                verified_selection(self.roots, self.state, "selection")[2],
                verified_selection(self.roots, self.state, "selection")[0],
            )
            parents.append(roles["paired_selection"][1])
            args = ["stage3_risk_evaluate.py", *baseargs, "--split", "development"]
        elif action == "example":
            args = ["stage3_risk_example.py", *baseargs, "--store", str(store), "--shelf", str(shelf)]
            f = load(self.dev)["families"][0]
            cell = next(k for k in load(self.dev)["cells"] if k["family"] == f)
            parents.append(cell["selected"]["reference"])
        elif action == "parity":
            cpu = {}
            for m in METHODS:
                p, _, rec = verified_selection(self.roots, self.state, "native-" + m)
                parents.append(p)
                roles["native_" + m] = (rec, p)
                cpu[m] = {"directory": str(rec.parent), "sha256": file_hash(rec)}
            cp = folder / "cpu-roster.json"
            publish(cp, cpu)
            roles["cpu"] = (cp, None)
            roles["protocol"] = (self.project / "protocol/stage3-native-risk-parity-v1.json", None)
            args = [
                "stage3_risk_parity.py",
                "--dataset",
                str(self.dev.parent),
                "--dataset-sha256",
                self.data_sha,
                "--producer",
                "{input:roster}",
                "--producer-sha256",
                file_hash(registry),
                "--cpu",
                "{input:cpu}",
                "--cpu-sha256",
                file_hash(cp),
                "--protocol",
                "{input:protocol}",
            ]
        else:
            raise ValueError("unknown scientific action")
        return self.record(action, parents, roles, args, retry=retry)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=SCIENTIFIC)
    for k in ("project", "store", "cpu-python"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--candidates", type=Path)
    p.add_argument("--candidates-sha256")
    p.add_argument("--shelf", type=Path)
    p.add_argument("--opening-sha256")
    p.add_argument("--retry-of")
    p.add_argument("--joint-evidence", type=Path)
    p.add_argument("--joint-evidence-sha256")
    a = p.parse_args()
    s = a.store.resolve()
    baselines_only = a.action in {"native-B-L1-transfer", "native-MS-L1"} and not (
        a.candidates or a.candidates_sha256
    )
    require(
        baselines_only or (a.candidates and a.candidates_sha256),
        "explicit paired candidate roster and digest required",
    )
    post = Post(
        a.project.resolve(),
        s,
        (a.shelf or s.parent / "cold-shelf").resolve(),
        a.candidates.resolve() if a.candidates else None,
        a.candidates_sha256,
        a.cpu_python.absolute(),
        baselines_only=baselines_only,
    )
    print(
        json.dumps(
            post.run(a.action, a.opening_sha256, a.retry_of, a.joint_evidence, a.joint_evidence_sha256)
        )
    )
