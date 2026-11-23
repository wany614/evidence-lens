"""Record the D1 release restore, source-only scoring and independent evaluation; no directory discovery."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from evidence_lens._bundle import load
from evidence_lens.formal_inputs import compact_pin, require
from evidence_lens.scoring_io import file_hash
from vision_record import parse_pin, publish
from vision_sources import find_run, member
from dtu_score import RELEASE_SHA

BUILD = (
    "20261123T173212401848Z-7045443abe424128=7af2c73e9fad9a0a85ba09c1789feef7380fc05aa39c6439676f9de431eb94f3"
)
V = "v0.3.0-rc.1"


class Recorder:
    def __init__(self, project, store, shelf, state):
        self.project = Path(project).resolve()
        self.store = Path(store).resolve()
        self.shelf = Path(shelf).resolve()
        self.roots = self.store, self.shelf
        self.state = Path(state).resolve()

    def original(self, encoded, name):
        pin = parse_pin(encoded) if isinstance(encoded, str) else compact_pin(encoded)
        r = find_run(self.roots, pin)
        require(r.run["execution_status"] == "succeeded", "failed selected original")
        path, spec = member(r, name)
        return pin, path, spec

    def stage(self, name, kind, parents, roles, argv, receipt, estimate, fields, retry=None):
        folder = self.state / name
        folder.mkdir(parents=True, exist_ok=True)
        parents = {p["run_id"]: p["manifest_sha256"] for p in parents}
        cfg = {
            "experiment_id": "d1-" + name,
            "trial_id": "d1-" + name,
            "recording": {"expected_output_bytes": estimate},
            "stage_role": name,
            "upstream": parents,
            "input_sha256": {k: file_hash(p) for k, (p, _) in roles.items()},
            **fields,
        }
        with (folder / "lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            publish(folder / "config.json", cfg)

            def accept(row):
                p = compact_pin(row)
                r = find_run(self.roots, p)
                require(
                    r.run["execution_status"] == "succeeded" and r.run["trial_id"] == "d1-" + name,
                    "unexpected recorded output",
                )
                n = "payload/" + receipt
                item = r.open(n)
                require(
                    load(r.open("config.resolved.yaml"))["input_sha256"] == cfg["input_sha256"],
                    "original input hashes differ",
                )
                result = {**p, "member": n, "sha256": file_hash(item)}
                publish(folder / "selected.json", result)
                return result

            if (folder / "selected.json").is_file():
                return accept(load(folder / "selected.json"))
            logs = sorted(folder.glob("attempt-*.jsonl"))
            if logs:
                previous = json.loads(logs[-1].read_text().strip().splitlines()[-1])
                if previous["execution_status"] == "succeeded":
                    return accept(previous)
                require(
                    retry == previous["run_id"] and len(logs) == 1,
                    "one explicit retry of original failure is allowed",
                )
            else:
                require(retry is None, "no original recorded failure for retry")
            cmd = [
                sys.executable,
                "-m",
                "evidence_lens.recorder",
                "--root",
                str(self.store),
                "--source-root",
                str(self.project),
                "--kind",
                kind,
                "--config",
                str(folder / "config.json"),
                "--storage-policy",
                str(self.project / "configs/storage.json"),
            ]
            for rid, sha in sorted(parents.items()):
                cmd += ["--parent", rid + "=" + sha]
            for k, (path, p) in sorted(roles.items()):
                cmd += ["--input", k + "=" + str(path)]
                if p:
                    cmd += ["--input-parent", k + "=" + p["run_id"]]
            if retry:
                cmd += ["--retry-of", retry]
            out = folder / ("attempt-%02d.jsonl" % (len(logs) + 1))
            with out.open("xb") as log:
                run = subprocess.run(
                    cmd + ["--", *argv, "--output", "{out}"],
                    cwd=self.project,
                    stdout=log,
                    stderr=log,
                    env={
                        **os.environ,
                        "PYTHONPATH": str(self.project / "src"),
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                )
            body = out.read_text().strip()
            require(run.returncode == 0, "failed recorder attempt " + str(out) + " " + body[-650:])
            return accept(json.loads(body.splitlines()[-1]))


def main(a):
    s = a.store.resolve()
    rec = Recorder(a.project, s, a.shelf or s.parent / "cold-shelf", a.state)
    if a.action == "restore":
        pin, cp, _ = rec.original(BUILD, "payload/evidence-lens-cpu-" + V + ".zip")
        _, eg, _ = rec.original(BUILD, "payload/evidence-lens-example-" + V + ".zip")
        return rec.stage(
            "published-assets",
            "derivation",
            [pin],
            {"cpu_zip": (cp, pin), "example_zip": (eg, pin)},
            [
                "{python}",
                "{source}/scripts/dtu_score_assets.py",
                "--cpu-zip",
                "{input:cpu_zip}",
                "--example-zip",
                "{input:example_zip}",
            ],
            "release-assets.json",
            650_000_000,
            {"purpose": "public CPU release and authored regression material; no DTU inputs"},
            a.retry_of,
        )
    if a.action == "score":
        gate = None
        arm = getattr(a, "arm", "main-256")
        require(arm == "main-256", "published risk scores are registered only for the main 256-step arm")
        if a.role != "regression":
            require(
                a.object_id == {"debug": "set1", "external": "set6"}[a.role], "wrong registered object role"
            )
        if a.role != "regression":
            require(
                a.authority and a.authority_member and a.authority_sha256,
                "debug or external scoring requires explicit controller authority",
            )
            authority, af, h = rec.original(a.authority, a.authority_member)
            require(h["sha256"] == a.authority_sha256, "wrong authority receipt")
            if a.role == "external":
                doc = load(af)
                require(
                    doc.get("schema") == "el-dtu-opening-1" and a.object_id in doc.get("object_ids", []),
                    "external case remains sealed",
                )
            gate = authority, af, h
        aid, asset, _ = rec.original(a.assets, "payload/release-assets.json")
        casep, casefile, meta = rec.original(a.case, a.case_member)
        require(
            meta["sha256"] == a.case_sha256 and load(casefile)["condition"] == a.condition,
            "wrong deployable case",
        )
        if a.role != "regression":
            _, pp, _ = rec.original(a.case, "payload/case-provenance.json")
            prov = load(pp)
            require(
                prov["schema"] == "el-dtu-case-provenance-1"
                and prov["object_id"] == a.object_id
                and prov["case_sha256"] == a.case_sha256
                and prov["condition"] == a.condition
                and prov["arm"] == arm
                and prov["optimization_steps"] == 256,
                "case belongs to another registered object or budget",
            )
            if a.role == "external":
                require(prov["opening_sha256"] == a.authority_sha256, "different source opening")
        src_asset = asset.parent / "cpu/release.json"
        require(file_hash(src_asset) == RELEASE_SHA, "unrecognized shared release")
        roles = {"release_index": (src_asset, aid), "case_index": (casefile, casep)}
        parents = [aid, casep]
        fields = {
            "release_sha256": RELEASE_SHA,
            "object_id": a.object_id,
            "condition_id": a.condition,
            "d1_role": a.role,
        }
        if a.role == "regression":
            require(
                a.object_id == "scene_057_offset_bell_cage" and aid == casep, "wrong public regression role"
            )
        else:
            authority, af, h = gate
            roles["authority"] = (af, authority)
            parents.append(authority)
            fields["authority_sha256"] = h["sha256"]
            fields["arm"] = arm
        return rec.stage(
            "score-"
            + a.role
            + "-"
            + a.object_id
            + "-"
            + a.condition
            + ("-" + arm if arm != "main-256" else ""),
            "model_score",
            parents,
            roles,
            [
                str(a.cpu_python.absolute()),
                "-I",
                "-B",
                "{source}/scripts/dtu_score.py",
                "--cpu-dir",
                str(src_asset.parent),
                "--release-sha256",
                RELEASE_SHA,
                "--case",
                str(casefile.parent),
                "--case-sha256",
                a.case_sha256,
                "--threads",
                "4",
            ],
            "scores.json",
            13_000_000,
            fields,
            a.retry_of,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("restore", "score"))
    for k in ("project", "store", "state"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--shelf", type=Path)
    p.add_argument("--cpu-python", type=Path)
    p.add_argument("--role", choices=("regression", "debug", "external"))
    p.add_argument("--condition", choices=("M2", "M4"))
    p.add_argument("--arm", choices=("main-256", "diagnostic-1024"), default="main-256")
    for k in (
        "assets",
        "case",
        "case-member",
        "case-sha256",
        "object-id",
        "authority",
        "authority-member",
        "authority-sha256",
        "retry-of",
    ):
        p.add_argument("--" + k)
    print(json.dumps(main(p.parse_args()), sort_keys=True))
