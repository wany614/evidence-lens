"""An immutable Mac stage selection around the existing experiment recorder."""

import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from evidence_lens._bundle import encoded, load
from evidence_lens.formal_inputs import compact_pin, hash_json, require
from evidence_lens.scoring_io import file_hash
from vision_sources import find_run


def publish(path, value):
    body = encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
        temp = Path(f.name)
    try:
        try:
            os.link(temp, path)
        except FileExistsError:
            require(
                path.is_file() and path.read_bytes() == body, "immutable stage record conflict: " + path.name
            )
    finally:
        temp.unlink(missing_ok=True)


def parse_pin(value):
    rid, sep, sha = value.partition("=")
    require(sep == "=", "pin must be RUN_ID=MANIFEST_SHA256")
    return compact_pin({"run_id": rid, "manifest_sha256": sha})


class Recorder:
    def __init__(self, project, store, shelf, state, catalog_sha, study_sha, retry=None):
        self.project, self.store, self.shelf, self.state = map(
            lambda p: Path(p).resolve(), (project, store, shelf, state)
        )
        self.catalog_sha, self.study_sha = catalog_sha, study_sha
        self.retry = retry or {}
        self.roots = (self.store, self.shelf)

    def asset(self, pin, member):
        run = find_run(self.roots, compact_pin(pin))
        require(run.run["execution_status"] == "succeeded", "stage did not succeed")
        p = run.open("payload/" + member)
        return p, file_hash(p)

    def checked(self, name, selected, request, receipt):
        run = find_run(self.roots, compact_pin(selected))
        require(
            run.run["execution_status"] == "succeeded"
            and run.run["trial_id"] == "v11-" + name
            and load(run.open("config.resolved.yaml"))["scientific_request_sha256"] == request,
            "selected stage uses another scientific request",
        )
        p = run.open("payload/" + receipt)
        if receipt == "fit.json":
            require(load(p)["status"] == "fitted", "the LR fit did not produce eligible parameters")
        return compact_pin(selected) | {
            "receipt": receipt,
            "receipt_sha256": file_hash(p),
            "trial_id": run.run["trial_id"],
            "request_sha256": request,
            "run_kind": run.run["run_kind"],
        }

    def stage(self, name, kind, parents, inputs, scripts, arguments, receipt, output_bytes, settings=None):
        # inputs map a stable role to (small pinned file, parent run id or None). Large assets are read in place
        # through that staged index and each consumed member SHA is independently checked by the worker.
        dr = self.state / name
        dr.mkdir(parents=True, exist_ok=True)
        pins = {p["run_id"]: p["manifest_sha256"] for p in parents}
        inputsig = {r: {"sha256": file_hash(Path(p)), "parent": par} for r, (p, par) in inputs.items()}
        sci = {
            "stage": name,
            "kind": kind,
            "catalog_sha256": self.catalog_sha,
            "study_sha256": self.study_sha,
            "parents": pins,
            "inputs": inputsig,
            "settings": settings or {},
        }
        req = hash_json(sci)
        selected = dr / "selected.json"
        with (dr / "lock").open("ab") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if selected.exists():
                prev = load(selected)
                checked = self.checked(name, prev, req, receipt)
                require(checked == prev, "selected stage receipt changed")
                return checked
            code_root = Path(__file__).resolve().parent
            worker_names = sorted(set(scripts))
            code = {n: file_hash(code_root / n) for n in worker_names}
            code_id = hash_json(code)
            workers = self.state / "workers" / code_id
            workers.mkdir(parents=True, exist_ok=True)
            for n, digest in code.items():
                dest = workers / n
                if not dest.exists():
                    with tempfile.NamedTemporaryFile(dir=workers, delete=False) as stream:
                        interim = Path(stream.name)
                    try:
                        shutil.copyfile(code_root / n, interim)
                        try:
                            os.link(interim, dest)
                        except FileExistsError:
                            pass
                    finally:
                        interim.unlink(missing_ok=True)
                require(file_hash(dest) == digest, "worker snapshot changed")
            config = {
                "schema": "el-vision-stage-config-1",
                "project_id": "evidence_lens",
                "experiment_id": "stage2-vision-data",
                "trial_id": "v11-" + name,
                "dataset_hash": self.catalog_sha,
                "scientific_request_sha256": req,
                "request": sci,
                "worker_files": code,
                "recording": {"expected_output_bytes": output_bytes},
            }
            publish(dr / "config.json", config)
            attempts = sorted(dr.glob("attempt-*.json"))
            logfiles = sorted(dr.glob("attempt-*.log"))
            if logfiles and (not attempts or logfiles[-1].stem != attempts[-1].stem):
                log = logfiles[-1].read_text()
                require(
                    log.strip(),
                    "previous recorder stopped before reporting its run identity; inspect " + str(logfiles[-1]),
                )
                try:
                    res = json.loads(log.strip().splitlines()[-1])
                except (ValueError, IndexError) as exc:
                    raise ValueError("recorder emitted no sealed result; inspect " + str(logfiles[-1])) from exc
                publish(logfiles[-1].with_suffix(".json"), res)
                attempts = sorted(dr.glob("attempt-*.json"))
            if attempts:
                prev = load(attempts[-1])
                if prev["execution_status"] == "succeeded":
                    out = self.checked(name, prev, req, receipt)
                    publish(selected, out)
                    return out
                require(
                    self.retry.get(name) == prev["run_id"],
                    "failed attempt " + prev["run_id"] + " requires an explicitly recorded retry: " + name,
                )
            outfile = dr / f"attempt-{len(attempts) + 1:02d}.log"
            args = [
                sys.executable,
                "-m",
                "evidence_lens.recorder",
                "--root",
                str(self.store),
                "--source-root",
                str(self.project),
                "--config",
                str(dr / "config.json"),
                "--kind",
                kind,
                "--input",
                "workers=" + str(workers),
            ]
            for rid, sha in pins.items():
                args.extend(("--parent", rid + "=" + sha))
            for role, (path, owner) in inputs.items():
                args.extend(("--input", role + "=" + str(path)))
                if owner:
                    args.extend(("--input-parent", role + "=" + owner))
            if attempts:
                args.extend(("--retry-of", load(attempts[-1])["run_id"]))
            cmd = ["{python}", "{input:workers}/" + arguments[0], *arguments[1:]]
            with outfile.open("xb") as log:
                # The recorder holds this lock across a controller restart and writes its genuine result to disk.
                r = subprocess.run(
                    [*args, "--", *map(str, cmd)],
                    cwd=self.project,
                    env={**os.environ, "PYTHONPATH": str(self.project / "src")},
                    stdout=log,
                    stderr=log,
                    pass_fds=(lock.fileno(),),
                )
            text = outfile.read_text().strip()
            try:
                actual = json.loads(text.splitlines()[-1])
            except (ValueError, IndexError) as exc:
                raise ValueError("recorder did not emit a sealed run; see " + str(outfile)) from exc
            publish(outfile.with_suffix(".json"), actual)
            require(r.returncode == 0, "sealed failed run " + actual["run_id"] + "; see " + str(outfile))
            out = self.checked(name, actual, req, receipt)
            publish(selected, out)
            return out
