"""Prepare, record and receive one bounded stage C neural fit."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "src")]
from multi_risk_pack import extracted, entries


def load(p):
    return json.loads(Path(p).read_bytes())


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def sig(p):
    return {"sha256": sha(p), "bytes": Path(p).stat().st_size}


def require(ok, text):
    if not ok:
        raise ValueError(text)


def inventory(data, split):
    from evidence_lens_vision.contract import HASH
    from evidence_lens_multirisk.data import CONDITIONS, VIEWS, source_ids

    require(
        data["schema"] == "el-stage3-risk-dataset-1"
        and data["cohort"] == "formal"
        and data["split"] == split
        and split in ("train", "development"),
        "training job requires a complete train or development dataset",
    )
    files = {}
    keys = set()
    for row in data["views"]:
        f, c, v = (row[k] for k in ("family", "condition", "view"))
        key = f, c, v
        require(key not in keys and c in CONDITIONS and v in VIEWS, "duplicate or unregistered acquired view")
        keys.add(key)
        items = []
        folder = f"views/{f}/{c}/{v}/"
        if split == "development":
            items.append(("truth.npz", row["truth"]))
        if row["status"] == "ready":
            items.append(("prediction-rgb.npy", row["rgb"]))
            for scope, r in row["modes"].items():
                require(r["source_ids"] == source_ids(c, scope), "unknown scorer source subset")
            if split == "train":
                items.append(("training.npz", row["modes"]["all"]["samples"]))
            else:
                items.append(("selection-support.npy", row["selection_support"]))
                for scope, r in row["modes"].items():
                    if not r["alias_of"]:
                        items.append((scope + "-deployment.npz", r["deploy"]))
        else:
            require(row["status"] == "render_failed" and row["rgb"] is None, "invalid missing prediction")
        for filename, entry in items:
            n = folder + filename
            require(
                entry["path"] == n
                and n not in files
                and HASH.fullmatch(entry["sha256"])
                and type(entry["bytes"]) is int
                and entry["bytes"] > 0,
                "unexpected data role or pathname " + n,
            )
            files[n] = {x: entry[x] for x in ("bytes", "sha256")}
    roster = set(data["families"])
    require(
        keys == {(f, c, v) for f in roster for c in CONDITIONS for v in VIEWS}
        and len(roster) == (8 if split == "train" else 4),
        "incomplete split",
    )
    return {"files": files, "views": len(keys), "families": sorted(roster), "split": split}


def check_original(root, shelf, pin):
    from vision_gpu_job import original, parse_pin

    return original(root, shelf, parse_pin(pin))


def prepare(a):
    from evidence_lens.scoring_io import new_output
    from evidence_lens_multirisk.train import CATALOG, SCHEMA, STUDY
    from evidence_lens_multirisk.model import B_INDEX, B_RUN, B_WEIGHTS, METHODS, SEEDS
    from vision_gpu_job import parse_pin

    require(a.method in METHODS and a.seed in SEEDS, "outside the registered four formal fits")
    root = a.store.resolve()
    shelf = (a.shelf or root.parent / "cold-shelf").resolve()
    raw = {k: getattr(a, k) for k in ("train", "development", "baseline", "B_parent")}
    parents = {k: parse_pin(v) for k, v in raw.items()}
    src = {k: check_original(root, shelf, v) for k, v in raw.items()}
    b = src["baseline"]
    pkg = b.open("payload/package/model.json")
    base = load(pkg)
    graded = b.open("payload/development.json")
    g = load(graded)
    require(
        b.run["run_kind"] == "threshold_select"
        and b.run["experiment_id"] == "stage3-risk-light-development"
        and base["schema"] == "el-stage3-light-package-1"
        and base["method"] == "MS-L1"
        and base["status"] == "frozen"
        and base["model_id"] == g["run_id"] == parents["baseline"]["run_id"]
        and g["schema"] == "el-stage3-light-development-1"
        and g["model_index_sha256"] == sha(pkg)
        and g["weights_sha256"] == base["weights_sha256"]
        and base["threshold"] == g["threshold"]["tau"]
        and g["transfer"]["model_index_sha256"]
        == "cd8573eaf809000be8510a7aa28f6f992215b75dbe2da44048ed1d0e34a18ca2",
        "a genuine frozen MS-L1 and measured original transfer are required before 3B",
    )
    require(
        set(g["scores"]) == {"MS-L1", "B-L1-transfer"}, "both measured 3A development methods are required"
    )
    for name, member in g["scores"].items():
        require(member["path"] == name + "/scores.json", "unexpected stage 3A score location")
        sp = b.open("payload/" + member["path"])
        scores = load(sp)
        require(
            sha(sp) == member["sha256"]
            and sp.stat().st_size == member["bytes"]
            and scores["method"] == name
            and scores["dataset_sha256"] == g["development_dataset_sha256"]
            and len(scores["views"]) == 48,
            "incomplete original 3A transfer or lightweight development scores: " + name,
        )
    prep = b.open("payload/package/preprocessing.json")
    original = src["B_parent"]
    require(
        parents["B_parent"]["run_id"] == B_RUN
        and sha(original.open("payload/package/model.json")) == B_INDEX
        and original.files["payload/package/weights.safetensors"]["sha256"] == B_WEIGHTS,
        "the registered B FT source and all-source public package are required",
    )
    with new_output(a.output) as dest:
        work = dest / "code"
        source = extracted(a.worker, a.worker_sha256, work)
        for path, expected in (
            ("protocol/stage3-study-v1.json", STUDY),
            ("protocol/multisource-v1.json", SCHEMA),
            ("protocol/stage3-families-v1.json", CATALOG),
        ):
            require(sha(work / path) == expected, "wrong scientific worker " + path)
        srcdir = dest / "inputs"
        srcdir.mkdir()
        shutil.copyfile(a.worker, srcdir / "worker.zip")
        shutil.copyfile(prep, srcdir / "preprocessing.json")
        shutil.copyfile(pkg, srcdir / "stage3a-model.json")
        shutil.copyfile(graded, srcdir / "stage3a-development.json")
        bs = srcdir / "B-package"
        bs.mkdir()
        for n in ("model.json", "DINOv2-LICENSE", "LICENSE", "NOTICE"):
            shutil.copyfile(original.open("payload/package/" + n), bs / n)
        datasets = {}
        for k in ("train", "development"):
            require(src[k].run["run_kind"] == "dataset_build", "wrong scientific dataset producer")
            p = src[k].open("payload/dataset.json")
            doc = load(p)
            rec = inventory(doc, k)
            rec["manifest"] = sig(p)
            require(
                doc["catalog_sha256"] == CATALOG
                and doc["study_sha256"] == STUDY
                and doc["feature_schema_sha256"] == SCHEMA,
                "different C data inputs",
            )
            for n, s in rec["files"].items():
                m = src[k].files.get("payload/" + n, {})
                require(
                    all(m.get(j) == s[j] for j in ("bytes", "sha256")),
                    "the original sealed dataset does not supply " + n,
                )
            shutil.copyfile(p, srcdir / (k + ".json"))
            datasets[k] = rec
        trainsha = datasets["train"]["manifest"]["sha256"]
        devsha = datasets["development"]["manifest"]["sha256"]
        require(
            load(prep)["training_dataset_sha256"] == trainsha
            and g["development_dataset_sha256"] == devsha
            and not set(datasets["train"]["families"]) & set(datasets["development"]["families"]),
            "stage 3A/3B source split mismatch",
        )
        anc = {r["run_id"]: r["manifest_sha256"] for r in load(b.open("inputs.json"))["parents"]}
        require(
            anc.get(parents["development"]["run_id"]) == parents["development"]["manifest_sha256"],
            "original development dataset is not a 3A ancestor",
        )
        fitness = []
        for rid, manifest in anc.items():
            if rid == parents["development"]["run_id"]:
                continue
            candidate = check_original(root, shelf, rid + "=" + manifest)
            if (
                candidate.run["run_kind"] == "model_fit"
                and candidate.run["experiment_id"] == "stage3-risk-light-fit"
            ):
                q = candidate.open("payload/package/model.json")
                if sha(q) == g["fit_index_sha256"]:
                    fit_parents = {
                        p["run_id"]: p["manifest_sha256"]
                        for p in load(candidate.open("inputs.json"))["parents"]
                    }
                    require(
                        fit_parents.get(parents["train"]["run_id"]) == parents["train"]["manifest_sha256"],
                        "3A fit and 3B do not share the same training dataset",
                    )
                    fitness.append((rid, manifest))
        require(len(fitness) == 1, "unverified actual lightweight fit ancestor")
        parents["light_fit"] = dict(zip(("run_id", "manifest_sha256"), fitness[0]))
        ids = {
            "catalog_sha256": CATALOG,
            "protocol_sha256": STUDY,
            "feature_schema_sha256": SCHEMA,
            "preprocessing_sha256": sha(prep),
            "training_dataset_sha256": trainsha,
            "development_dataset_sha256": devsha,
        }
        authority = {
            "schema": "el-stage3-neural-authority-1",
            "stage3a_completed": True,
            **ids,
            "parents": parents,
            "stage3a_model": dict(parents["baseline"], model_index_sha256=sha(pkg)),
            "stage3a_development": dict(parents["baseline"], measurement_sha256=sha(graded)),
        }
        (srcdir / "authority.json").write_bytes(entries(authority))
        cfg = {
            "schema": "el-stage3-neural-fit-1",
            "phase": "formal",
            **ids,
            "method": a.method,
            "seed": a.seed,
            "authority_sha256": sha(srcdir / "authority.json"),
            "worker_zip_sha256": a.worker_sha256,
            "code_files": {n: s["sha256"] for n, s in source["files"].items()},
        }
        (srcdir / "config.json").write_bytes(entries(cfg))
        job = {
            "schema": "el-stage3-neural-job-1",
            "trial_id": f"stage3-risk-{a.method.lower()}-{a.seed}",
            "experiment_id": "stage3-risk-neural-fit",
            "method": a.method,
            "seed": a.seed,
            "datasets": datasets,
            "parents": parents,
            "worker_sha256": a.worker_sha256,
            "config_sha256": sha(srcdir / "config.json"),
            "B_weights_sha256": B_WEIGHTS,
            "catalog_sha256": CATALOG,
            "study_sha256": STUDY,
            "expected_output_bytes": 256 * 2**20,
            "files": {p.relative_to(dest).as_posix(): sig(p) for p in sorted(dest.rglob("*")) if p.is_file()},
        }
        (dest / "job.json").write_bytes(entries(job))
    return {
        "job_sha256": sha(a.output / "job.json"),
        "trial_id": job["trial_id"],
        "science_configuration_sha256": job["config_sha256"],
    }


def check_job(root, expected):
    from vision_gpu_job import valid_path

    require(sha(root / "job.json") == expected, "job does not match the independent Mac dispatch")
    j = load(root / "job.json")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    require(
        j["schema"] == "el-stage3-neural-job-1" and actual == set(j["files"]) | {"job.json"},
        "unregistered or incomplete handoff",
    )
    for n, s in j["files"].items():
        p = root / valid_path(n)
        require(p.resolve() == p and sig(p) == s, "changed job file " + n)
    return j


def scientific(a):
    from vision_gpu_job import audit_dataset

    j = load(a.job)
    require(
        os.environ.get("EL_RUN_ID") == a.run_id
        and os.environ.get("EL_RUN_KIND") == "model_fit"
        and sha(a.worker) == j["worker_sha256"]
        and sha(a.config) == j["config_sha256"],
        "unrecorded or substituted C fit",
    )
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    roots = {k: Path(os.environ["EL_C_" + k.upper() + "_ROOT"]).resolve() for k in ("train", "development")}
    checked = {k: audit_dataset(v, getattr(a, k), j["datasets"][k]) for k, v in roots.items()}
    audit = {
        "schema": "el-stage3-neural-transport-audit-1",
        "run_id": a.run_id,
        "job_sha256": sha(a.job),
        "parents": j["parents"],
        "datasets": checked,
        "worker_sha256": sha(a.worker),
        "config_sha256": sha(a.config),
    }
    (out / "transport-audit.json").write_bytes(entries(audit))
    with tempfile.TemporaryDirectory(prefix="stage3-neural-source-") as td:
        code = Path(td)
        extracted(a.worker, j["worker_sha256"], code)
        args = [
            sys.executable,
            "-I",
            "-B",
            str(code / "scripts/multi_risk_worker.py"),
            "train",
            "--B-index",
            str(a.B_package / "model.json"),
            "--B-weights",
            str(a.B_weights),
            "--feature-schema",
            str(code / "protocol/multisource-v1.json"),
            "--protocol",
            str(code / "protocol/stage3-study-v1.json"),
            "--catalog",
            str(code / "protocol/stage3-families-v1.json"),
            "--authority",
            str(a.authority),
            "--config",
            str(a.config),
            "--config-sha256",
            j["config_sha256"],
            "--train",
            str(roots["train"] / "dataset.json"),
            "--development",
            str(roots["development"] / "dataset.json"),
            "--preprocessing",
            str(a.preprocessing),
            "--device",
            "cuda",
            "--run-id",
            a.run_id,
            "--output",
            str(out / "fit"),
        ]
        res = os.environ.get("EL_C_RESUME_SHA")
        if res:
            from evidence_lens._verify import verify_bundle

            r = load(out.parent / "inputs.json")["retry_of"]
            p = Path(os.environ["EL_C_STORE"]) / "runs" / r["run_id"]
            old = verify_bundle(p)
            matches = [
                n
                for n in ("payload/fit/training-state.pt", "payload/training-state.pt")
                if old["artifacts"].get(n, {}).get("sha256") == res
            ]
            require(
                old["manifest_sha256"] == r["manifest_sha256"] and len(matches) == 1,
                "resume does not identify the original sealed failed attempt",
            )
            args += ["--resume", str(p / matches[0]), "--resume-sha256", res]
        result = subprocess.run(args, cwd=code)
        if result.returncode:
            return result.returncode
        # Preserve partial failures under fit/. On success expose the canonical flat payload to all scientific readers.
        for item in (out / "fit").iterdir():
            item.rename(out / item.name)
        (out / "fit").rmdir()
        command = [
            sys.executable,
            "-I",
            "-B",
            str(code / "scripts/multi_risk_worker.py"),
            "verify-updates",
            "--B-index",
            str(a.B_package / "model.json"),
            "--B-weights",
            str(a.B_weights),
            "--training",
            str(out),
            "--training-sha256",
            sha(out / "training.json"),
            "--output",
            str(out / "independent-proof"),
        ]
        return subprocess.run(command, cwd=code).returncode


def windows(a):
    from evidence_lens._index import writer_gate
    from evidence_lens.recorder import worker_started
    from worker_jobs import frozen_json, matching_runs, outgoing_receipt

    require(os.name == "nt", "formal C training runs on the registered native Windows GPU")
    root = a.job.resolve()
    j = check_job(root, a.job_sha256)
    code = root / "code"
    require(
        HERE == code / "scripts/multi_risk_job.py" and sig(a.B_weights)["sha256"] == j["B_weights_sha256"],
        "unexpected stage source or base",
    )
    store = a.store.resolve()
    ret = a.return_dir.resolve()
    ret.mkdir(parents=True, exist_ok=True)
    for p in (root, store, ret, a.B_weights, a.train_root, a.development_root):
        require(
            p.resolve().drive.casefold() == root.drive.casefold(),
            "stage scientific work on the designated research drive",
        )
    dr = store / "stage3-risk-jobs" / j["trial_id"]
    dr.mkdir(parents=True, exist_ok=True)
    with writer_gate(dr):
        frozen_json(dr / "job.json", {"sha256": a.job_sha256})
        found, pending = matching_runs(store, j, a.job_sha256, "model_fit", a.retry_of)
        require(not pending, "a fit for this immutable job is still running")
        if not a.retry_of:
            for p in found:
                retried, active = matching_runs(store, j, a.job_sha256, "model_fit", p.name)
                require(not active, "a linked fit is in progress")
                successes = [r for r in retried if load(r / "run.json")["execution_status"] == "succeeded"]
                require(len(successes) <= 1, "multiple successful retries")
                if successes:
                    return outgoing_receipt(store, ret, successes[0], a.job_sha256)
        else:
            p = store / "runs" / a.retry_of
            meta = load(p / "run.json")
            require(
                meta["trial_id"] == j["trial_id"]
                and meta["execution_status"] != "succeeded"
                and not meta["retry_of_run_id"]
                and worker_started(p, meta)
                and load(p / "config.resolved.yaml")["formal_handoff_sha256"] == a.job_sha256,
                "invalid scientific retry",
            )
        active = [p for p in found if worker_started(p, load(p / "run.json"))]
        require(len(active) <= 1, "multiple scientific runs for this job")
        if active:
            return outgoing_receipt(store, ret, active[0], a.job_sha256)
        if found and not a.after_preflight:
            return outgoing_receipt(store, ret, found[-1], a.job_sha256)
        if a.after_preflight:
            require(found and found[-1].name == a.after_preflight, "wrong prior preflight failure")
        require(not a.resume_sha256 or a.retry_of, "resume requires its failed predecessor")
        marker = dr / (
            (a.retry_of or "initial") + ("-after-" + a.after_preflight if a.after_preflight else "") + ".json"
        )
        require(
            not marker.exists(), "reserved dispatch without a sealed record; inspect the original process"
        )
        config = {
            "dataset_hash": j["catalog_sha256"],
            "experiment_id": j["experiment_id"],
            "trial_id": j["trial_id"],
            "formal_handoff_sha256": a.job_sha256,
            "scientific_configuration": load(root / "inputs/config.json"),
            "original_parents": j["parents"],
            "recording": {"expected_output_bytes": j["expected_output_bytes"]},
        }
        frozen_json(dr / "config.json", config)
        args = [
            sys.executable,
            "-m",
            "evidence_lens.recorder",
            "--root",
            str(store),
            "--source-root",
            str(code),
            "--kind",
            "model_fit",
            "--config",
            str(dr / "config.json"),
        ]
        pins = {p["run_id"]: p["manifest_sha256"] for p in j["parents"].values()}
        for rid, h in pins.items():
            args += ["--parent", rid + "=" + h]
        for role, n, owner in (
            ("train", "train.json", "train"),
            ("development", "development.json", "development"),
            ("preprocessing", "preprocessing.json", "baseline"),
            ("light", "stage3a-model.json", "baseline"),
            ("light_dev", "stage3a-development.json", "baseline"),
            ("B_package", "B-package", "B_parent"),
            ("config", "config.json", None),
            ("authority", "authority.json", None),
            ("worker", "worker.zip", None),
        ):
            args += ["--input", role + "=" + str(root / "inputs" / n)]
            if owner:
                args += ["--input-parent", role + "=" + j["parents"][owner]["run_id"]]
        args += [
            "--input",
            "job=" + str(root / "job.json"),
            "--shared-input",
            "B_weights=" + str(a.B_weights.resolve()),
            "--input-parent",
            "B_weights=" + j["parents"]["B_parent"]["run_id"],
        ]
        if a.retry_of:
            args += ["--retry-of", a.retry_of]
        args += [
            "--",
            "{python}",
            "{source}/scripts/multi_risk_job.py",
            "scientific",
            "--run-id",
            "{run_id}",
            "--output",
            "{out}",
        ]
        for role in (
            "job",
            "train",
            "development",
            "preprocessing",
            "config",
            "authority",
            "worker",
            "B_package",
            "B_weights",
        ):
            args += ["--" + role.replace("_", "-"), "{input:" + role + "}"]
        env = dict(
            os.environ,
            PYTHONPATH=str(code / "src"),
            PYTHONDONTWRITEBYTECODE="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            EL_C_TRAIN_ROOT=str(a.train_root.resolve()),
            EL_C_DEVELOPMENT_ROOT=str(a.development_root.resolve()),
            EL_C_STORE=str(store),
        )
        env.pop("EL_C_RESUME_SHA", None)
        if a.resume_sha256:
            env["EL_C_RESUME_SHA"] = a.resume_sha256
        temp = store / "cache/stage3-neural-temp"
        temp.mkdir(parents=True, exist_ok=True)
        env.update(TEMP=str(temp), TMP=str(temp))
        frozen_json(
            marker, {"job_sha256": a.job_sha256, "retry_of": a.retry_of, "after_preflight": a.after_preflight}
        )
        try:
            result = subprocess.run(args, cwd=code, env=env, text=True, stdout=subprocess.PIPE)
        except OSError:
            marker.unlink()
            raise
        if not result.stdout.strip():
            again, active = matching_runs(store, j, a.job_sha256, "model_fit", a.retry_of)
            if not active and again == found:
                marker.unlink()
            raise ValueError("recorder exited without a sealed result, exit " + str(result.returncode))
        last = json.loads(result.stdout)
        path = store / "runs" / last["run_id"]
        require(
            path == Path(last["path"]).resolve()
            and load(path / "config.resolved.yaml")["formal_handoff_sha256"] == a.job_sha256
            and sha(path / "manifest.json") == last["manifest_sha256"],
            "wrong final recorder result",
        )
        return outgoing_receipt(store, ret, path, a.job_sha256)


def receive(a):
    from evidence_lens.archive import receive as take

    j = check_job(a.job.resolve(), a.job_sha256)
    r = load(a.receipt)
    p = a.archive
    require(
        r["schema"] == "el-worker-return-1"
        and r["handoff_sha256"] == a.job_sha256
        and r["run_kind"] == "model_fit"
        and r["trial_id"] == j["trial_id"]
        and r["package"] == p.name == r["run"] + ".zip"
        and sig(p) == {"bytes": r["bytes"], "sha256": r["sha256"]},
        "return does not identify this scientific job",
    )
    with zipfile.ZipFile(p) as z:
        get = lambda n: json.loads(z.read(n))
        m = get("run.json")
        require(
            m["run_id"] == r["run"]
            and m["execution_status"] == r["status"]
            and m["experiment_id"] == j["experiment_id"]
            and get("config.resolved.yaml")["formal_handoff_sha256"] == a.job_sha256
            and hashlib.sha256(z.read("manifest.json")).hexdigest() == r["manifest_sha256"],
            "wrong canonical C fit",
        )
        if r["status"] == "succeeded":
            fit = get("payload/training.json")
            proof = get("payload/independent-proof/verified-updates.json")
            audit = get("payload/transport-audit.json")
            require(
                fit["phase"] == "formal"
                and fit["run_id"] == proof["run_id"] == r["run"]
                and fit["method"] == j["method"]
                and fit["seed"] == j["seed"]
                and proof["all_exported_tensor_hashes_match"]
                and proof["weights_sha256"] == fit["weights_sha256"]
                and proof["model_index_sha256"] == fit["model_index_sha256"]
                and audit["parents"] == j["parents"]
                and audit["job_sha256"] == a.job_sha256,
                "wrong fit, inputs or independent tensor proof",
            )
            for k, d in j["datasets"].items():
                require(
                    audit["datasets"][k]["verified_asset_table_sha256"]
                    == hashlib.sha256(entries(d["files"])).hexdigest(),
                    "dataset audit mismatch",
                )
    result = take(p, a.store, r["sha256"])
    require(result["status"] in ("accepted", "identical"), "canonical Mac receipt " + str(result))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest="action", required=True)

    def paths(s, names):
        for n in names:
            s.add_argument("--" + n, type=Path, required=True)

    def strings(s, names):
        for n in names:
            s.add_argument("--" + n, required=True)

    s = sp.add_parser("prepare")
    paths(s, ("store", "worker", "output"))
    s.add_argument("--shelf", type=Path)
    strings(s, ("worker-sha256", "method", "train", "development", "baseline", "B-parent"))
    s.add_argument("--seed", type=int, required=True)
    s = sp.add_parser("windows")
    paths(s, ("job", "store", "train-root", "development-root", "B-weights", "return-dir"))
    strings(s, ("job-sha256",))
    for n in ("retry-of", "after-preflight", "resume-sha256"):
        s.add_argument("--" + n)
    s = sp.add_parser("scientific")
    paths(
        s,
        (
            "job",
            "train",
            "development",
            "preprocessing",
            "config",
            "authority",
            "worker",
            "B-package",
            "B-weights",
            "output",
        ),
    )
    strings(s, ("run-id",))
    s = sp.add_parser("receive")
    paths(s, ("store", "job", "archive", "receipt"))
    strings(s, ("job-sha256",))
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    value = {"prepare": prepare, "windows": windows, "scientific": scientific, "receive": receive}[a.action](
        a
    )
    if isinstance(value, int):
        raise SystemExit(value)
    print(json.dumps(value, sort_keys=True))
