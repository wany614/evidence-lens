"""The GPU transport exposes only registered split members before its sealed model_fit."""

import json
import sys
from pathlib import Path

import pytest

from evidence_lens_multirisk.data import source_ids

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import multi_risk_job as job


def stage_fixture(split):
    from evidence_lens_multirisk.data import VIEWS

    rows = []
    names = [f"fixture-{i}" for i in range(8 if split == "train" else 4)]
    for f in names:
        for c in ("M2", "M4"):
            for v in VIEWS:
                folder = f"views/{f}/{c}/{v}/"

                def asset(n):
                    return {"path": folder + n, "sha256": "d" * 64, "bytes": 3}

                modes = {}
                for scope in ("one", "two", "all"):
                    alias = "two" if (c, scope) == ("M2", "all") else None
                    modes[scope] = {
                        "source_ids": source_ids(c, scope),
                        "alias_of": alias,
                        "samples": asset("training.npz"),
                        "deploy": asset((alias or scope) + "-deployment.npz"),
                    }
                rows.append(
                    {
                        "family": f,
                        "condition": c,
                        "view": v,
                        "modes": modes,
                        "status": "ready",
                        "rgb": asset("prediction-rgb.npy"),
                        "truth": asset("truth.npz"),
                        "selection_support": asset("selection-support.npy"),
                        "reference_rgb": asset("reference-rgb.npy"),
                    }
                )
    return {
        "schema": "el-stage3-risk-dataset-1",
        "split": split,
        "cohort": "formal",
        "families": names,
        "views": rows,
    }


def test_gpu_transport_inventory_is_split_specific_and_deduplicates_m2():
    train = stage_fixture("train")
    t = job.inventory(train, "train")
    assert (
        t["views"] == 96
        and len(t["files"]) == 192
        and not any("truth" in n or "reference" in n for n in t["files"])
    )
    dev = stage_fixture("development")
    d = job.inventory(dev, "development")
    assert d["views"] == 48 and len(d["files"]) == 264 and all("reference-rgb" not in n for n in d["files"])
    assert not any("/M2/" in n and "/all-deployment" in n for n in d["files"])
    train["views"][0]["modes"]["all"]["samples"]["path"] = "../test/hidden.npz"
    with pytest.raises(ValueError, match="unexpected data role"):
        job.inventory(train, "train")
    with pytest.raises(ValueError, match="complete train"):
        job.inventory(dev, "test")


def test_closed_formal_fit_does_not_open_blinded_or_any_training_material(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from evidence_lens_multirisk.train import run_fit

    x = tmp_path / "config.json"
    x.write_text(
        json.dumps({"schema": "el-stage3-neural-fit-1", "phase": "formal", "seed": 1729, "method": "MS-ft"})
    )
    monkeypatch.delenv("EL_RUN_KIND", raising=False)
    monkeypatch.delenv("EL_RUN_ID", raising=False)
    with pytest.raises(ValueError, match="inside.*model_fit"):
        run_fit(SimpleNamespace(config=x, config_sha256=job.sha(x), run_id="outside-study"))


def test_prepare_requires_original_3a_scores_and_creates_durable_job(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace
    import evidence_lens_multirisk
    from evidence_lens_multirisk.model import B_INDEX, B_RUN, B_WEIGHTS
    from evidence_lens_multirisk.train import CATALOG, SCHEMA, STUDY

    bundle = os.environ.get("EL_C_WORKER_ZIP")
    upstream = os.environ.get("EL_C_B_PACKAGE")
    if not bundle or not upstream:
        pytest.skip("set EL_C_WORKER_ZIP and EL_C_B_PACKAGE to verify the full handoff")
    sys.path.insert(0, str(Path(evidence_lens_multirisk.__file__).resolve().parents[2] / "scripts"))
    from vision_gpu_job import parse_pin

    records = {}

    class Stub:
        def __init__(self, i, kind, experiment):
            self.run = {"run_kind": kind, "experiment_id": experiment}
            self.files = {}
            self.dir = tmp_path / f"original-{i}"
            self.dir.mkdir()
            self.pin = {
                "run_id": "20000101T000000000000Z-" + format(i, "016x"),
                "manifest_sha256": format(i, "064x"),
            }
            self.key = self.pin["run_id"] + "=" + self.pin["manifest_sha256"]
            records[self.key] = self

        def open(self, n):
            return self.dir / n

        def add(self, n, data):
            p = self.dir / n
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(job.entries(data))
            self.files[n] = job.sig(p)
            return p

    train = Stub(1, "dataset_build", "stage3-risk-training-data")
    dev = Stub(2, "dataset_build", "stage3-risk-development-data")
    fit = Stub(3, "model_fit", "stage3-risk-light-fit")
    base = Stub(4, "threshold_select", "stage3-risk-light-development")
    for resource, split in ((train, "train"), (dev, "development")):
        doc = stage_fixture(split)
        # Make the logical training/development IDs disjoint; no science files are opened.
        doc = json.loads(json.dumps(doc).replace("fixture-", split + "-family-"))
        doc.update(catalog_sha256=CATALOG, study_sha256=STUDY, feature_schema_sha256=SCHEMA)
        info = job.inventory(doc, split)
        resource.add("payload/dataset.json", doc)
        for n, s in info["files"].items():
            resource.files["payload/" + n] = s
    study_fit = fit.add("payload/package/model.json", {"method": "MS-L1", "status": "fit"})
    fit.add("inputs.json", {"parents": [train.pin]})
    base.add("inputs.json", {"parents": [dev.pin, fit.pin]})
    base.add(
        "payload/package/preprocessing.json",
        {"training_dataset_sha256": job.sha(train.open("payload/dataset.json"))},
    )
    idx = base.add(
        "payload/package/model.json",
        {
            "schema": "el-stage3-light-package-1",
            "status": "frozen",
            "model_id": base.pin["run_id"],
            "method": "MS-L1",
            "weights_sha256": "e" * 64,
            "threshold": 0.4,
        },
    )
    devsha = job.sha(dev.open("payload/dataset.json"))
    spec = {}
    for name in ("B-L1-transfer", "MS-L1"):
        file = base.add(
            "payload/" + name + "/scores.json", {"method": name, "dataset_sha256": devsha, "views": [{}] * 48}
        )
        spec[name] = {"path": name + "/scores.json", **job.sig(file)}
    grade = {
        "schema": "el-stage3-light-development-1",
        "run_id": base.pin["run_id"],
        "development_dataset_sha256": devsha,
        "model_index_sha256": job.sha(idx),
        "weights_sha256": "e" * 64,
        "threshold": {"tau": 0.4},
        "fit_index_sha256": job.sha(study_fit),
        "transfer": {
            "model_index_sha256": "cd8573eaf809000be8510a7aa28f6f992215b75dbe2da44048ed1d0e34a18ca2"
        },
        "scores": spec,
    }
    base.add("payload/development.json", grade)
    package = Path(upstream).resolve()
    b = SimpleNamespace(
        files={"payload/package/weights.safetensors": {"sha256": B_WEIGHTS}},
        open=lambda n: package / n.removeprefix("payload/package/"),
    )
    bp = B_RUN + "=" + "a" * 64
    records[bp] = b
    monkeypatch.setattr(job, "check_original", lambda root, shelf, key: records[key])
    a = SimpleNamespace(
        store=tmp_path / "store",
        shelf=None,
        method="MS-ft",
        seed=1729,
        train=train.key,
        development=dev.key,
        baseline=base.key,
        B_parent=bp,
        worker=Path(bundle),
        worker_sha256=job.sha(bundle),
        output=tmp_path / "handoff",
    )
    receipt = job.prepare(a)
    j = job.check_job(a.output.resolve(), receipt["job_sha256"])
    assert j["parents"]["light_fit"] == fit.pin and j["B_weights_sha256"] == B_WEIGHTS
    assert (
        len(j["datasets"]["development"]["files"]) == 264
        and job.sha(a.output / "inputs/B-package/model.json") == B_INDEX
    )
    assert parse_pin(bp) == j["parents"]["B_parent"] and j["worker_sha256"] == job.sha(
        a.output / "inputs/worker.zip"
    )
    # A signed 3A claim missing its true transfer score receipt is rejected before materialization.
    del grade["scores"]["B-L1-transfer"]
    base.add("payload/development.json", grade)
    a.output = tmp_path / "wrong"
    with pytest.raises(ValueError, match="both measured"):
        job.prepare(a)
    assert not a.output.exists()
