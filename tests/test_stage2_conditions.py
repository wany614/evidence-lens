"""Real recorder/CPU stages on explicit synthetic data. No confirmation bundles are opened."""

import importlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from evidence_lens import recorder
from evidence_lens._bundle import BundleError, encoded, load, new_id, save, sha

ROOT = Path.cwd()
ASSETS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ASSETS / "scripts"))
# The historical fixture prepends its own scripts folder. Load the staged controls
# first so both the pre-promotion overlay and the promoted checkout use this revision.
for module in ("condition_jobs", "condition_mac", "condition_selection", "condition_stage"):
    importlib.import_module(module)

import stage2_conditions as controller


def test_all_eight_stay_closed_for_both_operations(tmp_path, monkeypatch):
    import stage2_process

    def wrong(*args):
        pytest.fail("read an image, result or model before the cohort opened")

    monkeypatch.setattr(stage2_process, "handoff", wrong)
    monkeypatch.setattr(controller, "returned", wrong)
    catalog = ASSETS / "protocol/stage2-families-v1.json"
    base = [
        "--project",
        str(ROOT),
        "--store",
        str(tmp_path),
        "--catalog",
        str(catalog),
        "--catalog-sha256",
        sha(catalog),
    ]
    for row in load(catalog)["families"]:
        if row["split"] != "test":
            continue
        for c in ("R1", "Q1"):
            with pytest.raises(BundleError, match="still sealed"):
                controller.main(
                    [
                        "process",
                        *base,
                        "--family",
                        row["scene_family_id"],
                        "--condition",
                        c,
                        "--returns-dir",
                        str(tmp_path),
                    ]
                )
        with pytest.raises(BundleError, match="still sealed"):
            controller.main(
                [
                    "dispatch",
                    *base,
                    "--families",
                    row["scene_family_id"],
                    "--batch",
                    "fixture",
                    "--exchange",
                    str(tmp_path),
                ]
            )
    assert list(tmp_path.iterdir()) == []


def fixture_context(ctx, root):
    """Build a temporary input-only staged worker; the test method marker exists only in this temp dir."""
    from make_handoff import sha_bytes, source_files, zip_bytes

    # The real project's work tree also has named intake folders with summary run.json files.
    note = ctx["store"] / "work" / "unrelated-intake"
    note.mkdir(parents=True, exist_ok=True)
    save(note / "run.json", {"execution_status": "succeeded", "manifest_bytes": 123})
    ref = ctx["store"] / "runs" / ctx["b0_ids"]["reference"] / "payload"
    family = ctx["family"]["scene_family_id"]
    catalog = ctx["family"]["dataset_hash"]
    code = source_files(formal=True)
    data = {"code/" + n: v for n, v in code.items()}
    data.update({n: (ref / n).read_bytes() for n in ("input.png", "cameras.json")})
    job = {"schema": "fixture-baseline", "files": {n: sha_bytes(v) for n, v in data.items()}}
    data["handoff.json"] = encoded(job)
    zipped = zip_bytes(data)
    folder = ctx["store"] / "stage2" / catalog / "dispatches/fixture"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (family + ".zip")).write_bytes(zipped)
    ctx["b0_dispatch"] = {
        "container_sha256": sha_bytes(zipped),
        "handoff_sha256": sha_bytes(data["handoff.json"]),
        "worker_source_sha256": sha_bytes(zip_bytes(code)),
    }
    opening = root / "fixture-opening.json"
    save(opening, {"fixture_only": True})
    marker = {
        "catalog_sha256": catalog,
        "opening_sha256": sha(opening),
        "freeze_run_id": new_id(),
        "freeze_manifest_sha256": "a" * 64,
        "method_sha256": sha(ctx["method_file"]),
    }
    ctx.update(
        project=ROOT,
        catalog_hash=catalog,
        quality_experiment=controller.EXPERIMENT,
        worker_source=controller.worker_source,
        q1_render_response=controller.q1_render_response,
        opening_path=opening,
        opening_evidence=marker,
    )
    ctx["release"]["test_opening_sha256"] = sha(opening)
    return ctx


def test_r1_stage2_actual_half_source_and_input_only_handoff(tmp_path, monkeypatch, capsys):
    # Reuse the real 512-pixel synthetic RGB/K construction and independent focal check.
    from test_condition_orchestration import (
        test_r1_recorder_derivation_and_worker_contains_only_exact_reduced_source as exercise,
    )

    old = importlib.import_module("condition_mac")
    original = old.prepare
    seen = []

    def new_prepare(ctx, retry=None):
        if "quality_experiment" not in ctx:
            fixture_context(ctx, tmp_path)
            seen.append(ctx)
        return original(ctx, retry)

    monkeypatch.setattr(old, "prepare", new_prepare)
    exercise(tmp_path, capsys)
    (ctx,) = seen
    package = tmp_path / "r1.zip"
    with zipfile.ZipFile(package) as z:
        job = json.loads(z.read("handoff.json"))
    assert job["experiment_id"] == controller.EXPERIMENT and job["schema"] == "el-stage2-condition-handoff-1"
    assert job["opening"] == ctx["opening_evidence"]
    assert controller.verify_dispatch(ctx)["job"] == job
    from condition_stage import read_job

    extracted = tmp_path / "r1"
    for corrupt in (
        dict(job, experiment_id="p6-heldout-conditions"),
        dict(job, opening={**job["opening"], "method_sha256": "0" * 64}),
    ):
        save(extracted / "handoff.json", corrupt)
        with pytest.raises(BundleError, match="Mac-opened"):
            read_job(extracted)
    downgrade = dict(
        job,
        schema="el-condition-handoff-1",
        experiment_id="p6-heldout-conditions",
        identity={**job["identity"], "dataset_id": "evidence-lens-vision-families-v1"},
    )
    save(extracted / "handoff.json", downgrade)
    with pytest.raises(BundleError, match="legacy condition gate"):
        read_job(extracted)
    save(extracted / "handoff.json", job)


def test_q1_stage2_record_selection_from_real_fp16_and_fixed_six_pixel_render(tmp_path, monkeypatch, capsys):
    from condition_jobs import make_job
    from condition_stage import read_job, checked_producer
    from formal_lineage import VerifiedRuns, run_pin
    from test_stress_conditions import small_ply
    from test_trial_selection import build_frozen_fixture

    # supply fixture dataset lineage to the original six-pixel B0 producer as the Stage B worker checks it.
    resolver = recorder.resolve_config

    def with_fixture_id(path, interval, limits):
        data = resolver(path, interval, limits)
        for k, v in {
            "scene_version": "v001",
            "dataset_id": "fixture",
            "dataset_hash": "5" * 64,
            "split": "test",
        }.items():
            data.setdefault(k, v)
        return data

    monkeypatch.setattr(recorder, "resolve_config", with_fixture_id)
    original_record = recorder.record

    def with_experiment(args):
        if args.experiment == "unit":
            args.experiment = "stage2-single-image"
        return original_record(args)

    monkeypatch.setattr(recorder, "record", with_experiment)
    p = tmp_path / "test.ply"
    small_ply(p, 0.1234567)
    store, frozen = build_frozen_fixture(tmp_path, monkeypatch, capsys, p)
    baseline = load(frozen)
    ids = {k: v["run_id"] for k, v in baseline["selected"].items()}
    family = {
        "scene_family_id": baseline["scene_family_id"],
        "scene_id": baseline["scene_id"],
        "scene_version": "v001",
        "trial_id": baseline["trial_id"],
        "split": "test",
        "dataset_id": "fixture",
        "dataset_hash": "5" * 64,
    }
    method = tmp_path / "fixture-method.json"
    save(method, {"fixture": True})
    ctx = fixture_context(
        {
            "store": store,
            "family": family,
            "condition": "Q1",
            "quality_trial": "fixture-Q1",
            "baseline": baseline,
            "b0_ids": ids,
            "b0_selection_sha256": sha(frozen),
            "verified": VerifiedRuns(store),
            "method_file": method,
            "release": {"method_freeze_sha256": sha(method)},
        },
        tmp_path,
    )
    out = tmp_path / "q1.zip"
    make_job(ctx, None, None, out)
    handoff = tmp_path / "worker"
    with zipfile.ZipFile(out) as z:
        assert set(n for n in z.namelist() if not n.startswith("code/")) == {
            "input.png",
            "cameras.json",
            "handoff.json",
        }
        z.extractall(handoff)
    job, digest = read_job(handoff)
    assert job["condition"] == "Q1" and job["experiment_id"] == controller.EXPERIMENT
    assert checked_producer(store, job, digest)["run_id"] == ids["generation"]
    with pytest.raises(BundleError, match="registered Stage B B0"):
        checked_producer(store, dict(job, baseline_trial_id="wrong-B0"), digest)
    with pytest.raises(BundleError, match="exact selected baseline"):
        checked_producer(store, dict(job, baseline_ply_sha256="0" * 64), digest)
    cfg = tmp_path / "condition.json"
    settings = {
        **load(ROOT / "configs/windows-b0.json"),
        **job["identity"],
        "formal_handoff_sha256": digest,
        "worker_source_sha256": job["worker_source_sha256"],
        "timing_repeat": None,
        "stage2_opening": job["opening"],
        "method_freeze_sha256": sha(method),
        "b0_selection_sha256": sha(frozen),
    }
    save(cfg, settings)

    def seal(kind, parents, inputs, command):
        flags = [
            "--root",
            str(store),
            "--source-root",
            str(ROOT),
            "--config",
            str(cfg),
            "--kind",
            kind,
            "--trial",
            ctx["quality_trial"],
            "--experiment",
            controller.EXPERIMENT,
        ]
        for rid in parents:
            flags += ["--parent", rid]
        for key, (rid, n) in inputs.items():
            flags += ["--input", f"{key}={store / 'runs' / rid / 'payload' / n}"]
        result = recorder.main([*flags, "--", "{python}", *command])
        captured = capsys.readouterr()
        assert result == 0, (captured.out, captured.err)
        rid = json.loads(captured.out.strip().splitlines()[-1])["run_id"]
        from worker_jobs import outgoing_receipt

        outgoing = tmp_path / "returns"
        outgoing.mkdir(exist_ok=True)
        outgoing_receipt(store, outgoing, store / "runs" / rid, digest)
        return rid

    bgen, ref = ids["generation"], ids["reference"]
    derived = seal(
        "derivation",
        [bgen],
        {"ply": (bgen, "scene.ply"), "generation": (bgen, "generation.json")},
        [
            "{source}/scripts/stress_condition.py",
            "Q1",
            "--ply",
            "{input:ply}",
            "--generation",
            "{input:generation}",
            "--out",
            "{out}",
        ],
    )
    reduced = load(store / "runs" / derived / "payload/generation.json")
    assert (
        reduced["forward_calls_in_this_stage"] == 0
        and checked_producer(store, job, digest, derived)["run_id"] == derived
    )
    fixture_worker = (
        "import os,json,pathlib,sys,shutil;"
        "d=pathlib.Path(sys.argv[1]);shutil.copytree(sys.argv[2],d,dirs_exist_ok=True,copy_function=shutil.copyfile);"
        "p=d/'prediction.json';m=json.loads(p.read_text());m.update(producer_run_id=os.environ['EL_RUN_ID'],"
        "generation_run_id=sys.argv[3],ply_sha256=sys.argv[4]);p.write_text(json.dumps(m))"
    )
    rendered = seal(
        "render",
        [ref, derived],
        {
            "rgb": (ref, "input.png"),
            "cameras": (ref, "cameras.json"),
            "ply": (derived, "scene.ply"),
            "generation": (derived, "generation.json"),
        },
        [
            "-c",
            fixture_worker,
            "{out}",
            str(store / "runs" / ids["render"] / "payload"),
            derived,
            reduced["ply_sha256"],
        ],
    )
    monkeypatch.setattr(controller, "study_context", lambda *args: ctx)
    args = SimpleNamespace(
        project=ROOT,
        store=store,
        catalog=tmp_path / "fixture",
        catalog_sha256=family["dataset_hash"],
        family=family["scene_family_id"],
        condition="Q1",
        returns_dir=tmp_path / "returns",
        terminal_failure=None,
        producer=None,
        render=None,
        cpu_python=Path(sys.executable),
        retry_stage=None,
        retry_of=None,
    )
    item = controller.process(args)
    out = load(Path(item["selection"]))
    assert (
        item["status"] == "selected"
        and out["experiment_id"] == controller.EXPERIMENT
        and out["opening"] == job["opening"]
    )
    assert (
        out["selected"]["generation"]["run_id"] == bgen and out["selected"]["preparation"]["run_id"] == derived
    )
    assert out["evidence"]["q1_field_check"]["struct_binary16_values_checked"] == 9
    response = out["evidence"]["q1_render_response"]["views"]
    assert set(response) == {"source", *controller.TARGETS} and all(
        v["geometry_bitwise_equal"] for v in response.values()
    )
    assert response["v01"]["joint_valid_rgb"] == 0
    assert all(v["pixels_rgb_mae_strictly_over_0_05"] == 0 for v in response.values())
    assert out["evidence"]["target_denominators"]["v01"]["rgb_evaluable"] == 0
    # A second launch verifies the same inputs via explicit already-received pins.
    args.returns_dir = None
    args.producer = run_pin(ctx["verified"].get(derived))
    args.render = run_pin(ctx["verified"].get(rendered))
    assert controller.process(args)["sha256"] == item["sha256"]


def test_two_b0_package_manifests_preserve_each_exact_worker_and_one_scientific_interface(tmp_path):
    from make_handoff import sha_bytes, zip_bytes

    store = tmp_path / "store"
    catalog = "9" * 64
    root = store / "stage2" / catalog / "dispatches/workers"
    root.mkdir(parents=True)
    archive = root / "fixture.zip"
    ctx = {"store": store, "project": ROOT, "catalog_hash": catalog, "family": {"scene_family_id": "fixture"}}

    def bundle(deps, kernel):
        package = (
            "[project]\nrequires-python='>=3.11'\ndependencies=['numpy>=1.26,<3']\n[project.optional-dependencies]\nrecord=['psutil']\n"
            + deps
        )
        code = {"pyproject.toml": package.encode(), "src/evidence_lens/renderer_adapter.py": kernel}
        members = {"code/" + n: v for n, v in code.items()}
        meta = encoded({"files": {n: sha_bytes(b) for n, b in members.items()}})
        raw = zip_bytes({**members, "handoff.json": meta})
        archive.write_bytes(raw)
        ctx["b0_dispatch"] = {
            "handoff_sha256": sha_bytes(meta),
            "container_sha256": sha_bytes(raw),
            "worker_source_sha256": sha_bytes(zip_bytes(code)),
        }
        return controller.worker_source(ctx)[1]

    a = bundle("", b"fixture-identical")
    b = bundle("vision=['fixture']\n", b"fixture-identical")
    directory = store / "formal" / catalog / "conditions"
    assert a != b and {
        load(f)["sha256"] for f in (directory / "worker-sources").glob("*/worker-source.json")
    } == {a, b}
    with pytest.raises(BundleError, match="immutable file conflicts"):
        bundle("vision=['fixture']\n", b"numerical change")
