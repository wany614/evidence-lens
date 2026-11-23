"""Derive the primary frozen deployment from the one authenticated terminal Windows export failure."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "src")]
from evidence_lens_vision.contract import digest, file_record, json_bytes, output_dir, put, read, require

FAILED = {
    "run_id": "20261123T163117676676Z-81c14259dee84abf",
    "manifest_sha256": "2f811f866d6581f846484f089ddbeb9c36239499e213af0fff338243c149e536",
}
STATE = "8df402a3d8170c39bc9b1decd531f7dd5e2ff7e3a13a5fcb90030505a808bae9"
WEIGHTS = "00be5de5abd30bf3e7eb1686e458a96e3ca55dffd6b892af862c98913c85c0b6"
BEFORE = "781393a70acc7d43397d15736ed785899068be16698462bf888c85252a2f0f60"
AFTER = "3cba08e523112084ad352dadcb104d3bbd9a97ec503966dc1efb8eec9cbed045"
INDICES = {
    BEFORE: "e909d35bf1cb726ee3702a30a9420f3e5d86d4a9747eae17d39d772a4ec465fc",
    AFTER: "eef4b94b574af15ff03091bab9c9b3bf5f8ab82c15878b48051f1668553bf8d2",
}
MODULE = "src/evidence_lens_multirisk/deploy.py"
EXPERIMENT = "stage3-terminal-export-recovery"
HEAD_NAMES = {
    "projection.weight",
    "projection.bias",
    "head.0.weight",
    "head.0.bias",
    "head.2.weight",
    "head.2.bias",
}


def hex_bytes(b):
    return hashlib.sha256(b).hexdigest()


def source_equivalence(before, after):
    require(digest(before) == BEFORE and digest(after) == AFTER, "unregistered source archive")
    tables = []
    data = []
    for p, pin in ((before, BEFORE), (after, AFTER)):
        with zipfile.ZipFile(p) as z:
            raw = z.read("worker-source.json")
            require(hex_bytes(raw) == INDICES[pin], "unregistered worker index")
            idx = json.loads(raw)["files"]
            require(z.namelist() == sorted(set(idx) | {"worker-source.json"}), "incorrect source inventory")
            values = {n: z.read(n) for n in idx}
            require(
                all(idx[n] == {"sha256": hex_bytes(b), "bytes": len(b)} for n, b in values.items()),
                "worker member digest mismatch",
            )
            tables.append({n: q["sha256"] for n, q in idx.items()})
            data.append(values)
    old, new = data
    require(
        set(old) == set(new) and [k for k in old if old[k] != new[k]] == [MODULE],
        "unexpected worker differences",
    )
    a = b'        "schema": "el-multisource-risk-model-1",\n        **identity,\n'
    b = b'        **identity,\n        "schema": "el-multisource-risk-model-1",\n'
    require(
        old[MODULE].count(a) == 1 and old[MODULE].replace(a, b) == new[MODULE], "changed mathematical program"
    )
    return (
        tables[0],
        tables[1],
        {
            "original_optimizer_worker_sha256": BEFORE,
            "corrected_export_worker_sha256": AFTER,
            "members_checked": len(old),
            "only_changed_module": MODULE,
            "change": "Model-index dictionary key precedence",
            "mathematical_source_sha256": hex_bytes(json_bytes(tables[1])),
            "changed_source_before": tables[0][MODULE],
            "changed_source_after": tables[1][MODULE],
        },
    )


def recovery_identity(run, fit, index, proof):
    require(
        run["run_kind"] == "derivation"
        and run["experiment_id"] == EXPERIMENT
        and fit["method"] == "MS-frozen"
        and fit["seed"] == 1729
        and fit["optimizer_steps"] == 288
        and fit["optimizer_steps_in_this_record"] == proof["optimizer_steps_in_this_record"] == 0
        and fit["optimization_parent"] == proof["optimization_parent"] == FAILED
        and proof["terminal_checkpoint_sha256"] == STATE
        and fit["weights_sha256"] == proof["original_weights_sha256"] == WEIGHTS
        and proof["verified_terminal_tensors"] == 181
        and proof["original_all_scope_maps"] == 48
        and proof["source_equivalence"]["original_optimizer_worker_sha256"] == BEFORE
        and proof["source_equivalence"]["corrected_export_worker_sha256"]
        == index["worker_zip_sha256"]
        == AFTER
        and index["optimization_worker_zip_sha256"] == BEFORE
        and index["optimization_parent"] == FAILED,
        "unrecognized optimization/derivation lineage",
    )


def fetch_failure(store, shelf):
    from evidence_lens.formal_inputs import PinnedRun

    root = next((p for p in (store, shelf) if (p / "runs" / FAILED["run_id"]).is_dir()), None)
    require(root, "original Windows failure missing")
    run = PinnedRun(root, FAILED)
    require(
        run.run["run_kind"] == "model_fit"
        and run.run["execution_status"] == "failed"
        and run.run["experiment_id"] == "stage3-risk-neural-fit"
        and run.run["trial_id"] == "stage3-risk-ms-frozen-1729"
        and run.open("stderr.log")
        .read_text(encoding="utf-8")
        .rstrip()
        .endswith("ValueError: incomplete multi-source model index"),
        "wrong failure type",
    )
    return run


def terminal(original, B_weights):
    import torch
    from safetensors import safe_open

    def member(n):
        return original.open("payload/fit/" + n)

    setup, dev = read(member("setup.json")), read(member("development.json"))
    idx = read(member("package/model.json"))
    ck = member("training-state.pt")
    weights = member("package/weights.safetensors")
    require(
        digest(ck) == STATE
        and digest(weights) == WEIGHTS
        and idx["model_id"] == FAILED["run_id"]
        and idx["schema"] == "el-stage3-neural-fit-1"
        and (idx["method"], idx["seed"]) == ("MS-frozen", 1729)
        and idx["worker_zip_sha256"] == BEFORE
        and idx["threshold"] == dev["threshold"]["tau"],
        "wrong terminal source",
    )
    raw = torch.load(ck, weights_only=True, map_location="cpu")
    st = raw["state"]
    curves = read(member("curves.json"))
    witnessed = read(member("gradient-witnesses.json"))
    require(
        raw["origin_run_id"] == FAILED["run_id"]
        and raw["configuration_sha256"] == setup["effective_configuration_sha256"]
        and st["stop"] == "registered_epoch_limit"
        and st["steps"] == 288
        and st["epoch"] == len(curves) == 12
        and st["curve"] == curves
        and st["witnesses"] == witnessed
        and len(witnessed) == 8
        and dev["family_equal_ap"]
        == st["best_ap"]
        == max(e["development"]["family_equal_ap"] for e in curves)
        and st["best_epoch"]
        == next(e["epoch"] for e in curves if e["development"]["family_equal_ap"] == st["best_ap"])
        and set(raw["best"]) == set(raw["live"]) == HEAD_NAMES
        and all(int(v["step"].item()) == 288 for v in raw["optimizer"]["state"].values())
        and len(raw["optimizer"]["state"]) == len(HEAD_NAMES),
        "checkpoint is not the completed authorized optimization",
    )
    with (
        safe_open(str(B_weights), framework="pt", device="cpu") as base,
        safe_open(str(weights), framework="pt", device="cpu") as saved,
    ):
        backbone = {n for n in base.keys() if n.startswith("backbone.")}
        require(
            len(backbone) == 175 and set(saved.keys()) == backbone | HEAD_NAMES, "incomplete terminal state"
        )
        for n in saved.keys():
            expected = base.get_tensor(n) if n in backbone else raw["best"][n]
            actual = saved.get_tensor(n)
            require(
                expected.dtype == actual.dtype == torch.float32
                and expected.shape == actual.shape
                and expected.contiguous().numpy().tobytes() == actual.contiguous().numpy().tobytes(),
                "pretrained/selected checkpoint tensor differs: " + n,
            )
    return setup, idx, dev, st


def certify(a):
    from evidence_lens._verify import verify_bundle
    from evidence_lens_multirisk.model import B_INDEX, B_WEIGHTS

    require(
        os.environ.get("EL_RUN_KIND") == "archive_verify" and os.environ.get("EL_RUN_ID") == a.run_id,
        "record original terminal readback as an archive verification",
    )
    out = output_dir(a.output)
    old = fetch_failure(a.store, a.shelf)
    verify = verify_bundle(old.root)
    require(
        verify["manifest_sha256"] == FAILED["manifest_sha256"] == digest(a.failed_manifest)
        and digest(a.B_index) == B_INDEX
        and digest(a.B_weights) == B_WEIGHTS,
        "incorrect recorded original failure manifest or upstream",
    )
    _, idx, _, state = terminal(old, a.B_weights)
    _, _, eq = source_equivalence(old.open("inputs/worker/worker.zip"), a.worker)
    evidence = {
        "schema": "el-stage3-terminal-optimization-readback-1",
        "run_id": a.run_id,
        "optimization_parent": FAILED,
        "terminal_checkpoint_sha256": STATE,
        "original_weights_sha256": WEIGHTS,
        "optimizer_steps": state["steps"],
        "selected_epoch": state["best_epoch"],
        "terminal_epoch": state["epoch"],
        "stop": state["stop"],
        "source_equivalence": eq,
        "validated_tensors": 181,
        "selected_head_tensors_from_checkpoint": 6,
        "frozen_B_tensors_from_original": 175,
        "recorded_failure": "ValueError: incomplete multi-source model index",
        "original_model_index_sha256": old.files["payload/fit/package/model.json"]["sha256"],
        "original_index_schema": idx["schema"],
    }
    put(out / "terminal-optimization.json", evidence)
    print(json.dumps(evidence))


def recover(a):
    import numpy as np
    import torch
    from evidence_lens_vision.train import tensor_readback
    from evidence_lens_multirisk.data import Dataset, signed
    from evidence_lens_multirisk.deploy import PACKAGE_MEMBERS, load_model, score_grid
    from evidence_lens_multirisk.inference import save_scores
    from evidence_lens_multirisk.model import B_INDEX, B_RUN, B_WEIGHTS
    from evidence_lens_multirisk.proof import verify
    from evidence_lens_multirisk.train import CATALOG, task_sources
    from vision_gpu_job import original as successful, parse_pin

    require(
        os.environ.get("EL_RUN_KIND") == "derivation" and os.environ.get("EL_RUN_ID") == a.run_id,
        "derivation must use the research recorder",
    )
    out = output_dir(a.output)
    old = fetch_failure(a.store, a.shelf)
    parents = read(old.open("inputs.json"))["parents"]
    recorded = read(out.parent / "inputs.json")["parents"]
    certpin = parse_pin(a.certificate)
    cert = successful(a.store, a.shelf, certpin)
    creceipt = cert.open("payload/terminal-optimization.json")
    doc = read(creceipt)
    require(
        cert.run["run_kind"] == "archive_verify"
        and cert.run["experiment_id"] == "stage3-terminal-optimization-readback"
        and doc["optimization_parent"] == FAILED
        and doc["original_weights_sha256"] == WEIGHTS
        and doc["terminal_checkpoint_sha256"] == STATE
        and doc["validated_tensors"] == 181
        and doc["optimizer_steps"] == 288
        and doc["selected_epoch"] == 12,
        "invalid independent Windows readback",
    )
    require(
        {(p["run_id"], p["manifest_sha256"]) for p in recorded}
        == {(p["run_id"], p["manifest_sha256"]) for p in [*parents, certpin]},
        "missing original successful data/certificate parents",
    )
    require(
        digest(a.failed_manifest) == FAILED["manifest_sha256"],
        "derivation input differs from the exact original failed seal",
    )
    deps = {p["run_id"]: successful(a.store, a.shelf, p) for p in parents}
    B = deps[B_RUN]
    B_index, B_weights = (B.open("payload/package/" + p) for p in ("model.json", "weights.safetensors"))
    require(digest(B_index) == B_INDEX and digest(B_weights) == B_WEIGHTS, "wrong verified upstream")
    setup, broken, dev_receipt, state = terminal(old, B_weights)
    oldsrc, newsrc, equiv = source_equivalence(old.open("inputs/worker/worker.zip"), a.worker)
    src = task_sources()
    require(
        all(oldsrc.get(n) == h for n, h in setup["source_sha256"].items())
        and all(newsrc.get(n) == h for n, h in src.items())
        and broken["code_files"] == oldsrc,
        "source snapshot differs",
    )
    optim_identity = setup["identity"]
    cfg = read(old.open("inputs/config/config.json"))
    require(
        all(optim_identity[k] == v for k, v in cfg.items())
        and digest(old.open("inputs/config/config.json")) == optim_identity["config_sha256"],
        "original optimizer settings differ",
    )
    corrected_cfg = {**cfg, "worker_zip_sha256": AFTER, "code_files": newsrc}
    # The comparable 04 dispatch is derived by only substituting authenticated worker source.
    identity = {**corrected_cfg, "config_sha256": hex_bytes(json_bytes(corrected_cfg)), "model_id": a.run_id}
    design = {k: v for k, v in identity.items() if k not in ("model_id", "method", "seed", "config_sha256")}
    design_sha = hex_bytes(json_bytes(design))
    proof_dir = out / "optimization-original"
    proof_dir.mkdir()

    def member(n):
        return old.open("payload/fit/" + n)

    for n in ("setup.json", "package/model.json"):
        shutil.copyfile(member(n), proof_dir / ("setup.json" if n == "setup.json" else "broken-model.json"))
    shutil.copyfile(old.open("payload/transport-audit.json"), proof_dir / "transport-audit.json")
    put(proof_dir / "corrected-export-config.json", corrected_cfg)
    for n in (
        "initial-head.safetensors",
        "initialization.json",
        "curves.json",
        "training-state.pt",
        "gradient-witnesses.json",
        "parameter-changes.json",
        "development.json",
    ):
        shutil.copyfile(member(n), out / n)
    # Preserve original optimizer evidence and its identity in full. The envelope identifies this derivative.
    put(
        out / "setup.json",
        {
            **setup,
            "run_id": a.run_id,
            "paired_design_sha256": design_sha,
            "optimization_parent": FAILED,
            "optimization_paired_design_sha256": setup["paired_design_sha256"],
            "pairing_worker_sha256": AFTER,
            "optimization_worker_zip_sha256": BEFORE,
        },
    )
    package = out / "package"
    package.mkdir()
    require(set(broken["files"]) == PACKAGE_MEMBERS, "incomplete original parameter package")
    for n, r in broken["files"].items():
        p = member("package/" + n)
        dest = package / n
        dest.parent.mkdir(parents=True, exist_ok=True)
        require(r == file_record(p, old.root / "payload/fit/package"), "incorrect original package member")
        if n == "runtime/evidence_lens_multirisk/deploy.py":
            fixed = HERE.parents[1] / MODULE
            require(
                digest(fixed) == newsrc[MODULE] and digest(p) == oldsrc[MODULE],
                "unexpected runtime implementation",
            )
            shutil.copyfile(fixed, dest)
        else:
            shutil.copyfile(p, dest)
    table = {
        p.relative_to(package).as_posix(): file_record(p, package) for p in package.rglob("*") if p.is_file()
    }
    idx = {
        **broken,
        **identity,
        "schema": "el-multisource-risk-model-1",
        "files": table,
        "optimization_parent": FAILED,
        "optimization_worker_zip_sha256": BEFORE,
        "terminal_checkpoint_sha256": STATE,
    }
    put(package / "model.json", idx)
    msha = digest(package / "model.json")
    require(
        digest(package / "weights.safetensors") == WEIGHTS,
        "recovered artifact differs from original byte stream",
    )
    torch.set_num_threads(a.threads)
    stuff = load_model(package, msha)
    model, pre, _ = stuff
    tensor_check = tensor_readback(model, read(out / "parameter-changes.json"))
    devs = [p for p in deps.values() if p.run["experiment_id"] == "stage3-risk-development-data"]
    require(len(devs) == 1, "missing original development data")
    dataset = Dataset(
        devs[0].open("payload/dataset.json"),
        idx["development_dataset_sha256"],
        "development",
        signed(HERE.parents[1] / "protocol/stage3-families-v1.json", CATALOG),
        CATALOG,
        pre,
    )
    score_dir = out / "development-scores"
    score_dir.mkdir()
    rows = []
    numeric = []
    n = 0
    maximum = 0.0
    crossings = 0
    for r in dataset.rows:
        f, c, v = (r[k] for k in ("family", "condition", "view"))
        require(r["status"] == "ready", "expected the 48 original completed development cases")
        rel = f"{f}/{c}/all/{v}.npz"
        old_map = member("development-scores/" + rel)
        dst = score_dir / rel
        dst.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(old_map, dst)
        xs, ok = dataset.deployment(r, "all")
        current = score_grid(stuff, dataset.rgb(r), xs, ok, "cpu")
        with np.load(old_map, allow_pickle=False) as z:
            risk, valid, states = (z[k] for k in ("risk", "valid", "state"))
        require(
            risk.dtype == np.float32
            and valid.dtype == bool
            and states.dtype == np.uint8
            and valid.shape == risk.shape == states.shape == (256, 256)
            and np.array_equal(valid, np.isfinite(risk))
            and np.all(np.isnan(risk[~valid]))
            and np.all(states[~valid] == 255)
            and np.array_equal(states[valid], risk[valid].astype(np.float64) >= idx["threshold"])
            and np.array_equal(current["valid"], valid),
            "original threshold state/support differs",
        )
        delta = float(np.max(np.abs(current["risk"][valid].astype(np.float64) - risk[valid]), initial=0))
        require(delta <= 0.0002, "native Mac FP32 reconstruction exceeds registered equivalence tolerance")
        flips = int(np.count_nonzero(current["state"] != states))
        count = int(valid.sum())
        crossings += flips
        n += count
        maximum = max(maximum, delta)
        numeric.append(
            {
                "family": f,
                "condition": c,
                "view": v,
                "original_sha256": digest(old_map),
                "valid": count,
                "max_absolute_risk_error": delta,
                "threshold_crossings": flips,
            }
        )
        rows.append({"family": f, "condition": c, "view": v, "score": file_record(dst, score_dir)})
    require(len(rows) == 48 and n > 0, "original all-source risk evidence missing")
    save_scores(stuff, msha, dataset, score_dir, "cpu", None, rows)
    audit = {
        "schema": "el-stage3-terminal-export-recovery-1",
        "run_id": a.run_id,
        "optimization_parent": FAILED,
        "terminal_readback": {
            **certpin,
            "member": "payload/terminal-optimization.json",
            "sha256": digest(creceipt),
        },
        "terminal_checkpoint_sha256": STATE,
        "original_weights_sha256": WEIGHTS,
        "verified_terminal_tensors": 181,
        "source_equivalence": equiv,
        "original_paired_design_sha256": setup["paired_design_sha256"],
        "corrected_worker_paired_design_sha256": design_sha,
        "original_all_scope_maps": len(rows),
        "optimizer_steps_on_original": state["steps"],
        "optimizer_steps_in_this_record": 0,
        "canonical_all_source": "Original Windows selected FP32 maps, copied byte-for-byte including the M2 two/all alias.",
        "remaining_ablation_source": "Mac CPU inference from the same selected weights at the original operating point.",
        "native_development_readback": numeric,
        "max_absolute_risk_error": maximum,
        "threshold_crossings": crossings,
        "cuda_bitwise_reproduction_established": False,
        "original_training_receipts": {
            p: old.files["payload/fit/" + p]["sha256"]
            for p in (
                "setup.json",
                "initialization.json",
                "initial-head.safetensors",
                "gradient-witnesses.json",
                "curves.json",
                "parameter-changes.json",
                "development.json",
                "package/model.json",
            )
        },
    }
    put(out / "recovery.json", audit)
    fit = {
        "schema": "el-multisource-risk-training-1",
        "run_id": a.run_id,
        "phase": "formal",
        "status": "completed",
        "method": "MS-frozen",
        "seed": 1729,
        "source_B_index_sha256": B_INDEX,
        "source_B_weights_sha256": B_WEIGHTS,
        "training_dataset_sha256": idx["training_dataset_sha256"],
        "development_dataset_sha256": dataset.sha,
        "feature_schema_sha256": idx["feature_schema_sha256"],
        "source_sha256": setup["source_sha256"],
        "export_source_sha256": src,
        "optimizer_steps": state["steps"],
        "epochs": state["epoch"],
        "best_epoch": state["best_epoch"],
        "stop_reason": state["stop"],
        "primary_seed": True,
        "initial_head_sha256": setup["initial_head_sha256"],
        "best_development_family_equal_ap": state["best_ap"],
        "paired_design_sha256": design_sha,
        "threshold": idx["threshold"],
        "model_index_sha256": msha,
        "weights_sha256": WEIGHTS,
        "cpu_export": {
            **tensor_check,
            "native_sample_count": n,
            "scope": "all 48 native development all-source maps versus independently loaded CPU model",
            "support_identical": True,
            "max_absolute_risk_error": maximum,
            "strict_threshold_crossings": crossings,
        },
        "optimization_parent": FAILED,
        "optimizer_steps_in_this_record": 0,
        "optimization_worker_zip_sha256": BEFORE,
        "optimization_paired_design_sha256": setup["paired_design_sha256"],
        "recovery_sha256": digest(out / "recovery.json"),
        "peak_allocated_cuda": None,
        "peak_reserved_cuda": None,
        "wall_seconds": old.run["elapsed_seconds_monotonic"],
        "wall_seconds_scope": "Original Windows recorder wall time through terminal export exception.",
    }
    recover_run = {"run_kind": "derivation", "experiment_id": EXPERIMENT}
    recovery_identity(recover_run, fit, idx, audit)
    put(out / "training.json", fit)
    verify(B_index, B_weights, out, out / "independent-proof", digest(out / "training.json"))
    print(
        json.dumps(
            {
                "run_id": a.run_id,
                "original_steps": state["steps"],
                "new_steps": 0,
                "weights_sha256": WEIGHTS,
                "model_index_sha256": msha,
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("store", "shelf", "output", "worker"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--mode", choices=("certify", "recover"), required=True)
    for n in ("B-index", "B-weights", "failed-manifest"):
        p.add_argument("--" + n, type=Path)
    p.add_argument("--certificate")
    args = p.parse_args()
    (certify if args.mode == "certify" else recover)(args)
