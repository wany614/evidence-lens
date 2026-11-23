"""Fit one registered shared multi-source head under a sealed 3A-to-3B transition."""

import collections
import hashlib
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from evidence_lens_vision.contract import (
    HASH,
    RUN,
    digest,
    json_bytes,
    output_dir,
    put,
    read,
    require,
)
from evidence_lens_vision.model import image_tensor
from evidence_lens_vision.train import (
    change_record,
    delta_state,
    epoch_order,
    hashes,
    optimizer_for,
    restore_delta,
    save_state,
    seed_everything,
    tensor_readback,
)
from .data import CONDITIONS, Dataset, signed, validated_preprocessing
from .deploy import export, load_model
from .model import B_INDEX, B_WEIGHTS, METHODS, SEEDS, WITNESSES, groups, initialize, transform
from .selection import evaluate

CATALOG = "f14ca80777cf72e6f99075fb860e24cd38a6b4d9db342011173e9b216d3c66f5"
STUDY = "dd227c25648af4ab5e4e42d3fdd07702a8e46b3def36810ac9d65c55b07a7b23"
SCHEMA = "1f444f0dc9b7fce5c998bb677da20b2c18c25312d7c918e9bf17c6b290457e63"


def task_sources():
    import importlib.util

    # This table identifies the executable code read by optimizer, encoder, feature transformation and development selection.
    source = {}
    for pkg in ("evidence_lens", "evidence_lens_vision", "evidence_lens_multirisk"):
        if pkg == "evidence_lens":
            names = [
                "__init__",
                "formal_metrics",
                "evaluation",
                "risk_model",
                "features",
                "scoring_io",
                "labels",
                "formal_inputs",
                "comparison_board",
                "l1",
                "l1_data",
                "l1_freeze",
                "_bundle",
                "_metrics",
                "multisource",
                "multisource_io",
                "multisource_model",
            ]
        elif pkg == "evidence_lens_vision":
            names = ["__init__", "train", "model", "deploy", "base", "contract", "configure"]
        else:
            names = ["__init__", "data", "model", "train", "deploy", "selection", "proof", "inference"]
        for n in names:
            module = importlib.util.find_spec(pkg if n == "__init__" else pkg + "." + n)
            require(module and module.origin, "unavailable scientific module " + pkg + "." + n)
            source[f"src/{pkg}/{n}.py"] = digest(Path(module.origin))
    return source


def scheduled_samples(dataset):
    loaded = []
    counts = collections.Counter()
    dispositions = []
    for r in dataset.rows:
        f, c, v = (r[k] for k in ("family", "condition", "view"))
        if r["status"] == "ready":
            ix, x, y = dataset.samples(r)
            if len(ix):
                loaded.append((r, ix, x, y))
                counts[f, c] += 1
            status = "supervised" if len(ix) else "no_eligible_pixels"
        else:
            ix = []
            status = r["status"]
        dispositions.append(
            {
                "family": f,
                "condition": c,
                "view": v,
                "status": status,
                "n": len(ix),
                "reason": r.get("failure"),
            }
        )
    require(loaded, "no supervised multi-source RGB errors in the training split")
    fs = {f for f, c in counts}
    by = {f: sum(counts[f, c] > 0 for c in CONDITIONS) for f in fs}
    weights = [1 / (len(fs) * by[r["family"]] * counts[r["family"], r["condition"]]) for r, *_ in loaded]
    require(np.isclose(sum(weights), 1), "training family/condition/view weights do not sum to one")
    return loaded, weights, dispositions


def step(model, opt, train, pre, loaded, weights, chosen, device, state, witness_params):
    opt.zero_grad(set_to_none=True)
    losses = 0.0
    for i in chosen:
        r, ix, x, y = loaded[i]
        weight = weights[i] * len(loaded) / len(chosen)
        img, patches, frame, observed = image_tensor(train.rgb(r), device)
        require(observed.ravel()[ix].all(), "supervision selected a nonfinite native RGB pixel")
        features = torch.from_numpy(transform(x, pre)).to(device)
        prediction, ok = model.at(model.project(img), patches, frame, ix, features)
        require(
            ok.all().item() and torch.isfinite(prediction).all().item(),
            "supervised pixel lacks finite model support",
        )
        loss = F.binary_cross_entropy_with_logits(
            prediction, torch.from_numpy(y.astype(np.float32)).to(device)
        )
        require(torch.isfinite(loss).item(), "nonfinite supervised BCE")
        captured = {}
        hooks = []

        def hook(name):
            def inspect(grad):
                value = grad.detach().float() / weight
                require(torch.isfinite(value).all().item(), "nonfinite data-only task gradient")
                captured[name] = {
                    "l2_norm": float(torch.linalg.vector_norm(value)),
                    "max_abs": float(value.abs().max()),
                    "nonzero_coordinates": int(torch.count_nonzero(value)),
                }

            return inspect

        watch = len(state["witnesses"]) < 8
        if watch:
            hooks = [p.register_hook(hook(n)) for n, p in witness_params.items()]
        try:
            (loss * weight).backward()
        finally:
            for handle in hooks:
                handle.remove()
        value = float(loss.detach())
        losses += value * weights[i]
        if watch:
            require(set(captured) == set(witness_params), "a registered gradient hook was not reached")
            state["witnesses"].append(
                {
                    "family": r["family"],
                    "condition": r["condition"],
                    "view": r["view"],
                    "scope": "all",
                    "samples_sha256": r["modes"]["all"]["samples"]["sha256"],
                    "epoch": state["epoch"] + 1,
                    "upcoming_optimizer_step": state["steps"] + 1,
                    "n": len(ix),
                    "positive": int(y.sum()),
                    "bce": value,
                    "gradients_of_unregularized_view_mean_bce": captured,
                }
            )
    require(
        all(
            p.grad is not None and torch.isfinite(p.grad).all().item()
            for p in model.parameters()
            if p.requires_grad
        ),
        "nonfinite or absent trainable task gradient",
    )
    opt.step()
    return losses


def validate_witnesses(model, method, witnesses):
    roles = groups(method)
    params = dict(model.named_parameters())
    expected = {n for n in WITNESSES if n.startswith(roles)}
    require(
        set(WITNESSES) <= params.keys()
        and expected == {n for n in WITNESSES if params[n].requires_grad}
        and all(sum(n.startswith(g) for n in expected) == 1 for g in roles),
        "unwitnessed trainable group or incorrect frozen layer",
    )
    if witnesses is not None:
        require(
            len(witnesses) == 8
            and all(set(w["gradients_of_unregularized_view_mean_bce"]) == expected for w in witnesses),
            "requires eight complete data-only minibatch witnesses",
        )
        for n in expected:
            measurements = [w["gradients_of_unregularized_view_mean_bce"][n] for w in witnesses]
            require(
                sum(
                    r["l2_norm"] > 0 and r["max_abs"] > 0 and r["nonzero_coordinates"] > 0
                    for r in measurements
                )
                >= 3,
                "insufficient nonzero BCE evidence for " + n,
            )
    return {n: params[n] for n in sorted(expected)}


def fit_core(a, protocol, pre, train, dev, identity, scope="formal"):
    started = time.perf_counter()
    out = output_dir(a.output)
    method = identity["method"]
    seed = identity["seed"]
    device = torch.device(a.device)
    if device.type == "cuda":
        require(
            torch.cuda.is_available() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"),
            "deterministic CUDA workspace must be set before startup",
        )
    torch.set_num_threads(a.threads)
    seed_everything(seed)
    flags = {
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    require(
        flags["deterministic_algorithms"]
        and not any(flags[x] for x in ("cuda_matmul_tf32", "cudnn_tf32", "cudnn_benchmark", "warn_only")),
        "effective precision differs from formal FP32 protocol",
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, init = initialize(a.B_index, a.B_weights, pre["columns"], seed, method)
    put(out / "initialization.json", init)
    head = {
        n: t.detach().contiguous() for n, t in model.state_dict().items() if not n.startswith("backbone.")
    }
    save_file(head, str(out / "initial-head.safetensors"), metadata={"format": "pt"})
    model = model.to(device)
    initial_hash = hashes(model)
    start_delta = delta_state(model)
    witnesses = validate_witnesses(model, method, None)
    old_method = "L2-vision-ft" if method == "MS-ft" else "L2-frozen"
    opt, optimizer_groups = optimizer_for(
        model,
        {
            "branch": old_method,
            "head_lr": protocol["head_lr"],
            "backbone_lr": protocol["finetune_backbone_lr"],
        },
    )
    loaded, weights, dispositions = scheduled_samples(train)
    src = task_sources()
    if scope == "formal" or identity.get("code_files"):
        require(
            isinstance(identity.get("code_files"), dict)
            and HASH.fullmatch(str(identity.get("worker_zip_sha256", ""))),
            "formal optimizer requires the independently pinned scientific source",
        )
        require(
            all(identity["code_files"].get(n) == h for n, h in src.items()),
            "actual training imports differ from the signed worker",
        )
    design = {k: v for k, v in identity.items() if k not in ("model_id", "method", "seed", "config_sha256")}
    design_sha = hashlib.sha256(json_bytes(design)).hexdigest()
    context_sha = hashlib.sha256(
        json_bytes(
            {
                "design": design_sha,
                "method": method,
                "seed": seed,
                "protocol": protocol,
                "source_sha256": src,
                "initial": initial_hash,
            }
        )
    ).hexdigest()
    state = {
        "epoch": 0,
        "steps": 0,
        "curve": [],
        "bad_epochs": 0,
        "best_epoch": None,
        "best_ap": None,
        "witnesses": [],
        "stop": None,
    }
    best = None
    if a.resume is not None:
        require(a.resume_sha256 and digest(a.resume) == a.resume_sha256, "unmatched recovery state")
        saved = torch.load(a.resume, map_location="cpu", weights_only=True)
        require(
            saved["configuration_sha256"] == context_sha,
            "scientific restart changed its configuration, inputs or code",
        )
        restore_delta(model, saved["live"])
        opt.load_state_dict(saved["optimizer"])
        state = saved["state"]
        best = saved["best"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    else:
        require(not a.resume_sha256, "recovery hash needs its original checkpoint")
    evidence = {
        "schema": "el-multisource-training-setup-1",
        "run_id": identity["model_id"],
        "phase": scope,
        "identity": identity,
        "protocol": protocol,
        "source_sha256": src,
        "initial_tensors": initial_hash,
        "initial_head_sha256": digest(out / "initial-head.safetensors"),
        "optimizer_groups": optimizer_groups,
        "precision_readbacks": flags,
        "device": str(device),
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "training_dispositions": dispositions,
        "supervised_cells": len(loaded),
        "paired_design_sha256": design_sha,
        "effective_configuration_sha256": context_sha,
    }
    put(out / "setup.json", evidence)
    while state["stop"] is None:
        epoch = state["epoch"]
        order = epoch_order(seed, epoch, len(loaded))
        loss = 0.0
        for begin in range(0, len(order), protocol["accumulate"]):
            chosen = order[begin : begin + protocol["accumulate"]]
            loss += step(model, opt, train, pre, loaded, weights, chosen, device, state, witnesses)
            state["steps"] += 1
        val = evaluate(model, dev, pre, device)
        ap = val["family_equal_ap"]
        state["epoch"] = epoch + 1
        if state["best_ap"] is None or ap > state["best_ap"]:
            state.update(best_ap=ap, best_epoch=epoch + 1, bad_epochs=0)
            best = delta_state(model)
        else:
            state["bad_epochs"] += 1
        state["curve"].append(
            {
                "epoch": epoch + 1,
                "optimizer_steps": state["steps"],
                "weighted_mean_BCE": loss,
                "view_order_sha256": hashlib.sha256(json_bytes(order)).hexdigest(),
                "development": val,
            }
        )
        if state["bad_epochs"] >= protocol["patience"]:
            state["stop"] = "development_patience"
        elif state["epoch"] >= protocol["epochs"]:
            state["stop"] = "registered_epoch_limit"
        save_state(out, model, opt, state, best, context_sha, identity["model_id"])
        put(out / "curves.json", state["curve"])
    require(best is not None and state["steps"] > 0, "no selected multi-source checkpoint")
    restore_delta(model, best)
    validate_witnesses(model, method, state["witnesses"])
    changes = change_record(model, start_delta, initial_hash, "L2-frozen")
    for prefix in groups(method):
        require(
            sum(r["changed"] for n, r in changes["trainable"].items() if n.startswith(prefix)) > 0,
            "no selected change in " + prefix,
        )
    put(out / "parameter-changes.json", changes)
    put(out / "gradient-witnesses.json", state["witnesses"])
    put(out / "curves.json", state["curve"])
    if not (out / "training-state.pt").is_file():
        shutil.copyfile(a.resume, out / "training-state.pt")
    selected = evaluate(model, dev, pre, device, out / "development-scores")
    require(selected["family_equal_ap"] == state["best_ap"], "selected weights differ from best epoch")
    rows = selected.pop("score_rows")
    tau = selected["threshold"]["tau"]
    put(out / "development.json", selected)
    package = out / "package"
    export(model, package, a.preprocessing, a.feature_schema, a.B_index, {**identity, "phase": scope}, tau)
    msha = digest(package / "model.json")
    readback, _, _ = load_model(package, msha)
    tensor_check = tensor_readback(readback, changes)
    # Re-evaluate a training-only prediction at its native sampled coordinates on the reloaded CPU package.
    row, ix, x, _ = loaded[0]
    rgb = train.rgb(row)

    def sample(net, where):
        with torch.inference_mode():
            img, p, frame, _ = image_tensor(rgb, where)
            logits, sup = net.at(
                net.project(img), p, frame, ix, torch.from_numpy(transform(x, pre)).to(where)
            )
            return torch.sigmoid(logits).cpu().numpy(), sup.cpu().numpy()

    live, s1 = sample(model, device)
    cpu, s2 = sample(readback, "cpu")
    require(
        np.array_equal(s1, s2) and np.isfinite(cpu).all() and np.allclose(live, cpu, rtol=0, atol=0.0002),
        "CPU reload changed native risk/support",
    )
    from .inference import save_scores

    save_scores(
        (model, pre, read(package / "model.json")), msha, dev, out / "development-scores", device, None, rows
    )
    result = {
        "schema": "el-multisource-risk-training-1",
        "run_id": identity["model_id"],
        "phase": scope,
        "status": "completed",
        "method": method,
        "seed": seed,
        "source_B_index_sha256": B_INDEX,
        "source_B_weights_sha256": B_WEIGHTS,
        "training_dataset_sha256": train.sha,
        "development_dataset_sha256": dev.sha,
        "feature_schema_sha256": identity["feature_schema_sha256"],
        "source_sha256": src,
        "optimizer_steps": state["steps"],
        "epochs": state["epoch"],
        "best_epoch": state["best_epoch"],
        "stop_reason": state["stop"],
        "primary_seed": seed == SEEDS[0],
        "initial_head_sha256": evidence["initial_head_sha256"],
        "best_development_family_equal_ap": state["best_ap"],
        "paired_design_sha256": design_sha,
        "threshold": tau,
        "model_index_sha256": msha,
        "weights_sha256": digest(package / "weights.safetensors"),
        "cpu_export": {
            **tensor_check,
            "native_sample_count": len(ix),
            "support_identical": True,
            "max_absolute_risk_error": float(np.max(np.abs(live - cpu))),
        },
        "peak_allocated_cuda": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_reserved_cuda": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "wall_seconds": time.perf_counter() - started,
    }
    put(out / "training.json", result)
    return result


def run_fit(a):
    cfg = signed(a.config, a.config_sha256)
    require(
        cfg["schema"] == "el-stage3-neural-fit-1"
        and cfg["phase"] == "formal"
        and cfg["method"] in METHODS
        and cfg["seed"] in SEEDS,
        "unregistered stage 3B fit",
    )
    rid = os.environ.get("EL_RUN_ID")
    require(
        rid == a.run_id and RUN.fullmatch(rid or "") and os.environ.get("EL_RUN_KIND") == "model_fit",
        "a formal neural fit must execute inside the study model_fit recorder",
    )
    study = signed(a.protocol, STUDY)
    cat = signed(a.catalog, CATALOG)
    rule = study["neural"]
    require(
        cfg["protocol_sha256"] == STUDY
        and study["catalog_sha256"] == cfg["catalog_sha256"] == CATALOG
        and study["stage_b"]["encoder_parent_weights_sha256"] == B_WEIGHTS
        and cfg["feature_schema_sha256"] == study["feature_schema_sha256"] == SCHEMA
        and rule["branches"] == list(METHODS)
        and rule["candidate_seeds"] == list(SEEDS)
        and rule["train_scope"] == "all"
        and rule["max_formal_fits"] == 4
        and rule["microbatch"] == 1
        and rule["gradient_accumulation"] == 4
        and rule["weight_decay"] == 0.01
        and rule["precision"] == "float32"
        and rule["tf32"] is False,
        "changed roster, upstream weights, optimizer or scientific budget",
    )
    protocol = {
        "head_lr": rule["head_and_projection_lr"],
        "finetune_backbone_lr": rule["finetune_backbone_lr"],
        "weight_decay": rule["weight_decay"],
        "accumulate": rule["gradient_accumulation"],
        "epochs": rule["max_epochs"],
        "patience": rule["patience_epochs"],
        "registered_study_sha256": STUDY,
        "neural": rule,
    }
    seal = signed(a.authority, cfg["authority_sha256"])
    require(
        seal["schema"] == "el-stage3-neural-authority-1"
        and seal["stage3a_completed"] is True
        and seal["catalog_sha256"] == CATALOG
        and seal["protocol_sha256"] == cfg["protocol_sha256"]
        and all(
            seal[k] == cfg[k]
            for k in (
                "training_dataset_sha256",
                "development_dataset_sha256",
                "preprocessing_sha256",
                "feature_schema_sha256",
            )
        )
        and RUN.fullmatch(seal["stage3a_model"]["run_id"])
        and HASH.fullmatch(seal["stage3a_model"]["manifest_sha256"])
        and HASH.fullmatch(seal["stage3a_model"]["model_index_sha256"])
        and RUN.fullmatch(seal["stage3a_development"]["run_id"])
        and HASH.fullmatch(seal["stage3a_development"]["manifest_sha256"])
        and HASH.fullmatch(seal["stage3a_development"]["measurement_sha256"]),
        "completed, pinned 3A model and transfer/multi-source development evaluation are required before neural fitting",
    )
    pre = validated_preprocessing(
        a.preprocessing,
        cfg["preprocessing_sha256"],
        cfg["training_dataset_sha256"],
        cfg["feature_schema_sha256"],
    )
    schema = signed(a.feature_schema, SCHEMA)
    require(
        schema["columns"] == pre["columns"] and len(schema["numeric_columns"]) == pre["numeric_count"],
        "unmatched feature definition",
    )
    train = Dataset(a.train, cfg["training_dataset_sha256"], "train", cat, CATALOG, pre)
    dev = Dataset(a.development, cfg["development_dataset_sha256"], "development", cat, CATALOG, pre)
    require(
        len(train.families) == 8 and len(dev.families) == 4 and not set(train.families) & set(dev.families),
        "incomplete/disallowed scientific roster",
    )
    identity = {**cfg, "model_id": rid, "config_sha256": a.config_sha256}
    return fit_core(a, protocol, pre, train, dev, identity)
