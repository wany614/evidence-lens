"""Independent CPU/NumPy check of original versus exported FP32 tensor bytes and task-gradient evidence."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from safetensors import safe_open

from evidence_lens_vision.contract import bound, digest, output_dir, put, read, require
from .model import B_INDEX, B_WEIGHTS, METHODS, WITNESSES, groups


def signature(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast("B")).hexdigest()


def positive(witness, names):
    require(
        set(witness["gradients_of_unregularized_view_mean_bce"]) == set(names)
        and witness["scope"] == "all"
        and witness["condition"] in ("M2", "M4")
        and witness["n"] > 0
        and 0 <= witness["positive"] <= witness["n"]
        and math.isfinite(witness["bce"]),
        "invalid real minibatch evidence",
    )
    out = {}
    for n, rec in witness["gradients_of_unregularized_view_mean_bce"].items():
        norm, mx, num = rec["l2_norm"], rec["max_abs"], rec["nonzero_coordinates"]
        require(
            math.isfinite(norm)
            and math.isfinite(mx)
            and norm >= 0
            and mx >= 0
            and type(num) is int
            and num >= 0,
            "nonfinite or malformed independent gradient witness",
        )
        out[n] = norm > 0 and mx > 0 and num > 0
    return out


def verify(base_index, base_weights, output, validation, training_sha256=None):
    original = Path(output)
    require(digest(base_index) == B_INDEX and digest(base_weights) == B_WEIGHTS, "wrong source B model")
    fit_path = original / "training.json"
    fit = read(fit_path)
    if training_sha256:
        require(digest(fit_path) == training_sha256, "unmatched training receipt")
    package = original / "package"
    index = read(package / "model.json")
    w = bound(package, index["files"]["weights.safetensors"])
    setup = read(original / "setup.json")
    init = read(original / "initialization.json")
    ch = read(original / "parameter-changes.json")
    witness = read(original / "gradient-witnesses.json")
    method = fit["method"]
    names = groups(method)
    rid = fit["run_id"]
    require(
        method in METHODS
        and fit["status"] == "completed"
        and fit["optimizer_steps"] > 0
        and fit["phase"] == index["phase"] == setup["phase"]
        and rid == setup["run_id"] == index["model_id"]
        and index["method"] == method == init["method"]
        and fit["seed"] == index["seed"] == init["seed"]
        and fit["model_index_sha256"] == digest(package / "model.json")
        and digest(w) == fit["weights_sha256"]
        and setup["initial_head_sha256"]
        == fit["initial_head_sha256"]
        == digest(original / "initial-head.safetensors"),
        "published parameters differ from original training/optimizer or saved initialization",
    )
    require(
        setup["identity"]["training_dataset_sha256"]
        == index["training_dataset_sha256"]
        == fit["training_dataset_sha256"]
        and setup["identity"]["development_dataset_sha256"]
        == index["development_dataset_sha256"]
        == fit["development_dataset_sha256"]
        and fit["source_B_index_sha256"]
        == index["source_B_index_sha256"]
        == init["B_model_sha256"]
        == B_INDEX
        and fit["source_B_weights_sha256"]
        == index["source_B_weights_sha256"]
        == init["B_weights_sha256"]
        == B_WEIGHTS
        and setup["paired_design_sha256"] == fit["paired_design_sha256"]
        and setup["source_sha256"] == fit["source_sha256"],
        "scientific input mismatch",
    )
    frozen = {}
    changes = {}
    afterhash = {}
    with (
        safe_open(str(base_weights), framework="numpy") as b,
        safe_open(str(original / "initial-head.safetensors"), framework="numpy") as h,
        safe_open(str(w), framework="numpy") as selected,
    ):
        keys = set(selected.keys())
        backbone = {x for x in b.keys() if x.startswith("backbone.")}
        head = set(h.keys())
        require(
            len(backbone) == 175
            and len(head) == 6
            and keys == set(b.keys()) == backbone | head
            and set(init["retained"]) == backbone
            and set(init["reinitialized"]) == head
            and not init["unexplained_names"]
            and set(ch["trainable"]) == {k for k in keys if k.startswith(names)}
            and set(ch["frozen_before"]) == keys - set(ch["trainable"]),
            "incomplete retained/reinitialized/optimized tensor inventory",
        )
        for k in sorted(keys):
            old = b.get_tensor(k) if k in backbone else h.get_tensor(k)
            new = selected.get_tensor(k)
            require(
                old.shape == new.shape and old.dtype == new.dtype == np.float32 and np.isfinite(new).all(),
                "wrong export tensor " + k,
            )
            before, after = signature(old), signature(new)
            afterhash[k] = after
            require(before == setup["initial_tensors"][k], "wrong starting bytes for " + k)
            if k in ch["trainable"]:
                count = int(np.count_nonzero(new != old))
                proof = ch["trainable"][k]
                require(
                    before == proof["before_sha256"]
                    and after == proof["after_sha256"]
                    and old.size == proof["elements"]
                    and count == proof["changed"],
                    "independent parameter change mismatch " + k,
                )
                changes[k] = count
            else:
                require(
                    before == after == ch["frozen_before"][k] == ch["frozen_after"][k],
                    "frozen pretrained parameter changed: " + k,
                )
                frozen[k] = after
    require(
        len(afterhash) == 181
        and set(ch["frozen_before"]) == set(ch["frozen_after"])
        and set(index["files"])
        and fit["cpu_export"]["all_tensor_sha256_equal"]
        and hashlib.sha256(
            (json.dumps(afterhash, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
        == fit["cpu_export"]["tensor_table_sha256"],
        "independent complete tensor table disagrees with CPU-export check",
    )
    counts = {g: sum(n for k, n in changes.items() if k.startswith(g)) for g in names}
    expected = [k for k in WITNESSES if k.startswith(names)]
    require(
        len(witness) == 8 and all(counts.values()),
        "insufficient signal or optimizer did not update a named group",
    )
    history = [positive(w, expected) for w in witness]
    observed = {k: sum(row[k] for row in history) for k in expected}
    require(all(n >= 3 for n in observed.values()), "insufficient unregularized data gradients")
    result = {
        "schema": "el-verified-multisource-update-1",
        "phase": fit["phase"],
        "method": method,
        "run_id": rid,
        "seed": fit["seed"],
        "training_sha256": digest(fit_path),
        "model_index_sha256": fit["model_index_sha256"],
        "weights_sha256": fit["weights_sha256"],
        "initial_head_sha256": fit["initial_head_sha256"],
        "paired_design_sha256": fit["paired_design_sha256"],
        "source_B_weights_sha256": B_WEIGHTS,
        "source_B_index_sha256": B_INDEX,
        "all_exported_tensor_hashes_match": True,
        "verified_tensors": 181,
        "checked_original_backbone": 175,
        "frozen_tensors": len(frozen),
        "nonzero_task_gradient_minibatches": observed,
        "backbone_changed_scalars": sum(n for k, n in changes.items() if k.startswith("backbone.")),
        "changed_scalars_by_group": counts,
        "optimizer_steps": fit["optimizer_steps"],
        "native_export_max_abs_error": fit["cpu_export"]["max_absolute_risk_error"],
    }
    dest = output_dir(validation)
    put(dest / "verified-updates.json", result)
    return result
