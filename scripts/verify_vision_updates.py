"""Compare original, selected and exported tensors and the data-loss gradient witnesses on CPU."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(a):
    train, setup, changes, witnesses = map(read, (a.training, a.setup, a.changes, a.witnesses))
    run = train["run_id"]
    require(setup["run_id"] == run and train["status"] == "completed" and train["optimizer_steps"] > 0,
            "missing completed risk optimizer")
    with a.base.open("rb") as f:
        require(hashlib.file_digest(f, "sha256").hexdigest() == train["base_checkpoint_sha256"], "base changed")
    base = torch.load(a.base, map_location="cpu", weights_only=True)
    head = load_file(str(a.initial_head), device="cpu")
    original = {**{"backbone." + n: v for n, v in base.items()}, **head}
    selected = load_file(str(a.weights), device="cpu")
    before = {name: digest(tensor) for name, tensor in original.items()}
    require(set(original) == set(selected) and before == setup["initial_tensors"], "initial tensors differ")
    frozen, trained = changes["frozen_before"], changes["trainable"]
    require(set(frozen) | set(trained) == set(selected) and not (set(frozen) & set(trained))
            and frozen == changes["frozen_after"], "incomplete parameter partition")
    roles = ["projection.", "head."]
    vision = train["branch"] == "L2-vision-ft"
    require(vision or train["branch"] == "L2-frozen", "unexpected model branch")
    if vision:
        roles.extend(["backbone.blocks.10.", "backbone.blocks.11.", "backbone.norm."])
    require(set(trained) == {n for n in original if any(n.startswith(role) for role in roles)},
            "trainable names differ from the actual model contract")
    changed = {}
    for name, old in original.items():
        new = selected[name]
        require(old.shape == new.shape and old.dtype == new.dtype == torch.float32, "shape or precision differs")
        after = digest(new)
        if name in frozen:
            require(before[name] == frozen[name] == after, "exported frozen tensor differs")
            continue
        proof = trained[name]
        count = int(torch.count_nonzero(new != old))
        require(proof["before_sha256"] == before[name] and proof["after_sha256"] == after
                and proof["changed"] == count and proof["elements"] == old.numel(), "selected/exported tensor mismatch")
        changed[name] = count
    expected = {"projection.weight", "head.2.weight"}
    if vision:
        expected.update({"backbone.blocks.10.attn.qkv.weight", "backbone.blocks.11.attn.qkv.weight", "backbone.norm.weight"})
    counts = {n: 0 for n in expected}
    for m in witnesses:
        gradients = m["gradients_of_unregularized_view_mean_bce"]
        require(set(gradients) == expected and math.isfinite(m["bce"]) and m["n"] > 0,
                "missing named pre-decay task-loss gradient")
        for name, item in gradients.items():
            require(math.isfinite(item["max_abs"]) and math.isfinite(item["l2_norm"]), "nonfinite task gradient")
            counts[name] += int(item["l2_norm"] > 0 and item["max_abs"] > 0 and item["nonzero_coordinates"] > 0)
    require(all(n >= 3 for n in counts.values()) and all(sum(k for n, k in changed.items() if n.startswith(role)) > 0
            for role in roles), "insufficient measured gradients or parameter changes in a required group")
    report = {"schema": "el-verified-export-gradient-proof-1", "training_run_id": run, "branch": train["branch"],
              "optimizer_steps": train["optimizer_steps"], "upstream_tensors": len(base), "exported_tensors": len(selected),
              "frozen_tensors": len(frozen), "identical_frozen_tensors": True, "selected_hashes_equal_export": True,
              "trainable_tensors": len(trained), "changed_pretrained_scalars": sum(k for n, k in changed.items() if n.startswith("backbone.")),
              "changed_scalars_by_group": {role: sum(k for n, k in changed.items() if n.startswith(role)) for role in roles},
              "nonzero_task_gradient_witnesses": counts}
    a.output.mkdir(exist_ok=True, parents=True)
    with (a.output / "verified-updates.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, sort_keys=True, indent=2)
        f.write("\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("training", "setup", "changes", "witnesses", "base", "initial-head", "weights", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    print(json.dumps(verify(p.parse_args())))
