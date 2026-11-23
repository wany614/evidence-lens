"""Independent Stage C readbacks from original train/development members and a fixed ViT-S/14 parameter roster."""

import hashlib
from collections import Counter

import numpy as np
from safetensors import safe_open

from evidence_lens_vision.contract import digest, json_bytes, read, require
from evidence_lens_multirisk.data import SCOPES, VIEWS, source_ids
from evidence_lens_multirisk.model import B_INDEX, B_WEIGHTS
from evidence_lens_multirisk.train import CATALOG, SCHEMA, STUDY

BASE_SHAPES = {
    "attn.qkv.weight": (1152, 384),
    "attn.qkv.bias": (1152,),
    "attn.proj.weight": (384, 384),
    "attn.proj.bias": (384,),
    "mlp.fc1.weight": (1536, 384),
    "mlp.fc1.bias": (1536,),
    "mlp.fc2.weight": (384, 1536),
    "mlp.fc2.bias": (384,),
    "norm1.weight": (384,),
    "norm1.bias": (384,),
    "norm2.weight": (384,),
    "norm2.bias": (384,),
    "ls1.gamma": (384,),
    "ls2.gamma": (384,),
}
TAIL = {
    **{"backbone.blocks.11." + k: v for k, v in BASE_SHAPES.items()},
    "backbone.norm.weight": (384,),
    "backbone.norm.bias": (384,),
}
HEAD = {
    "projection.weight": (32, 384),
    "projection.bias": (32,),
    "head.0.weight": (32, 71),
    "head.0.bias": (32,),
    "head.2.weight": (1, 32),
    "head.2.bias": (1,),
}
WITNESSES = {"projection.weight", "head.2.weight"}
TAIL_WITNESSES = {"backbone.blocks.11.attn.qkv.weight", "backbone.norm.weight"}


def verify_base(index, weights):
    require(digest(index) == B_INDEX and digest(weights) == B_WEIGHTS, "wrong released Stage B baseline")
    with safe_open(str(weights), framework="numpy") as fh:
        keys = set(fh.keys())
        selected = {k for k in keys if k.startswith(("backbone.norm.", "backbone.blocks.11."))}
        require(
            selected == set(TAIL) and len([k for k in keys if k.startswith("backbone.")]) == 175,
            "unexpected actual DINOv2 final block",
        )
        for name, shape in TAIL.items():
            a = fh.get_tensor(name)
            require(
                a.shape == shape and a.dtype == np.float32 and np.isfinite(a).all(),
                "incorrect original shape/type " + name,
            )
    require(
        sum(int(np.prod(v)) for v in TAIL.values()) == 1776000
        and sum(int(np.prod(v)) for v in HEAD.values()) == 14657,
        "incorrect independent parameter budget",
    )
    return {
        "source_weights_sha256": B_WEIGHTS,
        "source_index_sha256": B_INDEX,
        "block11_tensors": 14,
        "final_norm_tensors": 2,
        "tail_scalars": 1776000,
        "head_scalars": 14657,
        "tail_shapes": TAIL,
        "head_shapes": HEAD,
    }


def source_member(original, spec):
    p = original.open("payload/" + spec["path"])
    s = original.files["payload/" + spec["path"]]
    require(
        s["sha256"] == spec["sha256"] and s["bytes"] == spec["bytes"],
        "original data role disagrees: " + spec["path"],
    )
    return p


class Inputs:
    def __init__(self, train, development):
        from evidence_lens.multisource_model import validate_rows

        self.train, self.dev = train, development
        self.train_path = train.open("payload/dataset.json")
        self.dev_path = development.open("payload/dataset.json")
        self.train_sha = digest(self.train_path)
        self.dev_sha = digest(self.dev_path)
        self.docs = [read(self.train_path), read(self.dev_path)]
        self.training = {}
        self.samples = {}
        self.dev_masks = {}
        self.train_cells = Counter()
        self.supervised = []
        for split, doc, original, num in zip(
            ("train", "development"), self.docs, (train, development), (8, 4)
        ):
            require(
                doc["schema"] == "el-stage3-risk-dataset-1"
                and doc["cohort"] == "formal"
                and doc["split"] == split
                and doc["feature_schema_sha256"] == SCHEMA
                and doc["catalog_sha256"] == CATALOG
                and doc["study_sha256"] == STUDY
                and len(set(doc["families"])) == len(doc["families"]) == num,
                "incorrect original formal split",
            )
            wanted = {(f, c, v) for f in doc["families"] for c in ("M2", "M4") for v in VIEWS}
            rows = sorted(doc["views"], key=lambda r: (r["family"], r["condition"], r["view"]))
            require(
                len(wanted) == len(rows)
                and {(r["family"], r["condition"], r["view"]) for r in rows} == wanted,
                "incomplete scheduled split",
            )
            for r in rows:
                k = tuple(r[n] for n in ("family", "condition", "view"))
                f, c, v = k
                if split == "train":
                    self.training[k] = r
                if r["status"] == "render_failed":
                    require(r["failure"] and r["rgb"] is None, "missing original failure")
                    if split == "development":
                        self.dev_masks[k] = None
                    continue
                require(
                    r["status"] == "ready" and (r["height"], r["width"]) == (256, 256),
                    "invalid native original",
                )
                rgb = source_member(original, r["rgb"])
                if split == "train":
                    spec = r["modes"]["all"]["samples"]
                    p = source_member(original, spec)
                    with np.load(p, allow_pickle=False) as z:
                        ix, x, y, prob = (z[n] for n in ("index", "X", "Y", "inclusion_probability"))
                    n = len(ix)
                    eligible = r["modes"]["all"]["eligible"]
                    require(
                        n == min(4096, eligible)
                        and ix.dtype == np.int32
                        and ix.shape == (n,)
                        and (not n or (ix[0] >= 0 and ix[-1] < 65536 and np.all(np.diff(ix) > 0)))
                        and x.shape == (n, 39)
                        and x.dtype == np.float32
                        and y.dtype == np.uint8
                        and y.shape == (n,)
                        and np.all(y <= 1)
                        and prob.dtype == np.float64
                        and prob.ndim == 0
                        and float(prob) == (n / eligible if eligible else 0.0),
                        "corrupt scientific training samples or eligibility",
                    )
                    validate_rows(x)
                    self.samples[k] = {
                        "sample": p,
                        "spec": spec,
                        "rgb": rgb,
                        "n": n,
                        "positive": int(y.sum()),
                    }
                    if n:
                        self.supervised.append(k)
                        self.train_cells[f, c] += 1
                else:
                    raw = np.load(rgb, allow_pickle=False)
                    require(
                        raw.dtype == np.float32 and raw.shape == (256, 256, 3),
                        "invalid true development prediction",
                    )
                    seen = np.isfinite(raw).all(-1)
                    self.dev_masks[k] = {}
                    for s in SCOPES:
                        m = r["modes"][s]
                        a = m["alias_of"]
                        require(
                            m["source_ids"] == source_ids(c, s)
                            and a == ("two" if (c, s) == ("M2", "all") else None),
                            "changed source subset",
                        )
                        if a:
                            require(m["deploy"] == r["modes"][a]["deploy"], "M2 alias input changed")
                            self.dev_masks[k][s] = self.dev_masks[k][a]
                            continue
                        dp = source_member(original, m["deploy"])
                        with np.load(dp, allow_pickle=False) as z:
                            ok = z["scoreable"]
                        require(
                            ok.shape == seen.shape
                            and ok.dtype == bool
                            and int(ok.sum()) == m["scoreable_pixels"],
                            "invalid original support",
                        )
                        self.dev_masks[k][s] = ok & seen
        require(
            not set(self.docs[0]["families"]) & set(self.docs[1]["families"]),
            "training and development overlap",
        )


def checked_development(run, fit, data):
    doc = read(run.open("payload/development-scores/scores.json"))
    idx = {}
    for row in doc["views"]:
        k = tuple(row[n] for n in ("family", "condition", "view"))
        require(k not in idx, "duplicate official score key")
        idx[k] = row
    require(
        doc["method"] == fit["method"]
        and doc["model_id"] == fit["run_id"]
        and doc["threshold"] == fit["threshold"]
        and doc["weights_sha256"] == fit["weights_sha256"]
        and doc["model_index_sha256"] == fit["model_index_sha256"]
        and doc["split"] == "development"
        and doc["dataset_sha256"] == fit["development_dataset_sha256"] == data.dev_sha
        and set(idx) == set(data.dev_masks),
        "fit score file uses a different model or native roster",
    )
    evidence = {}
    logical = 0
    cache = {}
    for k, masks in data.dev_masks.items():
        row = idx[k]
        require(
            (row["status"] == "render_failed") == (masks is None), "development score disposition changed"
        )
        require(set(row["modes"]) == set(SCOPES), "wrong set of source ablations")
        for scope in SCOPES:
            m = row["modes"][scope]
            c = k[1]
            alias = "two" if (c, scope) == ("M2", "all") else None
            require(
                m["source_ids"] == source_ids(c, scope) and m["alias_of"] == alias,
                "unexpected score evidence subset",
            )
            if alias:
                require(
                    m["file"] == row["modes"]["two"]["file"],
                    "M2 all score must reuse the exact two-source file",
                )
            logical += 1
            if masks is None:
                require(m["file"] is None, "failed original acquired a risk map")
                continue
            s = m["file"]
            n = s["path"]
            p = "payload/development-scores/" + n
            if n not in cache:
                f = run.open(p)
                r = run.files[p]
                require(
                    s["sha256"] == r["sha256"] and s["bytes"] == r["bytes"],
                    "sealed development member differs",
                )
                with np.load(f, allow_pickle=False) as z:
                    require(set(z.files) == {"risk", "valid", "state"}, "unknown development score fields")
                    a, valid, state = (z[t] for t in ("risk", "valid", "state"))
                mask = masks[scope]
                require(
                    a.dtype == np.float32
                    and valid.dtype == bool
                    and state.dtype == np.uint8
                    and a.shape == valid.shape == state.shape == (256, 256)
                    and np.array_equal(np.isfinite(a), valid)
                    and np.array_equal(valid, mask)
                    and np.isnan(a[~valid]).all()
                    and np.array_equal(state[valid], a[valid].astype(np.float64) >= fit["threshold"])
                    and (state[~valid] == 255).all(),
                    "development score, support or strict threshold states disagree",
                )
                evidence[n] = {
                    "sha256": s["sha256"],
                    "bytes": s["bytes"],
                    "low": int((state == 0).sum()),
                    "high": int((state == 1).sum()),
                    "unknown": int((state == 255).sum()),
                }
                cache[n] = (k, scope)
            else:
                require(alias and cache[n] == (k, alias), "unregistered score-file reuse")
    require(logical == 144, "missing scheduled development score cells")
    return {
        "score_index_sha256": run.files["payload/development-scores/scores.json"]["sha256"],
        "logical_scope_views": logical,
        "unique_array_files": len(evidence),
        "files": evidence,
    }


def experiment_setup(run, fit, data):
    setup = read(run.open("payload/setup.json"))
    init = read(run.open("payload/initialization.json"))
    changes = read(run.open("payload/parameter-changes.json"))
    curve = read(run.open("payload/curves.json"))
    flags = setup["precision_readbacks"]
    conditions = list(data.train_cells.items())
    require(
        set(flags)
        == {
            "cuda_matmul_tf32",
            "cudnn_tf32",
            "cudnn_benchmark",
            "deterministic_algorithms",
            "warn_only",
            "cublas_workspace",
        }
        and flags["deterministic_algorithms"] is True
        and all(flags[k] is False for k in ("cuda_matmul_tf32", "cudnn_tf32", "cudnn_benchmark", "warn_only"))
        and flags["cublas_workspace"] == ":4096:8"
        and setup["device"] == "cuda",
        "wrong formal effective precision",
    )
    wanted = {**HEAD, **(TAIL if fit["method"] == "MS-ft" else {})}
    require(
        set(changes["trainable"]) == set(init["trainable_names"]) == set(wanted)
        and init["trainable_scalars"] == sum(int(np.prod(v)) for v in wanted.values())
        and all(changes["trainable"][n]["elements"] == int(np.prod(shape)) for n, shape in wanted.items()),
        "optimizer roster differs from independent exact DINO layer and head names",
    )
    groups = {r["role"]: r for r in setup["optimizer_groups"]}
    expected_groups = {"projection_head": (HEAD, 0.0003)}
    if fit["method"] == "MS-ft":
        expected_groups["pretrained_suffix"] = (TAIL, 0.00001)
    require(set(groups) == set(expected_groups), "unexpected AdamW groups")
    for role, (params, lr) in expected_groups.items():
        r = groups[role]
        require(
            set(r["names"]) == set(params)
            and r["lr"] == lr
            and r["weight_decay"] == 0.01
            and r["scalars"] == sum(int(np.prod(s)) for s in params.values()),
            "named AdamW group differs",
        )
    dispositions = {
        tuple(r[x] for x in ("family", "condition", "view")): r for r in setup["training_dispositions"]
    }
    require(
        set(dispositions) == set(data.training)
        and len(dispositions) == len(setup["training_dispositions"]) == 96
        and len(conditions) == 16
        and setup["supervised_cells"] == len(data.supervised),
        "not all 16 family/acquisition cells contributed to supervision",
    )
    for k, r in dispositions.items():
        require(
            r["n"] == data.samples.get(k, {}).get("n", 0)
            and (r["status"] == "supervised") == (k in data.supervised),
            "wrong optimizer sample inventory",
        )
    for epoch, cut in enumerate(curve):
        order = (
            np.random.default_rng(np.random.SeedSequence([fit["seed"], epoch]))
            .permutation(len(data.supervised))
            .tolist()
        )
        require(
            cut["view_order_sha256"] == hashlib.sha256(json_bytes(order)).hexdigest()
            and cut["epoch"] == epoch + 1
            and cut["optimizer_steps"] == (epoch + 1) * ((len(data.supervised) + 3) // 4),
            "incorrect optimizer steps or registered training order",
        )
    vals = [r["development"]["family_equal_ap"] for r in curve]
    best = vals.index(max(vals)) + 1
    require(
        best == fit["best_epoch"]
        and fit["best_development_family_equal_ap"] == vals[best - 1]
        and fit["optimizer_steps"] == curve[-1]["optimizer_steps"]
        and 1 <= len(curve) == fit["epochs"] <= 12,
        "wrong earliest-best selected checkpoint or effective epoch budget",
    )
    return {
        "approved_tensor_names": sorted(wanted),
        "head_scalars": 14657,
        "eligible_backbone_scalars": 1776000 if fit["method"] == "MS-ft" else 0,
        "precision_readbacks": flags,
        "supervised_family_conditions": [
            {"family": f, "condition": c, "supervised_views": n} for (f, c), n in sorted(conditions)
        ],
        "training_arrays": len(data.samples),
        "training_samples": sum(r["n"] for r in data.samples.values()),
        "training_sample_sha256": {"/".join(k): v["spec"]["sha256"] for k, v in sorted(data.samples.items())},
        "source_train_dataset_sha256": data.train_sha,
        "best_epoch": best,
        "epoch_count": len(curve),
        "optimizer_steps": fit["optimizer_steps"],
    }


def first_step_replay(run, fit, bindex, bweights, data, threads):
    import torch
    from safetensors.torch import load_file
    from torch.nn.functional import binary_cross_entropy_with_logits
    from evidence_lens.multisource_model import transform
    from evidence_lens_vision.model import RiskModel, image_tensor

    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    require(digest(bindex) == B_INDEX and digest(bweights) == B_WEIGHTS, "wrong upstream initialization")
    init = load_file(str(run.open("payload/initial-head.safetensors")), device="cpu")
    require(
        set(init) == set(HEAD) and all(tuple(init[k].shape) == shape for k, shape in HEAD.items()),
        "incorrect original first head",
    )
    base = load_file(str(bweights), device="cpu")
    net = RiskModel(features=39).eval()
    values = {k: t for k, t in base.items() if k.startswith("backbone.")}
    values.update(init)
    net.load_state_dict(values, strict=True)
    tx = {**HEAD, **(TAIL if fit["method"] == "MS-ft" else {})}
    params = dict(net.named_parameters())
    for k, p in params.items():
        p.requires_grad_(k in tx)
    names = sorted(WITNESSES | (TAIL_WITNESSES if fit["method"] == "MS-ft" else set()))
    w = read(run.open("payload/gradient-witnesses.json"))
    pre = read(run.open("payload/package/preprocessing.json"))
    seed = fit["seed"]
    order = np.random.default_rng(np.random.SeedSequence([seed, 0])).permutation(len(data.supervised))
    require(
        len(w) == 8 and pre["training_dataset_sha256"] == data.train_sha,
        "wrong first-epoch sample/preprocessor",
    )
    rows = []
    observations = Counter()
    for i, rec in enumerate(w):
        key = data.supervised[int(order[i])]
        expected_step = 1 + i // 4
        item = data.samples[key]
        recs = rec["gradients_of_unregularized_view_mean_bce"]
        require(
            tuple(rec[k] for k in ("family", "condition", "view")) == key
            and rec["epoch"] == 1
            and rec["upcoming_optimizer_step"] == expected_step
            and rec["scope"] == "all"
            and rec["samples_sha256"] == item["spec"]["sha256"]
            and rec["n"] == item["n"]
            and rec["positive"] == item["positive"]
            and set(recs) == set(names),
            "BCE minibatch identity, labels or optimizer position differ",
        )
        for n in names:
            g = recs[n]
            if g["max_abs"] > 0 and g["l2_norm"] > 0 and g["nonzero_coordinates"] > 0:
                observations[n] += 1
        entry = {
            "family": key[0],
            "condition": key[1],
            "view": key[2],
            "upcoming_optimizer_step": expected_step,
            "sample_sha256": item["spec"]["sha256"],
            "recomputed_from_unchanged_initialization": i < 4,
            "recorded_bce": rec["bce"],
            "recorded_gradients": recs,
        }
        if i < 4:
            with np.load(item["sample"], allow_pickle=False) as z:
                ix, x, y = (z[k] for k in ("index", "X", "Y"))
            rgb = np.load(item["rgb"], allow_pickle=False)
            im, support, frame, observed = image_tensor(rgb, "cpu")
            features = torch.from_numpy(transform(x, pre, np.float32))
            logits, ok = net.at(net.project(im), support, frame, ix, features)
            require(
                ok.all().item() and observed.ravel()[ix].all(),
                "original first-step sample lacks native evidence",
            )
            loss = binary_cross_entropy_with_logits(logits, torch.from_numpy(y.astype(np.float32)))
            derivatives = torch.autograd.grad(loss, [params[n] for n in names])
            actual = float(loss.detach())
            proofs = {}
            for n, gradient in zip(names, derivatives, strict=True):
                g = gradient.float().detach()
                l2 = float(torch.linalg.vector_norm(g))
                max_abs = float(g.abs().max())
                nonzero = int(torch.count_nonzero(g))
                win = recs[n]["l2_norm"]
                delta = abs(l2 - win)
                limit = max(1e-6, 0.01 * max(l2, win))
                proofs[n] = {
                    "cpu_l2": l2,
                    "windows_l2": win,
                    "absolute_l2_difference": delta,
                    "relative_l2_difference": delta / max(l2, win) if max(l2, win) else 0.0,
                    "cpu_max_abs": max_abs,
                    "cpu_nonzero_coordinates": nonzero,
                    "within_numerical_check": delta <= limit,
                }
            entry.update(cpu_bce=actual, absolute_bce_difference=abs(actual - rec["bce"]), gradients=proofs)
        rows.append(entry)
    return {
        "first_step_replayed": 4,
        "original_witnesses_checked": 8,
        "witnesses": rows,
        "nonzero_minibatches_by_parameter": dict(observations),
        "within_numerical_check": all(v >= 3 for v in observations.values())
        and set(observations) == set(names)
        and all(
            r["absolute_bce_difference"] <= 0.0002
            and all(
                p["within_numerical_check"] and p["cpu_nonzero_coordinates"] > 0
                for p in r["gradients"].values()
            )
            for r in rows[:4]
        ),
        "rule": "CPU neutral FP32, absolute BCE difference at most 0.0002, L2 difference at most max(1e-6, 0.01 * max(CPU norm, recorded norm)).",
        "interpretation": "The first update's four independent view BCE gradients are reconstructed from original initialization and supervised data. The next four witnesses were recorded after one Windows AdamW update and checked against the original sample and optimizer schedule. Cross-device bitwise CUDA replication was not measured.",
    }
