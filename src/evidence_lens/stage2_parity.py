"""Formal Stage B checks for native deployment, fitted parameters and independent development selection."""

import hashlib
from collections import Counter
import re
import numpy as np

from .formal_inputs import require

CATALOG_SHA = "543adef7b3b3f0dbc11434738f5a7a6d4c6178622b63f452a5133eed6446dc79"
PROTOCOL_SHA = "ec5205a3022562899db81e2b6708617e37f5a87295b15feb2a813d52259dc00b"
SEEDS = (1729, 2718, 3141)
NEURAL = ("L2-frozen", "L2-vision-ft")
VIEWS = {f"v{n:02d}" for n in range(1, 7)}


def protocol(protocol, sha, study_sha, catalog_sha):
    require(
        (catalog_sha != CATALOG_SHA or sha == PROTOCOL_SHA)
        and protocol["schema"] == "el-stage2-cpu-parity-protocol-1"
        and protocol["training_protocol_sha256"] == study_sha
        and protocol["finite_support_must_match"] is True
        and protocol["minimum_common_finite_per_view"] == 1
        and protocol["fp32_max_absolute_score_difference"] == 1e-4
        and protocol["test_scoring_device"] == "macOS CPU",
        "different registered CPU inference parity protocol",
    )
    return sha


def parity(cost, protocol):
    require(set(cost) == set(NEURAL), "missing principal development CPU pair")
    checks = {}
    for m, c in cost.items():
        views = c["canonical_win_to_mac"]
        require(
            set(views) == VIEWS
            and all(
                r["support_disagreement"] == 0
                and r["common_finite"] >= protocol["minimum_common_finite_per_view"]
                and r["max_abs"] is not None
                and np.isfinite(r["max_abs"])
                and 0 <= r["max_abs"] <= protocol["fp32_max_absolute_score_difference"]
                and 0 <= r["threshold_crossings"] <= r["common_finite"]
                for r in views.values()
            ),
            "Mac/CUDA native support or registered tolerance differs: " + m,
        )
        checks[m] = {
            "verified_targets": len(views),
            "minimum_common_finite": min(r["common_finite"] for r in views.values()),
            "support_disagreement": 0,
            "maximum_absolute_error": max(r["max_abs"] for r in views.values()),
            "threshold_crossings": {v: views[v]["threshold_crossings"] for v in sorted(views)},
        }
    return checks


def registered_groups(branch):
    require(branch in NEURAL, "unexpected neural candidate")
    head = {n + "." + part for n in ("head.0", "head.2", "projection") for part in ("weight", "bias")}
    tail = set()
    if branch == NEURAL[1]:
        for block in (10, 11):
            prefix = f"backbone.blocks.{block}."
            tail.update(
                prefix + n + "." + part
                for n in ("attn.proj", "attn.qkv", "mlp.fc1", "mlp.fc2", "norm1", "norm2")
                for part in ("weight", "bias")
            )
            tail.update(prefix + n + ".gamma" for n in ("ls1", "ls2"))
        tail.update(("backbone.norm.weight", "backbone.norm.bias"))
    return head, tail


def trainable_parameters(branch, change, setup):
    head, tail = registered_groups(branch)
    allowed = head | tail
    groups = setup["optimizer"]["groups"]
    listed = {g["role"]: g["names"] for g in groups}
    expected = {"projection_head": head, **({"pretrained_suffix": tail} if tail else {})}
    require(
        set(change["trainable"]) == allowed
        and not (set(change["frozen_after"]) & allowed)
        and change["frozen_before"] == change["frozen_after"]
        and set(setup["initial_tensors"]) == allowed | set(change["frozen_after"])
        and len(listed) == len(groups) == len(expected)
        and set(listed) == set(expected)
        and all(len(listed[k]) == len(expected[k]) and set(listed[k]) == expected[k] for k in expected),
        "recorded trainable/frozen tensors or optimizer membership differ from the registered branch",
    )
    return {
        "count": len(allowed),
        "projection_and_head_count": len(head),
        "pretrained_count": len(tail),
        "names": sorted(allowed),
    }


def split_families(train, dev, catalog, original):
    def ids(cat, role):
        seq = [r["scene_family_id"] for r in cat["families"] if r["split"] == role]
        require(len(set(seq)) == len(seq), "duplicate registered family")
        return set(seq)

    def roster(ds):
        keys = [(r["family"], r["view"]) for r in ds["views"]]
        sizes = Counter(f for f, v in keys)
        require(
            len(set(keys)) == len(keys)
            and all(r["condition"] == "B0" and r["view"] in VIEWS for r in ds["views"])
            and all(n == 6 for n in sizes.values()),
            "real dataset has duplicated/missing target views",
        )
        return set(sizes)

    origins = {
        k: {role: ids(cat, role) for role in ("train", "development", "test")}
        for k, cat in (("new", catalog), ("original", original))
    }
    roles = [s for group in origins.values() for s in group.values()]
    require(
        sum(map(len, roles)) == len(set.union(*roles))
        and [[len(z[n]) for n in ("train", "development", "test")] for z in origins.values()]
        == [[12, 4, 8], [16, 4, 4]],
        "registered family populations overlap or counts differ",
    )
    require(
        train["split"] == "train"
        and dev["split"] == "development"
        and roster(train) == (origins["original"]["train"] | origins["new"]["train"])
        and roster(dev) == (origins["original"]["development"] | origins["new"]["development"]),
        "actual train/dev populations differ from frozen family rosters",
    )
    source_hashes = {}
    for name, ds in (("train", train), ("development", dev)):
        hashes = {}
        for row in ds["views"]:
            h = row.get("source_input_sha256")
            require(
                (h is None and row.get("status") != "completed")
                or (isinstance(h, str) and re.fullmatch("[0-9a-f]{64}", h)),
                "identified training/development source lacks its PNG checksum",
            )
            if h is not None:
                require(
                    hashes.setdefault(row["family"], h) == h, "B0 source changed within a learning family"
                )
        source_hashes[name] = sorted(set(hashes.values()))
    require(
        set(source_hashes["train"]).isdisjoint(source_hashes["development"]),
        "train/dev use identical source PNG bytes",
    )
    return {
        "source_input_sha256": source_hashes,
        "train": 28,
        "development": 8,
        "new_confirmation": 8,
        "historical_regression": 4,
        "pairwise_overlap": 0,
        "family_ids": {k: {r: sorted(s) for r, s in z.items()} for k, z in origins.items()},
    }


def supports(neural):
    masks = [np.isfinite(a) for a in neural.values()]
    require(
        len(masks) == 8 and all(np.array_equal(masks[0], other) for other in masks[1:]),
        "candidate native scoring support differs between the eight fits",
    )
    return hashlib.sha256(np.packbits(masks[0]).tobytes()).hexdigest()


def paired_domains(domains, expected_views=48):
    ordered = [
        {
            (r["family"], r["view"]): (r["mask_sha256"], r["candidate_support_sha256"], r["S7"], r["S7Y"])
            for r in rows
        }
        for rows in (domains.get(s, domains.get(str(s))) for s in SEEDS)
    ]
    require(
        all(
            len(o) == expected_views
            and all(
                re.fullmatch("[0-9a-f]{64}", x[0])
                and re.fullmatch("[0-9a-f]{64}", x[1])
                and 0 <= x[3] <= x[2]
                for x in o.values()
            )
            for o in ordered
        )
        and ordered[0] == ordered[1] == ordered[2],
        "the three paired seeds used different six-view development masks",
    )
    return expected_views


def independent_choice(individual, selection):
    chosen = {}
    for branch in NEURAL:
        ids = [i for i, r in individual.items() if r["branch"] == branch]
        a = [i for i in ids if individual[i]["seed"] == SEEDS[0]]
        require(len(ids) == 4 and len(a) == 2, "independent candidate roster omits a branch or seed-A rate")

        def key(i):
            r = individual[i]
            return (-r["ap"]["value"], r["backbone_lr"], r["head_lr"])

        winner = min(a, key=key)
        ref = individual[winner]
        chosen[branch] = {}
        require(
            winner == selection["chosen"][branch]["run_id"],
            "winner on independently reranked native development scores differs",
        )
        for seed in SEEDS:
            matching = [
                i
                for i in ids
                if individual[i]["seed"] == seed
                and all(individual[i][k] == ref[k] for k in ("backbone_lr", "head_lr"))
            ]
            require(len(matching) == 1, "seed repeat differs from independently selected rate")
            chosen[branch][str(seed)] = matching[0]
    return chosen


def original_scores(
    before_run, after_run, before, after, original_tau, final_tau, fit_sha, model_sha, qa, replay
):
    five = ("constant", "1-alpha", "L0", "L1-full", "L1-geometry")
    require(
        before["model_sha256"] == fit_sha
        and after["model_sha256"] == model_sha
        and before["method"] == after["method"] == "original-five"
        and before["dataset_sha256"] == after["dataset_sha256"]
        and original_tau["method_sources"]["original-five"]["model_sha256"] == fit_sha
        and final_tau["method_sources"]["original-five"]["model_sha256"] == model_sha
        and original_tau["thresholds"] == final_tau["thresholds"]
        and original_tau["choices"] == final_tau["choices"]
        and set(original_tau["choices"]) == set(five)
        and original_tau["thresholds"]["L0"] == 0.5,
        "five operating points or original producer identity changed on export",
    )

    def keyed(rows):
        return {(r["family"], r["condition"], r["view"]): r for r in rows}

    a, b = keyed(before["views"]), keyed(after["views"])
    require(
        len(a) == len(b) == len(before["views"]) == len(after["views"]) == 48 and set(a) == set(b),
        "original-five baseline native rows differ",
    )
    matched = {}
    stats = 0
    for (f, c, v), x in a.items():
        y = b[f, c, v]
        require(c == "B0" and x == y, "original-five score bytes or inputs differ for " + f + "/" + v)
        if x["score"] is not None:
            member = x["score"]
            rel = "payload/" + member["path"]
            for run in (before_run, after_run):
                path = run.open(rel)
                require(
                    run.files[rel]["sha256"] == member["sha256"] and path.stat().st_size == member["bytes"],
                    "native score file differs from original receipt",
                )
            matched[f + "/" + v] = member["sha256"]
        else:
            matched[f + "/" + v] = None
        for method in five:
            for domain in ("own", "five_common"):
                prior = qa["views"][f + "/" + v]["ranks"][method][domain]
                fresh = replay["primary_domains"][method][domain]["views"][f][v]
                ap, ref = fresh["ap"], prior["ap"]
                require(
                    fresh["n"] == prior["n"]
                    and fresh["positive"] == prior["positive"]
                    and (
                        (ap is None and ref is None)
                        or (ap is not None and ref is not None and abs(ap - ref) <= 1e-12)
                    ),
                    "re-export changed native-domain score counts or AP",
                )
                stats += 1
    return {
        "targets": len(matched),
        "per_target_identical_score_npz_sha256": matched,
        "own_and_five_common_view_method_comparisons": stats,
        "five_thresholds": original_tau["thresholds"],
    }
