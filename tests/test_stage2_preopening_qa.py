import copy
import itertools
import numpy as np
import pytest
from evidence_lens import stage2_parity as checks
from vision_sources import paired_render_sources
from vision_evaluate import read_scores


def test_protocol_masks_and_crossings():
    study = "2" * 64
    proto = {
        "schema": "el-stage2-cpu-parity-protocol-1",
        "training_protocol_sha256": study,
        "finite_support_must_match": True,
        "minimum_common_finite_per_view": 1,
        "fp32_max_absolute_score_difference": 0.0001,
        "test_scoring_device": "macOS CPU",
    }
    with pytest.raises(ValueError, match="protocol"):
        checks.protocol(proto, "a" * 64, study, checks.CATALOG_SHA)
    measurement = {"support_disagreement": 0, "common_finite": 21, "max_abs": 1e-4, "threshold_crossings": 2}
    costs = {m: {"canonical_win_to_mac": {v: dict(measurement) for v in checks.VIEWS}} for m in checks.NEURAL}
    out = checks.parity(costs, proto)
    assert out["L2-vision-ft"]["threshold_crossings"]["v04"] == 2
    for field, bad in (("support_disagreement", 1), ("common_finite", 0), ("max_abs", 0.000100001)):
        c = copy.deepcopy(costs)
        c["L2-frozen"]["canonical_win_to_mac"]["v01"][field] = bad
        with pytest.raises(ValueError, match="support or registered"):
            checks.parity(c, proto)
    grids = {i: np.array([[0.1, np.nan], [0.3, 0.4]], np.float32) for i in range(8)}
    assert len(checks.supports(grids)) == 64
    grids[6][0, 1] = 0.5
    with pytest.raises(ValueError, match="candidate native scoring support"):
        checks.supports(grids)
    rows = [
        {
            "family": str(f),
            "view": v,
            "mask_sha256": "e" * 64,
            "candidate_support_sha256": "6" * 64,
            "S7": 6,
            "S7Y": 2,
        }
        for f, v in itertools.product(range(8), checks.VIEWS)
    ]
    by = {s: copy.deepcopy(rows) for s in checks.SEEDS}
    assert checks.paired_domains(by) == 48
    by[3141][3]["mask_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="three paired seeds"):
        checks.paired_domains(by)


def test_trainable_and_split_membership():
    head = {n + "." + x for n in ("projection", "head.0", "head.2") for x in ("weight", "bias")}
    change = {
        "trainable": dict.fromkeys(head),
        "frozen_after": {"backbone.norm.bias": 1},
        "frozen_before": {"backbone.norm.bias": 1},
    }
    setup = {
        "optimizer": {"groups": [{"role": "projection_head", "names": sorted(head)}]},
        "initial_tensors": dict.fromkeys(head | {"backbone.norm.bias"}),
    }
    assert checks.trainable_parameters("L2-frozen", change, setup)["count"] == 6
    change["trainable"]["backbone.norm.bias"] = 1
    with pytest.raises(ValueError, match="optimizer membership"):
        checks.trainable_parameters("L2-frozen", change, setup)

    def cat(n, prefix):
        return {
            "families": [
                {"scene_family_id": prefix + role + str(i), "split": role}
                for role, len_ in zip(("train", "development", "test"), n)
                for i in range(len_)
            ]
        }

    new, old = cat((12, 4, 8), "new"), cat((16, 4, 4), "old")

    def ds(role):
        return {
            "split": role,
            "views": [
                {"family": f["scene_family_id"], "condition": "B0", "view": v}
                for f in new["families"] + old["families"]
                if f["split"] == role
                for v in checks.VIEWS
            ],
        }

    assert checks.split_families(ds("train"), ds("development"), new, old)["pairwise_overlap"] == 0
    new["families"][-1]["scene_family_id"] = old["families"][0]["scene_family_id"]
    with pytest.raises(ValueError, match="populations overlap"):
        checks.split_families(ds("train"), ds("development"), new, old)


def test_insert_a_b0_render_or_its_score_row_in_r1_is_rejected():
    def pin(n):
        return {"run_id": "20000101T000000000000Z-" + str(n) * 16, "manifest_sha256": str(n) * 64}

    def cell(n, condition, source, cam):
        return {
            "condition": condition,
            "input_sha256": source * 64,
            "camera_sha256": cam * 64,
            "parents": {"prediction": pin(n), "reference": pin(9)},
        }

    base = cell(1, "B0", "a", "b")
    r = cell(2, "R1", "c", "d")
    q = cell(3, "Q1", "a", "b")
    paired_render_sources([base, r, q])
    with pytest.raises(ValueError, match="cannot use the original B0 render"):
        paired_render_sources([base, {**r, "parents": base["parents"]}, q])
    with pytest.raises(ValueError, match="wrong input image or calibration"):
        paired_render_sources([base, {**r, "input_sha256": base["input_sha256"]}, q])
    row = {
        "family": "a",
        "view": "v01",
        "condition": "R1",
        "height": 2,
        "width": 2,
        "source_input_sha256": "c" * 64,
        "source_camera_sha256": "d" * 64,
        "source_prediction_sha256": "5" * 64,
        "target_prediction_sha256": "6" * 64,
        "render_pin": pin(2),
        "rgb": None,
        "full": {"deploy": None},
    }
    injected = {**row, "source_input_sha256": "a" * 64, "render_pin": pin(1)}
    with pytest.raises(ValueError, match="condition render differs"):
        read_scores(row, {"original-five": (None, {("a", "R1", "v01"): injected}, None)})


def test_independently_ranked_a_uses_lower_learning_rate_on_a_tie():
    individual = {}
    sel = {"chosen": {}}
    for m in checks.NEURAL:
        rates = [(0.0, 1e-4), (0.0, 3e-4)] if m == "L2-frozen" else [(1e-5, 3e-4), (3e-5, 3e-4)]
        all_rates = [rates[0], rates[1], rates[0], rates[0]]
        for i, (seed, rate) in enumerate(zip((1729, 1729, 2718, 3141), all_rates)):
            individual[m + str(i)] = {
                "branch": m,
                "seed": seed,
                "head_lr": rate[1],
                "backbone_lr": rate[0],
                "ap": {"value": 0.5},
            }
        sel["chosen"][m] = {"run_id": m + "0"}
    result = checks.independent_choice(individual, sel)
    assert result["L2-frozen"] == {"1729": "L2-frozen0", "2718": "L2-frozen2", "3141": "L2-frozen3"}
    individual["L2-frozen1"]["ap"]["value"] = 0.500001
    with pytest.raises(ValueError, match="independently reranked"):
        checks.independent_choice(individual, sel)
