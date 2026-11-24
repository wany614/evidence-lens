"""Project all five registered coverage budgets on the fixed authored B confirmation view."""

import json
from zipfile import ZipFile

import numpy as np
from PIL import Image

from evidence_lens._verify import require
from evidence_lens_bundle.bindings import array_sha
from foundation_case_io import (ATTRIBUTION, GALLERY, REASON_COLORS, arrays, categorical_counts, checked_zip,
                                decode_png, digest, encode_png, evaluation, legend, maps, rgb8)

VIEW = "h00"
BUDGETS = (4, 8, 16, 32, 128)
STATE_NAMES = ("selected_subset_only", "subset_comparability_gap", "audited_library_no_comparable_evidence",
               "library_incomplete_or_unavailable", "geometry_not_evaluable", "audited_comparable_evidence")
STATE_COLORS = ("#d1deeb", "#e2903f", "#705782", "#808080", "#3f4348", "#5c9a9c")


def mask_id(mask):
    return digest(np.asarray(mask.shape, dtype="<i8").tobytes() + mask.astype(np.uint8).tobytes())


def gallery_sources(inputs, pub, bank):
    native = inputs["gallery_source"]
    source = native.metadata
    original = source["original_supplement"]
    package = source["packages"]["R128"]
    require(source["H_photo_decodes"] == 0 and source["permission"]["license"].startswith("CC BY 4.0, wany614")
            and source["scene_id"] == GALLERY, "unexpected authored source or rights")
    rows = []
    with ZipFile(native.path(original["path"], original)) as originals, ZipFile(native.path(package["path"], package)) as analysis:
        metadata = json.loads(analysis.read("scene.json"))
        require(metadata["scene_id"] == GALLERY and metadata["R"] == metadata["A"] == bank["accessible_ids"]
                and metadata["H"] == bank["target_ids"], "signed R-only source and score bank differ")
        for sid in metadata["R"]:
            bank_row = bank["sources"][sid]
            deployed = inputs["gallery_bank"].read(bank_row["photo"])
            data = metadata["source_rgb"][sid]
            require(deployed == checked_zip(analysis, data["path"], data), "source photo differs from the native bank")
            spec = source["originals"][sid]
            require(spec["sha256"] == bank_row["original_sha256"] == metadata["original_rgb"][sid]["sha256"],
                    "incorrect physical source photo identity")
            raw = checked_zip(originals, spec["path"], spec)
            whole = Image.fromarray(decode_png(raw))
            require(whole.size == tuple(bank["grids"]["views"][sid]["original_size"])
                    and bank["grids"]["views"][sid]["crop_xywh"] == [0, 0, *whole.size]
                    and np.array_equal(decode_png(deployed), np.asarray(whole.resize((378, 252), Image.Resampling.LANCZOS))),
                    "original source does not reproduce the bank pixels")
            rows.append(pub.source(GALLERY, sid, deployed, raw, 144))
    return rows


def gallery_reference(inputs, pub, pred, expected):
    native = inputs["gallery_truth"]
    t = native.metadata
    opening = inputs["gallery_opening"]
    grant = opening.metadata
    terminal = grant["terminals"][GALLERY]["128"]
    require(t["scene_id"] == GALLERY and t["split"] == "confirmation" and VIEW in grant["allowed_targets"][GALLERY]
            and t["opening_sha256"] == opening.pin["sha256"] and terminal["status"] == "succeeded"
            and terminal["receipt"] == inputs["gallery_scores"].pin, "incorrect confirmation opening or score receipt")
    pkg = t["archive"]
    fields = t["targets"][VIEW]
    with ZipFile(native.path(pkg["path"], pkg)) as z:
        raw = checked_zip(z, fields["analysis_rgb"]["path"], fields["analysis_rgb"])
        full = checked_zip(z, fields["original_rgb"]["path"], fields["original_rgb"])
    require(fields["analysis_rgb"]["sha256"] == expected["source_reference_sha256"]
            and np.array_equal(decode_png(raw), np.asarray(Image.fromarray(decode_png(full)).resize(
                (378, 252), Image.Resampling.LANCZOS))), "native independent original and analysis photographs disagree")
    original = pub.asset(f"{GALLERY}/evaluation/{VIEW}-original.png", full, "independent evaluation only, original RGB8")
    entry, valid, high = evaluation(pub, GALLERY, VIEW, pred, decode_png(raw).astype(np.float32) / np.float32(255),
                                    raw, original=original,
                                    origin="Held-out authored camera. Independent analysis RGB8 photograph decoded on its registered grid and divided by FP32(255).")
    arith = next(c for c in inputs["gallery_arithmetic"].metadata["cases"] if c["family"] == GALLERY and c["R"] == 128)
    require(arith["target"] == VIEW and arith["matched_native_numbers_and_hashes"]
            and arith["labels_sha256"] == expected["labels_sha256"] == mask_id(high)
            and arith["valid_labels_sha256"] == expected["valid_labels_sha256"] == mask_id(valid)
            and all(entry["counts"][k] == expected[k] == arith[k] for k in ("reference", "labelled", "errors"))
            and abs(entry["counts"]["mean_rgb_mae"] - expected["mae_sum"] / expected["labelled"]) < 1e-12,
            "independent error denominator or pixel identity differs from the registered evaluation")
    entry.update(registered_labels_sha256=mask_id(high), registered_valid_labels_sha256=mask_id(valid),
                 p2_same_query=arith["P2"], p1=arith["P1"], opening=opening.pin,
                 correct_gap_by_budget={str(k): {f: expected["mechanism"][f"coverage-{k}"][f]
                                           for f in ("all_reference_correct", "correct_with_gap", "correct_gap_rate")}
                                        for k in BUDGETS})
    return entry, valid, high


def gallery_outputs(inputs, pub):
    b = inputs["gallery_bank"].metadata
    scores = inputs["gallery_scores"].metadata
    official = inputs["gallery_evaluation"].metadata
    freeze = inputs["gallery_freeze"].metadata
    require(b["scene_id"] == scores["family"] == official["scene"] == GALLERY
            and b["accessible_ids"] == b["reconstruction_ids"] == scores["source_ids"]
            and 128 == scores["A"] == scores["R"] == len(b["accessible_ids"])
            and b["grids"]["profile_id"] == "analysis-378x252-v1" and scores["references_accessed"] == 0
            and inputs["gallery_bank"].pin["sha256"] == scores["bank_sha256"] == official["bank_sha256"]
            and inputs["gallery_scores"].pin["sha256"] == official["scores_sha256"]
            and inputs["gallery_truth"].pin["sha256"] == official["truth_sha256"]
            and freeze["final_model"]["sha256"] == scores["models"]["B-CPU-LR"], "B case lineage mismatch")
    sources = gallery_sources(inputs, pub, b)
    d = b["targets"][VIEW]["render"]
    rec = scores["views"][VIEW]
    expected = official["views"][VIEW]
    require(d["sha256"] == rec["prediction"] == expected["prediction_sha256"] and rec["status"] == "ready",
            "B fixed prediction or checkpoint changed across the evaluation")
    raw = inputs["gallery_bank"].read(d)
    pred = arrays(raw)["rgb"]
    require(pred.shape == (252, 378, 3) and pred.dtype == np.float32, "incorrect B prediction")
    prediction = pub.asset(f"{GALLERY}/predictions/{VIEW}.npz", raw, "sealed native RGB, alpha and camera-Z")
    target = pub.png(f"{GALLERY}/targets/{VIEW}.png", encode_png(rgb8(pred)))
    measure, valid, high = gallery_reference(inputs, pub, pred, expected)
    snapshots = {}
    for k in BUDGETS:
        cell = rec["cells"][f"coverage-{k}"]
        xr = inputs["gallery_scores"].path(cell["file"], cell).read_bytes()
        require(cell["actual_k"] == len(set(cell["queries"])) == k and cell["strategy"] == "coverage"
                and cell["nominal_budgets"] == (["all"] if k == 128 else [k])
                and cell["queries"] == expected["queries"][f"coverage-{k}"]["source_ids"],
                "actual frozen B queries or requested budget differ")
        p = f"{GALLERY}/models/{VIEW}-coverage-{k}"
        native = pub.asset(p + ".npz", xr, "native checkpoint ranks, counts, reasons and per-pixel first witness")
        snapshots[k] = (arrays(xr), cell, p, native)
    full, full_cell, _, _ = snapshots[128]
    require(set(full_cell["queries"]) == set(b["accessible_ids"]) and not full["incomplete"].any()
            and full_cell["query_cost"]["source_geometry_projections"] == 128, "incomplete full-library audit")
    conditions = []
    previous_q = []
    previous_has = None
    first_has = snapshots[4][0]["comparable"] > 0
    for k in BUDGETS:
        a, cell, path, native = snapshots[k]
        q = cell["queries"]
        require(q[:len(previous_q)] == previous_q and set(q) <= set(b["accessible_ids"]), "non-nested budget")
        has = a["comparable"] > 0
        cond = budget_condition(pub, path, native, a, full, cell, expected["mechanism"][f"coverage-{k}"], b,
                                scores, inputs, valid, high)
        require(previous_has is None or not np.any(previous_has & ~has), "adding sources lost comparable evidence")
        cond["counts"]["new_comparable_since_q4"] = int((has & ~first_has).sum())
        cond["counts"]["new_comparable_since_previous"] = int((has & ~previous_has).sum()) if previous_has is not None else 0
        cond["new_query_ids"] = q[len(previous_q):]
        previous_q, previous_has = q, has
        conditions.append(cond)
    require(all(c["target_prediction_sha256"] == prediction["sha256"] for c in conditions),
            "budget switch did not preserve the same native bytes")
    view = dict(id=VIEW, target=target, target_prediction_sha256=prediction["sha256"],
                target_rgb_sha256=digest(pred.tobytes()), target_rgb_typed_sha256=array_sha(pred),
                native_prediction=prediction, identical_prediction=True, conditions=conditions, evaluation=measure)
    return dict(id=GALLERY, title="Perforated Gallery", rights=ATTRIBUTION,
                size=[378, 252], analysis_profile=b["grids"]["profile_id"], available_count=128, reconstruction_count=128,
                source_processing="128 registered authored source observations. Whole 756 × 504 camera frames reduced to 378 × 252 with Lanczos; the original and analysis PNGs are indexed separately.",
                source_ids=b["accessible_ids"], sources=sources, views=[view],
                renderer=b["renderer_id"], checkpoint_sha256=b["checkpoint_sha256"],
                provenance={**{key: inputs[key].pin for key in ("gallery_source", "gallery_bank", "gallery_scores",
                    "gallery_freeze", "gallery_opening", "gallery_truth", "gallery_evaluation", "gallery_arithmetic")},
                    "upstream_native_fit": b["provenance"]["native_run_id"], "final_ranking_model": freeze["final_model"]})


def budget_condition(pub, path, native, a, full, cell, expected, bank, scores, inputs, valid, high):
    rank = a["rank_B-CPU-LR"]
    finite = np.isfinite(rank)
    why = a["reason"]
    require(rank.shape == valid.shape and np.array_equal(finite, a["finite"]) and np.array_equal(finite, why == 0),
            "raw rank and reason support disagree")
    q = cell["queries"]
    has = a["comparable"] > 0
    all_has = full["comparable"] > 0
    incomplete = a["incomplete"] | full["incomplete"]
    bad = a["invalid_geometry"] | full["invalid_geometry"]
    require(not (has & ~all_has).any(), "full bank lost a query witness")
    state = np.where(all_has, np.where(has, 5, 1), 2).astype(np.uint8)
    state[incomplete] = 3
    state[bad] = 4
    gap = state == 1
    ids = bank["accessible_ids"]
    indices, freq = np.unique(full["witness"][gap], return_counts=True)
    require(all(0 <= ix < len(ids) and ids[ix] not in q for ix in indices), "invalid outside-query witness")
    witnesses = {str(int(ix)): int(n) for ix, n in zip(indices, freq, strict=True)}
    states = legend(STATE_NAMES, STATE_COLORS)
    reasons = legend(scores["reason_names"], REASON_COLORS[:len(scores["reason_names"])])
    counts = dict(canvas=rank.size, finite=int(finite.sum()), unknown=int((~finite).sum()), low=None, high=None,
                  reference=int(valid.size), query_has_comparable=int(has.sum()), selected_gap=int(gap.sum()),
                  library_no_comparable=int((state == 2).sum()), library_incomplete=int((state == 3).sum()),
                  invalid_geometry=int((state == 4).sum()), query_contains_supported=int((a["support"] > 0).sum()),
                  depth_conflicts=int((a["depth_conflict"] > 0).sum()), colour_conflicts=int((a["colour_conflict"] > 0).sum()),
                  reason_counts=categorical_counts(why, reasons), state_counts=categorical_counts(state, states))
    names = ("query_has_comparable", "query_contains_supported", "selected_gap", "library_no_comparable",
             "library_incomplete", "invalid_geometry", "depth_conflicts", "colour_conflicts")
    require(all(expected[k] == counts[k] for k in names) and expected["gap_witness_sources"] == witnesses
            and expected["correct_with_gap"] == int((gap & valid & ~high).sum()) and expected["audit_complete"],
            "independent native count or per-source witness discrepancy")
    return dict(id="coverage-" + str(len(q)), method="B-CPU-LR", k=len(q), requested_k=cell["nominal_budgets"][0],
                query_ids=q, actual_q=q, reconstruction_count=128, available_count=128, full_bank_audited=True,
                grain="one native analysis pixel", threshold=None, threshold_status="uncalibrated rank; no low/high operating threshold",
                target_prediction_sha256=scores["views"][VIEW]["prediction"], counts=counts,
                gap_witness_source_counts={ids[int(ix)]: n for ix, n in witnesses.items()},
                query_cost=cell["query_cost"], native_arrays=native,
                provenance=dict(model_sha256=scores["models"]["B-CPU-LR"], checkpoint_sha256=bank["checkpoint_sha256"],
                                scores_run=inputs["gallery_scores"].pin["run_id"], native_cell_sha256=cell["sha256"],
                                full_audit_cell_sha256=scores["views"][VIEW]["cells"]["coverage-128"]["sha256"],
                                array_digests={key: array_sha(a[key]) for key in ("rank_B-CPU-LR", "finite", "reason", "comparable", "incomplete", "invalid_geometry", "witness")},
                                scope_state_derived_sha256=array_sha(state), gap_witness_array_sha256=array_sha(full["witness"])),
                **maps(pub, path, rank, state, why, states, reasons))
