"""Replay native RGB, supervision and truth-isolated features for every registered Stage C train/dev target."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from evidence_lens._bundle import load
from evidence_lens.formal_inputs import compact_pin, require
from evidence_lens.multisource import COLUMNS, CONDITIONS, TARGETS
from evidence_lens.multisource_io import Case
from evidence_lens.scoring_io import file_hash, new_output, write_json
from stage3_risk_sources import CATALOG, STUDY, TRANSFER
from vision_sources import find_run, json_member, member, signed

B_TRAIN = {
    "run_id": "20261123T130016909864Z-261c749e33704daf",
    "manifest_sha256": "ea0d00e06d3510924e4f98465b93ae02d3b56eafea1f23851785ccd053bf99ba",
}
B_L1 = {
    "run_id": "20261123T130437328225Z-b36e63e03b88475b",
    "manifest_sha256": "1cca54bb3876f56019efb819370345ea54453831280bdd93ce1feb6f3e4daf0d",
}
B_FT = {
    "run_id": "20261123T133146588950Z-c1d19dcb78c74cf0",
    "manifest_sha256": "0100b3f27e79c4498d51854d249312fc54cd7c80291b812e68f42c1dbfd88583",
}


def bits(a, b, role):
    require(
        a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order="C") == b.tobytes(order="C"),
        "different original bit pattern: " + role,
    )


def array(p):
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=False)
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def independent_y(ref, pred):
    v = np.isfinite(ref).all(axis=2)
    e = np.zeros_like(v) if pred is None else v & np.isfinite(pred).all(axis=2)
    y = np.full(v.shape, 255, np.uint8)
    if pred is not None:
        delta = np.abs(pred[e] - ref[e])
        # Three original display-sRGB FP32 channels, in camera/array order.
        y[e] = (delta[:, 0] + delta[:, 1] + delta[:, 2]) / np.float32(3) > np.float32(0.05)
    return v, e, y


def indices(mask, family, condition, view):
    eligible = np.flatnonzero(mask)
    if len(eligible) > 4096:
        raw = ("9184\n" + CATALOG + "\n" + family + "\n" + condition + "/" + view).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(raw).digest()[:16], "big")
        eligible = np.sort(np.random.Generator(np.random.PCG64(seed)).choice(eligible, 4096, replace=False))
    return eligible.astype(np.int32)


def packed(run, spec):
    return array(member(run, "payload/" + spec["path"], spec)[0])


def replay_view(dr, ref, ren, case, row, split):
    family, condition, view = (row[k] for k in ("family", "condition", "view"))
    name = f"payload/reference/{view}.npz"
    rp, refspec = member(ref, name, row["reference_member_sha256"])
    with np.load(rp, allow_pickle=False) as z:
        original_reference = z["rgb"]
    require(
        original_reference.shape == (256, 256, 3) and original_reference.dtype == np.float32,
        "wrong native reference format",
    )
    ready = row["status"] == "ready"
    pred = None
    proof = {
        "split": split,
        "family": family,
        "condition": condition,
        "view": view,
        "status": row["status"],
        "reference_sha256": refspec["sha256"],
        "truth_sha256": row["truth"]["sha256"],
        "samples": 0,
        "positive_samples": 0,
        "rgb_channel_values_bitwise_verified": 0,
        "features_replayed_without_truth": 0,
    }
    if ready:
        pp, ps = member(ren, f"payload/{view}.npz", row["prediction_member_sha256"])
        with np.load(pp, allow_pickle=False) as z:
            require(set(z.files) == {"rgb", "z", "alpha"}, "original renderer exposed undeclared fields")
            pred = z["rgb"]
        bits(packed(dr, row["rgb"]), pred, "published training/development RGB versus original renderer")
        proof.update(
            prediction_sha256=ps["sha256"],
            dataset_rgb_sha256=row["rgb"]["sha256"],
            rgb_channel_values_bitwise_verified=int(pred.size),
        )
    else:
        require(row["status"] == "render_failed" and row["rgb"] is None, "failed frame has training RGB")
    t = packed(dr, row["truth"])
    v, e, y = independent_y(original_reference, pred)
    for k, a in (("reference_valid", v), ("rgb_evaluable", e), ("rgb_high_error", y)):
        bits(a, t[k], k)
    counts = {"V": int(v.sum()), "E": int(e.sum()), "Y": int((y == 1).sum())}
    require(
        not (e & ~v).any() and all(row[k] == value for k, value in counts.items()),
        "V/E/Y denominator changed",
    )
    proof.update(counts)
    if not ready:
        return proof
    if split == "development":
        bits(packed(dr, row["reference_rgb"]), original_reference, "independent evaluation reference")
    else:
        require(row["reference_rgb"] is None, "training row carries an evaluation-only reference image")
    for scope in ("all",) if split == "train" else ("one", "two", "all"):
        spec = row["modes"][scope]
        if scope == "all" and split == "development" and condition == "M2":
            require(spec == {**row["modes"]["two"], "alias_of": "two"}, "two/all alias differs")
            continue
        native_rgb, features = case.features(view, scope)
        bits(native_rgb, pred, "truth-isolated public feature entry point RGB")
        x, scoreable = features["X"], features["scoreable"]
        require(
            x.dtype == np.float32 and x.shape == (256, 256, len(COLUMNS)) and scoreable.dtype == bool,
            "wrong deployable input layout",
        )
        if split == "development":
            stored = packed(dr, spec["deploy"])
            require(set(features) == set(stored), "additional annotated channel inside deployment features")
            for k, a in features.items():
                bits(a, stored[k], k + " without reference")
            proof["features_replayed_without_truth"] += int(x.size)
        else:
            require(spec["deploy"] is None, "expected bounded training transport")
            data = packed(dr, spec["samples"])
            require(set(data) == {"X", "Y", "index", "inclusion_probability"}, "unexpected training columns")
            mask = e & scoreable
            ix = indices(mask, family, condition, view)
            bits(ix, data["index"], "independent bounded sample")
            bits(data["X"], x.reshape(-1, len(COLUMNS))[ix], "sampled deployable features without truth")
            bits(data["Y"], y.ravel()[ix], "original supervision labels")
            require(
                data["inclusion_probability"].item() == (len(ix) / int(mask.sum()) if mask.any() else 0)
                and spec["eligible"] == int(mask.sum()),
                "sample population differs",
            )
            proof.update(
                samples=len(ix),
                positive_samples=int(data["Y"].sum()),
                eligible=int(mask.sum()),
                sample_sha256=spec["samples"]["sha256"],
                features_replayed_without_truth=int(data["X"].size),
            )
    return proof


def ancestors(roots, starts, catalog, registration):
    # Earlier original models/datasets are a temporal boundary. Every Stage C node after registration is traversed.
    boundary = find_run(roots, registration).run["started_at_utc"]
    forbidden = set(catalog["test_families"])
    work = list(starts)
    seen = {}
    while work:
        p = compact_pin(work.pop())
        key = p["run_id"]
        if key in seen:
            require(p["manifest_sha256"] == seen[key]["manifest_sha256"], "multiple seals for one ancestor")
            continue
        r = find_run(roots, p)
        ids = [r.run.get(k) or "" for k in ("scene_family_id", "scene_id", "trial_id")]
        require(
            not any(f in part for f in forbidden for part in ids),
            "C held-out acquisition in training ancestry",
        )
        earlier = r.run["ended_at_utc"] < boundary
        require(not earlier or r.run["dataset_hash"] != CATALOG, "C experiment predates C registration")
        direct = [] if earlier else load(r.open("inputs.json"))["parents"]
        seen[key] = {
            **p,
            "run_kind": r.run["run_kind"],
            "experiment_id": r.run["experiment_id"],
            "scene_family_id": r.run.get("scene_family_id"),
            "ended_at_utc": r.run["ended_at_utc"],
            "pre_registration_boundary": earlier,
            "parent_count_traversed": len(direct),
        }
        work.extend(direct)
    return sorted(seen.values(), key=lambda j: j["run_id"])


def audit(roots, req, output, run_id):
    require(
        req["schema"] == "el-stage3-train-input-qa-request-1"
        and req["catalog_sha256"] == CATALOG
        and set(req["datasets"]) == {"train", "development"},
        "only the registered preopening inputs are admitted",
    )
    original_reg = find_run(roots, req["registration"])
    catalog, _ = json_member(original_reg, "payload/catalog.json", CATALOG)
    groups = {
        s: [r["scene_family_id"] for r in catalog["families"] if r["split"] == s]
        for s in ("train", "development", "test")
    }
    require(
        [len(v) for v in groups.values()] == [8, 4, 4] and len(set(sum(groups.values(), []))) == 16,
        "C registered family sets overlap",
    )
    bd = find_run(roots, B_TRAIN)
    bfit = find_run(roots, B_L1)
    bft = find_run(roots, B_FT)
    bdata, bs = json_member(bd, "payload/dataset.json")
    bj, _ = json_member(bfit, "payload/fit.json")
    bf, _ = json_member(bft, "payload/training.json")
    b = sorted({r["family"] for r in bdata["views"]})
    require(
        bdata["split"] == "train"
        and set(b) == set(bj["families"])
        and len(b) == 28
        and bs["sha256"] == bj["training_dataset_sha256"] == bf["training_dataset_sha256"]
        and not set(b).intersection(sum(groups.values(), [])),
        "B training cohort or encoder source overlaps C",
    )
    ri = find_run(roots, TRANSFER)
    require(B_L1 in load(ri.open("inputs.json"))["parents"], "wrong Stage B transfer fit")
    cfit = find_run(roots, req["fit"])
    fj, _ = json_member(cfit, "payload/fit.json")
    rows, totals, cells = [], {}, {}
    for split, fams in list(groups.items())[:2]:
        spec = req["datasets"][split]
        dr = find_run(roots, spec)
        d, ds = json_member(dr, "payload/dataset.json", spec["sha256"])
        require(
            d["split"] == split
            and d["catalog_sha256"] == CATALOG
            and d["study_sha256"] == STUDY
            and d["cohort"] == "formal"
            and d["families"] == fams,
            "unregistered learning run",
        )
        if split == "train":
            require(
                fj["training_dataset_sha256"] == ds["sha256"]
                and dr.pin["run_id"] in cfit.run["parent_run_ids"],
                "selected C fitted model uses another training dataset",
            )
        by = {(c["family"], c["condition"]): c for c in d["cells"]}
        keyed = {(r["family"], r["condition"], r["view"]): r for r in d["views"]}
        require(
            len(by) == len(d["cells"]) == len(fams) * 2 and len(keyed) == len(d["views"]) == len(fams) * 12,
            "duplicated or incomplete learning cohort",
        )
        splitrows = []
        for f in fams:
            for c in CONDITIONS:
                cell = by[f, c]
                ref = find_run(roots, cell["selected"]["reference"])
                case = None
                ren = None
                if not cell.get("failure"):
                    ren = find_run(roots, cell["selected"]["render"])
                    cp = member(dr, "payload/" + cell["case"]["path"], cell["case"])[0]
                    case = Case(cp.parent, cell["case"]["sha256"])
                    allowed = {
                        "case.json",
                        "source_cameras.json",
                        "render_cameras.json",
                        "prediction.json",
                        *(x["path"] for x in case.meta["source_images"].values()),
                        *(x["path"] for x in case.meta["renders"].values()),
                    }
                    pref = cp.parent.relative_to(dr.root).as_posix() + "/"
                    require(
                        {name[len(pref) :] for name in dr.files if name.startswith(pref)} == allowed,
                        "deployment case contains an unapproved reference or annotation member",
                    )
                for v in TARGETS:
                    splitrows.append(replay_view(dr, ref, ren, case, keyed[f, c, v], split))
        rows.extend(splitrows)
        totals[split] = {
            k: sum(r[k] for r in splitrows)
            for k in (
                "V",
                "E",
                "Y",
                "samples",
                "positive_samples",
                "rgb_channel_values_bitwise_verified",
                "features_replayed_without_truth",
            )
        }
        totals[split].update(
            families=len(fams),
            family_conditions=len(by),
            planned_targets=len(splitrows),
            ready_targets=sum(r["status"] == "ready" for r in splitrows),
        )
        cells[split] = [
            {
                "family": f,
                "condition": c,
                "planned": len(TARGETS),
                "supervised_rows": sum(
                    r["samples"] for r in splitrows if (r["family"], r["condition"]) == (f, c)
                ),
            }
            for f, c in by
        ]
    graph = ancestors(
        roots,
        [req["fit"], req["baseline_development"], *req["datasets"].values(), *req.get("additional_fits", [])],
        catalog,
        req["registration"],
    )
    require(sum(r["samples"] for r in rows) == fj["parity_rows"], "fit parity sample count differs")
    result = {
        "schema": "el-stage3-train-input-qa-1",
        "run_id": run_id,
        "catalog_sha256": CATALOG,
        "study_sha256": STUDY,
        "passed": True,
        "datasets": req["datasets"],
        "supervised_fit": req["fit"],
        "baseline_development": req["baseline_development"],
        "additional_fits": req.get("additional_fits", []),
        "totals": totals,
        "family_conditions": cells,
        "registered_splits": groups,
        "b_training": {
            "families": b,
            "dataset": B_TRAIN,
            "dataset_sha256": bs["sha256"],
            "logistic_fit": B_L1,
            "encoder_fit": B_FT,
            "transfer": TRANSFER,
        },
        "ancestry": {
            "verified_nodes": len(graph),
            "earlier_originals": sum(r["pre_registration_boundary"] for r in graph),
            "heldout_acquisitions": 0,
            "boundary": "Traverse Stage C parent seals recursively. Earlier originals ended before Stage C registration; the two B fits independently match the same 28-family B training dataset.",
        },
        "deployment_readback": "The case allowlist admits source PNGs, calibrations and producer RGB/Z/alpha. Features and raw prediction RGB were independently replayed from this case. Original reference RGB is accessed separately for labels.",
        "observations": rows,
        "ancestors": graph,
    }
    with new_output(output) as dst:
        write_json(dst / "input-qa.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("store", "shelf", "request", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--request-sha256", required=True)
    a = p.parse_args()
    q = audit((a.store, a.shelf), signed(a.request, a.request_sha256), a.output, os.environ["EL_RUN_ID"])
    print(json.dumps(q["totals"]))
