"""Repeat the registered P1/P2 arithmetic directly from physical prediction, independent reference and rank files."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image

from dual_records import resolve, selected, stage
from dtu_eval import read_pinned
from evidence_lens._bundle import load, save
from evidence_lens._verify import require
from many_data_prepare import confirmation_terminals


def record(s):
    s = s.resolve()
    pins = {
        k: selected(s, label)
        for k, label in {
            "protocol": "b-independent-arithmetic-registration",
            "roles": "b-data-registration",
            "freeze": "b-model-freeze",
            "opening": "b-confirmation-h-opening",
        }.items()
    }
    protocol = load(resolve((s,), pins["protocol"]))
    grant = load(resolve((s,), pins["opening"]))
    confirmation_terminals(grant, load(resolve((s,), pins["roles"])))
    for i, case in enumerate(protocol["cases"]):
        f, n = case["family"], case["R"]
        slug = f.replace("_", "-") + "-r" + str(n)
        pins[f"truth{i}"] = selected(s, "b-truth-" + f.replace("_", "-"))
        pins[f"eval{i}"] = selected(s, "b-eval-" + slug)
        terminal = grant["terminals"][f][str(n)]
        pins[f"endpoint{i}"] = terminal["receipt"]
        if (
            terminal["status"] == "succeeded"
            or (s / "work/controller" / ("b-bank-" + slug) / "selected.json").is_file()
        ):
            pins[f"bank{i}"] = selected(s, "b-bank-" + slug)
    return stage(
        s,
        "b-independent-arithmetic-check",
        "evaluation",
        pins,
        {
            "purpose": "native original predicted RGB / H PNG / per-query arrays; independent three-case P1 and P2",
            "cases": protocol["cases"],
            "rate_tolerance": 1e-15,
        },
        [
            "{python}",
            "-B",
            "{source}/scripts/many_data_arithmetic.py",
            "worker",
            "--store",
            str(s),
            "--config",
            "{config}",
            "--out",
            "{out}",
        ],
        "payload/arithmetic.json",
        70000,
    )


def member(native, ref):
    name = "payload/" + ref["path"]
    entry = native.files[name]
    require(
        entry["sha256"] == ref["sha256"] and ("bytes" not in ref or entry["bytes"] == ref["bytes"]),
        "array identity differs from native artifact",
    )
    return native.open(name)


def hand_labels(rgb, raw_rgb):
    require(raw_rgb.shape == (252, 378, 3) and raw_rgb.dtype == np.uint8, "wrong independently decoded H PNG")
    reference = raw_rgb.astype(np.float32) / np.float32(255)
    if rgb is None:
        rgb = np.full(reference.shape, np.nan, np.float32)
    require(rgb.dtype == np.float32 and rgb.shape == reference.shape, "wrong raw prediction grid")
    ref = np.isfinite(reference).all(axis=-1)
    valid = ref & np.isfinite(rgb).all(axis=-1)
    delta = np.abs(np.subtract(rgb, reference, dtype=np.float32))
    summed = np.add(
        np.add(delta[:, :, 0], delta[:, :, 1], dtype=np.float32), delta[:, :, 2], dtype=np.float32
    )
    high = valid & (summed / np.float32(3) > np.float32(0.05))
    return ref, valid, high


def mask_digest(mask):
    return hashlib.sha256(
        np.array(mask.shape, dtype="<i8").tobytes() + np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
    ).hexdigest()


def order_quota(flat_scores, eligible, k):
    ids = np.flatnonzero(eligible)
    if not k:
        return np.array([], np.int64)
    value = flat_scores.reshape(-1)[ids]
    kth = np.partition(value, k - 1)[k - 1]
    before = ids[value < kth]
    tied = ids[value == kth]
    require(len(before) < k <= len(before) + len(tied), "invalid order statistic")
    return np.sort(np.concatenate([before, tied[: k - len(before)]]))


def index_digest(idx):
    return hashlib.sha256(np.asarray(idx, dtype="<i4").tobytes()).hexdigest()


def rate(n, d):
    return n / d if d else None


def same_rate(a, b, eps=1e-15):
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= eps)


def verify_nullable(report, expected, names):
    for k in names:
        require(same_rate(report[k], expected[k]), "different independently recomputed rate " + k)


def images(owner_truth, truth, owner_bank, bank, view):
    ref = truth["targets"][view]["analysis_rgb"]
    with zipfile.ZipFile(member(owner_truth, truth["archive"])) as archive:
        content = archive.read(ref["path"])
    require(
        len(content) == ref["bytes"] and hashlib.sha256(content).hexdigest() == ref["sha256"],
        "different independent reference PNG",
    )
    with Image.open(io.BytesIO(content)) as image:
        require(
            image.mode == "RGB" and image.format == "PNG" and image.size == (378, 252),
            "invalid original H analysis PNG",
        )
        raw = np.array(image, dtype=np.uint8)
    spec = bank["targets"][view]["render"] if bank is not None else None
    if spec is not None:
        with np.load(member(owner_bank, spec), allow_pickle=False) as a:
            rgb = a["rgb"]
        prediction = spec["sha256"]
    else:
        rgb = None
        prediction = None
    return raw, rgb, ref["sha256"], prediction


def arrays(owner, cell, fields):
    spec = {k: cell[k] for k in ("sha256", "bytes")}
    spec["path"] = cell["file"]
    with np.load(member(owner, spec), allow_pickle=False) as d:
        require(set(fields) <= set(d.files), "a native rank or scope array is absent")
        return {k: d[k] for k in fields}


def mechanism(data, full, native_ids, subset_ids, valid, errors, q_order):
    correct = valid & ~errors
    require(
        set(subset_ids) <= set(q_order) and set(q_order) == set(native_ids),
        "wrong full or selected source membership",
    )
    cg = (data["comparable"] == 0) & (full["comparable"] > 0)
    cg &= ~(data["invalid_geometry"] | full["invalid_geometry"] | data["incomplete"] | full["incomplete"])
    found = np.unique(full["witness"][cg])
    require(
        np.all((found >= 0) & (found < len(native_ids)))
        and not {native_ids[int(j)] for j in found}.intersection(subset_ids),
        "gap witness is absent, outside the library or inside an ostensibly non-comparable query",
    )
    n = int(correct.sum())
    numerator = int(np.logical_and(cg, correct).sum())
    complete = not full["incomplete"].any()
    valid_query = not (
        correct & (data["incomplete"] | (data["invalid_geometry"] & ~full["invalid_geometry"]))
    ).any()
    return {
        "correct_with_gap": numerator,
        "all_reference_correct": n,
        "correct_gap_rate": rate(numerator, n) if complete and valid_query else None,
        "audit_complete": complete,
        "witness_source_ids": [native_ids[int(j)] for j in found],
    }


def compute_ready(endpoint, rank_owner, bank, view, valid, errors, ref, frozen, ev):
    rec = endpoint["views"][view]
    cells = rec["cells"]
    full_id = "coverage-" + str(endpoint["A"])
    names = ("coverage-4", "fixed-4", "coverage-16")
    common_fields = ("comparable", "incomplete", "invalid_geometry", "witness")
    native = {}
    snapshots = {}
    for c in dict.fromkeys((*names, full_id)):
        cell = cells[c]
        fields = list(common_fields)
        if c == "coverage-16":
            fields += ["rank_B-CPU-LR", "rank_" + frozen["strong_simple_rule"]]
        native[c] = arrays(rank_owner, cell, fields)
        require(all(arr.shape == ref.shape for arr in native[c].values()), "different rank canvas")
        snapshots[c] = cell["sha256"]
    source_ids = endpoint["source_ids"]
    full_order = cells[full_id]["queries"]
    require(
        len(source_ids) == len(set(source_ids)) == len(full_order) == endpoint["R"]
        and set(source_ids) == set(bank["accessible_ids"]),
        "wrong native bank cardinality",
    )
    scope = {}
    for c in names:
        q = cells[c]["queries"]
        k = 4 if c.endswith("-4") else 16
        require(
            len(q) == len(set(q)) == cells[c]["actual_k"] == k
            and (not c.startswith("coverage-") or full_order[:k] == q),
            "different or non-nested independent query",
        )
        row = mechanism(native[c], native[full_id], source_ids, q, valid, errors, full_order)
        old = ev["mechanism"][c]
        require(
            all(row[k] == old[k] for k in ("correct_with_gap", "all_reference_correct", "audit_complete")),
            "different independent P1 integer denominator",
        )
        verify_nullable(old, row, ["correct_gap_rate"])
        scope[c] = row

    def diff(a, b):
        if a["correct_gap_rate"] is None or b["correct_gap_rate"] is None:
            return None
        require(
            a["all_reference_correct"] == b["all_reference_correct"], "different correct target denominators"
        )
        return (a["correct_with_gap"] - b["correct_with_gap"]) / a["all_reference_correct"]

    p1 = {
        "coverage4_minus_fixed4": diff(scope["coverage-4"], scope["fixed-4"]),
        "coverage16_minus_coverage4": diff(scope["coverage-16"], scope["coverage-4"]),
    }
    verify_nullable(ev["p1"], p1, p1.keys())
    new = native["coverage-16"]["rank_B-CPU-LR"]
    simple = native["coverage-16"]["rank_" + frozen["strong_simple_rule"]]
    require(new.dtype == simple.dtype == np.float32, "different FP32 ranking type")
    common = valid & np.isfinite(new) & np.isfinite(simple)
    denominator = int(common.sum())
    k = denominator // 5
    i = order_quota(new, common, k)
    j = order_quota(simple, common, k)
    e1 = int(errors.reshape(-1)[i].sum())
    e2 = int(errors.reshape(-1)[j].sum())
    p2 = {
        "common": denominator,
        "quota": k,
        "rule": frozen["strong_simple_rule"],
        "new_errors": e1,
        "rule_errors": e2,
        "new_error_rate": rate(e1, k),
        "rule_error_rate": rate(e2, k),
        "new_minus_rule": rate(e1 - e2, k),
        "quota_over_reference": rate(k, int(ref.sum())),
        "common_hash": mask_digest(common),
        "new_selection": index_digest(i),
        "rule_selection": index_digest(j),
    }
    official = ev["p2"]
    integers = (
        "common",
        "quota",
        "rule",
        "new_errors",
        "rule_errors",
        "common_hash",
        "new_selection",
        "rule_selection",
    )
    require(
        all(official[n] == p2[n] for n in integers),
        "different independent P2 membership, eligibility or errors",
    )
    verify_nullable(
        official, p2, ("new_error_rate", "rule_error_rate", "new_minus_rule", "quota_over_reference")
    )
    return {
        "P1": p1,
        "G_counts": scope,
        "P2": p2,
        "same_Q16": cells["coverage-16"]["queries"],
        "native_rank_files_sha256": snapshots,
    }


def worker(a):
    require(os.environ.get("EL_RUN_ID"), "run this arithmetic under its sealed registration")
    start, cpu = time.perf_counter(), time.process_time()
    cfg = load(a.config)
    pins = cfg["sources"]
    owners = {}
    documents = {}
    for key, p in pins.items():
        owners[key], path = read_pinned((a.store,), p)
        documents[key] = load(path)
    grant = documents["opening"]
    registration = documents["protocol"]
    frozen = documents["freeze"]
    confirmation_terminals(grant, documents["roles"])
    require(
        grant["model_freeze"] == pins["freeze"] and cfg["cases"] == registration["cases"],
        "different prespecified cases or models",
    )
    rows = []
    for i, case in enumerate(cfg["cases"]):
        f, n, v = case["family"], case["R"], case["target"]
        end = documents[f"endpoint{i}"]
        truth = documents[f"truth{i}"]
        ev = documents[f"eval{i}"]
        bank = documents.get(f"bank{i}")
        term = grant["terminals"][f][str(n)]
        require(
            term["receipt"] == pins[f"endpoint{i}"]
            and v in grant["allowed_targets"][f]
            and truth["scene_id"] == ev["scene"] == f
            and ev["R"] == n
            and truth["opening_sha256"] == ev["opening_sha256"] == pins["opening"]["sha256"]
            and ev["truth_sha256"] == pins[f"truth{i}"]["sha256"],
            "different target evaluation, opening or independent references",
        )
        if bank is not None:
            require(
                bank["scene_id"] == f
                and bank["provenance"]["source_acquisition"] == truth["inputs"]["source"]["run_id"]
                and bank["target_ids"] == truth["H"]
                and pins[f"bank{i}"]["sha256"] == ev["bank_sha256"],
                "different source calibration or prediction bank",
            )
        physical, rgb, reference_sha, prediction_sha = images(
            owners[f"truth{i}"], truth, owners.get(f"bank{i}"), bank, v
        )
        ref, valid, high = hand_labels(rgb, physical)
        old = ev["views"][v]
        independent = {
            "reference": int(ref.sum()),
            "labelled": int(valid.sum()),
            "errors": int(high.sum()),
            "prediction_unavailable": int((ref & ~valid).sum()),
            "labels_sha256": mask_digest(high),
            "valid_labels_sha256": mask_digest(valid),
            "source_reference_sha256": reference_sha,
            "prediction_sha256": prediction_sha,
        }
        require(
            all(old[k] == val for k, val in independent.items()),
            "actual RGB labels or denominators differ for " + f,
        )
        if term["status"] == "succeeded":
            require(
                end["family"] == f
                and end["R"] == n
                and end["target_ids"] == truth["H"]
                and end["references_accessed"] == 0
                and end["models"] == {"B-CPU-LR": frozen["final_model"]["sha256"]}
                and end["views"][v]["prediction"] == prediction_sha
                and ev["scores_sha256"] == pins[f"endpoint{i}"]["sha256"],
                "native ranks belong to a different prediction, family or model",
            )
        if term["status"] == "succeeded" and end["views"][v]["status"] == "ready":
            result = compute_ready(end, owners[f"endpoint{i}"], bank, v, valid, high, ref, frozen, old)
            availability = "ready"
        else:
            require(
                old["p1"] is None and old["p2"] is None,
                "unavailable comparison has a non-null official endpoint",
            )
            result = {"P1": None, "P2": None, "reason": old.get("missing_reason", term["status"])}
            availability = "native_terminal_or_missing_target"
        rows.append(
            {
                **case,
                "availability": availability,
                "original_H_analysis_decodes": 1,
                **independent,
                **result,
                "matched_native_numbers_and_hashes": True,
                "official_evaluation": pins[f"eval{i}"],
            }
        )
    save(
        a.out / "arithmetic.json",
        {
            "schema": "el-many-independent-primary-arithmetic-1",
            "run_id": os.environ["EL_RUN_ID"],
            "protocol": pins["protocol"],
            "opening": pins["opening"],
            "model_freeze": pins["freeze"],
            "cases": rows,
            "count_compared": len(rows),
            "total_H_photographs_decoded": len(rows),
            "all_compared": True,
            "rate_absolute_tolerance": cfg["rate_tolerance"],
            "worker_wall_seconds": time.perf_counter() - start,
            "worker_cpu_seconds": time.process_time() - cpu,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "worker", "register"))
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    a = parser.parse_args()
    if a.action == "register":
        s = a.store.resolve()
        p = stage(
            s,
            "b-independent-arithmetic-registration",
            "archive_verify",
            {"study": selected(s, "b-study-registration"), "roles": selected(s, "b-data-registration")},
            {
                "scope": "three predetermined confirmation h00 views; independent arithmetic, no observations opened"
            },
            [
                "{python}",
                "-I",
                "-c",
                "import shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2])",
                "{input:protocol}",
                "{out}/arithmetic-protocol.json",
            ],
            "payload/arithmetic-protocol.json",
            9000,
            {"protocol": Path(__file__).resolve().parents[1] / "protocol/many-source-arithmetic-v1.json"},
        )
        print(json.dumps(p), flush=True)
    elif a.action == "record":
        print(json.dumps(record(a.store)), flush=True)
    else:
        worker(a)
