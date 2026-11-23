"""Independent AP and exact strict-threshold readbacks for the recorded development freeze."""

from fractions import Fraction

import numpy as np

from evidence_lens.formal_inputs import require


def ap_readback(score, y):
    from sklearn.metrics import average_precision_score

    n = int(y.size)
    pos = int(np.count_nonzero(y))
    reason = None if 0 < pos < n else "empty_mask" if n == 0 else "all_negative" if pos == 0 else "all_positive"
    return {
        "ap": float(average_precision_score(y, score)) if reason is None else None,
        "positive": pos,
        "n": n,
        "missing_reason": reason,
    }


def threshold_readback(lists, denominators, fixed=False):
    """Sum per-family score histograms. Exact rational L/V resolves the boundary and coverage ties."""
    valid = {f: int(n) for f, n in denominators.items() if n > 0}
    require(valid, "no positive family V in threshold verification")
    if fixed:
        coverage = sum(Fraction(int(np.count_nonzero(lists[f] < 0.5)), n) for f, n in valid.items()) / len(
            valid
        )
        return {"tau": 0.5, "coverage": float(coverage)}
    hist = {f: np.unique(lists[f], return_counts=True) for f in valid}
    levels = np.unique(np.concatenate([v for v, _ in hist.values()]))
    mass = np.zeros(len(levels), np.float64)
    run_counts = {}
    for f, (values, counts) in hist.items():
        require(len(lists[f]) <= valid[f] and np.isfinite(values).all(), "invalid operating score counts")
        mass[np.searchsorted(levels, values)] += counts.astype(np.float64) / valid[f] / len(valid)
        run_counts[f] = np.r_[0, np.cumsum(counts)]

    def coverage(k):
        if not k:
            return Fraction(0)
        return sum(
            Fraction(int(run_counts[f][np.searchsorted(hist[f][0], levels[k - 1], side="right")]), n)
            for f, n in valid.items()
        ) / len(valid)

    goal = Fraction(4, 5)
    k = min(len(levels), int(np.searchsorted(np.cumsum(mass), 0.8)) + 1)
    while k and coverage(k - 1) >= goal:
        k -= 1
    while k < len(levels) and coverage(k) < goal:
        k += 1
    options = {k, max(0, k - 1)}
    j = min(options, key=lambda i: (abs(coverage(i) - goal), coverage(i)))
    if not j:
        tau = 0.0
    elif j == len(levels):
        tau = float(np.nextafter(float(levels[-1]), np.inf))
    else:
        tau = (float(levels[j - 1]) + float(levels[j])) / 2
    return {
        "tau": tau,
        "coverage": float(coverage(j)),
        "accepted_distinct_groups": j,
        "candidate_count": len(levels) + 1,
        "eligible_families": sorted(valid),
    }


def verify_tensor_payloads(path, expected):
    """Read and SHA-256 each named tensor directly from the little-endian safetensors member."""
    import hashlib
    import json
    import math
    import struct

    with path.open("rb") as stream:
        head = stream.read(8)
        require(len(head) == 8, "missing safetensors header")
        size = struct.unpack("<Q", head)[0]
        require(
            2 <= size <= 32 * 1024 * 1024 and size <= path.stat().st_size - 8, "invalid safetensors header size"
        )

        def unique(rows):
            out = dict(rows)
            require(len(rows) == len(out), "duplicate safetensors metadata key")
            return out

        fields = json.loads(stream.read(size), object_pairs_hook=unique)
        fields.pop("__metadata__", None)
        require(set(fields) == set(expected), "actual safetensors names differ from the learned/frozen table")
        start = 8 + size
        end = start
        element = {
            "F32": 4,
            "F64": 8,
            "F16": 2,
            "BF16": 2,
            "I64": 8,
            "U64": 8,
            "I32": 4,
            "U32": 4,
            "I16": 2,
            "U16": 2,
            "I8": 1,
            "U8": 1,
            "BOOL": 1,
        }
        for name, t in sorted(fields.items(), key=lambda x: tuple(x[1]["data_offsets"])):
            lo, hi = t["data_offsets"]
            shape = t["shape"]
            require(
                isinstance(shape, list)
                and all(type(s) is int and s >= 0 for s in shape)
                and t["dtype"] in element
                and type(lo) is int
                and type(hi) is int
                and start + lo == end
                and hi - lo == math.prod(shape) * element[t["dtype"]],
                "invalid learned tensor layout " + name,
            )
            digest = hashlib.sha256()
            left = hi - lo
            while left:
                part = stream.read(min(1 << 20, left))
                require(part, "truncated learned tensor " + name)
                digest.update(part)
                left -= len(part)
            require(digest.hexdigest() == expected[name], "actual saved tensor SHA differs: " + name)
            end = start + hi
        require(not stream.read(1), "unaccounted bytes after learned tensors")
    return len(fields)


def baseline_qa(roots, stages, output, run_id):
    from evidence_lens.scoring_io import new_output, write_json
    from vision_data import open_dataset, file_for, packed
    from vision_evaluate import FIVE, deploy_contract, ranking, read_scores, score_sources
    from vision_sources import find_run, member, signed

    def source(key, name):
        run = find_run(roots, stages[key])
        p, s = member(run, "payload/" + name)
        require(run.run["execution_status"] == "succeeded", "unfinished QA input")
        return p, s["sha256"]

    dp, dh = source("development", "dataset.json")
    sp, sh = source("scores", "scores.json")
    tp, th = source("thresholds", "thresholds.json")
    op, oh = source("old_development_replay", "readback.json")
    old = signed(op, oh)
    ds = open_dataset(dp.parent, dh, "development")
    t = signed(tp, th)
    sources = score_sources(ds, dh, [(sp, sh)], ("original-five",))
    require(
        t["schema"] == "el-vision-baseline-thresholds-1"
        and t["method_sources"]["original-five"] == sources["original-five"][2]
        and t["development_dataset_sha256"] == dh
        and len(ds["views"]) == 48
        and len({r["family"] for r in ds["views"]}) == 8,
        "wrong baseline or development rows",
    )
    require(
        old["schema"] == "el-stage2-original-development-qa-1"
        and old["cohort"] == "original_development_only"
        and old["targets"] == 24
        and old["unequal_finite_comparisons"] == old["state_disagreements"] == 0,
        "original development engineering replay differs",
    )
    from collections import defaultdict

    den = defaultdict(int)
    vals = {m: defaultdict(list) for m in FIVE}
    counts = {m: 0 for m in FIVE}
    detail = {}
    worst = 0.0
    compared = 0
    for row in ds["views"]:
        f, v = row["family"], row["view"]
        src = packed(
            file_for(dp.parent, row["full"]["truth"]), ("reference_valid", "rgb_evaluable", "rgb_high_error")
        )
        E = src["rgb_evaluable"]
        Y = src["rgb_high_error"] == 1
        V = int(src["reference_valid"].sum())
        den[f] += V
        scores, _ = read_scores(row, sources, deploy=deploy_contract(dp.parent, row))
        common = E & np.logical_and.reduce([np.isfinite(scores[m]) for m in FIVE])
        ranks = {}
        for m, s in scores.items():
            own = E & np.isfinite(s)
            vals[m][f].append(s[own])
            counts[m] += int(own.sum())
            ranks[m] = {}
            for label, mask in (("own", own), ("five_common", common)):
                a = ap_readback(s[mask], Y[mask])
                b = ranking(s[mask], Y[mask])
                require(
                    all(a[k] == b[k] for k in ("n", "positive", "missing_reason"))
                    and (a["ap"] is None) == (b["ap"] is None),
                    "whole-tie AP numerator or availability mismatch",
                )
                if a["ap"] is not None:
                    worst = max(worst, abs(a["ap"] - b["ap"]))
                    compared += 1
                    require(worst <= 1e-12, "independent rank AP disagreement")
                ranks[m][label] = a
        detail[f + "/" + v] = {
            "V": V,
            "E": int(E.sum()),
            "Y": int(Y.sum()),
            "S5": int(common.sum()),
            "ranks": ranks,
        }
    thresholds = {}
    for m, items in vals.items():
        fresh = threshold_readback({f: np.concatenate(seq) for f, seq in items.items()}, den, m == "L0")
        orig = t["choices"][m]
        require(
            fresh["tau"] == orig["tau"]
            and np.isclose(fresh["coverage"], orig["coverage"], rtol=0, atol=1e-15)
            and (
                m == "L0"
                or all(
                    fresh[k] == orig[k]
                    for k in ("accepted_distinct_groups", "candidate_count", "eligible_families")
                )
            ),
            "independent coverage or strictly grouped threshold differs",
        )
        thresholds[m] = fresh
    with new_output(output) as out:
        write_json(
            out / "readback.json",
            {
                "schema": "el-vision-five-development-qa-1",
                "run_id": run_id,
                "stages": stages,
                "source_sha256": {
                    "development": dh,
                    "scores": sh,
                    "thresholds": th,
                    "original_development_replay": oh,
                },
                "cohort": "eight registered development B0 families",
                "views": detail,
                "V": sum(den.values()),
                "E": sum(r["E"] for r in detail.values()),
                "finite_E": counts,
                "defined_view_domain_ap_comparisons": compared,
                "max_abs_ap_difference": worst,
                "thresholds": thresholds,
                "historical_development": {
                    "targets": old["targets"],
                    "V": old["V"],
                    "finite_score_comparisons": sum(old["finite_by_method"].values()),
                    "unequal_scores": old["unequal_finite_comparisons"],
                    "unequal_states": old["state_disagreements"],
                },
            },
        )


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from vision_sources import signed

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "shelf", "roster", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--run-id", required=True)
    a = p.parse_args()
    baseline_qa((a.store.resolve(), a.shelf.resolve()), signed(a.roster, a.sha256), a.output, a.run_id)
