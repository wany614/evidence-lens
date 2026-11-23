"""Exact registered matched coverage with a fixed, label-independent break of cutoff ties."""

import hashlib
import numpy as np
from dtu_score import METHODS
from evidence_lens.formal_inputs import require

PERCENTS = (1, 2, 5, 10, 20, 40, 60, 80)


def ranks(shape, obj, condition, view):
    seed = int.from_bytes(
        hashlib.sha256(f"el-recovery-coverage-v1|{obj}|{condition}|{view}".encode("utf-8")).digest()[:8],
        "big",
    )
    n = int(np.prod(shape))
    perm = np.random.Generator(np.random.PCG64(seed)).permutation(n)
    order = np.empty(n, np.int64)
    order[perm] = np.arange(n)
    return order.reshape(shape)


def counts_at(values, y, index_rank, V, ks):
    require(
        values.ndim == y.ndim == index_rank.ndim == 1
        and len(values) == len(y) == len(index_rank)
        and np.isfinite(values).all()
        and ((y == 0) | (y == 1)).all(),
        "invalid matched-coverage input",
    )
    ids = np.lexsort((index_rank, values))
    s = values[ids]
    pos = np.r_[0, np.cumsum(y[ids].astype(np.int64))]
    out = []
    size = len(s)
    for name, k in ks:
        row = {
            "anchor": name,
            "requested_k": k,
            "V": V,
            "eligible": size,
            "eligible_positive": int(pos[-1]),
            "realized_coverage": None,
            "accepted": None,
            "accepted_positive": None,
            "risk": None,
            "cutoff_score": None,
            "cutoff_group_size": None,
            "accepted_in_cutoff_group": None,
            "min_possible_accepted_positive": None,
            "max_possible_accepted_positive": None,
        }
        if k > size:
            row["reason"] = "outside_support"
        elif not k:
            row.update(
                reason="empty_low", accepted=0, accepted_positive=0, realized_coverage=0 if V else None
            )
        else:
            cutoff = s[k - 1]
            a = int(np.searchsorted(s, cutoff, "left"))
            b = int(np.searchsorted(s, cutoff, "right"))
            choose = k - a
            gp = int(pos[b] - pos[a])
            err = int(pos[k])
            lo = int(pos[a]) + max(0, choose - (b - a - gp))
            hi = int(pos[a]) + min(choose, gp)
            require(lo <= err <= hi, "incorrect whole-tie error bounds")
            row.update(
                reason=None,
                accepted=k,
                accepted_positive=err,
                risk=err / k,
                realized_coverage=k / V,
                cutoff_score=float(cutoff),
                cutoff_group_size=b - a,
                accepted_in_cutoff_group=choose,
                min_possible_accepted_positive=lo,
                max_possible_accepted_positive=hi,
            )
        out.append(row)
    return out


def matched(obj, condition, view, versions, V):
    require(
        V.ndim == 2 and V.dtype == bool and set(versions) == {"old", "new"},
        "matched coverage requires old and new on one Boolean reference grid",
    )
    ranking = ranks(V.shape, obj, condition, view)
    count = int(V.sum())
    poss = {}
    for name, z in versions.items():
        e, y = z["E"], z["Y"]
        require(
            z["V"].dtype == bool
            and np.array_equal(z["V"], V)
            and e.shape == y.shape == V.shape
            and e.dtype == bool
            and not np.any(e & ~V)
            and np.all((y[e] == 0) | (y[e] == 1)),
            "matched E must lie within the same V with binary labels on E",
        )
        require(
            set(z["scores"]) == set(z["thresholds"]) == set(METHODS)
            and all(r.shape == V.shape and r.dtype == np.float32 for r in z["scores"].values()),
            "all four original score maps on the same grid are required",
        )
        common = e.copy()
        for m in METHODS:
            common &= np.isfinite(z["scores"][m])
        require(
            not np.any(common & ~e) and all(np.isfinite(r[common]).all() for r in z["scores"].values()),
            "nonfinite member of four-method common support",
        )
        poss[name] = common
    masks = [("within_" + n, common, [n]) for n, common in poss.items()]
    paired = poss["old"] & poss["new"]
    require(
        all(np.isfinite(r[paired]).all() for z in versions.values() for r in z["scores"].values()),
        "nonfinite member of eight-score paired support",
    )
    masks.append(("paired_old_new", paired, list(versions)))
    out = []
    for kind, domain, names in masks:
        ks = [(f"{p}%_V", (2 * p * count + 99) // 200) for p in PERCENTS]
        for n in names:
            z = versions[n]
            baseline = z["scores"]["MS-L1"]
            ks.append(
                (
                    n + "_frozen_MS-L1",
                    int((domain & (baseline.astype(np.float64) < z["thresholds"]["MS-L1"])).sum()),
                )
            )
        for n in names:
            ver = versions[n]
            y = ver["Y"][domain]
            for m, score in ver["scores"].items():
                for row in counts_at(score[domain], y, ranking[domain], count, ks):
                    out.append({"domain": kind, "version": n, "method": m, **row})
    return out
