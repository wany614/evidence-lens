"""Authenticated train/development transport and native prediction support."""

import re
from pathlib import Path

import numpy as np

from evidence_lens_vision.contract import HASH, bound, digest, read, require

VIEWS = tuple(f"v{i:02d}" for i in range(1, 7))
CONDITIONS = ("M2", "M4")
SCOPES = ("one", "two", "all")
MODES = {"one": ["s00"], "two": ["s00", "s03"], "all": ["s00", "s01", "s02", "s03"]}


def source_ids(condition, scope):
    require(condition in CONDITIONS and scope in SCOPES, "unknown acquisition or scoring subset")
    return MODES["two"] if condition == "M2" and scope == "all" else MODES[scope]


def signed(path, expected):
    require(
        isinstance(expected, str) and HASH.fullmatch(expected) and digest(path) == expected,
        "unmatched input SHA-256: " + Path(path).name,
    )
    return read(path)


def catalog_families(catalog, split):
    rows = catalog["families"]
    families = [r["scene_family_id"] for r in rows if r["split"] == split]
    require(
        len(families) == len(set(families))
        and all(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", f) for f in families),
        "invalid family roster",
    )
    return sorted(families)


def validated_preprocessing(path, expected, train_sha, schema_sha):
    pre = signed(path, expected)
    require(
        pre["schema"] == "el-stage3-risk-preprocessing-1"
        and pre["feature_schema_sha256"] == schema_sha
        and pre["training_dataset_sha256"] == train_sha,
        "preprocessing does not match the train-only data and feature schema",
    )
    names, count = pre["columns"], pre["numeric_count"]
    require(
        isinstance(names, list)
        and len(names) == len(set(names))
        and type(count) is int
        and 0 < count <= len(names),
        "invalid ordered multi-source features",
    )
    for key in ("medians", "means", "scales"):
        a = np.asarray(pre[key], dtype=np.float64)
        require(
            a.shape == (count,) and np.isfinite(a).all() and (key != "scales" or (a > 0).all()),
            "invalid train-only " + key,
        )
    return pre


class Dataset:
    def __init__(self, path, expected, split, catalog, catalog_sha, pre):
        require(split in ("train", "development", "test"), "unexpected split")
        self.path = Path(path)
        self.root = self.path.parent
        self.sha = expected
        self.split = split
        d = signed(path, expected)
        require(
            d["schema"] == "el-stage3-risk-dataset-1"
            and d["split"] == split
            and d["feature_schema_sha256"] == pre["feature_schema_sha256"]
            and d["catalog_sha256"] == catalog_sha
            and d["columns"] == pre["columns"],
            "wrong C split, catalog, schema or feature order",
        )
        self.columns = list(d["columns"])
        self.families = catalog_families(catalog, split)
        self.rows = d["views"]
        self.document = d
        require(
            d["cohort"] == "formal"
            and set(d["families"]) == set(self.families)
            and len(d["families"]) == len(self.families),
            "the neural study requires the complete registered roster",
        )
        expected_keys = {(f, c, v) for f in self.families for c in CONDITIONS for v in VIEWS}
        actual = set()
        for row in self.rows:
            f, c, v = (row[k] for k in ("family", "condition", "view"))
            require(
                (f, c, v) in expected_keys
                and (f, c, v) not in actual
                and row["width"] == row["height"] == 256,
                "duplicate, unauthorized, missing or incorrectly sized scheduled view",
            )
            actual.add((f, c, v))
            require(set(row["modes"]) == set(SCOPES), "incomplete scoring mode manifest")
            for scope, mode in row["modes"].items():
                require(
                    mode["source_ids"] == source_ids(c, scope)
                    and mode["alias_of"] == ("two" if (c, scope) == ("M2", "all") else None),
                    "scorer was supplied a different source subset",
                )
            if c == "M2":
                a, b = row["modes"]["all"], row["modes"]["two"]
                require(
                    a["samples"] == b["samples"] and a["deploy"] == b["deploy"],
                    "M2 two and all must share the same bytes",
                )
            if row["status"] == "ready":
                require(
                    isinstance(row["rgb"], dict) and row["failure"] is None, "completed view lacks native RGB"
                )
            else:
                require(
                    row["status"] == "render_failed"
                    and row["rgb"] is None
                    and row.get("failure")
                    and all(m["deploy"] is None for m in row["modes"].values()),
                    "failed view must retain its upstream disposition",
                )
        require(
            actual == expected_keys and expected_keys, "incomplete registered family/condition/view roster"
        )
        self.rows = sorted(self.rows, key=lambda r: (r["family"], r["condition"], r["view"]))
        self.verified = {}

    def file(self, spec):
        k = (spec["path"], spec["sha256"], spec["bytes"])
        if k not in self.verified:
            self.verified[k] = bound(self.root, spec)
        return self.verified[k]

    def rgb(self, row):
        rgb = np.load(self.file(row["rgb"]), allow_pickle=False)
        require(
            rgb.dtype == np.float32 and rgb.shape == (row["height"], row["width"], 3),
            "malformed native predicted RGB",
        )
        return rgb

    def samples(self, row):
        require(self.split == "train", "supervision samples are restricted to training")
        mode = row["modes"]["all"]
        with np.load(self.file(mode["samples"]), allow_pickle=False) as z:
            ix, x, y, p = (z[k] for k in ("index", "X", "Y", "inclusion_probability"))
        n = len(ix)
        eligible = mode["eligible"]
        require(
            type(eligible) is int
            and 0 <= eligible <= row["height"] * row["width"]
            and n == min(4096, eligible)
            and x.dtype == np.float32
            and x.shape == (n, len(self.columns))
            and y.dtype == np.uint8
            and y.shape == (n,)
            and np.all(y <= 1)
            and ix.dtype == np.int32
            and ix.shape == (n,)
            and (not n or (ix[0] >= 0 and ix[-1] < 256**2 and np.all(ix[1:] > ix[:-1]))),
            "wrong fixed native sample or supervision counts",
        )
        require(
            p.dtype == np.float64 and p.ndim == 0 and float(p) == (n / eligible if eligible else 0.0),
            "wrong inclusion probability",
        )
        return ix, x, y

    def deployment(self, row, scope):
        require(scope in SCOPES, "unknown scoring subset")
        with np.load(self.file(row["modes"][scope]["deploy"]), allow_pickle=False) as z:
            x, ok = z["X"], z["scoreable"]
        require(
            x.dtype == np.float32
            and x.shape == (row["height"], row["width"], len(self.columns))
            and ok.dtype == np.bool_
            and ok.shape == x.shape[:2]
            and int(ok.sum()) == row["modes"][scope]["scoreable_pixels"],
            "wrong dense deployment feature/mask",
        )
        return x, ok

    def dev_truth(self, row):
        require(self.split == "development", "training selection may only read development error labels")
        with np.load(self.file(row["truth"]), allow_pickle=False) as z:
            V, E, Y = (z[k] for k in ("reference_valid", "rgb_evaluable", "rgb_high_error"))
        require(
            V.dtype == E.dtype == np.bool_
            and Y.dtype == np.uint8
            and V.shape == E.shape == Y.shape == (row["height"], row["width"])
            and not np.any(E & ~V)
            and np.all(Y[E] <= 1)
            and np.all(Y[~E] == 255),
            "invalid independent RGB error labels",
        )
        return V, E, Y == 1
