"""Optional static replay metadata. Pose uncertainty and dynamic-mask references are stored verbatim."""

from __future__ import annotations

import copy
import math

import numpy as np

from .bindings import TASK
from .common import encoded, exact_keys, require, sha, text

SCHEMA = "el-static-replay-sidecar-1"
OPTIONAL = {
    "acquisition_timestamp",
    "clock_domain",
    "pose_covariance",
    "dynamic_mask",
    "source_selection_policy",
    "task_schema",
}
CAPABILITIES = {
    "sequence_frame_identity": {"accepted": True, "used": True, "validated": "static replay association"},
    "map_source_model_renderer_revisions": {
        "accepted": True,
        "used": True,
        "validated": "exact snapshot identity",
    },
    "acquisition_timestamp": {
        "accepted": True,
        "used": False,
        "validated": False,
        "metadata_check": "number and named clock",
    },
    "clock_domain": {"accepted": True, "used": False, "validated": False, "metadata_check": "name only"},
    "source_selection_policy": {
        "accepted": True,
        "used": False,
        "validated": False,
        "metadata_check": "stored as declared provenance",
    },
    "pose_covariance": {
        "accepted": True,
        "used": False,
        "validated": False,
        "metadata_check": "shape, coordinate frame and units",
    },
    "dynamic_mask": {
        "accepted": True,
        "used": False,
        "validated": False,
        "metadata_check": "unopened reference and provenance",
    },
    "task_schema": {"accepted": True, "used": True, "validated": "registered RGB photometric risk only"},
    "cpu_risk": {"accepted": True, "used": True, "validated": "see platform run receipts"},
    "cuda_risk": {"accepted": False, "used": False, "validated": False},
    "upstream_cuda_render": {"accepted": True, "used": False, "validated": "separate gsplat registration"},
}


def optional_metadata(doc=None):
    if doc is None:
        doc = {"schema": SCHEMA}
    require(
        isinstance(doc, dict) and doc.get("schema") == SCHEMA and set(doc) <= OPTIONAL | {"schema"},
        "unsupported replay sidecar fields or schema",
    )
    result = {k: copy.deepcopy(doc.get(k)) for k in sorted(OPTIONAL)}
    stamp, clock = result["acquisition_timestamp"], result["clock_domain"]
    if clock is not None:
        text(clock, "acquisition clock domain")
    if stamp is not None:
        require(
            type(stamp) in (int, float) and math.isfinite(stamp) and clock is not None,
            "numeric acquisition timestamp requires a named clock domain",
        )
    policy = result["source_selection_policy"]
    if policy is not None:
        text(policy, "source selection policy")
    cov = result["pose_covariance"]
    if cov is not None:
        exact_keys(cov, ("matrix", "coordinates", "units"), "pose covariance")
        text(cov["coordinates"], "covariance coordinates, axis order and frame")
        text(cov["units"], "covariance axis units")
        mat = np.asarray(cov["matrix"])
        require(
            mat.shape == (6, 6) and mat.dtype.kind in "ifu" and np.isfinite(mat).all(),
            "pose covariance requires a finite 6 x 6 matrix",
        )
    mask = result["dynamic_mask"]
    if mask is not None:
        exact_keys(mask, ("reference", "source", "time", "clock_domain"), "dynamic mask metadata")
        text(mask["reference"], "unopened dynamic mask reference")
        text(mask["source"], "dynamic mask source")
        if mask["clock_domain"] is not None:
            text(mask["clock_domain"], "mask clock")
        if mask["time"] is not None:
            require(
                type(mask["time"]) in (int, float)
                and math.isfinite(mask["time"])
                and mask["clock_domain"] is not None,
                "mask timestamp requires a clock domain",
            )
    require(result["task_schema"] in (None, TASK), "unsupported task schema")
    result["task_schema"] = TASK
    return {"schema": SCHEMA, **result}


def register_views(document, sequences):
    """Bind optional metadata to actual sequence/view IDs before starting the timed loop."""
    if document is None:
        document = {"schema": SCHEMA}
    require(isinstance(document, dict), "sidecar must be a JSON object")
    defaults = optional_metadata({k: v for k, v in document.items() if k != "views"})
    default_id = sha(encoded(defaults))
    records, assignments = {default_id: defaults}, {}
    views = document.get("views", {})
    require(isinstance(views, dict), "sidecar views must map sequence IDs to target IDs")
    roster = {}
    for seq in sequences:
        roster.setdefault(seq.sequence_id, set()).update(seq.bundle.targets)
    for sequence, observations in views.items():
        require(
            sequence in roster and isinstance(observations, dict) and set(observations) <= roster[sequence],
            "sidecar declares an unknown source sequence or target view",
        )
        for view, fields in observations.items():
            require(isinstance(fields, dict) and set(fields) <= OPTIONAL, "unknown per-view metadata")
            value = optional_metadata({**defaults, **fields})
            digest = sha(encoded(value))
            records[digest] = value
            assignments[sequence, view] = digest
    return default_id, records, assignments
