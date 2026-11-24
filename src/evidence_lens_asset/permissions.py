"""Validate branch-local source grants for prospectively registered upstream views."""

from evidence_lens_bundle.common import encoded, require, sha


def source_grant(registration, branch, choice=None):
    require(
        registration.get("schema") == "el-upstream-layout-authority-1",
        "invalid upstream layout registration",
    )
    fixed = registration["fixed_layouts"]
    for name, count in (("M2", 2), ("M4", 4)):
        order = fixed[name]
        require(
            isinstance(order, list)
            and len(order) == len(set(order)) == count
            and all(isinstance(v, str) and v for v in order),
            "invalid fixed source order",
        )
    require(
        fixed["M2"] == [fixed["M4"][0], fixed["M4"][-1]],
        "fixed layouts must share the registered endpoints",
    )
    if branch in ("fixed-M2", "fixed-M4"):
        return list(fixed[branch[-2:]])
    require(branch == "optional-M4", "unknown upstream acquisition branch")
    require(
        isinstance(choice, dict)
        and choice.get("schema") == "el-upstream-optional-choice-1"
        and choice.get("registration_sha256") == sha(encoded(registration)),
        "missing sealed optional selection or failure",
    )
    if choice.get("status") == "failed":
        require(
            choice.get("selected_ids") is None and choice.get("reason"),
            "incomplete optional failure",
        )
        return list(fixed["M2"])
    pair = choice.get("selected_ids")
    require(
        choice.get("status") == "selected"
        and isinstance(pair, list)
        and len(pair) == len(set(pair)) == 2
        and set(pair) <= set(registration["optional_pool"])
        and set(pair).isdisjoint(fixed["M2"]),
        "invalid selected pair",
    )
    return [fixed["M2"][0], *pair, fixed["M2"][1]]
