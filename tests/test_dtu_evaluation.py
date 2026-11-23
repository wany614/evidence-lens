"""Synthetic numeric contract checks. These arrays never enter recorded DTU observations."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dtu_score import METHODS


def metric_module():
    pytest.importorskip("pyarrow")
    from dtu_eval_metrics import labels, observation, aggregate_views

    return labels, observation, aggregate_views


def test_error_neighbour_and_permissions():
    labels, _, _ = metric_module()
    ref = np.zeros((3, 4, 3), np.float32)
    ref[1, 2:] = 1
    pred = ref.copy()
    pred[1, 1] = 1
    pred[0, 0] = np.nan
    allowed = np.ones((3, 4), bool)
    allowed[1, 2] = False
    v, e, y, err, aux = labels(ref, pred, allowed)
    assert (v.sum(), e.sum(), np.count_nonzero(y == 1)) == (11, 10, 1)
    assert err[1, 1] == aux[1, 1] == 1 and np.isnan(err[0, 0])
    allowed[1, 2] = True
    v, e, y, err, aux = labels(ref, pred, allowed)
    assert aux[1, 1] == 0 and err[1, 1] == 1 and (aux[e] <= err[e]).all()
    sklearn = pytest.importorskip("sklearn")
    from dtu_eval_replay import independent_labels

    assert all(
        np.array_equal(a, b, equal_nan=True)
        for a, b in zip((v, e, y, err, aux), independent_labels(ref, pred, allowed))
    )


def test_exact_threshold_label_ties_unknown_and_fair_domain():
    labels, observe, aggregate = metric_module()
    ref = np.zeros((1, 6, 3), np.float32)
    pred = ref.copy()
    pred[0, :, 0] = [0, 0.75, 0, 0.75, 0, 0.75]
    p = labels(ref, pred, np.ones((1, 6), bool))
    rs = {m: np.array([[0.25, 0.5, 0.5, 0.75, 0.75, np.nan]], np.float32) for m in METHODS}
    rs[METHODS[1]] = rs[METHODS[1]].copy()
    rs[METHODS[1]][0, 0] = np.nan
    tau = {m: 0.5 for m in METHODS}
    rows, groups = observe(*p, rs, tau)
    out = rows[METHODS[0]]
    assert [out["counts"][k] for k in ("V", "E", "C", "CY", "Y", "L", "LY", "H", "HY", "U", "UY")] == [
        6,
        6,
        4,
        2,
        3,
        1,
        0,
        4,
        2,
        1,
        1,
    ]
    assert out["own"]["ap"] == pytest.approx(0.5) and out["common"]["ap"] == pytest.approx(0.5)
    agg = aggregate([out, out], [groups[METHODS[0]]] * 2, 0.5, planned_views=6)
    assert agg["AP"]["common"] == {"ap": 0.5, "defined_views": 2, "planned_views": 6}
    assert agg["risk_coverage"]["frozen_L"] == 2
    v, e, y, err, aux = labels(
        np.zeros((1, 2, 3), np.float32), np.full((1, 2, 3), np.float32(0.05)), np.ones((1, 2), bool)
    )
    assert y.max() == 0


def test_zero_denominators_remain_separate():
    labels, observe, _ = metric_module()
    ref = np.zeros((2, 2, 3), np.float32)
    ms = np.ones((2, 2), bool)
    risks = {m: np.full((2, 2), np.nan, np.float32) for m in METHODS}
    tau = dict.fromkeys(METHODS, 0.5)
    d, _ = observe(*labels(ref, None, ms), risks, tau)
    o = d[METHODS[0]]
    assert (
        o["counts"]["V"] == 4 and o["counts"]["V_minus_E"] == 4 and o["rates"]["low_coverage"]["value"] == 0
    )
    assert (
        o["rates"]["low_error"]["value"] is None
        and o["rates"]["rgb_error_fraction"]["value"] is None
        and o["rates"]["unknown"]["value"] is None
    )
    d, _ = observe(*labels(ref, ref.copy(), ms), risks, tau)
    o = d[METHODS[0]]
    assert (
        o["counts"]["E"] == o["counts"]["U"] == 4
        and o["rates"]["unknown"]["value"] == 1
        and o["rates"]["rgb_error_fraction"]["value"] == 0
    )
    assert o["common"]["missing_reason"] == "empty_mask"
    rs = {m: np.zeros((2, 2), np.float32) for m in METHODS}
    d, _ = observe(*labels(ref, ref.copy(), ms), rs, tau)
    assert d[METHODS[0]]["own"]["missing_reason"] == "all_negative"


def test_external_requires_photo_gate_before_reading_inputs(tmp_path, monkeypatch):
    metric_module()
    import dtu_eval

    f = tmp_path / "wrong.json"
    f.write_text('{"schema":"el-dtu-opening-1","object_ids":["set6"]}')
    calls = []

    def read(roots, pin, success=True):
        calls.append(pin)
        return None, f

    monkeypatch.setattr(dtu_eval, "read_pinned", read)
    authority = {"identity": "source only"}
    req = {
        "schema": "el-dtu-eval-request-1",
        "object_id": "set6",
        "condition": "M4",
        "arm": "main-256",
        "role": "external",
        "status": "ready",
        "authority": authority,
    }
    with pytest.raises(ValueError, match="remain sealed"):
        dtu_eval.assess((), req, tmp_path / "not-created", "synthetic-test")
    assert calls == [authority] and not (tmp_path / "not-created").exists()


def test_public_cohorts_keep_empty_domains_and_equal_objects():
    labels, observe, aggregate = metric_module()
    from dtu_report import method_rows, summaries

    ref = np.zeros((1, 2, 3), np.float32)
    pred = ref.copy()
    pred[0, 0, :] = 1
    dom = labels(ref, pred, np.ones((1, 2), bool))
    risk = dict.fromkeys(METHODS, np.array([[1, 0.3]], np.float32))
    tau = dict.fromkeys(METHODS, 0.5)
    ob, g = observe(*dom, risk, tau)
    ag = {m: aggregate([ob[m]] * 6, [g[m]] * 6, 0.5, planned_views=6) for m in METHODS}
    rows = [
        {
            "view": "v0" + str(i),
            "disposition": "ready",
            "domains": {"full_crop": ob, "object": None},
            "domain_reasons": {"object": "No verified mask"},
        }
        for i in range(1, 7)
    ]
    data = {
        "aggregate": {
            "full_crop": {"methods": ag, "reference_present_views": 6},
            "object": {"methods": {}, "reference_present_views": 0},
        },
        "thresholds": tau,
        "rows": rows,
    }
    out = {k: [] for k in ("object", "view", "curve", "bins")}
    identity = {
        "object_id": "fixture-a",
        "role": "debug",
        "family_id": "fixture-f",
        "condition": "M2",
        "arm": "main-256",
    }
    for domain in ("full_crop", "object"):
        method_rows(identity, data, domain, out)
    res = summaries(out["object"])
    good = [r for r in res if r["domain"] == "full_crop"][0]
    no = [r for r in res if r["domain"] == "object"][0]
    assert len(out["view"]) == 48 and len(out["curve"]) == 4 * 22
    assert (
        good["V_pool"] == 12
        and good["Y_pool"] == 6
        and good["AP_common_object_equal"] == 1
        and good["low_coverage_object_equal"] == 0.5
    )
    assert no["V_pool"] is None and no["AP_common_defined_objects"] == 0 and no["scheduled_views"] == 6
