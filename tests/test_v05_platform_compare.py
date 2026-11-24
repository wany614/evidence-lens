import numpy as np

from v05_platform_compare import compare_view, summary


def test_changed_support_and_tie_membership_are_separate_from_common_finite_thresholds():
    def channels(values):
        risk = np.array(values, np.float32).reshape(2, 5)
        valid = np.isfinite(risk)
        state = np.where(valid, risk >= 0.5, 255).astype(np.uint8)
        return {"risk": risk, "valid": valid, "state": state, "unknown_reason": (~valid).astype(np.uint8)}

    mac = channels([0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, np.nan])
    other = channels([0.15, 0.05, 0.1, 0.2, 0.3, 0.49, 0.6, 0.7, np.nan, 0.8])
    diff = compare_view(mac, other, 0.5)
    assert diff["changed_pixels"] == {"risk": 5, "state": 3, "valid": 2, "unknown_reason": 2}
    assert diff["common_finite"] == 8 and diff["threshold_crossings"] == 1
    assert [q["added"] for q in diff["quota_on_method_finite"]] == [0, 1]
    assert [q["removed"] for q in diff["quota_on_method_finite"]] == [0, 1]
    total = summary([diff, compare_view(mac, mac, 0.5)])
    assert total["bit_exact_views"] == 1 and total["common_finite"] == 17
    assert total["changed_method_finite_quota_tests"] == 1 and total["method_finite_quota_tests"] == 4
