"""Record the general closeout without mixing its trials with either preceding batch."""

from t0_records import bind, record, resolve


def stage(store, history, name, kind, pins, details, command, member, size, extra=None, retry=None):
    return record(store, history, name, kind, pins, {
        **details, "experiment_id": "general-1", "trial_id": "general-" + name,
    }, command, member, size, extra, retry)
