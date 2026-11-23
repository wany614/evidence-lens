"""Turn a stage's optional metrics.jsonl into the typed run-level table."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ._bundle import BundleError

TEXT = ("run_id trial_id method_id model_id scene_family_id scene_id condition_id "
        "camera_id metric_name metric_version unit status missing_reason mask_id mask_hash "
        "region_name label_version threshold_version").split()
COUNTS = ("denominator reference_pixel_count prediction_valid_count "
          "scoreable_count").split()
SCHEMA = pa.schema([(key, pa.string()) for key in TEXT] + [("value", pa.float64()), ("numerator", pa.float64())]
                   + [(key, pa.int64()) for key in COUNTS]
                   + [("source_artifact_ids", pa.list_(pa.string()))])


def export(bundle: Path, identity: dict, *, empty: bool = False) -> None:
    rows = []
    source = bundle / "payload" / "metrics.jsonl"
    if source.is_file() and not empty:
        with source.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                row = json.loads(line)
                if not isinstance(row, dict) or (row.keys() - SCHEMA.names):
                    raise BundleError(f"invalid metric fields at line {number}")
                for key in ("run_id", "trial_id", "scene_family_id", "scene_id"):
                    if key in row and identity.get(key) is not None and row[key] != identity[key]:
                        raise BundleError(f"metric {key} differs from the run at line {number}")
                    row.setdefault(key, identity.get(key))
                if not all(row.get(key) for key in ("metric_name", "metric_version", "unit", "status")):
                    raise BundleError(f"missing metric identity at line {number}")
                if row.get("value") is None:
                    if not row.get("missing_reason"):
                        raise BundleError(f"missing metric value needs a reason at line {number}")
                elif not math.isfinite(row["value"]):
                    raise BundleError(f"non-finite metric value at line {number}")
                rows.append(row)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, bundle / "metrics.parquet", compression="zstd")
    with (bundle / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA.names)
        writer.writeheader()
        for row in table.to_pylist():
            if row["source_artifact_ids"] is not None:
                row["source_artifact_ids"] = json.dumps(row["source_artifact_ids"])
            writer.writerow(row)
