"""Turn a stage's optional metrics.jsonl into the typed run-level table."""

from __future__ import annotations

import csv
import json
import math
from itertools import zip_longest
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


def verify(bundle: Path, identity: dict, hashes: set[str]) -> None:
    parquet = pq.ParquetFile(bundle / "metrics.parquet")
    columns = parquet.schema_arrow
    if set(columns.names) != set(SCHEMA.names) or len(columns.names) != len(SCHEMA):
        raise BundleError("unexpected metric columns")
    for field in SCHEMA:
        actual = columns.field(field.name).type
        # Early P0 smoke bundles contained an empty integer numerator column.
        legacy_empty = field.name == "numerator" and parquet.metadata.num_rows == 0 and actual == pa.int64()
        if actual != field.type and not legacy_empty:
            raise BundleError(f"invalid metric column type: {field.name}")
    expected_rows = (row for batch in parquet.iter_batches(batch_size=4096) for row in batch.to_pylist())
    with (bundle / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        records = csv.DictReader(handle)
        if records.fieldnames != columns.names:
            raise BundleError("CSV and Parquet metric columns differ")
        for row, human in zip_longest(expected_rows, records):
            if row is None or human is None:
                raise BundleError("CSV and Parquet metric counts differ")
            if row["run_id"] != identity["run_id"] or row["trial_id"] != identity["trial_id"]:
                raise BundleError("metric belongs to a different run/trial")
            for ref in row["source_artifact_ids"] or []:
                if ":sha256:" in ref:
                    digest = ref.rsplit(":sha256:", 1)[1]
                elif ref.startswith("sha256:"):
                    digest = ref[7:]
                else:
                    raise BundleError("metric lacks a content-addressed source asset")
                if digest not in hashes:
                    raise BundleError(f"metric source asset missing from bundle: {digest}")
            for key, actual in row.items():
                if key == "source_artifact_ids" and actual is not None:
                    actual = json.dumps(actual)
                if human[key] != ("" if actual is None else str(actual)):
                    raise BundleError(f"CSV and Parquet metric values differ: {key}")
