"""Store the exact engineering comparison protocol before generating comparison data."""

import argparse
import hashlib
import json
import os
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = args.protocol.read_bytes()
    protocol = json.loads(data)
    assert protocol["schema"] == "el-asset-acceptance-protocol-1"
    (args.out / "protocol.json").write_bytes(data)
    receipt = {
        "schema": "el-asset-registration-1",
        "run_id": os.environ["EL_RUN_ID"],
        "protocol_sha256": hashlib.sha256(data).hexdigest(),
        "comparisons": protocol["comparisons"],
    }
    (args.out / "registration.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
