"""Measure an immutable local exchange publication and independently checksum the staged bytes."""

import argparse
import os
from pathlib import Path
import time

from evidence_lens._bundle import save, sha, utc
from make_handoff import publish


def transfer(source, destination, receipt):
    started = utc()
    before = time.monotonic()
    data = source.read_bytes()
    digest = sha(source)
    existing = destination.exists()
    publish(destination, data)
    if destination.stat().st_size != len(data) or sha(destination) != digest:
        raise ValueError("local exchange readback differs")
    save(receipt, {"schema":"el-general-exchange-local-1","run_id":os.environ["EL_RUN_ID"],
                   "start_utc":started,"finish_utc":utc(),"elapsed_seconds":time.monotonic()-before,
                   "filename":destination.name,"bytes":len(data),"sha256":digest,"already_present":existing,
                   "measured_boundary":"Local study storage to local synchronized project folder, including checksum readback.",
                   "provider_network_elapsed_seconds":None})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source","destination","receipt"):
        p.add_argument("--"+name, type=Path, required=True)
    a=p.parse_args()
    transfer(a.source,a.destination,a.receipt)
