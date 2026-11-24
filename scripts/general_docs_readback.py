"""Seal a fresh revision read supplied by the authorized document client."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import time
import uuid

from evidence_lens._bundle import load, save, utc
from evidence_lens._verify import require


def text_sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_document(handoff, last_entry, out, expected=None):
    challenge = str(uuid.uuid4())
    started = utc()
    run_id = os.environ["EL_RUN_ID"]
    save(handoff / "waiting-for-read.json", {"challenge": challenge, "run_id": run_id, "at": started})
    response = handoff / "google-read.json"
    deadline = time.monotonic() + 240
    while not response.is_file() and time.monotonic() < deadline:
        time.sleep(1)
    require(response.is_file(), "the document client did not return a revision before the deadline")
    doc = load(response)
    require(doc["challenge"] == challenge and doc["revision"], "document read belongs to a different challenge")
    require(re.fullmatch(r"[0-9a-f]{64}", doc["full_text_sha256"]), "invalid document digest")

    entries = []
    for item in doc["entries"]:
        entry = item["entry"]
        value = item["text"]
        require(re.fullmatch(r"RL-[0-9]{3}", entry) and item["count"] == 1 and value.startswith(entry)
                and len(value) > len(entry), "missing or duplicate research log entry")
        entries.append({"entry": entry, "count": item["count"], "characters": len(value),
                        "paragraph_text_sha256": text_sha256(value)})
    names = [row["entry"] for row in entries]
    require(names and len(set(names)) == len(names) and names[-1] == last_entry, "different entry boundaries")
    linkage = {}
    if expected:
        registered = load(expected)
        require(registered["entry"] == last_entry and registered["revision_sha256"] == text_sha256(doc["revision"])
                and registered["entry_sha256"] == entries[-1]["paragraph_text_sha256"],
                "the readback differs from the registered write revision or text")
        require(all(registered[k] in doc["entries"][-1]["text"] for k in ("tag", "code_commit", "manifest_sha256")),
                "the last entry omits a release identity")
        linkage["verified_release_linkage"] = registered
    save(out / "research-log-readback.json", {
        "schema": "el-general-private-log-projection-1", "run_id": run_id,
        "begin_utc": started, "finish_utc": utc(), "revision_sha256": text_sha256(doc["revision"]),
        "document_text_sha256": doc["full_text_sha256"], "entry_checks": entries, **linkage,
        "read_boundary": "Authorized Google Docs connector parsed the current native document at its returned revision. Only digests and entry counts are public.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected", type=Path, help="Previously returned write revision and registered entry digests.")
    args = parser.parse_args()
    read_document(args.handoff, args.entry, args.out, args.expected)
