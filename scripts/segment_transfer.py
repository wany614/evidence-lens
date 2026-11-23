"""Split or reconstruct an immutable private run ZIP for bounded-size artifact transports."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from evidence_lens._bundle import HASH, encoded, local_path, sha

SEGMENT_BYTES = 128 * 2**20


def segment(path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    source = hashlib.sha256()
    with path.open("rb") as inp:
        while block := inp.read(SEGMENT_BYTES):
            name = path.stem + f".part{len(rows) + 1:03}.bin"
            piece = destination / name
            d = hashlib.sha256(block).hexdigest()
            if piece.exists():
                if piece.is_symlink() or sha(piece) != d:
                    raise ValueError("existing transport segment differs")
            else:
                with tempfile.NamedTemporaryFile(dir=destination) as temp:
                    temp.write(block)
                    temp.flush()
                    os.fsync(temp.fileno())
                    try:
                        os.link(temp.name, piece)
                    except (PermissionError, OSError):
                        with piece.open("xb") as out:
                            out.write(block)
            rows.append({"name": name, "sha256": d, "bytes": len(block)})
            source.update(block)
    return {
        "name": path.name,
        "sha256": source.hexdigest(),
        "bytes": path.stat().st_size,
        "parts": rows,
    }


def reassemble(manifest, pieces, destination):
    if not HASH.fullmatch(manifest["sha256"]) or not manifest["parts"]:
        raise ValueError("invalid original package identity")
    seen = set()
    with tempfile.NamedTemporaryFile(dir=destination.parent) as joint:
        h = hashlib.sha256()
        size = 0
        for i, p in enumerate(manifest["parts"], 1):
            name = local_path(p["name"])
            if (
                name in seen
                or "/" in name
                or name != Path(manifest["name"]).stem + f".part{i:03}.bin"
                or not HASH.fullmatch(p["sha256"])
                or type(p["bytes"]) is not int
                or not 0 < p["bytes"] <= SEGMENT_BYTES
            ):
                raise ValueError("invalid segment manifest")
            piece = pieces / name
            if piece.is_symlink() or piece.stat().st_size != p["bytes"]:
                raise ValueError("segment size differs")
            d = hashlib.sha256()
            with piece.open("rb") as inp:
                while block := inp.read(1 << 20):
                    d.update(block)
                    h.update(block)
                    joint.write(block)
                    size += len(block)
            if d.hexdigest() != p["sha256"]:
                raise ValueError("segment hash differs")
            seen.add(name)
        joint.flush()
        if h.hexdigest() != manifest["sha256"] or size != manifest["bytes"]:
            raise ValueError("restored transfer bytes differ")
        if destination.exists():
            if sha(destination) != manifest["sha256"]:
                raise ValueError("existing destination differs")
        else:
            os.link(joint.name, destination)
    return {
        "reassembled_bytes": size,
        "sha256": manifest["sha256"],
        "segments": len(seen),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="op", required=True)
    c = s.add_parser("split")
    c.add_argument("--file", type=Path, required=True)
    c.add_argument("--folder", type=Path, required=True)
    c.add_argument("--manifest", type=Path, required=True)
    c = s.add_parser("join")
    c.add_argument("--manifest", type=Path, required=True)
    c.add_argument("--folder", type=Path, required=True)
    c.add_argument("--destination", type=Path, required=True)
    a = p.parse_args()
    if a.op == "split":
        rec = segment(a.file, a.folder)
        with a.manifest.open("xb") as f:
            f.write(encoded(rec))
    else:
        rec = reassemble(json.loads(a.manifest.read_bytes()), a.folder, a.destination)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
