"""Verify private Drive family segments and restore original recorder files within a 4 GiB working budget."""

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from evidence_lens._bundle import HASH, RUN_ID, encoded, freeze_tree, load, local_path, sha
from evidence_lens._verify import require, verify_bundle
from make_handoff import publish
from stage2_confirmation_archive import DOC, contents
from run_stage2_confirmation import lock

BUDGET = 4 * 2**30
PART = 128 * 2**20
FLOOR_MARGIN = 2**20


def provider_location(store=None, path=None, digest=None):
    """Authenticate the private account-specific Drive folder selection; no identifiers live in this source."""
    if path is None and digest is None:
        raw = os.environ.get("EVIDENCE_LENS_DRIVE_ARCHIVE_CONFIG")
        wanted = os.environ.get("EVIDENCE_LENS_DRIVE_ARCHIVE_CONFIG_SHA256")
        if raw is not None or wanted is not None:
            path, digest = Path(raw) if raw else None, wanted
        else:
            path = (
                Path(store) if store is not None else Path(".research")
            ).resolve() / "protocol/stage2/cloud-archive-location.json"
            sidecar = path.with_suffix(".sha256")
            require(
                sidecar.is_file() and not sidecar.is_symlink(),
                "A private anchored Drive location is required",
            )
            digest = sidecar.read_text(encoding="ascii").strip()
    require(
        path is not None and isinstance(digest, str) and HASH.fullmatch(digest),
        "Supply the private Drive location and its expected SHA together",
    )
    path = Path(path)
    require(
        path.is_file() and not path.is_symlink() and sha(path) == digest,
        "Private Drive location digest differs",
    )
    value = load(path)
    roles = ("sealed_runs_parent_id", "scientific_archive_parent_id", "engineering_fixture_parent_id")
    require(
        set(value) == {"schema", "provider", *roles}
        and value["schema"] == "el-stage2-drive-location-1"
        and value["provider"] == "Google Drive"
        and all(
            isinstance(value[k], str) and re.fullmatch(r"[A-Za-z0-9_-]{5,200}", value[k]) for k in roles
        )
        and len({value[k] for k in roles}) == len(roles),
        "Invalid or overlapping private Drive folder roles",
    )
    return {**value, "sha256": digest}


def role_parent(location, fixture):
    return location["engineering_fixture_parent_id" if fixture else "scientific_archive_parent_id"]


def offer(source, digest, folder, proof_dir, *, fixture=False, part_bytes=PART, location=None):
    location = location or provider_location()
    doc, _ = contents(source, digest)
    require(0 < part_bytes <= PART and (fixture or part_bytes == PART), "Invalid scientific chunk size")
    if fixture:
        require(
            doc["identity"].get("purpose") == "synthetic_engineering_fixture", "Not a synthetic archive"
        )
    else:
        require(
            isinstance(doc["family"], str) and HASH.fullmatch(doc["identity"]["authorization_sha256"]),
            "Archive lacks real opening provenance",
        )
    total = source.stat().st_size
    name = "synthetic-recovery-family-" + digest[:8] if fixture else doc["family"] + "-" + digest[:12]
    rows = []
    folder.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    with source.open("rb") as f:
        while block := f.read(part_bytes):
            n = len(rows)
            part = name + ".zip" if total <= part_bytes else name + f".part{n + 1:03}.bin"
            h = hashlib.sha256(block).hexdigest()
            source_hash.update(block)
            publish(folder / part, block)
            rows.append({"name": part, "bytes": len(block), "sha256": h, "offset": n * part_bytes})
    require(source_hash.hexdigest() == digest, "Archive changed during publication")
    report = {
        "schema": "el-stage2-private-cloud-offer-1",
        "fixture": fixture,
        "family": doc["family"],
        "family_identity": doc["identity"],
        "remote_parent_id": role_parent(location, fixture),
        "private_location_sha256": location["sha256"],
        "archive_sha256": digest,
        "archive_bytes": total,
        "archive_index_sha256": hashlib.sha256(encoded(doc)).hexdigest(),
        "part_limit_bytes": part_bytes,
        "recovery_budget_bytes": BUDGET,
        "parts": rows,
        "run_manifests": doc["run_manifests"],
    }
    dest = proof_dir / (name + ".offer.json")
    publish(dest, encoded(report))
    publish(folder / dest.name, encoded(report))
    return dest


def read_offer(path, digest, location=None):
    location = location or provider_location()
    require(sha(path) == digest, "Offer digest mismatch")
    offer = load(path)
    require(
        offer["schema"] == "el-stage2-private-cloud-offer-1"
        and offer["remote_parent_id"] == role_parent(location, offer["fixture"])
        and (
            offer.get("private_location_sha256") == location["sha256"]
            or (offer["fixture"] is True and "private_location_sha256" not in offer)
        )
        and offer["recovery_budget_bytes"] == BUDGET
        and 0 < offer["part_limit_bytes"] <= PART
        and HASH.fullmatch(offer["archive_sha256"]),
        "Unexpected restore plan",
    )
    rows = offer["parts"]
    n = 0
    names = set()
    for i, r in enumerate(rows):
        name = local_path(r["name"])
        require(
            "/" not in name
            and name not in names
            and HASH.fullmatch(r["sha256"])
            and r["offset"] == n
            and 0 < r["bytes"] <= offer["part_limit_bytes"]
            and (i == len(rows) - 1 or r["bytes"] == offer["part_limit_bytes"]),
            "Bad or overlapping part",
        )
        names.add(name)
        n += r["bytes"]
    require(rows and n == offer["archive_bytes"], "Missing archive bytes")
    return offer


def metadata(offer, rows):
    require(
        len(rows) == len(offer["parts"]) and len({r["id"] for r in rows}) == len(rows),
        "Need one distinct Drive object for every registered part",
    )
    values = []
    for item, p in zip(rows, offer["parts"]):
        require(
            item["name"] == p["name"]
            and int(item["bytes"]) == p["bytes"]
            and item["parent_ids"] == [offer["remote_parent_id"]]
            and item["shared"] is False
            and isinstance(item["id"], str)
            and item["id"]
            and isinstance(item["mime_type"], str)
            and item["mime_type"]
            and not item["mime_type"].startswith("application/vnd.google-apps."),
            "Drive metadata does not match registered private folder, name and size",
        )
        parsed = urllib.parse.urlsplit(item["download_url"])
        host = parsed.hostname or ""
        require(
            parsed.scheme == "https"
            and not parsed.username
            and not parsed.password
            and (
                host.endswith(".googleapis.com")
                or host.endswith(".googleusercontent.com")
                or host.endswith(".oaiusercontent.com")
                or host in ("drive.google.com", "drive.usercontent.google.com")
            ),
            "Expected authenticated provider or connector download",
        )
        values.append({k: item[k] for k in ("id", "name", "bytes", "parent_ids", "shared", "mime_type")})
    return values


def fetch_part(item, expected, dst, full=None):
    h = hashlib.sha256()
    n = 0
    start = time.monotonic()
    request = urllib.request.Request(
        item["download_url"], headers={"User-Agent": "Evidence-Lens-private-research-verification"}
    )
    with urllib.request.urlopen(request, timeout=120) as source:
        while block := source.read(1 << 20):
            n += len(block)
            require(n <= expected["bytes"], "Excess Drive bytes")
            h.update(block)
            if full is not None:
                full.update(block)
            if dst is not None:
                dst.write(block)
    require(
        n == expected["bytes"] and h.hexdigest() == expected["sha256"],
        "Independent downloaded part differs from registered source",
    )
    return {"bytes": n, "sha256": h.hexdigest(), "elapsed_seconds": time.monotonic() - start}


class RemoteArchive(io.RawIOBase):
    """ZIP seek support over checked Drive chunks. At most one chunk is cached in memory."""

    def __init__(self, offer, entries):
        self.offer = offer
        self.entries = entries
        self.pos = 0
        self.index = None
        self.buffer = None

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        end = self.offer["archive_bytes"]
        target = offset if whence == 0 else self.pos + offset if whence == 1 else end + offset
        require(0 <= target <= end, "Unexpected remote ZIP seek")
        self.pos = target
        return target

    def read(self, size=-1):
        end = self.offer["archive_bytes"]
        remain = end - self.pos if size < 0 else min(size, end - self.pos)
        out = bytearray()
        require(remain <= 256 * 2**20, "Archive member request is too large to read as one block")
        while remain:
            i = self.pos // self.offer["part_limit_bytes"]
            p = self.offer["parts"][i]
            if i != self.index:
                self.buffer = None
                b = io.BytesIO()
                fetch_part(self.entries[i], p, b)
                self.buffer = b
                self.index = i
            off = self.pos - p["offset"]
            n = min(remain, p["bytes"] - off)
            self.buffer.seek(off)
            out.extend(self.buffer.read(n))
            self.pos += n
            remain -= n
        return bytes(out)


def bounded_restore(offer, rows, ids, output, index, workspace, budget=BUDGET, provider_spool=None):
    require(
        set(ids) <= set(offer["run_manifests"]) and len(set(ids)) == len(ids) and ids, "Unexpected run list"
    )
    require(0 < budget <= BUDGET, "Recovery budget exceeds registered maximum")
    unique = {}
    logical = {}
    for n, v in index["files"].items():
        local_path(n)
        if not n.startswith("runs/") or n.split("/")[1] in ids:
            logical[n] = v
            unique[v["sha256"]] = v["bytes"]
    # Returning an already verified restoration only scans it and never mutates it.
    marker = output / "restore.json"
    config = {
        "schema": "el-stage2-bound-restore-1",
        "archive_sha256": offer["archive_sha256"],
        "runs": sorted(ids),
        "bytes_written": sum(unique.values()),
        "logical_files": len(logical),
        "source": "independently_downloaded_Google_Drive_segments",
    }
    if output.exists():
        require(
            marker.is_file()
            and load(marker) == config
            and all((output / n).is_file() and sha(output / n) == v["sha256"] for n, v in logical.items())
            and all(
                verify_bundle(output / "runs" / rid)["manifest_sha256"] == offer["run_manifests"][rid]
                for rid in ids
            ),
            "Existing restoration differs from its original logical files",
        )
        return config
    workspace.mkdir(exist_ok=True, parents=True)
    output.parent.mkdir(exist_ok=True, parents=True)
    require(
        workspace.stat().st_dev == output.parent.stat().st_dev,
        "Atomic restoration requires same-volume staging",
    )
    rounded = sum(((v + 4095) // 4096) * 4096 for v in unique.values())
    estimate = (
        rounded
        + min(offer["archive_bytes"], offer["part_limit_bytes"])
        + FLOOR_MARGIN
        + 4096 * len(logical)
    )
    require(
        estimate <= budget,
        "Requested restoration exceeds 4 GiB recovery limit; use one registered run at a time",
    )
    require(
        shutil.disk_usage(workspace).free >= estimate, "Insufficient local space for bounded restoration"
    )
    stream = provider_spool if provider_spool is not None else RemoteArchive(offer, rows)
    with (
        tempfile.TemporaryDirectory(dir=workspace, prefix=".bounded-drive-") as tmp,
        zipfile.ZipFile(stream) as z,
    ):
        tmp = Path(tmp)
        obj = tmp / "objects"
        tree = tmp / "tree"
        obj.mkdir()
        tree.mkdir()
        # Follow original ZIP order so large archives fetch each occupied segment sequentially.
        for h in sorted(unique, key=lambda key: z.getinfo("objects/" + key).header_offset):
            path = obj / h
            n = 0
            dig = hashlib.sha256()
            with z.open("objects/" + h) as inp, path.open("xb") as dst:
                while block := inp.read(1 << 20):
                    n += len(block)
                    dig.update(block)
                    dst.write(block)
            require(
                n == unique[h] and dig.hexdigest() == h, "Restored remote object differs from its manifest"
            )
            path.chmod(0o444)
        for name, v in logical.items():
            p = tree / name
            p.parent.mkdir(exist_ok=True, parents=True)
            os.link(obj / v["sha256"], p)
        for rid in ids:
            rp = tree / "runs" / rid
            require(
                verify_bundle(rp)["manifest_sha256"] == offer["run_manifests"][rid],
                "Restored recorder manifest differs",
            )
            freeze_tree(rp)
        publish(tree / "restore.json", encoded(config))
        os.rename(tree, output)
    return config


@contextlib.contextmanager
def recovery_workspace(root):
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / "recovery-lock.sqlite3"):
        tmp = root / ".stage2-private-recovery"
        owner = {
            "schema": "el-stage2-private-recovery-workspace-1",
            "managed_contents": "transient downloaded bytes, independent of original scientific stores",
        }
        if tmp.exists():
            require(
                not tmp.is_symlink()
                and (tmp / "owner.json").is_file()
                and load(tmp / "owner.json") == owner,
                "Recovery staging directory has an unknown owner",
            )
        else:
            tmp.mkdir()
            publish(tmp / "owner.json", encoded(owner))
        working = tmp / "in-flight"
        if working.exists():
            require(not working.is_symlink() and working.is_dir(), "Unexpected recovery staging path")
            # This directory belongs solely to our previous interrupted I/O under the same exclusive lock.
            shutil.rmtree(working)
        working.mkdir()
        try:
            yield working
        finally:
            shutil.rmtree(working)


def confirm(
    offer_path, expected, rows, workspace, output, *, audit=False, restore=None, rids=None, location=None
):
    offer = read_offer(offer_path, expected, location)
    sources = metadata(offer, rows)
    output.mkdir(exist_ok=True, parents=True)
    with recovery_workspace(workspace) as active:
        return retrieve(offer, expected, rows, sources, active, output, audit, restore, rids)


def retrieve(offer, expected, rows, sources, active, output, audit, restore, rids):
    proof_path = output / "cloud-proof.json"
    index_path = output / "independent-archive-index.json"
    pending = []
    if audit:
        for rid in sorted(offer["run_manifests"]):
            p = output / "restored-runs" / (rid + ".json")
            if p.is_file():
                r = load(p)
                require(
                    r["archive_sha256"] == offer["archive_sha256"] and r["runs"] == [rid],
                    "Wrong previous per-run cloud restoration",
                )
            else:
                pending.append(rid)
    needs_bytes = bool(
        pending or (restore and not restore.exists()) or ((audit or restore) and not index_path.is_file())
    )
    # Small and medium archives need one independent sequential download even when many runs are audited.
    # A large archive is instead read by pinned segment, keeping the full 4 GiB for the original run.
    spool = (
        active / "independently-downloaded.zip"
        if needs_bytes and offer["archive_bytes"] <= BUDGET // 2
        else None
    )
    proof = None
    if proof_path.is_file():
        proof = load(proof_path)
        require(
            proof["offer_sha256"] == expected
            and proof["objects"] == sources
            and proof["bytes"] == offer["archive_bytes"]
            and proof["archive_sha256"] == offer["archive_sha256"]
            and proof["archive_index_sha256"] == offer["archive_index_sha256"],
            "Published cloud proof differs",
        )
    if proof is None or spool is not None:
        if spool is not None:
            require(
                shutil.disk_usage(active).free >= offer["archive_bytes"] + FLOOR_MARGIN,
                "Local provider spool would exceed available disk",
            )
        full = hashlib.sha256()
        measured = []
        with spool.open("xb") if spool is not None else contextlib.nullcontext(None) as dst:
            for item, part in zip(rows, offer["parts"]):
                measured.append(fetch_part(item, part, dst, full))
        require(full.hexdigest() == offer["archive_sha256"], "Full independent archive digest differs")
        if proof is None:
            proof = {
                "schema": "el-stage2-private-cloud-proof-1",
                "offer_sha256": expected,
                "archive_sha256": full.hexdigest(),
                "archive_index_sha256": offer["archive_index_sha256"],
                "bytes": sum(r["bytes"] for r in measured),
                "objects": sources,
                "part_receipts": measured,
                "family": offer["family"],
                "fixture": offer["fixture"],
                "run_manifests": offer["run_manifests"],
                "download_path": "authenticated Drive files.get and raw fetch; verified independent HTTPS bytes",
                "workspace_budget_bytes": BUDGET,
            }
            publish(proof_path, encoded(proof))
    if not (audit or restore):
        return proof
    if index_path.is_file():
        raw = index_path.read_bytes()
    else:
        with zipfile.ZipFile(spool if spool is not None else RemoteArchive(offer, rows)) as z:
            raw = z.read(DOC)
        require(
            hashlib.sha256(raw).hexdigest() == offer["archive_index_sha256"],
            "Provider ZIP returned an incorrect logical index",
        )
        publish(index_path, raw)
    require(
        hashlib.sha256(raw).hexdigest() == offer["archive_index_sha256"], "Remote logical file map differs"
    )
    doc = json.loads(raw)
    require(
        doc["identity"] == offer["family_identity"] and doc["run_manifests"] == offer["run_manifests"],
        "Wrong scientific archive identity",
    )
    available = BUDGET - (offer["archive_bytes"] if spool else 0)
    for rid in pending:
        with tempfile.TemporaryDirectory(dir=active, prefix=".remote-audit-") as scratch:
            restored = bounded_restore(
                offer, rows, [rid], Path(scratch) / "restored", doc, active, available, spool
            )
            publish(output / "restored-runs" / (rid + ".json"), encoded(restored))
    if restore:
        bounded_restore(
            offer, rows, rids or sorted(offer["run_manifests"]), restore, doc, active, available, spool
        )
    return proof


def certify(store, shelf, offer_path, offer_sha, proof_path, proof_sha, location=None):
    from evidence_lens._index import writer_gate
    from evidence_lens.archive import make_receipt, project_receipt
    from working_set import verify_receipt

    location = location or provider_location(store)
    offer = read_offer(offer_path, offer_sha, location)
    require(
        not offer["fixture"] and sha(proof_path) == proof_sha,
        "Only a verified scientific provider proof can certify production runs",
    )
    q = load(proof_path)
    ident = offer["family_identity"]
    family = offer["family"]
    require(
        q["schema"] == "el-stage2-private-cloud-proof-1"
        and q["fixture"] is False
        and q["family"] == family
        and q["offer_sha256"] == offer_sha
        and q["archive_sha256"] == offer["archive_sha256"]
        and q["bytes"] == offer["archive_bytes"]
        and q["run_manifests"] == offer["run_manifests"]
        and q["archive_index_sha256"] == offer["archive_index_sha256"],
        "Unrelated or incomplete provider proof",
    )
    require(
        len(q["objects"]) == len(q["part_receipts"]) == len(offer["parts"]), "Unverified provider segments"
    )
    for remote, readback, part in zip(q["objects"], q["part_receipts"], offer["parts"]):
        require(
            remote["name"] == part["name"]
            and remote["parent_ids"] == [role_parent(location, False)]
            and remote["shared"] is False
            and int(remote["bytes"]) == readback["bytes"] == part["bytes"]
            and readback["sha256"] == part["sha256"],
            "Provider object or original segment differs",
        )
    object_ids = [r["id"] for r in q["objects"]]
    require(len(set(object_ids)) == len(object_ids), "Repeated provider object")
    base = store / "stage2" / ident["catalog_sha256"] / "confirmation"
    family_file = base / "families" / (family + ".json")
    require(
        sha(family_file) == ident["family_sha256"]
        and sha(base / "authorization.json") == ident["authorization_sha256"],
        "Wrong archive selection or opening",
    )
    expected = load(family_file)
    known = {
        r["run_id"]: r["manifest_sha256"]
        for trial in expected["trials"].values()
        for r in (*trial["run_pins"].values(), *trial["returns"])
    }
    require(
        all(q["run_manifests"].get(r) == v for r, v in known.items()),
        "Provider archive misses original family pins",
    )
    for rid, h in q["run_manifests"].items():
        require(RUN_ID.fullmatch(rid) and HASH.fullmatch(h), "Malformed original identity")
        paths = [root / "runs" / rid for root in (store, shelf) if (root / "runs" / rid).is_dir()]
        require(
            len(paths) == 1
            and sha(paths[0] / "manifest.json") == h
            and load(paths[0] / "run.json")["scene_family_id"] == family,
            "Unrelated recorder run in archive",
        )
        audit = load(proof_path.parent / "restored-runs" / (rid + ".json"))
        require(
            audit["schema"] == "el-stage2-bound-restore-1"
            and audit["archive_sha256"] == q["archive_sha256"]
            and audit["runs"] == [rid],
            "Run-independent Drive restoration missing",
        )
    dest = base / "archives" / (q["archive_sha256"] + ".json")
    with writer_gate(store):
        if dest.is_file():
            old = load(dest)
            require(
                old["provider_proof_sha256"] == proof_sha and old["object_ids"] == object_ids,
                "Existing provider receipt differs",
            )
            for rid, r in old["receipts"].items():
                verify_receipt(store, rid, r["manifest_sha256"], r["archive_receipt_id"])
            return old
        receipts = {}
        for rid, h in q["run_manifests"].items():
            f, r = make_receipt(
                store,
                "archive",
                run_id=rid,
                manifest_sha256=h,
                status="verified",
                reason=None,
                container_sha256=q["archive_sha256"],
                bytes=q["bytes"],
                provider="Google Drive",
                remote_object_ids=object_ids,
                remote_relative_path="Sealed runs/Stage B confirmation v1/" + offer_path.name,
                provider_proof_sha256=proof_sha,
                offer_sha256=offer_sha,
                verification_method="Individually downloaded all segments, verified full archive and independently restored this original recorder run within 4 GiB.",
            )
            project_receipt(store, f)
            receipts[rid] = {"manifest_sha256": h, "archive_receipt_id": r["receipt_id"]}
        out = {
            "schema": "el-stage2-family-cloud-archive-1",
            "family": family,
            "container_sha256": q["archive_sha256"],
            "object_ids": object_ids,
            "provider_proof_sha256": proof_sha,
            "offer_sha256": offer_sha,
            "receipts": receipts,
        }
        publish(dest, encoded(out))
        return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="mode", required=True)
    o = subs.add_parser("offer")
    for k in ("source", "folder", "proof-dir"):
        o.add_argument("--" + k, required=True, type=Path)
    o.add_argument("--sha256", required=True)
    o.add_argument("--fixture", action="store_true")
    c = subs.add_parser("confirm")
    for k in ("offer", "workspace", "output"):
        c.add_argument("--" + k, required=True, type=Path)
    c.add_argument("--sha256", required=True)
    c.add_argument("--audit-runs", action="store_true")
    c.add_argument("--restore", type=Path)
    c.add_argument("--run", action="append")
    reg = subs.add_parser("certify")
    for n in ("store", "shelf", "offer", "proof"):
        reg.add_argument("--" + n, required=True, type=Path)
    for n in ("offer-sha256", "proof-sha256"):
        reg.add_argument("--" + n, required=True)
    for parser in (o, c, reg):
        parser.add_argument("--provider-config", type=Path, help="Private Drive folder declaration")
        parser.add_argument("--provider-config-sha256", help="Expected SHA of the private declaration")
    a = p.parse_args(argv)
    location = provider_location(getattr(a, "store", None), a.provider_config, a.provider_config_sha256)
    if a.mode == "certify":
        print(
            json.dumps(
                certify(a.store, a.shelf, a.offer, a.offer_sha256, a.proof, a.proof_sha256, location)
            )
        )
        return
    if a.mode == "offer":
        dest = offer(a.source, a.sha256, a.folder, a.proof_dir, fixture=a.fixture, location=location)
        print(json.dumps({"offer": str(dest), "sha256": sha(dest)}))
        return
    rows = json.loads(sys.stdin.readline())
    p = confirm(
        a.offer,
        a.sha256,
        rows,
        a.workspace,
        a.output,
        audit=a.audit_runs,
        restore=a.restore,
        rids=a.run,
        location=location,
    )
    print(json.dumps(p))


if __name__ == "__main__":
    main()
