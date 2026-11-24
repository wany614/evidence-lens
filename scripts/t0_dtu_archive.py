"""Read exact DTU central metadata and individual frozen source members over HTTP Range."""
import hashlib
import io
import json
import re
import struct
import zlib
import zipfile
from pathlib import PurePosixPath

from dtu_sample import Network


class MetadataError(ValueError):
    pass


def check(condition, message):
    if not condition:
        raise MetadataError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


class ExactArchive:
    """A Range is interpreted only after its predecessor has identified it as metadata or a named member."""

    def __init__(self, net, url, expected=None):
        self.net = net
        self.url = url
        first, hdr = net.get(url, limit=4, span=(0, 3))
        check(first == b"PK\x03\x04", "wrong first ZIP signature")
        self.length = int(hdr["content-range"].split("/")[-1])
        self.etag = hdr.get("etag")
        self.modified = hdr.get("last-modified")
        check(bool(self.etag) and not self.etag.startswith("W/"), "strong archive ETag required")
        self.identity = {"url": url, "bytes": self.length, "etag": self.etag, "last_modified": self.modified}
        if expected is not None:
            check(self.identity == {k: expected[k] for k in self.identity}, "archive identity differs from frozen directory")

    def exact(self, pos, length):
        check(0 < length <= 32 * 1024**2 and 0 <= pos <= self.length - length, "invalid ZIP range")
        body, hdr = self.net.get(self.url, limit=length, span=(pos, pos + length - 1), total=self.length, etag=self.etag)
        check(hdr.get("last-modified") == self.modified, "archive changed during transfer")
        return body

    def directory(self):
        end = self.length - 22
        eocd = struct.unpack("<4s4H2IH", self.exact(end, 22))
        sig, disk, start_disk, per_disk, count, length, offset, comment = eocd
        check(sig == b"PK\x05\x06" and (disk, start_disk, comment) == (0, 0, 0), "unsupported EOCD or ZIP comment")
        if 65535 in (per_disk, count) or 0xffffffff in (length, offset):
            lsig, ldisk, loc, disks = struct.unpack("<4sIQI", self.exact(end - 20, 20))
            check(lsig == b"PK\x06\x07" and ldisk == 0 and disks == 1, "invalid ZIP64 locator")
            zsig, size, _, _, zdisk, zstart, per_disk, count, length, offset = struct.unpack("<4sQ2H2I4Q", self.exact(loc, 56))
            check(zsig == b"PK\x06\x06" and zdisk == zstart == 0 and loc + 12 + size == end - 20, "invalid ZIP64 record")
            end = loc
        check(per_disk == count and offset + length == end, "unsupported ZIP central directory layout")
        raw = self.exact(offset, length)
        stream = io.BytesIO(raw)
        members = {}
        for _ in range(count):
            fields = struct.unpack("<4s6H3I5H2I", stream.read(46))
            sig, _, _, flags, method, _, _, crc, comp, size, nname, nextra, ncomment, _, _, _, local = fields
            check(sig == b"PK\x01\x02" and not flags & 1, "invalid or encrypted central record")
            encoded = stream.read(nname)
            name = encoded.decode("utf-8" if flags & 0x800 else "cp437")
            parts = PurePosixPath(name)
            check(name not in members and not parts.is_absolute() and ".." not in parts.parts and "\\" not in name, "unsafe or duplicate ZIP path")
            extra = stream.read(nextra)
            if 0xffffffff in (size, comp, local):
                fields64 = None
                index = 0
                while index + 4 <= len(extra):
                    token, n = struct.unpack_from("<HH", extra, index)
                    check(index + 4 + n <= len(extra), "truncated ZIP extra field")
                    if token == 1:
                        fields64 = io.BytesIO(extra[index + 4:index + 4 + n])
                        break
                    index += 4 + n
                check(fields64 is not None, "missing ZIP64 member sizes/offset")
                size, comp, local = [struct.unpack("<Q", fields64.read(8))[0] if v == 0xffffffff else v
                                     for v in (size, comp, local)]
            stream.seek(ncomment, 1)
            members[name] = {"name": name, "bytes": size, "compressed": comp, "offset": local,
                             "method": method, "flags": flags, "crc32": f"{crc:08x}"}
        check(stream.tell() == length, "central record size or count differs")
        return {**self.identity, "directory_offset": offset, "directory_bytes": length, "directory_sha256": digest(raw),
                "entries": count}, members, raw

    def member(self, row):
        offset = row["offset"]
        header = self.exact(offset, 30)
        sig, _, flags, method, _, _, crc, sz, raw_size, nname, nextra = struct.unpack("<4s5H3I2H", header)
        check(sig == b"PK\x03\x04" and flags == row["flags"] and method == row["method"], "local/central ZIP metadata disagree")
        name = self.exact(offset + 30, nname + nextra)[:nname].decode("utf-8" if flags & 0x800 else "cp437")
        check(name == row["name"], "different member at expected location")
        if not flags & 8:
            check(f"{crc:08x}" == row["crc32"] and sz in (row["compressed"], 0xffffffff)
                  and raw_size in (row["bytes"], 0xffffffff), "local member sizes/CRC differ")
        check(0 < row["bytes"] <= 32 * 1024**2 and 0 < row["compressed"] <= 32 * 1024**2, "member size outside limit")
        body = self.exact(offset + 30 + nname + nextra, row["compressed"])
        if method == zipfile.ZIP_STORED:
            data = body
        elif method == zipfile.ZIP_DEFLATED:
            decompressor = zlib.decompressobj(-15)
            data = decompressor.decompress(body, row["bytes"] + 1)
            check(decompressor.eof and not decompressor.unused_data and not decompressor.unconsumed_tail, "incomplete or excessive deflate stream")
        else:
            raise MetadataError("unsupported ZIP member compression")
        check(len(data) == row["bytes"] and f"{zlib.crc32(data):08x}" == row["crc32"], "uncompressed size or CRC differs")
        return data
