"""Fetch DTU's official listing and explicit ZIP members with bounded HTTP ranges."""
import argparse
import hashlib
import io
import json
import re
import ssl
import struct
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path

from dtu_gate import verify_members

SITE = "https://roboimagedata.compute.dtu.dk/?page_id=36"
HOSTS = {"roboimagedata.compute.dtu.dk", "roboimagedata2.compute.dtu.dk"}
BLOCK = 256 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def official(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"https", "http"} or parsed.hostname not in HOSTS or parsed.username or parsed.password:
        raise ValueError("unexpected data origin")
    return url


class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, official(newurl))


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.href = None
        self.title = []
        self.links = {}

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.title = []

    def handle_data(self, data):
        if self.href:
            self.title.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.href:
            self.links["".join(self.title).strip()] = urllib.parse.urljoin(SITE, self.href)
            self.href = None


class Network:
    def __init__(self, budget, log):
        self.opener = urllib.request.build_opener(Redirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        self.budget = budget
        self.total = 0
        self.log = log

    def get(self, url, *, limit, span=None, total=None, etag=None):
        url = official(url)
        headers = {"Accept-Encoding": "identity", "User-Agent": "EvidenceLens-DTU-Research/1"}
        if span:
            headers["Range"] = "bytes=%d-%d" % span
        if etag and not etag.startswith("W/"):
            headers["If-Match"] = etag
        if limit > self.budget - self.total:
            raise ValueError("per-stage network budget exhausted")
        row = {"url": url, "range": span, "body_bytes": 0}
        try:
            with self.opener.open(urllib.request.Request(url, headers=headers), timeout=30) as response:
                row.update(status=response.status, final_url=official(response.geturl()))
                metadata = {k.lower(): response.headers[k] for k in ("Content-Length", "Content-Range", "ETag", "Last-Modified") if k in response.headers}
                row.update(metadata)
                if response.headers.get("Content-Encoding", "identity") not in {"identity", ""}:
                    raise ValueError("unexpected response encoding")
                if span:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", metadata.get("content-range", ""))
                    if response.status != 206 or not match:
                        raise ValueError("server did not honor Range; refused to stream full body")
                    lo, hi, length = map(int, match.groups())
                    if (lo, hi) != span or (total is not None and length != total):
                        raise ValueError("incorrect range or changed length")
                    if etag and metadata.get("etag") != etag:
                        raise ValueError("archive revision changed")
                advertised = metadata.get("content-length")
                if advertised is not None and int(advertised) > limit:
                    raise ValueError("response larger than registered cap")
                body = response.read(limit if span else limit + 1)
                row["body_bytes"] = len(body)
                self.total += len(body)
                if len(body) > limit or (span and len(body) != span[1] - span[0] + 1):
                    raise ValueError("truncated or excessive response")
                return body, metadata
        except Exception as exc:
            row["error"] = str(exc)
            raise
        finally:
            with self.log.open("a") as out:
                out.write(json.dumps(row) + "\n")


class RemoteZip(io.RawIOBase):
    def __init__(self, net, url):
        self.net = net
        self.url = url
        first, headers = net.get(url, limit=4, span=(0, 3))
        if first != b"PK\x03\x04":
            raise ValueError("official sample is not a conventional ZIP member archive")
        self.size = int(headers["content-range"].split("/")[-1])
        self.etag = headers.get("etag")
        self.modification = headers.get("last-modified")
        self.pos = self.start = 0
        self.block = b""

    def prime_member(self, info):
        """For photographs, cache precisely the one indexed member, without adjacent image data."""
        start = info.header_offset
        head, _ = self.net.get(self.url, span=(start, start + 29), total=self.size,
                               etag=self.etag, limit=30)
        if head[:4] != b"PK\x03\x04":
            raise ValueError("incorrect ZIP local header")
        filename_len, extra_len = struct.unpack_from("<HH", head, 26)
        length = filename_len + extra_len + info.compress_size
        if not 0 < length <= 32 * 1024**2 + 131070:
            raise ValueError("invalid member length")
        data, metadata = self.net.get(self.url, span=(start + 30, start + 29 + length),
                                      total=self.size, etag=self.etag, limit=length)
        if self.modification and metadata.get("last-modified") != self.modification:
            raise ValueError("archive changed during member read")
        self.start, self.block = start, head + data

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        position = (0, self.pos, self.size)[whence] + offset
        if position < 0:
            raise OSError("negative seek")
        self.pos = position
        return position

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n > 32 * 1024**2:
            raise ValueError("single ZIP read exceeds 32 MiB")
        result = bytearray()
        while len(result) < n:
            if not self.start <= self.pos < self.start + len(self.block):
                self.start = self.pos // BLOCK * BLOCK
                end = min(self.size, max(self.start + BLOCK, self.pos + n - len(result))) - 1
                self.block, hdr = self.net.get(self.url, span=(self.start, end), total=self.size,
                                               etag=self.etag, limit=end - self.start + 1)
                if self.modification and hdr.get("last-modified") != self.modification:
                    raise ValueError("archive Last-Modified changed")
            chunk = self.block[self.pos - self.start:self.pos - self.start + n - len(result)]
            self.pos += len(chunk)
            result.extend(chunk)
        return bytes(result)


def inventory(z):
    return [{"name": i.filename, "bytes": i.file_size, "compressed": i.compress_size,
             "crc32": "%08x" % i.CRC, "offset": i.header_offset, "method": i.compress_type}
            for i in z.infolist() if not i.is_dir()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    for key in ("view-set", "marker", "freeze"):
        p.add_argument("--" + key, type=Path)
    args = p.parse_args()
    spec = json.loads(args.request.read_text())
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cap = spec["network_cap_bytes"]
    if not 0 < cap <= 512 * 1024**2:
        raise ValueError("invalid bounded-member budget")
    net = Network(cap, out / "http.jsonl")
    action = spec["action"]
    try:
        if action == "catalog":
            data, _ = net.get(SITE, limit=1024**2)
            (out / "official.html").write_bytes(data)
            parser = Links()
            parser.feed(data.decode("utf-8"))
            keep = {key: official(parser.links[key]) for key in ("SampleSet", "Rectified", "Cleaned", "Points")}
            dump(out / "links.json", keep)
            return
        if action not in {"index", "members"}:
            raise ValueError("unknown acquisition stage")
        verify_members(spec, args.view_set, args.marker, args.freeze)
        stream = RemoteZip(net, official(spec["url"]))
        with zipfile.ZipFile(stream) as archive:
            rows = inventory(archive)
            identity = {"url": stream.url, "bytes": stream.size, "etag": stream.etag,
                        "last_modified": stream.modification, "directory_sha256": digest(json.dumps(rows, separators=(",", ":")).encode())}
            if action == "index":
                dump(out / "directory.json", rows)
                dump(out / "archive.json", identity)
                return
            if identity != spec["archive"]:
                raise ValueError("official archive identity differs from registered index")
            names = spec["members"]
            if not names or len(set(names)) != len(names) or sum(archive.getinfo(n).file_size for n in names) > spec["output_cap_bytes"]:
                raise ValueError("invalid member list or output budget")
            if len({row["name"] for row in rows}) != len(rows):
                raise ValueError("duplicate archive member")
            split_names = {name: name.split("/") for name in names}
            if any(any(x in {".", "..", ""} or "\\" in x or ":" in x for x in parts)
                   for parts in split_names.values()):
                raise ValueError("unsafe member path")
            hashes = []
            for name, parts in split_names.items():
                info = archive.getinfo(name)
                if info.file_size > 32 * 1024**2:
                    raise ValueError("individual member exceeds cap")
                if "/Rectified/" in name:
                    stream.prime_member(info)
                data = archive.read(info)  # verifies the official ZIP CRC
                dst = out.joinpath("members", *parts)
                dst.parent.mkdir(parents=True, exist_ok=True)
                with dst.open("xb") as f:
                    f.write(data)
                hashes.append({"member": name, "sha256": digest(data), "bytes": len(data), "crc32": "%08x" % info.CRC})
            dump(out / "members.json", {"archive": identity, "files": hashes})
    finally:
        dump(out / "network.json", {"body_bytes": net.total, "cap_bytes": cap, "action": action,
                                    "accounting": "HTTP entity bodies read, excluding transport headers"})


if __name__ == "__main__":
    main()
