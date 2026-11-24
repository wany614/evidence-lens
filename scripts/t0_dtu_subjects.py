"""Extract only scan IDs and subject names from an independently authored textual DTU table."""
import argparse
import json
import re
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from t0_dtu_archive import check, digest, write_json

URL = "https://arxiv.org/html/2610.06472v1"


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.table = None
        self.row = None
        self.cell = None
        self.tables = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.table = []
        elif self.table is not None and tag == "tr":
            self.row = []
        elif self.row is not None and tag in ("td", "th"):
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        if tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("unregistered publication redirect")


def subjects(config, out, local=None):
    cap = config["network_cap_bytes"]
    check(0 < cap <= 8 * 1024**2, "invalid HTML transfer cap")
    start = time.monotonic()
    log = {"url": URL, "body_bytes": 0, "cap_bytes": cap, "subresources_requested": 0}
    try:
        if local is not None:
            raw = local.read_bytes()
            check(digest(raw) == config["captured_publication_sha256"], "captured publication changed")
            log.update(source="sealed HTTP response from the recorded acquisition attempt", observed_html_sha256=digest(raw))
        else:
            request = urllib.request.Request(URL, headers={"Accept-Encoding": "identity", "User-Agent": "EvidenceLens-DTU-Research/1"})
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
                log.update(status=response.status, final_url=response.geturl(), content_type=response.headers.get_content_type())
                check(response.status == 200 and response.geturl() == URL and log["content_type"] == "text/html",
                      "unexpected versioned publication response")
                check(int(response.headers.get("Content-Length", 0)) <= cap, "oversized HTML response")
                raw = response.read(cap + 1)
                log["body_bytes"] = len(raw)
        check(len(raw) <= cap, "HTML exceeded cap")
        (out / "publication.html").write_bytes(raw)
        parser = Tables()
        parser.feed(raw.decode("utf-8"))
        candidates = [t for t in parser.tables if t and t[0][:2] == ["Scene", "Object"] and "Object Ratio" in " ".join(t[0])
                      and t[1:] and all(r and re.fullmatch(r"scan[1-9][0-9]*", r[0]) for r in t[1:])]
        check(len(candidates) == 1, "expected one dataset-configuration table")
        table = candidates[0]
        ids = {}
        for row in table[1:]:
            check(len(row) >= 2 and re.fullmatch(r"scan[1-9][0-9]*", row[0]), "unexpected dataset configuration row")
            n = int(row[0][4:])
            check(n not in ids and 0 < len(row[1]) <= 80, "duplicate scan or missing subject label")
            ids[n] = row[1]
        check(len(ids) >= 20, "incomplete object metadata")
        write_json(out / "identity-labels.json", {"schema": "el-t0-literature-identity-1", "url": URL, "version": "2610.06472v1",
                   "locator": "Appendix, Table 12, DTU dataset Configuration; Scene and Object columns",
                   "raw_publication_sha256": digest(raw), "objects": ids, "scope": "author-named primary subjects, not exhaustive inventories of every background accessory"})
    except Exception as e:
        log["error"] = str(e)
        raise
    finally:
        log["elapsed_seconds"] = time.monotonic() - start
        write_json(out / "network.json", log)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--publication", type=Path)
    a = p.parse_args()
    subjects(json.loads(a.config.read_bytes()), a.out, a.publication)
