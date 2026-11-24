"""Acquire text-only identity evidence or the complete official ZIP central directory."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from dtu_sample import Network
from t0_dtu_archive import ExactArchive, check, digest, write_json


BASE = "https://roboimagedata2.compute.dtu.dk/data/"
PAPERS = {"cvpr-2014": BASE + "text/multiViewCVPR2014.pdf", "ijcv-2016": BASE + "text/IJCV_MVS_Stereo_data_2016.pdf"}
PDF_TEXT = """
ObjC.import("Foundation"); ObjC.import("PDFKit");
function run(argv) {
    var pdf = $.PDFDocument.alloc.initWithURL($.NSURL.fileURLWithPath(argv[0]));
    if (!pdf) throw Error("bad publication PDF");
    var pages = [];
    for (var i = 0; i < pdf.pageCount; i++) pages.push(ObjC.unwrap(pdf.pageAtIndex(i).string));
    return JSON.stringify(pages);
}
"""


def metadata(config, out):
    action = config["action"]
    cap = config["network_cap_bytes"]
    check(0 < cap <= 96 * 1024**2, "invalid network budget")
    net = Network(cap, out / "http.jsonl")
    try:
        if action == "directory":
            archive = ExactArchive(net, BASE + "MVS/Rectified.zip")
            ident, rows, raw = archive.directory()
            (out / "central-directory.bin").write_bytes(raw)
            write_json(out / "archive.json", ident)
            write_json(out / "members.json", list(rows.values()))
        elif action == "author-metadata":
            archive = ExactArchive(net, BASE + "MVS/SampleSet.zip")
            ident, rows, raw = archive.directory()
            files = []
            allowed = ["SampleSet/ReadMe.txt", *["SampleSet/Matlab evaluation code/" + name for name in
                       ("GetUsedSets.m", "GetUsedLightSets.m", "ComputeStat_web.m", "BaseEvalMain_web.m")]]
            for name in allowed:
                raw = archive.member(rows[name])
                destination = name.rsplit("/", 1)[1]
                (out / destination).write_bytes(raw)
                files.append({**rows[name], "sha256": digest(raw), "path": destination})
            write_json(out / "identity-sources.json", {"archive": ident, "files": files})
        elif action == "papers":
            for name, url in PAPERS.items():
                data, headers = net.get(url, limit=config["per_paper_max_bytes"])
                check(data[:5] == b"%PDF-", "unexpected publication content")
                pdf = out / (name + ".pdf")
                pdf.write_bytes(data)
                result = subprocess.run(["/usr/bin/osascript", "-l", "JavaScript", "-e", PDF_TEXT, str(pdf)],
                                        check=True, capture_output=True, text=True)
                pages = json.loads(result.stdout)
                check(len(pages) >= 5 and all(isinstance(p, str) for p in pages), "text extraction failed")
                write_json(out / (name + "-text.json"), {"url": url, "publication_sha256": digest(data),
                                                        "extraction": "PDFKit page.string only, no image extraction or rendering", "pages": pages})
        else:
            raise ValueError("unknown metadata action")
    finally:
        write_json(out / "network.json", {"action": action, "http_entity_bytes_read": net.total, "transfer_cap_bytes": cap,
                                          "raw_dtu_images_read": 0, "surface_or_scan_geometry_members_read": 0,
                                          "output_bytes_before_summary": sum(p.stat().st_size for p in out.glob("*") if p.is_file()),
                                          "disk_free_bytes_end": shutil.disk_usage(out).free})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    metadata(json.loads(a.config.read_bytes()), a.out)
