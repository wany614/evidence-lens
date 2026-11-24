"""Record the editions and page ranges used for the three humanities source cards."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request


SOURCES = {
    "bolter-grusin": ("https://www.dss-edit.com/plu/Bolter-Grusin_Remediation.pdf", (1, 2)),
    "london-charter": ("https://www.london-charter.org/media/files/london_charter_2_1_en.pdf",
                       (1, 7, 8, 9, 12, 13)),
    "seville-principles": ("https://icomos.es/wp-content/uploads/2020/06/Seville-Principles-IN-ES-FR.pdf",
                          (1, 2, 3, 6, 8)),
}


def extract(path, pages, output):
    import pypdf
    reader = pypdf.PdfReader(str(path))
    paragraphs = {str(i): reader.pages[i - 1].extract_text() for i in pages}
    if not all(paragraphs.values()):
        raise ValueError("A reading page lacks an accessible text layer")
    output.write_text(json.dumps(paragraphs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pdf_pages": len(reader.pages), "parser": "pypdf " + pypdf.__version__}))


def capture(out, pdf_python):
    from evidence_lens._bundle import save, sha
    editions = {}
    for name, (url, pages) in SOURCES.items():
        request = urllib.request.Request(url, headers={"User-Agent": "EvidenceLens-research/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            blob = response.read(10_000_001)
            final_url = response.geturl()
        if not blob.startswith(b"%PDF-") or len(blob) > 10_000_000:
            raise ValueError("Unexpected reading edition for " + name)
        document = out / (name + ".pdf")
        document.write_bytes(blob)
        selection = out / (name + "-reading.json")
        summary = json.loads(subprocess.check_output(
            [pdf_python, str(Path(__file__).resolve()), "--extract", document, "--pages", *map(str, pages),
             "--out", selection], text=True))
        editions[name] = dict(url=url, retrieved_url=final_url, bytes=len(blob),
                              sha256=hashlib.sha256(blob).hexdigest(), pdf_page_numbers=list(pages),
                              excerpt_sha256=sha(selection), **summary)
    save(out / "reading.json", dict(schema="el-humanities-primary-editions-1",
         run_id=os.environ["EL_RUN_ID"], retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         sources=editions, public_scope="Bibliography, edition fingerprints and original notes."))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("store", "out", "pdf-python", "extract"):
        p.add_argument("--" + field, type=Path)
    p.add_argument("--pages", type=int, nargs="+")
    args = p.parse_args()
    if args.store:
        from dual_records import selected, stage
        print(stage(args.store, "c-primary-editions", "archive_verify",
              {"registration": selected(args.store, "c-foundation-registration")},
              {"sources": SOURCES, "public_members": ["reading.json"]},
              ["{python}", "scripts/foundation_reading.py", "--out", "{out}",
               "--pdf-python", str(args.pdf_python.resolve())], "payload/reading.json", 16000000))
    elif args.extract:
        extract(args.extract, args.pages, args.out)
    else:
        capture(args.out, args.pdf_python)
