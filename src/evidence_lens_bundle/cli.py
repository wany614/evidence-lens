"""Evidence Lens prepared-bundle CPU commands."""

import argparse
import json
import sys
from pathlib import Path
from zipfile import BadZipFile


def main(argv=None):
    from .common import Unsupported

    p = argparse.ArgumentParser(prog="evidence-lens", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("adapt-legacy", help="Convert an authenticated legacy Stage C Case")
    a.add_argument("--case", type=Path, required=True)
    a.add_argument("--case-sha256", required=True)
    a.add_argument("--proof", type=Path, help="Optional sealed legacy metadata and camera proof directory")
    a.add_argument("--proof-sha256", help="Independently obtained SHA-256 of proof.json")
    a.add_argument("--input-domain", default="Legacy Stage C Case; acquisition domain unspecified")
    a.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("validate", help="Verify manifest, authorized bytes, metric cameras, and all arrays")
    s = sub.add_parser("score", help="Apply a released CPU risk model")
    for cmd in (v, s):
        cmd.add_argument("--bundle", type=Path, required=True)
        cmd.add_argument("--bundle-sha256", help="Require this independently recorded bundle identity")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--model", choices=("MS-L1", "MS-ft", "MS-frozen", "B-L1-transfer"), default="MS-L1")
    s.add_argument(
        "--models", type=Path, help="Extracted published v0.3.0-rc.1 CPU release (optional for MS-L1)"
    )
    s.add_argument("--device", choices=["cpu"], default="cpu")
    s.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads for optional visual models")
    r = sub.add_parser("report", help="Render a self-contained HTML report from numerical score outputs")
    r.add_argument("--scores", type=Path, required=True)
    r.add_argument("--scores-sha256")
    r.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.command == "adapt-legacy":
            from .legacy import adapt

            result = adapt(
                args.case,
                args.case_sha256,
                args.out,
                input_domain=args.input_domain,
                proof=args.proof,
                proof_sha256=args.proof_sha256,
            )
        elif args.command == "validate":
            from .bundle import Bundle

            result = Bundle(args.bundle, args.bundle_sha256).validate()
        elif args.command == "score":
            from .scoring import score

            receipt = score(
                args.bundle,
                args.out,
                args.model,
                cpu_dir=args.models,
                expected_sha256=args.bundle_sha256,
                threads=args.threads,
            )
            result = {
                "status": "completed",
                "method": args.model,
                "counts": receipt["counts"],
                "bundle_sha256": receipt["bundle_sha256"],
            }
        else:
            from .report import render

            result = render(args.scores, args.out, args.scores_sha256)
    except BadZipFile as e:
        print(json.dumps({"status": "invalid", "message": "Invalid NPZ archive: " + str(e)}), file=sys.stderr)
        return 2
    except (ValueError, TypeError, KeyError, IndexError, OSError, EOFError) as e:
        status = "unsupported" if isinstance(e, Unsupported) else "invalid"
        print(json.dumps({"status": status, "message": str(e)}, ensure_ascii=False), file=sys.stderr)
        return 3 if status == "unsupported" else 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
