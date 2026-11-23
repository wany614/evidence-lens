"""Register the fixed engineering-only steps before preparing any fixture."""
import argparse
from pathlib import Path
from evidence_lens._bundle import save, sha
p = argparse.ArgumentParser()
p.add_argument("--candidate", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
save(a.out / "registration.json", {"scope": "synthetic engineering only", "recipe_candidate_sha256": sha(a.candidate),
     "point_cap": 20000, "resource_probe_resolution": [256, 256], "resume_fixture_steps": [0, 4, 8, 12],
     "recipe_horizon": 5000, "absolute_tolerance": 1e-4, "relative_tolerance": 1e-4, "absolute_loss_tolerance": 3e-5,
     "checks": ["cap/prune/clone/split/reset and Adam moments at every mutation", "fixed/active parameter deltas",
                "metric camera projection, expected depth and black display composite", "complete checkpoint resume parity",
                "all-checkpoint render versus one selected step", "1-second cutoff produces a sealed failure"]})
