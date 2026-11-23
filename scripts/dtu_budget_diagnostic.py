"""Record the developmental reconstruction-budget decision from the two original source traces."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["m2", "m4", "config", "out"]:
        p.add_argument("--"+key, type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_bytes())
    rows = {}
    for cond, path in (("M2",a.m2),("M4",a.m4)):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != cfg["traces"][cond]:
            raise ValueError("unregistered trace")
        trace = [json.loads(line) for line in data.splitlines()]
        if [r["update"] for r in trace] != list(range(1,257)):
            raise ValueError("incomplete main developmental trace")
        means = [sum(r["loss_display_srgb_mae"] for r in trace[i:i+32])/32 for i in (192,224)]
        rows[cond] = {"source_minibatch_mae_193_through_224":means[0],
                      "source_minibatch_mae_225_through_256":means[1],"main_updates":256}
    decision = {"schema":"el-dtu-development-diagnostic-1", "run_id":os.environ["EL_RUN_ID"],
                "object_id":"set1", "conditions":["M2","M4"], "updates":1024, "source_trace":rows,
                "reason":"Both last 32-step source-loss means are lower than the preceding 32 at the registered main endpoint. Quantify how additional scene optimization alters source and six target errors on the development object.",
                "comparison":"Independent execution from the same original seed and fixed topology. Only the update count changes. Compare source and six target RGB endpoints; no new risk fits, scores, thresholds or external runs.",
                "external_updates":256,"maximum_diagnostic_jobs":2}
    (a.out/"diagnostic-decision.json").write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n")
if __name__ == "__main__":
    main()
