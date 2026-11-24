"""Project a clean public replay into per-view signed differences, flips and exact-quota membership counts."""
import argparse
import csv
from pathlib import Path
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require


def project(root,out):
    top=load(root/"acceptance.json")
    rows=[]
    for group,receipt in top["groups"].items():
        p=root/group/"acceptance.json"
        require(sha(p)==receipt["acceptance_sha256"],"different platform grade")
        data=load(p)
        for method,r in data["output"].items():
            for name,v in r["views"].items():
                delta=v["difference"]; matched=v["matched_coverage"]
                for q in matched["anchors"]:
                    rows.append({"platform":top["platform"]["os"],"arch":top["platform"]["arch"],"group":group,
                                 "method":method,"view":name,"bit_exact":delta["bit_exact"],"finite_common":delta["common_finite"],
                                 **{"changed_"+k:n for k,n in delta["changed_pixels"].items()},
                                 "min_signed_new_minus_old":delta["min_signed_new_minus_old"],
                                 "max_signed_new_minus_old":delta["max_signed_new_minus_old"],
                                 "mean_signed_new_minus_old":delta["mean_signed_new_minus_old"],
                                 "max_abs":delta["max_abs_difference_common_finite"],
                                 "threshold_crossings":delta["threshold_crossings_common_finite"],
                                 "common_mask_changed":matched["changed_common_mask"],**q})
    require(rows,"no replay outputs")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"numerical-differences.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
    save(out/"platform.json",{"clean_replay":top,"table_sha256":sha(out/"numerical-differences.csv"),
         "basis":"Each row is one matched quota on one original target and method. Threshold changes repeat across quota rows."})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();project(a.root,a.out)
