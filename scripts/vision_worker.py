"""Train or run the optional Evidence Lens risk model using explicitly supplied local files."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

BOOT=time.perf_counter()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    def paths(s,names):
        for k in names.split():
            s.add_argument("--"+k,type=Path,required=True)
    def words(s,names):
        for k in names.split():
            s.add_argument("--"+k,required=True)
    base=sub.add_parser("pin-base")
    paths(base,"output")
    inputs=base.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--official-source",type=Path)
    inputs.add_argument("--source-archive",type=Path)
    words(base,"revision expected-checkpoint-sha256")
    freeze=sub.add_parser("configure")
    paths(freeze,"train preprocessing base-pin catalog output")
    words(freeze,"train-sha256 preprocessing-sha256 base-pin-sha256 catalog-sha256")
    freeze.add_argument("--phase",choices=("pilot","train"),required=True)
    freeze.add_argument("--branch",choices=("L2-frozen","L2-vision-ft"),required=True)
    freeze.add_argument("--seed",choices=(1729,2718,3141),type=int,default=1729)
    freeze.add_argument("--epochs",type=int,default=20)
    freeze.add_argument("--max-updates",type=int)
    freeze.add_argument("--head-lr",type=float,required=True)
    freeze.add_argument("--backbone-lr",type=float,required=True)
    for key in ("development","lr-selection"):
        freeze.add_argument("--"+key,type=Path)
        freeze.add_argument("--"+key+"-sha256")
    verify=sub.add_parser("verify-base")
    paths(verify,"official-source base base-pin output")
    for action in ("pilot","train"):
        t=sub.add_parser(action)
        paths(t,"train preprocessing official-source base base-pin config output")
        t.add_argument("--development",type=Path)
        t.add_argument("--lr-selection",type=Path)
        t.add_argument("--resume",type=Path)
        t.add_argument("--resume-sha256")
        t.add_argument("--run-id")
        t.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    select=sub.add_parser("select")
    paths(select,"roster store output")
    select.add_argument("--stage",required=True,choices=("learning-rate","primary"))
    infer=sub.add_parser("infer")
    paths(infer,"package rgb deploy output")
    infer.add_argument("--model-sha256")
    infer.add_argument("--device",default="cpu",choices=("cpu","cuda"))
    dataset=sub.add_parser("infer-dataset")
    paths(dataset,"package dataset output")
    words(dataset,"dataset-sha256 model-sha256")
    dataset.add_argument("--split",required=True,choices=("development","test"))
    for k in ("store","shelf","catalog","method-freeze"):
        dataset.add_argument("--"+k,type=Path)
    dataset.add_argument("--catalog-sha256")
    dataset.add_argument("--method-sha256")
    dataset.add_argument("--device",default="cpu",choices=("cpu","cuda"))
    native=sub.add_parser("infer-native")
    paths(native,"package input cameras predictions output")
    words(native,"model-sha256")
    native.add_argument("--view",choices=["all",*[f"v{i:02d}" for i in range(1,7)]],default="all")
    native.add_argument("--warm-repeats",type=int,default=0)
    return p


def main():
    args=parser().parse_args()
    from evidence_lens_vision.contract import digest,output_dir,put
    if args.action=="pin-base":
        from evidence_lens_vision.base import pin_base
        pin_base(args.official_source,args.revision,args.expected_checkpoint_sha256,args.output,args.source_archive)
        name="base.json"
    elif args.action=="configure":
        from evidence_lens_vision.configure import configure
        configure(args)
        name="config.json"
    elif args.action=="verify-base":
        from evidence_lens_vision.base import verified_base
        out=output_dir(args.output)
        _,receipt=verified_base(args.official_source,args.base,args.base_pin)
        put(out/"base-verification.json",receipt)
        name="base-verification.json"
    elif args.action in ("pilot","train"):
        from evidence_lens_vision.train import run_fit
        run_fit(args,args.action)
        name="training.json"
    elif args.action=="select":
        from evidence_lens_vision.selection import selection
        selection(args)
        name="selection.json"
    elif args.action=="infer":
        from evidence_lens_vision.deploy import infer
        infer(args.package,args.rgb,args.deploy,args.output,args.device,args.model_sha256)
        name="inference.json"
    elif args.action=="infer-dataset":
        from evidence_lens_vision.batch import dataset_scores
        dataset_scores(args)
        name="scores.json"
    else:
        from evidence_lens_vision.native import native
        native(args,BOOT)
        name="native-cost.json"
    print(json.dumps({"file":name,"sha256":digest(args.output/name)},sort_keys=True))


if __name__=="__main__":
    main()
