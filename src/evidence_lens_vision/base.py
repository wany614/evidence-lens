"""Verify one official generic DINOv2 base and its portable CPU implementation."""
import hashlib
import importlib
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

from .contract import HASH, digest, output_dir, put, read, require
from .model import Backbone, DIM, GRID, PATCH

SOURCE="https://github.com/facebookresearch/dinov2"
BASE_URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"


def registered():
    return read(Path(__file__).resolve().parents[2]/"configs/vision-base-v1.json")


def pin_base(source, revision, weight_sha, output, archive=None):
    fixed=registered()
    require(revision==fixed["revision"] and weight_sha==fixed["checkpoint_sha256"],
            "base differs from the registered ordinary DINOv2 ViT-S/14")
    require(bool(source)!=bool(archive),"choose the official Git checkout or its registered archive")
    require(re.fullmatch(r"[a-f0-9]{40}",revision) and HASH.fullmatch(weight_sha),"use full source and weight identities")
    if archive:
        require(Path(archive).stat().st_size==fixed["source_archive_bytes"] and digest(archive)==fixed["source_archive_sha256"],
                "upstream source archive differs from the registered original")
        files={}
        with tarfile.open(archive) as bundle:
            entries=bundle.getmembers()
            candidates=[e.name for e in entries if e.name.endswith("dinov2/hub/backbones.py")]
            require(len(candidates)==1,"official source archive has an ambiguous root")
            prefix=candidates[0][:-len("dinov2/hub/backbones.py")]
            for entry in entries:
                path=entry.name.removeprefix(prefix)
                if path.startswith("dinov2/") and path.endswith(".py") or path in {"LICENSE","MODEL_CARD.md"}:
                    require(entry.isfile() and path not in files and ".." not in path.split("/"),"unsafe official source member")
                    files[path]=hashlib.sha256(bundle.extractfile(entry).read()).hexdigest()
    else:
        git=["git","-C",str(source)]
        require(subprocess.check_output([*git,"rev-parse",revision+"^{commit}"],text=True).strip()==revision,
                "the requested source revision is not a local Git commit")
        raw=subprocess.check_output([*git,"ls-tree","-rz",revision,"--","dinov2","LICENSE","MODEL_CARD.md"])
        files={}
        for entry in raw.split(b"\0"):
            if not entry:
                continue
            head,name=entry.split(b"\t",1)
            path=name.decode()
            if path.endswith(".py") or path in {"LICENSE","MODEL_CARD.md"}:
                mode,kind,oid=head.split()
                require(mode in (b"100644",b"100755") and kind==b"blob","unsafe upstream member")
                content=subprocess.check_output([*git,"cat-file","blob",oid])
                files[path]=hashlib.sha256(content).hexdigest()
    require({"LICENSE","dinov2/hub/backbones.py","dinov2/models/vision_transformer.py"}<=files.keys(),"incomplete DINOv2 source")
    require(files["LICENSE"]==fixed["license_sha256"],"registered official license differs")
    out=output_dir(output)
    record={"schema":"el-dinov2-base-1","architecture":"dinov2_vits14","registers":0,"repository":SOURCE,
            "revision":revision,"checkpoint_url":BASE_URL,"checkpoint_sha256":weight_sha,"license":"Apache-2.0",
            "license_sha256":files["LICENSE"],"sources":files,"checkpoint_bytes":fixed["checkpoint_bytes"],
            "source_archive_sha256":fixed["source_archive_sha256"],"source_archive_bytes":fixed["source_archive_bytes"]}
    put(out/"base.json",record)
    return record


def verified_base(source, weights, pin):
    source=Path(source).resolve()
    base=read(pin)
    fixed=registered()
    require(all(base[k]==v for k,v in fixed.items() if k!="schema") and Path(weights).stat().st_size==fixed["checkpoint_bytes"],
            "base pin differs from the registered official artifact tuple")
    require(base["schema"]=="el-dinov2-base-1" and base["architecture"]=="dinov2_vits14" and
            base["registers"]==0 and base["repository"]==SOURCE and base["checkpoint_url"]==BASE_URL
            and re.fullmatch(r"[a-f0-9]{40}",base["revision"]) and base["license"]=="Apache-2.0"
            and base["license_sha256"]==base["sources"]["LICENSE"] and digest(weights)==base["checkpoint_sha256"],
            "official plain generic base identity differs")
    files=base["sources"]
    expected={p.relative_to(source).as_posix() for p in (source/"dinov2").rglob("*.py")}
    require(expected=={k for k in files if k.endswith(".py")} and {"LICENSE","dinov2/hub/backbones.py"}<=files.keys(),
            "official Python source inventory differs")
    for name,sha in files.items():
        p=source/name
        require(p.resolve().is_relative_to(source) and not p.is_symlink() and digest(p)==sha,
                "official source differs: "+name)
    os.environ["XFORMERS_DISABLED"]="1"
    sys.path.insert(0,str(source))
    try:
        upstream=importlib.import_module("dinov2.hub.backbones")
        require(all(Path(mod.__file__).resolve().is_relative_to(source) for n,mod in sys.modules.items()
                    if n=="dinov2" or n.startswith("dinov2.") if getattr(mod,"__file__",None)),
                "preloaded DINO module differs from the selected local Git source")
        reference=upstream.dinov2_vits14(pretrained=False).cpu().eval()
    finally:
        sys.path.pop(0)
    state=torch.load(weights,map_location="cpu",weights_only=True)
    require(isinstance(state,dict) and all(isinstance(n,str) and isinstance(t,torch.Tensor) and
            t.dtype==torch.float32 and torch.isfinite(t).all() for n,t in state.items()),
            "expected finite generic FP32 state dictionary")
    own=Backbone().cpu().eval()
    reference.load_state_dict(state,strict=True)
    own.load_state_dict(state,strict=True)
    require(all(torch.equal(t,own.state_dict()[k]) for k,t in state.items()),"copied base tensor differs")
    # This synthetic input tests architecture equivalence independently of study labels or media.
    x=torch.arange(3*GRID*GRID,dtype=torch.int32).remainder(257).float().div(128).sub(1).reshape(1,3,GRID,GRID)
    with torch.inference_mode():
        a=reference.forward_features(x)["x_norm_patchtokens"]
        b=own(x)
    delta=(a-b).abs()
    require(a.shape==b.shape==(1,(GRID//PATCH)**2,DIM) and torch.isfinite(delta).all()
            and torch.allclose(a,b,rtol=2e-5,atol=2e-4),"official/portable FP32 patch token comparison failed")
    audit={"schema":"el-vision-base-check-1","base":base,"base_pin_sha256":digest(pin),"parameters":sum(t.numel() for t in state.values()),
           "tensors":len(state),"strict_equal_loaded_tensors":True,"probe":"integer ramp modulo 257, CPU FP32, 252 square",
           "probe_input_sha256":hashlib.sha256(x.numpy().tobytes()).hexdigest(),"probe_shape":list(a.shape),
           "official_patch_sha256":hashlib.sha256(a.numpy().tobytes()).hexdigest(),
           "portable_patch_sha256":hashlib.sha256(b.numpy().tobytes()).hexdigest(),
           "max_abs_difference":float(delta.max()),"mean_abs_difference":float(delta.mean()),"atol":2e-4,"rtol":2e-5}
    return own,audit
