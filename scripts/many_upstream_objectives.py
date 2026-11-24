"""Check the registered valid SSIM value/gradient against the upstream test oracle, and KNN against independent distances."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request
import zipfile

from evidence_lens._bundle import load, save, sha
from evidence_lens_bundle.common import require


FUSED_COMMIT="328dc9836f513d00c4b5bc38fe30478b4435cbb5"


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"EvidenceLens-source-control"}),timeout=35) as response:
        return response.read()


def worker(out,cfg):
    import numpy as np
    import torch
    from many_upstream_native import valid_ssim, exact_knn
    torch.set_num_threads(2)
    base="https://raw.githubusercontent.com/rahul-goel/fused-ssim/"+FUSED_COMMIT+"/"
    for n,name in (("tests/test.py","upstream-test.py"),("fused_ssim/__init__.py","upstream-fused-entry.py"),("ssim.cu","upstream-ssim.cu")):
        (out/name).write_bytes(get(base+n))
    tests=(out/"upstream-test.py").read_text()
    require("from pytorch_msssim import SSIM" in tests and "assert torch.isclose(mine_ssim_val_valid, pm_ssim_val)" in tests,
            "the pinned fused-ssim reference test changed its independent valid-image oracle")
    meta=json.loads(get("https://pypi.org/pypi/pytorch-msssim/1.0.0/json"))
    wheel,=[x for x in meta["urls"] if x["filename"].endswith("py3-none-any.whl")]
    binary=get(wheel["url"])
    require(hashlib.sha256(binary).hexdigest()==wheel["digests"]["sha256"],"independent SSIM reference archive differs")
    (out/wheel["filename"]).write_bytes(binary)
    rng=torch.Generator().manual_seed(1927); results=[]
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(io.BytesIO(binary)) as z:
            for n in ("pytorch_msssim/__init__.py","pytorch_msssim/ssim.py"):
                p=Path(td)/n;p.parent.mkdir(exist_ok=True);p.write_bytes(z.read(n))
        sys.path.insert(0,td)
        from pytorch_msssim import SSIM
        ref=SSIM(data_range=1.,channel=3)
        for family,(h,w) in (("random",(36,53)),("ramp",(41,59)),("constant",(32,29)),("study-grid",(252,378))):
            x=torch.rand((1,3,h,w),generator=rng);y=torch.rand((1,3,h,w),generator=rng)
            if family=="ramp":x=torch.linspace(0,1,w).expand_as(x).clone();y=(.94*x+.06*y).clamp(0,1)
            if family=="constant":x=torch.full_like(x,.33);y=torch.full_like(y,.35)
            x.requires_grad_();xx=x.detach().clone().requires_grad_()
            v=valid_ssim(x,y,"valid");r=ref(xx,y)
            v.backward();r.backward()
            err=abs(float(v.detach()-r.detach()));gradient=float((x.grad-xx.grad).abs().max())
            require(err<=cfg["max_ssim_error"] and gradient<=cfg["max_grad_error"],"valid window or first-image gradient differs from upstream oracle")
            results.append({"case":family,"shape":[h,w],"value_abs_error":err,"maximum_gradient_abs_error":gradient})
    points=torch.rand((255,3),generator=rng)
    d=exact_knn(points,4).numpy();g=points.double().numpy()
    reference=np.sort(np.linalg.norm(g[:,None,:]-g[None,:,:],axis=2),axis=1)[:,:4].astype(np.float32)
    require(np.array_equal(d,reference),"official K=4 Euclidean distance convention differs")
    save(out/"objectives.json",{"schema":"el-many-official-objective-control-1","run_id":os.environ["EL_RUN_ID"],"fused_ssim_commit":FUSED_COMMIT,
         "official_test_sha256":sha(out/"upstream-test.py"),"official_cuda_sha256":sha(out/"upstream-ssim.cu"),
         "official_entry_sha256":sha(out/"upstream-fused-entry.py"),"independent_oracle":"pytorch-msssim 1.0.0 SSIM, same library used by the pinned fused-ssim valid-region test",
         "oracle_wheel_sha256":wheel["digests"]["sha256"],"registered_publication_hash":cfg["sources"]["vendor"],
         "ssim_checks":results,"max_value_tolerance":cfg["max_ssim_error"],"max_gradient_tolerance":cfg["max_grad_error"],
         "knn_fp32_after_fp64_euclidean_exact":True,"knn_distances":int(d.size),"photographs":0,"fits":0,"scope":"CPU FP32 value and derivative, no native fused CUDA timing claim"})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("record","worker"))
    for k in ("store","config","out","torch-python"):p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.action=="worker":worker(a.out,load(a.config))
    else:
        from dual_records import selected,stage
        s=a.store.resolve();results=stage(s,"b-upstream-objectives","archive_verify",{"vendor":selected(s,"b-upstream-vendor"),"study":selected(s,"b-study-registration")},
             {"fused_commit":FUSED_COMMIT,"max_ssim_error":.0003,"max_grad_error":.000006,"independent_package":"pytorch-msssim==1.0.0"},
             [str(a.torch_python.absolute()),"-B","{source}/scripts/many_upstream_objectives.py","worker","--out","{out}","--config","{config}"],
             "payload/objectives.json",180000)
        print(json.dumps(results))
