"""Exercise rank and threshold imports from the exact stand-alone Windows worker."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def test_complete_isolated_worker_and_rank_checkpoint(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    source=Path(__file__).resolve().parents[1]
    repo=Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip())
    archive=tmp_path/"worker.zip"
    commit="60d73ccc89c36903ba0182de8922e3f6ba0a15a2"
    subprocess.run([sys.executable,str(source/"scripts/vision_worker_pack.py"),"--source-root",str(source),
                    "--core-repo",str(repo),"--core-commit",commit,"--output",str(archive)],check=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(tmp_path/"isolated")
    code="""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('src').resolve()))
from evidence_lens_vision.train import scientific_sources
from evidence_lens.formal_metrics import rank
from evidence_lens.l1_freeze import threshold_for
import numpy as np
import torch
pins=scientific_sources()
assert all('src/evidence_lens/'+n+'.py' in pins for n in ('formal_metrics','l1_freeze','comparison_board'))
x=np.array([.25,.25,.8],np.float32)
y=np.array([True,False,True])
for labels,mask in [(y,y|True),(y&False,y|True),(y|True,y|True),(y,y&False)]:
    r,_=rank(x,labels,mask)
    torch.save(r,'rank.pt')
    assert torch.load('rank.pt',weights_only=True,map_location='cpu')==json.loads(json.dumps(r,allow_nan=False))==r
print(json.dumps(dict(imports=len(pins),rank_cases=4)))
"""
    p=subprocess.run([sys.executable,"-I","-B","-c",code],cwd=tmp_path/"isolated",text=True,capture_output=True,check=True)
    assert json.loads(p.stdout)["rank_cases"]==4
