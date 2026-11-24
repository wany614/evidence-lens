"""Read the registered source-only COLMAP model for the pinned official gsplat training loop."""

import collections
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def check(ok, message):
    if not ok:
        raise ValueError(message)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rotation(q):
    w,x,y,z=q/np.linalg.norm(q)
    return np.asarray([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                       [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                       [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def exact_knn(points, k):
    """Exact Euclidean KNN including the query itself, as in native utils.knn."""
    check(k==4 and len(points)>=k, "insufficient independent SfM geometry")
    values = points.to(torch.float64)
    result=[]
    for a in range(0,len(points),128):
        chunks=torch.cdist(values[a:a+128],values,p=2,compute_mode="donot_use_mm_for_euclid_dist")
        result.append(torch.topk(chunks,k,largest=False,sorted=True).values)
    return torch.cat(result).to(points)


def valid_ssim(x, y, padding):
    """Differentiable 11x11, sigma 1.5, valid, data-range-one population SSIM used for the registered fit."""
    check(padding=="valid" and x.shape==y.shape, "only the registered valid RGB structural objective is supported")
    axis=torch.arange(-5,6,dtype=x.dtype,device=x.device)
    kernel=torch.exp(-axis.square()/(2*1.5**2));kernel=kernel/kernel.sum()
    fields=torch.cat((x,y,x*x,y*y,x*y),dim=1)
    n=fields.shape[1]
    t=F.conv2d(fields,kernel.view(1,1,1,-1).expand(n,1,1,11),groups=n)
    t=F.conv2d(t,kernel.view(1,1,-1,1).expand(n,1,11,1),groups=n)
    a,b,xx,yy,xy=t.chunk(5,dim=1)
    return (((2*a*b+.01**2)*(2*(xy-a*b)+.03**2))/
            ((a*a+b*b+.01**2)*(xx-a*a+yy-b*b+.03**2))).mean()


class Parser:
    def __init__(self,data_dir,factor,normalize,test_every):
        check(factor==1 and normalize is False and test_every==0,"built-in split and normalization are disabled")
        self.root=Path(data_dir)
        self.roles=json.loads((self.root/"roles.json").read_bytes())
        self.ids=self.roles["R"]
        check(len(self.ids)==len(set(self.ids)) and len(self.ids)>=4,"duplicate or missing registered training sources")
        folder=self.root/"sparse/0"
        intrinsic={}
        for row in (folder/"cameras.txt").read_text().splitlines():
            i,model,w,h,*v=row.split();fx,fy,cx,cy=map(float,v)
            check(model=="PINHOLE" and (int(w),int(h))==(378,252),"unsupported source camera")
            intrinsic[int(i)]=np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]],np.float32)
        rows=(folder/"images.txt").read_text().splitlines()
        check(len(rows)==2*len(self.ids) and len(intrinsic)==len(self.ids),"source-only COLMAP roster differs")
        self.images=[]; self.cameras=[]; self.K=[]
        for i,key in enumerate(self.ids):
            image_id,*fields=rows[2*i].split()
            q,t=np.asarray(fields[:4],float),np.asarray(fields[4:7],float)
            camera_id,filename=int(fields[7]),fields[8]
            check(int(image_id)==camera_id==i+1 and filename==f"{i+1:04d}.png","COLMAP source view order differs")
            M=np.eye(4);M[:3,:3]=rotation(q);M[:3,3]=t
            c2w=np.linalg.inv(M);K=intrinsic[camera_id]
            cam=self.roles["cameras"][key]
            edge=np.array(cam["K"],dtype=float);edge[:2,2]+=.5
            check(np.allclose(K,edge,rtol=0,atol=1e-4) and np.allclose(c2w,cam["c2w"],atol=1e-6,rtol=1e-6),"native calibration disagrees with scene")
            path=self.root/"images"/filename
            check(file_hash(path)==self.roles["photos"][key],"source image digest differs")
            with Image.open(path) as im:
                check(im.size==(378,252) and im.mode=="RGB","wrong source photography dimensions")
                self.images.append(torch.from_numpy(np.asarray(im).copy()))
            self.cameras.append(torch.from_numpy(c2w.astype(np.float32)))
            self.K.append(torch.from_numpy(K))
        points=[];rgb=[]
        for line in (folder/"points3D.txt").read_text().splitlines():
            v=line.split();check(len(v)==12 and len({v[8],v[10]})==2,"each initialization track must name two registered sources")
            check(all(1<=int(n)<=len(self.ids) for n in (v[8],v[10])),"initialization track exceeds the registered source role")
            points.append([float(w) for w in v[1:4]]);rgb.append([int(w) for w in v[4:7]])
        self.points=np.asarray(points,np.float32);self.points_rgb=np.asarray(rgb,np.uint8)
        check(len(points)==self.roles["initial_points"] and 16<=len(points)<=15000,"incorrect initialization size")
        centers=np.stack([c[:3,3].numpy().astype(float) for c in self.cameras])
        self.scene_scale=float(np.linalg.norm(centers-centers.mean(0),axis=1).max())


class Dataset(torch.utils.data.Dataset):
    def __init__(self,parser,split,patch_size,load_depths):
        check(split=="train" and patch_size is None and load_depths is False,"unregistered dataset role")
        self.parser=parser
        self.visits=collections.Counter()
        self.sequence=hashlib.sha256()
        self.trace=[]
        self.epoch=time.monotonic()
        self.nearest_bound=parser.roles["time_limit_seconds"]

    def __len__(self):
        return len(self.parser.ids)

    def __getitem__(self,i):
        return {"image_id":i, "image":self.parser.images[i], "camtoworld":self.parser.cameras[i],"K":self.parser.K[i]}

    def record_step(self,step,indices,loss,l1,ssim,count):
        key=self.parser.ids[int(indices.item())]
        self.visits[key]+=1;self.sequence.update((str(step)+" "+key+"\n").encode())
        t=time.monotonic()-self.epoch
        check(t<self.nearest_bound and count<=60000 and bool(torch.isfinite(loss)), "training exceeded its registered budget or became nonfinite")
        if step%100==0 or step==5999:
            self.trace.append({"step":step,"source":key,"gaussians":int(count),"loss":float(loss.detach().item()),
                               "display_l1":float(l1.detach().item()),"one_minus_ssim":float(ssim.detach().item()),"elapsed_s":t})
