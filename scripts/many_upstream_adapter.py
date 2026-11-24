"""Decode and independently project the supported official gsplat simple_trainer SH3 checkpoint."""

import numpy as np
import torch


def need(ok, message):
    if not ok:
        raise ValueError(message)


PARAMS={"means":(3,), "scales":(3,), "quats":(4,), "opacities":(), "sh0":(1,3), "shN":(15,3)}


def load_checkpoint(path):
    blob=torch.load(path,map_location="cpu",weights_only=True)
    need(isinstance(blob,dict) and set(blob)=={"splats","step"} and type(blob["step"]) is int,"unsupported official native checkpoint")
    p=blob["splats"]
    need(isinstance(p,dict) and set(p)==set(PARAMS),"incomplete native means, scales, quaternion, opacity or SH")
    need(isinstance(p["means"],torch.Tensor) and p["means"].ndim==2,"invalid Gaussian position array")
    n=len(p["means"])
    need(0<n<=60000,"actual fitted Gaussian count exceeds the registered cap")
    for k,shape in PARAMS.items():
        t=p[k]
        need(isinstance(t,torch.Tensor) and t.dtype==torch.float32 and tuple(t.shape)==(n,*shape) and bool(torch.isfinite(t).all()),
             "unsupported native parameter shape/dtype/value: "+k)
    need(bool((torch.linalg.vector_norm(p["quats"],dim=1)>1e-12).all()) and bool((torch.exp(p["scales"])>0).all())
         and bool(torch.isfinite(torch.exp(p["scales"])).all()),"degenerate native Gaussian orientation or scale")
    return blob


def check_frame(row):
    need(set(row)=={"rgb","z","alpha"},"missing native RGB, alpha or expected camera-Z")
    rgb,alpha,z=(row[k] for k in ("rgb","alpha","z"))
    need(rgb.dtype==alpha.dtype==z.dtype==np.float32 and rgb.shape==(252,378,3) and alpha.shape==z.shape==(252,378),
         "native frame shape or dtype differs")
    need(np.isfinite(rgb).all() and np.isfinite(alpha).all() and (rgb>=0).all() and (rgb<=1).all()
         and (alpha>=0).all() and (alpha<=1).all(),"invalid display RGB or alpha")
    need(np.all(np.isfinite(z[alpha>1e-6])) and np.all(z[alpha>1e-6]>0) and np.all(np.isnan(z[alpha<=1e-6])),
         "missing positive expected optical-axis depth on registered support")


def fields(value, opacity):
    rgb=value[0,...,:3].clamp(0,1).contiguous().cpu().numpy()
    alpha=opacity[0,...,0].contiguous().cpu().numpy()
    z=value[0,...,3].contiguous().cpu().numpy().copy()
    z[alpha<=1e-6]=np.nan
    row={"rgb":rgb,"alpha":alpha,"z":z}
    check_frame(row)
    return row


def render(p, cam, clip):
    from gsplat import rasterization
    k=torch.tensor(cam["K"],dtype=torch.float32,device="cuda");k[:2,2]+=.5
    pose=torch.tensor(cam["c2w"],dtype=torch.float32,device="cuda")
    image,alpha,_=rasterization(means=p["means"],quats=p["quats"],scales=torch.exp(p["scales"]),
              opacities=torch.sigmoid(p["opacities"]),colors=torch.cat((p["sh0"],p["shN"]),dim=1),
              viewmats=torch.linalg.inv(pose[None]),Ks=k[None],width=cam["width"],height=cam["height"],sh_degree=3,
              render_mode="RGB+ED",packed=False,near_plane=clip[0],far_plane=clip[1],rasterize_mode="classic",camera_model="pinhole")
    return fields(image,alpha)


def depth_probe():
    from gsplat import rasterization
    f=lambda a:torch.as_tensor(a,dtype=torch.float32,device="cuda")
    points=f([[0,0,2],[0,0,4],[.5,.25,2],[1,.5,4]])
    args=dict(means=points,quats=f([[1,0,0,0]]*4),scales=f([[.025]*3]*4),opacities=f([.5]*4),colors=f([[1,1,1]]*4),
              viewmats=f(np.eye(4)[None]),Ks=f([[[64,0,32.5],[0,64,32.5],[0,0,1]]]),width=65,height=65,
              packed=False,camera_model="pinhole",rasterize_mode="classic",near_plane=.01,far_plane=100.)
    ed,a,_=rasterization(**args,render_mode="RGB+ED");d,_,_=rasterization(**args,render_mode="RGB+D")
    samples=[]
    for x,y in ((32,32),(48,40)):
        alpha=float(a[0,y,x,0]);z=float(ed[0,y,x,3]);summed=float(d[0,y,x,3])
        need(abs(alpha-.75)<1e-4 and abs(z-8/3)<1e-4 and abs(z-summed/alpha)<1e-4,"expected camera-Z analytic probe failed")
        samples.append({"x":x,"y":y,"alpha":alpha,"expected_z":z,"allocated_depth_sum":summed})
    return {"schema":"el-many-native-expected-z-probe-1","passed":True,"samples":samples}
