"""Render the frozen C scene and measure ED/opacity with each actual cropped DTU camera."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from evidence_lens import stage3_gaussians as native


def probe(camera, budget, device):
    import torch
    k=np.asarray(camera["K"],np.float64)
    pose=np.asarray(camera["c2w"],np.float64)
    native.require(abs(k[0,2]-127.5)>0.5 and abs(k[1,2]-127.5)>0.5,
                   "This DTU probe requires its registered off-centre right-camera principal point")
    xy=[(41,61),(41,61),(199,154)]
    depths=[.41,.91,.73]
    opacity=[.40,.65,.72]
    cam_xyz=np.array([np.linalg.solve(k,[x,y,1])*z for (x,y),z in zip(xy,depths,strict=True)])
    world=(pose[:3,:3]@cam_xyz.T).T+pose[:3,3]
    def t(x):
        return torch.tensor(x,dtype=torch.float32,device=device)
    values=(t(world),t([[.0018]*3]*3),t([[1.,0,0,0]]*3),t(opacity),t([[.2,.45,.62]]*3))
    view=native.view_tensor(camera,device)
    actual_edge_k=view[0][0].detach().cpu().numpy()
    expected_edge=np.array([[1,0,.5],[0,1,.5],[0,0,1]])@k
    native.require(np.max(np.abs(actual_edge_k-expected_edge))<0.0001,"gsplat actual intrinsics differ from cropped K+half-pixel")
    with torch.no_grad():
        ed,alpha=native.raster(*values,view,"RGB+ED",budget["eps2d"])
        acc,alpha_acc=native.raster(*values,view,"RGB+D",budget["eps2d"])
    b=.4+.6*.65
    d=.4*.41+.6*.65*.91
    cases=[]
    for (x,y),a,z,depth_sum,first,ray in [
        (xy[0],b,d/b,d,.41,d/b*np.linalg.norm(cam_xyz[0])/.41),
        (xy[2],.72,.73,.72*.73,.73,np.linalg.norm(cam_xyz[2]))]:
        expected_rgb=np.array([.2,.45,.62])*a
        actual_rgb=ed[y,x,:3].cpu().numpy()
        sides=[float(alpha[y+dy,x+dx]) for dx,dy in ((-1,0),(1,0),(0,-1),(0,1))]
        actual_a=float(alpha[y,x])
        actual_z=float(ed[y,x,3])
        actual_d=float(acc[y,x,3])
        native.require(abs(actual_a-a)<=budget["probe_alpha_tolerance"] and
                       abs(float(alpha_acc[y,x])-a)<=budget["probe_alpha_tolerance"] and
                       abs(actual_z-z)<=budget["probe_depth_tolerance_m"] and abs(actual_d-depth_sum)<=budget["probe_depth_tolerance_m"] and
                       np.max(abs(actual_rgb-expected_rgb))<.0005 and min(actual_a-v for v in sides)>.004,
                       "Actual calibrated gsplat pixel, RGB, ED, D or opacity disagrees with independent prediction")
        cases.append({"xy":[x,y],"expected_alpha":a,"actual_alpha":actual_a,"expected_ED_m":z,"actual_ED_m":actual_z,
                      "expected_D_m":depth_sum,"actual_D_m":actual_d,"first_surface_m":first,"expected_radial_range_m":float(ray),
                      "neighbour_max_alpha":max(sides),"expected_linear_rgb":expected_rgb.tolist(),"actual_linear_rgb":actual_rgb.tolist()})
    native.require(float(alpha[0,0])==0 and abs(cases[0]["actual_ED_m"]-cases[0]["first_surface_m"])>.1
                   and abs(cases[0]["actual_ED_m"]-cases[0]["expected_radial_range_m"])>.003,
                   "No empty/normalized off-axis depth distinction measured")
    return {"passed":True,"integer_K":k.tolist(),"measured_gsplat_edge_K":actual_edge_k.tolist(),
            "c2w":pose.tolist(),"measured_empty_alpha":float(alpha[0,0]),"cases":cases}


def main(a):
    native.render(a.config,a.cameras,a.checkpoint,a.reconstruction,a.out)
    cfg=native.read(a.config)
    cam=native.read(a.cameras)
    device=native.check_gpu()
    checks={k:probe(v,cfg["optimizer"]["render"],device) for k,v in cam["views"].items()}
    native.write(a.out/"dtu-camera-probe.json",{"schema":"el-dtu-native-gpu-probe-1","passed":True,
                 "render_camera_sha256":native.sha(a.cameras),"original_C_probe_sha256":native.sha(a.out/"depth-probe.json"),
                 "prediction_sha256":native.sha(a.out/"prediction.json"),"cameras":checks})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("config","cameras","checkpoint","reconstruction","out"):p.add_argument("--"+n,required=True,type=Path)
    main(p.parse_args())
