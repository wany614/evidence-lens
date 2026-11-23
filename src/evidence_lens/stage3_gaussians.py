"""An original source-only joint photometric optimizer and its separate gsplat renderer."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image


SETS={'M2':('s00','s03'),'M4':('s00','s01','s02','s03')}
TARGETS=tuple(f'v{i:02d}' for i in range(1,7))
PARAMS=('means','log_scales','quats','opacity_logits','color_logits')


def require(ok,message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,data):
    path.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_gpu():
    import torch
    require(importlib.metadata.version('gsplat')=='1.5.3' and torch.cuda.is_available(),'gsplat 1.5.3 and CUDA are required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return torch.device('cuda:0')


def runtime():
    import torch
    d=torch.cuda.get_device_properties(0)
    return {'device':d.name,'gpu_total_bytes':d.total_memory,'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda,
            'gsplat':importlib.metadata.version('gsplat'),'numpy':np.__version__,'python':platform.python_version(),
            'torch_cuda_allocated_peak_bytes':torch.cuda.max_memory_allocated(),'torch_cuda_reserved_peak_bytes':torch.cuda.max_memory_reserved()}


def initial_values(s):
    gen=np.random.Generator(np.random.PCG64(s['seed']))
    n=s['count']
    means=gen.uniform(s['means_uniform_min'],s['means_uniform_max'],size=(n,3))
    scale=gen.uniform(*s['scale_uniform_m'],size=(n,3))
    q=gen.normal(size=(n,4))
    q=q/np.linalg.norm(q,axis=1,keepdims=True)
    opacity=gen.uniform(*s['opacity_uniform'],size=n)
    rgb=gen.uniform(*s['linear_rgb_uniform'],size=(n,3))
    logit=lambda x:np.log(x/(1-x))
    return {k:np.ascontiguousarray(x,dtype='<f4') for k,x in
            zip(PARAMS,(means,np.log(scale),q,logit(opacity),logit(rgb)),strict=True)}


def digest_values(arr):
    h=hashlib.sha256()
    for n in PARAMS:
        v=np.ascontiguousarray(arr[n],dtype='<f4')
        h.update(n.encode('ascii')+b'\x00')
        h.update(np.asarray(v.shape,dtype='<i4').tobytes())
        h.update(v.tobytes())
    return h.hexdigest()


def valid_cameras(doc, ids):
    require(doc['schema']=='el-camera-1' and doc['unit']=='m' and doc['pixel_center']=='integer'
            and doc['axis_convention']=='OpenCV x right, y down, z forward' and set(doc['views'])==set(ids),'Wrong camera role/coordinate convention')
    for cam in doc['views'].values():
        k=np.asarray(cam['K'])
        pose=np.asarray(cam['c2w'])
        require(cam['width']==cam['height']==256 and k.shape==(3,3) and pose.shape==(4,4) and
                np.isfinite(pose).all() and np.isfinite(k).all() and np.allclose(pose[3],[0,0,0,1]) and
                k[0,0]>0 and k[1,1]>0 and np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-5) and
                abs(np.linalg.det(pose[:3,:3])-1)<1e-5,'Unsupported camera')
    return doc['views']


def view_tensor(cam,device):
    import torch
    k=np.array(cam['K'],dtype=np.float32)
    k[:2,2]+=0.5
    w2c=np.linalg.inv(np.array(cam['c2w'],dtype=np.float64)).astype(np.float32)
    return (torch.tensor(k[None],device=device),torch.tensor(w2c[None],device=device),cam['width'],cam['height'],cam['clip_m'])


def raster(means,scales,quats,opacity,color,view,mode,eps):
    from gsplat import rasterization
    k,w2c,width,height,clip=view
    val,alpha,_=rasterization(means=means,quats=quats,scales=scales,opacities=opacity,colors=color,viewmats=w2c,Ks=k,
        width=width,height=height,near_plane=clip[0],far_plane=clip[1],eps2d=eps,render_mode=mode,rasterize_mode='classic',
        packed=False,backgrounds=None)
    return val[0],alpha[0,...,0]


def native_parameters(params):
    import torch
    return (params['means'],params['log_scales'].exp(),torch.nn.functional.normalize(params['quats'],dim=-1),
            params['opacity_logits'].sigmoid(),params['color_logits'].sigmoid())


def display(linear):
    import torch
    x=linear.clamp_min(0)
    return torch.where(x<=0.0031308,12.92*x,1.055*x.clamp_min(.0031308).pow(1/2.4)-.055).clamp(0,1)


def train(config, camera_path, images, out):
    import torch
    started=time.perf_counter()
    device=check_gpu()
    cfg=read(config)
    budget=cfg['optimizer']
    inits=budget['initialization']
    opt=budget['optimization']
    ids=SETS[cfg['condition_id']]
    require(tuple(cfg['generator_source_ids'])==ids and sha(camera_path)==cfg['source_camera_sha256'],'Optimizer sources differ')
    cams=valid_cameras(read(camera_path),ids)
    require(len(images)==len(ids) and set(images)==set(ids),'Source image roster must exactly match the optimizer')
    sources={}
    source_sha={}
    for s in ids:
        src=images[s]
        source_sha[s]=sha(src)
        require(source_sha[s]==cfg['source_rgb_sha256'][s],'Source RGB differs')
        with Image.open(src) as im:
            require(im.mode=='RGB' and im.size==(256,256),'Unexpected source RGB')
            sources[s]=torch.as_tensor(np.array(im,dtype=np.float32)/255,device=device)
    steps=cfg['optimization_steps']
    require(steps%len(ids)==0 and 0<steps<=budget['limits']['max_updates'] and inits['count']<=budget['limits']['max_gaussians'],'Invalid fixed budget')
    inputs_s=time.perf_counter()-started
    torch.cuda.reset_peak_memory_stats()
    apriori=initial_values(inits)
    np.savez(out/'initial.npz',**apriori)
    params={k:torch.nn.Parameter(torch.from_numpy(v.copy()).to(device)) for k,v in apriori.items()}
    optimizer=torch.optim.Adam([{'params':[params[k]],'lr':opt['learning_rates'][k]} for k in PARAMS],
                               betas=tuple(opt['betas']),eps=opt['eps'],foreach=False)
    views={k:view_tensor(cams[k],device) for k in ids}
    def source_loss(k):
        colors,alpha=raster(*native_parameters(params),views[k],'RGB',budget['render']['eps2d'])
        return (display(colors)-sources[k]).abs().mean(),alpha
    def current_losses():
        with torch.no_grad():
            return {s:float(source_loss(s)[0].item()) for s in ids}
    initial_losses=current_losses()
    stats={s:{'samples':0,'pixels':0,'gradient_nonzero_steps':{p:0 for p in PARAMS}} for s in ids}
    begin=time.perf_counter()
    with (out/'optimizer.jsonl').open('x',encoding='utf-8') as log:
        for i in range(steps):
            source=ids[i%len(ids)]
            optimizer.zero_grad(set_to_none=True)
            loss,alpha=source_loss(source)
            require(torch.isfinite(loss).item() and alpha.max().item()>0,'Nonfinite source loss or empty support')
            loss.backward()
            grads=torch.stack([torch.linalg.vector_norm(params[k].grad) for k in PARAMS]).detach().cpu().numpy()
            require(np.isfinite(grads).all(),'Nonfinite source gradients')
            optimizer.step()
            values={k:float(v) for k,v in zip(PARAMS,grads,strict=True)}
            s=stats[source]
            s['samples']+=1
            s['pixels']+=cams[source]['width']*cams[source]['height']
            for k in PARAMS: s['gradient_nonzero_steps'][k]+=int(values[k]>0)
            log.write(json.dumps({'update':i+1,'source':source,'image_sha256':source_sha[source],'pixels':256*256,
                                  'loss_display_srgb_mae':float(loss.item()),'gradient_l2':values},allow_nan=False)+'\n')
    torch.cuda.synchronize()
    optimization_s=time.perf_counter()-begin
    final={k:p.detach().cpu().numpy().astype('<f4',copy=True) for k,p in params.items()}
    require(all(np.isfinite(a).all() for a in final.values()),'Nonfinite final common scene')
    deltas={k:float(np.linalg.norm((final[k]-apriori[k]).astype(np.float64))) for k in PARAMS}
    require(all(v>0 for v in deltas.values()) and all(s['samples']==steps//len(ids) and all(v>0 for v in s['gradient_nonzero_steps'].values()) for s in stats.values()),
            'Geometry, appearance or a registered source failed to update the shared scene')
    final_losses=current_losses()
    np.savez(out/'scene.npz',**final)
    write(out/'reconstruction.json',{'schema':'el-stage3-reconstruction-1','run_id':os.environ['EL_RUN_ID'],'backend':'gsplat','backend_version':'1.5.3',
          'scene_frame':'Blender world z up, metres; quaternion wxyz; anisotropic scale exp(log_scales), linear RGB/opacity sigmoid(logits)',
          'generator_source_ids':ids,'source_rgb_sha256':source_sha,'source_camera_sha256':sha(camera_path),'optimizer_sha256':cfg['optimizer_sha256'],
          'initial_checkpoint_sha256':sha(out/'initial.npz'),'initial_parameter_sha256':digest_values(apriori),'checkpoint_sha256':sha(out/'scene.npz'),
          'final_parameter_sha256':digest_values(final),'parameter_delta_l2':deltas,'parameter_names':PARAMS,'gaussian_optimization_steps':steps,
          'initial_gaussian_count':int(len(apriori['means'])),'final_gaussian_count':int(len(final['means'])),'topology_events':[],
          'source_sampling':stats,'sampled_source_pixels':sum(x['pixels'] for x in stats.values()),'loss_initial_by_source':initial_losses,
          'loss_final_by_source':final_losses,'loss':'mean absolute display-sRGB on one actual whole source per step',
          'checkpoint_rule':'final fixed update','trace_sha256':sha(out/'optimizer.jsonl'),'precision':'float32',
          'read_seconds':inputs_s,'optimization_seconds':optimization_s,'stage_seconds':time.perf_counter()-started,'runtime':runtime()})


def analytic_probe(device, budget):
    import torch
    def t(v): return torch.tensor(v,dtype=torch.float32,device=device)
    cam={'K':[[24,0,16],[0,24,16],[0,0,1]],'c2w':np.eye(4).tolist(),'width':33,'height':33,'clip_m':[.05,30.]}
    pts=t([[0,0,1.5],[0,0,2.5],[.625,-.3125,2.5]])
    scale=t([[.025]*3]*3)
    q=t([[1,0,0,0]]*3)
    opacity=t([.4,.65,.72])
    col=t([[.20,.45,.62]]*3)
    val,alpha=raster(pts,scale,q,opacity,col,view_tensor(cam,device),'RGB+ED',budget['eps2d'])
    accum,_=raster(pts,scale,q,opacity,col,view_tensor(cam,device),'RGB+D',budget['eps2d'])
    x,y=22,13
    a0=.4+(1-.4)*.65
    d0=.4*1.5+(1-.4)*.65*2.5
    readouts=[{'xy':[16,16],'expected_alpha':a0,'alpha':float(alpha[16,16]),'expected_z':d0/a0,'actual_z':float(val[16,16,3]),'expected_D':d0,'actual_D':float(accum[16,16,3])},
              {'xy':[x,y],'expected_alpha':.72,'alpha':float(alpha[y,x]),'expected_z':2.5,'actual_z':float(val[y,x,3]),'expected_D':.72*2.5,'actual_D':float(accum[y,x,3])}]
    shifted=np.eye(4);shifted[0,3]=.3125
    moved={**cam,'c2w':shifted.tolist()}
    v,a=raster(pts[2:],scale[2:],q[2:],opacity[2:],col[2:],view_tensor(moved,device),'RGB+ED',budget['eps2d'])
    tol=budget['probe_depth_tolerance_m']
    at=budget['probe_alpha_tolerance']
    require(all(abs(x['alpha']-x['expected_alpha'])<=at and abs(x['actual_z']-x['expected_z'])<=tol and abs(x['actual_D']-x['expected_D'])<=tol for x in readouts)
            and abs(float(a[13,19])-.72)<=at and abs(float(v[13,19,3])-2.5)<=tol and float(alpha[0,0])==0,
            'Analytic opacity/half-pixel/normalized camera-Z/translation fixture failed')
    return {'passed':True,'samples':readouts,'shifted_xy':[19,13],'translated_alpha':float(a[13,19]),'translated_z':float(v[13,19,3]),'empty_alpha':float(alpha[0,0])}


def render(config,camera_path,checkpoint,reconstruction,out):
    import torch
    begun=time.perf_counter()
    device=check_gpu()
    cfg=read(config)
    r=read(reconstruction)
    ids=SETS[cfg['condition_id']]
    require(tuple(r['generator_source_ids'])==ids and r['run_id']==cfg['reconstruction_run_id'] and r['checkpoint_sha256']==sha(checkpoint)
            and r['source_rgb_sha256']==cfg['source_rgb_sha256'] and r['source_camera_sha256']==cfg['source_camera_sha256'] and
            sha(camera_path)==cfg['render_camera_sha256'],'Renderer must consume the selected reconstruction')
    views=valid_cameras(read(camera_path),ids+TARGETS)
    with np.load(checkpoint,allow_pickle=False) as arr:
        require(set(arr.files)==set(PARAMS) and digest_values(arr)==r['final_parameter_sha256'],'Checkpoint schema/content differs')
        p={k:torch.as_tensor(arr[k],device=device) for k in PARAMS}
    program=cfg['optimizer']['render']
    torch.cuda.reset_peak_memory_stats()
    probe=analytic_probe(device,program)
    write(out/'depth-probe.json',probe)
    index={}
    with torch.no_grad():
        for k in ids+TARGETS:
            start=time.perf_counter()
            val,alpha=raster(*native_parameters(p),view_tensor(views[k],device),'RGB+ED',program['eps2d'])
            rgb=display(val[...,:3]).cpu().numpy().astype(np.float32)
            a=alpha.cpu().numpy().astype(np.float32)
            z=val[...,3].cpu().numpy().astype(np.float32)
            require(np.isfinite(rgb).all() and np.isfinite(a).all() and np.all((a>=0)&(a<=1)),'Invalid rendered colour or opacity')
            z[(a<=0)|(~np.isfinite(z))|(z<=0)]=np.nan
            path=out/(k+'.npz'); png=out/(k+'.png')
            np.savez_compressed(path,rgb=rgb,z=z,alpha=a)
            Image.fromarray(np.floor(rgb*255+.5).astype(np.uint8)).save(png)
            index[k]={'sha256':sha(path),'bytes':path.stat().st_size,'png_sha256':sha(png),'pixels':a.size,
                      'finite_depth_pixels':int(np.isfinite(z).sum()),'zero_alpha_pixels':int((a==0).sum()),'seconds':time.perf_counter()-start}
    write(out/'prediction.json',{'schema':'el-stage3-prediction-1','run_id':os.environ['EL_RUN_ID'],'reconstruction_run_id':r['run_id'],
          'generator_source_ids':ids,'target_ids':TARGETS,'source_rgb_sha256':r['source_rgb_sha256'],'source_camera_sha256':r['source_camera_sha256'],
          'render_camera_sha256':sha(camera_path),'checkpoint_sha256':sha(checkpoint),'reconstruction_sha256':sha(reconstruction),
          'rgb_encoding':'display_srgb_0_1','alpha_semantics':'accumulated_opacity','depth_semantics':'expected_camera_z_m',
          'depth_verified':True,'depth_probe_sha256':sha(out/'depth-probe.json'),'views':index,'runtime':runtime(),'stage_seconds':time.perf_counter()-begun})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('optimize','render'))
    for k in ('config','cameras','out','checkpoint','reconstruction'):
        p.add_argument('--'+k,type=Path,required=k in ('config','cameras','out'))
    p.add_argument('--source',action='append',nargs=2,default=[],metavar=('ID','PNG'))
    a=p.parse_args()
    if a.action=='optimize':
        images={i:Path(x) for i,x in a.source}
        require(len(images)==len(a.source),'Duplicate source ID')
        train(a.config,a.cameras,images,a.out)
    else:
        render(a.config,a.cameras,a.checkpoint,a.reconstruction,a.out)
