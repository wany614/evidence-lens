"""Calibrated, source-only sparse initialization for the recovery study."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

SETS={"M2":("s00","s03"),"M4":("s00","s01","s02","s03")}
NAMES=("means","log_scales","quats","opacity_logits","color_logits")
PROTOCOL_SHA="024a4a08552e748afafc3ead27ba745d6f96e84757572e5fbf9aec387653d086"


def require(ok,message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def write(path,item):
    with Path(path).open("x",encoding="utf-8") as out:
        json.dump(item,out,indent=2,sort_keys=True,allow_nan=False)
        out.write("\n")


def summary(values):
    a=np.asarray(values,np.float64)
    if not a.size:return {"n":0}
    require(np.isfinite(a).all(),"Nonfinite summary")
    return {"n":int(a.size),**{k:float(v) for k,v in zip(("min","p10","median","p90","max"),np.quantile(a,[0,.1,.5,.9,1]),strict=True)}}


def cameras(doc,ids):
    require(doc["schema"]=="el-camera-1" and doc["unit"]=="m" and doc["pixel_center"]=="integer" and
            doc["axis_convention"]=="OpenCV x right, y down, z forward" and set(doc["views"])==set(ids),"Unauthorized source camera set")
    out={}
    for s in ids:
        c=doc["views"][s];k=np.asarray(c["K"],np.float64);pose=np.asarray(c["c2w"],np.float64)
        clip=np.asarray(c["clip_m"],np.float64)
        require((c["width"],c["height"])==(256,256) and k.shape==(3,3) and np.isfinite(k).all() and np.allclose(k[2],[0,0,1]) and
                k[0,0]>0 and k[1,1]>0 and pose.shape==(4,4) and np.isfinite(pose).all() and np.allclose(pose[3],[0,0,0,1]) and
                np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-5) and np.linalg.det(pose[:3,:3])>0 and
                clip.shape==(2,) and 0<clip[0]<clip[1],"Incompatible calibrated source")
        out[s]={"K":k,"invK":np.linalg.inv(k),"pose":pose,"w2c":np.linalg.inv(pose)[:3],"C":pose[:3,3],"clip":clip}
    return out


def measure_features(image,algorithm):
    keypoints,desc=algorithm.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),None)
    xy=np.array([k.pt for k in keypoints],np.float64).reshape(-1,2)
    bucket={}
    for i,k in enumerate(keypoints):
        slot=tuple(np.rint(xy[i]*1000).astype(np.int64))
        if slot not in bucket or k.response>keypoints[bucket[slot]].response:bucket[slot]=i
    aliases=[bucket[tuple(v)] for v in np.rint(xy*1000).astype(np.int64)]
    return {"xy":xy,"desc":np.empty((0,128),np.float32) if desc is None else desc,"alias":aliases,"unique":len(bucket)}


def knn(src,dst,matcher):
    if len(src)==0 or len(dst)<2:return {},0
    pairs=matcher.knnMatch(src,dst,k=2)
    passed={}
    for match in pairs:
        if len(match)!=2:continue
        a,b=match
        if b.distance>0 and a.distance/b.distance<.8:
            passed[a.queryIdx]=(a.trainIdx,a.distance/b.distance,a.distance)
    return passed,len(pairs)


def fundamental(a,b):
    transform=b["w2c"]@a["pose"]
    r,t=transform[:,:3],transform[:,3]
    x,y,z=t
    cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return b["invK"].T@cross@r@a["invK"]


def pair_edges(i,j,fa,fb,a,b):
    matcher=cv2.BFMatcher(cv2.NORM_L2,crossCheck=False)
    forward,nf=knn(fa["desc"],fb["desc"],matcher);back,nb=knn(fb["desc"],fa["desc"],matcher)
    F=fundamental(a,b);edges=[];mutual=0;ep=[];ratios=[]
    for ida,(idb,ratio,distance) in forward.items():
        if idb not in back or back[idb][0]!=ida:continue
        mutual+=1
        ia,ib=fa["alias"][ida],fb["alias"][idb]
        left=np.append(fa["xy"][ia],1);right=np.append(fb["xy"][ib],1)
        l1=F@left;l0=F.T@right
        den=min(np.linalg.norm(l1[:2]),np.linalg.norm(l0[:2]))
        if den<=1e-12:continue
        error=float(abs(right@l1)/den)
        if error<=1.5:
            both=max(ratio,back[idb][1])
            edges.append((both,error,distance,(i,ia),(j,ib)))
            ep.append(error);ratios.append(both)
    return edges,{"knn_forward":nf,"knn_reverse":nb,"ratio_forward":len(forward),"ratio_reverse":len(back),"reciprocal":mutual,
                  "epipolar_pass":len(edges),"accepted_point_line_error_px":summary(ep),"accepted_worst_ratios":summary(ratios)}


def build_tracks(edges):
    roots={};members={};conflicts=cycles=0
    def root(x):
        if x not in roots:roots[x]=x;members[x]={x[0]:x}
        while roots[x]!=x:roots[x]=roots[roots[x]];x=roots[x]
        return x
    for *_,u,v in sorted(edges):
        a,b=root(u),root(v)
        if a==b:cycles+=1;continue
        if any(k in members[b] and n!=members[b][k] for k,n in members[a].items()):
            conflicts+=1;continue
        if b<a:a,b=b,a
        roots[b]=a;members[a].update(members.pop(b))
    tracks=sorted(tuple(x for _,x in sorted(group.items())) for group in members.values() if len(group)>=2)
    return tracks,{"ambiguous_component_edges_rejected":conflicts,"redundant_compatible_edges":cycles,"candidate_tracks":len(tracks)}


def point_from_track(observations,cams):
    rows=[]
    for i,uv in observations:
        c=cams[i]
        u,v,_=c["invK"]@np.r_[uv,1]
        p=c["w2c"]
        rows.extend([u*p[2]-p[0],v*p[2]-p[1]])
    _,_,vt=np.linalg.svd(np.stack(rows),full_matrices=False)
    h=vt[-1]
    if abs(h[3])<1e-12 or not np.isfinite(h).all():return None,"invalid_DLT"
    point=h[:3]/h[3]
    directions=[];errors=[];zs=[]
    for i,uv in observations:
        c=cams[i]
        xyz=c["w2c"]@np.r_[point,1]
        if not c["clip"][0]<xyz[2]<c["clip"][1]:return None,"outside_positive_source_clip"
        ray=point-c["C"];directions.append(ray/np.linalg.norm(ray));zs.append(float(xyz[2]))
        q=c["K"]@xyz;errors.append(float(np.linalg.norm(q[:2]/q[2]-uv)))
    cos=np.clip(np.stack(directions)@np.stack(directions).T,-1,1)
    parallax=float(np.degrees(np.arccos(cos.min())))
    if parallax<2:return None,"insufficient_parallax"
    if max(errors)>1.5:return None,"reprojection_error"
    return {"point":point,"observations":observations,"depths":zs,"errors":errors,"max_error":max(errors),"parallax_deg":parallax},None


def unique_tracks(tracks,radius):
    order=sorted(range(len(tracks)),key=lambda i:(-len(tracks[i]["observations"]),tracks[i]["max_error"],tracks[i]["ids"]))
    if not order:return []
    xyz=np.stack([x["point"] for x in tracks]);tree=cKDTree(xyz);suppressed=set();good=[]
    for i in order:
        if i in suppressed:continue
        good.append(tracks[i]);suppressed.update(tree.query_ball_point(xyz[i],r=radius))
    return good


def eligibility(points,ids):
    counts=[0]*len(ids);grids=[np.zeros((4,4),int) for _ in ids]
    for t in points:
        for i,uv in t["observations"]:
            counts[i]+=1
            x,y=np.clip(np.floor((uv+.5)/64).astype(int),0,3);grids[i][y,x]+=1
    bad=[]
    if len(points)<24:bad.append("fewer_than_24_distinct_triangulated_tracks")
    for i,s in enumerate(ids):
        if counts[i]<8:bad.append(s+"_fewer_than_8_tracks")
        if np.count_nonzero(grids[i])<4:bad.append(s+"_fewer_than_4_occupied_cells")
    return {"qualified":not bad,"failed_gates":bad,"independent_tracks":len(points),
            "source_coverage":{s:{"observed_tracks":counts[i],"occupied_cells":int(np.count_nonzero(grids[i])),"histogram_4x4":grids[i].tolist()} for i,s in enumerate(ids)}}


def triangulate(images,rig,ids):
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);cv2.setRNGSeed(43171)
    sift=cv2.SIFT_create(nfeatures=4000,nOctaveLayers=3,contrastThreshold=.02,edgeThreshold=10,sigma=1.6,enable_precise_upscale=True)
    feats=[measure_features(images[s],sift) for s in ids];cams=[rig[s] for s in ids]
    pairs={};edges=[]
    for i,s in enumerate(ids):
        for j in range(i+1,len(ids)):
            found,pairs[s+"-"+ids[j]]=pair_edges(i,j,feats[i],feats[j],cams[i],cams[j]);edges.extend(found)
    raw,stats=build_tracks(edges)
    good=[];rejected=Counter()
    for track in raw:
        obs=[(i,feats[i]["xy"][k]) for i,k in track]
        got,reason=point_from_track(obs,cams)
        if reason:rejected[reason]+=1
        else:got["ids"]=track;good.append(got)
    baseline=float(np.linalg.norm(rig["s03"]["C"]-rig["s00"]["C"]))
    require(np.isfinite(baseline) and baseline>1e-9,"Invalid shared endpoint baseline")
    points=unique_tracks(good,.0005*baseline)
    quality=eligibility(points,ids)
    xyz=np.asarray([p["point"] for p in points],np.float64).reshape(-1,3)
    stats.update(rejected={k:rejected[k] for k in ("invalid_DLT","outside_positive_source_clip","insufficient_parallax","reprojection_error")},
                 retained_before_spatial_dedup=len(good),duplicate_points_removed=len(good)-len(points),
                 track_length_histogram=dict(Counter(len(x["ids"]) for x in points)),
                 triangulation_angle_deg=summary([x["parallax_deg"] for x in points]),all_observation_reprojection_px=summary([e for x in points for e in x["errors"]]),
                 centre_bounds_m={"min":xyz.min(0).tolist(),"max":xyz.max(0).tolist()} if points else None)
    return points,baseline,quality,{"features":{s:{"raw":len(f["xy"]),"unique_at_0p001px":f["unique"]} for s,f in zip(ids,feats,strict=True)},"pairs":pairs,"tracks":stats}


def random_prior(rig,ids,scale):
    rng=np.random.Generator(np.random.PCG64(43171))
    endpoints=[rig[s] for s in ("s00","s03")]
    f=[c["pose"][:3,2]/np.linalg.norm(c["pose"][:3,2]) for c in endpoints]
    ang=float(np.degrees(np.arccos(np.clip(f[0]@f[1],-1,1))))
    d,_,rank,_=np.linalg.lstsq(np.stack([f[0],-f[1]],axis=1),endpoints[1]["C"]-endpoints[0]["C"],rcond=None)
    valid=(1<=ang<=179 and rank==2 and all(np.isfinite(x) and c["clip"][0]<x<c["clip"][1] for x,c in zip(d,endpoints,strict=True)))
    if not valid:d=[3*scale]*2
    points=[];bounds={}
    for s,c,focus in zip(("s00","s03"),endpoints,d,strict=True):
        near,far=c["clip"];eps=(far-near)*1e-6
        centre=float(np.clip(focus,near+eps,far-eps));lo=max(near+eps,centre-1.25*scale);hi=min(far-eps,centre+1.25*scale)
        require(0<lo<hi,"Invalid positive random inverse-depth interval")
        bounds[s]={"focus_m":centre,"positive_interval_m":[float(lo),float(hi)]}
        for y in range(32):
            for x in range(32):
                rx,ry,rz=rng.random(3)
                uv=np.array([(x+rx)*8-.5,(y+ry)*8-.5])
                z=1/(1/hi+rz*(1/lo-1/hi));ray=c["invK"]@np.r_[uv,1]
                pt=c["pose"][:3,:3]@(ray*z)+c["C"]
                points.append({"point":pt,"observations":[(ids.index(s),uv)],"depths":[z]})
    return points,{"method":"closest_endpoint_axes" if valid else "three_times_endpoint_baseline","endpoint_axis_angle_deg":ang,"sources":bounds,
                   "samples_per_endpoint":1024,"total":len(points),"seed":43171}


def linear_sample(im,uv):
    x,y=np.clip(uv,0,255);i,j=int(x),int(y);u,v=x-i,y-j
    a=im[j,i]*(1-u)+im[j,min(i+1,255)]*u
    b=im[min(j+1,255),i]*(1-u)+im[min(j+1,255),min(i+1,255)]*u
    rgb=(a*(1-v)+b*v)/255.
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)


def parameters(points,images,rig,ids,baseline):
    xyz=np.stack([x["point"] for x in points])
    require(len(xyz)>=4 and len(np.unique(xyz,axis=0))==len(xyz),"Initialization contains duplicate or degenerate point centres")
    nearest=cKDTree(xyz).query(xyz,k=4,workers=1)[0][:,1:]
    footprint=[];rgb=[]
    for point in points:
        metre=[];colors=[]
        for (i,uv),z in zip(point["observations"],point["depths"],strict=True):
            s=ids[i];k=rig[s]["K"]
            metre.append(z/np.sqrt(k[0,0]*k[1,1]))
            colors.append(linear_sample(images[s],uv))
        footprint.append(np.median(metre));rgb.append(np.median(colors,axis=0))
    footprint=np.array(footprint);rgb=np.array(rgb)
    proposed=np.maximum(footprint,.5*np.median(nearest,axis=1));low=np.maximum(.75*footprint,.001*baseline)
    raw_high=np.minimum(8*footprint,.075*baseline);high=np.maximum(low,raw_high)
    sigma=np.clip(proposed,low,high);color=np.clip(rgb,.001,.999)
    logits=np.log(color/(1-color));n=len(points)
    arrays={"means":xyz,"log_scales":np.tile(np.log(sigma)[:,None],(1,3)),"quats":np.tile(np.array([1,0,0,0]),(n,1)),
            "opacity_logits":np.full(n,np.log(.1/.9)),"color_logits":logits}
    arr={k:np.ascontiguousarray(v,dtype="<f4") for k,v in arrays.items()}
    require(np.isfinite(np.column_stack(list(arr.values()))).all() and len(np.unique(arr["means"],axis=0))==n,"Nonfinite or coincident stored init")
    stats={"sigma_m":summary(sigma),"footprint_m_per_px":summary(footprint),"nearest_three_median_m":summary(np.median(nearest,axis=1)),
           "sigma_lower_clipped":int((proposed<low).sum()),"sigma_upper_clipped":int((proposed>high).sum()),"scale_bound_conflicts":int((low>raw_high).sum()),
           "linear_rgb_before_clip":summary(rgb),"color_components_low_clipped":int((rgb<.001).sum()),"color_components_high_clipped":int((rgb>.999).sum()),"initial_opacity":.1,
           "actual_means_min_m":xyz.min(0).tolist(),"actual_means_max_m":xyz.max(0).tolist()}
    return arr,stats


def parameter_digest(arr):
    h=hashlib.sha256()
    for key in NAMES:
        v=np.ascontiguousarray(arr[key],dtype="<f4")
        h.update(key.encode()+b"\0");h.update(np.asarray(v.shape,dtype="<i4").tobytes());h.update(v.tobytes())
    return h.hexdigest()


def details(path,tracks,ids):
    n=len(tracks);nview=len(ids)
    xy=np.full((n,nview,2),np.nan,dtype="<f4"); index=np.full((n,nview),-1,dtype="<i4")
    residual=np.full((n,nview),np.nan,dtype="<f4"); depths=np.full_like(residual,np.nan)
    for r,p in enumerate(tracks):
        for (i,uv),(_,idx),e,z in zip(p["observations"],p["ids"],p["errors"],p["depths"],strict=True):
            xy[r,i]=uv;index[r,i]=idx;residual[r,i]=e;depths[r,i]=z
    np.savez_compressed(path,xyz_m=np.asarray([p["point"] for p in tracks],dtype="<f8").reshape(-1,3),uv=xy,feature_id=index,
                        reprojection_px=residual,source_camera_z_m=depths,parallax_deg=np.asarray([p["parallax_deg"] for p in tracks],dtype="<f4"))


def run(a):
    start=time.perf_counter()
    require(sha(a.protocol)==PROTOCOL_SHA,"Changed source geometry rules")
    p=load(a.protocol);cfg=load(a.config); c=cfg["condition_id"];ids=SETS[c]
    require(cfg["initialization_protocol_sha256"]==PROTOCOL_SHA and tuple(cfg["generator_source_ids"])==ids and
            sha(a.cameras)==cfg["source_camera_sha256"],"Initialization identity differs")
    versions={n:importlib.metadata.version(n) for n in ("opencv-python-headless","numpy","scipy","pillow")}
    require(all(versions[n]==p["runtime"][n.replace("-","_")] for n in versions) and cv2.__version__==p["runtime"]["opencv"],"Unregistered feature runtime")
    paths={s:Path(f) for s,f in a.source}
    require(len(a.source)==len(paths)==len(ids) and set(paths)==set(ids),"Extra source or target input")
    rig=cameras(load(a.cameras),ids);images={}
    for s,path in paths.items():
        require(sha(path)==cfg["source_rgb_sha256"][s],"Source photograph SHA differs")
        with Image.open(path) as image:
            require(image.format=="PNG" and image.mode=="RGB" and image.size==(256,256),"Original RGB8 crop required")
            images[s]=np.array(image)
    tracks,scale,quality,stats=triangulate(images,rig,ids); fallback=None
    if quality["qualified"]:chosen=tracks;method="known_pose_sift_triangulation"
    else:chosen,fallback=random_prior(rig,ids,scale);method="endpoint_stratified_inverse_depth"
    arr,scale_stats=parameters(chosen,images,rig,ids,scale)
    details(a.out/"source-tracks.npz",tracks,ids)
    np.savez(a.out/"init.npz",**arr)
    write(a.out/"init.json",{"schema":"el-recovery-init-1","run_id":os.environ["EL_RUN_ID"],"object_id":cfg["scene_family_id"],"condition":c,
          "registration":cfg["registration"],"recipe_sha256":cfg["recipe_sha256"],"initialization_protocol_sha256":PROTOCOL_SHA,
          "source_parent":cfg["source_parent"],"generator_source_ids":list(ids),"source_camera_sha256":sha(a.cameras),"source_rgb_sha256":cfg["source_rgb_sha256"],
          "scene_scale_m":scale,"shared_endpoints":["s00","s03"],"actual_method":method,"N":len(chosen),"parameter_names":list(NAMES),
          "parameter_sha256":parameter_digest(arr),"init_npz_sha256":sha(a.out/"init.npz"),"init_npz_bytes":(a.out/"init.npz").stat().st_size,
          "source_tracks_sha256":sha(a.out/"source-tracks.npz"),"eligibility":quality,"matching":stats,"parameters":scale_stats,"fallback":fallback,
          "frame":"Original el-camera-1 world in metres, OpenCV integer-centre source pixels, quaternion wxyz, scales exp(log_scales), linear RGB and opacity sigmoid(logits)",
          "versions":versions,"elapsed_seconds":time.perf_counter()-start})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("protocol","config","cameras","out"):p.add_argument("--"+n,required=True,type=Path)
    p.add_argument("--source",nargs=2,action="append",required=True)
    run(p.parse_args())
