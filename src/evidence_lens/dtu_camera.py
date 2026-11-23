"""Metric rectified DTU cameras and the dtu-crop256-v1 external adapter.

All view choices and the shared rigid world frame use calibration alone. The
original projective matrices remain available for an independent pixel check.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

SOURCES = ("s00", "s01", "s02", "s03")
TARGETS = tuple(f"v{n:02d}" for n in range(1, 7))
CONDITIONS = {"M2": ("s00", "s03"), "M4": SOURCES}
PRIOR_CENTRE_M = np.array([0.0, 0.4, 0.85])
CAMERA_HEADER = {"schema": "el-camera-1", "unit": "m", "pixel_center": "integer",
                 "image_origin": "top_left", "axis_convention": "OpenCV x right, y down, z forward"}
CLIP_M = [0.05, 30.0]


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def record(data):
    return {"sha256": digest(data), "bytes": len(data)}


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def crop_geometry(width, height):
    require(all(type(i) is int and i > 0 for i in (width, height)), "Bad original image size")
    side = min(width, height)
    left, top = (width-side)//2, (height-side)//2
    scale = 256.0 / side
    affine = [[scale, 0.0, scale*(0.5-left)-0.5], [0.0, scale, scale*(0.5-top)-0.5], [0., 0., 1.]]
    return {"schema":"el-dtu-crop-1", "id":"dtu-crop256-v1", "width":width, "height":height,
            "left":left, "top":top, "side":side, "output":[256,256], "scale":scale, "affine":affine,
            "resampling":"Pillow LANCZOS in the supplied display RGB encoding; half-pixel centres",
            "color":"RGB8 display values without exposure, alignment or fitted colour transformation"}


def convert_rgb(data, crop):
    with Image.open(io.BytesIO(data)) as image:
        require(image.size == (crop["width"],crop["height"]) and image.mode == "RGB", "Expected native three-channel RGB")
        require(image.getexif().get(274,1)==1, "Unregistered pixel orientation")
        require(not image.info.get("icc_profile"), "An embedded colour profile needs explicit handling before freeze")
        if image.format == "PNG":
            require(data[24:26] == b"\x08\x02", "Expected truecolour, eight bits per channel")
        if image.format == "PNG" and image.size == (256,256):
            return data
        left, top, side = crop["left"], crop["top"], crop["side"]
        resized = image.crop((left,top,left+side,top+side)).resize((256,256),Image.Resampling.LANCZOS)
        out = io.BytesIO()
        resized.save(out,format="PNG",compress_level=6)
        return out.getvalue()


def unit_vector(v):
    length = np.linalg.norm(v)
    require(np.isfinite(length) and length>1e-9, "Degenerate camera-frame vector")
    return v/length


def angle(a,b):
    return float(np.arccos(np.clip(np.dot(unit_vector(a),unit_vector(b)),-1,1)))


def decompose(matrix, metres_per_unit, first_pixel):
    p = np.asarray(matrix,np.float64)
    require(p.shape==(3,4) and np.isfinite(p).all() and np.linalg.matrix_rank(p[:,:3])==3,
            "Rectified projection must be finite and full rank")
    require(np.isfinite(metres_per_unit) and metres_per_unit>0 and first_pixel in (0,1), "Unverified metric or pixel units")
    p = p * np.sign(np.linalg.det(p[:,:3]))
    q, upper = np.linalg.qr(np.flipud(p[:,:3]).T)
    k = np.fliplr(np.flipud(upper.T))
    rot = np.flipud(q.T)
    sign = np.diag(np.sign(np.diag(k)))
    k, rot = k @ sign, sign @ rot
    k /= k[2,2]
    require(np.allclose(rot@rot.T,np.eye(3),atol=1e-10) and np.linalg.det(rot)>0, "Not a right-handed world-to-camera rotation")
    centre = -np.linalg.solve(p[:,:3],p[:,3]) * metres_per_unit
    k[:2,2] -= first_pixel
    pose = np.eye(4)
    pose[:3,:3], pose[:3,3] = rot.T, centre
    return k, pose


def ray_frame(a,b):
    u,v = a[:3,2], b[:3,2]
    baseline = b[:3,3]-a[:3,3]
    axes = np.column_stack([u,-v])
    lengths,_,rank,_ = np.linalg.lstsq(axes,baseline,rcond=None)
    require(rank==2 and all(lengths>0), "Endpoints have no stable forward-axis convergence")
    endpoints = [a[:3,3]+lengths[0]*u,b[:3,3]+lengths[1]*v]
    aim = (endpoints[0]+endpoints[1])/2
    x = unit_vector(a[:3,0]+b[:3,0])
    f = unit_vector(u+v)
    y = unit_vector(f-np.dot(f,x)*x)
    z = unit_vector(np.cross(x,y))
    basis = np.stack([x,y,z])
    dist = float(np.linalg.norm(baseline))
    require(dist>1e-8 and np.dot(x,baseline)>0, "Endpoint order must run left to right")
    return {"basis":basis, "aim":aim, "angle":angle(u,v), "distance":dist, "range":float(lengths.mean()),
            "mid":(a[:3,3]+b[:3,3])/2,"ray_gap":float(np.linalg.norm(endpoints[0]-endpoints[1])),
            "baseline_direction":basis@baseline/dist}


def relative_views(ids,poses,frame):
    return {i:((frame["basis"]@(poses[i][:3,3]-frame["mid"]))/frame["distance"],
               frame["basis"]@poses[i][:3,2]) for i in ids}


def select_views(poses, published):
    """Rank endpoint shape first, then assign interior sources and six targets by geometry."""
    old = {key:np.asarray(x["c2w"],np.float64) for key,x in published["views"].items()}
    old_frame = ray_frame(old["s00"],old["s03"])
    logical = ("s01","s02",*TARGETS)
    layout = relative_views(logical,old,old_frame)
    candidates=[]
    for ia,ib in itertools.combinations(sorted(poses),2):
        a,b=poses[ia],poses[ib]
        if np.dot(b[:3,3]-a[:3,3],a[:3,0]+b[:3,0])<0:
            ia,ib=ib,ia
            a,b=b,a
        try:
            f=ray_frame(a,b)
        except ValueError:
            continue
        if not (np.deg2rad(1)<=f["angle"]<np.pi/3 and f["ray_gap"]<=f["distance"]):
            continue
        angle_gap=(f["angle"]-old_frame["angle"])/old_frame["angle"]
        ratio_gap=(f["distance"]/f["range"])/(old_frame["distance"]/old_frame["range"])-1
        direction_gap=np.linalg.norm(f["baseline_direction"]-old_frame["baseline_direction"])
        endpoint_cost=angle_gap**2 + ratio_gap**2 + 4*direction_gap**2 + (f["ray_gap"]/f["distance"])**2
        candidates.append((float(endpoint_cost),ia,ib,f))
    require(candidates, "No usable metric camera pair")
    # Test each eligible pair against the complete rig shape. The finite tie-break is the official numeric camera ID.
    best=None
    for ep_cost,ia,ib,f in candidates:
        if best and ep_cost>best[0][0]:
            continue
        available=sorted(set(poses)-{ia,ib})
        rel=relative_views(available,poses,f)
        assignment={"s00":ia,"s03":ib}
        errors={}
        for name in logical:
            position,forward=layout[name]
            def cost(i):
                p,d=rel[i]
                return float(np.linalg.norm(position-p)**2 + (angle(forward,d)/max(old_frame["angle"],0.01))**2)
            require(available, "At least ten distinct official rectified views are required")
            picked=min(available,key=lambda i:(cost(i),i))
            errors[name]=cost(picked)
            assignment[name]=picked
            available.remove(picked)
        total=ep_cost+float(np.mean(list(errors.values())))
        key=(total,ia,ib,*(assignment[n] for n in logical))
        if best is None or key<best[0]:
            best=(key,assignment,f,ep_cost,errors)
    _,mapping,f,ep,errs=best
    def stats(k):
        return {"endpoint_angle_deg":float(np.rad2deg(k["angle"])),"endpoint_distance_m":k["distance"],
                "convergence_range_m":k["range"],"closest_ray_gap_m":k["ray_gap"],
                "baseline_over_range":k["distance"]/k["range"]}
    return mapping, f, {"algorithm":"dtu-geometry-layout-v1","endpoint_cost":ep,"assigned_role_costs":errs,
                        "stage3":stats(old_frame),"dtu":stats(f),"roles":logical,
                        "rank":"endpoint angular and baseline/range squared relative errors, 4× baseline direction squared and relative ray gap squared; plus mean squared normalized 3D positions and relative optical-angle errors. Greedy role order s01,s02,v01..v06. Ties use official ID."}


def camera_outputs(projections, width, height, factor, first_pixel, published):
    original={i:decompose(p,factor,first_pixel) for i,p in projections.items()}
    poses={i:v[1] for i,v in original.items()}
    ids,frame,fit=select_views(poses,published)
    world=np.eye(4)
    world[:3,:3]=frame["basis"]
    world[:3,3]=PRIOR_CENTRE_M-frame["basis"]@frame["aim"]
    crop=crop_geometry(width,height)
    affine=np.array(crop["affine"])
    views,checks={},{}
    for key in SOURCES+TARGETS:
        idx=ids[key]
        native_k,pose=original[idx]
        k=affine@native_k
        # The frozen deployment reader supports zero skew. Any projected impact from dropping
        # published coefficient roundoff is measured against the original 3x4 P below.
        require(abs(k[0,1])<=0.003 and abs(k[1,0])<=1e-9,"Material rectified skew is unsupported by the released scorer")
        raw_skew=float(k[0,1]); k[0,1]=k[1,0]=0
        c2w=world@pose
        views[key]={"width":256,"height":256,"K":k.tolist(),"c2w":c2w.tolist(),"clip_m":CLIP_M}
        # Independent world points span the actual raster and three positive axial depths.
        pix=np.array([[.25,.75],[254.8,.12],[.8,254.4],[253.75,254.25],[52.12,143.07],[177.3,78.1],[127.5,127.5]])
        u=np.column_stack([pix,np.ones(len(pix))])
        error=deptherr=0.
        p=np.asarray(projections[idx])
        for z in (.21,.87,2.43):
            camera_xyz=np.linalg.solve(k,u.T).T*z
            native_m=(pose[:3,:3]@camera_xyz.T).T+pose[:3,3]
            native_h=np.column_stack([native_m/factor,np.ones(len(pix))])
            uvw=(p@native_h.T).T
            uv0=uvw[:,:2]/uvw[:,2:3]-first_pixel
            expected=(affine@np.column_stack([uv0,np.ones(len(pix))]).T).T[:,:2]
            canon=(world[:3,:3]@native_m.T).T+world[:3,3]
            xyz=(c2w[:3,:3].T@(canon-c2w[:3,3]).T).T
            got=(k@xyz.T).T
            error=max(error,float(np.max(np.abs(got[:,:2]/got[:,2:3]-expected))))
            deptherr=max(deptherr,float(np.max(np.abs(xyz[:,2]-z))))
        require(error<=0.005 and deptherr<=1e-10, "Original DTU projection and metric crop disagree")
        checks[key]={"official_view_id":idx,"probe_points":21,"max_abs_reprojection_px":error,"max_abs_axial_z_m":deptherr,
                     "raw_cropped_skew_px":raw_skew}
    return {"cameras":{**CAMERA_HEADER,"views":views},"crop":crop,"ids":ids,"fit":fit,"checks":checks,
            "frame":{"schema":"el-dtu-metric-frame-1","endpoint_ids":[ids["s00"],ids["s03"]],
                     "metres_per_native_unit":factor,"prior_centre_m":PRIOR_CENTRE_M.tolist(),
                     "triangulated_axis_centre_native_m":frame["aim"].tolist(),"dtu_metres_to_canonical_metres":world.tolist(),
                     "canonical_metres_to_dtu_metres":np.linalg.inv(world).tolist(),"additional_scale":1.0,
                     "axis_definition":"x mean endpoint image right; y mean optical forward orthogonalized to x; z=x cross y"}}
