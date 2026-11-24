"""Build a source-only COLMAP reconstruction from calibrated feature and optical-flow matches."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image

from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens_bundle.common import require, parts


RULE = {"algorithm": "SIFT mutual ratio plus forward-backward Lucas-Kanade, calibrated two-view triangulation",
        "opencv": "4.11.0", "source_neighbours": 6, "sift_max": 2400, "ratio": 0.80, "lk_fb_px": 0.75,
        "max_reprojection_px": 1.25, "min_parallax_deg": 0.75, "max_parallax_deg": 80,
        "max_mean_rgb_discrepancy_8bit": 45, "max_radius_in_source_extents": 80,
        "fusion_voxel_in_source_extents": 0.0025, "minimum_points": 16, "maximum_points": 15000}


def source_inputs(z):
    require(z.namelist().count("scene.json") == 1 and len(z.namelist()) == len(set(z.namelist())), "non-unique source archive")
    scene = json.loads(z.read("scene.json"))
    ids = scene["R"]
    require(scene["schema"] == "el-many-source-data-1" and scene["A"] == ids and 4 <= len(ids) == len(set(ids))
            and len(scene["H"]) == len(set(scene["H"])) and set(ids).isdisjoint(scene["H"])
            and set(scene["source_rgb"]) == set(scene["original_rgb"]) == set(ids), "wrong registered source/held-out roles")
    require(set(scene["views"]) == set(ids + scene["H"]) and scene["unit"] in ("m", "scene_units"), "camera or unit contract")
    photos = {}
    for key in ids:
        row = scene["source_rgb"][key]
        parts(row["path"])
        data = z.read(row["path"])
        require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], "changed analysis photograph")
        with Image.open(io.BytesIO(data)) as im:
            require(im.mode == "RGB" and im.size == (378, 252) and im.format == "PNG", "wrong analysis photo grid")
            photos[key] = np.asarray(im).copy()
    return scene, photos


def camera_arrays(cam):
    k = np.asarray(cam["K"], np.float64)
    c = np.asarray(cam["c2w"], np.float64)
    require((cam["width"], cam["height"]) == (378, 252) and k.shape == (3, 3) and c.shape == (4, 4)
            and np.isfinite(k).all() and np.isfinite(c).all() and np.allclose(c[3], [0,0,0,1])
            and np.allclose(c[:3,:3].T @ c[:3,:3], np.eye(3), atol=1e-5)
            and np.linalg.det(c[:3,:3]) > .9999, "invalid calibrated source camera")
    return k, c, k @ np.linalg.inv(c)[:3]


def pairs(cameras):
    centers = np.stack([c[:3, 3] for _, c, _ in cameras])
    directions = np.stack([c[:3, 2] for _, c, _ in cameras])
    distances = np.linalg.norm(centers[:,None] - centers[None,:], axis=2)
    np.fill_diagonal(distances, np.inf)
    extent = float(np.max(np.linalg.norm(centers - centers.mean(0), axis=1)))
    spacing = np.median(np.min(distances, axis=1))
    require(extent > 1e-8 and spacing > 1e-10, "coincident or degenerate source camera track")
    metric = distances / spacing + 3 * (1 - np.clip(directions @ directions.T, -1, 1))
    graph = set()
    for i, values in enumerate(metric):
        for j in np.argsort(values, kind="stable")[:min(len(values)-1, RULE["source_neighbours"])]:
            graph.add(tuple(sorted((i, int(j)))))
    return sorted(graph), extent, centers.mean(0)


def gray_features(rgb, features):
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    keys, descriptors = features.detectAndCompute(gray, None)
    xy = np.asarray([k.pt for k in keys], np.float32).reshape(-1, 2)
    corners = cv2.goodFeaturesToTrack(gray, 1400, 0.001, 4, blockSize=5)
    y, x = np.mgrid[4:gray.shape[0]-4:7, 4:gray.shape[1]-4:7]
    gradient = cv2.cornerMinEigenVal(gray, blockSize=5)[y, x]
    grid = np.column_stack((x[gradient > 1e-5], y[gradient > 1e-5])).astype(np.float32)
    track = np.vstack((corners.reshape(-1, 2), grid)) if corners is not None else grid
    return gray, xy, descriptors, track.reshape(-1, 1, 2)


def correspondences(first, second):
    import cv2
    ga, ka, da, seeds = first
    gb, kb, db, _ = second
    aa, bb = [], []
    if da is not None and db is not None and min(len(da), len(db)) >= 2:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        def mutual(d1, d2):
            return {m.queryIdx:m.trainIdx for m, n in matcher.knnMatch(d1, d2, k=2) if m.distance < RULE["ratio"] * n.distance}
        forward, reverse = mutual(da, db), mutual(db, da)
        good = [(i,j) for i,j in forward.items() if reverse.get(j) == i]
        if good:
            i,j = np.asarray(good).T
            aa.append(ka[i]); bb.append(kb[j])
    if len(seeds):
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.015)
        opts = dict(winSize=(21,21), maxLevel=3, criteria=criteria)
        out, ok, _ = cv2.calcOpticalFlowPyrLK(ga, gb, seeds, None, **opts)
        back, ok2, _ = cv2.calcOpticalFlowPyrLK(gb, ga, out, None, **opts)
        ss, tt, uu = (x.reshape(-1,2) for x in (seeds, out, back))
        keep = ok[:,0].astype(bool) & ok2[:,0].astype(bool) & (np.linalg.norm(ss-uu, axis=1) <= RULE["lk_fb_px"])
        aa.append(ss[keep]); bb.append(tt[keep])
    if not aa:
        return np.empty((0,2)), np.empty((0,2))
    return np.vstack(aa), np.vstack(bb)


def triangulate(first, second, p, q, im_a, im_b, extent, center):
    import cv2
    if not len(p):
        return []
    coords = np.column_stack((p,q))
    in_frame = np.isfinite(coords).all(1) & (coords >= 0).all(1) & (coords <= [377,251,377,251]).all(1)
    p, q = p[in_frame], q[in_frame]
    if not len(p):
        return []
    with np.errstate(invalid="ignore", divide="ignore"):
        X = cv2.triangulatePoints(first[2], second[2], p.astype(np.float64).T, q.astype(np.float64).T).T
        X /= X[:,3:4]
        a, b = X @ first[2].T, X @ second[2].T
        err = np.maximum(np.linalg.norm(a[:,:2]/a[:,2:3]-p, axis=1), np.linalg.norm(b[:,:2]/b[:,2:3]-q, axis=1))
        rays = [X[:,:3] - v[1][:3,3] for v in (first, second)]
        cs = np.sum(rays[0]*rays[1], axis=1) / (np.linalg.norm(rays[0],axis=1)*np.linalg.norm(rays[1],axis=1))
        angle = np.degrees(np.arccos(np.clip(cs, -1, 1)))
    ip, iq = np.rint(p).astype(int), np.rint(q).astype(int)
    c1, c2 = im_a[ip[:,1], ip[:,0]].astype(float), im_b[iq[:,1], iq[:,0]].astype(float)
    photo_error = np.abs(c1-c2).mean(1)
    good = ((a[:,2]>0) & (b[:,2]>0) & (err<=RULE["max_reprojection_px"]) &
            (angle>=RULE["min_parallax_deg"]) & (angle<=RULE["max_parallax_deg"]) &
            (np.linalg.norm(X[:,:3]-center,axis=1)<=RULE["max_radius_in_source_extents"]*extent) &
            (photo_error<=RULE["max_mean_rgb_discrepancy_8bit"]))
    use = np.flatnonzero(good)
    rgb = np.rint((c1+c2) *.5).astype(np.uint8)
    return [(X[n,:3], rgb[n], float(err[n]), p[n], q[n]) for n in use]


def quaternion(rotation):
    import cv2
    v = cv2.Rodrigues(rotation)[0].ravel()
    angle = np.linalg.norm(v)
    return np.r_[np.cos(angle/2), v * (np.sin(angle/2)/angle if angle > 1e-12 else 0.5)]


def num(v):
    return " ".join(format(float(x), ".17g") for x in v)


def run(scene, photos, out):
    import cv2
    require(cv2.__version__.split("-")[0] == RULE["opencv"], "unregistered OpenCV version")
    start_cpu, start = time.process_time(), time.perf_counter()
    cv2.setNumThreads(2); cv2.setRNGSeed(42)
    ids = scene["R"]
    cams = [camera_arrays(scene["views"][i]) for i in ids]
    graph, extent, center = pairs(cams)
    sift = cv2.SIFT_create(nfeatures=RULE["sift_max"], contrastThreshold=.012, edgeThreshold=12)
    features = [gray_features(photos[v], sift) for v in ids]
    survivors, stats = {}, []
    for i,j in graph:
        p, q = correspondences(features[i], features[j])
        points = triangulate(cams[i],cams[j],p,q,photos[ids[i]],photos[ids[j]],extent,center)
        for xyz,rgb,err,xy0,xy1 in points:
            cell = tuple(np.floor((xyz-center)/(extent*RULE["fusion_voxel_in_source_extents"])).astype(np.int64).tolist())
            if cell not in survivors or err < survivors[cell][2]:
                survivors[cell]=(xyz.copy(),rgb.copy(),err,i,j,xy0.copy(),xy1.copy())
        stats.append({"source_ids":[ids[i],ids[j]], "matches":len(p),"positive_triangulations":len(points)})
    require(len(survivors) >= RULE["minimum_points"], "insufficient independent R-only triangulations")
    keys = sorted(survivors, key=lambda k: hashlib.sha256(np.asarray(k,dtype="<i8").tobytes()).digest())[:RULE["maximum_points"]]
    obs = [[] for _ in ids]
    rows = []
    for point_id,key in enumerate(keys, 1):
        xyz,rgb,err,i,j,pa,pb = survivors[key]
        track = []
        for idx,xy in ((i,pa),(j,pb)):
            track += [idx+1,len(obs[idx])]
            obs[idx].append(num(xy+.5)+" "+str(point_id))
        rows.append(" ".join((str(point_id),num(xyz)," ".join(map(str,rgb)),format(err,".17g")," ".join(map(str,track)))))
    cameras, images = [], []
    for i,(key,(K,pose,_)) in enumerate(zip(ids,cams),1):
        M = np.linalg.inv(pose)
        cameras.append(f"{i} PINHOLE 378 252 "+num([K[0,0],K[1,1],K[0,2]+.5,K[1,2]+.5]))
        images += [f"{i} "+num(quaternion(M[:3,:3]))+" "+num(M[:3,3])+f" {i} {i:04d}.png"," ".join(obs[i-1])]
    folder=out/"sparse/0";folder.mkdir(parents=True)
    for n,strings in (("cameras",cameras),("images",images),("points3D",rows)):
        (folder/(n+".txt")).write_text("\n".join(strings)+"\n", encoding="ascii")
    return {"schema":"el-many-source-sfm-1","run_id":os.environ["EL_RUN_ID"],"recipe":RULE,"source_ids":ids,"scene_id":scene["scene_id"],
            "photo_sha256":{k:scene["source_rgb"][k]["sha256"] for k in ids},"source_camera_extent":extent,"unit":scene["unit"],
            "pairs":stats,"unique_points_before_cap":len(survivors),"initial_points":len(keys),"source_tracks":sum(map(len,obs)),
            "components":{f"sparse/0/{p.name}":sha(p) for p in folder.iterdir()},"sift_features":{v:len(features[i][1]) for i,v in enumerate(ids)},
            "source_cpu_seconds":time.process_time()-start_cpu,"wall_seconds":time.perf_counter()-start,"official_points_or_tracks_read":0,"target_photos_opened":0}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    with zipfile.ZipFile(a.data) as z:
        scene, photos=source_inputs(z)
    result=run(scene,photos,a.out)
    save(a.out/"sfm.json",result)
