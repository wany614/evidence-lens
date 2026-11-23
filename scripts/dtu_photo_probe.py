"""Check registered camera geometry against development source photographs, without fitting a pose."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter, sobel


def patches(path):
    with Image.open(path) as image:
        a = np.asarray(image.convert("L"), dtype=np.float32) / 255
    if a.shape != (256, 256):
        raise ValueError("expected the registered crop")
    x, y = sobel(a, axis=1), sobel(a, axis=0)
    xx, xy, yy = [gaussian_filter(v, 1.5) for v in (x*x, x*y, y*y)]
    strength = (xx + yy - np.sqrt((xx-yy)**2 + 4*xy**2)) / 2
    yy, xx = np.nonzero((strength == maximum_filter(strength, 7)) & (strength > 0.002))
    keep = (xx >= 6) & (yy >= 6) & (xx < 250) & (yy < 250)
    xx, yy = xx[keep], yy[keep]
    order = np.argsort(-strength[yy, xx], kind="stable")[:384]
    xy = np.column_stack((xx[order], yy[order]))
    if len(xy) < 2:
        raise ValueError("insufficient source texture for appearance matching")
    vectors = np.stack([a[v-5:v+6, u-5:u+6].ravel() for u,v in xy])
    vectors -= vectors.mean(axis=1,keepdims=True)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    return xy, vectors


def compare(left, right, camera_a, camera_b):
    points_a, da = left
    points_b, db = right
    similarity = da @ db.T
    j = similarity.argmax(axis=1)
    i = np.arange(len(j))
    second = np.partition(similarity, -2, axis=1)[:,-2]
    keep = (similarity.argmax(axis=0)[j] == i) & (similarity[i,j] >= 0.85) & (similarity[i,j]-second >= 0.035)
    pa, pb = [np.column_stack((p, np.ones(len(p)))) for p in (points_a[keep], points_b[j[keep]])]
    a,b = (np.asarray(c["c2w"]) for c in (camera_a,camera_b))
    ka,kb = (np.asarray(c["K"]) for c in (camera_a,camera_b))
    r = b[:3,:3].T @ a[:3,:3]
    t = b[:3,:3].T @ (a[:3,3] - b[:3,3])
    tx = np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    fundamental = np.linalg.inv(kb).T @ tx @ r @ np.linalg.inv(ka)
    lb,la = pa @ fundamental.T, pb @ fundamental
    cross = np.abs(np.sum(lb*pb, axis=1))
    symmetric = cross*0.5*(1/np.maximum(np.linalg.norm(la[:,:2],axis=1),1e-14)
                           +1/np.maximum(np.linalg.norm(lb[:,:2],axis=1),1e-14))
    return {"corners":[len(da),len(db)], "mutual_appearance_matches":int(keep.sum()),
            "below_1px":int(np.sum(symmetric<1)), "below_2px":int(np.sum(symmetric<2)),
            "median_px":float(np.median(symmetric)) if len(symmetric) else None,
            "p90_px":float(np.quantile(symmetric,0.9)) if len(symmetric) else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for role in ("sources","cameras","out"):
        p.add_argument("--"+role, type=Path, required=True)
    a = p.parse_args()
    cams = json.loads(a.cameras.read_bytes())["views"]
    source_ids = ["s00","s01","s02","s03"]
    if set(cams) != set(source_ids):
        raise ValueError("source-only camera table required")
    features = {name:patches(a.sources / (name+".png")) for name in source_ids}
    pairs = {i+"_"+j:compare(features[i], features[j], cams[i], cams[j]) for i,j in itertools.combinations(source_ids,2)}
    doc = {"schema":"el-dtu-physical-photo-probe-1", "object":"set1", "pairs":pairs,
           "basis":"Image-only, cross-checked normalized 11x11 patches on up to 384 corners per photograph. No epipolar or spatial gate during selection. Symmetric pixel-to-epipolar-line distance from the supplied K and poses. No homography, pose or scale fitted. False matches and weak texture can change the tail."}
    (a.out/"photo-probe.json").write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
