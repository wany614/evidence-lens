"""Fit the registered small NumPy logistic head using authored training rows only."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


LAMBDAS = (0.001, 0.01, 0.1)
TRAIN = ("b01-staggered-arcade", "b02-terraced-court", "b03-woven-truss")
VIEWS = tuple(f"h{i:02d}" for i in range(6))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_bytes())


def save(path, data):
    Path(path).write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fp32_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a,dtype="<f4").tobytes()).hexdigest()


def sample_ids(family, count, view, labelled):
    n = int(count)
    identity = f"617312/{family}/{n}/{view}".encode("ascii")
    seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
    ids = np.flatnonzero(labelled)
    return np.sort(np.random.Generator(np.random.PCG64(seed)).permutation(ids)[:1000]).astype(np.int32)


def balanced_weights(unit, pixel):
    units, inv = np.unique(unit, return_inverse=True)
    weights = np.zeros(len(unit), np.float64)
    for idx in range(len(units)):
        pos = np.flatnonzero(inv == idx)
        _, mapped, repetitions = np.unique(pixel[pos], return_inverse=True, return_counts=True)
        w = 1.0 / repetitions[mapped]
        weights[pos] = w / (w.sum() * len(units))
    return weights


def collect(manifests):
    from evidence_lens_many import FEATURE_NAMES, FEATURE_SCHEMA_SHA256

    X, ys, unit, pixel, ledger = [], [], [], [], []
    seen = set()
    for path in manifests:
        d = load(path)
        family, count = d["family"], d["R"]
        check(d["schema"] == "el-many-authored-training-sample-1" and d["role"] == "train"
              and family in TRAIN and count in (16,128) and (family,count) not in seen,
              "training family, identity or role differs")
        check(d["feature_schema_sha256"] == FEATURE_SCHEMA_SHA256 and tuple(d["columns"]) == tuple(FEATURE_NAMES)
              and d["target_ids"] == list(VIEWS), "training schema or cameras differ")
        seen.add((family,count))
        file = Path(path).parent / "samples.npz"
        check(digest(file) == d["samples_sha256"], "training row bytes differ")
        with np.load(file, allow_pickle=False) as a:
            x, y, v, px, k = (a[n] for n in ("X","y","view","pixel","k"))
            check(x.dtype==np.float32 and x.ndim==2 and x.shape[1]==len(FEATURE_NAMES) and
                  all(arr.shape==(len(x),) for arr in (y,v,px,k)) and np.isin(y,[0,1]).all()
                  and not np.isinf(x).any() and (px>=0).all() and (px<378*252).all(),
                  "invalid sampled matrix")
            indices = np.stack([v.astype(np.int64),px,k], axis=1)
            check(len(np.unique(indices,axis=0))==len(indices), "duplicated K-specific row")
            check(set(np.unique(v)) == set(range(6)) and np.isin(k,[4,8,16,32,128]).all()
                  and (k<=count).all(), "missing registered training view or unsupported budget")
            X.append(x); ys.append(y); pixel.append(px)
            unit.append(v.astype(np.int32) + 6*(2*TRAIN.index(family)+(count==128)))
        ledger.append(dict(family=family,R=count,status="train",sample_receipt_sha256=digest(path),
                           samples_sha256=d["samples_sha256"],bank_sha256=d["bank_sha256"],
                           labels_sha256=d["label_hashes"],rows=len(x)))
    check(seen=={(f,n) for f in TRAIN for n in (16,128)}, "incomplete six-checkpoint training corpus")
    X=np.concatenate(X); y=np.concatenate(ys).astype(np.float64)
    check(len(X)<=200000 and y.min()==0 and y.max()==1, "row budget or binary classes invalid")
    unit=np.concatenate(unit); pixel=np.concatenate(pixel)
    check(len(np.unique(unit))==36,"missing family/condition/view mass")
    w=balanced_weights(unit,pixel)
    return X,y,w,ledger


def preprocess(X,w):
    ok=np.isfinite(X)
    mass=w @ ok.astype(np.float64)
    clean=np.where(ok,X,0).astype(np.float64)
    mean=np.divide(w @ clean,mass,out=np.zeros(X.shape[1]),where=mass>0)
    values=np.where(ok,X,mean)
    delta=values-mean
    variance=np.divide(w @ (delta*delta),mass,out=np.zeros_like(mean),where=mass>0)
    scale=np.sqrt(variance)
    scale[scale<1e-12]=1.
    return np.ascontiguousarray(delta/scale),mean,scale,mass


def sigmoid(z):
    return np.exp(-np.logaddexp(0.,-z))


def fit(A,y,w,penalty):
    # A weighted Newton step with backtracking makes the fit and termination reproducible without a training runtime.
    x=np.empty((len(A),A.shape[1]+1),np.float64)
    x[:,:-1]=A
    x[:,-1]=1
    theta=np.zeros(x.shape[1],np.float64)
    prior=float(w@y)
    theta[-1]=np.log(prior/(1-prior))
    def loss(t):
        z=x@t
        return float(w @ (np.logaddexp(0,z) - y*z) + .5*penalty*(t[:-1]@t[:-1]))
    history=[]
    converged=False
    for iteration in range(60):
        prob=sigmoid(x@theta)
        gradient=x.T@(w*(prob-y))
        gradient[:-1]+=penalty*theta[:-1]
        norm=float(np.max(np.abs(gradient)))
        value=loss(theta)
        history.append(dict(iteration=iteration,loss=value,max_absolute_gradient=norm))
        if norm<=2e-9:
            converged=True
            break
        hess=x.T@((w*prob*(1-prob))[:,None]*x)
        regularized=np.arange(len(theta)-1)
        hess[regularized,regularized]+=penalty
        direction=np.linalg.solve(hess,gradient)
        slope=gradient@direction
        check(np.isfinite(slope) and slope>0,"invalid Newton descent direction")
        step=1.
        if .5*slope>1e-13:
            while loss(theta-step*direction)>value-1e-4*step*slope:
                step*=.5
                check(step>=2**-28,"fit failed to find a descent step")
        else:
            history[-1]["near_optimum_full_newton_step"]=True
        theta-=step*direction
        history[-1]["accepted_step"]=step
    check(converged and np.isfinite(theta).all(),"registered optimization did not converge")
    return theta,history


def fit_models(manifests,out,penalties,selected=None):
    from evidence_lens_many import FEATURE_NAMES, FEATURE_SCHEMA_SHA256
    from evidence_lens_many.schema import BASELINE_ID

    cpu=time.process_time(); wall=time.perf_counter()
    X,y,w,ledger=collect(manifests)
    A,mean,scale,mass=preprocess(X,w)
    out.mkdir(exist_ok=True, parents=True)
    ids=[]
    for lam in penalties:
        theta,history=fit(A,y,w,lam)
        model=dict(schema="el-b-cpu-lr-1",baseline_id=BASELINE_ID,feature_schema_sha256=FEATURE_SCHEMA_SHA256,
            columns=list(FEATURE_NAMES),impute=mean.tolist(),mean=mean.tolist(),scale=scale.tolist(),
            coef=theta[:-1].tolist(),intercept=float(theta[-1]),tau=None,lambda_l2=lam,
            method="B-CPU-LR",model_id=f"b-cpu-lr-l2-{lam:g}"+("-final" if selected else ""),
            training_role="original_authored_only",training_receipts_sha256=[v["sample_receipt_sha256"] for v in ledger])
        # Recalculate each logit from original rows and exported JSON representation; retain numerical evidence.
        decoded=json.loads(json.dumps(model))
        back=(np.where(np.isfinite(X),X,decoded["impute"])-decoded["mean"])/decoded["scale"]
        ref=sigmoid(A@theta[:-1]+theta[-1])
        candidate=sigmoid(np.sum(back*np.asarray(decoded["coef"]),axis=1)+decoded["intercept"])
        diff=float(np.max(np.abs(ref-candidate)))
        check(diff<1e-12,"CPU export differs on the training replay")
        path=out/("model.json" if selected else "candidate-"+format(lam,"g")+".json")
        save(path,model)
        if selected:
            initial=load(selected)
            for k in ("coef","intercept","impute","mean","scale","lambda_l2","training_receipts_sha256"):
                check(initial[k]==model[k],"final deterministic refit differs: "+k)
        ids.append(dict(file=path.name,sha256=digest(path),lambda_l2=lam,converged=True,trace=history,
                        cpu_recalculation_max_abs=diff,training_score_fp32_sha256=fp32_hash(ref)))
    save(out/"fits.json",dict(schema="el-many-head-fits-1",run_id=os.getenv("EL_RUN_ID"),rows=int(len(X)),
         positive_rows=int(y.sum()),weighted_error_base_rate=float(w@y),training_weights_sum=float(w.sum()),
         feature_schema_sha256=FEATURE_SCHEMA_SHA256,observed_weight_per_feature=mass.tolist(),training=ledger,
         candidates_or_final=ids,fit_count=len(ids),calibration_fits=0,neural_updates=0,
         cpu_core_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample",type=Path,action="append",required=True)
    p.add_argument("--selected",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    fit_models(a.sample,a.out,[load(a.selected)["lambda_l2"]] if a.selected else LAMBDAS,a.selected)
