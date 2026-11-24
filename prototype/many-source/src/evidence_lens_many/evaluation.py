"""Independent evaluation of frozen ranks, with explicit label and quota denominators."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_file(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def mask_id(mask):
    a = np.ascontiguousarray(mask, dtype=np.uint8)
    h = hashlib.sha256(np.asarray(a.shape,dtype="<i8").tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


def labels(prediction, reference):
    p=np.asarray(prediction)
    r=np.asarray(reference)
    if r.shape!=(252,378,3) or r.dtype!=np.float32:
        raise ValueError("reference must be 378 × 252 display-sRGB FP32")
    if p.shape!=r.shape or p.dtype!=np.float32:
        raise ValueError("different prediction and reference colour or grid")
    ref=np.isfinite(r).all(axis=2)
    valid=ref & np.isfinite(p).all(axis=2)
    difference=np.full(ref.shape,np.nan,np.float32)
    difference[valid]=np.mean(np.abs(r[valid]-p[valid]),axis=1,dtype=np.float32)
    high=valid & (difference>np.float32(0.05))
    return ref,valid,high,difference


def ap_and_base(scores, y):
    y=np.asarray(y,bool)
    n=len(y)
    pos=int(y.sum())
    if not n or not pos:
        return None, pos/n if n else None, "no_labelled_scores" if not n else "no_positive_labels"
    order=np.argsort(-scores,kind="stable")
    ranked=scores[order]
    ends=np.r_[np.nonzero(ranked[:-1]!=ranked[1:])[0]+1,n]
    cumulative=np.cumsum(y[order],dtype=np.int64)[ends-1]
    positives_at_tie=np.diff(np.r_[0,cumulative])
    return float(np.sum(positives_at_tie*(cumulative/ends))/pos),pos/n,None


def quota(risk, domain, high, q, *, highest=False):
    ids=np.flatnonzero(domain)
    k=(int(q)*len(ids))//100
    score=np.asarray(risk).ravel()[ids]
    order=np.lexsort((ids,-score if highest else score))
    accepted=ids[order[:k]]
    num=int(high.ravel()[accepted].sum())
    return dict(percent=int(q),direction="highest" if highest else "lowest",
        eligible=len(ids),selected=k,errors=num,error_rate=num/k if k else None,
        pixel_indices_sha256=hashlib.sha256(np.sort(accepted).astype("<i4").tobytes()).hexdigest())


def ranking_metrics(risk, ref, valid, high, *, domain=None):
    finite=np.isfinite(risk)
    if risk.shape!=valid.shape or not (valid.shape==ref.shape==high.shape):
        raise ValueError("ranking and reference masks have different grids")
    eligible=valid & finite
    if domain is not None:
        eligible &= domain
    count=int(eligible.sum())
    full=int(ref.sum())
    positives=int(high[eligible].sum())
    ap,base,reason=ap_and_base(risk[eligible],high[eligible])
    # One stable sort per direction. Row-major ids also define deterministic whole quota membership at ties.
    ids=np.flatnonzero(eligible)
    val=risk.ravel()[ids]
    low=ids[np.lexsort((ids,val))]
    higher=ids[np.lexsort((ids,-val))]
    def measured(indices,percent,which):
        k=percent*count//100
        accepted=indices[:k]
        n=int(high.ravel()[accepted].sum())
        return dict(percent=percent,direction=which,eligible=count,selected=k,errors=n,
            error_rate=n/k if k else None,share_full_reference=k/full if full else None,
            error_capture=n/positives if positives else None,
            pixel_indices_sha256=hashlib.sha256(np.sort(accepted).astype("<i4").tobytes()).hexdigest())
    prefix=np.r_[0,np.cumsum(high.ravel()[low],dtype=np.int64)]
    return dict(canvas=int(ref.size),reference=full,reference_errors=int(high.sum()),labelled=int(valid.sum()),
        denominator=count,finite_scores=int(finite.sum()),positive=positives,ap=ap,base=base,ap_undefined=reason,
        quota=[measured(arr,q,d) for q in (10,20) for arr,d in ((low,"lowest"),(higher,"highest"))],
        coverage_curve=[dict(percent=q,accepted=q*count//100,errors=int(prefix[q*count//100]))
                        for q in range(0,101,5)])


def opening_priority(score):
    values=score[np.isfinite(score)]
    if not len(values):
        return {"no_scores":True,"top_decile_mean":None,"unknown_fraction":1.,"scored":0}
    n=(len(values)+9)//10
    top=np.partition(values,len(values)-n)[-n:]
    return dict(no_scores=False,top_decile_mean=float(np.mean(top,dtype=np.float64)),
                unknown_fraction=1-len(values)/score.size,scored=len(values))


def load_reference(path):
    with Image.open(path) as im:
        if im.mode!="RGB" or im.size!=(378,252):
            raise ValueError("reference must be an RGB8 PNG on the registered 378 × 252 analysis grid")
        return np.asarray(im,dtype=np.float32)/np.float32(255)


def evaluate_pair(prediction_npz, reference_png, risk_npz, out):
    with np.load(prediction_npz,allow_pickle=False) as pred, np.load(risk_npz,allow_pickle=False) as scored:
        if "rgb" not in pred or not {"rank","finite"} <= set(scored.files):
            raise ValueError("expected a native target RGB array and a saved risk array")
        ref,valid,high,mae=labels(pred["rgb"],load_reference(reference_png))
        rank=scored["rank"]
        if rank.dtype!=np.float32 or not np.array_equal(np.isfinite(rank),scored["finite"]):
            raise ValueError("saved ranking and finite mask disagree")
        result=ranking_metrics(rank,ref,valid,high)
    result.update(schema="el-many-independent-evaluation-1",event="FP32 display-sRGB MAE > 0.05",
        prediction_sha256=sha256_file(prediction_npz),reference_sha256=sha256_file(reference_png),
        scores_sha256=sha256_file(risk_npz),label_hash=mask_id(high),valid_label_hash=mask_id(valid),
        mean_rgb_error=float(mae[valid].mean(dtype=np.float64)) if valid.any() else None)
    dest=Path(out)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    return result
