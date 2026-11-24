"""Select an empirical fixed-layout MS-L1 operating point using equal physical-object weights."""
from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from pathlib import Path

import numpy as np


ANCHORS=200
MAX_ERROR=Fraction(1,20)
MIN_COVERAGE=Fraction(1,5)


@dataclass
class ObjectData:
    object_id: str
    V: int
    E: int
    Y: int
    deployment: np.ndarray
    evaluable_scores: np.ndarray
    events_prefix: np.ndarray

    @classmethod
    def from_views(cls,object_id,views):
        deploy=[];score=[];events=[];V=E=Y=0
        for risk,v,e,y in views:
            if risk.dtype!=np.float32 or v.dtype!=bool or e.dtype!=bool or risk.shape!=v.shape or e.shape!=v.shape or y.shape!=v.shape:
                raise ValueError("wrong native supervision arrays")
            finite=np.isfinite(risk)
            if (e&~v).any() or not np.isin(y[e],[0,1]).all() or np.isinf(risk).any() or ((risk[finite]<0)|(risk[finite]>1)).any():
                raise ValueError("invalid reference/score domain")
            s=e&finite
            V+=int(v.sum());E+=int(e.sum());Y+=int((e&(y==1)).sum())
            deploy.append(risk[finite]);score.append(risk[s]);events.append((y[s]==1).astype(np.int64))
        if not views:raise ValueError("the planned views must be retained")
        all_scores=np.sort(np.concatenate(deploy).astype(np.float64),kind="stable")
        raw=np.concatenate(score).astype(np.float64)
        order=np.argsort(raw,kind="stable")
        prefix=np.r_[np.int64(0),np.cumsum(np.concatenate(events)[order],dtype=np.int64)]
        return cls(object_id,V,E,Y,all_scores,raw[order],prefix)

    def count(self,tau):
        if tau is None:
            return {"object_id":self.object_id,"V":self.V,"E":self.E,"Y":self.Y,"S":len(self.evaluable_scores),
                    "U_E":self.E-len(self.evaluable_scores),"V_minus_E":self.V-self.E,"L":None,"LY":None,"coverage":None,"low_error":None}
        l=int(np.searchsorted(self.evaluable_scores,np.float64(tau),side="left"))
        ly=int(self.events_prefix[l])
        return {"object_id":self.object_id,"V":self.V,"E":self.E,"Y":self.Y,"S":len(self.evaluable_scores),
                "U_E":self.E-len(self.evaluable_scores),"V_minus_E":self.V-self.E,"L":l,"LY":ly,
                "coverage":l/self.V if self.V else None,"low_error":ly/l if l else None}


def candidates(objects):
    groups=[o.deployment for o in objects if len(o.deployment)]
    if not groups:return [None]*(ANCHORS+1)
    union=np.unique(np.concatenate(groups))
    unit=lcm(*(len(g) for g in groups))
    weights=[unit//len(g) for g in groups]
    total=len(groups)*unit
    out=[float(union[0])]
    for j in range(1,ANCHORS+1):
        lo=0;hi=len(union)-1
        while lo<hi:
            mid=(lo+hi)//2
            mass=sum(w*int(np.searchsorted(g,union[mid],side="right")) for g,w in zip(groups,weights))
            if mass*ANCHORS>=j*total:hi=mid
            else:lo=mid+1
        out.append(float(np.nextafter(np.float64(union[lo]),np.float64(np.inf))))
    return out


def assessment(objects,tau):
    rows=[o.count(tau) for o in objects]
    coverage=None
    error=None
    if tau is not None and all(r["V"]>0 for r in rows):
        coverage=sum((Fraction(r["L"],r["V"]) for r in rows),Fraction())/len(rows)
    if tau is not None and all(r["L"] for r in rows):
        error=sum((Fraction(r["LY"],r["L"]) for r in rows),Fraction())/len(rows)
    reasons=[]
    if tau is None:reasons.append("no_finite_deployment_score")
    if any(r["V"]==0 for r in rows):reasons.append("object_without_valid_reference")
    if any(not r["L"] for r in rows):reasons.append("one_or_more_empty_acceptance_sets")
    if coverage is not None and coverage<MIN_COVERAGE:reasons.append("mean_coverage_below_20_percent")
    if error is not None and error>MAX_ERROR:reasons.append("mean_low_error_above_5_percent")
    row={"tau":tau,"mean_coverage":float(coverage) if coverage is not None else None,
         "mean_low_error":float(error) if error is not None else None,
         "minimum_object_coverage":min((r["coverage"] for r in rows if r["coverage"] is not None),default=None),
         "maximum_object_low_error":max((r["low_error"] for r in rows if r["low_error"] is not None),default=None),
         "nonempty_objects":sum(bool(r["L"]) for r in rows),"objects":len(rows),"feasible":not reasons,"reasons":reasons}
    return row,rows,(coverage,error)


def fit(objects):
    if not objects or len({g.object_id for g in objects})!=len(objects):raise ValueError("nonempty distinct families required")
    thresholds=candidates(objects)
    curves=[];counts=[];cache={};best_key=None;best=None
    for anchor,tau in enumerate(thresholds):
        if tau not in cache:cache[tau]=assessment(objects,tau)
        aggregate,per_object,exact=cache[tau]
        curves.append({"anchor":anchor,"quantile_numerator":anchor,"quantile_denominator":ANCHORS,**aggregate})
        counts.extend({"anchor":anchor,"tau":tau,**r} for r in per_object)
        if aggregate["feasible"]:
            coverage,error=exact
            key=(-coverage,error,tau,anchor)
            if best_key is None or key<best_key:best_key=key;best=curves[-1]
    return {"status":"empirical_calibration_only" if best else "no_feasible_operating_point",
            "tau":best["tau"] if best else None,"selected_anchor":best["anchor"] if best else None,
            "selected":best,"candidate_anchors":len(thresholds),"distinct_thresholds":len({t for t in thresholds if t is not None}),
            "participating_quantile_objects":sum(bool(len(o.deployment)) for o in objects),"curves":curves,"counts":counts}


def leave_one_out(objects):
    if len(objects)<2:raise ValueError("leave-one-out needs multiple objects")
    folds=[]
    for i,left in enumerate(objects):
        train=objects[:i]+objects[i+1:]
        result=fit(train)
        folds.append({"held_out":left.object_id,"training_ids":[g.object_id for g in train],"training":result,
                      "held_out_result":left.count(result["tau"]),"held_out_status":"evaluated" if result["tau"] is not None else "no_fitted_operating_point"})
    return folds
