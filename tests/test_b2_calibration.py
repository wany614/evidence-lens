from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from b2_calibration import ObjectData,assessment,candidates,fit,leave_one_out


def obj(name,values,errors,valid=None):
    r=np.asarray(values,np.float32)
    v=np.ones(r.shape,bool); e=v.copy() if valid is None else np.array(valid,bool)
    y=np.array(errors,np.uint8)
    return ObjectData.from_views(name,[(r,v,e,y)])


def test_exact_object_equal_quantiles_and_strict_ties():
    a=obj("many",[.9]*100,[0]*100)
    b=obj("one",[.1],[0])
    tau=candidates([a,b]);low=float(np.float32(.1))
    assert len(tau)==201 and tau[0]==low and tau[100]==np.nextafter(low,np.inf)
    assert b.count(tau[0])["L"]==0 and a.count(tau[100])["L"]==0 and b.count(tau[100])["L"]==1
    assert a.count(tau[101])["L"]==100
    assert candidates([obj("one",[1],[0])])[-1]>1


def test_nonmonotonic_feasibility_and_missing_score_denominator():
    values=[.1]+[.2]+[.3]*48+[np.nan]*10
    events=[0]+[1]+[0]*58
    x=obj("a",values,events)
    assert assessment([x],.25)[0]["mean_low_error"]==.5
    result=fit([x])
    assert result["status"]=="empirical_calibration_only" and result["selected"]["mean_coverage"]==50/60
    row=x.count(result["tau"]);assert row["LY"]==1 and row["U_E"]==10
    assert x.count(float(np.float32(.2)))["L"]==1
    masked=obj("b",[.001,.8],[255,1],valid=[False,True])
    assert candidates([masked])[0]==float(np.float32(.001))
    assert masked.count(1)["V_minus_E"]==1 and masked.count(1)["L"]==1


def test_leave_one_out_regenerates_candidates_and_infeasible_folds():
    objs=[obj(str(i),([.9] if i==5 else [.2]),[0]) for i in range(6)]
    folds=leave_one_out(objs)
    assert len(folds)==6
    final=folds[-1]
    assert final["training"]["tau"]==np.nextafter(float(np.float32(.2)),np.inf)
    assert final["held_out_result"]["L"]==0 and final["held_out_result"]["low_error"] is None
    bad=obj("bad",[np.nan],[1])
    assert fit([objs[0],bad])["status"]=="no_feasible_operating_point"
    badfolds=leave_one_out([objs[0],bad])
    assert badfolds[0]["held_out_status"]=="no_fitted_operating_point" and badfolds[0]["held_out_result"]["L"] is None


def test_strict_goal_boundaries_and_per_object_feasibility():
    low=[.1]*20+[np.nan]*80
    yes=obj("a",low,[1]+[0]*99); row=fit([yes])["selected"]
    assert row["mean_coverage"]==.2 and row["mean_low_error"]==.05
    bad=obj("b",[np.nan]*100,[0]*100)
    assert not assessment([yes,bad],1.0)[0]["feasible"]
