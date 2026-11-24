"""The two coverage policies share geometry and differ solely by risk weights."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from b2_nbv_select import choices, visible
from evidence_lens.scoring_io import Camera


def test_pair_union_and_value_independent_shared_support():
    pool=[7,10,13]
    points=np.zeros((5,3))
    weights=np.array([0.01,0.02,0.1,0.3,0.99],np.float32)
    vis={7:np.array([1,1,1,0,0],bool),10:np.array([1,1,0,1,0],bool),13:np.array([0,0,0,0,1],bool)}
    axes={7:np.array([1,0,0]),10:np.array([0,1,0]),13:np.array([0,0,1])}
    result,why,rows=choices(pool,points,weights,vis,axes,"set1",[137101,137102,137103])
    # All pairs accept four points here, so both coverage methods use common geometry.
    assert why=="constant_utilities" and result["risk"]==result["uniform"]==(7,10)
    vis[13]=np.array([0,0,1,0,1],bool)
    vis[10]=np.array([1,1,0,1,0],bool)
    result,why,rows=choices(pool,points,weights,vis,axes,"set1",[137101,137102,137103])
    assert why is None
    row=next(x for x in rows if x["pair"]==[7,10])
    assert row["uniform"]==4 and abs(row["risk"]-float(weights[:4].sum()))<1e-6
    assert result["risk"]==result["uniform"]==(10,13)
    assert len(rows)==3 and len(result)==5
    empty={k:np.zeros(0,bool) for k in pool}
    r,why,_=choices(pool,points[:0],weights[:0],empty,axes,"set1",[137101,137102,137103])
    assert why=="empty_support" and r["risk"]==r["uniform"]


def test_camera_z_uses_nearest_pixel_centres_and_rejects_occlusion():
    cam=Camera(2,2,np.eye(3),np.eye(4))
    pts=np.array([[0,0,1],[.6,0,1],[1.6,0,1],[0,0,5],[0,0,-1]],np.float64)
    ed=np.array([[1,3],[1,np.nan]],np.float32)
    rules={"absolute_depth_tolerance_in_baselines":.01,"relative_depth_tolerance":.05,"clip":[.05,30],
           "minimum_candidate_alpha":.1}
    np.testing.assert_array_equal(visible(pts,cam,{"z":ed,"alpha":np.ones((2,2))},1.,rules),
                                  [True,False,False,False,False])
