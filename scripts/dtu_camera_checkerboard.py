"""Independent physical camera/zero-origin check against the 49 paired DTU calibration plates."""
import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from evidence_lens.dtu_camera import decompose,digest,encoded,require,write


def measured(board,i):
    X=np.asarray(board[f"X_{i}"],np.float64)
    observed=np.asarray(board[f"x_{i}"],np.float64)
    rot=Rotation.from_rotvec(board[f"omc_{i}"]).as_matrix()
    t=board[f"Tc_{i}"]
    xyz=rot@X+t[:,None]
    require(np.all(xyz[2]>0),"Calibration target behind camera")
    x,y=xyz[:2]/xyz[2]
    r2=x*x+y*y
    k1,k2,p1,p2,k3=board["kc"]
    factor=1+k1*r2+k2*r2**2+k3*r2**3
    distorted=np.stack([x*factor+2*p1*x*y+p2*(r2+2*x*x),y*factor+p1*(r2+2*y*y)+2*p2*x*y])
    distorted[0]+=board["alpha_c"]*distorted[1]
    uv=board["fc"][:,None]*distorted+board["cc"][:,None]
    return rot,t,uv-observed


def main(a):
    sets={s:loadmat(a.calibration/f"Calib_Results_{s}.mat",squeeze_me=True) for s in ("left","right")}
    viewnames=__import__("json").loads((a.rig/"view-set.json").read_bytes())
    ids=viewnames["available_common_capture_ids"]
    r1,t1,_=measured(sets["left"],1)
    records={}
    residuals={"left":[],"right":[]}
    for i in ids:
        _,pose=decompose(np.loadtxt(a.calibration/f"pos_{i:03d}.txt"),.001,0)
        row={}
        for side,bd in sets.items():
            rot,t,errs=measured(bd,i);residuals[side].append(errs.ravel())
            centre=(r1@(-rot.T@t)+t1)*.001
            row[side]={"physical_pose_gap_to_P_m":float(np.linalg.norm(centre-pose[:3,3])),
                       "optical_rotation_gap_to_P_deg":float(np.rad2deg(Rotation.from_matrix(pose[:3,:3]@rot@r1.T).magnitude())),
                       "measured_corner_count":errs.shape[1],"native_raw_corner_coordinate_RMSE_px":float(np.sqrt(np.mean(errs**2)))}
        records[str(i)]=row
    stats={}
    for side in ("right","left"):
        d=np.concatenate(residuals[side])
        pose=np.asarray([v[side]["physical_pose_gap_to_P_m"] for v in records.values()])
        orientation=np.asarray([v[side]["optical_rotation_gap_to_P_deg"] for v in records.values()])
        stats[side]={"corners":int(d.size//2),"raw_native_rmse_px":float(np.sqrt(np.mean(d*d))),"if_shifted_one_px_rmse_px":float(np.sqrt(np.mean((d+1)**2))),
                     "max_abs_error_raw_native_px":float(np.max(np.abs(d))),"centre_gap_min_m":float(pose.min()),"centre_gap_max_m":float(pose.max()),
                     "centre_gap_median_m":float(np.median(pose)),"orientation_gap_max_deg":float(orientation.max()),"orientation_gap_min_deg":float(orientation.min())}
    require(stats["right"]["centre_gap_max_m"]<.010 and stats["left"]["centre_gap_min_m"]>.10 and stats["right"]["orientation_gap_max_deg"]<2 and
            stats["right"]["raw_native_rmse_px"]<.5 and stats["right"]["if_shifted_one_px_rmse_px"]>.8,"Incorrect physical camera, scale or original coordinate convention")
    write(a.out/"checkerboard.json",encoded({"schema":"el-dtu-original-camera-checkerboard-1","passed":True,"camera_plan_sha256":digest((a.rig/"camera-plan.json").read_bytes()),
         "source":"Original cal18 plate observations and paired Brown–Conrady calibrations. Express each plate-frame camera in first left camera coordinates; independently compare its centre/rotation to the published right-colour rectified pos_N. Raw predicted-versus-observed original corners are independent of the rectified projective algebra.",
         "objects_read":[],"capture_poses":len(ids),"summary":stats,"views":records}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("calibration","rig","out"):p.add_argument("--"+n,required=True,type=Path)
    main(p.parse_args())
