"""Measure complete CPU inference from the condition's original, truth-free source and prediction."""
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .contract import COLUMNS, digest, file_record, output_dir, put, require


@contextmanager
def peak_tree():
    import psutil
    current=psutil.Process()
    stop=threading.Event()
    peak=[0]
    def sample():
        total=0
        for p in (current,*current.children(recursive=True)):
            try:
                total+=p.memory_info().rss
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                continue
        peak[0]=max(total,peak[0])
    def poll():
        while not stop.is_set():
            sample()
            stop.wait(.01)
    task=threading.Thread(target=poll,daemon=True)
    sample()
    task.start()
    try:
        yield peak
    finally:
        stop.set()
        task.join()
        sample()


def one(model,pre,index,source_path,camera_path,prediction_dir,view,out):
    from evidence_lens.features import extract_features
    from evidence_lens.l1 import ALL_COLUMNS, schema_digest
    from evidence_lens.scoring_io import PredictionBundle, ReadAudit, load_cameras, load_source_png
    from .deploy import score
    start=time.perf_counter()
    audit=ReadAudit()
    cams=load_cameras(camera_path,audit)
    image=load_source_png(source_path,cams["source"],audit)
    predictions=PredictionBundle(prediction_dir,audit)
    src=predictions.read("source",cams["source"])
    target=predictions.read(view,cams[view])
    loaded=time.perf_counter()
    require(ALL_COLUMNS==COLUMNS and schema_digest()==index["feature_schema_digest"],"native feature contract differs from model")
    shape=cams[view].shape
    if target is None or not predictions.depth_verified:
        arrays={"risk":np.full(shape,np.nan,np.float32),"valid":np.zeros(shape,np.bool_)}
        if index["threshold"] is not None:
            arrays["state"]=np.full(shape,255,np.uint8)
        reason=predictions.failed.get(view) if target is None else "producer depth validation failed"
        details={"reason":reason,"status":"unsupported","supported_pixels":0}
        feature_elapsed=0.
    else:
        features=extract_features(image,src,target,cams["source"],cams[view],predictions.depth_verified)
        raw=np.stack([features[k] for k in COLUMNS],axis=-1).astype(np.float32,copy=False)
        feature_elapsed=time.perf_counter()-loaded
        arrays,details=score(model,target["rgb"],raw,features["scoreable"],pre,"cpu",index["threshold"])
        details.update(status="scored",reason=None)
    np.savez_compressed(out,**arrays)
    return {"view":view,"height":shape[0],"width":shape[1],"inputs":audit.files,"producer_run_id":predictions.meta["producer_run_id"],
            "read_seconds":loaded-start,"camera_projection_and_legal_features_seconds":feature_elapsed,
            "wall_through_npz_seconds":time.perf_counter()-start,"output_sha256":digest(out),**details}


def native(args, boot):
    import platform
    import torch
    from .deploy import open_model
    from .contract import VIEWS
    out=output_dir(args.output)
    runs=[]
    views=sorted(VIEWS) if args.view=="all" else [args.view]
    require(set(views)<=VIEWS and 0<=args.warm_repeats<=20,"invalid native view or warm repeats count")
    with peak_tree() as peak:
        model,pre,idx=open_model(args.package,args.model_sha256,"cpu")
        loaded=time.perf_counter()
        for rep in range(args.warm_repeats+1):
            dest=out/("cold" if rep==0 else "warm-"+str(rep))
            dest.mkdir()
            for view in views:
                result=one(model,pre,idx,args.input,args.cameras,args.predictions,view,dest/(view+".npz"))
                result["pass"]=rep
                if not runs:
                    result["fresh_process_to_first_output_seconds"]=time.perf_counter()-boot
                runs.append(result)
    warm=[r["wall_through_npz_seconds"] for r in runs if r["pass"]>0]
    report={"schema":"el-vision-native-cost-1","model_id":idx["model_id"],"model_sha256":digest(Path(args.package)/"model.json"),
            "method":idx["branch"],"device":"cpu","boot_to_model_loaded_seconds":loaded-boot,
            "platform":platform.platform(),"architecture":platform.machine(),"python":platform.python_version(),
            "torch":str(torch.__version__),"numpy":np.__version__,"torch_cpu_threads":torch.get_num_threads(),
            "torch_interop_threads":torch.get_num_interop_threads(),
            "cold_process_to_first_output_seconds":runs[0]["fresh_process_to_first_output_seconds"],"views":runs,
            "warm_views":len(warm),"hot_p50_seconds":float(np.percentile(warm,50)) if warm else None,
            "hot_p95_seconds":float(np.percentile(warm,95)) if warm else None,"process_tree_peak_rss_bytes":peak[0],
            "rss_interval_seconds":0.01,"scope":"source, cameras, prediction decoding, projection, features, complete backbone, head, NPZ",
            "all_measured_views_512":all((r["height"],r["width"])==(512,512) for r in runs),
            "all_measured_views_scored":all(r["status"]=="scored" and r["supported_pixels"]>0 for r in runs)}
    put(out/"native-cost.json",report)
    return report


def standalone(package,boot):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("input","cameras","predictions","output"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--model-sha256",required=True)
    p.add_argument("--view",default="all",choices=["all",*[f"v{i:02d}" for i in range(1,7)]])
    p.add_argument("--warm-repeats",type=int,default=0)
    a=p.parse_args()
    a.package=package
    native(a,boot)
