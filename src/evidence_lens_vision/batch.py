"""Score an explicit study dataset with markers for original render failures."""
from pathlib import Path
import numpy as np

from .contract import COLUMNS, HASH, SEEDS, VIEWS, bound, digest, file_record, output_dir, put, read, require
from .deploy import open_model, score


def dataset_scores(a):
    out=output_dir(a.output)
    model,pre,idx=open_model(a.package,a.model_sha256,a.device)
    branch=idx["branch"]
    method=None
    marker_sha=None
    if a.split=="test":
        from vision_evaluate import test_access, model_matches
        require(a.store and a.shelf and a.catalog and a.catalog_sha256 and a.method_freeze and a.method_sha256 and
                digest(a.method_freeze)==a.method_sha256,"study test scoring requires the canonical opening and exact freeze")
        marker_sha,method=test_access([a.store,a.shelf],[a.catalog,a.catalog_sha256])
        require(method==read(a.method_freeze) and method["models"][branch]["model_sha256"]==a.model_sha256
                and method["models"][branch]["run_id"]==idx["model_id"] and idx["seed"]==method["primary_seed"]==SEEDS[0]
                and method["thresholds"][branch]["domain"]=="development-B0" and method["thresholds"][branch]["tau"]==idx["threshold"],
                "test model or threshold differs from registered joint freeze")
        require(model_matches(method,branch,a.model_sha256),"freeze files omit the original model index")
    else:
        require(a.split=="development" and idx["development_dataset_sha256"]==a.dataset_sha256,
                "development run refers to a different selection dataset")
    require(digest(a.dataset)==a.dataset_sha256,"dataset differs from the explicit pin")
    ds=read(a.dataset)
    root=Path(a.dataset).parent
    require(ds["schema"]=="el-vision-dataset-1" and ds["columns"]==COLUMNS and ds["split"]==a.split and
            idx["phase"]=="train" and idx["threshold"] is not None and idx["feature_schema_digest"]==ds["feature_schema_digest"],
            "wrong split or model, threshold or feature schema")
    if method is not None:
        from vision_evaluate import guard_test
        require(ds["test_unlock_sha256"]==marker_sha and guard_test(ds,[a.store,a.shelf])==method,
                "dataset is not the registered eight-family test opening")
    rows=[]
    for view in ds["views"]:
        f,c,v=(view[key] for key in ("family","condition","view"))
        require(isinstance(f,str) and f.replace("_","").replace("-","").isalnum() and c in ("B0","R1","Q1") and v in VIEWS,
                "invalid stage identity")
        pair=view["rgb"],view["full"]["deploy"]
        record={"family":f,"condition":c,"view":v,"reason":view["reason"],"rgb_sha256":pair[0]["sha256"] if pair[0] else None,
                "deploy_sha256":pair[1]["sha256"] if pair[1] else None}
        if view["status"]=="render_failed":
            require(pair==(None,None) and record["reason"],"failed render has substituted arrays or lacks reason")
            record.update(status="render_failed",score=None)
        else:
            require(view["status"]=="completed" and all(pair),"unknown render status or missing inputs")
            rgb=np.load(bound(root,pair[0]),allow_pickle=False)
            require(rgb.shape==(view["height"],view["width"],3),"target RGB shape differs")
            with np.load(bound(root,pair[1]),allow_pickle=False) as z:
                require(set(z.files)=={"X","scoreable","alpha_score","l0"},"unknown deployment features")
                arrays,details=score(model,rgb,z["X"],z["scoreable"],pre,a.device,idx["threshold"])
            path=out/f/c/(v+".npz")
            path.parent.mkdir(parents=True,exist_ok=True)
            require(not path.exists(),"duplicate inference view")
            np.savez_compressed(path,**arrays)
            record.update(status="scored",reason=None,score=file_record(path,out),support=details["supported_pixels"])
        rows.append(record)
    require(len({(r["family"],r["condition"],r["view"]) for r in rows})==len(rows),"duplicate evaluated case")
    result={"schema":"el-vision-scores-1","method":branch,"model_sha256":a.model_sha256,"dataset_sha256":a.dataset_sha256,
            "method_freeze_sha256":a.method_sha256 if method else None,"views":rows}
    put(out/"scores.json",result)
    return result
