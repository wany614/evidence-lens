"""Local pinned file formats for authorized training inputs and label-free deployment."""
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

HASH = re.compile(r"[a-f0-9]{64}")
RUN = re.compile(r"\d{8}T\d{12}Z-[a-f0-9]{16}")
NUMERIC = ("target_alpha","source_alpha","c_behind","c_front","border_distance","relative_translation","view_angle",
           "local_depth_range","local_rgb_difference","source_rgb_difference","source_render_residual")
COLUMNS = [*NUMERIC,"frame_unprojectable","frame_inside","frame_outside","frame_behind_camera", *("missing_"+k for k in NUMERIC)]
BRANCHES = ("L2-frozen","L2-vision-ft")
SEEDS = (1729,2718,3141)
VIEWS = {f"v{n:02d}" for n in range(1,7)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def json_bytes(value):
    return (json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(",",":")) + "\n").encode()


def put(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_bytes(json_bytes(value))
    os.replace(tmp,path)


def output_dir(path):
    root=Path(path)
    root.mkdir(exist_ok=True,parents=True)
    require(not root.is_symlink() and not any(root.iterdir()),"output must be a new or empty directory")
    return root


def bound(root, entry):
    require(isinstance(entry,dict) and {"path","bytes","sha256"}<=entry.keys() and
            isinstance(entry["path"],str) and HASH.fullmatch(entry["sha256"]), "invalid file receipt")
    tokens=entry["path"].split("/")
    require(all(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*",p) for p in tokens),"nonportable relative data path")
    path=Path(root).joinpath(*tokens)
    require(path.resolve().is_relative_to(Path(root).resolve()) and path.is_file() and not path.is_symlink()
            and path.stat().st_size==entry["bytes"] and digest(path)==entry["sha256"],
            "data differs from selected bytes: "+entry["path"])
    return path


def file_record(path, root):
    return {"path":Path(path).relative_to(root).as_posix(),"bytes":Path(path).stat().st_size,"sha256":digest(path)}


class Dataset:
    def __init__(self, manifest, expected_split, expected_sha):
        self.path=Path(manifest)
        self.sha=digest(self.path)
        require(self.sha==expected_sha, "dataset manifest differs from the registered hash")
        doc=read(self.path)
        require(doc["schema"]=="el-vision-dataset-1" and doc["split"]==expected_split and doc["columns"]==COLUMNS
                and HASH.fullmatch(doc["feature_schema_digest"]),"wrong dataset, feature order or split")
        self.schema=doc["feature_schema_digest"]
        self.root=self.path.parent
        self.rows=doc["views"]
        require(isinstance(self.rows,list) and self.rows,"dataset has no declared views")
        self.families={}
        for row in self.rows:
            f,v=row["family"],row["view"]
            require(isinstance(f,str) and f and v in VIEWS and row["condition"]=="B0"
                    and type(row["height"]) is int and type(row["width"]) is int and min(row["height"],row["width"])>0,
                    "training and model selection accept original-grid B0 views only")
            members=self.families.setdefault(f,set())
            require(v not in members and row["status"] in ("completed","render_failed"),"duplicate or invalid study view")
            members.add(v)
            if row["status"]=="render_failed":
                require(row["rgb"] is None, "render failure may not substitute a prediction")
            if expected_split=="train":
                require(row.get("full") is None, "training manifest must isolate development and truth arrays")
        require(all(views==VIEWS for views in self.families.values()), "retain all six target rows in each family")
        self.verified=set()

    def file(self, entry):
        key=(entry["path"],entry["sha256"],entry["bytes"])
        if key not in self.verified:
            path=bound(self.root,entry)
            self.verified.add(key)
            return path
        return self.root/entry["path"]

    def rgb(self,row):
        rgb=np.load(self.file(row["rgb"]),allow_pickle=False)
        require(rgb.shape==(row["height"],row["width"],3) and rgb.dtype==np.float32,"wrong native RGB shape or dtype")
        return rgb

    def samples(self,row):
        with np.load(self.file(row["samples"]),allow_pickle=False) as z:
            ix,x,y=z["index"],z["X"],z["Y"]
        n=len(ix)
        require(ix.dtype==np.int32 and ix.shape==(n,) and x.shape==(n,len(COLUMNS)) and x.dtype==np.float32
                and y.dtype==np.uint8 and y.shape==(n,) and n==row["sample_count"]==min(4096,row["eligible"]) and (y<=1).all() and
                (not n or (0<=ix[0] and ix[-1]<row["height"]*row["width"] and np.all(ix[1:]>ix[:-1]))),
                "invalid registered sample rows")
        if row["status"]=="render_failed":
            require(not n, "render failure has fabricated samples")
        return ix,x,y


def preprocessing(path, *, train_sha=None, feature_sha=None):
    data=read(path)
    require(data["schema"]=="el-vision-preprocessing-1" and data["columns"]==COLUMNS and
            HASH.fullmatch(data["training_dataset_sha256"]) and HASH.fullmatch(data["feature_schema_digest"])
            and (not train_sha or train_sha==data["training_dataset_sha256"])
            and (not feature_sha or feature_sha==data["feature_schema_digest"]),"preprocessing is not bound to training and schema")
    for k in ("medians","means","scales"):
        a=np.asarray(data[k],dtype=np.float64)
        require(a.shape==(len(NUMERIC),) and np.all(np.isfinite(a)) and (k!="scales" or np.all(a>0)),
                "invalid train-only "+k)
    return data


def transform(raw, pre):
    require(raw.dtype==np.float32 and raw.ndim==2 and raw.shape[1]==len(COLUMNS),"unexpected feature matrix")
    nv=len(NUMERIC)
    values=raw[:,:nv].astype(np.float64)
    flags=raw[:,nv:]
    require(not np.any(np.isinf(values)) and np.all((flags==0)|(flags==1)) and np.all(flags[:,:4].sum(axis=1)==1)
            and np.array_equal(flags[:,4:],~np.isfinite(values)),"invalid numeric value or feature flag")
    z=(np.where(np.isfinite(values),values,pre["medians"])-pre["means"])/pre["scales"]
    out=np.concatenate((z,flags),axis=1).astype(np.float32)
    require(np.isfinite(out).all(),"nonfinite transformed feature")
    return out
