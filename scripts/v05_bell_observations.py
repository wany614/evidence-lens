"""Publish the actual authored originals and independently rebuild the 24 source PNGs of the six bundles."""

import argparse
import io
import json
import os
from pathlib import Path
import zipfile

from PIL import Image

from dual_records import bind, resolve, selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens_bundle.common import encoded, require
from evidence_lens_bundle.common import sha as digest
from evidence_lens_bundle.grids import resample_photo
from v05_bell_common import OLD,SOURCES


def control(s):
    roots=s,s/"history"
    bundles=selected(s,"a-bell-bundle-publication");photos=selected(s,"a-bell-source-observations")
    source=load(resolve(roots,photos));book=load(resolve(roots,bundles));pins={"photos":photos,"bundles":bundles}
    for tag,group in source["products"].items():
        for v,record in group.items():
            pins[tag+"-"+v]=bind(s,photos["run_id"],"payload/"+record["rgb8"]["path"])
    for tag,row in book["routes"].items():
        pins[tag]=bind(s,bundles["run_id"],"payload/"+row["path"])
        pins["evidence-"+tag]=bind(s,bundles["run_id"],"payload/bundles/"+tag+"/source-observations.json")
    cmd=["{python}","-B","{source}/scripts/v05_bell_observations.py","worker","--out","{out}","--config","{config}"]
    for n in pins:cmd += ["--input",n,"{input:"+n+"}"]
    return stage(s,"a-bell-source-delivery","dataset_build",pins,{"license":"CC-BY-4.0","originals":20,"derived":24},cmd,"payload/observations.json",15_000_000)


def worker(cfg,inputs,out):
    src,book=load(inputs["photos"]),load(inputs["bundles"])
    require(src["role"]=="sources" and src["target_reference_renders"]==0 and book["source_observations"]==cfg["sources"]["photos"],"wrong source-only study")
    assets={"photos.json":inputs["photos"].read_bytes()};rows=[]
    for tag,group in src["products"].items():
        for v,record in group.items():
            f=record["rgb8"];raw=inputs[tag+"-"+v].read_bytes()
            require(digest(raw)==f["sha256"] and len(raw)==f["bytes"],"wrong original source photo")
            assets[f["path"]]=raw
    for tag,route in book["routes"].items():
        b=load(inputs[tag]);evidence=load(inputs["evidence-"+tag]);kind="landscape" if tag==OLD else tag
        require(sha(inputs[tag])==route["sha256"] and sha(inputs["evidence-"+tag])==route["source_observation_sha256"],"changed bundle input")
        for v in SOURCES:
            original=src["products"][kind][v]["rgb8"]
            with Image.open(io.BytesIO(assets[original["path"]])) as im:
                if tag==OLD:
                    sampled=Image.new("RGB",(256,256));sampled.paste(im.resize((256,144),Image.Resampling.LANCZOS),(0,56))
                else:
                    grid=b["grids"];sampled=resample_photo(im,grid["views"][v],grid["analysis_size"])
            buf=io.BytesIO();sampled.save(buf,format="PNG");h=digest(buf.getvalue())
            lineage=evidence["original_observations"][v]
            require(h==b["sources"][v]["sha256"]==lineage["analysis_png_sha256"] and original["sha256"]==lineage["original_rgb8_sha256"],"source lineage mismatch")
            rows.append({"view":v,"route":tag,"source":original["path"],"original_sha256":original["sha256"],"analysis_sha256":h})
    require(len(assets)==21 and len(rows)==24,"wrong source counts")
    index={"schema":"el-v05-authored-observations-1","run_id":os.environ["EL_RUN_ID"],"original_count":20,"reconstructed_analysis_pngs":24,
           "mismatches":0,"target_reference_photos":0,"bundles":cfg["sources"]["bundles"],"photos":cfg["sources"]["photos"],"rows":rows,
           "license":"CC BY 4.0 — wany614, Evidence Lens; https://creativecommons.org/licenses/by/4.0/"}
    assets["index.json"]=encoded(index)
    archive=out/"v05-bell-source-observations.zip"
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,raw in sorted(assets.items()):z.writestr(name,raw)
    save(out/"observations.json",{**index,"archive":{"path":archive.name,"bytes":archive.stat().st_size,"sha256":sha(archive)}})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("publish","worker"))
    for k in ("config","out","store"):p.add_argument("--"+k,type=Path)
    p.add_argument("--input",action="append",nargs=2,default=[]);a=p.parse_args()
    if a.action=="worker":worker(load(a.config),{k:Path(v) for k,v in a.input},a.out)
    else:print(json.dumps(control(a.store.resolve())))
