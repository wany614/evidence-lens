"""Acquire only registered C6 photographs for one fixed condition or collectively opened targets."""
import argparse
import io
import json
import os
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from PIL import Image
from evidence_lens.dtu_camera import convert_rgb
from dtu_sample import Network
from t0_dtu_archive import ExactArchive, check, digest, write_json
from t0_dtu_convert import pin

END=("s00","s03")
SOURCES={"M2":END,"M4":("s00","s01","s02","s03")}
VIEWS=tuple(f"v{i:02}" for i in range(1,7))


def loads(path):
    return json.loads(path.read_bytes())


REFERENCE_CODE=("scripts/b2_intake.py","scripts/b2_intake_photos.py","scripts/t0_dtu_archive.py",
                "scripts/dtu_sample.py","scripts/t0_dtu_convert.py","src/evidence_lens/dtu_camera.py")


def verify_commitment(cohort,cohort_pin,method_pin,opening,candidates):
    check(candidates["schema"]=="el-b2-score-candidates-1" and candidates["run_id"]==opening["score_candidates"]["run_id"]
          and candidates["cohort"]==cohort_pin and candidates["calibration_method"]==method_pin
          and candidates["C6_reference_reads"]==candidates["T1_photo_reads"]==0,
          "candidates did not precede collective opening")
    thresholds=candidates["candidate_thresholds"]
    check(set(thresholds)==set(SOURCES) and
          all(set(rows)=={"full",*("held-out-"+oid for oid in cohort["splits"]["C6"])} and
              all(len(values)==201 for values in rows.values()) for rows in thresholds.values()),
          "incomplete full or held-out candidate grid")


def valid_opening(cohort,cohort_pin,method_pin,opening):
    ids=cohort["splits"]["C6"]
    check(opening["schema"]=="el-b2-c6-opening-1" and pin(opening["cohort"])==pin(cohort_pin) and
          pin(opening["calibration_method"])==pin(method_pin) and opening["object_ids"]==ids and
          opening["conditions"]==list(SOURCES) and opening["target_ids"]==list(VIEWS) and
          opening["predictions_and_risks_sealed"] is True,"wrong calibration target authority")
    pin(opening["score_candidates"])
    expected={(i,c) for i in ids for c in SOURCES}
    cells=opening["terminal_cells"]
    check({(r["object_id"],r["condition"]) for r in cells}==expected and len(cells)==len(expected) and
          opening["scheduled_entries"]==len(expected)*24 and 0<=opening["scored_entries"]<=opening["scheduled_entries"],
          "full C6 matrix is required before opening")
    for c in cells:
        if c["status"]=="scored":
            for key in ("case","readback","scores"):pin(c[key])
        else:
            check(c["status"] in ("failed","not_executed") and bool(c["reason"]),"incomplete terminal cell")
            pin(c["disposition"])


def run(args):
    cfg=loads(args.config);pins=cfg["pinned"]
    cohort=loads(args.cohort)
    check(cohort["schema"]=="el-b2-dtu-cohort-1" and cohort["run_id"]==pins["cohort"]["run_id"] and
          digest(args.cohort.read_bytes())==pins["cohort"]["sha256"],"wrong cohort source")
    oid=cfg["object_id"];cond=cfg.get("condition")
    check(oid in cohort["splits"]["C6"] and oid not in cohort["splits"]["T1"] and oid not in cohort["splits"]["reserve"],"C6 source/reference reader rejects T1 and reserve")
    subject,=[o for o in cohort["objects"] if o["object_id"]==oid and o["role"]=="C6"]
    source=cfg["role"]=="sources"
    if source:
        check(cond in SOURCES and args.opening is None,"source job cannot use target authority")
        ids=SOURCES[cond]
    else:
        check(cfg["role"]=="references" and not cond and args.opening is not None and args.method is not None,"separate reference/algorithm authority required")
        op=loads(args.opening);algorithm=loads(args.method)
        check(digest(args.opening.read_bytes())==pins["opening"]["sha256"] and op["run_id"]==pins["opening"]["run_id"]
              and digest(args.method.read_bytes())==pins["method"]["sha256"] and algorithm["schema"]=="el-b2-calibration-method-1" and
              algorithm["cohort"]==pins["cohort"] and algorithm["target_rgb_reads"]==0,"unverified method or opening")
        valid_opening(cohort,pins["cohort"],pins["method"],op)
        check(digest(args.candidates.read_bytes())==pins["candidates"]["sha256"] and op["score_candidates"]==pins["candidates"],
              "wrong sealed score-only evidence")
        verify_commitment(cohort,pins["cohort"],pins["method"],op,loads(args.candidates))
        code_root=Path(__file__).resolve().parents[1]
        check(all(digest((code_root/rel).read_bytes())==algorithm["scientific_files"][rel] for rel in REFERENCE_CODE),
              "reference conversion software differs from the algorithm lock")
        ids=VIEWS
    index=subject["source_members" if source else "target_members"]
    mapping=cohort["source_view_ids" if source else "target_view_ids"]
    for v in ids:
        check(index[v]["name"]==f"Rectified/scan{subject['scan_id']}/rect_{mapping[v]:03}_3_r5000.png","wrong physical photograph or light")
    crop=loads(args.crop)
    check(digest(args.crop.read_bytes())==cohort["camera"]["crop_sha256"],"crop changed")
    producer=None
    if source and cond=="M4":
        producer=loads(args.endpoint_provenance)
        check(producer["object_id"]==oid and producer["condition"]=="M2" and producer["cohort"]==pins["cohort"] and
              producer["run_id"]==pins["endpointProvenance"]["run_id"] and digest(args.endpoint_provenance.read_bytes())==pins["endpointProvenance"]["sha256"],
              "independent M2 endpoints do not match the physical subject")
    else:
        check(not (args.endpoint_provenance or args.endpoint0 or args.endpoint3),"unauthorized inputs")
    camera_hashes={}
    if source:
        for typ,p in (("source",args.source_camera),("render",args.render_camera)):
            raw=p.read_bytes()
            check(digest(raw)==cohort["camera"]["conditions"][cond][typ+"_cameras"]["sha256"] and
                  set(loads(p)["views"])==(set(ids) if typ=="source" else set(ids)|set(VIEWS)),"wrong approved native camera set")
            (args.out/(typ+"_cameras.json")).write_bytes(raw)
            camera_hashes[typ+"_camera_sha256"]=digest(raw)
    net=Network(cohort["budget"]["max_response_bytes_per_data_cell"],args.out/"http.jsonl")
    begin=time.monotonic();rows={};received=0;reused=0
    try:
        remote=ExactArchive(net,cohort["archive"]["url"],cohort["archive"])
        for v in ids:
            m=index[v];cached=producer is not None and v in END
            if cached:
                raw=(args.endpoint0 if v=="s00" else args.endpoint3).read_bytes()
                check(digest(raw)==producer["sources"][v]["original_sha256"],"cached endpoint source differs")
                reused+=len(raw)
            else:
                raw=remote.member(m);received+=len(raw)
            check(len(raw)==m["bytes"] and f"{zlib.crc32(raw):08x}"==m["crc32"],"wrong official member")
            png=convert_rgb(raw,crop)
            p=f"{'sources' if source else 'targets'}/{v}.png";original=f"original/{v}.png"
            for n,data in ((p,png),(original,raw)):
                f=args.out/n;f.parent.mkdir(parents=True,exist_ok=True)
                with f.open("xb") as h:h.write(data)
            with Image.open(io.BytesIO(png)) as im:pixels=digest(im.tobytes())
            rows[v]={"official_view_id":mapping[v],"original_sha256":digest(raw),"original_crc32":m["crc32"],"original_bytes":len(raw),
                     "archive_member":m["name"],"original_path":original,"rgb":{"path":p,"bytes":len(png),"sha256":digest(png)},
                     "cropped_display_rgb8_pixel_sha256":pixels,"acquisition":"sealed_M2_endpoint" if cached else "official_exact_HTTP_range"}
            if cached:check(rows[v]["rgb"]==producer["sources"][v]["rgb"],"M4 differs from shared M2 endpoints")
            if not source:rows[v].update(valid_mask={"kind":"all_true","width":256,"height":256},object_mask=None,
                                          object_mask_reason="No independently verified two-dimensional object mask for the registered full central crop.")
        shared={"run_id":os.environ["EL_RUN_ID"],"object_id":oid,"physical_family_id":subject["physical_family_id"],"cohort":pins["cohort"],
                "roster_sha256":pins["cohort"]["sha256"],"cohort_sha256":pins["cohort"]["sha256"],"scan_id":subject["scan_id"],"lighting_id":3,
                "archive":cohort["archive"],"camera_plan_sha256":cohort["camera"]["plan_sha256"],"crop_sha256":cohort["camera"]["crop_sha256"],
                "camera_sha256":cohort["camera"]["camera_sha256"],"view_set_sha256":cohort["camera"]["view_set_sha256"],
                "source_view_ids":cohort["source_view_ids"],"target_view_ids":cohort["target_view_ids"],"T1_photographs_opened":0}
        if source:write_json(args.out/"source-provenance.json",{**shared,**camera_hashes,"schema":"el-b2-dtu-source-1","condition":cond,
                        "source_ids":list(ids),"sources":rows,"reference_photo_reads":0,"opening":pins["cohort"]})
        else:write_json(args.out/"references.json",{**shared,"schema":"el-b2-dtu-references-1","views":rows,"opening":pins["opening"],
                        "calibration_method":pins["method"],"target_ids":list(VIEWS)})
    finally:
        write_json(args.out/"network.json",{"role":cfg["role"],"object_id":oid,"condition":cond,"http_bytes":net.total,"network_cap_bytes":net.budget,
                   "unpacked_from_HTTP_bytes":received,"verified_endpoint_cache_bytes":reused,"completed_views":list(rows),
                   "cropped_png_bytes":sum(r["rgb"]["bytes"] for r in rows.values()),"elapsed_seconds":time.monotonic()-begin})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("config","out","cohort","crop","opening","method","candidates","endpoint-provenance","endpoint0","endpoint3","source-camera","render-camera"):
        p.add_argument("--"+k,type=Path,required=k in ("config","out","cohort","crop"))
    run(p.parse_args())
