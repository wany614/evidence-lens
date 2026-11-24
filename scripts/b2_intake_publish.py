"""Project C6 cohort and intake receipts onto a public table of identifiers, counts and hashes."""
import argparse
import csv
import json
from pathlib import Path

from b2_intake_record import ROOT, locate, record_step, sealed_member
from t0_dtu_archive import check, digest, write_json


def read(path):
    return json.loads(path.read_bytes())


def emit_table(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def put_copy(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    if dest.exists():
        check(dest.read_bytes() == data, "refusing to replace an existing public export")
    else:
        with dest.open("xb") as stream:
            stream.write(data)


def selected(store, key):
    return read(store / "work/controller" / key / "selected.json")


def project(cfg, out):
    pins=cfg["pinned"]
    roots=[Path(r) for r in cfg["read_only_roots"]]
    evidence=lambda key: read(locate(roots,pins[key]))
    cohort=evidence("cohort")
    cal=evidence("cameras")
    plan=evidence("plan")
    check(cohort["allocation_plan"]==pins["plan"] and plan["seed"]==137031 and
          cohort["photo_bodies_read"]==0 and len(cohort["objects"])==16, "wrong allocation")
    eligibility=[]
    for row in sorted(cohort["objects"], key=lambda o:o["rank"]):
        eligibility.append({
            "rank":row["rank"],"role":row["role"],"object_id":row["object_id"],"scan_id":row["scan_id"],
            "physical_family_id":row["physical_family_id"],"documented_principal_subject":row["primary_subject"],
            "documented_or_possible_shared_ids":" ".join(map(str,row["documented_or_possible_shared_group"])),
            "registered_rotation_ids":" ".join(map(str,row["declared_rotation_group"])),
            "priority_sha256":row["priority_sha256"],"eligible":row["eligible"],
            "number_of_verified_captures":len(row["members"])-len(row["missing_or_incompatible_capture_ids"]),
        })
    emit_table(out/"physical-families.csv",eligibility)
    geometry=[{"official_view_id":r["official_view_id"],"calibration_P_sha256":r["projection_sha256"],
               "maximum_native_to_crop_residual_px":r["max_reprojection_px"]}
              for r in sorted(cal["cameras"].values(),key=lambda r:r["official_view_id"])]
    emit_table(out/"rig-checks.csv",geometry)
    network=[];photos=[];unique=set()
    for obj in cohort["splits"]["C6"]:
        for cond in (("M2","M4","references") if cfg["include_references"] else ("M2","M4")):
            tag=obj+"-"+cond
            result=evidence(tag)
            stats=evidence(tag+"net")
            reqs=[json.loads(line) for line in locate(roots,pins[tag+"http"]).read_text().splitlines()]
            check(stats["http_bytes"]==sum(r["body_bytes"] for r in reqs), "transfer ledger differs")
            records=result["views"] if cond=="references" else result["sources"]
            network.append({"object_id":obj,"role":cond,"run_id":pins[tag]["run_id"],
                            "manifest_sha256":pins[tag]["manifest_sha256"],"member_sha256":pins[tag]["sha256"],
                            "http_requests":len(reqs),"http_body_bytes":stats["http_bytes"],
                            "decompressed_from_http_bytes":stats["unpacked_from_HTTP_bytes"],
                            "verified_endpoint_cache_bytes":stats["verified_endpoint_cache_bytes"],
                            "output_crop_png_bytes":stats["cropped_png_bytes"],
                            "elapsed_seconds":stats["elapsed_seconds"],"completed_members":len(records)})
            for v,row in records.items():
                key=(obj,row["official_view_id"])
                check((key in unique)==(row["acquisition"]=="sealed_M2_endpoint"),
                      "duplicate identity was not the reused common endpoint")
                unique.add(key)
                photos.append({"object_id":obj,"role":cond,"logical_view":v,"official_view_id":row["official_view_id"],
                               "official_archive_member":row["archive_member"],"acquisition":row["acquisition"],
                               "original_bytes":row["original_bytes"],"original_crc32":row["original_crc32"],
                               "original_sha256":row["original_sha256"],"crop_bytes":row["rgb"]["bytes"],
                               "crop_sha256":row["rgb"]["sha256"],"crop_rgb8_pixels_sha256":row["cropped_display_rgb8_pixel_sha256"]})
    emit_table(out/"transfers.csv",network)
    emit_table(out/"photograph-hashes.csv",photos)
    summary={"schema":"el-b2-public-cohort-1","allocation_plan":pins["plan"],"cohort":pins["cohort"],
             "physical_identity_register":cohort["upstream_roster"],"camera_eligibility":pins["cameras"],
             "allocation_seed":plan["seed"],"arm_order":cohort["splits"],"eligible_families":len(eligibility),
             "registered_captures_per_family":len(geometry),"lights":[3],
             "candidate_capture_ids":cohort["candidate_cameras"],"fixed_sources":cohort["source_view_ids"],
             "evaluation_views":cohort["target_view_ids"],"camera":cohort["camera"],
             "max_reprojection_error_px":cal["max_reprojection_px"],"tolerance_px":cal["reprojection_tolerance_px"],
             "T1_photograph_reads":0,"unique_C6_photographs":len(unique),"included_reference_photographs":cfg["include_references"],
             "totals":{k:sum(r[k] for r in network) for k in ("http_requests","http_body_bytes","decompressed_from_http_bytes",
                       "verified_endpoint_cache_bytes","output_crop_png_bytes","completed_members")},
             "source_files":{name:digest((out/name).read_bytes()) for name in
                             ("physical-families.csv","rig-checks.csv","transfers.csv","photograph-hashes.csv")}}
    write_json(out/"cohort-summary.json",summary)


def run(store,batch_one,history,include):
    roots=(store,batch_one,history)
    cohort=selected(store,"b2-dtu-cohort")
    document=read(locate(roots,cohort))
    pins={"cohort":cohort,"plan":document["allocation_plan"],
          "cameras":sealed_member(roots,cohort["run_id"],"payload/camera-eligibility.json")}
    for o in document["splits"]["C6"]:
        for c in (("M2","M4","references") if include else ("M2","M4")):
            tag=o+"-"+c
            base="b2-C6-reference-"+o if c=="references" else "b2-C6-source-"+tag
            pin=selected(store,base)
            pins[tag]=pin
            pins[tag+"net"]=sealed_member(roots,pin["run_id"],"payload/network.json")
            pins[tag+"http"]=sealed_member(roots,pin["run_id"],"payload/http.jsonl")
    return record_step(roots,"b2-public-complete-intake" if include else "b2-public-source-intake",
                       "b2_intake_publish.py","payload/cohort-summary.json",
                       {"read_only_roots":list(map(str,roots)),"include_references":include},pins,worker_args=("worker",),
                       extra_code=["scripts/b2_intake_record.py","src/evidence_lens/_bundle.py","src/evidence_lens/_verify.py"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("sources","complete","worker"))
    for k in ("store","batch-one","history","out","config","dest"):p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.action=="worker":
        project(read(a.config),a.out)
    else:
        roots=(a.store.resolve(),a.batch_one.resolve(),(a.history or ROOT/".research").resolve())
        pin=run(*roots,a.action=="complete")
        if a.dest:
            pf=locate(roots,pin)
            meta=read(pf)
            for name in [*meta["source_files"],"cohort-summary.json"]:
                put_copy(pf.parent/name,a.dest/name)
        print(json.dumps(pin))
