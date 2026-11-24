"""Audit explicitly pinned DTU acquisition runs and publish only counts, identities and hashes."""
import argparse
import csv
import hashlib
import json
import os
import sys
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from evidence_lens._verify import verify_bundle
from t0_dtu_archive import check, digest, write_json


def read(path):
    return json.loads(path.read_bytes())


def csv_rows(path,rows):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frozen_run(store, pin):
    folder=store/"runs"/pin["run_id"]
    evidence=verify_bundle(folder)
    check(evidence["manifest_sha256"] == pin["manifest_sha256"] and evidence["run"]["run_id"] == pin["run_id"], "wrong sealed stage")
    if "member" in pin:
        path=folder/pin["member"]
        check(evidence["run"]["execution_status"]=="succeeded" and
              any(r["path"]==pin["member"] and r["sha256"]==pin["sha256"] for r in evidence["manifest"]["files"])
              and digest(path.read_bytes())==pin["sha256"], "changed acquisition receipt")
    return folder,evidence


def geometry_data_inventory(root):
    rows=[]
    for parent,dirs,names in os.walk(root,followlinks=False):
        for name in sorted(dirs+names):
            p=Path(parent)/name
            st=p.lstat()
            rows.append({"relative_path":p.relative_to(root).as_posix(),"kind":"symlink" if p.is_symlink() else "directory" if p.is_dir() else "file",
                         "stat_bytes":st.st_size,"content_opened":False})
    return rows


def audit(store,config,out):
    roster_root,_=frozen_run(store,config["roster"])
    roster=read(roster_root/config["roster"]["member"])
    planned={o["object_id"]:o for o in roster["objects"]}
    role=config["role"]
    check(role in ("sources","complete"),"unexpected statistics phase")
    expected={f"{i}-{c}" for i in planned for c in ("M2","M4")}
    check(set(config["source_pins"])==expected,"source accounting missing a planned condition")
    refs=config.get("reference_pins",{})
    check((role=="complete" and set(refs)==set(planned)) or (role=="sources" and not refs),"wrong reference accounting cohort")
    frozen_run(store,config["freeze"])
    if role=="complete":frozen_run(store,config["opening"])
    stats=[]
    for m in config["metadata_runs"]:
        folder,proof=frozen_run(store,m)
        payload=folder/"payload"
        receipt=read(payload/"network.json") if (payload/"network.json").is_file() else {}
        status=proof["run"]["execution_status"]
        check(status==m["status"],"metadata attempt changed disposition")
        body=receipt.get("body_bytes",receipt.get("http_entity_bytes_read",0))
        if (payload/"http.jsonl").exists():
            audit_http=[json.loads(l) for l in (payload/"http.jsonl").read_text().splitlines()]
            check(body==sum(h["body_bytes"] for h in audit_http),"HTTP byte ledger differs")
        stats.append({"stage":m["stage"],"run_id":m["run_id"],"manifest_sha256":m["manifest_sha256"],"status":status,
                      "http_entity_bytes":body,"sealed_manifest_member_bytes":sum(r["bytes"] for r in proof["manifest"]["files"]),
                      "elapsed_s":proof["run"].get("elapsed_seconds_monotonic"),"notes":m["notes"]})
    data=[]; photos=[]; seen=set(); endpoints={}; acquisition_failures=[]
    for kind,group in (("sources",config["source_pins"]),("references",refs)):
        for key,item in group.items():
            folder,proof=frozen_run(store,item)
            meta=read(folder/item["member"]);net=read(folder/"payload/network.json")
            oid=meta["object_id"]; cond=meta.get("condition") or "reference"
            ids=roster["target_ids"] if kind=="references" else (["s00","s03"] if cond=="M2" else ["s00","s01","s02","s03"])
            check((key==oid if kind=="references" else key==oid+"-"+cond) and oid in planned and meta["roster_sha256"]==config["roster"]["sha256"]
                  and meta["freeze_sha256"]==config["freeze"]["sha256"] and meta["lighting_id"]==3, "wrong cohort or freeze")
            check(set(net["completed_views"])==set(ids) and net["role"]==kind, "wrong photograph roster")
            if kind=="references": check(meta["opening"]==config["opening"], "wrong reference authority")
            else:
                check(meta["reference_photo_reads"]==0 and meta["source_ids"]==ids and
                      digest((folder/"payload/source_cameras.json").read_bytes())==roster["camera"]["conditions"][cond]["source_cameras"]["sha256"]
                      and digest((folder/"payload/render_cameras.json").read_bytes())==roster["camera"]["conditions"][cond]["render_cameras"]["sha256"],"source aperture changed")
            stage=store/"work/controller"/("dtu-acquire-"+oid+"-"+(cond if kind=="sources" else "evaluation-reference"))
            check(read(stage/"selected.json")==item, "selected archive differs from the retained attempt history")
            attempts=[json.loads(line.read_text().splitlines()[-1]) for line in sorted(stage.glob("attempt-*.jsonl"))]
            check(attempts and attempts[-1]["run_id"]==item["run_id"] and attempts[-1]["execution_status"]=="succeeded",
                  "acquisition attempt history incomplete")
            for attempt in attempts[:-1]:
                failure_root,failure=frozen_run(store,attempt)
                check(failure["run"]["execution_status"]!="succeeded", "undisclosed replacement of a successful scientific acquisition")
                acquisition_failures.append({"object_id":oid,"condition":cond,"run_id":attempt["run_id"],"manifest_sha256":attempt["manifest_sha256"],
                                             "status":failure["run"]["execution_status"],"reason":failure["run"].get("failure_reason")})
            rows=meta["sources" if kind=="sources" else "views"]
            http=[json.loads(l) for l in (folder/"payload/http.jsonl").read_text().splitlines()]
            check(all(h["status"]==206 and h.get("range") and h["body_bytes"]==h["range"][1]-h["range"][0]+1 for h in http),"server did not honor exact Range")
            check(sum(h["body_bytes"] for h in http)==net["body_bytes"]<=net["approved_network_cap_bytes"],"wire bytes exceed recorded cap")
            raw_from_http=0; cached=0;png_bytes=0
            intervals={(h["range"][0],h["range"][1]+1) for h in http}
            allowed_intervals={(0,4)}
            for view in ids:
                row=rows[view]; original=planned[oid]["target_members" if kind=="references" else "source_members"][view]
                raw=(folder/"payload"/row["original_path"]).read_bytes()
                cropped=(folder/"payload"/row["rgb"]["path"]).read_bytes()
                check(row["archive_member"]==original["name"] and row["original_crc32"]==original["crc32"]==f"{zlib.crc32(raw):08x}"
                      and digest(raw)==row["original_sha256"] and len(raw)==original["bytes"]
                      and digest(cropped)==row["rgb"]["sha256"] and len(cropped)==row["rgb"]["bytes"],"official or derived image mismatch")
                png_bytes+=len(cropped)
                direct=row["acquisition"]=="official_exact_HTTP_range"
                if direct:
                    check((oid,view) not in seen, "unplanned duplicate photograph transfer")
                    seen.add((oid,view));raw_from_http+=len(raw)
                    start=original["offset"];head=(start,start+30)
                    matches=[p for p in intervals if p[0]==start+30]
                    check(head in intervals and len(matches)==1 and matches[0][1]>=start+30+len(original["name"].encode()),
                          "missing exact local header and local filename/extra")
                    data_pos=matches[0][1];body=(data_pos,data_pos+original["compressed"])
                    check(body in intervals,"wrong compressed-member range")
                    allowed_intervals.update((head,matches[0],body))
                else:
                    check(kind=="sources" and cond=="M4" and view in ("s00","s03"),"unauthorized input cache read")
                    cached+=len(raw)
                if kind=="sources" and view in ("s00","s03"):
                    endpoints[(oid,cond,view)]=(row["original_sha256"],row["rgb"]["sha256"],row["cropped_display_rgb8_pixel_sha256"])
                photos.append({"object_id":oid,"condition":cond,"view":view,"official_capture_id":row["official_view_id"],
                               "source_or_reference":kind,"original_crc32":row["original_crc32"],"original_bytes":len(raw),
                               "original_sha256":row["original_sha256"],"cropped_bytes":len(cropped),"cropped_sha256":row["rgb"]["sha256"],
                               "obtained_from":"HTTP range" if direct else "sealed M2 endpoint"})
            check(intervals==allowed_intervals and len(intervals)==len(http),"extra or repeated HTTP range beyond the approved members")
            check(net["members_uncompressed_from_http_bytes"]==raw_from_http and net["verified_local_original_endpoint_bytes"]==cached and
                  net["converted_png_bytes"]==png_bytes,"recorded payload and data bytes differ")
            data.append({"role":kind,"object_id":oid,"condition":cond,"views":len(ids),"http_requests":len(http),"http_body_bytes":net["body_bytes"],
                         "unpacked_from_http_bytes":raw_from_http,"cached_endpoint_bytes":cached,"cropped_png_bytes":png_bytes,
                         "run_id":item["run_id"],"manifest_sha256":item["manifest_sha256"],"member_sha256":item["sha256"],
                         "elapsed_s":proof["run"]["elapsed_seconds_monotonic"],"sealed_manifest_member_bytes":sum(r["bytes"] for r in proof["manifest"]["files"])})
    for oid in planned:
        for view in ("s00","s03"):
            check(endpoints[(oid,"M2",view)]==endpoints[(oid,"M4",view)],"shared endpoint changed")
    check(len(seen)==(10 if role=="complete" else 4)*len(planned),"wrong unique authorized photograph count")
    compact=[]
    for row in roster["inventory"]:
        compact.append({"scan_id":row["scan_id"],"physical_family_id":row["physical_family_id"] or "",
                        "official_paper_evaluation":int(row["official_evaluation_member"]),"ten_fixed_members_available":int(row["required_views_complete"]),
                        "dturotation_scan_ids":" ".join(map(str,row["declared_rotation_group"])),
                        "known_or_possible_shared_scan_ids":" ".join(map(str,row["documented_or_possible_shared_group"])),"decision":row["disposition"]})
    csv_rows(out/"physical-eligibility.csv",compact);csv_rows(out/"metadata-operations.csv",stats);csv_rows(out/"photograph-transfers.csv",data)
    csv_rows(out/"photo-hashes.csv",photos)
    aggregates={}
    for key in ("sources","references"):
        group=[r for r in data if r["role"]==key]
        aggregates[key]={k:sum(r[k] for r in group) for k in ("views","http_requests","http_body_bytes","unpacked_from_http_bytes","cached_endpoint_bytes","cropped_png_bytes","sealed_manifest_member_bytes","elapsed_s")}
        aggregates[key]["unique_photographs"]=len({(r["object_id"],r["view"]) for r in photos if r["source_or_reference"]==key})
    result={"schema":"el-t0-intake-audit-1","phase":role,"roster":config["roster"],"freeze":config["freeze"],"opening":config.get("opening"),
            "objects":[{"object_id":r["object_id"],"physical_family_id":r["physical_family_id"],"documented_subject":r["primary_subject"]} for r in planned.values()],
            "inventory":roster["inventory_counts"],"calibration_cross_archive_anchors":20,"archive_bytes":roster["archive"]["bytes"],
            "official_directories":roster["archive"]["entries"],"per_role":aggregates,"metadata_http_body_bytes":sum(r["http_entity_bytes"] for r in stats),
            "metadata_sealed_runs":len(stats),"metadata_failed_runs":sum(r["status"]!="succeeded" for r in stats),
            "successful_scientific_photo_acquisitions":len(data),"photo_acquisition_failure_runs":len(acquisition_failures),"photo_acquisition_failures":acquisition_failures,"network_accounting":"HTTP entity bytes read, exclusive of transport headers. HTTP discovery-engine results are not measured by the acquisition recorder.",
            "checksum":"All successful photograph requests checked for exact HTTP 206 ranges, strong ETag, original ZIP CRC and received SHA-256. The M4 common endpoints match M2 in original bytes, PNG bytes and decoded pixel SHA.",
            "bounds":"No transfer of an entire collection archive; same lab setup and incidental background objects can recur; new physical identity is documented for the named principal subject.",
            "outputs":{n:digest((out/n).read_bytes()) for n in ("physical-eligibility.csv","metadata-operations.csv","photograph-transfers.csv","photo-hashes.csv")}}
    if config.get("local_dtu_metadata_root"):
        rows=geometry_data_inventory(Path(config["local_dtu_metadata_root"]))
        write_json(out/"local-volume-metadata.json",{"scope":"stat only; no file contents opened","entries":rows})
        result["existing_data_volume_metadata"]={"directories":sum(r["kind"]=="directory" for r in rows),"regular_files":sum(r["kind"]=="file" for r in rows)}
    write_json(out/"intake-summary.json",result)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("store","config","out"):p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args();audit(a.store,read(a.config),a.out)
