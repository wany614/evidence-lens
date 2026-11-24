"""Fetch and crop only the DTU source pair/quartet or the collectively opened evaluation targets."""
import argparse
import io
import json
import os
import re
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image, __version__ as pillow_version
from evidence_lens.dtu_camera import convert_rgb
from dtu_sample import Network
from t0_dtu_archive import ExactArchive, check, digest, write_json

TARGETS = tuple(f"v{i:02}" for i in range(1, 7))
ALL_SOURCES = ("s00", "s01", "s02", "s03")
ENDPOINTS = ("s00", "s03")
MEMBERS = {"M2": ENDPOINTS, "M4": ALL_SOURCES}
RECIPE = "57730c706aaaa2f88c7f39ff179ba04ff971f31e599bf3409a3d545c61d72c98"
METHODS = {"B-L1-transfer", "MS-L1", "MS-frozen", "MS-ft"}
PIN_FIELDS = ("run_id", "manifest_sha256", "member", "sha256")


def read_checked(path, sha):
    data = path.read_bytes()
    check(digest(data) == sha, "recorded input bytes differ")
    return json.loads(data)


def pin(value):
    check(isinstance(value, dict) and all(isinstance(value.get(k),str) for k in PIN_FIELDS), "missing strong run/member pin")
    check(bool(re.fullmatch(r"[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}", value["run_id"]))
          and all(re.fullmatch(r"[0-9a-f]{64}",value[k]) for k in ("sha256","manifest_sha256"))
          and value["member"].startswith("payload/") and ".." not in Path(value["member"]).parts, "bad acquisition pin")
    return {k:value[k] for k in PIN_FIELDS}


REFERENCE_SOFTWARE = ("scripts/t0_dtu_convert.py", "scripts/t0_dtu_archive.py",
                      "scripts/dtu_sample.py", "src/evidence_lens/dtu_camera.py")


def verify_reference_software(lock, freeze_pin, project):
    check(lock["schema"] == "el-t0-opening-software-1" and pin(lock["sources"]["freeze"]) == pin(freeze_pin),
          "reference software lock belongs to a different freeze")
    recorded = lock["files"]
    for name in REFERENCE_SOFTWARE:
        expected = recorded.get(name)
        check(isinstance(expected,str) and re.fullmatch("[0-9a-f]{64}",expected)
              and digest((project / name).read_bytes()) == expected, "unlocked reference implementation: " + name)


def authorise(roster, config, freeze, opening):
    roster_pin, freeze_pin = pin(config["roster"]), pin(config["freeze"])
    check(roster_pin["member"] == "payload/roster.json" and freeze_pin["member"] == "payload/freeze.json" and
          freeze["schema"] == "el-t0-freeze-1" and freeze["run_id"] == freeze_pin["run_id"] and
          roster["schema"] == "el-t0-dtu-roster-1" and pin(freeze["sources"]["roster"]) == roster_pin and
          freeze["roster_sha256"] == roster_pin["sha256"] and freeze["recipe_sha256"] == RECIPE and
          freeze["stop_step"] == 5000 and set(freeze["models"]) == METHODS, "wrong freeze, methods or object registry")
    objects = [r["object_id"] for r in roster["objects"]]
    check(0 < len(objects) <= 6 and len(objects) == len(set(objects)) and
          [r["scan_id"] for r in roster["objects"]] == sorted({r["scan_id"] for r in roster["objects"]}) and
          objects == freeze["object_ids"] == roster["object_ids"] and config["object_id"] in objects and
          freeze["conditions"] == roster["conditions"] == list(MEMBERS) and freeze["target_ids"] == list(TARGETS) and
          freeze["target_reference_permission"] == "closed" and roster["lighting_id"] == 3, "wrong roster scope")
    obj, = [r for r in roster["objects"] if r["object_id"] == config["object_id"]]
    check(obj["object_id"] == "set" + str(obj["scan_id"]) and obj["scan_id"] not in (1,6) and
          len({r["physical_family_id"] for r in roster["objects"]}) == len(objects), "wrong physical identity")
    if config["role"] == "sources":
        check(config["condition"] in MEMBERS and opening is None and config.get("opening") is None, "source job requires only its freeze")
        return obj, MEMBERS[config["condition"]]
    check(config["role"] == "references" and config["condition"] is None and opening is not None, "unauthorized photograph role")
    opening_pin = pin(config["opening"])
    check(pin(opening["software_lock"]) == pin(config["software_lock"]), "reference software pin differs from common opening")
    check(opening_pin["member"] == "payload/opening.json" and opening["schema"] == "el-t0-reference-opening-1" and
          opening["run_id"] == opening_pin["run_id"] and pin(opening["freeze"]) == freeze_pin and
          pin(opening["roster"]) == roster_pin and opening["roster_sha256"] == roster_pin["sha256"] and
          opening["object_ids"] == objects and opening["conditions"] == list(MEMBERS) and opening["target_ids"] == list(TARGETS)
          and opening["permitted_reader"] == "independent_evaluator" and opening["predictions_and_risks_sealed"] is True,
          "missing common independent reference opening")
    cells = opening["terminal_cells"]
    expected = {(o, c) for o in objects for c in MEMBERS}
    check(len(cells) == len(expected) and {(c["object_id"],c["condition"]) for c in cells} == expected and
          opening["scheduled_entries"] == freeze["expected_scores"] == len(expected) * 6 * 4 and
          0 <= opening["scored_entries"] <= opening["scheduled_entries"] and bool(re.fullmatch(r"[0-9a-f]{64}", opening["sealed_matrix_sha256"])),
          "incomplete frozen prediction and risk matrix")
    for cell in cells:
        if cell["status"] == "scored":
            for key in ("case", "readback", "scores"):
                pin(cell[key])
        else:
            check(cell["status"] in ("failed","not_executed") and cell["reason"], "missing terminal reason")
            pin(cell["disposition"])
    return obj, TARGETS


def put(path, raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f:
        f.write(raw)


def convert(a):
    config = json.loads(a.config.read_bytes())
    roster = read_checked(a.roster, config["roster"]["sha256"])
    fr = read_checked(a.freeze, config["freeze"]["sha256"])
    opening = read_checked(a.opening, config["opening"]["sha256"]) if a.opening is not None else None
    obj, ids = authorise(roster,config,fr,opening)
    role, cond = config["role"], config["condition"]
    if role == "references":
        check(a.software_lock is not None, "missing original reference-software receipt")
        lock = read_checked(a.software_lock,pin(config["software_lock"])["sha256"])
        verify_reference_software(lock,config["freeze"],Path(__file__).resolve().parents[1])
    rc = roster["camera"]
    crop = read_checked(a.crop, rc["crop_sha256"])
    check(crop["id"] == "dtu-crop256-v1" and (crop["width"],crop["height"],crop["output"]) == (1600,1200,[256,256]),
          "wrong registered crop")
    endpoint = None
    views = {}
    received = []
    clock = time.monotonic()
    condition_hashes = {}
    if role == "sources":
        check(a.source_cameras and a.render_cameras, "condition calibrations required")
        for key, path in (("source_cameras",a.source_cameras),("render_cameras",a.render_cameras)):
            expected = rc["conditions"][cond][key]
            raw = path.read_bytes()
            check(digest(raw) == expected["sha256"] and len(raw) == expected["bytes"], "wrong condition calibration bytes")
            expected_ids = set(ids) if key=="source_cameras" else set(ids) | set(TARGETS)
            check(set(json.loads(raw)["views"]) == expected_ids, "wrong camera roles")
            put(a.out / (key + ".json"),raw)
            condition_hashes["source_camera_sha256" if key == "source_cameras" else "render_camera_sha256"] = digest(raw)
    check(bool(a.endpoint_provenance) == (role == "sources" and cond == "M4") and
          bool(a.endpoint0) == bool(a.endpoint3) == bool(a.endpoint_provenance), "unexpected or unbound cached endpoints")
    if a.endpoint_provenance is not None:
        endpoint = read_checked(a.endpoint_provenance, pin(config["endpoints"])["sha256"])
        check(endpoint["schema"] == "el-t0-dtu-source-1" and endpoint["run_id"] == config["endpoints"]["run_id"] and
              endpoint["condition"] == "M2" and endpoint["object_id"] == obj["object_id"] and
              endpoint["roster_sha256"] == config["roster"]["sha256"] and endpoint["freeze_sha256"] == config["freeze"]["sha256"]
              and set(endpoint["sources"]) == set(ENDPOINTS), "cached endpoints belong to a different trial")
    raw_index = obj["source_members" if role == "sources" else "target_members"]
    official = roster["source_view_ids" if role == "sources" else "target_view_ids"]
    for view in ids:
        row = raw_index[view]
        check(row["name"] == f"Rectified/scan{obj['scan_id']}/rect_{official[view]:03}_3_r5000.png", "wrong role in frozen ZIP metadata")
    max_bytes = roster["budget"]["source_http_cap_per_condition_bytes" if role=="sources" else "reference_http_cap_per_object_bytes"]
    net = Network(max_bytes, a.out / "http.jsonl")
    fetched_bytes = cached_bytes = 0
    try:
        archive = ExactArchive(net,roster["archive"]["url"],roster["archive"])
        for view in ids:
            row = raw_index[view]
            cached = endpoint is not None and view in ENDPOINTS
            if cached:
                raw = (a.endpoint0 if view=="s00" else a.endpoint3).read_bytes()
                prior = endpoint["sources"][view]
                check(digest(raw) == prior["original_sha256"] and prior["archive_member"] == row["name"], "cached endpoint identity mismatch")
                cached_bytes += len(raw)
            else:
                raw = archive.member(row)
                fetched_bytes += len(raw)
            check(len(raw) == row["bytes"] and f"{zlib.crc32(raw):08x}" == row["crc32"] and raw.startswith(b"\x89PNG\r\n\x1a\n"), "official original is not the registered PNG")
            png = convert_rgb(raw,crop)
            with Image.open(io.BytesIO(png)) as image:
                check(image.mode == "RGB" and image.size == (256,256), "wrong native output")
                pixel_sha = digest(image.tobytes())
            filename = f"{'sources' if role=='sources' else 'targets'}/{view}.png"
            put(a.out / filename,png)
            put(a.out / "original" / (view+".png"),raw)
            entry = {"official_view_id":official[view],"archive_member":row["name"],"original_crc32":row["crc32"],"original_sha256":digest(raw),
                     "original_bytes":len(raw),"original_path":f"original/{view}.png","official_compressed_bytes":row["compressed"],
                     "rgb":{"path":filename,"sha256":digest(png),"bytes":len(png)},"cropped_display_rgb8_pixel_sha256":pixel_sha,
                     "acquisition":"verified_M2_endpoint" if cached else "official_exact_HTTP_range"}
            if cached:
                check(entry["rgb"] == prior["rgb"] and pixel_sha == prior["cropped_display_rgb8_pixel_sha256"], "M2/M4 endpoints differ after native conversion")
            if role == "references":
                entry.update(valid_mask={"kind":"all_true","width":256,"height":256}, object_mask=None,
                             object_mask_reason="The original DTU ObsMask is volumetric. No independently verified 2D object segmentation is available for these native views.")
            views[view] = entry
            received.append(view)
        common = {"run_id":os.environ["EL_RUN_ID"],"object_id":obj["object_id"],"scan_id":obj["scan_id"],
                  "physical_family_id":obj["physical_family_id"],"roster":pin(config["roster"]),"freeze":pin(config["freeze"]),
                  "roster_sha256":config["roster"]["sha256"],"freeze_sha256":config["freeze"]["sha256"],
                  "lighting_id":3,"camera_plan_sha256":rc["plan_sha256"],"camera_sha256":rc["camera_sha256"],
                  "crop_sha256":rc["crop_sha256"],"view_set_sha256":rc["view_set_sha256"],
                  "source_view_ids":roster["source_view_ids"],"target_view_ids":roster["target_view_ids"],"archive":roster["archive"],
                  "crop_implementation":"evidence_lens.dtu_camera.convert_rgb","pillow_version":pillow_version}
        if role == "sources":
            write_json(a.out/"source-provenance.json", {**common,**condition_hashes,"schema":"el-t0-dtu-source-1","condition":cond,
                      "opening":pin(config["freeze"]),"sources":views,"source_ids":list(ids),"cached_endpoints":config.get("endpoints"),"reference_photo_reads":0})
        else:
            write_json(a.out/"references.json",{**common,"schema":"el-t0-dtu-references-1","opening":pin(config["opening"]),
                                                "software_lock":pin(config["software_lock"]),"target_ids":list(TARGETS),"views":views})
    finally:
        write_json(a.out/"network.json",{"role":role,"condition":cond,"object_id":obj["object_id"],"body_bytes":net.total,
                   "approved_network_cap_bytes":max_bytes,"members_uncompressed_from_http_bytes":fetched_bytes,
                   "verified_local_original_endpoint_bytes":cached_bytes,"completed_views":received,"elapsed_seconds":time.monotonic()-clock,
                   "converted_png_bytes":sum(r["rgb"]["bytes"] for r in views.values()),"photo_pixels_accessed_outside_role":0})


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("config","roster","freeze","crop","out"):
        p.add_argument("--"+key,type=Path,required=True)
    for key in ("opening","software-lock","source-cameras","render-cameras","endpoint-provenance","endpoint0","endpoint3"):
        p.add_argument("--"+key,type=Path)
    convert(p.parse_args())
