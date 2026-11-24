"""Register the full DTU metadata pool, documentary identity decisions and ordered T0 cohort."""
import argparse
import collections
import json
import math
import re
import shutil
from pathlib import Path

from t0_dtu_archive import check, digest, write_json


SOURCES = {"s00": 24, "s01": 23, "s02": 17, "s03": 22}
TARGETS = {"v01": 16, "v02": 18, "v03": 25, "v04": 37, "v05": 35, "v06": 36}
EXPOSED = {1: "painted tapered container", 6: "architectural miniature"}
# These are exclusions for uncertainty. They are not declarations that the articles proved an exact physical match.
POSSIBLE_SHARED = ((4, 49), (10, 95), (12, 59, 97), (34, 36), (118, 123))
ROTATION = re.compile(r"^(\d+)-(\d+)\s*$", re.MULTILINE)


def read(path):
    return json.loads(path.read_bytes())


def file_record(path):
    raw = path.read_bytes()
    return {"sha256": digest(raw), "bytes": len(raw)}


def matlab_ids(raw):
    lines = raw.decode("cp1252").splitlines()
    choices = [re.fullmatch(r"\s*UsedSets\s*=\s*\[([\d: ]+)\];\s*", line) for line in lines]
    active = [m.group(1) for m in choices if m]
    check(len(active) == 1, "ambiguous active author evaluation membership")
    ids = []
    for exp in active[0].split():
        numbers = list(map(int, exp.split(":")))
        check(len(numbers) in (1, 2), "unsupported official ID expression")
        ids.extend(range(numbers[0], numbers[-1] + 1))
    check(len(set(ids)) == len(ids), "repeated official evaluation ID")
    return set(ids)


def components(scans, rotations, labels):
    parent = {i: i for i in scans}

    def endpoint(n):
        while parent[n] != n:
            n = parent[n]
        return n

    for members in [*rotations, *(g for g in POSSIBLE_SHARED if all(n in labels for n in g))]:
        use = sorted(n for n in members if n in scans)
        if use:
            for n in use[1:]:
                a, b = endpoint(use[0]), endpoint(n)
                parent[max(a, b)] = min(a, b)
    return {n: endpoint(n) for n in scans}


def verify_cameras(path):
    plan = read(path / "camera-plan.json")
    check(plan["source_view_ids"] == SOURCES and plan["target_view_ids"] == TARGETS and plan["lighting_id"] == 3,
          "prior registered camera layout or lighting changed")
    camera = read(path / "cameras.json")
    for member, key in (("cameras.json", "camera_sha256"), ("crop.json", "crop_sha256"),
                        ("metric-frame.json", "metric_frame_sha256"), ("projection-checks.json", "projection_checks_sha256"),
                        ("view-set.json", "view_set_sha256")):
        check(file_record(path / member)["sha256"] == plan[key], "prior calibration hash mismatch: " + member)
    check(read(path / "projection-checks.json")["passed"] is True and camera["unit"] == "m", "metric validation failed")
    for view in camera["views"].values():
        check(view["width"] == view["height"] == 256 and all(math.isfinite(v) for k in ("K", "c2w") for row in view[k] for v in row),
              "invalid registered camera")
    for condition, names in (("M2", {"s00","s03"}), ("M4", set(SOURCES))):
        for role, ids in (("source_cameras", names), ("render_cameras", names | set(TARGETS))):
            p = path / "condition" / condition / (role + ".json")
            check(file_record(p) == plan["condition_cameras"][condition][role] and read(p)["views"] == {i:camera["views"][i] for i in ids},
                  "condition member differs from the official calibration")
    return plan


def make_roster(a, config):
    archive = read(a.archive)
    rows = read(a.members)
    files = {r["name"]:r for r in rows}
    check(len(rows) == len(files) == archive["entries"] and digest(a.central.read_bytes()) == archive["directory_sha256"], "unstable directory")
    scans = sorted({int(m.group(1)) for n in files if (m := re.fullmatch(r"Rectified/scan(\d+)/", n))})
    check(len(scans) == 124, "official site/archive scan counts differ")
    readme = a.readme.read_bytes().decode("cp1252")
    ranges = [(int(lo), int(hi)) for lo, hi in ROTATION.findall(readme.replace("\r", ""))]
    check(len(ranges) == 8 and "360 degree rotations" in readme, "unrecognized author rotation manifest")
    rotations = [list(range(lo, hi + 1)) for lo, hi in ranges]
    benchmark = matlab_ids(a.used.read_bytes())
    check(benchmark <= set(scans), "unknown author-evaluation scan ID")
    source = read(a.subjects)
    labels = {int(k):v for k,v in source["objects"].items()}
    check(set(labels) <= set(scans), "secondary documentary labels contain unknown scan IDs")
    # Rotated members share the same subject. A label applies to all scans in a documented author rotation.
    asserted = dict(labels)
    for group in rotations:
        names = {labels[i] for i in group if i in labels}
        if len(names) == 1:
            for i in group:
                asserted.setdefault(i, next(iter(names)))
    roots = components(scans, rotations, asserted)
    family_members = collections.defaultdict(list)
    for scan, group in roots.items():
        family_members[group].append(scan)
    plan = verify_cameras(a.calibration)
    sample = {r["name"]:r for r in read(a.sample)}
    # Cross-package CRC/size anchors the raw rectified view index against the already registered official sample/rig.
    anchors = []
    for scan in EXPOSED:
        for view, num in {**SOURCES, **TARGETS}.items():
            name = f"Rectified/scan{scan}/rect_{num:03}_3_r5000.png"
            full, old = files[name], sample["SampleSet/MVS Data/" + name]
            check(all(full[k] == old[k] for k in ("bytes", "crc32")), "SampleSet/Rectified member differs")
            anchors.append({"scan_id":scan, "view":view, "crc32":full["crc32"], "bytes":full["bytes"]})
    planned, inventory = [], []
    identities = {}
    for scan in scans:
        records = {key:files.get(f"Rectified/scan{scan}/rect_{num:03}_3_r5000.png") for key,num in {**SOURCES,**TARGETS}.items()}
        missing = [key for key,r in records.items() if r is None or r["bytes"] <= 0]
        fingerprint = tuple((r["bytes"],r["crc32"]) for r in records.values()) if not missing else ()
        if fingerprint and fingerprint in identities:
            raise ValueError("Two complete CRC/size sequences coincide; adjudicate the documented families before registration")
        if fingerprint:
            identities[fingerprint] = scan
        family = roots[scan]
        members = family_members[family]
        subject = EXPOSED.get(scan) or asserted.get(scan)
        if scan in EXPOSED:
            disposition = "previously_exposed"
        elif missing:
            disposition = "required_image_member_missing"
        elif not subject:
            disposition = "physical_identity_unresolved"
        elif any(i in EXPOSED for i in members):
            disposition = "grouped_with_exposed_subject"
        elif scan > min(i for i in members if i in asserted):
            disposition = "conservatively_grouped_or_rotated_duplicate"
        elif len(planned) < 6:
            disposition = "selected"
        else:
            disposition = "eligible_beyond_registered_budget"
        obj = {"object_id":f"set{scan}", "scan_id":scan, "official_evaluation_member": scan in benchmark,
               "primary_subject":subject, "object_label_published_for_scan_id": next((i for i in members if i in labels and labels[i]==subject), None),
               "physical_family_id": (f"dtu-subject-{family}" if subject else None), "documented_or_possible_shared_group":members if subject or len(members)>1 else [],
               "declared_rotation_group": next((g for g in rotations if scan in g),[]),
               "required_views_complete":not missing, "missing_view_ids":missing, "disposition":disposition}
        inventory.append(obj)
        if disposition == "selected":
            planned.append({**obj, "physical_identity_evidence":{"reference":source["url"], "locator":source["locator"],"independent_label":subject,
                            "official_rotation_check":"ReadMe.txt verified", "previous_subject_check":EXPOSED},
                            "source_members":{k:records[k] for k in SOURCES}, "target_members":{k:records[k] for k in TARGETS}})
    check(planned and planned == sorted(planned,key=lambda x:x["scan_id"]), "unsupported empty or unordered sample")
    record = {"schema":"el-t0-dtu-roster-1", "scientific_role":"T0 frozen confirmation", "selection_version":"v1.2 ascending eligible independent physical family",
              "object_ids":[o["object_id"] for o in planned], "objects":planned, "inventory":inventory, "archive":archive,
              "inventory_counts":dict(collections.Counter(row["disposition"] for row in inventory)), "conditions":["M2","M4"],
              "source_view_ids":SOURCES,"target_view_ids":TARGETS,"target_ids":list(TARGETS),"lighting_id":3,
              "camera":{"plan_sha256":file_record(a.calibration/"camera-plan.json")["sha256"],"crop_sha256":plan["crop_sha256"],
                        "camera_sha256":plan["camera_sha256"],"view_set_sha256":plan["view_set_sha256"],"conditions":plan["condition_cameras"],
                        "rig_source":"official right-camera Rectified P coordinates, common first 49 robotic views; cross-archive anchors on the two exposed scenes"},
              "published_identity_source":source["url"], "author_evaluation_count":len(benchmark),
              "author_rotation_groups":rotations,"conservative_additional_groups":POSSIBLE_SHARED, "previous_exposure":EXPOSED,
              "identity_limit":"The original publications count scenes, and provide no physical-serial-number map. Distinct named principal subjects and the explicit rotations support this family-level comparison. Unnamed scenes are unresolved. The same laboratory, fixed background, props and incidental objects may recur. Publication object/error ratios and outcomes never enter the selection.",
              "scope":"All cropped pixels are evaluated after the independent target gate. The documentary primary subject is not an object segmentation.",
              "registration_inputs":config["evidence"],"budget":config["budget"], "original_photo_bodies_read_during_selection":0,
              "explicit_unlisted_scan_numbers":[i for i in range(min(scans),max(scans)+1) if i not in scans]}
    write_json(a.out/"roster.json",record)
    write_json(a.out/"calibration-index.json",{"schema":"el-t0-rig-1","camera":record["camera"],"metadata_only_cross_archive_anchors":anchors})
    for name in ("camera-plan.json","cameras.json","crop.json","view-set.json","metric-frame.json","projection-checks.json"):
        shutil.copyfile(a.calibration/name,a.out/name)
    shutil.copytree(a.calibration/"condition",a.out/"condition")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("config","out","archive","members","central","readme","used","subjects","sample","calibration"):
        p.add_argument("--"+flag,type=Path,required=True)
    args=p.parse_args()
    make_roster(args,read(args.config))
