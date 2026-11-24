"""Seal the staged A build, clean installs, immutable score matrix and separate reference permission."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from dual_records import bind, resolve, selected, stage


OLD_DOWNLOAD = "20261124T053239837552Z-8ec25846c9824287"
WEIGHT_RECEIPT = "20261124T041052425868Z-d308c134e69d4d4b"
METHODS = {"B-L1-transfer", "MS-L1", "MS-frozen", "MS-ft"}
TARGETS = [f"v{i:02d}" for i in range(1, 7)]


def seal_scores(root, grade, bundles, out):
    report = load(grade)
    require(report["wheel_version"] == "0.5.0" and set(report["methods"]) == METHODS
            and len(report["cells"]) == 24 and all(len(c["views"]) == 6 for c in report["cells"]),
            "all four methods must finish the six authored grids")
    index = load(bundles)
    cells = {(row["tag"], row["method"]):row for row in report["cells"]}
    require(set(cells) == {(tag,m) for tag in index["routes"] for m in METHODS}, "incomplete score matrix")
    archive = out/"v05-bell-scores.zip"
    files = []
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for (tag,method), row in sorted(cells.items()):
            cell = root/tag/method
            raw = cell/"scores/scores.json"
            summary = load(raw)
            expected_bundle = index["routes"][tag]["sha256"]
            require(summary["bundle_sha256"] == row["bundle_sha256"] == expected_bundle and
                    sha(raw) == row["scores_sha256"] and summary["target_order"] == TARGETS
                    and summary["method"] == method and set(row["views"]) == set(TARGETS),
                    "scored cell differs from its pinned models or sources")
            listed = [(raw,"scores/scores.json",sha(raw)),(cell/"grade.json","grade.json",sha(cell/"grade.json")),
                      (cell/"report.html","report.html",row["html"]["html_sha256"])]
            for spec in [v["arrays"] for v in summary["views"]] + [
                v["prediction"] for v in summary["views"] if v["prediction"]] + [s["display"] for s in summary["sources"]]:
                listed.append((cell/"scores"/spec["path"],"scores/"+spec["path"],spec["sha256"]))
            for path,name,expected in listed:
                require(sha(path)==expected,"score member differs "+name)
                rel=f"{tag}/{method}/{name}"
                z.write(path,rel)
                files.append(dict(path=rel,sha256=expected,bytes=path.stat().st_size))
        z.write(grade,"acceptance.json")
    save(out/"scores-index.json",{"schema":"el-v05-sealed-bell-scores-1","run_id":os.environ["EL_RUN_ID"],
         "grade_sha256":sha(grade),"bundles_index_sha256":sha(bundles),"archive":archive.name,
         "bytes":archive.stat().st_size,"sha256":sha(archive),"score_members":files,
         "method_views":sum(len(x["views"]) for x in cells.values()),
         "comparison_method_views":sum(len(x["views"]) for x in cells.values() if x["tag"] in {"legacy-letterbox","landscape"})})


def operations(a):
    s=a.store
    roots=(s,s/"history")
    prior=s/"history"
    reg=selected(s,"a-grid-registration")
    if a.action=="build":
        import subprocess
        commit = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
        return stage(s,a.label or "a-release-build","archive_verify",{"registration":reg},
            {"exact_source":commit,"wheel_version":"0.5.0","risk_weights":"v0.3.0-rc.1"},
            ["{python}","scripts/general_build.py","--repo",str(Path.cwd()),"--commit",commit,"--out","{out}"],
            "payload/build.json",30_000_000)
    bundle = selected(s,"a-bell-bundle-publication")
    bell = bind(s,bundle["run_id"],"payload/v05-bell-bundles.zip")
    if a.action=="accept":
        build=selected(s,a.build_label)
        w=bind(s,build["run_id"],"payload/evidence_lens-0.5.0-py3-none-any.whl")
        old=bind(prior,OLD_DOWNLOAD,"payload/evidence_lens-0.4.0-py3-none-any.whl")
        oldzip=bind(prior,OLD_DOWNLOAD,"payload/evidence-lens-original-replays-v1.zip")
        model=bind(prior,WEIGHT_RECEIPT,"payload/cpu/release.json")
        command=["{python}","scripts/v05_install_replay.py","--old-wheel","{input:oldwheel}","--wheel","{input:wheel}",
                 "--original","{input:original}","--examples","{input:examples}","--worker","{source}/scripts/v05_profile_accept.py",
                 "--models",str(resolve(roots,model).parent),"--out","{out}","--groups","default","vision","--keep-scores",
                 "--wheel-sha256",w["sha256"],"--examples-sha256",bell["sha256"]]
        return stage(s,a.label or "a-mac-install","archive_verify",{"wheel":w,"examples":bell,
            "original":oldzip,"oldwheel":old,"model_release":model,"registration":reg},
            {"old":"0.4.0","new":"0.5.0","models":sorted(METHODS),"readable_roles":"source and prediction, evaluator references closed",
             "neural_threads":4,"default_and_full_isolated":True}, command, "payload/installation.json",430_000_000)
    if a.action=="scores":
        install=selected(s,a.install_label)
        grade=bind(s,install["run_id"],"payload/profiles-vision.json")
        root=resolve(roots,grade).parent/"profiles-vision"
        return stage(s,"a-bell-scores-sealed","archive_verify",{"install":install,"grade":grade,"bundles":bundle},
            {"matrix":"six profiles, all four models, six planned targets per cell","references":"sealed"},
            ["{python}","scripts/v05_local_release.py","scores-worker","--profiles",str(root),
             "--grade","{input:grade}","--bundles","{input:bundles}","--out","{out}"],"payload/scores-index.json",240_000_000)
    if a.action=="opening":
        scores=selected(s,"a-bell-scores-sealed")
        scores_archive=bind(s,scores["run_id"],"payload/v05-bell-scores.zip")
        sources={"scores":scores,"scores_archive":scores_archive,"bundles_archive":bell,
                 "registration":selected(s,"a-bell-register-six")}
        return stage(s,"a-bell-reference-opening","archive_verify",sources,
             {"authorization":"six authored landscape target cameras for independent A comparison; no other views"},
             ["{python}","scripts/v05_local_release.py","opening-worker","--grade","{input:scores}",
              "--config","{config}","--out","{out}"],"payload/opening.json",6500)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["build","accept","scores","opening","scores-worker","opening-worker"])
    for k in ("store","profiles","grade","bundles","config","out"):
        parser.add_argument("--"+k,type=Path)
    parser.add_argument("--label")
    parser.add_argument("--build-label",default="a-release-build")
    parser.add_argument("--install-label",default="a-mac-install")
    a=parser.parse_args()
    if a.action=="scores-worker":
        seal_scores(a.profiles,a.grade,a.bundles,a.out)
    elif a.action=="opening-worker":
        parents=load(a.config)["sources"]
        require(load(a.grade)["comparison_method_views"]==48 and load(a.grade)["method_views"]==144 and
                load(a.grade)["sha256"]==parents["scores_archive"]["sha256"],"missing sealed score cells")
        save(a.out/"opening.json",dict(schema="el-v05-bell-reference-opening-1",run_id=os.environ["EL_RUN_ID"],
             registered_scores_sealed=True,conditions=["landscape","legacy-letterbox"],targets=TARGETS,
             terminal_score_count=48,**parents))
    else:
        print(json.dumps(operations(a)))
