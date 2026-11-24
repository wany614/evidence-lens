"""Export the trained NumPy parameters with their measured evaluation and material-specific notices."""

import argparse
import os
from pathlib import Path
import sys
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

sys.path.insert(0,str(Path(__file__).resolve().parent))
from dual_records import bind,selected,stage
from evidence_lens._bundle import encoded,load,save,sha
from evidence_lens._verify import require
from many_science_control import cpu_command,runtime,suite


ARCHIVE="evidence-lens-many-source-model-0.1.0.dev0.zip"


def control(s):
    pins=suite(s)
    for k,label in (("frozen","b-model-freeze"),("fit","b-head-final"),("candidates","b-head-candidates"),
                    ("choice","b-development-selection"),("results","b-scientific-results"),
                    ("opening","b-confirmation-h-opening"),("upstream","b-upstream-closeout")):
        pins[k]=selected(s,label)
    pins["model"]=bind(s,pins["fit"]["run_id"],"payload/model.json")
    cmd=cpu_command(runtime(s,pins),"many_model_release.py")+["--config","{config}","--out","{out}"]
    for k in (*pins,"license","rights"):
        cmd+=["--file",k,"{input:"+k+"}"]
    return stage(s,"b-model-package","archive_verify",pins,{"published_rights":"original head Apache-2.0; original writing and grantable measurements CC BY 4.0"},
        cmd,"payload/model-release.json",190000,{"license":Path("LICENSE").resolve(),
        "rights":Path("docs/research/many-source-v1/rights.md").resolve()})


def export(a):
    from evidence_lens_many import FEATURE_SCHEMA_SHA256
    from evidence_lens_many.schema import BASELINE_ID,feature_schema
    f={k:Path(v) for k,v in a.file}
    cfg=load(a.config)
    for k,p in f.items():
        expected=cfg["sources"][k]["sha256"] if k in cfg["sources"] else cfg["extra_sha256"][k]
        require(sha(p)==expected,"model release input changed")
    model,fit,cands,choice,results,freeze,opening,up=(load(f[k]) for k in
        ("model","fit","candidates","choice","results","frozen","opening","upstream"))
    require(fit["fit_count"]==1 and cands["fit_count"]==3 and fit["rows"]<=200000
            and model["tau"] is None and model["baseline_id"]==BASELINE_ID
            and model["feature_schema_sha256"]==FEATURE_SCHEMA_SHA256
            and model["lambda_l2"]==choice["lambda_l2"] and sha(f["model"])==freeze["final_model"]["sha256"]
            and opening["model_freeze"]["sha256"]==sha(f["frozen"])
            and len(results["scenes"])==9 and results["heldout_external_scenes"]==2,
            "inconsistent frozen model or report")
    external=[r for r in results["scenes"] if r["scene"] in ("living_room","bridge")]
    def percent(x):
        return "N/A" if x is None else f"{x:.3%}"
    card=["# Evidence Lens multi-source CPU model","","## Intended use","",
        "For a fixed static Gaussian reconstruction, the model orders target pixels by their risk of an RGB error. "
        "Its training event is FP32 display-sRGB mean absolute difference above 0.05 on the frozen 378 × 252 grid. "
        "A finite score is an uncalibrated ranking. A request supplies a fixed target prediction, the frame, source photographs and matching source self-renders. "
        "A separate evaluator can add independent reference labels.","",
        "## Training, model identity and operating point","",
        f"The final model is `{model['model_id']}`, SHA-256 `{sha(f['model'])}`. It has {len(model['coef'])} coefficients and one intercept. "
        f"The final fit used {fit['rows']:,} source-query rows, including {fit['positive_rows']:,} positive rows from six checkpoints and 36 target units across the three original training families. "
        "Repeated querying of a physical pixel redistributes that pixel's training weight. "
        "The fitted weighted finite means, imputation values and scales are included in model.json.","",
        f"The development endpoint selected L2 = {model['lambda_l2']:g} from three candidates, followed by one deterministic refit. "
        f"The selected simple comparison was `{choice['strong_simple_rule']}`. The choice used the authored b04 family and ETH3D delivery_area with equal scene weights. "
        "All coefficient fitting used b01-staggered-arcade, b02-terraced-court and b03-woven-truss. "
        "The operating threshold is null. Calibration fits and neural updates both total zero. At Q4 the study retained the released MS-L1 and the genuine fine-tuned MS-ft as separate controls.","",
        "## Frozen external evaluation","",
        "| Scene | Labelled errors / pixels | Common K16 pixels | Accepted | New errors | Simple errors | Target-equal difference |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in external:
        card.append(f"| {r['scene']} | {r['error_pixels']:,} / {r['labelled_pixels']:,} | {r['common']:,} | {r['quota']:,} | {r['new_errors']:,} | {r['rule_errors']:,} | {percent(r['new_minus_rule'])} |")
    card+=["",f"The primary comparison status is `{results['comparison_status']}`. Each target accepts floor(20% of its shared finite domain). "
        "Each of the two ETH3D confirmation scenes supplies ten repeated target views. P1 measures missed comparable evidence using an explicitly costed full audit. "
        "Independent mesh visibility is available for original families. External scene geometry uses supplied camera calibration in scene units. "
        "Outputs describe these two external cases and the registered families. Their use for new environments, dynamic video, physical safety or historical authenticity remains unvalidated.","",
        "## Provenance and rights","",
        f"Scientific source commit `{freeze['science_code_commit']}`. Stage A `{BASELINE_ID}`. "
        f"Final fit `{fit['run_id']}`, confirmation permission `{opening['run_id']}`, results `{results['run_id']}`. "
        f"The upstream ledger includes {up['issued_fit_count']} registered Gaussian fits. The CPU study uses gsplat v1.5.3 reconstructions, the registered analysis profile, and known calibration. "
        "See training.json, development-selection.json and freeze.json for the input manifests and feature identities.","",
        "Copyright 2026 wany614. The original logistic parameters and preprocessing are Apache-2.0. Original explanatory text and grantable measurements are CC BY 4.0. "
        "ETH3D imagery is subject to its own terms. The CPU parameters do not require the optional DINO control package. "
        "[ETH3D](https://www.eth3d.net/) · [Stage A](https://github.com/wany614/evidence-lens/releases/tag/v0.5.0) · "
        "[Legacy risk models](https://github.com/wany614/evidence-lens/releases/tag/v0.3.0-rc.1)"]
    md=("\n".join(card)+"\n").encode()
    operating=encoded(dict(schema="el-many-rank-operating-point-1",model_id=model["model_id"],tau=None,
        units="uncalibrated error ranking",calibration_fits=0,quota="evaluation only; per-target floor(20% of common finite labelled pixels)"))
    contents={"model.json":f["model"].read_bytes(),"feature-schema.json":encoded(feature_schema()),
        "operating-point.json":operating,"training.json":f["fit"].read_bytes(),"development-selection.json":f["choice"].read_bytes(),
        "freeze.json":f["frozen"].read_bytes(),"MODEL_CARD.md":md,"LICENSE":f["license"].read_bytes(),
        "RIGHTS.md":f["rights"].read_bytes()}
    import hashlib
    roster={k:{"bytes":len(v),"sha256":hashlib.sha256(v).hexdigest()} for k,v in contents.items()}
    require(roster["feature-schema.json"]["sha256"]==FEATURE_SCHEMA_SHA256,"schema bytes differ")
    contents["manifest.json"]=encoded(dict(schema="el-many-model-files-1",files=roster,
        science_code_commit=freeze["science_code_commit"],protocol_sha256=load(f["study"])["protocol_sha256"],
        study_registration_sha256=cfg["sources"]["study"]["sha256"],
        model_id=model["model_id"],final_fit=cfg["sources"]["fit"],feature_schema_sha256=FEATURE_SCHEMA_SHA256))
    zip_path=a.out/ARCHIVE
    with ZipFile(zip_path,"x",compression=ZIP_DEFLATED) as archive:
        for path,data in sorted(contents.items()):
            entry=ZipInfo(path,(2000,1,1,0,0,0));entry.external_attr=0o100644<<16
            archive.writestr(entry,data,compress_type=ZIP_DEFLATED)
    with ZipFile(zip_path) as archive:
        require(set(archive.namelist())==set(contents) and all(archive.read(n)==v for n,v in contents.items()),
                "model ZIP readback differs")
    (a.out/"model-card.md").write_bytes(md)
    (a.out/"feature-schema.json").write_bytes(contents["feature-schema.json"])
    (a.out/"operating-point.json").write_bytes(operating)
    save(a.out/"model-release.json",dict(schema="el-many-model-release-1",run_id=os.environ["EL_RUN_ID"],
        model_id=model["model_id"],weights_sha256=sha(f["model"]),model_bytes=f["model"].stat().st_size,
        science_code_commit=freeze["science_code_commit"],feature_schema_sha256=FEATURE_SCHEMA_SHA256,
        archive=dict(name=ARCHIVE,bytes=zip_path.stat().st_size,sha256=sha(zip_path)),contents=roster,
        training=cfg["sources"]["fit"],results=cfg["sources"]["results"]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("store","config","out"):p.add_argument("--"+k,type=Path)
    p.add_argument("--file",nargs=2,action="append")
    a=p.parse_args()
    if a.store:print(control(a.store))
    else:export(a)
