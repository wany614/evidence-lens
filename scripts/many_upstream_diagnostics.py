"""Measure fixed development reconstructions against evaluator-only RGB and account for the shared compute budget."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import shutil
import time

import numpy as np

from dual_records import bind, resolve, selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from many_data_access import EvaluatorTruth
from many_upstream_control import publish_packet


PILOT="b04-layered-louvers-r16"
DEV=["b04-layered-louvers-r16","b04-layered-louvers-r128","delivery-area-r34"]
COHORT=[(f"b{i:02d}-{name}-r{n}",n,6) for i,name in enumerate(
        ("staggered-arcade","terraced-court","woven-truss","layered-louvers","perforated-gallery","corner-alcoves"),1) for n in (16,128)
        ]+[("delivery-area-r34",34,10),("living-room-r55",55,10),("bridge-r100",100,10)]
FIRST_QUEUE=["delivery-area-r34","b01-staggered-arcade-r16","b01-staggered-arcade-r128","b02-terraced-court-r16",
             "b02-terraced-court-r128","b03-woven-truss-r16","b03-woven-truss-r128","b04-layered-louvers-r128"]


def command(action,pins):
    c=["{python}","-B","{source}/scripts/many_upstream_diagnostics.py",action,"--config","{config}","--out","{out}"]
    for name in pins:c+=["--input",name,"{input:"+name+"}"]
    return c


def control(s,action,tag=None):
    roots=s,s/"history"
    if action=="evaluate":
        require(tag in DEV,"only the three development configurations are in this evaluation")
        bank=selected(s,"b-bank-"+tag);scene=tag.rsplit("-r",1)[0];target=selected(s,"b-truth-"+scene)
        proof=bind(s,bank["run_id"],"payload/upstream-evidence.json");meta=load(resolve(roots,bank));tr=load(resolve(roots,target))
        pins={"bank":bank,"proof":proof,"truth":target,"truth_zip":bind(s,target["run_id"],"payload/"+tr["archive"]["path"]),
              "opening":selected(s,"b-train-dev-h-opening")}
        for i,k in enumerate(meta["target_ids"]):pins[f"pred-{i}"]=bind(s,bank["run_id"],"payload/"+meta["targets"][k]["render"]["path"])
        return stage(s,"b-upstream-dev-"+tag,"evaluation",pins,{"role":"independent development-only reconstruction evaluation",
             "scene":meta["scene_id"],"configuration":tag,"error_threshold":.10,"opaque_alpha":.5,"near_black_threshold":.02},
             command("evaluate-worker",pins),"payload/diagnostic.json",45000)
    if action=="budget":
        bank=selected(s,"b-bank-"+PILOT);pins={"bank":bank,"proof":bind(s,bank["run_id"],"payload/upstream-evidence.json"),
            "evaluation":selected(s,"b-upstream-dev-"+PILOT),"study":selected(s,"b-study-registration")}
        for name in [PILOT,*FIRST_QUEUE]:
            pins["sfm-"+name]=selected(s,"b-sfm-"+name)
        return stage(s,"b-pilot-budget","report",pins,{"cohort":COHORT,"conservative_factor":1.5,"cap_seconds":28800,"fit_cap":20,
           "diagnostic":"worst planned 128+10 poses added to projected voluntary fits; overall/per-fit ceilings govern each subsequent admission"},
           command("budget-worker",pins),"payload/budget.json",15000)
    if action=="queue":
        pins={"budget":selected(s,"b-pilot-budget"),**{name:selected(s,"b-fit-ticket-"+name) for name in FIRST_QUEUE}}
        return stage(s,"b-training-development-fit-queue","archive_verify",pins,{"order":FIRST_QUEUE,"pilot":PILOT,"confirmation_allowed":False},
                     command("queue-worker",pins),"payload/queue.json",35000)
    pins={"budget":selected(s,"b-pilot-budget"),**{k:selected(s,"b-upstream-dev-"+k) for k in DEV}}
    for k in DEV:
        bank=selected(s,"b-bank-"+k)
        proof=load(resolve(roots,bind(s,bank["run_id"],"payload/upstream-evidence.json")))
        pins["training-"+k]=proof["native_registry"]["training"]
        pins["sfm-"+k]=proof["native_registry"]["sfm"]
    return stage(s,"b-development-diagnostics","report",pins,{"scope":"authored and external development only","configurations":DEV},
                 command("summary-worker",pins),"payload/diagnostics.json",68000)


def evaluate(cfg,paths,out):
    bank,proof,truth,gate=(load(paths[k]) for k in ("bank","proof","truth","opening"))
    ids=bank["target_ids"]
    require(bank["scene_id"]==truth["scene_id"]==cfg["scene"] and truth["split"]=="development"
            and gate["allowed_targets"][cfg["scene"]]==ids and truth["H"]==ids
            and proof["bank_sha256"]==sha(paths["bank"]) and truth["opening_sha256"]==sha(paths["opening"]),
            "development-only independent evaluator lineage differs")
    rgb_mean=np.asarray(proof["source_only_mean"],np.float32)
    rows=[]
    with tempfile.TemporaryDirectory(dir=out.parent) as td:
        root=Path(td);(root/"truth.json").write_bytes(paths["truth"].read_bytes())
        p=root/truth["archive"]["path"]
        try:os.link(paths["truth_zip"],p)
        except OSError:shutil.copyfile(paths["truth_zip"],p)
        vault=EvaluatorTruth(root/"truth.json",cfg["sources"]["truth"]["sha256"])
        for i,k in enumerate(ids):
            require(sha(paths[f"pred-{i}"])==bank["targets"][k]["render"]["sha256"],"different sealed development prediction")
            with np.load(paths[f"pred-{i}"],allow_pickle=False) as frame:
                rgb,alpha,z=(frame[c] for c in ("rgb","alpha","z"))
            h=vault.target_rgb(k);err=np.abs(rgb-h).mean(2);base=np.abs(rgb_mean-h).mean(2)
            mask=alpha>=cfg["opaque_alpha"];is_black=(rgb.max(2)<=cfg["near_black_threshold"])
            rows.append({"target":k,"analysis_pixels":int(err.size),"mae":float(err.mean()),"input_mean_rgb_baseline_mae":float(base.mean()),
                "base_rate_gt_0_10":float((err>cfg["error_threshold"]).mean()),"opaque_pixels":int(mask.sum()),
                "opaque_mae":float(err[mask].mean()) if mask.any() else None,"finite_expected_z":int(np.isfinite(z).sum()),
                "predicted_near_black_pixels":int(is_black.sum()),"reference_near_black_pixels":int((h.max(2)<=cfg["near_black_threshold"]).sum())})
    save(out/"diagnostic.json",{"schema":"el-many-upstream-development-1","run_id":os.environ["EL_RUN_ID"],"configuration":cfg["configuration"],
        "bank":cfg["sources"]["bank"],"truth":cfg["sources"]["truth"],"opening":cfg["sources"]["opening"],
        "source_self_mae":proof["mean_source_self_mae"],"source_constant_mean_baseline_mae":proof["mean_source_colour_baseline_mae"],
        "novel_target_mae":float(np.mean([r["mae"] for r in rows])),"novel_constant_mean_baseline_mae":float(np.mean([r["input_mean_rgb_baseline_mae"] for r in rows])),
        "gaussians":proof["gaussians"],"source_self":proof["source_self"],"targets":rows,"checkpoint_sha256":bank["checkpoint_sha256"],
        "mean_source_only_rgb":proof["source_only_mean"],"upstream_runtime":proof["runtime"],"target_rgb_decodes":len(ids),
        "target_reference_reads_by_trainer":0,"scope":"development diagnostic for a source-only fit; no confirmation photographs"})


def budget(cfg,paths,out):
    proof,dev=load(paths["proof"]),load(paths["evaluation"]);native=proof["runtime"]
    fit=float(native["fit_seconds"]);worker=float(native["job_wall_seconds"])
    require(0<fit<=worker<=3600 and dev["bank"]==cfg["sources"]["bank"],"a successful timed pilot and its independent development result are required")
    # Includes setup, packet verification, native+adapter readback and compressed encoding of every planned frame.
    nonfit=(worker-fit)/(len(proof["source_self"])+len(dev["targets"]))
    factor=cfg["conservative_factor"]
    items=[{"case":key,"R":nr,"target_poses":nh,"seconds":worker if key==PILOT else factor*(fit+nonfit*(nr+nh))}
           for key,nr,nh in cfg["cohort"]]
    needed=sum(item["seconds"] for item in items)
    sx=[load(paths[k]) for k in paths if k.startswith("sfm-")]
    worst=max(item["seconds"] for item in items)
    optional_count=cfg["fit_cap"]-len(items)
    possible_20=needed+optional_count*worst
    require(len(items)==15,"required reconstruction matrix changed")
    save(out/"budget.json",{"schema":"el-many-fit-pilot-budget-1","run_id":os.environ["EL_RUN_ID"],"actual_pilot":cfg["sources"]["proof"],
         "evaluated_pilot":cfg["sources"]["evaluation"],"fit_count_consumed":1,"fit_seconds":fit,"observed_gpu_job_seconds":worker,
         "observed_nonfit_seconds":worker-fit,"allocated_peak_bytes":native["peak_torch_allocated_bytes"],"reserved_peak_bytes":native["peak_torch_reserved_bytes"],
         "max_cumulative_gpu_seconds":cfg["cap_seconds"],"remaining_gpu_seconds_after_pilot":cfg["cap_seconds"]-worker,
         "planning_multiplier":factor,"projection_per_frame_nonfit_seconds":nonfit,"mandatory_15":items,"projected_15_total_seconds":needed,
         "projected_15_remaining_seconds":needed-worker,"projected_15_fits_budget":needed<=cfg["cap_seconds"] and worst<=3600,
         "potential_20_total_seconds":possible_20,"potential_20_fits_budget":possible_20<=cfg["cap_seconds"] and worst<=3600,
         "optional_fits_dispatched":0,"observed_source_sfm_cpu_seconds":sum(x["source_cpu_seconds"] for x in sx),
         "observed_source_sfm_cases":len(sx),"observed_sfm_cpu_sec_by_sources":{k:load(paths[k])["source_cpu_seconds"] for k in paths if k.startswith("sfm-")},
         "policy":"Monitor measured native job seconds after each terminal. Reserve remaining confirmation fits first and omit optional second developer or retries if the projected budget fails."})


def worker(action,cfg,pins,out):
    if action=="evaluate-worker":return evaluate(cfg,pins,out)
    if action=="budget-worker":return budget(cfg,pins,out)
    if action=="queue-worker":
        require(load(pins["budget"])["projected_15_fits_budget"],"pilot projection exceeds cumulative or single-fit cap")
        rows=[{"tag":k,"pin":cfg["sources"][k],"trial":load(pins[k])["trial"],"packet":load(pins[k])["packet"]} for k in cfg["order"]]
        return save(out/"queue.json",{"schema":"el-many-development-fit-queue-1","run_id":os.environ["EL_RUN_ID"],"pilot":cfg["pilot"],"budget":cfg["sources"]["budget"],"fits":rows,
             "confirmations":0,"instructions":"One sequential job per immutable ticket, one D: recorder run each, check 60 minutes per fit and cumulative 8 GPU hours."})
    studies={key:load(pins[key]) for key in DEV}
    for key,study in studies.items():
        fit=load(pins["training-"+key]);sfm=load(pins["sfm-"+key])
        require(fit["native_checkpoint"]["sha256"]==study["checkpoint_sha256"] and fit["R"]==sfm["source_ids"]
                and fit["gaussians"]==study["gaussians"],"development morphology and evaluator checkpoint disagree")
        sources,targets=study["source_self"],study["targets"]
        study["initialization_and_projection"]={
            "sfm":cfg["sources"]["sfm-"+key],"native_training":cfg["sources"]["training-"+key],
            "initial_source_triangulations":sfm["unique_points_before_cap"],"retained_initial_gaussians":sfm["initial_points"],
            "calibrated_source_pairs":len(sfm["pairs"]),"active_opacity_over_0_005":fit["alpha_above_0_005"],
            "opacity_percentiles_min_q1_median_q3_max":fit["opacity_percentiles"],"world_scale_percentiles":fit["scale_percentiles"],
            "mean_projected_gaussians":float(np.mean([r["projected_gaussians"] for r in sources])),
            "source_max_projected_radius_px":max(r["max_projected_radius_px"] for r in sources),
            "source_alpha_over_0_5_fraction":sum(r["alpha_gt_0_5"] for r in sources)/(len(sources)*378*252),
            "target_alpha_at_least_0_5_fraction":sum(r["opaque_pixels"] for r in targets)/sum(r["analysis_pixels"] for r in targets),
            "target_near_black_fraction":sum(r["predicted_near_black_pixels"] for r in targets)/sum(r["analysis_pixels"] for r in targets),
            "reference_near_black_fraction":sum(r["reference_near_black_pixels"] for r in targets)/sum(r["analysis_pixels"] for r in targets)}
    save(out/"diagnostics.json",{"schema":"el-many-development-upstream-closeout-1","run_id":os.environ["EL_RUN_ID"],"development":studies,"parents":cfg["sources"],
         "development_reconstructions":len(studies),"external_development_reconstructions":1,"confirmation_views_read":0})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("evaluate","budget","queue","summary","evaluate-worker","budget-worker","queue-worker","summary-worker"))
    for k in ("store","config","out","exchange"):p.add_argument("--"+k,type=Path)
    p.add_argument("--tag");p.add_argument("--input",action="append",nargs=2,default=[]);a=p.parse_args()
    if a.action.endswith("-worker"):worker(a.action,load(a.config),{k:Path(v) for k,v in a.input},a.out)
    else:
        s=a.store.resolve();pin=control(s,a.action,a.tag)
        if a.action=="queue" and a.exchange:
            from make_handoff import publish
            from evidence_lens._bundle import encoded
            for key in FIRST_QUEUE:publish_packet(s,selected(s,"b-fit-ticket-"+key),a.exchange)
            doc=resolve((s,s/"history"),pin)
            publish(a.exchange/"development-queue.json",doc.read_bytes());publish(a.exchange/"development-queue-pin.json",encoded(pin))
        print(json.dumps(pin))
