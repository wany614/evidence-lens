"""Engineering-only forced branch coverage at the actual 20k cap on a synthetic 256px projection."""
import argparse
import os
import time
from pathlib import Path

import torch

from evidence_lens._bundle import load, save
from evidence_lens._verify import require
from recovery.engine import optimizers
from recovery.scene import draw, runtime, setup
from recovery.topology import BoundedRefinement, assert_adam

p = argparse.ArgumentParser()
p.add_argument("--recipe", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
require(os.environ.get("EL_RUN_KIND") == "archive_verify", "Record the topology fixture")
start = time.perf_counter()
dev = setup()
torch.manual_seed(2217)
torch.cuda.manual_seed_all(2217)
n = 20000
means = torch.rand(n, 3, device=dev)
means[:, :2] = (means[:, :2] - .5) * 2.8
means[:, 2] = 2 + means[:, 2]
scales = (.004 + .002 * torch.rand(n, 3, device=dev))
scales[1::2] *= 4
params = {k: torch.nn.Parameter(v) for k, v in {
    "means": means, "scales": scales.log(), "quats": torch.randn(n, 4, device=dev),
    "opacities": torch.full((n,), -1., device=dev), "colors": torch.randn(n, 3, device=dev)}.items()}
recipe = load(a.recipe)
tc = dict(recipe["topology"], start=1, stop=5, every=1, reset_steps=[3], prune_scale_fraction=0,
          clone_scale_fraction=.01, grow_grad2d=1., cap=20000)
s = BoundedRefinement(tc, 1.)
opts = optimizers(params, recipe, 1.)
view = (torch.tensor([[[205., 0., 128.], [0., 208., 127.], [0., 0., 1.]]], device=dev),
        torch.eye(4, device=dev)[None], 256, 256, .05, 20.)
rows = []
for step in range(1, 6):
    for o in opts.values():
        o.zero_grad(set_to_none=True)
    rgb, alpha, _, info = draw(params, view, .3)
    s.retain(params, opts, step, info)
    loss = (rgb - .35).abs().mean()
    loss.backward()
    s.observe(params, step, info)
    before = {k: t.detach().clone() for k, t in params.items()}
    for o in opts.values():
        o.step()
    delta = {k: float((v.detach() - before[k]).norm()) for k, v in params.items()}
    pre_n = len(params["means"])
    # Deliberate branch fixture. Only synthetic state is overridden, after a real raster/Adam update.
    with torch.no_grad():
        remove_n = {2: 2, 3: 4}.get(step, 0)
        if remove_n:
            params["opacities"][-remove_n:] = -20.
        if step < 5:
            s.state["count"].fill_(1)
            s.state["grad2d"].zero_()
            if step != 3:
                small = params["scales"].exp().amax(1) <= .01
                for rank, sel in enumerate((torch.where(small)[0][:4], torch.where(~small)[0][:4])):
                    s.state["grad2d"][sel] = torch.tensor([20., 16., 12., 8.], device=dev) - rank
    events = s.refine(params, opts, step)
    assert_adam(params, opts, step)
    radii = info["radii"]
    visible = (radii > 0).all(-1)
    rows.append({"step": step, "before": pre_n, "after": len(params["means"]), "events": events, "delta_l2": delta,
                 "max_radius_px": float(radii.max()), "max_abs_visible_means2d_px": float(info["means2d"].detach()[visible].abs().max()),
                 "visible": int(visible.sum()), "loss": float(loss.detach()), "phase": "fixed" if step == 5 else "active"})
ev = lambda i, k: [e for e in rows[i]["events"] if e["operation"] == k]
require(ev(0, "growth_budget")[0]["admitted"] == 0 and ev(0, "growth_budget")[0]["blocked"] == 8, "Cap did not block growth")
require(ev(1, "prune")[0]["selected"] == 2 and ev(1, "clone")[0]["selected"] == 1 and
        ev(1, "split")[0]["selected"] == 1, "Prune and both newborn types were not exercised")
require(rows[2]["after"] == n - 4 and ev(2, "reset") and rows[3]["after"] == n and
        all(r["after"] <= n for r in rows) and not rows[-1]["events"], "Regrowth/fixed tail contract")
require(all(r["delta_l2"]["means"] > 0 and r["delta_l2"]["colors"] > 0 for r in rows), "Missing active/fixed update")
torch.cuda.synchronize()
save(a.out / "forced-topology.json", {"passed": True, "fixture": "synthetic-20k-controlled-branches",
     "nominal_resolution": [256, 256], "wall_seconds": time.perf_counter() - start, "runtime": runtime(), "steps": rows})
