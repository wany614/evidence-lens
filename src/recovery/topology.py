"""Bounded refinement with verified gsplat 1.5.3 Adam state surgery."""
from __future__ import annotations

import torch
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import duplicate, remove, reset_opa, split

from evidence_lens._verify import require


def assert_adam(params, optimizers, step):
    for key, p in params.items():
        opt = optimizers[key]
        require(len(opt.param_groups) == 1 and len(opt.param_groups[0]["params"]) == 1
                and opt.param_groups[0]["params"][0] is p, "Adam binding differs for " + key)
        st = opt.state.get(p, {})
        if step == 0:
            require(not st, "Unexpected Adam state before update one")
        else:
            require(len(opt.state) == 1 and set(st) == {"step", "exp_avg", "exp_avg_sq"}
                    and int(st["step"].item()) == step
                    and all(st[k].shape == p.shape and st[k].dtype == p.dtype and st[k].device == p.device
                            for k in ("exp_avg", "exp_avg_sq")), "Adam state differs for " + key)


@torch.no_grad()
def mutate(kind, params, opts, state, mask=None, value=None):
    names = ["opacities"] if kind == "reset" else list(params)
    before = {n: dict(opts[n].state[params[n]]) for n in names}
    total = len(params["means"])
    if kind == "reset":
        changed = int((params["opacities"].sigmoid() > value).sum())
        reset_opa(params, opts, state, value=value)
        selected = total
    else:
        selected = int(mask.sum())
        changed = selected
        if kind == "clone":
            duplicate(params, opts, state, mask=mask)
        elif kind == "split":
            split(params, opts, state, mask=mask, revised_opacity=False)
        else:
            require(kind == "prune", "Unknown topology operation")
            remove(params, opts, state, mask=mask)
    for n in names:
        after = opts[n].state[params[n]]
        require(torch.equal(before[n]["step"], after["step"]), "Topology changed Adam step")
        for k in ("exp_avg", "exp_avg_sq"):
            old, new = before[n][k], after[k]
            if kind == "reset":
                ok = not bool(torch.count_nonzero(new))
            else:
                surviving = old if kind == "clone" else old[~mask]
                tail = new[len(surviving):]
                created = selected if kind == "clone" else 2 * selected if kind == "split" else 0
                ok = torch.equal(new[:len(surviving)], surviving) and len(tail) == created and not bool(torch.count_nonzero(tail))
            require(ok, f"Incorrect {kind} Adam moments for {n}/{k}")
    return {"operation": kind, "selected": selected, "affected_values": changed, "before": total,
            "after": len(params["means"]), "adam_moments_checked": True}


class BoundedRefinement:
    def __init__(self, config, scale):
        self.config = config
        # This installed, pinned implementation supplies projection-gradient normalization.
        self.base = DefaultStrategy(absgrad=False, refine_scale2d_stop_iter=0)
        self.state = self.base.initialize_state(scene_scale=scale)

    def retain(self, params, opts, step, info):
        if step < self.config["stop"]:
            self.base.step_pre_backward(params, opts, self.state, step, info)

    @torch.no_grad()
    def observe(self, params, step, info):
        if step < self.config["stop"]:
            self.base._update_state(params, self.state, info, packed=False)

    @torch.no_grad()
    def refine(self, params, opts, step):
        t, state = self.config, self.state
        events = []
        if not t["start"] <= step < t["stop"]:
            return events
        if (step - t["start"]) % t["every"] == 0:
            opacity = params["opacities"].sigmoid()
            drop = opacity < t["prune_opacity"]
            if step >= t["prune_scale_from"] and t["prune_scale_fraction"] > 0:
                drop |= params["scales"].exp().amax(dim=1) > t["prune_scale_fraction"] * state["scene_scale"]
            if bool(drop.all()):
                drop[torch.argmax(opacity)] = False
            if bool(drop.any()):
                events.append(mutate("prune", params, opts, state, drop))
            avg = state["grad2d"] / state["count"].clamp_min(1)
            require(bool(torch.isfinite(avg).all()), "Nonfinite projected-gradient statistic")
            candidates = torch.where((state["count"] > 0) & (avg > t["grow_grad2d"]))[0]
            room = t["cap"] - len(params["means"])
            require(room >= 0, "Point cap already exceeded")
            # Descending score, then ascending pre-mutation row index on ties.
            order = torch.argsort(avg[candidates], descending=True, stable=True)
            chosen = candidates[order[:room]]
            grow = torch.zeros_like(avg, dtype=torch.bool)
            grow[chosen] = True
            small = params["scales"].exp().amax(dim=1) <= t["clone_scale_fraction"] * state["scene_scale"]
            clones, splits = grow & small, grow & ~small
            events.append({"operation": "growth_budget", "candidates": len(candidates), "available": room,
                           "admitted": len(chosen), "blocked": len(candidates) - len(chosen)})
            if bool(clones.any()):
                events.append(mutate("clone", params, opts, state, clones))
                splits = torch.cat((splits, splits.new_zeros(int(clones.sum()))))
            if bool(splits.any()):
                events.append(mutate("split", params, opts, state, splits))
            require(len(params["means"]) <= t["cap"], "Point cap exceeded during refinement")
            state["count"].zero_()
            state["grad2d"].zero_()
        if step in t["reset_steps"]:
            events.append(mutate("reset", params, opts, state, value=t["reset_opacity"]))
        if events:
            assert_adam(params, opts, step)
        return events
