"""Schedule optimizer on a one-day MODEL plan. Savings are ESTIMATES from model assumptions only.
Method: for each flexible machine, the same number of running steps is placed in the cheapest tariff steps inside its
shift window (sort by price). For this structure (fixed run steps, no contiguity limit) that greedy choice is exactly
optimal, so no external solver is needed. Idle draw is reduced by idle_cut (e.g. auto-standby) for all machines."""
import numpy as np, pandas as pd
from .simulator import STEP_H
from .kpis import tou_price

def plan(machines, rates, idle_cut=0.7, flex=True, idle_stop=True):
    hours = np.arange(96) * STEP_H
    price, rows = tou_price(hours, rates), []
    for name, c in machines.items():
        w = np.where((hours >= c["start"]) & (hours < c["end"]))[0]
        n = int(round(c["run_frac"] * len(w)))
        base = np.zeros(96, bool); base[w[:n]] = True                 # baseline: run as soon as shift starts
        opt = base.copy()
        if flex and c.get("flexible"):
            opt = np.zeros(96, bool); opt[w[np.lexsort((w, price[w]))][:n]] = True
        win = np.zeros(96, bool); win[w] = True
        cut = idle_cut if idle_stop else 0.0
        idle_kw = c["kw"] * c["idle_pct"]
        rows.append(pd.DataFrame(dict(step=np.arange(96), hour=hours, machine=name, price=price, in_window=win,
            base_run=base, opt_run=opt,
            base_kw=np.where(base, c["kw"] * 0.9, np.where(win, idle_kw, 0.0)),
            opt_kw=np.where(opt, c["kw"] * 0.9, np.where(win & ~opt, idle_kw * (1 - cut), 0.0)))))
    return pd.concat(rows, ignore_index=True)

def summarize(p):
    out = {}
    for k in ("base", "opt"):
        out[f"{k}_kwh"] = p[f"{k}_kw"].sum() * STEP_H
        out[f"{k}_cost"] = (p[f"{k}_kw"] * p.price).sum() * STEP_H
        out[f"{k}_peak"] = p.groupby("step")[f"{k}_kw"].sum().max()
    pct = lambda b, o: (b - o) / b * 100 if b > 0 else 0.0
    out.update(kwh_red_pct=pct(out["base_kwh"], out["opt_kwh"]), cost_red_pct=pct(out["base_cost"], out["opt_cost"]),
               peak_red_pct=pct(out["base_peak"], out["opt_peak"]))
    return out

def check_constraints(p):
    """Returns a list of violations (empty = all constraints satisfied)."""
    v = []
    for m, g in p.groupby("machine"):
        if g.base_run.sum() != g.opt_run.sum(): v.append(f"{m}: run duration (production) changed")
        if (g.opt_run & ~g.in_window).any(): v.append(f"{m}: runs outside shift window/deadline")
    return v
