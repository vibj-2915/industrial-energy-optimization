"""Rule-based waste detection on SIMULATED data. Excess energy is an ESTIMATE."""
import pandas as pd
from .kpis import add_price
COLS = ["machine", "start", "end", "steps", "type", "reason", "excess_kwh", "est_cost"]

def detect_waste(df, min_steps=2, off_threshold_kw=0.05, avoidable_frac=0.8):
    if "price" not in df: df = add_price(df)
    rows = []
    for m, g in df.sort_values("timestamp").groupby("machine"):
        g = g.reset_index(drop=True)
        rules = [("Confirmed rule violation", (~g.scheduled) & (g.power_kw > off_threshold_kw), 1.0,
                  "Drawing power outside scheduled period"),
                 ("Suspected waste", g.scheduled & (g.status == "Idle"), avoidable_frac,
                  "Idle during schedule; only part is avoidable")]
        for kind, mask, frac, reason in rules:
            run = (mask != mask.shift()).cumsum()
            for _, r in g[mask].groupby(run[mask]):
                if len(r) < min_steps: continue
                rows.append(dict(machine=m, start=r.timestamp.iloc[0], end=r.timestamp.iloc[-1], steps=len(r), type=kind,
                    reason=reason, excess_kwh=r.energy_kwh.sum() * frac, est_cost=(r.energy_kwh * r.price).sum() * frac))
    return pd.DataFrame(rows, columns=COLS).sort_values("start").reset_index(drop=True)
