"""Factory simulator. ALL output is SIMULATED (synthetic) data."""
import numpy as np, pandas as pd
STEP_H = 0.25  # 15 min = 0.25 h
DEFAULT_MACHINES = {
    "Conveyor":  dict(kw=2, idle_pct=0.30, start=6, end=22, units_per_hour=0,   flexible=False, run_frac=0.85),
    "Compressor":dict(kw=5, idle_pct=0.35, start=6, end=22, units_per_hour=0,   flexible=True,  run_frac=0.80),
    "Pump":      dict(kw=3, idle_pct=0.25, start=6, end=22, units_per_hour=0,   flexible=True,  run_frac=0.75),
    "Mixer":     dict(kw=4, idle_pct=0.20, start=6, end=22, units_per_hour=0,   flexible=False, run_frac=0.80),
    "Packaging": dict(kw=3, idle_pct=0.30, start=6, end=22, units_per_hour=120, flexible=False, run_frac=0.85),
}
# Assumptions: Running power ~ N(90% rated, 5%); Idle power = idle_pct * rated (+-8% noise); Off = 0 kW.
# Only Packaging counts finished units (final output). Schedules are daily windows [start,end) in hours.
# Injected anomalies: 'spike' (140-180% rated while Running) and 'after_hours_idle' (3-8 steps of idle draw off-shift).

def simulate(days=7, seed=42, machines=None, spike_prob=0.01, blocks_per_day=0.5, start_date="2025-01-06"):
    rng = np.random.default_rng(seed)
    machines = machines or DEFAULT_MACHINES
    ts = pd.date_range(start_date, periods=int(days) * 96, freq="15min")
    hours = (ts.hour + ts.minute / 60).to_numpy()
    n, frames = len(ts), []
    for name, c in machines.items():
        sched = (hours >= c["start"]) & (hours < c["end"])
        running = sched & (rng.random(n) < c["run_frac"])
        status = np.where(running, "Running", np.where(sched, "Idle", "Off")).astype(object)
        anomaly, atype = np.zeros(n, bool), np.full(n, "", dtype=object)
        off_idx = np.where(~sched)[0]
        if len(off_idx):
            for s in rng.choice(off_idx, size=min(int(days * blocks_per_day), len(off_idx)), replace=False):
                blk = np.zeros(n, bool); blk[s:s + int(rng.integers(3, 9))] = True; blk &= ~sched
                status[blk] = "Idle"; anomaly |= blk; atype[blk] = "after_hours_idle"
        sp = running & (rng.random(n) < spike_prob)
        anomaly |= sp; atype[sp] = "spike"
        power = np.where(status == "Running", c["kw"] * np.clip(rng.normal(0.9, 0.05, n), 0.6, 1.0),
                np.where(status == "Idle", c["kw"] * c["idle_pct"] * np.clip(rng.normal(1, 0.08, n), 0.7, 1.3), 0.0))
        power = np.where(sp, c["kw"] * rng.uniform(1.4, 1.8, n), power)
        prod = np.where(status == "Running", c["units_per_hour"] * STEP_H * rng.uniform(0.9, 1.1, n), 0.0).round()
        frames.append(pd.DataFrame(dict(timestamp=ts, machine=name, rated_kw=c["kw"], scheduled=sched, status=status,
            power_kw=power.round(3), energy_kwh=(power * STEP_H).round(4), production_units=prod,
            injected_anomaly=anomaly, anomaly_type=atype, data_source="SIMULATED")))
    return pd.concat(frames).sort_values(["timestamp", "machine"]).reset_index(drop=True)
