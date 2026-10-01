"""KPIs, tariff and validation. energy_kWh = power_kW x time_h."""
import numpy as np, pandas as pd
from .simulator import STEP_H
DEFAULT_RATES = dict(standard=7.0, peak=9.5, offpeak=5.0, peak_start=18, peak_end=22, off_start=22, off_end=6)  # INR/kWh, ASSUMED
VALID_STATUS = {"Running", "Idle", "Off"}

def energy_kwh(power_kw, hours=STEP_H):
    return power_kw * hours

def tou_price(hours, r=DEFAULT_RATES):
    h = np.asarray(hours, float)
    p = np.full(h.shape, r["standard"], float)
    p[(h >= r["peak_start"]) & (h < r["peak_end"])] = r["peak"]
    p[(h >= r["off_start"]) | (h < r["off_end"])] = r["offpeak"]
    return p

def add_price(df, rates=DEFAULT_RATES):
    d = df.copy()
    d["price"] = tou_price((d.timestamp.dt.hour + d.timestamp.dt.minute / 60).to_numpy(), rates)
    d["cost"] = d.energy_kwh * d.price
    return d

def compute_kpis(df):
    if "cost" not in df: df = add_price(df)
    factory = df.groupby("timestamp").power_kw.sum()          # coincident demand
    total, units = df.energy_kwh.sum(), df.production_units.sum()
    return dict(total_kwh=total, avg_kw=factory.mean(), peak_kw=factory.max(),
                idle_kwh=df.loc[df.status == "Idle", "energy_kwh"].sum(), units=units,
                kwh_per_unit=total / units if units > 0 else float("nan"), cost=df.cost.sum())

def by_machine(df):
    if "cost" not in df: df = add_price(df)
    out = df.groupby("machine").agg(energy_kwh=("energy_kwh", "sum"), avg_kw=("power_kw", "mean"),
                                    peak_kw=("power_kw", "max"), cost=("cost", "sum"))
    out["idle_kwh"] = df[df.status == "Idle"].groupby("machine").energy_kwh.sum()
    return out.fillna(0).reset_index()

def validate(df):
    issues = []
    if df.isna().any().any(): issues.append("Missing values found")
    if not set(df.status.unique()) <= VALID_STATUS: issues.append("Invalid status values")
    if (df.power_kw < 0).any(): issues.append("Negative power found")
    for m, g in df.groupby("machine"):
        if not (g.timestamp.sort_values().diff().dropna() == pd.Timedelta("15min")).all():
            issues.append(f"Inconsistent timestamps for {m}")
    return issues
