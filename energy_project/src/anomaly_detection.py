"""Anomaly detectors. They flag UNUSUAL behaviour, not necessarily faults or waste."""
import pandas as pd

def baseline_detector(df, z_thr=3.0):
    """Robust z-score of power within each (machine, status) group (median/MAD)."""
    p = df.groupby(["machine", "status"]).power_kw
    med = p.transform("median")
    mad = p.transform(lambda x: (x - x.median()).abs().median())
    z = (df.power_kw - med) / (1.4826 * mad.clip(lower=0.01))
    return (z.abs() > z_thr).rename("flag")

def isolation_forest_detector(df, contamination=0.03, seed=42):
    from sklearn.ensemble import IsolationForest
    d = df.sort_values(["machine", "timestamp"]).copy()
    d["rel"] = d.power_kw / d.rated_kw
    d["roll"] = d.groupby("machine").rel.transform(lambda x: x.rolling(3, min_periods=1).mean())
    X = pd.DataFrame(dict(rel=d.rel, status=d.status.map({"Off": 0, "Idle": 1, "Running": 2}),
                          hour=d.timestamp.dt.hour, sched=d.scheduled.astype(int), roll=d.roll))
    pred = IsolationForest(contamination=contamination, random_state=seed).fit_predict(X)
    return pd.Series(pred == -1, index=d.index, name="flag").reindex(df.index)

def evaluate(df, flags):
    t, f = df.injected_anomaly.astype(bool), flags.astype(bool)
    tp, fp, fn = int((f & t).sum()), int((f & ~t).sum()), int((~f & t).sum())
    return dict(true_positives=tp, false_positives=fp, missed=fn,
                precision=tp / (tp + fp) if tp + fp else 0.0, recall=tp / (tp + fn) if tp + fn else 0.0)
