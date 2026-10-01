from src import kpis, optimizer as opt, waste_detection as wd
from src.simulator import simulate, DEFAULT_MACHINES

def test_energy_formula():
    assert kpis.energy_kwh(4.0) == 1.0  # 4 kW x 0.25 h

def test_deterministic_seed():
    a, b, c = simulate(2, 1), simulate(2, 1), simulate(2, 2)
    assert a.equals(b) and not a.power_kw.equals(c.power_kw)

def test_kpis():
    df = simulate(3, 5); k = kpis.compute_kpis(df)
    assert abs(k["total_kwh"] - df.power_kw.sum() * 0.25) < 0.05
    assert k["peak_kw"] >= k["avg_kw"] and 0 <= k["idle_kwh"] <= k["total_kwh"]

def test_offschedule_waste():
    df = simulate(7, 3, blocks_per_day=1.0)
    ev = wd.detect_waste(df, min_steps=1)
    conf = ev[ev.type == "Confirmed rule violation"].excess_kwh.sum()
    expected = df[(~df.scheduled) & (df.power_kw > 0.05)].energy_kwh.sum()
    assert abs(conf - expected) < 1e-6 and conf > 0

def test_optimizer_constraints():
    p = opt.plan(DEFAULT_MACHINES, kpis.DEFAULT_RATES)
    s = opt.summarize(p)
    assert opt.check_constraints(p) == [] and s["opt_kwh"] <= s["base_kwh"] and s["opt_cost"] <= s["base_cost"]

def test_validation_catches_negative():
    df = simulate(1, 1); df.loc[0, "power_kw"] = -1
    assert "Negative power found" in kpis.validate(df)
