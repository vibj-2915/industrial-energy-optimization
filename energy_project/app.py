import random
import pandas as pd, plotly.express as px, streamlit as st
from src import kpis, waste_detection as wd, anomaly_detection as ad, optimizer as opt, visualizations as viz
from src.simulator import simulate, DEFAULT_MACHINES

st.set_page_config(page_title="Industrial Energy Prototype", page_icon="⚡", layout="wide")
BANNER = "⚠️ All data is SIMULATED and all savings are ESTIMATED from model assumptions. This is a software-based energy simulation and optimization prototype, not a validated digital twin."
if "seed" not in st.session_state: st.session_state.seed = 42
def _regen(): st.session_state.seed = random.randint(0, 99999)

sb = st.sidebar
sb.title("⚡ Energy Prototype")
page = sb.radio("Page", ["Executive Overview", "Factory Simulator", "Energy Analysis", "Waste Detection", "Anomaly Detection",
                         "Optimization Lab", "Data Explorer", "Engineering Concepts"])
days = sb.slider("Simulation days", 1, 30, 7)
sb.number_input("Random seed", 0, 99999, key="seed")
sb.button("🔄 Regenerate simulation data", on_click=_regen)
names = sb.multiselect("Machines", list(DEFAULT_MACHINES), default=list(DEFAULT_MACHINES))
if not names:
    st.warning("Select at least one machine in the sidebar."); st.stop()
spike = sb.slider("Spike probability (%)", 0.0, 5.0, 1.0) / 100
blocks = sb.slider("After-hours anomaly blocks per day", 0.0, 3.0, 0.5)
cfg = {}
for n in names:
    d = DEFAULT_MACHINES[n]
    with sb.expander(f"{n} settings"):
        kw = st.number_input("Rated power (kW)", 0.5, 50.0, float(d["kw"]), key=n + "kw")
        s, e = st.slider("Shift hours", 0, 24, (d["start"], d["end"]), key=n + "s")
        ip = st.slider("Idle power (% of rated)", 5, 60, int(d["idle_pct"] * 100), key=n + "i") / 100
    cfg[n] = {**d, "kw": kw, "start": s, "end": e, "idle_pct": ip}
with sb.expander("Tariff (INR/kWh, assumed)"):
    ps, pe = st.slider("Peak hours", 0, 24, (18, 22))
    rates = dict(standard=st.number_input("Standard", 0.0, 50.0, 7.0), peak=st.number_input("Peak", 0.0, 50.0, 9.5),
                 offpeak=st.number_input("Off-peak", 0.0, 50.0, 5.0), peak_start=ps, peak_end=pe, off_start=22, off_end=6)

@st.cache_data
def load(days, seed, cfg, spike, blocks, rates):
    return kpis.add_price(simulate(days, seed, cfg, spike, blocks), rates)
df = load(days, st.session_state.seed, cfg, spike, blocks, rates)
issues = kpis.validate(df)
if issues: st.error("Data validation issues: " + "; ".join(issues))
d0, d1 = df.timestamp.dt.date.min(), df.timestamp.dt.date.max()
rg = sb.date_input("Date range", (d0, d1), min_value=d0, max_value=d1)
fdf = df[(df.timestamp.dt.date >= rg[0]) & (df.timestamp.dt.date <= rg[1])] if len(rg) == 2 else df
if fdf.empty: st.warning("No data in the selected range."); st.stop()
st.caption(BANNER)
k = kpis.compute_kpis(fdf)
ev = wd.detect_waste(fdf)

def report():
    return (f"SIMULATED DATA REPORT (seed {st.session_state.seed}, {days} days)\n{BANNER}\n\n"
            f"Total energy: {k['total_kwh']:.1f} kWh\nAverage power: {k['avg_kw']:.2f} kW\nPeak demand: {k['peak_kw']:.2f} kW\n"
            f"Idle energy: {k['idle_kwh']:.1f} kWh\nEnergy per unit: {k['kwh_per_unit']:.4f} kWh/unit\n"
            f"Estimated cost: INR {k['cost']:.0f}\nWaste events: {len(ev)}; estimated excess {ev.excess_kwh.sum():.1f} kWh\n")

if page == "Executive Overview":
    st.title("Executive Overview")
    c = st.columns(5)
    c[0].metric("Total energy", f"{k['total_kwh']:,.0f} kWh", help="Sum of power x 0.25 h over all machines")
    c[1].metric("Peak demand", f"{k['peak_kw']:.1f} kW", help="Max of summed machine power at one timestamp")
    c[2].metric("Idle energy", f"{k['idle_kwh']:,.0f} kWh", f"{k['idle_kwh']/k['total_kwh']*100:.1f}% of total", delta_color="off")
    c[3].metric("Energy / unit", f"{k['kwh_per_unit']:.3f} kWh" if k["units"] else "n/a")
    c[4].metric("Est. cost", f"INR {k['cost']:,.0f}")
    st.plotly_chart(viz.line_power(fdf), use_container_width=True)
    st.plotly_chart(viz.machine_bar(kpis.by_machine(fdf)), use_container_width=True)
elif page == "Factory Simulator":
    st.title("Factory Simulator")
    st.info("Use the sidebar for duration, seed, machines, shifts, ratings and idle %. Same seed = identical data.")
    st.markdown("**Assumptions:** Running power ~ N(90% rated, 5%); Idle = idle% x rated (+-8%); Off = 0 kW; only Packaging counts units; "
                "injected anomalies = power spikes (140-180% rated) and after-hours idle draw (3-8 steps).")
    st.dataframe(fdf.groupby(["machine", "status"]).size().unstack(fill_value=0), use_container_width=True)
    st.dataframe(fdf.head(200), use_container_width=True)
elif page == "Energy Analysis":
    st.title("Energy Analysis")
    st.dataframe(kpis.by_machine(fdf).round(2), use_container_width=True)
    f = fdf.assign(hour=fdf.timestamp.dt.hour).groupby(["hour", "machine"]).power_kw.mean().reset_index()
    st.plotly_chart(px.area(f, x="hour", y="power_kw", color="machine", title="Average power by hour of day (kW)"), use_container_width=True)
    with st.expander("Engineering notes: formulas and units", expanded=True):
        st.markdown("- Energy (kWh) = Power (kW) x time (h); a 15-min step = 0.25 h.\n- Average power (kW) = mean of summed machine power.\n"
                    "- Peak demand (kW) = max over timestamps of summed power (a power value, not energy).\n"
                    "- Idle energy = sum of energy where status is Idle.\n- Energy intensity = total kWh / units produced.\n- Cost = sum(kWh x tariff).")
elif page == "Waste Detection":
    st.title("Waste Detection")
    a, b, c = st.columns(3)
    ms = a.slider("Minimum event duration (steps of 15 min)", 1, 16, 2)
    th = b.slider("Off-schedule power threshold (kW)", 0.0, 1.0, 0.05)
    af = c.slider("Avoidable fraction of idle energy", 0.0, 1.0, 0.8)
    ev2 = wd.detect_waste(fdf, ms, th, af)
    if ev2.empty: st.success("No events at these thresholds.")
    else:
        st.metric("Estimated avoidable energy", f"{ev2.excess_kwh.sum():.1f} kWh", f"INR {ev2.est_cost.sum():.0f} (estimate)", delta_color="off")
        st.dataframe(ev2.round(2), use_container_width=True)
        st.plotly_chart(px.bar(ev2.groupby(["machine", "type"]).excess_kwh.sum().reset_index(), x="machine", y="excess_kwh", color="type"), use_container_width=True)
    st.caption("'Confirmed' = rule violation in simulated data (power while off-schedule). 'Suspected' = idle in schedule; avoidable share is an assumption.")
elif page == "Anomaly Detection":
    st.title("Anomaly Detection")
    m = st.radio("Method", ["Baseline robust z-score", "Isolation Forest"], horizontal=True)
    flags = ad.baseline_detector(fdf, st.slider("z threshold", 2.0, 6.0, 3.0)) if m.startswith("Baseline") else \
            ad.isolation_forest_detector(fdf, st.slider("Contamination", 0.005, 0.1, 0.03), st.session_state.seed)
    r = ad.evaluate(fdf, flags)
    c = st.columns(5)
    for col, (lab, key) in zip(c, [("Precision", "precision"), ("Recall", "recall"), ("True positives", "true_positives"),
                                   ("False positives", "false_positives"), ("Missed", "missed")]):
        col.metric(lab, f"{r[key]:.2f}" if isinstance(r[key], float) else r[key])
    st.plotly_chart(viz.flag_chart(fdf, flags, st.selectbox("Machine", sorted(fdf.machine.unique()))), use_container_width=True)
    st.info("Anomaly detection finds UNUSUAL behaviour; it does not prove an equipment fault or energy waste. Scores use injected simulation labels.")
elif page == "Optimization Lab":
    st.title("Optimization Lab")
    a, b, c = st.columns(3)
    flex, stop, cut = a.checkbox("Shift flexible loads to cheap tariff steps", True), b.checkbox("Reduce idle draw (auto-standby)", True), c.slider("Idle reduction", 0.0, 1.0, 0.7)
    p = opt.plan(cfg, rates, cut, flex, stop); s = opt.summarize(p)
    c = st.columns(3)
    c[0].metric("Energy / day", f"{s['opt_kwh']:.1f} kWh", f"-{s['kwh_red_pct']:.1f}% vs {s['base_kwh']:.1f} (est.)", delta_color="off")
    c[1].metric("Cost / day", f"INR {s['opt_cost']:.0f}", f"-{s['cost_red_pct']:.1f}% vs {s['base_cost']:.0f} (est.)", delta_color="off")
    c[2].metric("Peak demand", f"{s['opt_peak']:.1f} kW", f"{-s['peak_red_pct']:.1f}% vs {s['base_peak']:.1f}", delta_color="off")
    t = p.groupby("hour")[["base_kw", "opt_kw"]].sum().reset_index().melt("hour", var_name="plan", value_name="kW")
    st.plotly_chart(px.line(t, x="hour", y="kW", color="plan", title="One-day demand: baseline vs optimized (model)"), use_container_width=True)
    v = opt.check_constraints(p)
    st.success("All constraints satisfied.") if not v else st.error("; ".join(v))
    st.markdown("**Constraints:** same running steps per machine (production preserved), runs only inside shift window (deadline), one plan per day. "
                "**Method:** flexible machines run in the cheapest tariff steps (exact for this structure); idle draw reduced by the chosen %. "
                "Savings exist only relative to this modeled baseline (machines run from shift start; idle left on).")
elif page == "Data Explorer":
    st.title("Data Explorer & Downloads")
    sel = st.multiselect("Filter machines", sorted(fdf.machine.unique()), default=sorted(fdf.machine.unique()))
    view = fdf[fdf.machine.isin(sel)]
    st.dataframe(view.head(5000), use_container_width=True)
    st.download_button("⬇ Simulated data (CSV)", view.to_csv(index=False), "simulated_energy_data.csv")
    st.download_button("⬇ Waste events (CSV)", ev.to_csv(index=False), "waste_events.csv")
    st.download_button("⬇ Analysis report (TXT)", report(), "energy_report.txt")
else:
    st.title("Engineering Concepts & Limitations")
    st.markdown("""**kW vs kWh:** kW is instantaneous power; kWh is energy over time. **Peak demand** drives demand charges.
**Idle energy:** machines on but not producing still draw a fraction of rated power.
**Isolation Forest:** isolates points via random splits; rare points need fewer splits.

**Limitations:** synthetic data; simplified power model (no power factor, harmonics, three-phase detail, ramp-up, thermal effects);
single-day greedy planner without inter-machine process dependencies; tariff and idle assumptions are illustrative;
detectors are scored against labels the simulator itself injected. Not validated against any real plant.""")
