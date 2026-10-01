import plotly.express as px, plotly.graph_objects as go
def line_power(df):
    f = df.groupby("timestamp").power_kw.sum().reset_index()
    return px.line(f, x="timestamp", y="power_kw", labels={"power_kw": "Factory demand (kW)"}, title="Total factory demand (simulated)")
def machine_bar(by):
    return px.bar(by, x="machine", y="energy_kwh", labels={"energy_kwh": "Energy (kWh)"}, title="Energy by machine (simulated)")
def flag_chart(df, flags, machine):
    g = df[df.machine == machine]; fl = g[flags.reindex(g.index).fillna(False)]
    fig = px.line(g, x="timestamp", y="power_kw", title=f"{machine}: power with flagged points")
    fig.add_trace(go.Scatter(x=fl.timestamp, y=fl.power_kw, mode="markers", marker=dict(color="red", size=7), name="Flagged"))
    return fig
