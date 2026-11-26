"""Calibrate the network model to Katrina (the fitting itself is in calib.py).

Before fitting, we check whether evacuees moved on from one host city to another. If they did, we would need
host-to-host coupling (S_med > 0). The IRS data say they didn't, so S_med is set to 0.

Writes output/onward_evidence.json and output/calibration.json.
"""
import json

import numpy as np
import statsmodels.api as sm

from common import OUT, NOLA, OTHER_STRUCK, BASE_YEARS, metros, metro_flows
from calib import calibrate

# ---- Did host cities that received many evacuees send extra people on to other hosts? ----

m = metros()
codes = set(m.index)
struck = {NOLA} | set(OTHER_STRUCK)  # leave out the places hit by Katrina and Rita themselves

flows = metro_flows().reset_index()
flows = flows[flows.o.isin(codes) & flows.d.isin(codes)]
base = flows[BASE_YEARS].mean(axis=1)  # normal flow for each pair
received = (m.o0506 - m.o_base).clip(lower=0)  # extra arrivals from New Orleans in 2005-06

host_flows = flows[~flows.o.isin(struck) & ~flows.d.isin(struck)]
hosts = m.drop(index=[c for c in struck if c in m.index])

onward = {}
for y in ["0506", "0607", "0708"]:
    extra = host_flows[y] - base[host_flows.index]  # flow above normal between two hosts
    extra_out = extra.groupby(host_flows.o).sum().reindex(m.index).fillna(0)
    extra_in = extra.groupby(host_flows.d).sum().reindex(m.index).fillna(0)
    net_out = extra_out - extra_in

    # regress each host's net extra outflow on how many evacuees it received
    fit = sm.OLS(net_out[hosts.index], sm.add_constant(received[hosts.index])).fit(cov_type="HC1")
    slope, se = float(fit.params.iloc[1]), float(fit.bse.iloc[1])
    onward[y] = dict(slope=slope, se=se, ci95=[slope - 1.96 * se, slope + 1.96 * se], n=int(len(hosts)))
    print(f"onward {y}: net extra outflow to other hosts per evacuee received = {slope:.3f} ± {se:.3f}")

json.dump(onward, open(OUT / "onward_evidence.json", "w"), indent=1)

# ---- Calibration ----

res = calibrate()  # main model: displaced residents tracked
alt = calibrate(ret="gradient")  # for comparison: return only through the occupancy gradient
res["alternative_gradient_return"] = alt

A, B, C = res["A"], res["B"], res["C"]
print(f"A: d0={A['d0']:.3f} dinf={A['dinf']:.3f} rho_s={A['rho_s']:.3f} "
      f"(jackknife SE {np.round(A['se'], 3).tolist()}); resid {np.round(A['resid'], 4).tolist()}")
for label, b in [("B (tracked)", B), ("B (gradient)", alt["B"])]:
    fitted = "  ".join(f"{k}={b[k]:.3f} {np.round(ci, 3).tolist()}" for k, ci in zip(b["names"], b["ci95"]))
    print(f"{label}: {fitted}; pop model {np.round(b['pop_model'], 4).tolist()} obs {np.round(b['pop_obs'], 4).tolist()}; "
          f"share {b['share_model']:.3f} vs {b['share_obs']:.3f}; SSE_pop={b['sse_pop']:.2e}")
print(f"C: rho_h={C['rho_h']:.3f} {np.round(C['ci95'], 3).tolist()}")

json.dump(res, open(OUT / "calibration.json", "w"), indent=1, default=float)
