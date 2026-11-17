"""Figures for the paper (run after 01-08): fig_calibration, fig_simulations, fig_campfire (fig_data is made by 01)."""
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from common import OUT, LANDFALL, NOLA, nola_series
from model import capacity_struck

import figstyle as fs
fs.apply()

C = json.load(open(OUT / "calibration.json")); V = json.load(open(OUT / "validation.json")); A = C["A"]
U = np.load(OUT / "uncertainty_bands.npz"); ns = nola_series()
vm = pd.read_csv(OUT / "validation_metros.csv", dtype={"code": str}); vm = vm[vm.code != NOLA]
G = lambda a: a.plot([-1e9, 1e9], [-1e9, 1e9], color="0.6", lw=0.5, zorder=0)        # 1:1 line

# ---------------- Fig. 3: calibration and validation
fig, ax = plt.subplots(2, 3, figsize=(fs.W, 4.5)); ax = ax.ravel()
t = U["t"]; yr = LANDFALL + t; obs_t = np.array(ns["years"]) + 0.5
ax[0].fill_between(yr, U["pop_lo"], U["pop_hi"], color=fs.BAND, lw=0)
ax[0].plot(yr, capacity_struck(t, 1.0, A["d0"], A["dinf"], A["rho_s"]), color=fs.ALT, lw=0.8, ls="--")
ax[0].plot(yr, U["pop_med"], color=fs.MOD, lw=1.0)
ax[0].plot(obs_t, ns["hu_rel"], "s", mfc="white", mec=fs.ALT, mew=0.7, ms=3.2)
ax[0].plot(obs_t, ns["pop_rel"], "o", color=fs.OBS, ms=3.2)
fs.label(ax[0], 2009.75, 0.885, "capacity", color="0.4"); fs.label(ax[0], 2009.75, 0.80, "population")
ax[0].set_xlim(2005, 2010.9); ax[0].set_xticks(range(2005, 2011, 1)); ax[0].set_ylim(0.5, 1.05)
ax[0].set_xlabel("year"); ax[0].set_ylabel("New Orleans metro / 2001–05 trend")
yrs = np.arange(2007, 2010); lab = [f"{y-1}–{str(y)[2:]}" for y in yrs]
ax[1].fill_between(yrs, U["dm_lo"][1:] / 1e3, U["dm_hi"][1:] / 1e3, color=fs.BAND, lw=0)
gm = np.array(V["gradient/empirical"]["nola_dm_model"][1:]) / 1e3; tm = np.array(V["tracked/empirical"]["nola_dm_model"][1:]) / 1e3
ob = np.array(V["tracked/empirical"]["nola_dm_obs"][1:]) / 1e3
ax[1].plot(yrs, gm, color=fs.ALT, lw=0.8, ls="--"); ax[1].plot(yrs, tm, color=fs.MOD, lw=1.0); ax[1].plot(yrs, ob, "o", color=fs.OBS, ms=3.2)
fs.label(ax[1], 2009.08, gm[-1], "no tracking", color="0.4"); fs.label(ax[1], 2009.08, tm[-1] - 2, "tracked")
ax[1].set_xticks(yrs, lab); ax[1].set_xlim(2006.8, 2009.75); ax[1].set_ylim(0, 60)
ax[1].set_xlabel("year (July–June)"); ax[1].set_ylabel("net in-migration to New Orleans (thousands)")
def loglog(a, o, m, xl, yl):
    k = o > 0
    a.plot(m[k], o[k], "o", mfc="white", mec=fs.OBS, mew=0.5, ms=2.6, ls="none"); G(a)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(1e-4, 0.6); a.set_ylim(1e-4, 0.6); a.set_xlabel(xl); a.set_ylabel(yl)
    h = vm.name.str.startswith("Houston").values & k
    a.plot(m[h], o[h], "o", color=fs.OBS, ms=2.6)
    a.annotate("Houston", (m[h][0], o[h][0]), xytext=(-5, 0), textcoords="offset points", ha="right", va="center", fontsize=6.5)
loglog(ax[2], vm.obs_dest.values, vm.mod_dest.values, "model share of host gains", "observed share of evacuees")
loglog(ax[3], vm.obs_ret.values, vm.mod_ret.values, "model share of returns", "observed share of returnees")
nr = vm[vm.near]
ax[4].plot(nr.mod_gain06 / 1e3, nr.dm_excess2006 / 1e3, "o", mfc="white", mec=fs.OBS, mew=0.5, ms=2.8, ls="none")
for nm, off, ha in [("Houston", (4, 0), "left"), ("Baton Rouge", (3, -5), "left"), ("Dallas", (4, -3), "left"), ("Atlanta", (-4, 2), "right")]:
    r = nr[nr.name.str.startswith(nm)].iloc[0]
    ax[4].annotate(nm, (r.mod_gain06 / 1e3, r.dm_excess2006 / 1e3), xytext=off, textcoords="offset points", fontsize=6.5, va="center", ha=ha)
G(ax[4]); ax[4].set_xlim(-10, 90); ax[4].set_ylim(-10, 90)
ax[4].set_xlabel("model population gain (thousands)"); ax[4].set_ylabel("net domestic migration (thousands)")
o5 = V["tracked/empirical"]["V5_occ"]
for y, mk in ((2006, dict(color=fs.OBS)), (2007, dict(mfc="white", mec=fs.OBS, mew=0.6))):
    ax[5].plot([100 * o[f"mod{y}"] for o in o5], [100 * o[f"obs{y}"] for o in o5], "o", ms=3, ls="none", label=f"July {y}", **mk)
for o in o5:
    if o["name"].startswith("Baton"):
        ax[5].annotate("Baton Rouge", (100 * o["mod2006"], 100 * o["obs2006"]), xytext=(-4, 0), textcoords="offset points", fontsize=6.5, va="center", ha="right")
G(ax[5]); ax[5].set_xlim(-0.5, 3.5); ax[5].set_ylim(-0.5, 3.5)
ax[5].set_xlabel("model occupancy above capacity (%)"); ax[5].set_ylabel("occupancy above trend (%)"); ax[5].legend(loc="lower right")
fig.tight_layout(w_pad=2.0, h_pad=1.8); fs.letters(ax, dx=-0.24, dy=1.02)
fig.savefig(OUT / "fig_calibration.png"); fig.savefig(OUT / "fig_calibration.pdf")

# ---------------- Fig. 4: what determines severity
S = json.load(open(OUT / "simulations.json")); sm = pd.read_csv(OUT / "simulations_metros.csv", dtype={"code": str})
fig, ax = plt.subplots(1, 3, figsize=(fs.W, 2.3))
d = sm[sm.scenario == "intensity"]; u = sm[sm.scenario == "uniform"]
ax[0].plot(d.S_i, d.total, "o", mfc="white", mec=fs.OBS, mew=0.5, ms=2.4, ls="none")
n = u[u.code == NOLA].iloc[0]
ax[0].axhspan(u.total.min(), u.total.max(), color=fs.BAND, lw=0, zorder=0)
ax[0].plot([n.S_i], [n.total], "o", color=fs.OBS, ms=3.2)
ax[0].annotate("New Orleans", (n.S_i, n.total), xytext=(5, 5), textcoords="offset points", fontsize=6.5)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel(r"evacuation rate $S_s$ (yr$^{-1}$)")
for axis, ticks in [(ax[0].xaxis, [0.3, 1, 3]), (ax[0].yaxis, [0.2, 0.3, 0.4])]:
    axis.set_major_locator(mpl.ticker.FixedLocator(ticks)); axis.set_minor_formatter(mpl.ticker.NullFormatter())
    axis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax[0].set_ylabel("severity (person-years per resident)")
dd = S["sweeps_nola"]["delta"]
ax[1].plot([x["d0"] for x in dd], [x["total"] for x in dd], color=fs.MOD)
ax[1].plot([x["d0"] for x in dd], [x["hosts"] for x in dd], color=fs.ALT, ls="--", lw=0.8)
ax[1].axvline(A["d0"], color="0.7", lw=0.5, ls=":", zorder=0)
fs.label(ax[1], 0.68, dd[-1]["total"], "total", ha="left"); fs.label(ax[1], 0.68, dd[-1]["hosts"], "hosts", color="0.4", ha="left")
ax[1].set_xlim(0, 0.82); ax[1].set_ylim(0, None); ax[1].set_xlabel(r"capacity lost at landfall $d_0$"); ax[1].set_ylabel("severity (person-years per resident)")
rr = S["sweeps_nola"]["rho_h"]
ax[2].axvspan(*C["C"]["ci95"], color=fs.BAND, lw=0)
ax[2].plot([x["rho_h"] for x in rr], [x["total"] for x in rr], color=fs.MOD)
ax[2].plot([x["rho_h"] for x in rr], [x["hosts"] for x in rr], color=fs.ALT, ls="--", lw=0.8)
fs.label(ax[2], 0.06, rr[0]["total"] + 0.06, "total", ha="left"); fs.label(ax[2], 0.055, rr[0]["hosts"] - 0.3, "hosts", color="0.4", ha="left")
ax[2].set_xscale("log"); ax[2].set_ylim(0, None); ax[2].set_xlabel(r"host capacity response $\rho_h$ (yr$^{-1}$)"); ax[2].set_ylabel("severity (person-years per resident)")
ax[2].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
fig.tight_layout(w_pad=2.0); fs.letters(ax, dx=-0.2)
fig.savefig(OUT / "fig_simulations.png"); fig.savefig(OUT / "fig_simulations.pdf")

# ---------------- Fig. 5: Camp Fire
CF = json.load(open(OUT / "campfire.json")); ct = np.load(OUT / "campfire_traj.npz")
fig, ax = plt.subplots(1, 2, figsize=(fs.W * 0.72, 2.3))
ax[0].plot(ct["t"], ct["K"], color=fs.ALT, lw=0.8, ls="--"); ax[0].plot(ct["t"], ct["N"], color=fs.MOD)
po = CF["population_rel_trend"]["observed"]
ax[0].plot([2018.5, 2019.5, 2020.5], [CF["housing_rel_trend"][k] for k in ("2018", "2019", "2020")], "s", mfc="white", mec=fs.ALT, mew=0.7, ms=3.2)
ax[0].plot([2019.5, 2020.5], [po["2019"], po["2020"]], "o", color=fs.OBS, ms=3.2)
fs.label(ax[0], 2020.95, ct["N"][-1], "model", ha="left"); fs.label(ax[0], 2020.95, ct["K"][-1] - 0.008, "capacity", color="0.4", ha="left")
fs.label(ax[0], 2020.6, po["2020"], "Census", ha="left")
ax[0].set_xlim(2018.25, 2021.6); ax[0].set_xticks([2019, 2020, 2021]); ax[0].set_ylim(0.8, 1.02); ax[0].set_yticks(np.arange(0.8, 1.01, 0.05))
ax[0].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
ax[0].set_xlabel("year"); ax[0].set_ylabel("Butte County / 2014–18 trend")
top = CF["destinations"]["top"]
ax[1].plot([x[2] for x in top], [x[1] for x in top], "o", color=fs.OBS, ms=3, ls="none")
NAMES = {"06103": "Tehama", "06101": "Sutter", "06061": "Placer", "06089": "Shasta", "06115": "Yuba", "06067": "Sacramento"}
LOFF = {"06061": (-4, 2, "right"), "06089": (4, -5, "left"), "06101": (4, 3, "left"), "06067": (0, 6, "center")}
for x in top:
    dx, dy, ha = LOFF.get(x[0], (4, 2, "left"))
    ax[1].annotate(NAMES.get(x[0], x[0]), (x[2], x[1]), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, ha=ha)
G(ax[1]); ax[1].set_xlim(0, 0.13); ax[1].set_ylim(0, 0.13)
ax[1].set_xlabel("pre-disaster flow share"); ax[1].set_ylabel("share of excess out-migration")
fig.tight_layout(w_pad=2.0); fs.letters(ax, dx=-0.2)
fig.savefig(OUT / "fig_campfire.png"); fig.savefig(OUT / "fig_campfire.pdf")
print("figures written")
