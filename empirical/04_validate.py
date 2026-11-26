"""Check the calibrated model against data it was not fitted to.

  V1  where the extra 2005-06 out-migration went (IRS): each host's share
  V2  where the extra 2006-07 returners came from (IRS): each host's share
  V3  timing of return: extra returns in 2006-07 compared with 2007-08 (IRS)
  V4  extra net domestic migration into each host in the year to June 2006 (Census), metros within 900 km
  V5  crowding (persons per housing unit relative to trend) in the 10 main hosts, July 2006 and 2007 (Census)

Two more comparisons are not fully independent of the calibration: New Orleans occupancy and New Orleans net
domestic migration, 2006-09.

For V1 we also compare with three simple predictions that don't use the model: the national gravity model, a
gravity model fitted to New Orleans' own flows before the storm, and the pre-storm flow shares themselves.

Writes output/validation.json, output/validation_metros.csv and output/traj.npz.
"""
import json

import numpy as np
import pandas as pd

from common import OUT, NOLA, OTHER_STRUCK, APRIL, JULY, nola_series, nola_totals, LANDFALL
from model import Network, simulate, unpack
from calib import top_hosts, sim_from

C = json.load(open(OUT / "calibration.json"))
G = json.load(open(OUT / "gravity.json"))
A = C["A"]
net = Network()
M = net.M
s = net.s
t_eval = np.linspace(0, 4.0, 801)
HOUSTON = "26420"


def shares(x, mask):
    """Turn a vector into shares that sum to 1, ignoring negative values and masked-out places."""
    x = np.clip(np.asarray(x, float), 0, None) * mask
    return x / x.sum()


def r2(obs, pred):
    return float(1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())


def misallocation(obs, pred):
    """Share of people sent to the wrong place (half the summed absolute difference in shares)."""
    return float(0.5 * np.abs(obs - pred).sum())


host_mask = np.ones(net.n)
host_mask[s] = 0
not_struck = host_mask.copy()  # also leave out the other places hit by Katrina and Rita
for code in OTHER_STRUCK:
    if code in net.codes:
        not_struck[net.codes.index(code)] = 0

obs_dest = shares(M.excess.values, host_mask)
obs_ret = shares(M.excess_ret.values, host_mask)

# simple predictions of destinations that don't use the model
g_nola = G["nola_metro"]["o_base"]
dist = np.where(M.dist.values > 0, M.dist.values, np.inf)
static = {
    "national gravity": shares(net.gravity()[s], host_mask),
    "NOLA gravity (pre-disaster)": shares(M.p2005.values ** g_nola["beta"] * dist ** -g_nola["gamma"], host_mask),
    "pre-disaster flow shares": shares(M.o_base.values, host_mask),
}

totals = nola_totals()
in_base = totals.loc[["0304", "0405"], "nola_in_total"].mean()
ret_ratio_obs = (totals.loc["0607", "nola_in_total"] - in_base) / (totals.loc["0708", "nola_in_total"] - in_base)

near = (M.dist.values < 900) & (not_struck > 0)
hosts = top_hosts(net)

res = {"static_benchmarks": {k: dict(dest_R2=r2(obs_dest, v), dest_misalloc=misallocation(obs_dest, v))
                             for k, v in static.items()}}
rows = None

# main model, the same with a pure gravity network, and the version without tracked returners
VARIANTS = {
    "tracked/empirical": (C, "empirical"),
    "tracked/gravity": (C, "gravity"),
    "gradient/empirical": (C["alternative_gradient_return"], "empirical"),
}

for kind, (cal, net_kind) in VARIANTS.items():
    B, C_fit = cal["B"], cal["C"]
    u = unpack(sim_from(net, A, B, C_fit["rho_h"], C["S_med"], net_kind, t_eval=t_eval), net.n)
    t, N, K = u["t"], u["N"], u["K"]
    phi = N / K

    # returns from each host between two times
    if B["ret"] == "tracked":
        cum_ret = u["cum_ret"]

        def returns_between(a, b):
            return np.array([np.interp(b, t, cr) - np.interp(a, t, cr) for cr in cum_ret])
    else:
        W = net.W(B["S_s"], B["S_ret"], C["S_med"], net_kind)
        flow_in = W[:, s][:, None] * N * np.maximum(phi - phi[s], 0)

        def returns_between(a, b):
            k = (t >= a) & (t <= b)
            return np.trapezoid(flow_in[:, k], t[k], axis=1)

    # V1, V2, V3
    dest = shares(N[:, np.searchsorted(t, APRIL["0506"])] - net.K0, host_mask)
    ret = shares(returns_between(APRIL["0506"], APRIL["0607"]), host_mask)
    ret_ratio_model = (returns_between(APRIL["0506"], APRIL["0607"]).sum()
                       / returns_between(APRIL["0607"], APRIL["0708"]).sum())

    # V4: population gained by each host by July 2006, compared with Census net migration
    gain06 = (N[:, np.searchsorted(t, JULY[2006])] - net.K0) * 1e6
    dm = M.dm_excess2006.values

    # V5: crowding in the hosts
    occ = {y: phi[:, np.searchsorted(t, JULY[y])] - 1 for y in (2006, 2007)}

    # New Orleans occupancy and net migration (consistency checks)
    nola = nola_series()
    nola_occ = [float(phi[s, np.searchsorted(t, x)]) for x in nola["t"]]
    nola_dm_model = []
    for y in (2006, 2007, 2008, 2009):
        a, b = max(y - 0.5 - LANDFALL, 0), y + 0.5 - LANDFALL  # Census year runs July to June
        cum_in = np.interp([a, b], t, u["cum_in"])
        cum_out = np.interp([a, b], t, u["cum_out"])
        nola_dm_model.append(float(((cum_in[1] - cum_in[0]) - (cum_out[1] - cum_out[0])) * 1e6))
    nola_dm_obs = [float(M.iloc[s][f"dm{y}"] - M.iloc[s].dm_base) for y in (2006, 2007, 2008, 2009)]

    def houston(v):
        return float(v[net.codes.index(HOUSTON)])

    occ_obs = np.r_[M.iloc[hosts].occ2006, M.iloc[hosts].occ2007]
    occ_model = np.r_[occ[2006][hosts], occ[2007][hosts]]
    res[kind] = dict(
        V1_dest_R2=r2(obs_dest, dest),
        V1_dest_misalloc=misallocation(obs_dest, dest),
        V1_houston=[houston(obs_dest), houston(dest)],
        V2_return_R2=r2(obs_ret, ret),
        V2_return_misalloc=misallocation(obs_ret, ret),
        V2_houston=[houston(obs_ret), houston(ret)],
        V3_return_ratio=[float(ret_ratio_obs), float(ret_ratio_model)],
        V4_dm_corr=float(np.corrcoef(dm[near], gain06[near])[0, 1]),
        V4_dm_slope=float(np.polyfit(gain06[near], dm[near], 1)[0]),
        V4_n=int(near.sum()),
        V5_occ=[dict(code=net.codes[i], name=M.iloc[i]["name"],
                     obs2006=float(M.iloc[i].occ2006), mod2006=float(occ[2006][i]),
                     obs2007=float(M.iloc[i].occ2007), mod2007=float(occ[2007][i])) for i in hosts],
        V5_occ_corr=float(np.corrcoef(occ_obs, occ_model)[0, 1]),
        nola_occ_model=nola_occ,
        nola_occ_obs=[float(x) for x in nola["occ"]],
        nola_dm_model=nola_dm_model,
        nola_dm_obs=nola_dm_obs,
        totals=dict(
            cum_out_M=float(u["cum_out"][-1]),
            cum_in_M=float(u["cum_in"][-1]),
            crowding_loss=float(u["loss"][-1] * 1e6),
            excess_occupancy_person_years=float(u["excess_occ"][-1] * 1e6),
        ),
    )

    if kind == "tracked/empirical":
        rows = pd.DataFrame(dict(
            code=net.codes, name=M.name, dist=M.dist.round(0),
            obs_dest=obs_dest, mod_dest=dest, obs_ret=obs_ret, mod_ret=ret,
            dm_excess2006=dm.round(0), mod_gain06=gain06.round(0), near=near,
            occ2006=M.occ2006.values, mod_occ2006=occ[2006], occ2007=M.occ2007.values, mod_occ2007=occ[2007],
        ))
        np.savez(OUT / "traj.npz", t=t, N_s=N[s], K_s=K[s], cum_in=u["cum_in"], cum_out=u["cum_out"], K0s=net.K0[s])

rows.to_csv(OUT / "validation_metros.csv", index=False)
json.dump(res, open(OUT / "validation.json", "w"), indent=1, default=float)

for k, v in res["static_benchmarks"].items():
    print(f"static {k:28s} dest R2={v['dest_R2']:.2f} misalloc={v['dest_misalloc']:.3f}")
for kind in VARIANTS:
    v = res[kind]
    print(f"\n[{kind}] V1 dest R2={v['V1_dest_R2']:.2f} misalloc={v['V1_dest_misalloc']:.3f} "
          f"Houston obs/mod={np.round(v['V1_houston'], 3).tolist()}")
    print(f"  V2 return R2={v['V2_return_R2']:.2f} misalloc={v['V2_return_misalloc']:.3f} "
          f"Houston={np.round(v['V2_houston'], 3).tolist()}")
    print(f"  V3 return ratio 06-07/07-08 obs={v['V3_return_ratio'][0]:.2f} model={v['V3_return_ratio'][1]:.2f}")
    print(f"  V4 host dm corr={v['V4_dm_corr']:.2f} slope={v['V4_dm_slope']:.2f} (n={v['V4_n']})")
    print(f"  V5 host occupancy corr={v['V5_occ_corr']:.2f}:",
          [(o['name'][:12], round(o['obs2006'], 3), round(o['mod2006'], 3)) for o in v['V5_occ'][:4]])
    print(f"  NOLA occ model {np.round(v['nola_occ_model'], 3).tolist()} obs {np.round(v['nola_occ_obs'], 3).tolist()}")
    print(f"  NOLA dm model {np.round(np.array(v['nola_dm_model']) / 1e3, 1).tolist()}k "
          f"obs {np.round(np.array(v['nola_dm_obs']) / 1e3, 1).tolist()}k")
    print(f"  totals {v['totals']}")
