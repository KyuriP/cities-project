"""Gravity model estimates, used to fill in the migration flows the IRS hides.

Three parts:
  1. Network-wide: all ordered pairs of the model's metros, normal flows before the storm (average of 2003-04
     and 2004-05). This gives alpha (origin size), beta (destination size) and gamma (distance), which the model
     uses to fill the hidden flows.
  2. New Orleans only, by year and type of flow, at metro and county level. This is descriptive: it shows whether
     moves after the storm followed the same pattern as normal migration.
  3. Bootstrap over metros, for the uncertainty in part 1.

Fitted by Poisson regression (PPML) with robust standard errors, which handles the many zero flows.
Writes output/gravity.json.
"""
import json

import numpy as np
import statsmodels.api as sm

from common import OUT, NOLA, BASE_YEARS, metros, metro_flows, counties, county_flows, hav, NOLA_PARISHES

rng = np.random.default_rng(20051829)

# ---- 1. all metro pairs ----

m = metros()
codes = m.index.tolist()
idx = {c: i for i, c in enumerate(codes)}
n = len(codes)

flows = metro_flows().reset_index()
flows = flows[flows.o.isin(codes) & flows.d.isin(codes)]
Y = np.zeros((n, n))  # Y[i, j] = average yearly flow from i to j before the storm (0 if hidden)
for _, row in flows.iterrows():
    Y[idx[row.o], idx[row.d]] = np.mean([row[y] for y in BASE_YEARS])

log_pop = np.log(m.p2005.values)
lat, lon = m.lat.values, m.lon.values
D = hav(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
I, J = np.where(~np.eye(n, dtype=bool))  # every ordered pair i != j


def fit_pairs(I, J):
    """Poisson regression of flow on log origin size, log destination size and minus log distance."""
    X = sm.add_constant(np.column_stack([log_pop[I], log_pop[J], -np.log(D[I, J])]))
    fit = sm.GLM(Y[I, J], X, family=sm.families.Poisson()).fit(cov_type="HC1")
    return np.asarray(fit.params), np.asarray(fit.bse), 1 - fit.deviance / fit.null_deviance


params, se, r2 = fit_pairs(I, J)
res = {
    "network": dict(
        alpha=params[1], alpha_se=se[1],
        beta=params[2], beta_se=se[2],
        gamma=params[3], gamma_se=se[3],
        pseudoR2=r2,
        n_pairs=int(len(I)),
        n_metros=n,
        share_pairs_positive=float((Y[I, J] > 0).mean()),
    )
}

# ---- 3. bootstrap: resample metros, keep all pairs among the resampled set ----

boots = []
for b in range(200):
    sample = rng.choice(n, n, replace=True)
    ii, jj = np.meshgrid(sample, sample, indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    off_diag = ii != jj
    try:
        boots.append(fit_pairs(ii[off_diag], jj[off_diag])[0][1:])
    except Exception:  # a resample can occasionally fail to converge; skip it
        pass
boots = np.array(boots)
res["network"]["bootstrap"] = dict(
    n=int(len(boots)),
    alpha_ci=np.percentile(boots[:, 0], [2.5, 97.5]).tolist(),
    beta_ci=np.percentile(boots[:, 1], [2.5, 97.5]).tolist(),
    gamma_ci=np.percentile(boots[:, 2], [2.5, 97.5]).tolist(),
    alpha_sd=float(boots[:, 0].std()),
    beta_sd=float(boots[:, 1].std()),
    gamma_sd=float(boots[:, 2].std()),
)


# ---- 2. New Orleans only ----

def ppml(y, pop, dist):
    """Gravity fit for flows out of (or into) one place: only destination size and distance."""
    X = sm.add_constant(np.column_stack([np.log(pop), np.log(dist)]))
    fit = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type="HC1")
    q, s = np.asarray(fit.params), np.asarray(fit.bse)
    return dict(beta=q[1], beta_se=s[1], gamma=-q[2], gamma_se=s[2], N=float(np.sum(y)),
                pseudoR2=1 - fit.deviance / fit.null_deviance)


FLOW_TYPES = ["o_base", "o0506", "excess", "o0607", "i_base", "i0607"]  # o = out of New Orleans, i = into it

# metro level
hosts = m.drop(index=NOLA)
res["nola_metro"] = {y: ppml(hosts[y].values, hosts.p2005.values, hosts.dist.values) for y in FLOW_TYPES}

# county level: every county with any flow to or from New Orleans, plus every county over 100k people
c = counties()
F = county_flows()
parishes = c[c.fips.isin(NOLA_PARISHES)]
lat0 = np.average(parishes.lat, weights=parishes.p2005)
lon0 = np.average(parishes.lon, weights=parishes.p2005)
cc = c[~c.fips.isin(NOLA_PARISHES)].copy()
cc["dist"] = hav(lat0, lon0, cc.lat, cc.lon)
for y in ["0304", "0405", "0506", "0607"]:
    f = F[F.year == y]
    cc[f"o{y}"] = f[f.orig.isin(NOLA_PARISHES)].groupby("dest").exemptions.sum().reindex(cc.fips).fillna(0).values
    cc[f"i{y}"] = f[f.dest.isin(NOLA_PARISHES)].groupby("orig").exemptions.sum().reindex(cc.fips).fillna(0).values
cc["o_base"] = (cc.o0304 + cc.o0405) / 2
cc["i_base"] = (cc.i0304 + cc.i0405) / 2
cc["excess"] = (cc.o0506 - cc.o_base).clip(lower=0)

any_flow = cc[["o0304", "o0405", "o0506", "o0607", "i0304", "i0405", "i0607"]].sum(1) > 0
sample = cc[any_flow | (cc.p2005 >= 100_000)]
res["nola_county"] = {y: ppml(sample[y].values, sample.p2005.values, sample.dist.values) for y in FLOW_TYPES}
res["nola_county"]["n_counties"] = int(len(sample))

json.dump(res, open(OUT / "gravity.json", "w"), indent=1, default=float)

g = res["network"]
print(f"network ({g['n_pairs']} pairs, {g['share_pairs_positive']:.0%} non-zero): "
      f"alpha={g['alpha']:.3f}±{g['alpha_se']:.3f} beta={g['beta']:.3f}±{g['beta_se']:.3f} "
      f"gamma={g['gamma']:.3f}±{g['gamma_se']:.3f} R2={g['pseudoR2']:.2f}; bootstrap CIs {g['bootstrap']}")
for level in ["nola_metro", "nola_county"]:
    for k, v in res[level].items():
        if isinstance(v, dict):
            print(f"{level:11s} {k:7s} beta={v['beta']:.2f}±{v['beta_se']:.2f} gamma={v['gamma']:.2f}±{v['gamma_se']:.2f} "
                  f"N={v['N']:.0f} R2={v['pseudoR2']:.2f}")
