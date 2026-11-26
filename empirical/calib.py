"""Fitting the network model to Katrina (used by 03_calibrate.py and 06_sensitivity.py).

Each group of parameters is fitted to one data set, and everything else is kept for checking the fit:

  A  d_0, d_inf, rho_s  New Orleans housing units, July 2006-09, relative to the 2001-05 trend
  B  S_s, kappa, lam    New Orleans population, July 2006-09, relative to trend, plus the share of the two-year
                        extra out-migration (IRS) that falls in the 2005-06 filing year.
                        S_s is the evacuation rate, kappa the return rate when the city has room, and lam the rate
                        at which displaced residents settle for good where they are.
                        With ret="gradient" we fit S_s and S_ret instead, without tracking the displaced.
  C  rho_h              Housing built above trend in the 10 metros that received most evacuees, July 2006-07,
                        converted to people with 2005 persons per unit.

Host-to-host coupling S_med is 0, because the IRS data show no onward moves between hosts (see 03_calibrate.py).
We don't assume noise levels: each block estimates its own residual variance, and standard errors come from
Cov = s^2 (J'J)^-1 on the scale we fit on (logs).
"""
import numpy as np
from scipy.optimize import least_squares

from common import nola_series, nola_totals, APRIL, OTHER_STRUCK, NOLA, BASE_YEARS, TOP_K
from model import Network, simulate, unpack, capacity_struck

A_BOUNDS = ([0, 0, 1e-3], [0.9, 0.5, 10])


def fit_A():
    """Housing loss and rebuilding in New Orleans."""
    nola = nola_series()
    t = np.array(nola["t"])
    y = np.array(nola["hu_rel"])

    def res(p):
        d0, dinf, rho = p
        return capacity_struck(t, 1.0, d0, dinf, rho) - y

    sol = least_squares(res, [0.4, 0.12, 0.6], bounds=A_BOUNDS)

    # Only 4 points for 3 parameters, so we use leave-one-year-out (jackknife) spread as the uncertainty.
    jk = []
    for k in range(len(t)):
        keep = np.arange(len(t)) != k
        r = least_squares(lambda p: capacity_struck(t[keep], 1.0, *p) - y[keep], sol.x, bounds=A_BOUNDS)
        jk.append(r.x)
    jk = np.array(jk)
    n = len(t)
    se = np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0))
    return dict(d0=sol.x[0], dinf=sol.x[1], rho_s=sol.x[2], se=se.tolist(), resid=sol.fun.tolist(),
                jackknife=jk.tolist())


def irs_first_share():
    """Share of the two-year extra out-migration from New Orleans that happened in filing year 2005-06."""
    totals = nola_totals()
    base = totals.loc[BASE_YEARS, "nola_out_total"].mean()
    extra1 = totals.loc["0506", "nola_out_total"] - base
    extra2 = totals.loc["0607", "nola_out_total"] - base
    return extra1 / (extra1 + extra2)


def top_hosts(net, k=None):
    """The k metros that received the most extra movers from New Orleans (other struck metros excluded)."""
    k = TOP_K if k is None else k
    others = [NOLA] + [c for c in OTHER_STRUCK if c in net.M.index]
    m = net.M.drop(index=others)
    best = m.sort_values("excess", ascending=False).index[:k]
    return [net.codes.index(c) for c in best]


def model_B(net, A, p, rho_h, S_med=0.0, kind="empirical", ret="tracked"):
    """Model population of New Orleans at the July dates, and the model's first-year share of out-migration."""
    nola = nola_series()
    t_eval = np.sort(np.r_[nola["t"], APRIL["0506"], APRIL["0607"]])
    if ret == "tracked":
        S_s, kappa, lam = p
        W = net.W(S_s, 1.0, S_med, kind)
        sol = simulate(net, W, A["d0"], A["dinf"], A["rho_s"], rho_h, t_eval=t_eval, kappa=kappa, lam=lam)
    else:
        S_s, S_ret = p
        W = net.W(S_s, S_ret, S_med, kind)
        sol = simulate(net, W, A["d0"], A["dinf"], A["rho_s"], rho_h, t_eval=t_eval)
    u = unpack(sol, net.n)
    pop = np.array([u["N"][net.s, np.searchsorted(t_eval, t)] for t in nola["t"]]) / net.K0[net.s]
    cum_out = np.interp([APRIL["0506"], APRIL["0607"]], u["t"], u["cum_out"])
    return pop, cum_out[0] / cum_out[1], u, t_eval


def fit_B(net, A, rho_h, S_med=0.0, kind="empirical", ret="tracked", x0=None):
    """Departure, return and settlement, fitted to New Orleans population and the IRS timing share."""
    nola = nola_series()
    y = np.array(nola["pop_rel"])
    f = irs_first_share()
    names = ["S_s", "kappa", "lam"] if ret == "tracked" else ["S_s", "S_ret"]
    if not x0:
        x0 = (1.6, 10.0, 0.5) if ret == "tracked" else (4.0, 1.0)

    def parts(log_p):
        pop, share, _, _ = model_B(net, A, np.exp(log_p), rho_h, S_med, kind, ret)
        return pop - y, np.array([share - f])

    # The population and timing residuals get their own noise levels, estimated from the fit (we iterate once).
    sig = np.array([0.01, 0.02])
    for _ in range(2):
        def weighted(log_p):
            return np.concatenate([parts(log_p)[0] / sig[0], parts(log_p)[1] / sig[1]])

        sol = least_squares(weighted, np.log(x0), diff_step=1e-3)
        res_pop, res_share = parts(sol.x)
        sig = np.array([max(np.sqrt(np.mean(res_pop**2)), 1e-3), max(abs(res_share[0]), 0.005)])

    J = sol.jac
    cov = np.linalg.pinv(J.T @ J)
    se_log = np.sqrt(np.diag(cov))
    ci95 = [[float(np.exp(sol.x[i] - 1.96 * se_log[i])), float(np.exp(sol.x[i] + 1.96 * se_log[i]))]
            for i in range(len(names))]

    out = {k: float(np.exp(v)) for k, v in zip(names, sol.x)}
    out.update(
        names=names,
        ret=ret,
        se_log=se_log.tolist(),
        cov_log=cov.tolist(),
        x_log=sol.x.tolist(),
        ci95=ci95,
        pop_model=(res_pop + y).tolist(),
        pop_obs=y.tolist(),
        share_model=float(res_share[0] + f),
        share_obs=float(f),
        sigma=sig.tolist(),
        chi2=float(np.sum((res_pop / sig[0]) ** 2) + (res_share[0] / sig[1]) ** 2),
        sse_pop=float(np.sum(res_pop**2)),
    )
    return out


def host_targets(net, hosts):
    """Housing built above trend in each host, 2006 and 2007, in millions of people."""
    M = net.M
    return np.array([[M.iloc[i][f"hu_excess{y}"] * M.iloc[i].ppu2005 / 1e6 for y in (2006, 2007)] for i in hosts])


def sim_from(net, A, B, rho_h, S_med=0.0, kind="empirical", **kw):
    """Run the network model with fitted blocks A and B."""
    if B["ret"] == "tracked":
        W = net.W(B["S_s"], 1.0, S_med, kind)
        return simulate(net, W, A["d0"], A["dinf"], A["rho_s"], rho_h, kappa=B["kappa"], lam=B["lam"], **kw)
    W = net.W(B["S_s"], B["S_ret"], S_med, kind)
    return simulate(net, W, A["d0"], A["dinf"], A["rho_s"], rho_h, **kw)


def fit_C(net, A, B, S_med=0.0, kind="empirical", hosts=None):
    """How fast receiving cities add housing when crowded."""
    from common import JULY

    hosts = top_hosts(net) if hosts is None else hosts
    target = host_targets(net, hosts)
    t_eval = np.array([JULY[2006], JULY[2007]])

    def res(log_rho):
        u = unpack(sim_from(net, A, B, np.exp(log_rho[0]), S_med, kind, t_eval=t_eval), net.n)
        added = u["K"][hosts] - net.K0[hosts, None]
        return (added - target).ravel()

    sol = least_squares(res, [np.log(1.0)], diff_step=1e-3, bounds=([np.log(1e-3)], [np.log(1e3)]))
    n_res = len(sol.fun)
    s2 = np.mean(sol.fun**2) * n_res / max(n_res - 1, 1)
    se_log = float(np.sqrt(s2 / (sol.jac.T @ sol.jac)[0, 0]))
    rho_h = float(np.exp(sol.x[0]))
    return dict(
        rho_h=rho_h,
        se_log=se_log,
        ci95=[float(np.exp(sol.x[0] - 1.96 * se_log)), float(np.exp(sol.x[0] + 1.96 * se_log))],
        hosts=[net.codes[i] for i in hosts],
        target_k=target.tolist(),
        model_k=(sol.fun.reshape(target.shape) + target).tolist(),
    )


def calibrate(net=None, S_med=0.0, kind="empirical", ret="tracked", verbose=True, A=None):
    """Fit A once, then alternate B and C until rho_h stops changing (they affect each other only a little)."""
    net = Network() if net is None else net
    A = fit_A() if A is None else A
    rho_h, B = 1.0, None
    for it in range(4):
        x0 = None if B is None else tuple(B[k] for k in B["names"])
        B = fit_B(net, A, rho_h, S_med, kind, ret, x0=x0)
        C = fit_C(net, A, B, S_med, kind)
        if verbose:
            fitted = " ".join(f"{k}={B[k]:.3f}" for k in B["names"])
            print(f"  iter {it}: {fitted} rho_h={C['rho_h']:.3f}")
        if abs(np.log(C["rho_h"] / rho_h)) < 0.01:
            break
        rho_h = C["rho_h"]
    return dict(A=A, B=B, C=C, S_med=S_med, kind=kind, ret=ret, n_metros=net.n)
