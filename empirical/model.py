"""The network model, set up from data.

Each metro i has population N_i and housing capacity K_i (people in millions, time in years since landfall).
Occupancy is phi_i = N_i / K_i, and people move from fuller to emptier places:

    dN_i/dt = r_i N_i (1 - phi_i) + sum_j (F_ji - F_ij),    F_ij = W_ij N_i (phi_i - phi_j)_+

The struck city's capacity is lost and rebuilt along a fitted curve,
    K_s(t) = K_s(0) [1 - d_inf - (d_0 - d_inf) exp(-rho_s t)],
and receiving cities add housing only when they are crowded, dK_i/dt = rho_h (N_i - K_i)_+.

The weights W start from the observed IRS flows before the storm. Where a flow is suppressed, we use the gravity
model instead, capped at the suppression threshold. The struck city's outgoing weights are then rescaled so they
sum to S_s, its evacuation rate.

Compared with the summer-school model (../prototype): hosts build in response to crowding instead of on a timer,
the evacuation, return and host couplings are separate instead of one normalised m0, distances are great-circle,
and the struck city's capacity follows the Census housing estimates.
"""
import json

import numpy as np
from scipy.integrate import solve_ivp

from common import metros, metro_flows, hav, NOLA, BASE_YEARS, OUT, MIN_POP, MAX_KM


class Network:
    def __init__(self, min_pop=MIN_POP, max_km=MAX_KM, struck=NOLA):
        self.M = metros(min_pop, max_km)
        self.codes = self.M.index.tolist()
        self.n = len(self.codes)
        self.s = self.codes.index(struck)
        self.K0 = self.M.p2005.values / 1e6  # everyone starts at capacity
        self.r = self.M.natinc_rate_0205.clip(lower=0).values  # natural increase per year

        lat, lon = self.M.lat.values, self.M.lon.values
        self.D = hav(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        np.fill_diagonal(self.D, np.inf)

        g = json.load(open(OUT / "gravity.json"))["network"]
        self.alpha, self.beta, self.gamma = g["alpha"], g["beta"], g["gamma"]

        # average observed flow over the base years
        idx = {c: i for i, c in enumerate(self.codes)}
        flows = metro_flows().reset_index()
        flows = flows[flows.o.isin(idx) & flows.d.isin(idx)]
        self.flow_base = np.zeros((self.n, self.n))
        for y in BASE_YEARS:
            for o, d, v in zip(flows.o, flows.d, flows[y]):
                self.flow_base[idx[o], idx[d]] += v / len(BASE_YEARS)

    def gravity(self):
        pop = self.M.p2005.values
        w = pop[:, None] ** self.alpha * pop[None, :] ** self.beta / self.D**self.gamma
        return w * self.flow_base.sum() / w.sum()  # same total as the observed flows

    # IRS hides flows below 10 returns (about 20 exemptions), so a filled-in flow can't be larger than that.
    SUPPRESSION_CAP = 20.0

    def empirical(self):
        filled = np.minimum(self.gravity(), self.SUPPRESSION_CAP)
        return np.where(self.flow_base > 0, self.flow_base, filled)

    def W(self, S_s, S_ret, S_med, kind="empirical"):
        """Weights: out of the struck city sum to S_s, into it set by S_ret, between hosts set by S_med."""
        if kind == "empirical":
            w = self.empirical().copy()
        else:
            w = self.gravity().copy()
        np.fill_diagonal(w, 0)
        s, n = self.s, self.n
        hosts = np.ones(n, bool)
        hosts[s] = False

        # between hosts: scale so the median host has total outgoing weight S_med
        w_hosts = w.copy()
        w_hosts[s, :] = 0
        w_hosts[:, s] = 0
        W = w_hosts * (S_med / np.median(w_hosts[hosts].sum(1)))

        # out of the struck city: keep the shape of the observed flows, total S_s
        W[s, :] = w[s] / w[s].sum() * S_s

        # into the struck city (only used when return is not tracked separately)
        col = w[:, s].copy()
        col[s] = 0
        W[:, s] = col / (col @ self.K0) * S_ret * self.K0[s]
        return W


def capacity_struck(t, K0s, d0, dinf, rho_s):
    """Capacity of the struck city: drops by d0 at landfall and recovers to a long-run loss of dinf."""
    return K0s * (1 - dinf - (d0 - dinf) * np.exp(-rho_s * t))


def simulate(net, W, d0, dinf, rho_s, rho_h, T=4.0, t_eval=None, r=None, K0=None, s=None, kappa=None, lam=0.0,
             rtol=1e-6, atol=1e-9):
    """Run the network model.

    If kappa is None, people return only through the ordinary occupancy gradient (the in-weights of the struck city).
    If kappa is given, we track displaced residents separately: D_j is the number of people from the struck city now
    living in j. They return at rate kappa * (1 - phi_s)_+, i.e. only when the struck city has room, and settle for
    good at rate lam. The in-weights of the struck city are then not used.
    """
    K0 = net.K0 if K0 is None else np.asarray(K0, float)
    s = net.s if s is None else s
    r = net.r if r is None else np.asarray(r, float)
    n = len(K0)
    track = kappa is not None

    Wm = W.copy()
    if track:
        Wm[:, s] = 0

    def rhs(t, y):
        N = y[:n]
        K = y[n:2 * n].copy()
        D = y[2 * n:3 * n]
        K[s] = capacity_struck(t, K0[s], d0, dinf, rho_s)
        phi = N / K

        F = Wm * N[:, None] * np.maximum(phi[:, None] - phi[None, :], 0)  # F[i, j] = flow from i to j
        growth = r * N * (1 - phi)
        dN = growth + F.sum(0) - F.sum(1)

        dD = np.zeros(n)
        ret = np.zeros(n)
        if track:
            ret = kappa * max(1 - phi[s], 0.0) * D
            dD = F[s] - ret - lam * D
            dN = dN - ret
            dN[s] += ret.sum()

        dK = rho_h * np.maximum(N - K, 0)
        dK[s] = K0[s] * (d0 - dinf) * rho_s * np.exp(-rho_s * t)

        # running totals: out of s, into s, population lost to crowding, and excess occupancy
        totals = [F[s].sum(), F[:, s].sum() + ret.sum(), np.maximum(-growth, 0).sum(), np.maximum(N - K, 0).sum()]
        return np.concatenate([dN, dK, dD, totals, ret])

    K_init = K0.copy()
    K_init[s] = capacity_struck(0.0, K0[s], d0, dinf, rho_s)
    y0 = np.concatenate([K0, K_init, np.zeros(n), [0, 0, 0, 0], np.zeros(n)])
    return solve_ivp(rhs, (0, T), y0, t_eval=t_eval, method="LSODA", rtol=rtol, atol=atol)


def unpack(sol, n):
    """Split the solver output into named pieces."""
    y = sol.y
    return dict(
        t=sol.t,
        N=y[:n],
        K=y[n:2 * n],
        D=y[2 * n:3 * n],
        cum_out=y[3 * n],
        cum_in=y[3 * n + 1],
        loss=y[3 * n + 2],
        excess_occ=y[3 * n + 3],
        cum_ret=y[3 * n + 4:4 * n + 4],
    )
