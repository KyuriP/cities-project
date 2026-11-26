"""The county model: one struck place and its displaced residents.

This is what the network model becomes when the receiving places stay close to their usual occupancy
(see 28_reduction_check.py). We follow the struck place's population N and capacity K(t), both relative to the
pre-event trend (so N starts at 1), and the displaced residents D who live elsewhere and may come back:

    dN/dt = -S N (N/K - phi*)_+ + kappa (phi* - N/K)_+ D
    dD/dt =  S N (N/K - phi*)_+ - kappa (phi* - N/K)_+ D - lam D

phi* = 1 gives the network case. Other values of phi* shift the occupancy at which people start to leave or can
come back. We also keep running totals of moves out and back, to compare with the IRS filing years.

Used by 10_events.py, 13_acs_panel.py and the held-out tests.
"""
import numpy as np
from scipy.integrate import solve_ivp


def simulate(K, S_s, kappa, lam, T, phis=1.0):
    def rhs(t, y):
        N, D, cum_out, cum_in = y
        phi = N / K(t)
        out = S_s * N * max(phi - phis, 0)
        ret = kappa * max(phis - phi, 0) * D
        return [-out + ret, out - ret - lam * D, out, ret]

    return solve_ivp(rhs, (0, T), [1.0, 0.0, 0.0, 0.0], method="LSODA", dense_output=True,
                     rtol=1e-8, atol=1e-10, max_step=0.02)


def predictions(ev, K, S_s, kappa, lam, phis=1.0):
    """Population at the observation times, plus two things to compare with the IRS data: the share of the
    two-year extra out-migration that falls in the event's filing year, and the returns in the second year."""
    fy = ev.get("fy_end")
    T = max(ev["t_obs"][-1], fy[1] if fy is not None else 0) + 0.05
    sol = simulate(K, S_s, kappa, lam, T, phis)
    pop = sol.sol(ev["t_obs"])[0]

    share = ret = np.nan
    if fy is not None:
        cum_out = sol.sol(np.array([0.0, *fy]))[2]
        cum_in = sol.sol(np.array([fy[0], fy[1]]))[3]
        share = (cum_out[1] - cum_out[0]) / max(cum_out[2] - cum_out[0], 1e-12)
        ret = (cum_in[1] - cum_in[0]) * ev["P0"]
    return pop, share, ret


def capacity_piecewise(t_obs, k_rel):
    """Capacity path through the housing estimates: drops at the event to the first estimate, straight lines
    between estimates, flat after the last one."""
    t_obs = np.asarray(t_obs, float)
    k_rel = np.asarray(k_rel, float)
    return lambda s: np.interp(s, np.r_[0, t_obs], np.r_[k_rel[0], k_rel], right=k_rel[-1])
