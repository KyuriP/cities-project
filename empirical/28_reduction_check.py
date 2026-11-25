"""Why the reduced county model is used for the cross-disaster test, and how close it is to the network model.

1. Katrina: run the calibrated network (calibration.json) and the county equation (struck.py) with the same
   evacuation, return and settlement rates, phi* = 1 and no local demographic change. Report how far receiving
   areas' occupancy rises above its reference and the largest difference in New Orleans population.
2. Tax-record coverage: for each of the 21 prediction counties, the share of out-migrants (IRS exemptions) in the
   filing year before the event that appear in individually reported county-to-county flows, and the number of
   reported destination counties. The remainder is suppressed into aggregate 'other flows'.
Writes output/reduction_check.json."""
import io, json, pickle, warnings, zipfile
import numpy as np, pandas as pd
from scipy.integrate import solve_ivp
from model import Network, unpack, capacity_struck
from calib import sim_from
from common import OUT, HERE
RAW = HERE / "raw"
warnings.filterwarnings("ignore")

# 1. network vs reduced model for New Orleans
c = json.load(open(OUT / "calibration.json")); A, B, rho_h = c["A"], c["B"], c["C"]["rho_h"]
net = Network(); n, s = net.n, net.s; te = np.linspace(0, 4, 401)
u = unpack(sim_from(net, A, B, rho_h, S_med=c["S_med"], kind=c["kind"], t_eval=te), n)
hosts = [i for i in range(n) if i != s]; phi = u["N"] / u["K"]
excess = (phi[hosts] - 1).max(1)
dweighted = ((u["D"][hosts] * (phi[hosts] - 1)).sum(0) / np.maximum(u["D"][hosts].sum(0), 1e-12)).max()
def rhs(t, y):
    N, D = y; f = N / capacity_struck(t, 1.0, A["d0"], A["dinf"], A["rho_s"])
    E = B["S_s"] * N * max(f - 1, 0); R = B["kappa"] * max(1 - f, 0) * D
    return [-E + R, E - R - B["lam"] * D]
red = solve_ivp(rhs, (0, 4), [1, 0], t_eval=te, method="LSODA", rtol=1e-8, atol=1e-10).y[0]
diff = np.abs(u["N"][s] / net.K0[s] - red)
top = np.argsort(-excess)[:3]
katrina = dict(max_host_excess=float(excess.max()), top_hosts=[[net.codes[hosts[i]], float(excess[i])] for i in top],
               max_displaced_weighted_excess=float(dweighted), max_pop_diff=float(diff.max()))
print(f"hosts: max occupancy excess {100*excess.max():.1f}%, displaced-weighted {100*dweighted:.1f}%; "
      f"New Orleans network vs reduced: max diff {100*diff.max():.2f} pp")

# 2. IRS county-flow coverage before each event
ev10 = pickle.load(open(OUT / "events_inputs.pkl", "rb")); ev13 = pickle.load(open(OUT / "acs_inputs.pkl", "rb"))
group = lambda d: d.rsplit(" ", 1)[0] if d[-4:].isdigit() else d
EV = ev10 + [e for e in ev13 if e["loss"] >= 0.04 and not (group(e["disaster"]) == "Camp Fire" and e["fips"] == "06007")]
ST = {"22": "LA", "28": "MS", "48": "TX", "06": "CA", "12": "FL", "37": "NC", "15": "HI", "01": "AL"}
cov = []
for e in EV:
    fy = int(np.floor(e["t0"])) - 1; st, co = e["fips"][:2], e["fips"][2:]
    try:
        if fy >= 2008:
            d = pd.read_csv(RAW / f"countyoutflow{str(fy)[2:]}{str(fy + 1)[2:]}.csv", encoding="latin1", dtype=str)
            d.columns = [x.lower() for x in d.columns]
            d = d[(d.y1_statefips.str.zfill(2) == st) & (d.y1_countyfips.str.zfill(3) == co)]
            o, dc, num = d.y2_statefips.str.zfill(2), d.y2_countyfips.str.zfill(3), pd.to_numeric(d.n2, errors="coerce")
        else:
            z = zipfile.ZipFile(RAW / f"county{str(fy)[2:]}{str(fy + 1)[2:]}.zip")
            d = pd.read_excel(io.BytesIO(z.read([x for x in z.namelist() if x.endswith(f"{ST[st]}o.xls")][0])), header=None, dtype=str)
            d = d[(d[0] == st) & (d[1] == co)]; o, dc, num = d[2].str.zfill(2), d[3].str.zfill(3), pd.to_numeric(d[7], errors="coerce")
        tot = num[(o == "97") & (dc == "000")].sum()
        rep = (~o.isin(["96", "97", "98", "57", "58", "59"])) & ~((o == st) & (dc == co))
        cov.append(dict(name=e["name"], disaster=group(e["disaster"]), filing_year=fy, out_exemptions=int(tot),
                        n_destinations=int(rep.sum()), reported_share=float(num[rep].sum() / tot)))
    except Exception as ex:   # file for that year not in raw/
        cov.append(dict(name=e["name"], disaster=group(e["disaster"]), filing_year=fy, error=type(ex).__name__))
ok = [x for x in cov if "reported_share" in x]; sh = [x["reported_share"] for x in ok]; nd = [x["n_destinations"] for x in ok]
irs = dict(n=len(ok), median_share=float(np.median(sh)), min_share=float(min(sh)), max_share=float(max(sh)),
           median_destinations=float(np.median(nd)), min_destinations=int(min(nd)), rows=cov)
print(f"IRS coverage, {len(ok)} of {len(cov)} counties: median {100*np.median(sh):.0f}% of out-migrants in reported county flows "
      f"(range {100*min(sh):.0f}-{100*max(sh):.0f}%); destinations median {np.median(nd):.0f}, minimum {min(nd)}")
json.dump(dict(katrina=katrina, irs_coverage=irs), open(OUT / "reduction_check.json", "w"), indent=1)
