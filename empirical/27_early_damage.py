"""How much does prediction depend on knowing the rebuilding path? (Planning use of the model.)

Same 21 counties, folds and training parameters as 14_holdout.py (read from output/holdout.json). For each omitted
county the population path is predicted from three capacity inputs:
  observed  the measured capacity path (main analysis)
  typical   only the first post-event capacity loss L1 is known; rebuilding follows the median path of the
            training counties, K(t) = 1 - L1 * m(t), with m(t) the median of (1 - K_j(t)) / L1_j across training
            counties (held at its last value beyond their follow-up)
  flat      only L1 is known and no rebuilding is assumed, K(t) = 1 - L1
Population tracking capacity (N = K) is reported for each input as the comparison. Writes output/early_damage.json."""
import json, pickle
import numpy as np
from common import OUT
from struck import predictions

ev10 = pickle.load(open(OUT / "events_inputs.pkl", "rb")); ev13 = pickle.load(open(OUT / "acs_inputs.pkl", "rb"))
group = lambda d: d.rsplit(" ", 1)[0] if d[-4:].isdigit() else d
EV = ev10 + [e for e in ev13 if e["loss"] >= 0.04 and not (group(e["disaster"]) == "Camp Fire" and e["fips"] == "06007")]
for e in EV:
    e["g"] = group(e["disaster"]); e["K"] = (lambda kt, kv: (lambda s: np.interp(s, kt, kv)))(e["Kt"], e["Kv"])
rows = {(r["g"], r["name"]): r for r in json.load(open(OUT / "holdout.json"))["rows"]}
L1 = lambda e: max(1 - float(e["K"](e["t_obs"][0])), 1e-3)
grid = np.linspace(0, 6, 241)
rmse = lambda p, o: float(np.sqrt(np.mean((np.asarray(p) - o) ** 2)))
out = []
for e in EV:
    r = rows[(e["g"], e["name"])]; P = r["train_H"]; tr = [q for q in EV if q["g"] != e["g"]]
    m = np.median([np.clip((1 - q["K"](np.minimum(grid, q["t_obs"][-1]))) / L1(q), 0, 2) for q in tr], axis=0)
    l1 = L1(e); m0 = np.interp(e["t_obs"][0], grid, m)
    caps = {"observed": e["K"],
            "typical": (lambda l1, m, m0: (lambda s: 1 - l1 * np.interp(s, grid, m) / m0))(l1, m, m0),
            "flat": (lambda l1: (lambda s: 1 - l1 + 0 * np.asarray(s, float)))(l1)}
    S = np.exp(P["a"] + P["b"] * np.log(max(e["loss"], 0.01))); o = e["pop_rel"]
    res = dict(g=e["g"], name=e["name"], loss=float(e["loss"]))
    for k, K in caps.items():
        res[k] = rmse(predictions(e, K, S, P["kap"], P["lam"], r["mu_train"])[0], o)
        res[k + "_NK"] = rmse(K(e["t_obs"]), o)
    out.append(res)
assert abs(np.median([x["observed"] for x in out]) - np.median([r["rmse"]["H"] for r in rows.values()])) < 1e-9
summ = {}
for k in ("observed", "typical", "flat"):
    a, b = np.array([x[k] for x in out]), np.array([x[k + "_NK"] for x in out])
    big = [x for x in out if x["loss"] >= 0.10]
    summ[k] = dict(model=float(np.median(a)), NK=float(np.median(b)), wins=int(np.sum(a < b)),
                   model_big=float(np.median([x[k] for x in big])), NK_big=float(np.median([x[k + "_NK"] for x in big])))
    print(f"{k:9s} model {100*summ[k]['model']:.1f}  N=K {100*summ[k]['NK']:.1f}  model better {summ[k]['wins']}/21 | >=10% loss: model {100*summ[k]['model_big']:.1f}  N=K {100*summ[k]['NK_big']:.1f}")
json.dump(dict(rows=out, summary=summ), open(OUT / "early_damage.json", "w"), indent=1)
