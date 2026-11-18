# Empirical calibration: Hurricane Katrina (and a second test on the 2018 Camp Fire)

Everything here is rebuilt from official IRS and Census files with `bash run_all.sh` (about 15 minutes; `pip install
numpy scipy pandas statsmodels matplotlib xlrd pyshp pyproj pytest`). See [`data/DATA.md`](data/DATA.md) for sources, how each data set
maps onto the model, and limitations, and [`../MODEL_AUDIT.md`](../MODEL_AUDIT.md) for the model checks.

## Model (`model.py`)

Metros are patches with population `N` and capacity `K`; residents move down occupancy gradients `phi = N/K` along a
directed network built from observed pre-disaster IRS flows. Compared with the report model in `src/` and `network/`:
host building is demand-driven (not on a timer), rates are per capita, the evacuation, return and host couplings are
separate (no max-normalised `m0`), distances are great-circle, the struck city's capacity follows measured housing units,
and **displaced residents are tracked**: they return when the struck city has room and settle in their host at a constant
rate. The tracked-return model fits New Orleans' population 19 times better (squared error) than gradient-driven return,
and is the only version that reproduces where returnees came from and when.

## Calibration protocol (`calib.py`, `03_calibrate.py`)

| Stage | Parameters | Fitted to | Estimate (95% CI) |
|---|---|---|---|
| A | capacity lost at landfall `d_0`, permanently `d_inf`, rebuild rate `rho_s` | Census housing units, NOLA metro, 2006-09 | 0.42 (jackknife SE 0.16), 0.11, 0.76/yr |
| B | evacuation `S_s`, return `kappa`, settling `lam` | Census population 2006-09; IRS timing | 1.62/yr (1.54-1.69), 8.8/yr (6.1-12.5), 0.50/yr (0.40-0.64) |
| C | host capacity response `rho_h` | housing units above trend in 10 main host metros, 2006-07 | 1.9/yr (0.4-8.6) |
| — | host-host coupling `S_med` | IRS onward flows (none detected) | 0 |
| — | network | observed 2003-05 IRS flows; gravity fill: alpha 1.02, beta 0.90, gamma 1.12 (38,220 metro pairs) | — |

## Ten-disaster panel (`10_events.py`)
The same struck-place model is calibrated on every county whose Census housing estimates record a disaster loss:
Orleans, St Bernard, Plaquemines, Jefferson (LA), Hancock, Harrison, Jackson (MS) after Katrina; Cameron (LA) after
Rita; Aransas (TX) after Harvey 2017; Butte (CA) after the 2018 Camp Fire. Ike (2008) and Michael (2018) were
checked and excluded (no housing loss in the Census estimates). County IRS totals including suppressed flows are built
by `00_build_data.py` (`data/irs_county_totals.psv`). Three fits: housing capacity only (per-event evacuation and
settling, shared return rate), plus a per-event tolerated occupancy phi*, and fully pooled rates; leave-one-event-out
prediction; relation of rates to shock size and to pre-disaster vacancy and tenure (Census 2000/2010 profiles).
Main results (`output/events.json`): phi* ranges 0.80-1.25 and explains most cross-event differences (AIC 209 vs 2277);
rates from other events predict a held-out event with a median error of 6.4 percentage points of population.

## Larger panel and held-out prediction (`12_acs.py` ... `16_damage_geography.py`)
- `12_acs.py`: ACS 1-year county tables 2005-2024 (needs a free Census API key in `CENSUS_API_KEY` or `~/.census_api_key`;
  never committed). Writes `data/acs_county.psv`; usable units = housing units - other vacant - seasonal.
- `13_acs_panel.py`: 43 county-disaster pairs fixed in advance (2008-2024; Orleans for Ida added later), 37 with enough ACS years; capacity from
  usable units, population from Census estimates spliced across vintages. `output/acs_panel.json`.
- `14_holdout.py`: 21 counties (10 Census-capacity + 11 ACS counties losing >= 4%), 14 disasters. Partial pooling of
  phi* (sd 0.1); each disaster predicted from the others. Median held-out error 3.0 pp (pooled phi*), 3.6 (phi* = 1),
  4.9 (population = capacity), 6.3 (no change). `output/holdout.json`; figure by `15_holdout_fig.py`.
- `16_damage_geography.py`: FEMA registrations per ZIP housing unit; does not predict phi* in the expected direction
  (Spearman -0.44 for share of housing spared, n = 20). `output/damage_geography.json`.

- `17_hazard_geography.py`, `browser/hazard_geography.js`: share of pre-disaster housing outside the FEMA flood zone,
  NOAA Category 3 surge zone and NOAA levee areas, and above 3 m, for the 17 storm/flood counties of the held-out
  panel (computed in a browser; results in `data/hazard_geography.json`). Spearman 0.55 (p 0.02) and 0.68 (p 0.005)
  with phi*, strongest within Katrina. `18_holdout_geo.py`: phi* predicted from geography does not beat pooled phi*
  for held-out disasters (median 4.1 vs 3.7 pp; elevation variant 3.9 vs 4.4). Figure by `19_hazard_fig.py` (Fig. 7).

- `20_holdout_uncertainty.py`: cluster bootstrap over held-out disasters, Wilcoxon tests, and the sensitivity runs of
  `14_holdout.py` (`HOLDOUT_TAU` = 0.05/0.2, `HOLDOUT_MIN_LOSS` = 0.03/0.05, `HOLDOUT_TAG` sets the output suffix).
  Pooled phi* beats capacity tracking by 1.9 pp (95% 0.4-2.5, p = 0.006); beats phi* = 1 by 0.5 pp (n.s.).
- `21_fema_mechanism.py`: OpenFEMA registrant records (Harvey, Irma, Florence, Michael, Laura; 6 panel counties).
  Rental location is blank in the data; share placed in FEMA hotels vs phi*: Spearman -0.94 (n = 6).
- `22_pulse.py`: Household Pulse Survey displacement questions; Florida after Ian and Hawaii after Lahaina, compared
  with ACS-measured housing loss (ratios 6.4 and 2.1). Week 55 is rejected by the Census server and omitted.

- `14_holdout.py` with `HOLDOUT_ACSCAP=1 HOLDOUT_TAG=_acscap`: survey capacity for Orleans, Jefferson, Harrison, Jackson, Butte;
  pooled median error unchanged (3.0 pp vs 6.1 for capacity tracking). The hazard ranking largely disappears (17_hazard_geography.py).
- `23_capacity_check.py`: profile-likelihood intervals for phi* (all narrower than +-0.02) and a refit of Orleans,
  Jefferson, Harrison, Jackson and Butte with ACS-measured capacity: phi* rises in all five, strongly in Orleans
  (0.97 -> 1.27) and Jefferson (1.01 -> 1.13).
- `24_baseline.py`: statistical forecasts in the held-out folds (median 5.6 / 5.2 / 4.0 pp vs 3.0 for the model).
- `25_destinations.py`: destination shares of excess IRS out-migration for seven post-2017 struck counties
  (pre-disaster flow shares median R^2 0.17; gravity 0.03-0.06).
- `26_pred_intervals.py`: held-out prediction intervals (90% intervals cover 95% of county-years).

## Results in brief

- **Validation (not fitted):** destinations of evacuees R² 0.70; origins of returnees R² 0.55; host net domestic
  migration correlation 0.79 (model underestimates host gains: slope 1.55); New Orleans net in-migration 2006-09
  front-loaded as observed (model 43k, 38k, 16k; Census 45k, 28k, 22k).
- **Failures:** Houston's share of evacuees (28%) and returnees (41%) is underestimated by half or more; IRS return timing
  is still too slow (ratio 0.72 vs 1.87); host crowding persists longer than modelled.
- **Robustness:** 11 analyst-choice variants change `S_s`, `kappa`, `lam` by < 15% and validation R² by < 0.04; `rho_h`
  is fragile (0.6-2.3). A pure gravity network performs much worse (destinations R² 0.35).
- **Simulations:** with calibrated parameters, severity (excess-occupancy person-years per resident) is nearly
  independent of which metro is struck (0.239-0.250), scales smoothly with shock size (no threshold) and depends strongly on
  host capacity response (1.14 to 0.15 over `rho_h` = 0.05-10/yr). Hosts bear about half of the crowding.
- **Camp Fire (no refitting):** the Katrina rates overpredict Butte County's population loss (predicted 0.90 and 0.87 of
  trend in 2019-20; observed 0.94 and 0.91): much of the displaced population stayed in the county, doubled up.
