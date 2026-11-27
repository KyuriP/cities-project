# Analysis

This folder has all the analysis code. To rebuild everything from the public data:

```bash
python3 00_download.py   # downloads the raw files into raw/
bash run_all.sh          # runs the scripts below in order
```

A full run takes a few hours, most of it for the held-out tests. The data sources and how each one is used are
described in [`data/DATA.md`](data/DATA.md). The check of our original summer-school code is in
[`../prototype/MODEL_AUDIT.md`](../prototype/MODEL_AUDIT.md).

## The model

- `model.py` is the network model. Each metro area has a population `N` and a housing capacity `K`, and people
  move from fuller to emptier places (occupancy `phi = N/K`). The links between places come from the IRS migration
  flows before the disaster, with a gravity model filling in the flows the IRS hides. People who leave the struck
  city are tracked: they can return once it has room again, and they settle for good where they are at a constant
  rate. Receiving cities build housing when they get crowded.
- `struck.py` is the same model seen from one struck county, used for the other disasters. It has one extra
  parameter, the occupancy threshold `phis` (phi*): people leave when `N/K` is above it and can return when it's
  below. `phis = 1` is the network case.
- `calib.py` fits the network model to Katrina. `common.py` has shared settings and data loaders, and
  `figstyle.py` the figure style.

## Scripts

They run in the order listed in `run_all.sh`.

**Data**

| Script | What it does | Output |
|---|---|---|
| `00_download.py` | Downloads the IRS and Census files | `raw/` |
| `00_build_data.py` | Builds the county and metro tables and checks them against files extracted by hand (`data/legacy/`) | `data/` |
| `12_acs.py` | American Community Survey county tables, 2005-2024 (needs a Census API key) | `data/acs_county.psv` |
| `01_describe_data.py` | Summary of the Katrina data, data figure | `data_summary.json` |

**Katrina**

| Script | What it does | Output |
|---|---|---|
| `02_gravity.py` | Gravity model for the hidden IRS flows | `gravity.json` |
| `03_calibrate.py` | Fits housing loss and rebuilding, leaving/return/settling, and host building | `calibration.json` |
| `04_validate.py` | Checks the fit against data it wasn't fitted to (destinations, where returners came from, host migration) | `validation.json` |
| `05_uncertainty.py` | Carries the uncertainty in the fitted values through to the predictions | `uncertainty.json` |
| `06_sensitivity.py` | Reruns everything with different analysis choices | `sensitivity.json` |
| `07_simulations.py` | What makes a disaster worse, for every metro in the network | `simulations.json` |
| `08_campfire.py` | The Camp Fire, with the Katrina rates and no refitting | `campfire.json` |
| `28_reduction_check.py` | How close the county model is to the network model, and how much of each county's out-migration the IRS reports | `reduction_check.json` |

**Other disasters**

| Script | What it does | Output |
|---|---|---|
| `10_events.py` | County model for ten counties hit by Katrina, Rita, Harvey and the Camp Fire (Census housing) | `events.json` |
| `13_acs_panel.py` | 43 county-disaster pairs from 2008-2024, with ACS housing | `acs_panel.json` |
| `23_capacity_check.py` | Does phi* change with the housing measure? Plus intervals for phi* | `capacity_check.json` |
| `14_holdout.py` | Leave each disaster out and predict it (21 counties, 14 disasters) | `holdout.json` |
| `20_holdout_uncertainty.py` | Bootstrap over disasters and tests for the held-out comparison | `holdout_uncertainty.json` |
| `24_baseline.py` | Simple statistical forecasts in the same test | `baseline.json` |
| `26_pred_intervals.py` | Prediction intervals and how often they contain the data | `pred_intervals.json` |
| `27_early_damage.py` | Prediction when only the first housing-loss estimate is known | `early_damage.json` |

**Extra checks**

| Script | What it does | Output |
|---|---|---|
| `16_damage_geography.py` | FEMA registrations by ZIP code against phi* | `damage_geography.json` |
| `browser/hazard_geography.js`, `17_hazard_geography.py` | Share of housing outside flood, surge and levee zones and above 3 m, against phi* (the map part runs in a browser; its result is saved in `data/hazard_geography.json`) | `hazard_geography.json` |
| `18_holdout_geo.py` | Held-out prediction with phi* from hazard geography | `holdout_geo.json` |
| `21_fema_mechanism.py` | FEMA hotel stays and rentals against phi* | `fema_mechanism.json` |
| `22_pulse.py` | Household Pulse Survey displacement against housing loss (Hurricane Ian, Lahaina fire) | `pulse.json` |
| `25_destinations.py` | Where people went after the other disasters (IRS county flows) | `destinations.json` |

**Figures**: `05_figures.py`, `09_map.py`, `11_schematic.py`, `15_holdout_fig.py` and `19_hazard_fig.py` write
`output/fig_*.pdf`. In the paper these are `fig_events` (Fig. 1), `fig_schematic` (Fig. 2), `fig_holdout` (Fig. 3)
and `fig_panel` (Fig. 4), with `fig_map`, `fig_data`, `fig_calibration`, `fig_hazard` and `fig_simulations` as
Supplementary Figs. S1-S5.

Everything goes to `output/`. The tests are in `tests/` (`python3 -m pytest -q tests`).

## Main results

- Katrina: people left at 1.62 per year, returned at 8.8 per year when there was room, and settled elsewhere at
  0.50 per year; host cities built at 1.9 per year (0.4-8.6). Without being fitted to them, the model gets the
  destinations of evacuees (R² 0.70) and where returners came from (R² 0.55) roughly right, but it gives Houston
  less than half its real share and brings people back too slowly.
- Ten counties: housing loss alone fits the flooded New Orleans parishes but not the other places. Letting each
  county have its own occupancy threshold (0.80-1.25) fits all ten. With ACS housing instead of Census housing, the
  threshold goes up in all five counties covered, a lot in Orleans (0.97 to 1.27).
- Held-out prediction: median error 3.0 percentage points with a pooled threshold, 3.6 with phi* = 1, 4.9 if
  population just follows housing, 6.3 for no change, and 4.0-6.0 for the statistical forecasts. Most of the gain
  over capacity tracking comes from modelling leaving and returning over time; the pooled threshold adds a smaller
  improvement.
- The extra checks are weak or only descriptive: hazard geography doesn't improve prediction, FEMA hotel stays are
  lower where phi* is higher (6 counties), the Household Pulse Survey shows more displacement than the measured
  housing loss, and destinations after the other disasters follow normal migration only loosely (median R² 0.17).
