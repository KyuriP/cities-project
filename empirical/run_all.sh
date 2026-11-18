#!/usr/bin/env bash
# Reproduce every empirical number and figure from official sources. Outputs in ./output. ~15 minutes.
set -euo pipefail
cd "$(dirname "$0")"
python3 00_download.py          # raw files from irs.gov and census.gov (checksums in raw/MANIFEST.txt)
python3 00_build_data.py        # processed inputs in data/, verified against hand-extracted files in data/legacy/
python3 01_describe_data.py     # descriptive statistics, fig_data
python3 02_gravity.py           # network-wide and New Orleans gravity estimates, bootstrap
python3 03_calibrate.py         # calibration (stages A-C), onward-migration evidence
python3 04_validate.py          # out-of-sample validation
python3 05_uncertainty.py       # jackknife and Monte Carlo propagation
python3 07_simulations.py       # severity across all metros, sweeps
python3 08_campfire.py          # second disaster (Camp Fire 2018), no refitting
python3 10_events.py            # ten-disaster panel: per-county calibration, tolerated occupancy, transfer tests
python3 05_figures.py           # fig_calibration, fig_simulations, fig_campfire, fig_events, fig_panel
python3 09_map.py               # fig_map (Fig. 2): evacuee destinations and returnee origins
python3 11_schematic.py         # fig_schematic (Fig. 1): model illustration from calibrated rates
python3 12_acs.py               # ACS 1-year county housing and household size (needs a Census API key)
python3 13_acs_panel.py         # 36 county-events with ACS-measured capacity
python3 14_holdout.py           # leave-one-disaster-out prediction on the combined panel (~40 min)
python3 15_holdout_fig.py       # fig_holdout (Fig. 6)
python3 16_damage_geography.py  # FEMA registrations per ZIP vs tolerated occupancy
python3 17_hazard_geography.py  # pre-disaster hazard geography vs phi* (browser output in data/)
python3 18_holdout_geo.py       # held-out prediction with phi* from hazard geography
python3 19_hazard_fig.py        # fig_hazard (Fig. 7)
for v in "HOLDOUT_TAU=0.05 HOLDOUT_TAG=_tau005" "HOLDOUT_TAU=0.2 HOLDOUT_TAG=_tau02" "HOLDOUT_MIN_LOSS=0.03 HOLDOUT_TAG=_loss3" "HOLDOUT_MIN_LOSS=0.05 HOLDOUT_TAG=_loss5"; do env $v python3 14_holdout.py; done
python3 20_holdout_uncertainty.py  # bootstrap intervals and sensitivity summary
python3 21_fema_mechanism.py    # FEMA registrant records vs phi*
python3 22_pulse.py             # Household Pulse displacement vs ACS housing loss
python3 23_capacity_check.py    # profile likelihood; ACS-capacity refit of five counties
HOLDOUT_ACSCAP=1 HOLDOUT_TAG=_acscap python3 14_holdout.py  # held-out test with survey capacity (after 23)
python3 17_hazard_geography.py  # rerun to add the survey-capacity variant
python3 24_baseline.py          # statistical baselines in the held-out folds
python3 25_destinations.py      # destinations beyond Katrina (IRS county flows)
python3 26_pred_intervals.py    # held-out prediction intervals and coverage
python3 06_sensitivity.py       # 11 analyst-choice variants + gravity network (writes to output/sensitivity/)
python3 -m pytest -q tests      # model correctness tests
