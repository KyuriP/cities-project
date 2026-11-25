# A migration model of population recovery after disasters

This repository has the code and data behind a paper on how population recovers after disasters. It started as our
group project "Ecology of Urban Systems" at the Santa Fe Institute Complex Systems Summer School 2026.

The model treats each place as a population with a housing capacity. When a disaster destroys housing, people leave.
As housing is rebuilt, some of them come back, and others settle where they went. We calibrate the model on Hurricane
Katrina using IRS migration data and Census estimates. We then test it on other US disasters between 2005 and 2024:
for each disaster, we fit the model to the others and predict the population of the affected counties.

## What's where

- `empirical/` has the analysis. The scripts are numbered in the order they run, and
  [`empirical/README.md`](empirical/README.md) says what each one does.
  - `data/` holds the processed inputs, described in [`DATA.md`](empirical/data/DATA.md).
  - `output/` holds the results and figures.
  - `tests/` has the model tests.
  - `browser/` has the hazard-map calculation, which has to run in a web browser.
- `prototype/` is the original summer-school model. We keep it for reference, but the analysis no longer uses it.

## Running it

You need Python 3.11 or later.

```bash
pip install -r requirements.txt
cd empirical
python3 00_download.py      # downloads the public source files into raw/
bash run_all.sh             # runs all analyses and makes the figures
python3 -m pytest -q tests  # runs the tests
```

The American Community Survey step needs a free Census API key. Put it in the environment variable `CENSUS_API_KEY`
or in the file `~/.census_api_key`.

The hazard-map step runs in a browser console, so its result is already saved in
`empirical/data/hazard_geography.json` and the rest of the pipeline runs without it.

A full run takes a few hours, most of it for the held-out tests.

## Data

All the data are public: IRS county-to-county migration files, Census population and housing estimates, American
Community Survey tables, OpenFEMA assistance records, the Household Pulse Survey, FEMA flood maps, NOAA storm-surge
maps and terrain tiles. Sources and versions are listed in [`empirical/data/DATA.md`](empirical/data/DATA.md).

## Licence

The code is under the [MIT License](LICENSE). The data stay subject to the terms of the agencies that publish them.
