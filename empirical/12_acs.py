"""American Community Survey (1-year) housing and household measures for counties, 2005-2025.

Tables: B01003 population, B25001 housing units, B25002 occupancy, B25004 vacancy status, B25010 average
household size, B11001 households. The 2020 1-year release wasn't published in the usual form (pandemic), so it's
missing. The ACS only publishes counties with at least 65,000 residents.

Measures:
  usable_units  housing units minus 'other vacant' minus seasonal units
                ('other vacant' goes up when homes are damaged or uninhabitable)
  hh_size       average household size; a rise above trend means people doubling up

Needs a Census API key in CENSUS_API_KEY or ~/.census_api_key (never stored in the repo). The raw API answers are
cached in raw/acs/; the processed table is data/acs_county.psv.
"""

import json
import os
import pathlib
import time
import urllib.request
import pandas as pd
from common import HERE

RAW = HERE / "raw" / "acs"
RAW.mkdir(parents=True, exist_ok=True)
KEY = os.environ.get("CENSUS_API_KEY") or (pathlib.Path.home() / ".census_api_key").read_text().strip()
VARS = {
    "B01003_001E": "pop",
    "B25001_001E": "hu",
    "B25002_002E": "occupied",
    "B25002_003E": "vacant",
    "B25004_002E": "v_for_rent",
    "B25004_003E": "v_rented_not_occ",
    "B25004_004E": "v_for_sale",
    "B25004_005E": "v_sold_not_occ",
    "B25004_006E": "v_seasonal",
    "B25004_007E": "v_migrant",
    "B25004_008E": "v_other",
    "B25010_001E": "hh_size",
    "B11001_001E": "households",
    "B25001_001M": "hu_moe",
    "B25004_008M": "v_other_moe",
    "B25010_001M": "hh_size_moe",
    "B01003_001M": "pop_moe",
}
YEARS = [y for y in range(2005, 2026) if y != 2020]


def fetch(year):
    f = RAW / f"acs1_{year}.json"
    if not f.exists():
        url = f"https://api.census.gov/data/{year}/acs/acs1?get=NAME,{','.join(VARS)}&for=county:*&in=state:*&key={KEY}"
        for attempt in range(5):
            try:
                txt = urllib.request.urlopen(url, timeout=180).read().decode()
                json.loads(txt)  # only cache valid JSON
                f.write_text(txt)
                break
            except Exception as e:
                print(year, "retry", attempt, str(e)[:80])
                time.sleep(10)
    j = json.loads(f.read_text())
    d = pd.DataFrame(j[1:], columns=j[0])
    d["fips"] = d.state + d.county
    d["year"] = year
    for v, n in VARS.items():
        d[n] = pd.to_numeric(d[v], errors="coerce")
    return d[["fips", "year", "NAME"] + list(VARS.values())]


A = pd.concat([fetch(y) for y in YEARS], ignore_index=True)
for c in [c for c in A.columns if c not in ("fips", "year", "NAME")]:
    A.loc[A[c] < 0, c] = float("nan")  # Census negative codes mean not available
A["usable_units"] = A.hu - A.v_other - A.v_seasonal
out = HERE / "data" / "acs_county.psv"
with open(out, "w") as fh:
    fh.write("# ACS 1-year county estimates 2005-2025 (no standard 2020 release); see 12_acs.py\n")
    A.to_csv(fh, sep="|", index=False)
print(len(A), "county-years;", A.fips.nunique(), "counties;", A.year.min(), "-", A.year.max())
