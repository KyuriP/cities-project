"""Shared settings and data loaders.

Every input file in data/ is built from the official downloads by 00_download.py and 00_build_data.py
(see data/DATA.md).
"""
import gzip
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
# The sensitivity runs (06_sensitivity.py) set CAL_OUT so that each variant writes to its own folder.
OUT = Path(os.environ.get("CAL_OUT", HERE / "output"))
OUT.mkdir(parents=True, exist_ok=True)

NOLA = "35380"  # New Orleans-Metairie-Kenner metro area (2009 definition)
NOLA_PARISHES = ["22051", "22071", "22075", "22087", "22089", "22095", "22103"]
OTHER_STRUCK = {
    "25060": "Gulfport (Katrina)",
    "37700": "Pascagoula (Katrina)",
    "13140": "Beaumont (Rita)",
    "29340": "Lake Charles (Rita)",
}
LANDFALL = 2005 + 240 / 365  # 29 August 2005, as a decimal year

# Choices that 06_sensitivity.py varies through environment variables.
BASE_YEARS = os.environ.get("CAL_BASE", "0304,0405").split(",")  # IRS filing years before the storm
TREND_YEARS = tuple(int(x) for x in os.environ.get("CAL_TREND", "2001,2005").split(","))  # first, last
MIN_POP = int(os.environ.get("CAL_MINPOP", "250000"))
MAX_KM = float(os.environ.get("CAL_MAXKM", "600"))
TOP_K = int(os.environ.get("CAL_TOPK", "10"))

# Model time is years since landfall.
JULY = {y: y + 0.5 - LANDFALL for y in range(2006, 2010)}  # 1 July of each year
APRIL = {"0506": 0.62, "0607": 1.62, "0708": 2.62}  # end of each IRS filing year


def read(name, **kw):
    return pd.read_csv(DATA / name, sep="|", comment="#", **kw)


def hav(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


@lru_cache(None)
def counties():
    return read("counties.psv", dtype={"fips": str})


@lru_cache(None)
def metros_all():
    return read("metros.psv", dtype={"code": str}).set_index("code")


@lru_cache(None)
def county_flows():
    return pd.read_csv(
        DATA / "irs_county_flows.psv.gz",
        sep="|",
        comment="#",
        dtype={"year": str, "orig": str, "dest": str},
    )


@lru_cache(None)
def nola_totals():
    return read("irs_nola_totals.psv", dtype={"year": str}).set_index("year")


def linear_trend(row, prefix, years=None):
    """Straight line through the trend years (2001-2005 by default), as a function of calendar year."""
    if years is None:
        years = np.arange(TREND_YEARS[0], TREND_YEARS[1] + 1)
    else:
        years = np.array(list(years))
    coef = np.polyfit(years, [row[f"{prefix}{y}"] for y in years], 1)
    return lambda t: np.polyval(coef, t)


@lru_cache(None)
def metro_flows():
    """IRS flows (exemptions) between metro areas, one column per filing year."""
    crosswalk = read("county_metro.psv", dtype={"fips": str, "code": str})
    county_to_metro = dict(zip(crosswalk.fips, crosswalk.code))
    flows = county_flows().copy()
    flows["o"] = flows.orig.map(county_to_metro)
    flows["d"] = flows.dest.map(county_to_metro)
    flows = flows.dropna(subset=["o", "d"])
    flows = flows[flows.o != flows.d]  # moves within a metro are not migration between metros
    return flows.groupby(["year", "o", "d"]).exemptions.sum().unstack("year").fillna(0)


def select_metros(min_pop=None, max_km=None):
    """Metros in the model: at least min_pop residents in 2005, or within max_km of New Orleans."""
    if min_pop is None:
        min_pop = MIN_POP
    if max_km is None:
        max_km = MAX_KM
    allm = metros_all()
    nola = allm.loc[NOLA]
    dist = hav(nola.lat, nola.lon, allm.lat, allm.lon)
    return allm[(allm.p2005 >= min_pop) | (dist <= max_km)].index.tolist()


@lru_cache(None)
def metros(min_pop=None, max_km=None):
    """Table of the model metros: flows to and from New Orleans, housing, population components."""
    allm = metros_all()
    codes = select_metros(min_pop, max_km)
    m = allm.loc[codes].copy()
    nola = allm.loc[NOLA]
    m["dist"] = hav(nola.lat, nola.lon, m.lat, m.lon)

    flows = metro_flows()
    for y in ["0304", "0405", "0506", "0607", "0708"]:
        if y in flows:
            m[f"o{y}"] = flows[y].xs(NOLA, level="o").reindex(m.index).fillna(0)  # out of New Orleans
            m[f"i{y}"] = flows[y].xs(NOLA, level="d").reindex(m.index).fillna(0)  # into New Orleans
        else:
            m[f"o{y}"] = 0
            m[f"i{y}"] = 0
    m["o_base"] = m[[f"o{y}" for y in BASE_YEARS]].mean(1)
    m["i_base"] = m[[f"i{y}" for y in BASE_YEARS]].mean(1)
    m["excess"] = (m.o0506 - m.o_base).clip(lower=0)  # extra moves out of New Orleans in 2005-06
    m["excess_ret"] = (m.i0607 - m.i_base).clip(lower=0)  # extra moves back in 2006-07

    dm_years = range(max(TREND_YEARS[0], 2002), TREND_YEARS[1] + 1)
    m["dm_base"] = m[[f"dm{y}" for y in dm_years]].mean(1)
    m["dm_excess2006"] = m.dm2006 - m.dm_base
    m["ppu2005"] = m.p2005 / m.hu2005  # persons per housing unit

    for y in (2006, 2007, 2008, 2009):
        hu_excess, occ = [], []
        for _, r in m.iterrows():
            hu_trend = linear_trend(r, "hu")(y)
            p_trend = linear_trend(r, "p")(y)
            hu_excess.append(r[f"hu{y}"] - hu_trend)
            occ.append((r[f"p{y}"] / r[f"hu{y}"]) / (p_trend / hu_trend) - 1)
        m[f"hu_excess{y}"] = hu_excess
        m[f"occ{y}"] = occ
    return m


def nola_series():
    """New Orleans population and housing relative to their 2001-05 trends, on 1 July 2006-2009."""
    r = metros_all().loc[NOLA]
    p_trend = linear_trend(r, "p")
    hu_trend = linear_trend(r, "hu")
    years = [2006, 2007, 2008, 2009]
    return dict(
        years=years,
        t=[JULY[y] for y in years],
        pop_rel=[r[f"p{y}"] / p_trend(y) for y in years],
        hu_rel=[r[f"hu{y}"] / hu_trend(y) for y in years],
        occ=[(r[f"p{y}"] / r[f"hu{y}"]) / (p_trend(y) / hu_trend(y)) for y in years],
    )
