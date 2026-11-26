"""Build the processed data files in data/ from the raw downloads in raw/ (see 00_download.py).

Files written (pipe-separated text):
  counties.psv             every US county: population and housing units 2000-2009, coordinates
  metros.psv               every metro area (2009 definitions): population, housing, components of change, centre
  county_metro.psv         which metro each county belongs to
  irs_county_flows.psv.gz  every county-to-county IRS flow, filing years 2003-04 to 2007-08
  irs_county_totals.psv    total moves out of and into each county, including the small flows the IRS hides
  irs_nola_totals.psv      the same totals for the New Orleans metro

At the end we compare the New Orleans flows and the metro populations with files I extracted by hand earlier
(data/legacy/). The script stops with an error if anything differs.
"""
import gzip
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
DATA = HERE / "data"
YEARS = ["0304", "0405", "0506", "0607", "0708"]
NOLA_PARISHES = ["22051", "22071", "22075", "22087", "22089", "22095", "22103"]
DECADE = range(2000, 2010)


def csv(name, **kw):
    return pd.read_csv(RAW / name, encoding="latin1", **kw)


# ---- counties: Census population and housing estimates, plus coordinates ----

pop = csv("co-est00int-tot.csv", dtype={"STATE": str, "COUNTY": str})
pop = pop[pop.SUMLEV == 50].copy()  # county rows only
pop["fips"] = pop.STATE.str.zfill(2) + pop.COUNTY.str.zfill(3)

hu = csv("hu-est00int-tot.csv", dtype={"STATE": str, "COUNTY": str, "SUMLEV": str})
hu = hu[hu.SUMLEV == "050"].copy()
hu["fips"] = hu.STATE.str.zfill(2) + hu.COUNTY.str.zfill(3)

gaz_zip = zipfile.ZipFile(RAW / "Gaz_counties_national.zip")
gaz = pd.read_csv(io.BytesIO(gaz_zip.read(gaz_zip.namelist()[0])), sep="\t", dtype={"GEOID": str}, encoding="latin1")
gaz.columns = [c.strip() for c in gaz.columns]

pop_cols = {f"POPESTIMATE{y}": f"p{y}" for y in DECADE}
hu_cols = {f"HUEST_{y}": f"hu{y}" for y in DECADE}
cty = pop[["fips", "CTYNAME", "STNAME"] + list(pop_cols)].rename(columns=pop_cols | {"CTYNAME": "name", "STNAME": "state"})
cty = cty.merge(hu[["fips"] + list(hu_cols)].rename(columns=hu_cols), on="fips", how="left")
coords = gaz[["GEOID", "INTPTLAT", "INTPTLONG"]].rename(columns={"GEOID": "fips", "INTPTLAT": "lat", "INTPTLONG": "lon"})
cty = cty.merge(coords, on="fips", how="left")
assert cty.lat.notna().all() and cty.hu2005.notna().all(), "county join failed"


# ---- metros (2009 definitions): add up their counties ----

cbsa = csv("cbsa-est2009-alldata.csv", dtype={"CBSA": str, "STCOU": str})
metro_rows = cbsa[cbsa.LSAD == "Metropolitan Statistical Area"].copy()
county_rows = cbsa[(cbsa.LSAD == "County or equivalent") & cbsa.CBSA.isin(metro_rows.CBSA)].copy()
county_rows["fips"] = county_rows.STCOU.str.zfill(5)

crosswalk = county_rows[["fips", "CBSA"]].drop_duplicates().rename(columns={"CBSA": "code"})
assert crosswalk.fips.isin(cty.fips).all(), "crosswalk county missing"

c2 = cty.merge(crosswalk, on="fips")
met = c2.groupby("code")[[f"p{y}" for y in DECADE] + [f"hu{y}" for y in DECADE]].sum()

# metro centre = population-weighted average of its counties' coordinates
c2["wlat"] = c2.lat * c2.p2005
c2["wlon"] = c2.lon * c2.p2005
centre = c2.groupby("code")[["wlat", "wlon", "p2005"]].sum()
met["lat"] = centre.wlat / centre.p2005
met["lon"] = centre.wlon / centre.p2005

# components of change: natural increase, domestic and international migration
components = metro_rows.set_index("CBSA")
keep = {"NAME": "name"}
for y in range(2001, 2010):
    keep |= {f"NATURALINC{y}": f"nat{y}", f"DOMESTICMIG{y}": f"dm{y}", f"INTERNATIONALMIG{y}": f"im{y}"}
met = met.join(components[list(keep)].rename(columns=keep))
met.index.name = "code"
# yearly natural increase rate, averaged over 2002-2005
met["natinc_rate_0205"] = sum(met[f"nat{y}"] / met[f"p{y-1}"] for y in (2002, 2003, 2004, 2005)) / 4


# ---- IRS county-to-county flows ----
# The old files come as fixed-width text (.dat), except 2003-04, which only has one spreadsheet per state.
# Each row is origin state, origin county, destination state, destination county, returns, exemptions.
# Exemptions are roughly the number of people. Codes 96-98 in the state column are summary rows
# (97 000 = all moves within the US, including the small flows that are hidden as individual rows).

def parse_dat(text):
    rows = []
    for line in text.splitlines():
        if len(line) < 60 or not line[:2].strip().isdigit():
            continue
        nums = line[50:].split()
        if len(nums) < 2:
            continue
        rows.append((line[0:2], line[3:6], line[7:9], line[10:13], int(nums[0]), int(nums[1])))
    return rows


def parse_xls(data):
    import xlrd

    sheet = xlrd.open_workbook(file_contents=data).sheet_by_index(0)
    rows = []
    for i in range(sheet.nrows):
        v = sheet.row_values(i)
        if not (isinstance(v[0], str) and v[0].strip().isdigit() and len(v[0].strip()) <= 2):
            continue
        try:
            rows.append((v[0].strip().zfill(2), v[1].strip().zfill(3), v[2].strip().zfill(2), v[3].strip().zfill(3),
                         int(v[6]), int(v[7])))
        except (ValueError, TypeError):  # hidden (-1) or blank
            continue
    return rows


flows, totals, county_out, county_in = [], [], [], []
for y in YEARS:
    irs_zip = zipfile.ZipFile(RAW / f"county{y}.zip")
    names = irs_zip.namelist()
    out_files = [n for n in names if re.fullmatch(r"(countyout\d{4}(us1)?|co\d{4}us)\.dat", n.lower())]
    if out_files:
        assert len(out_files) == 1, (y, out_files)
        rows = parse_dat(irs_zip.read(out_files[0]).decode("latin1"))
    else:
        rows = []
        for n in names:
            if (n.lower().endswith("o.xls") or n.lower().endswith("0.xls")) and "outflow" in n.lower():
                rows += parse_xls(irs_zip.read(n))

    df = pd.DataFrame(rows, columns=["os", "oc", "ds", "dc", "returns", "exemptions"])
    df["orig"] = df.os + df.oc
    df["dest"] = df.ds + df.dc

    # total moves out of the New Orleans parishes, hidden flows included (their "97 000" rows)
    nola_out = df[df.orig.isin(NOLA_PARISHES) & (df.dest.str[:2] == "97") & (df.dc == "000")].exemptions.sum()

    # real county-to-county flows: actual states (codes up to 56), actual counties, not moves within a county
    real = df[(df.os.astype(int) <= 56) & (df.ds.astype(int) <= 56) & (df.oc != "000") & (df.dc != "000")
              & (df.orig != df.dest)]
    real = real[(real.exemptions >= 0) & (real.returns >= 0)].drop_duplicates(["orig", "dest"])
    flows.append(real.assign(year=y)[["year", "orig", "dest", "returns", "exemptions"]])
    totals.append(dict(year=y, nola_out_total=int(nola_out)))

    us_total = df[(df.ds == "97") & (df.dc == "000") & (df.oc != "000") & (df.exemptions >= 0)].drop_duplicates("orig")
    county_out.append(us_total.assign(year=y)[["year", "orig", "exemptions"]]
                      .rename(columns={"orig": "fips", "exemptions": "out_total"}))
    print(y, "flows:", len(real), "NOLA total out:", nola_out)
F = pd.concat(flows, ignore_index=True)

# Moves into New Orleans need the inflow files, where the destination comes first.
for t in totals:
    y = t["year"]
    irs_zip = zipfile.ZipFile(RAW / f"county{y}.zip")
    names = irs_zip.namelist()
    in_files = [n for n in names if re.fullmatch(r"(countyin\d{4}(us1)?|ci\d{4}us)\.dat", n.lower())]
    if in_files:
        rows = parse_dat(irs_zip.read(in_files[0]).decode("latin1"))
        df = pd.DataFrame(rows, columns=["ds", "dc", "os", "oc", "returns", "exemptions"])
        nola_rows = (df.ds + df.dc).isin(NOLA_PARISHES) & (df.os == "97") & (df.oc == "000")
        t["nola_in_total"] = int(df[nola_rows].exemptions.sum())
    else:  # 2003-04: Louisiana spreadsheet for the New Orleans total, then all states for the county totals
        la_file = [n for n in names if "inflow" in n.lower() and n.lower().endswith("lai.xls")][0]
        df = pd.DataFrame(parse_xls(irs_zip.read(la_file)), columns=["ds", "dc", "os", "oc", "returns", "exemptions"])
        nola_rows = (df.ds + df.dc).isin(NOLA_PARISHES) & (df.os == "97") & (df.oc == "000")
        t["nola_in_total"] = int(df[nola_rows].exemptions.sum())
        rows = []
        for n in names:
            if "inflow" in n.lower() and n.lower().endswith("i.xls"):
                rows += parse_xls(irs_zip.read(n))
        df = pd.DataFrame(rows, columns=["ds", "dc", "os", "oc", "returns", "exemptions"])

    us_total = df[(df.os == "97") & (df.oc == "000") & (df.dc != "000") & (df.exemptions >= 0)]
    us_total = us_total.assign(fips=lambda x: x.ds + x.dc).drop_duplicates("fips")
    county_in.append(us_total.assign(year=y)[["year", "fips", "exemptions"]].rename(columns={"exemptions": "in_total"}))


# ---- write ----

DATA.mkdir(exist_ok=True)
HEADER = "# Built by 00_build_data.py from raw/ (see raw/MANIFEST.txt). Do not edit by hand.\n"


def write(df, name, index=False):
    with open(DATA / name, "w") as f:
        f.write(HEADER)
        df.to_csv(f, sep="|", index=index)


write(cty, "counties.psv")
write(met.reset_index(), "metros.psv")
write(crosswalk, "county_metro.psv")
write(pd.DataFrame(totals), "irs_nola_totals.psv")
county_totals = pd.concat(county_out).merge(pd.concat(county_in), on=["year", "fips"], how="outer")
write(county_totals.sort_values(["fips", "year"]), "irs_county_totals.psv")
with gzip.open(DATA / "irs_county_flows.psv.gz", "wt") as f:
    f.write(HEADER)
    F.to_csv(f, sep="|", index=False)


# ---- check against the files I extracted by hand ----

legacy = DATA / "legacy"
old = pd.read_csv(legacy / "irs_nola_flows_2004_2008.psv", sep="|", comment="#", dtype={"fips": str})
old = old.set_index("fips").fillna(0)
for y in ["0405", "0506", "0607", "0708"]:
    year = F[F.year == y]
    out_ = year[year.orig.isin(NOLA_PARISHES) & ~year.dest.isin(NOLA_PARISHES)].groupby("dest").exemptions.sum()
    in_ = year[year.dest.isin(NOLA_PARISHES) & ~year.orig.isin(NOLA_PARISHES)].groupby("orig").exemptions.sum()
    assert (out_.reindex(old.index).fillna(0) == old[f"o{y}"]).all() and out_.sum() == old[f"o{y}"].sum(), f"out {y} mismatch"
    assert (in_.reindex(old.index).fillna(0) == old[f"i{y}"]).all() and in_.sum() == old[f"i{y}"].sum(), f"in {y} mismatch"

old_metros = pd.read_csv(legacy / "cbsa_metros_2009def.psv", sep="|", comment="#", dtype={"code": str}).set_index("code")
check_cols = ["p2001", "p2004", "p2005", "p2006", "p2007"]
assert (met.loc[old_metros.index, check_cols].values == old_metros[check_cols].values).all(), "metro populations mismatch"
print("check against hand-extracted files: OK")
print(pd.DataFrame(totals).to_string(index=False))
