"""Download the raw data from the official sources into raw/ and record a checksum for each file.

Run once with:  python3 00_download.py
That's about 75 MB. raw/ itself isn't in git, but raw/MANIFEST.txt is, so you can check you have the same files.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
RAW.mkdir(exist_ok=True)

IRS = "https://www.irs.gov/pub/irs-soi/"
CEN = "https://www2.census.gov/"
NEW_IRS_YEARS = ["0809", "0910", "1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718", "1819", "1920",
                 "2021", "2122", "2223"]

SOURCES = {
    # IRS county-to-county migration, one file per pair of filing years (Katrina period)
    "county0304.zip": IRS + "2003to2004countymigration.zip",
    "county0405.zip": IRS + "county0405.zip",
    "county0506.zip": IRS + "county0506.zip",
    "county0607.zip": IRS + "county0607.zip",
    "county0708.zip": IRS + "county0708.zip",
    # Census population and housing estimates for 2000-2009, and metro components of change
    "co-est00int-tot.csv": CEN + "programs-surveys/popest/datasets/2000-2010/intercensal/county/co-est00int-tot.csv",
    "hu-est00int-tot.csv": CEN + "programs-surveys/popest/datasets/2000-2010/intercensal/housing/hu-est00int-tot.csv",
    "cbsa-est2009-alldata.csv": CEN + "programs-surveys/popest/datasets/2000-2009/metro/totals/cbsa-est2009-alldata.csv",
    # Later disasters: newer IRS files (csv format from 2008-09 on) and Census estimates to 2020
    **{f"countyoutflow{y}.csv": IRS + f"countyoutflow{y}.csv" for y in NEW_IRS_YEARS},
    **{f"countyinflow{y}.csv": IRS + f"countyinflow{y}.csv" for y in NEW_IRS_YEARS},
    "co-est2020-alldata.csv": CEN + "programs-surveys/popest/datasets/2010-2020/counties/totals/co-est2020-alldata.csv",
    "HU-EST2020_6.csv": CEN + "programs-surveys/popest/datasets/2010-2020/housing/HU-EST2020_6.csv",  # California
    "HU-EST2020_48.csv": CEN + "programs-surveys/popest/datasets/2010-2020/housing/HU-EST2020_48.csv",  # Texas
    # County coordinates (2010 Gazetteer)
    "Gaz_counties_national.zip": CEN + "geo/docs/maps-data/data/gazetteer/Gaz_counties_national.zip",
    # ZIP-code housing units, 2000 and 2010 (used in 16_damage_geography.py)
    "zcta5.zip": CEN + "geo/docs/maps-data/data/gazetteer/zcta5.zip",
    "Gaz_zcta_national.zip": CEN + "geo/docs/maps-data/data/gazetteer/Gaz_zcta_national.zip",
    # Housing occupancy in the 2000 and 2010 Census profiles
    "2khxx.zip": CEN + "census_2000/datasets/demographic_profile/0_All_State/2khxx.zip",
    "tx2010.dp.zip": CEN + "census_2010/03-Demographic_Profile/Texas/tx2010.dp.zip",
    "ca2010.dp.zip": CEN + "census_2010/03-Demographic_Profile/California/ca2010.dp.zip",
    # Census estimates to 2025
    "co-est2025-alldata.csv": CEN + "programs-surveys/popest/datasets/2020-2025/counties/totals/co-est2025-alldata.csv",
    # State outlines for the map
    "cb_2018_us_state_20m.zip": CEN + "geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    lines = []
    for name, url in SOURCES.items():
        dest = RAW / name
        if not dest.exists():
            print("downloading", url, file=sys.stderr)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; cities-project)"})
            with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
                f.write(r.read())
        lines.append(f"{sha256(dest)}  {dest.stat().st_size:>10d}  {name}  {url}")
    (RAW / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
