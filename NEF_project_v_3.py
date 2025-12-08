import os
import zipfile
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# --------------------------
# Configuration
# --------------------------
ACSYear = "2023"

# NEF-style thresholds
highPOVthreshold = 10.0     # percent of people below poverty line
lowPOPthreshold = 5000      # total county population

# ACS endpoints
ACS_subject_base = f"https://api.census.gov/data/{ACSYear}/acs/acs5/subject"
ACS_profile_base = f"https://api.census.gov/data/{ACSYear}/acs/acs5/profile"

# TIGER/Line shapefiles
ZCTA_SHP_URL = "https://www2.census.gov/geo/tiger/TIGER2023/ZCTA520/tl_2023_us_zcta520.zip"
ZCTA_SHP_NAME = "tl_2023_us_zcta520.shp"

county_SHP_URL = "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
county_SHP_NAME = "tl_2023_us_county.shp"

state_SHP_URL = "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"
state_SHP_NAME = "tl_2023_us_state.shp"

# FIPS codes for Nebraska, Iowa, South Dakota
FIPS = ["19", "31", "46"]


# --------------------------
# Shapefiles
# --------------------------
def download_and_extract_zip(url: str, shp_name: str, extract_path: str = "."):
    """
    Downloads a zip file from the given URL and extracts its contents
    if the main shapefile isn't already present.
    """
    if os.path.exists(shp_name):
        print(f"{shp_name} already exists, skipping download.")
        return

    zip_filename = url.split("/")[-1]
    print(f"Downloading {zip_filename} from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(zip_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    os.remove(zip_filename)
    print(f"Extracting {zip_filename}...")
    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    print(f"{shp_name} ready.")


# --------------------------
# ACS County Data
# --------------------------
def fetch_acs_poverty_income_county() -> pd.DataFrame:
    """
    County-level poverty rate and median income from ACS subject tables.

    S1701_C03_001E: Percent of people below poverty level (estimate)
    S1901_C01_012E: Median household income (estimate)
    """
    params = {
        "get": "S1701_C03_001E,S1901_C01_012E,NAME",
        "for": "county:*",
    }

    print("Requesting ACS county-level poverty & income data...")
    response = requests.get(ACS_subject_base, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data[1:], columns=data[0])

    # state, county columns are FIPS codes
    df["state"] = df["state"].astype(str).str.zfill(2)
    df["county"] = df["county"].astype(str).str.zfill(3)
    df["county_geoid"] = df["state"] + df["county"]

    # Restrict to IA, NE, SD
    df = df[df["state"].isin(FIPS)].copy()

    df = df.rename(
        columns={
            "S1701_C03_001E": "poverty_rate",
            "S1901_C01_012E": "median_income",
        }
    )

    df["poverty_rate"] = pd.to_numeric(df["poverty_rate"], errors="coerce")
    df["median_income"] = pd.to_numeric(df["median_income"], errors="coerce")

    df = df[["county_geoid", "NAME", "poverty_rate", "median_income"]]
    print(f"Retrieved poverty & income for {len(df)} counties (IA/NE/SD).")
    return df


def fetch_acs_population_county() -> pd.DataFrame:
    """
    County-level total population from ACS profile tables.

    DP05_0001E: Total population estimate.
    """
    params = {
        "get": "DP05_0001E,NAME",
        "for": "county:*",
    }

    print("Requesting ACS county-level population data...")
    response = requests.get(ACS_profile_base, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data[1:], columns=data[0])

    df["state"] = df["state"].astype(str).str.zfill(2)
    df["county"] = df["county"].astype(str).str.zfill(3)
    df["county_geoid"] = df["state"] + df["county"]

    # Restrict to IA, NE, SD
    df = df[df["state"].isin(FIPS)].copy()

    df = df.rename(columns={"DP05_0001E": "population"})
    df["population"] = pd.to_numeric(df["population"], errors="coerce")

    df = df[["county_geoid", "population"]]
    print(f"Retrieved population for {len(df)} counties (IA/NE/SD).")
    return df


# --------------------------
# NEF loans locations: Zip Code -> County
# --------------------------
def load_nef_loans(path: str = "nef_loans.csv") -> pd.DataFrame:
    """
    Load NEF loans file and normalize zip to 5-digit code.
    Returns loan counts per ZIP/ZCTA-like code.
    """
    print(f"Loading NEF loans from {path}...")
    loans = pd.read_csv(path)

    if "zip" not in loans.columns:
        raise ValueError("nef_loans.csv must contain a column named 'zip'.")

    loans["zip_zcta"] = pd.to_numeric(loans["zip"], errors="coerce")
    loans = loans.dropna(subset=["zip_zcta"])
    loans["zip_zcta"] = loans["zip_zcta"].astype(int).astype(str).str.zfill(5)

    loan_counts_zcta = loans.groupby("zip_zcta").size().reset_index(name="loan_count")
    loan_counts_zcta = loan_counts_zcta.rename(columns={"zip_zcta": "zcta"})

    print(f"Found loans in {len(loan_counts_zcta)} distinct ZIPs/ZCTAs.")
    return loan_counts_zcta


def build_zcta_to_county_crosswalk(zcta_shp_name: str,
                                   county_shp_name: str,
                                   state_shp_name: str) -> pd.DataFrame:
    """
    Build a ZCTA -> county crosswalk for IA/NE/SD using spatial join:

    1. Limit ZCTAs to the tri-state region via state polygons
    2. Take ZCTA centroids
    3. Spatially join to counties (within)
    """
    print("Loading state, county, and ZCTA geometries for crosswalk...")

    # States for 3-state region
    states = gpd.read_file(state_shp_name)[["STATEFP", "NAME", "geometry"]]
    states = states[states["STATEFP"].isin(FIPS)].copy()

    # Counties (full US, then filter)
    counties = gpd.read_file(county_shp_name)
    counties = counties[counties["STATEFP"].isin(FIPS)].copy()

    # ZCTAs (full US)
    zctas = gpd.read_file(zcta_shp_name)
    zctas["ZCTA5CE20"] = zctas["ZCTA5CE20"].astype(str).str.zfill(5)

    # Reproject everything to equal-area CRS
    states = states.to_crs(epsg=2163)
    counties = counties.to_crs(epsg=2163)
    zctas = zctas.to_crs(epsg=2163)

    # Restrict ZCTAs to tri-state region by intersection with union of states
    tri_union = states.unary_union
    zctas_tri = zctas[zctas.geometry.intersects(tri_union)].copy()
    print(f"ZCTAs intersecting IA/NE/SD: {len(zctas_tri)}")

    # ZCTA centroids
    zctas_tri["centroid"] = zctas_tri.geometry.centroid
    zcta_pts = gpd.GeoDataFrame(
        zctas_tri[["ZCTA5CE20", "centroid"]],
        geometry="centroid",
        crs=zctas_tri.crs,
    )

    # Spatial join: which county contains each ZCTA centroid
    print("Performing spatial join ZCTA centroids -> counties...")
    cross = gpd.sjoin(
        zcta_pts,
        counties[["GEOID", "geometry"]],
        how="inner",
        predicate="within",
    )

    crosswalk = cross[["ZCTA5CE20", "GEOID"]].drop_duplicates()
    crosswalk = crosswalk.rename(
        columns={"ZCTA5CE20": "zcta", "GEOID": "county_geoid"}
    )

    print(f"Built ZCTA->county crosswalk with {len(crosswalk)} rows.")
    return crosswalk


def aggregate_loans_to_county(
    loan_counts_zcta: pd.DataFrame,
    zcta_to_county: pd.DataFrame
) -> pd.DataFrame:
    """
    Join loan counts per ZCTA to county via crosswalk and sum by county.
    """
    merged = loan_counts_zcta.merge(zcta_to_county, on="zcta", how="left")

    # Drop ZCTAs that didn't match a county (e.g., PO Box-only ZCTAs)
    merged = merged.dropna(subset=["county_geoid"]).copy()

    loans_county = (
        merged
        .groupby("county_geoid")["loan_count"]
        .sum()
        .reset_index()
    )

    print(f"Aggregated to {len(loans_county)} counties with at least one NEF loan.")
    return loans_county


# --------------------------
# Main script
# --------------------------
def main():
    # 1. Download shapefiles if needed
    download_and_extract_zip(ZCTA_SHP_URL, ZCTA_SHP_NAME)
    download_and_extract_zip(county_SHP_URL, county_SHP_NAME)
    download_and_extract_zip(state_SHP_URL, state_SHP_NAME)

    # 2. Fetch ACS county-level data
    acs_pov_inc = fetch_acs_poverty_income_county()
    acs_pop = fetch_acs_population_county()

    # 3. Build a ZCTA->county crosswalk and aggregate NEF loans to county
    loan_counts_zcta = load_nef_loans("nef_loans.csv")
    zcta_to_county = build_zcta_to_county_crosswalk(
        ZCTA_SHP_NAME,
        county_SHP_NAME,
        state_SHP_NAME,
    )
    loans_county = aggregate_loans_to_county(loan_counts_zcta, zcta_to_county)

    # 4. Load counties & states, compute land area for counties
    print("Loading county and state geometries for mapping...")
    counties = gpd.read_file(county_SHP_NAME)
    counties = counties[counties["STATEFP"].isin(FIPS)].copy()

    states = gpd.read_file(state_SHP_NAME)[["STATEFP", "NAME", "geometry"]]
    states = states[states["STATEFP"].isin(FIPS)].copy()

    # Reproject to equal-area CRS
    counties = counties.to_crs(epsg=2163)
    states = states.to_crs(epsg=2163)

    # County GEOID (state+county) matches ACS county_geoid
    counties["county_geoid"] = counties["STATEFP"] + counties["COUNTYFP"]

    # Compute land area for counties in square miles
    SQM_TO_SQMI = 3.861021585424458e-7
    counties["land_area_sq_miles"] = counties.geometry.area * SQM_TO_SQMI

    land_df = counties[["county_geoid", "land_area_sq_miles"]].copy()

    # 5. Merge ACS pieces + land area + loans to get county-level data frame
    print("Merging ACS county data, land area, and NEF loans...")
    acs_all = (
        acs_pov_inc
        .merge(acs_pop, on="county_geoid", how="left")
        .merge(land_df, on="county_geoid", how="left")
        .merge(loans_county, on="county_geoid", how="left")
    )

    acs_all["loan_count"] = acs_all["loan_count"].fillna(0)
    acs_all["population_density"] = (
        acs_all["population"] / acs_all["land_area_sq_miles"]
    )

    # Normalize loan metrics
    acs_all["loans_per_1k_pop"] = np.where(
        acs_all["population"] > 0,
        acs_all["loan_count"] / acs_all["population"] * 1000,
        np.nan,
    )

    acs_all["people_in_poverty"] = (
        acs_all["population"] * acs_all["poverty_rate"] / 100.0
    )

    acs_all["loans_per_1k_poor"] = np.where(
        acs_all["people_in_poverty"] > 50,
        acs_all["loan_count"] / acs_all["people_in_poverty"] * 1000,
        np.nan,
    )

    # 6. Build a GeoDataFrame for counties with all attributes
    county_map = counties.merge(acs_all, on="county_geoid", how="left")

    # Centroids for overlay points (counties with loans)
    county_map["centroid"] = county_map.geometry.centroid
    centroids_gdf = county_map[county_map["loan_count"] > 0].copy()
    centroids_gdf = gpd.GeoDataFrame(
        centroids_gdf,
        geometry="centroid",
        crs=county_map.crs,
    )

    # -----------------------
    # Story Categories Map
    # -----------------------
    print("Computing story categories (using fixed NEF-style thresholds)...")

    county_map["has_loans"] = county_map["loan_count"] > 0
    county_map["high_poverty"] = county_map["poverty_rate"] >= highPOVthreshold
    county_map["low_population"] = county_map["population"] <= lowPOPthreshold

    def story_label(row):
        if row["high_poverty"] and row["has_loans"]:
            return "High poverty (\u226510%), NEF loans"
        elif row["high_poverty"] and not row["has_loans"]:
            return "High poverty (\u226510%), no NEF loans"
        elif row["low_population"] and row["has_loans"]:
            return "Low population (\u22645k), NEF loans"
        elif row["low_population"] and not row["has_loans"]:
            return "Low population (\u22645k), no NEF loans"
        else:
            return "Other"

    county_map["story_cat"] = county_map.apply(story_label, axis=1)

    story_order = [
        "High poverty (\u226510%), no NEF loans",
        "High poverty (\u226510%), NEF loans",
        "Low population (\u22645k), no NEF loans",
        "Low population (\u22645k), NEF loans",
        "Other",
    ]

    color_map = {
        "High poverty (\u226510%), no NEF loans": "#b30000",   # dark red (priority gap)
        "High poverty (\u226510%), NEF loans": "#fd8d3c",      # orange
        "Low population (\u22645k), no NEF loans": "#3182bd", # medium blue
        "Low population (\u22645k), NEF loans": "#9ecae1",    # light blue
        "Other": "#d9d9d9",                              # light gray
    }

    county_map["story_color"] = county_map["story_cat"].map(color_map)

    print("Creating county-level story categories map...")
    fig, ax = plt.subplots(figsize=(12, 10))

    county_map.plot(
        ax=ax,
        color=county_map["story_color"],
        linewidth=0.3,
        edgecolor="white",
    )

    states.boundary.plot(ax=ax, linewidth=1.0, edgecolor="black")

    ax.set_title(
        "Where NEF Lending Reaches (and Misses) High-Need Rural Counties\n"
        "Iowa, Nebraska, and South Dakota\n"
        f"(High poverty \u2265 {highPOVthreshold}%, Low population \u2264 {lowPOPthreshold:,})",
        fontsize=15,
    )
    ax.axis("off")

    legend_elements = [
        Patch(facecolor=color_map[cat], edgecolor="black", label=cat)
        for cat in story_order
    ]
    ax.legend(
        handles=legend_elements,
        title="County type",
        loc="upper right",
        frameon=True,
    )

    plt.tight_layout()
    out1 = "map_county_story_categories.png"
    plt.savefig(out1, dpi=300)
    plt.close()
    print(f"Saved {out1}")

    # -----------------------
    # Poverty and Loans Per 1000 Residents Map
    # -----------------------
    print("Creating county-level poverty + NEF loans per 1k residents map...")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Choropleth: poverty rate (quantiles still useful for visual contrast)
    county_map.plot(
        column="poverty_rate",
        cmap="Reds",
        linewidth=0.3,
        ax=ax,
        legend=True,
        scheme="quantiles",
        k=5,
        legend_kwds={
            "title": "Poverty rate (%) \u2013 county quantiles"
        },
    )

    # Overlay points: NEF loans per 1k residents
    pts = centroids_gdf.copy()
    loan_intensity = pts["loans_per_1k_pop"].replace(0, np.nan)

    if loan_intensity.notna().any():
        min_val = loan_intensity.min()
        min_val = max(min_val, 1e-6)  # avoid log(0)
        loan_intensity = loan_intensity.clip(lower=min_val)

        size = 20 + 150 * (np.log10(loan_intensity) - np.log10(min_val))
    else:
        size = pd.Series(20, index=pts.index)

    pts.plot(
        ax=ax,
        markersize=size,
        color="blue",
        alpha=0.6,
        marker="o",
        label="NEF loans (per 1,000 residents)",
    )

    states.boundary.plot(ax=ax, linewidth=1.0, edgecolor="black")
    counties.boundary.plot(ax=ax, linewidth=0.3, edgecolor="gray") # Added county lines

    ax.set_title(
        "NEF Lending Intensity Overlaid on County Poverty Rates\n"
        "Iowa, Nebraska, and South Dakota",
        fontsize=15,
    )
    ax.axis("off")
    plt.legend(loc="lower left")

    plt.tight_layout()
    out2 = "map_county_poverty_loans_per_capita.png"
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"Saved {out2}")

    print("All done.")


if __name__ == "__main__":
    main()
