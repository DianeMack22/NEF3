# 
# Create acs_poverty_income.csv
# 

import requests
import pandas as pd

# ACS 5-year dataset
YEAR = "2022"   # Use the most recent ACS available
BASE_URL = f"https://api.census.gov/data/{YEAR}/acs/acs5/subject"

params = {
    "get": "S1701_C03_001E,S1901_C01_012E,NAME",
    "for": "zip code tabulation area:*"
}

response = requests.get(BASE_URL, params=params)
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data[1:], columns=data[0])

# Rename fields
df = df.rename(columns={
    "S1701_C03_001E": "poverty_rate",
    "S1901_C01_012E": "median_income",
    "zip code tabulation area": "zcta"
})

# Convert numeric fields
df["poverty_rate"] = pd.to_numeric(df["poverty_rate"], errors="coerce")
df["median_income"] = pd.to_numeric(df["median_income"], errors="coerce")

# Keep only necessary fields
df = df[["zcta", "poverty_rate", "median_income"]]

df.to_csv("acs_poverty_income.csv", index=False)

print("Created acs_poverty_income.csv!")

# 
# Create acs_population_area.csv
# 

import requests
import pandas as pd
import geopandas as gpd
import zipfile
import os

# --------------------------
# CONFIGURATION
# --------------------------
ACS_YEAR = "2022"
ACS_BASE = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5/profile"
SHAPEFILE_URL = "https://www2.census.gov/geo/tiger/TIGER2022/ZCTA520/tl_2022_us_zcta520.zip"
SHAPEFILE_NAME = "tl_2022_us_zcta520.shp"

def get_acs_population():
    """
    Fetch total population for all ZCTAs from ACS DP05 (2022).
    Uses DP05_0001E: Estimate!!Total population
    """
    params = {
        "get": "DP05_0001E,NAME",
        "for": "zip code tabulation area:*"
    }

    print("Requesting ACS 2022 population data for all ZCTAs...")
    response = requests.get(ACS_BASE, params=params)

    if response.status_code != 200:
        raise RuntimeError(f"ACS API request failed: {response.status_code} {response.text}")

    data = response.json()
    df = pd.DataFrame(data[1:], columns=data[0])

    # Clean up and rename
    df = df.rename(columns={
        "DP05_0001E": "population",
        "zip code tabulation area": "zcta"
    })

    df["zcta"] = df["zcta"].astype(str).str.zfill(5)
    df["population"] = pd.to_numeric(df["population"], errors="coerce")

    print(f"Retrieved population for {len(df)} ZCTAs.")
    return df[["zcta", "population"]]


def get_land_area_from_shapefile(shapefile_path=SHAPEFILE_NAME):
    """
    Load TIGER/Line ZCTA shapefile and compute land area in square miles.
    Uses an equal-area projection (EPSG:2163) for accurate area calc.
    """
    print(f"Loading ZCTA shapefile from {shapefile_path} ...")
    zcta_geo = gpd.read_file(shapefile_path)

    # Ensure consistent ZCTA code formatting
    zcta_geo["ZCTA5CE20"] = zcta_geo["ZCTA5CE20"].astype(str).str.zfill(5)

    # Project to an equal-area CRS (US National Atlas Equal Area)
    print("Reprojecting to EPSG:2163 for area calculations...")
    zcta_geo = zcta_geo.to_crs(epsg=2163)

    # Area in square meters, convert to square miles
    # 1 square meter = 3.861021585424458e-7 square miles
    SQM_TO_SQMI = 3.861021585424458e-7
    zcta_geo["land_area_sq_miles"] = zcta_geo["geometry"].area * SQM_TO_SQMI

    land_area_df = zcta_geo[["ZCTA5CE20", "land_area_sq_miles"]].rename(
        columns={"ZCTA5CE20": "zcta"}
    )

    print(f"Computed land area for {len(land_area_df)} ZCTAs.")
    return land_area_df


def download_and_extract_shapefile(url, target_name, extract_path="."):
    """
    Downloads a zip file from the given URL and extracts its contents.
    """
    zip_filename = url.split('/')[-1]
    print(f"Downloading {zip_filename} from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status() # Raise an exception for bad status codes

    with open(zip_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Extracting {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")

    # Clean up the downloaded zip file
    os.remove(zip_filename)

def main():
    # Download and extract shapefile if not already present
    if not os.path.exists(SHAPEFILE_NAME):
        download_and_extract_shapefile(SHAPEFILE_URL, SHAPEFILE_NAME)

    # 1. Get population from ACS
    pop_df = get_acs_population()

    # 2. Get land area from TIGER shapefile
    land_df = get_land_area_from_shapefile(SHAPEFILE_NAME)

    # 3. Merge population + land area
    print("Merging population and land area...")
    merged = pop_df.merge(land_df, on="zcta", how="left")

    # 4. Compute population density
    print("Computing population density (people per sq mile)...")
    merged["population_density"] = merged["population"] / merged["land_area_sq_miles"]

    # 5. Save to CSV
    output_file = "acs_population_area.csv"
    merged.to_csv(output_file, index=False)
    print(f"✅ Created {output_file} with {len(merged)} rows.")


if __name__ == "__main__":
    main()

# 
# Main script
# 

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Load NEF loan data
# -----------------------
loans = pd.read_csv("nef_loans.csv")

# Clean and format the zip column for merging to ZCTA
loans["zip_zcta"] = pd.to_numeric(loans["zip"], errors="coerce")
loans = loans.dropna(subset=["zip_zcta"])
loans["zip_zcta"] = loans["zip_zcta"].astype(int).astype(str).str.zfill(5)

# -----------------------
# Load ACS poverty + income
# zcta, poverty_rate, median_income
# -----------------------
acs = pd.read_csv("acs_poverty_income.csv")
acs["zcta"] = acs["zcta"].astype(str).str.zfill(5)

# -----------------------
# Load ACS population + land area for density
# zcta, population, land_area_sq_miles
# -----------------------
dens = pd.read_csv("acs_population_area.csv")
dens["zcta"] = dens["zcta"].astype(str).str.zfill(5)

# Compute population density if not already present
if "population_density" not in dens.columns:
    dens["population_density"] = dens["population"] / dens["land_area_sq_miles"]

# -----------------------
# Count loans per ZCTA
# -----------------------
loan_counts = loans.groupby("zip_zcta").size().reset_index(name="loan_count")
loan_counts = loan_counts.rename(columns={"zip_zcta": "zcta"})

# -----------------------
# Combine ACS + density + loans
# -----------------------
acs_all = (
    acs
    .merge(dens[["zcta", "population", "population_density"]], on="zcta", how="left")
    .merge(loan_counts, on="zcta", how="left")
)

acs_all["loan_count"] = acs_all["loan_count"].fillna(0)

# -----------------------
# Create normalized loan metrics
# -----------------------
# loans per 1,000 residents
acs_all["loans_per_1k_pop"] = np.where(
    acs_all["population"] > 0,
    acs_all["loan_count"] / acs_all["population"] * 1000,
    np.nan
)

# approximate number of people in poverty
acs_all["people_in_poverty"] = (
    acs_all["population"] * acs_all["poverty_rate"] / 100.0
)

# loans per 1,000 people in poverty (where that denominator isn't tiny)
acs_all["loans_per_1k_poor"] = np.where(
    acs_all["people_in_poverty"] > 50,  # avoid crazy ratios in tiny places
    acs_all["loan_count"] / acs_all["people_in_poverty"] * 1000,
    np.nan
)

# Filter to ZCTAs where NEF is actually active (at least 1 loan)
active = acs_all[acs_all["loan_count"] > 0].copy()

# -----------------------
# Load ZCTA shapefile (TIGER/Line) and join
# -----------------------
zcta_geo = gpd.read_file("tl_2022_us_zcta520.shp")
zcta_geo["ZCTA5CE20"] = zcta_geo["ZCTA5CE20"].astype(str).str.zfill(5)

# Reproject to an equal-area CRS for mapping
zcta_geo = zcta_geo.to_crs(epsg=2163)

acs_map = zcta_geo.merge(
    acs_all, left_on="ZCTA5CE20", right_on="zcta", how="left"
)

# Pre-compute centroids for point overlays (in same CRS)
acs_map["centroid"] = acs_map.geometry.centroid
centroids_gdf = acs_map.dropna(subset=["loan_count"]).copy()
centroids_gdf = gpd.GeoDataFrame(
    centroids_gdf,
    geometry="centroid",
    crs=acs_map.crs
)

# -----------------------
# 1) Map: Poverty rate + NEF loans per 1k residents
# -----------------------
fig, ax = plt.subplots(figsize=(12, 10))

# Choropleth: poverty rate
acs_map.plot(
    column="poverty_rate",
    cmap="Reds",
    linewidth=0,
    ax=ax,
    legend=True,
    legend_kwds={"label": "Poverty rate (%)"}
)

# Overlay points: NEF loans per 1k residents (size encodes intensity)
pts = centroids_gdf[centroids_gdf["loan_count"] > 0].copy()
# rescale marker size for display
size = pts["loans_per_1k_pop"].fillna(0)
size = 20 + 200 * (size / max(size.max(), 1))

pts.plot(
    ax=ax,
    markersize=size,
    color="blue",
    alpha=0.6,
    marker="o",
    label="NEF loans per 1,000 residents"
)

ax.set_title("NEF Lending Overlaid on Poverty Rates (by ZCTA)", fontsize=16)
ax.axis("off")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("map_poverty_loans_per_capita.png", dpi=300)
plt.show()

# -----------------------
# 2) Scatterplots: visual "correlations"
# -----------------------
scatter = active.dropna(
    subset=["poverty_rate", "median_income", "population_density", "loans_per_1k_pop"]
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# a) Poverty vs loans per 1k pop
axes[0].scatter(
    scatter["poverty_rate"],
    scatter["loans_per_1k_pop"],
    alpha=0.4,
    s=15
)
axes[0].set_xlabel("Poverty rate (%)")
axes[0].set_ylabel("NEF loans per 1,000 residents")
axes[0].set_title("Lending vs Poverty")

# b) Median income vs loans per 1k pop
axes[1].scatter(
    scatter["median_income"],
    scatter["loans_per_1k_pop"],
    alpha=0.4,
    s=15
)
axes[1].set_xlabel("Median household income ($)")
axes[1].set_title("Lending vs Income")

# c) Population density vs loans per 1k pop
axes[2].scatter(
    scatter["population_density"],
    scatter["loans_per_1k_pop"],
    alpha=0.4,
    s=15
)
axes[2].set_xlabel("Population density (people / sq mile)")
axes[2].set_title("Lending vs Population Density")

plt.suptitle("How NEF Lending Intensity Relates to Community Characteristics", fontsize=18)
plt.tight_layout()
plt.savefig("scatter_lending_vs_context.png", dpi=300)
plt.show()

print("Story-focused maps and scatterplots generated!")
