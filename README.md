Inputs: The only required file outside of the .py file is the NEF_loans.csv file. 

NEF Mapping Pipeline (v5)

This project generates county-level maps showing how NEF lending aligns with economic need across Iowa, Nebraska, and South Dakota. The script integrates NEF loan data, ACS poverty and population data, and TIGER/Line shapefiles to identify high-poverty or low-population rural counties and visualize where NEF lending is occurring.

The pipeline:
Fetches ACS data and downloads required shapefiles
Builds a ZCTA → County crosswalk
Aggregates loan counts to counties
Creates two maps:
     Story Categories Map
     Poverty + Lending Intensity Map

A GitHub Actions workflow automates execution and uploads the resulting PNG map files.
