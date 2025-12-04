Input files in the project folder:

    nef_loans.csv
    acs_poverty_income.csv
    acs_population_area.csv
    tl_2022_us_zcta520.shp (plus shapefile components) (included in the acs_population_area.csv code)

Outputs:

Map: poverty + per-capita lending
     Darker red = higher poverty.
     Bigger blue circles = more loans per 1,000 residents.
     You can visually say: “Do big circles tend to be in darker red places?”

Scatterplots:
Poverty rate vs loans_per_1k_pop
     Upward pattern → NEF lends more intensely in higher-poverty areas.
Median income vs loans_per_1k_pop
     Downward pattern → NEF concentrates more in lower-income areas.
Population density vs loans_per_1k_pop
     Shape tells you if you’re more active in urban or rural contexts.
