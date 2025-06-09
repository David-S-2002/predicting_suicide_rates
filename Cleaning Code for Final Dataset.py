import pandas as pd

anxiety_df=pd.read_csv(r"Anxiety2021.csv")
anxiety = anxiety_df[["location","sex","age","val"]] 
anxiety = anxiety.rename(columns={"val": "anxiety_percent"})


bipolar_df=pd.read_csv(r"Bipolar2021.csv")
bipolar = bipolar_df[["location","sex","val"]] 
bipolar = bipolar.rename(columns={"val": "bipolar_percent"})


depress_df=pd.read_csv(r"Depression2021.csv")
depress = depress_df[["location","sex","val"]] 
depress = depress.rename(columns={"val": "depress_percent"})


suicide_df=pd.read_csv(r"Suicide2021.csv")
suicide = suicide_df[["location","sex","val"]] 
suicide = suicide.rename(columns={"val": "suicide_rate_per100k"})


gdp_df=pd.read_csv(r"GDP.csv")

gdp_long = gdp_df.drop(columns=["Indicator Name", "Indicator Code"])


gdp_long = gdp_long.rename(columns={"Country Name": "location", "Country Code": "country_code"})

# Converting the individual year columns to a single column
gdp = gdp_long.melt(
    id_vars=["location", "country_code"],
    var_name="year",
    value_name="gdp_usd"
)

gdp["year"] = pd.to_numeric(gdp["year"], errors="coerce")

# Getting only the year 2021
gdp_2021 = gdp[gdp["year"] == 2021].copy()

# Using the list of locations from Anxiety dataset to compare and join with GDP dataset:
valid_locations = set(anxiety["location"].unique())
gdp_2021 = gdp_2021[gdp_2021["location"].isin(valid_locations)]

# Merge anxiety and bipolar
merged = pd.merge(anxiety, bipolar, on=["location", "sex"], how="inner")

# Adding depression
merged = pd.merge(merged, depress, on=["location", "sex"], how="inner")

# Adding suicide rate
merged = pd.merge(merged, suicide, on=["location", "sex"], how="inner")

# Adding GDP
merged = pd.merge(merged, gdp_2021, on=["location"], how="inner")

merged

merged.to_csv("intermediate_suicide_2021.csv", index=False)
