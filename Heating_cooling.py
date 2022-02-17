#This program calculates the heating and cooling degree days of CA, CO, and the cooling
#degree days of DNK and ESP. We use a threshold of 17˚C for HDD.

import pandas as pd

heatCA = pd.read_csv("data/TemperatureData/ninja_2011_weather_country_US.CA_merra-2_population_weighted.csv",  header = 2, index_col=0)
heatCA.index = pd.to_datetime(heatCA.index)
heatCO = pd.read_csv("data/TemperatureData/ninja_2011_weather_country_US.CO_merra-2_population_weighted.csv",  header = 2, index_col=0)
heatCO.index = pd.to_datetime(heatCO.index)

heatCA["HDD"] = heatCA.apply(lambda row: 17 - row["temperature"] if row["temperature"] < 17 else 0, axis = 1)
heatCO["HDD"] = heatCO.apply(lambda row: 17 - row["temperature"] if row["temperature"] < 17 else 0, axis = 1)

heatCA["CDD"] = heatCA.apply(lambda row: row["temperature"] - 18 if row["temperature"] > 18 else 0, axis = 1)
heatCO["CDD"] = heatCO.apply(lambda row: row["temperature"] - 18 if row["temperature"] > 18 else 0, axis = 1)


df_cal_elec = pd.read_csv('data_extra/CaliforniaTimeSeries.csv', index_col=0)
df_cal_elec.index = pd.to_datetime(df_cal_elec.index)
df_co_elec = pd.read_csv('data_extra/ColoradoTimeSeries.csv', index_col=0)
df_co_elec.index = pd.to_datetime(df_co_elec.index)

df_cal_elec["HDD"] = heatCA["HDD"]
df_cal_elec["CDD"] = heatCA["CDD"]
df_cal_elec["heating_demand"] = df_cal_elec.apply(lambda row: 1715 * row["HDD"] + 6356, axis = 1)# 1715 is California's G factor, MWh/HDD. 6356 is the constant, that we get from water heating
df_cal_elec["adjust_elec_demand"] =  df_cal_elec["demand_mwh"] + 1/3 * df_cal_elec["heating_demand"]

df_co_elec["HDD"] = heatCO["HDD"]
df_co_elec["CDD"] = heatCO["CDD"]
df_co_elec["heating_demand"]= df_co_elec.apply(lambda row: 1782 * row["HDD"] + 6472, axis = 1)
df_co_elec["adjust_elec_demand"] =  df_co_elec["demand_mwh"] + 1/3 * df_co_elec["heating_demand"]

#print(heatCO["HDD"].sum())

#print(df_co_elec["demand_mwh"].sum())
print(df_co_elec[4000:4020])