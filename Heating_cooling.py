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





df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0)# in MWh
df_elec.index = pd.to_datetime(df_elec.index) #change index to datatime

df_cal_elec = pd.read_csv('data_extra/CaliforniaTimeSeries.csv', index_col=0)
df_cal_elec.index = pd.to_datetime(df_cal_elec.index)

df_co_elec = pd.read_csv('data_extra/ColoradoTimeSeries.csv', index_col=0)
df_co_elec.index = pd.to_datetime(df_co_elec.index)

#We want to simulate electrification of heating. We can then add to Denmark and Spain
df_heat = pd.read_csv('data/heat_demand.csv', sep=';', index_col=0)# in MWh
df_heat.index = pd.to_datetime(df_heat.index) #change index to datatime
heatCA = pd.read_csv("data/TemperatureData/ninja_2011_weather_country_US.CA_merra-2_population_weighted.csv",  header = 2, index_col=0)
heatCA.index = pd.to_datetime(heatCA.index)
heatCO = pd.read_csv("data/TemperatureData/ninja_2011_weather_country_US.CO_merra-2_population_weighted.csv",  header = 2, index_col=0)
heatCO.index = pd.to_datetime(heatCO.index)



df_elec["DNKheat"] = df_heat["DNK"]
df_elec["ESPheat"] = df_heat["ESP"]

df_elec["DNKcombine"] = df_elec.apply(lambda row: row["DNK"] + row["DNKheat"]/3, axis = 1)
df_elec["ESPcombine"] = df_elec.apply(lambda row: row["ESP"] + row["ESPheat"]/3, axis = 1)

heatCA["HDD"] = heatCA.apply(lambda row: 17 - row["temperature"] if row["temperature"] < 17 else 0, axis = 1)
heatCO["HDD"] = heatCO.apply(lambda row: 17 - row["temperature"] if row["temperature"] < 17 else 0, axis = 1)

df_cal_elec["HDD"] = heatCA["HDD"]
df_cal_elec["heating_demand"] = df_cal_elec.apply(lambda row: 1715 * row["HDD"] + 6356, axis = 1)# 1715 is California's G factor, MWh/HDD. 6356 is the constant, that we get from water heating
df_cal_elec["adjust_elec_demand"] =  df_cal_elec["demand_mwh"] + 1/3 * df_cal_elec["heating_demand"]

df_co_elec["HDD"] = heatCO["HDD"]
df_co_elec["heating_demand"]= df_co_elec.apply(lambda row: 1782 * row["HDD"] + 6472, axis = 1)
df_co_elec["adjust_elec_demand"] =  df_co_elec["demand_mwh"] + 1/3 * df_co_elec["heating_demand"]


def increase_temp(degree_change, slope_factor):
    '''Here, I am trying to model what would happen to electricity demand in Denmark if
    the temperature increases uniformly by x degrees due to global warming
    
    For Denmark we assume that the electricity demand would be constant with change in 
    temperature until it reaches a threshold temperature (15.79 degrees). Then, there is
    a linear increase'''


    ESData= pd.read_csv('data/TemperatureData/ninja_weather_country_ES_merra-2_population_weighted.csv', skiprows = 2, index_col=0)
    ESData.index = pd.to_datetime(ESData.index)
    DKData = pd.read_csv('data/TemperatureData/ninja_weather_country_DK_merra-2_population_weighted.csv', skiprows = 2, index_col=0)
    DKData.index = pd.to_datetime(DKData.index)
    COData = pd.read_csv('data/TemperatureData/ninja_weather_country_US.CO_merra-2_population_weighted.csv', skiprows = 2, index_col=0)
    COData.index = pd.to_datetime(COData.index)
    CAData = pd.read_csv('data/TemperatureData/ninja_weather_country_US.CA_merra-2_population_weighted.csv', skiprows = 2, index_col=0)
    CAData.index = pd.to_datetime(CAData.index)

    #temperature is x, combined electricity demand is y
    
    hours_in_2015 = pd.date_range('2015-01-01T00:00Z','2015-12-31T23:00Z', freq='H')

    ESTemp = ESData['temperature']['2015-01-01 00:00:00':'2015-12-31 23:00:00']
    DKTemp = DKData['temperature']['2015-01-01 00:00:00':'2015-12-31 23:00:00']
    COTemp = COData[ 'temperature']['2011-01-01 00:00:00':'2011-12-31 23:00:00']
    CATemp = CAData['temperature']['2011-01-01 00:00:00':'2011-12-31 23:00:00']

    ESTemp.index = hours_in_2015
    DKTemp.index = hours_in_2015

    df_elec["DNKTemp"] = DKData["temperature"]
    df_elec["ESPTemp"] = ESTemp["temperature"]
    df_cal_elec["CATemp"] = CATemp["temperature"]
    df_co_elec["COTemp"] = COTemp["temperature"]




    df_elec["DNKcombine"] = df_elec.apply(lambda row: row ["DNKcombine"] if row["DNKTemp"] > 15.8
        else row["DNKcombine"] - 273.665 * (15.8-row["DNKTemp"]) if row["DNKTemp"]+ degree_change - 15.8 > 0 
        else row["DNKcombine"] - 273.665 * degree_change, axis = 1)
    #Like California, there are only three cases. Unlike california, the three cases are a bit different
    #flat to flat, heat to flat, heat to heat

    df_elec["DNKTemp"] = df_elec.apply(lambda row: row["DNKTemp"] + degree_change, axis = 1)





    df_elec["ESPcombine"] = df_elec.apply(lambda row: row["ESPcombine"] + 865.2 * degree_change if row["ESPTemp"] > 22.267 #Add according to positive slope if starts in cooling region
        else row["ESPcombine"] + 865.2 * (row["ESPTemp"] - 22.267 + degree_change) if row["ESPTemp"]+ degree_change  >22.267 and row["ESPTemp"] > 16
        else row["ESPcombine"] if row["ESPTemp"] > 16 
        else row["ESPcombine"] - 1356.544 * (16- row["ESPTemp"]) + 865.2 * (row["ESPTemp"] + degree_change - 22.267) if row["ESPTemp"] + degree_change > 16 and row["ESPTemp"] + degree_change  > 22.267 
        else row["ESPcombine"] - 1356.544 * (16- row["ESPTemp"]) if row["ESPTemp"] + degree_change > 16
        else row["ESPcombine"] - 1356.544 * degree_change, axis = 1)

    df_elec["ESPcombine"] = df_elec.apply(lambda row: row["ESPcombine"] + (row ["ESPcombine"] - 30000) * (slope_factor-1) if row["ESPTemp"] > 22.267 and row["ESPcombine"] > 30000
    else row["ESPcombine"], axis = 1)

    df_elec["ESPTemp"] = df_elec.apply(lambda row: row["ESPTemp"] + degree_change, axis = 1)




    df_cal_elec["adjust_elec_demand"] = df_cal_elec.apply(lambda row: row["adjust_elec_demand"] + 1093.304 * degree_change if row["CATemp"] > 16.14 #Add according to positive slope if starts in cooling region
        else row["adjust_elec_demand"] + 1093.304 * (row["CATemp"] - 16.14 + degree_change) if row["CATemp"]+ degree_change  >16.14 and row["CATemp"] > 14.22
        else row["adjust_elec_demand"] if row["CATemp"] > 14.22
        else row["adjust_elec_demand"] - 640.248 * (14.22- row["CATemp"]) + 1093.304 * (row["CATemp"] + degree_change - 16.14) if row["CATemp"] + degree_change > 14.22 and row["CATemp"] + degree_change  > 16.14
        else row["adjust_elec_demand"] - 640.248 * (14.22- row["CATemp"]) if row["CATemp"] + degree_change > 14.22
        else row["adjust_elec_demand"] - 640.248 * degree_change, axis = 1)


    #temperature is x, combined electricity demand is y


    df_cal_elec["CATemp"] = df_cal_elec.apply(lambda row: row["CATemp"] + degree_change, axis = 1)
    
    #Use this line if you also want to include slope
    df_cal_elec["adjust_elec_demand"] = df_cal_elec.apply(lambda row: row["adjust_elec_demand"] + (row ["adjust_elec_demand"] - 35000) * (slope_factor-1) if row["CATemp"] > 16.14 and row["adjust_elec_demand"] > 35000
    else row["adjust_elec_demand"], axis = 1)





    df_co_elec["adjust_elec_demand"] = df_co_elec.apply(lambda row: row["adjust_elec_demand"] + 249.56 * degree_change if row["COTemp"] > 16.966 #Add according to positive slope if starts in cooling region
        else row["adjust_elec_demand"] + 249.56 * (row["COTemp"] - 16.966 + degree_change) if row["COTemp"]+ degree_change  >16.966 and row["COTemp"] > 13.801
        else row["adjust_elec_demand"] if row["COTemp"] > 13.801
        else row["adjust_elec_demand"] - 646.373 * (13.801- row["COTemp"]) + 249.56 * (row["COTemp"] + degree_change - 16.966) if row["COTemp"] + degree_change > 13.801 and row["COTemp"] + degree_change  > 16.966
        else row["adjust_elec_demand"] - 646.373 * (13.801- row["COTemp"]) if row["COTemp"] + degree_change > 13.801
        else row["adjust_elec_demand"] - 646.373 * degree_change, axis = 1)


    #temperature is x, combined electricity demand is y


    df_co_elec["COTemp"] = df_co_elec.apply(lambda row: row["COTemp"] + degree_change, axis = 1)
    
    #Use this line if you also want to include slope
    df_co_elec["adjust_elec_demand"] = df_co_elec.apply(lambda row: row["adjust_elec_demand"] + (row ["adjust_elec_demand"] - 9000) * (slope_factor-1) if row["COTemp"] > 16.966 and row["adjust_elec_demand"] > 9000
    else row["adjust_elec_demand"], axis = 1) 


    return df_elec, df_cal_elec, df_co_elec   



    
  
  
   




