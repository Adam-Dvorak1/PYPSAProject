import pandas as pd
from helpers import annual_cost




def find_solar_data(n, name, solar_cost, dirname):
    #Takes annualized coefficient and multiplies by investment cost
  
    annualized_solar_cost =  0.07846970300338728* solar_cost
    n.generators.loc[['solar'],['capital_cost']] = annualized_solar_cost
    
    #this substitutes the current solar cost in our generator for a new cost

    
    n.lopf(n.snapshots, 
             pyomo=False,
             solver_name='gurobi')


    n.export_to_netcdf("NetCDF/"+ name + f"/{dirname}/{solar_cost}solar_cost.nc")



def find_wind_data(n, name, wind_cost, dirname):
    #Takes annualized coefficient and multiplies by investment cost
  
    annualized_wind_cost = 0.08442684282600257 * wind_cost
    n.generators.loc[['onshorewind'],['capital_cost']] = annualized_wind_cost
    
    #this substitutes the current solar cost in our generator for a new cost

    
    n.lopf(n.snapshots, 
             pyomo=False,
             solver_name='gurobi')

    n.export_to_netcdf("NetCDF/"+ name + f"/{dirname}/{wind_cost}wind_cost.nc")


def find_batt_data(n, name, batt_cost, dirname):

    annualized_batt_cost = 0.09439292574325567 * batt_cost
    n.stores.loc[['battery'],['capital_cost']] = annualized_batt_cost
    
    #this substitutes the current solar cost in our generator for a new cost

    
    n.lopf(n.snapshots, 
             pyomo=False,
             solver_name='gurobi')

    n.export_to_netcdf("NetCDF/"+ name + f"/{dirname}/{batt_cost}batt_cost.nc")



if __name__ == "__main__":
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

