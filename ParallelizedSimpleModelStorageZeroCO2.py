import pypsa
import pandas as pd
import numpy as np
import matplotlib.ticker as mtick
import time
from itertools import repeat
import os
import re
import matplotlib.image as mpimg

import pathlib
import matplotlib
import csv
import glob
from multiprocessing import Pool
from buildnetwork import set_hours, add_buses, add_load, add_generators, add_storage
from helpers import annuity, make_a_list
from modifynetwork import find_solar_data, find_batt_data, find_wind_data

'''Author: Adam Dvorak. Inspired by code from the RES project by Marta. 
Thanks to Marta Victoria, Ebbe Gøtske, and Tim Pedersen for help'''
#network = Denmark, nspain, ncal, ncolorado
__spec__ = "ModuleSpec(name='builtins', loader=<class'_frozen_importlib.BuiltinImporter'>)"  ##This has been necessary for some of my multiprocessing statements








#Carriers, generators, wind and solar loading





def reset_stats(n):
    n.generators.loc[['solar'], ['capital_cost']] = annuity(35,0.07)*529000*(1+0.016)
    n.generators.loc[['onshorewind'], ['capital_cost']] = annuity(27,0.07)*1118000*(1+0.012)
    n.generators.loc[['OCGT'], ['capital_cost']] = annuity(25,0.07)*453000*(1+0.018)
    n.stores.loc[['battery'], ['capital_cost']] =  annuity(20, 0.07) * 232000
    return n

def make_dir(foldername):
    dir = pathlib.Path().resolve() #This is the current path
    mypath = str(dir) + "/NetCDF" #Going into the NetCDF folder

    for item in os.listdir(mypath):
        if os.path.isdir(os.path.join(mypath, item)): #searching for directories in my path
            newpath = os.path.join(mypath, item) #Make the path to be our previous path + directory 
            os.mkdir(newpath + f"/{foldername}") #Make new folder in the directory


def import_cdf_data(filepath):
    '''This is critical to getting the data that we previously stored'''

    n = pypsa.Network()
    n.import_from_netcdf(filepath)

    solar_penetration = n.generators_t.p['solar'].sum()/sum(n.generators_t.p.sum())
    wind_penetration = n.generators_t.p['onshorewind'].sum()/sum(n.generators_t.p.sum())
    gas_penetration = n.generators_t.p['OCGT'].sum()/sum(n.generators_t.p.sum())

    s_curtailment = (n.generators_t.p-n.generators.p_nom_opt * n.generators_t.p_max_pu)['solar'].sum()
    w_curtailment = (n.generators_t.p-n.generators.p_nom_opt * n.generators_t.p_max_pu)['onshorewind'].sum()

    max_gen_solar = (n.generators.p_nom_opt * n.generators_t.p_max_pu)['solar'].sum()
    max_gen_wind = (n.generators.p_nom_opt * n.generators_t.p_max_pu)['onshorewind'].sum()

    battery_pen = n.links_t.p1["battery discharger"].sum()/sum(n.generators_t.p.sum())
   
    solar_cost = n.generators.loc[['solar'],['capital_cost']].values[0]
    wind_cost = n.generators.loc[['onshorewind'],['capital_cost']].values[0]
    batt_cost = n.stores.loc[['battery'], ['capital_cost']].values[0]
    if max_gen_solar == 0:
        s_curtailment = 0
    else:
        s_curtailment = s_curtailment/max_gen_solar

    if max_gen_wind == 0:
        w_curtailment = 0
    else:
        w_curtailment = w_curtailment/max_gen_wind
 
    

    return solar_cost, solar_penetration, wind_penetration, s_curtailment, w_curtailment, gas_penetration, wind_cost, batt_cost, battery_pen


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def iterate_netcdf(country, dataset):
    '''Takes the folders of the country and the particular 
    set of gurobi solutions as strings'''
    solution_list = []
    mypath = "NetCDF/" + country + f"/{dataset}"
    for filename in natural_sort(os.listdir(mypath)):
        f = os.path.join(mypath, filename)
        solution_list += [import_cdf_data(f)]
    
    return solution_list

def netcdf_to_csv(country, dataset):
    header_list = ["solar_cost", "solar_penetration", 'wind_penetration', 's_curtailment', 'w_curtailment', 'gas_penetration', 'wind_cost', 'batt_cost', 'battery_pen']
    with open (f"results/csvs/{country}/{dataset}.csv", "w", newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(header_list)
    mypath = "NetCDF/" + country + f"/{dataset}"
    for filename in natural_sort(os.listdir(mypath)):
        f = os.path.join(mypath, filename)
        solution_list = import_cdf_data(f)
        with open(f"results/csvs/{country}/{dataset}.csv", mode = "a") as afile:
            writer = csv.writer(afile)
            writer.writerow(solution_list)
        

def netcdf_csv_all(datasets):
    countries = ["Denmark", "Spain", "CA", "CO"]
    # datasets = ["battcostLOGFeb24_2deg_2sl", "windcostLOGFeb24_2deg_2sl", "solarcostLOGFeb24_2deg_2sl", "battcostLOGFeb8", "windcostLOGFeb7", "solarcostLOGFeb7"]

    for country, data in [(country, data) for country in countries for data in datasets]:
        netcdf_to_csv(country, data)






####NEW SECTION 24/2###
#This section will take the given relationship of temperature and electricity demand for each country and extend it, creating


###-----------------<<<Global warming and increased cooling demand>>>-------------------------


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
    df_elec["ESPTemp"] = ESData["temperature"]
    df_cal_elec["CATemp"] = CAData["temperature"]
    df_co_elec["COTemp"] = COData["temperature"]




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


    return df_elec, df_co_elec, df_cal_elec








def mod_csvs():
    '''The purpose of this function is to turn the columns of the csvs into only having float values.
    Before we had a problem where we would also get values that are not floats, in square brackets'''
    for file in glob.glob('results/csvs/**/*'):
        path = file
        df = pd.read_csv(path)
        for column in df:

            if type(df[column][0]) == str:
                df[column] = df[column].str.replace(r'[][]', '', regex=True)
                df[column] = df[column].astype(float)

        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)

        df.to_csv(path)
            
#mod_csvs()





    
if __name__ == "__main__":

    '''We are interested in making '''
    runname = ""

    Denmark = pypsa.Network(name = "Denmark")
    Spain = pypsa.Network(name = "Spain")
    California = pypsa.Network(name = "California")
    Colorado = pypsa.Network(name = "Colorado")    
        
    mynetworks = [Denmark, Spain, California, Colorado]

    

    for network in mynetworks:
        network = set_hours(network)
        network = add_buses(network)
        network = add_load(network)
        network = add_generators(network)
        network = add_storage(network)

    
    make_dir("solarcost_elec_27_Sept")

    f = make_a_list(California, "CA", np.logspace(4, 5, 4), "This_is_a_test")
    g = make_a_list(Denmark, "Denmark", np.logspace(4, 5, 4), "This_is_a_test")

    with Pool(processes=4) as pool:
        pool.starmap(find_solar_data, f)
        pool.starmap(find_solar_data, g)  

    ##--------------------<<<Model Runs: Just electricity>>>------------------------

    #These four below return 100 points of cost vs solar penetration. 
    # for network in mynetworks:
    #     reset_stats(network)

    #make_dir("This_is_a_test")
    #find_solar_data(Denmark, "Denmark", 100000, "This_is_a_test")






    # make_dir("solarcost_elec_27_Sept")

    #DNK_solar_data = list(map(find_solar_data, repeat(Denmark), repeat("Denmark"), np.logspace(4, 6.31, 100), repeat("solarcost_elec_27_Sept")))
    # ESP_solar_data = list(map(find_solar_data, repeat(Spain), repeat("Spain"), np.logspace(4, 6.31, 100), repeat("solarcost_elec_27_Sept")))
    # CA_solar_data = list(map(find_solar_data, repeat(CA), repeat("CA"), np.logspace(4, 6.31, 100), repeat("solarcost_elec_27_Sept")))
    # CO_solar_data = list(map(find_solar_data, repeat(CO), repeat("CO"), np.logspace(4, 6.31, 100), repeat("solarcost_elec_27_Sept")))

    # #find_wind_data()
    # for network in mynetworks:
    #     reset_stats(network)



    ###Jan 31 currently running wind and battery data
    # list(map(find_wind_data, repeat(Denmark), repeat("Denmark"), np.logspace(5, 6.5, 100), repeat("dirname here")))
    # list(map(find_wind_data, repeat(Spain), repeat("Spain"), np.logspace(5, 6.5, 100), repeat("dirname here")))
    # list(map(find_wind_data, repeat(CA), repeat("CA"), np.logspace(5, 6.5, 100), repeat("dirname here")))
    # list(map(find_wind_data, repeat(CO), repeat("CO"), np.logspace(5, 6.5, 100), repeat("dirname here")))

    # for network in mynetworks:
    #     reset_stats(network)
    # list(map(find_batt_data, repeat(Denmark), repeat("Denmark"), np.logspace(4.5, 6, 100), repeat("dirname here")))
    # list(map(find_batt_data, repeat(Spain), repeat("Spain"), np.logspace(4.5, 6, 100), repeat("dirname here")))
    # list(map(find_batt_data, repeat(CA), repeat("CA"), np.logspace(4.5, 6, 100), repeat("dirname here")))
    # list(map(find_batt_data, repeat(CO), repeat("CO"), np.logspace(4.5, 6, 100), repeat("dirname here")))






    ##------------------------<<<Mod electricity loads with electrification of heating>>>---------------

    Denmark.remove("Load", "load")
    Denmark.add("Load",
                "load", 
                bus="electricity bus", 
                p_set=df_elec['DNKcombine'])
    Spain.remove("Load", "load")
    Spain.add("Load",
                "load", 
                bus="electricity bus", 
                p_set=df_elec['ESPcombine'])
    CA.remove("Load", "load")
    CA.add("Load",
                "load", 
                bus="electricity bus", 
                p_set=df_cal_elec["adjust_elec_demand"])
    CO.remove("Load", "load")
    CO.add("Load",
                "load", 
                bus="electricity bus", 
                p_set=df_co_elec["adjust_elec_demand"])


    