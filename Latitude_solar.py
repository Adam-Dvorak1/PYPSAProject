import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time
from itertools import repeat
import os
import re
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import pathlib
import matplotlib
import csv
import urllib

'''We are using new costs, found in costs.csv. This is what pypsa-eur uses'''

def set_hours(mynetworks):
    hours_in_2015 = pd.date_range('2015-01-01T00:00Z','2015-12-31T23:00Z', freq='H') #for network, nspain
    hours_in_2011 = pd.date_range('2011-01-01T00:00:00','2011-12-31T23:00:00', freq='H') #for ncal, nco

    for n in mynetworks:
        if mynetworks.index(n)//2 == 0:
            n.set_snapshots(hours_in_2015)
        else:
            n.set_snapshots(hours_in_2011)

def add_bus(n):
    n.add("Bus", "electricity bus")  


def annuity(n,r):
    """Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20,0.05)*20 = 1.6"""

    if r > 0:
        return r/(1. - 1./(1.+r)**n)
    else:
        return 1/n



#Carriers, generators, wind and solar loading
def add_carriers(mynetworks):
    for n in mynetworks:
        n.add("Carrier", "gas", co2_emissions = 0.19)
        n.add("Carrier", "onshorewind")
        n.add("Carrier", "solar")






#utility scale solar




#For gas, we assume the same cost in all countries. we just need to add a generator to each network,
# network, nspain, etc




def add_generators(networks, CF_wind, cap_cost_wind, CF_solar, cap_cost_solar
                              , cap_cost_gas, marginal_cost_gas):
    for n in networks:
        n.add("Generator",
                   "onshorewind",
                   bus = "electricity bus",
                   p_nom_extendable = True,
                   carrier = "onshorewind",
                   capital_cost = cap_cost_wind,
                   marginal_cost = 0,
                 p_max_pu = CF_wind[networks.index(n)])
        
        n.add("Generator",
                   "solar",
                   bus = "electricity bus",
                   p_nom_extendable = True,
                   carrier = "solar",
                   capital_cost = cap_cost_solar,
                   marginal_cost = 0,
                p_max_pu = CF_solar[networks.index(n)])
        
        n.add("Generator",
                    "OCGT",
                    bus="electricity bus",
                    p_nom_extendable=True,
                    carrier="gas",
                    #p_nom_max=1000,
                    capital_cost = cap_cost_gas,
                    marginal_cost = marginal_cost_gas)

        co2_limit=1 #tonCO2 #global as in system global
        n.add("GlobalConstraint",
            "co2_limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=co2_limit)

    





#These are the same variables but two orders of magnitude cheaper
# capital_cost_electrolysis = annuity(25, 0.07) * 6500 * (1 + 0.02) 

# capital_cost_H2 = annuity(100, 0.07) * 30 #EUR/MWh

# capital_cost_fuelcell = annuity(10, 0.07) * 13000 * (1 + 0.05)

def add_storage(n, cost_electro, cost_H2, cost_fuelcell, cost_batt, cost_inverter):
    n.add("Carrier", "H2")
    n.add("Bus", "H2", carrier = "H2")
    
    n.add("Link",
         "H2 Electrolysis",
         bus1 = "H2",
         bus0 = "electricity bus",
         p_nom_extendable = True,
         carrier = "H2 Electrolysis",
         efficiency = 0.66,
         capital_cost = cost_electro * 0.66 #EUR/MW_el, making sure to multiply by efficiency
         )
    
    n.add("Link",
         "H2 Fuel Cell",
         bus0 = "H2",
         bus1 = "electricity bus",
         p_nom_extendable = True,
         carrier = "H2 Fuel Cell",
         efficiency = .5,
         capital_cost = cost_fuelcell * .5)#need to multiply by efficiency 
    
    n.add("Store",
         "H2 Store",
         bus = "H2",
         e_nom_extendable = True,
         e_cyclic = True,
         carrier = "H2 Store",
         capital_cost = cost_H2)
    
    
    
    
    n.add("Carrier", "battery")
    n.add("Bus", "battery", carrier = "battery")
    
    n.add("Link",
         "battery charger",
         bus0 = "electricity bus",
         bus1 = "battery",
         carrier = "battery charger",
         efficiency = 0.95**0.5,
         p_nom_extendable = True,
         capital_cost = cost_inverter)
    
    n.add("Link",
         "battery discharger",
         bus0 = "battery",
         bus1 = "electricity bus",
         carrier = "battery discharger",
         efficiency = 0.95**0.5,
         p_nom_extendable = True,
         )#No costs because they are all included in the charger? not sure
    
    n.add("Store",
         "battery",
         bus = "battery",
         e_cyclic = True, #NO FREE LUNCH must return back to original position by end of the year
         e_nom_extendable = True,
         capital_cost = cost_batt)

    #add CO2 constraint
    
    
    


#If I want to do other storage stuff I need to include it in reset_stats
def reset_stats(n):
    n.generators.loc[['solar'], ['capital_cost']] = annuity(25,0.04)*600000*(1+0.02)
    n.generators.loc[['onshorewind'], ['capital_cost']] = annuity(30,0.04)*1040000*(1+0.0245)
    n.generators.loc[['OCGT'], ['capital_cost']] = annuity(30,0.4)*400000*(1+0.0375)
    n.stores.loc[['battery'], ['capital_cost']] =  annuity(20, 0.07) * 232000


    
def make_dir(foldername):
    dir = pathlib.Path().resolve() #This is the current path
    mypath = str(dir) + "/NetCDF" #Going into the NetCDF folder

    for item in os.listdir(mypath):
        if os.path.isdir(os.path.join(mypath, item)): #searching for directories in my path
            newpath = os.path.join(mypath, item) #Make the path to be our previous path + directory 
            os.mkdir(newpath + f"/{foldername}") #Make new folder in the directory




def find_solar_data(n, name, solar_cost, dirname):
    #Takes annualized coefficient and multiplies by investment cost
    
    annualized_solar_cost =  0.06529220204218368* solar_cost
    n.generators.loc[['solar'],['capital_cost']] = annualized_solar_cost
    
    #this substitutes the current solar cost in our generator for a new cost

    
    n.lopf(n.snapshots, 
             pyomo=False,
             solver_name='gurobi')
    n.export_to_netcdf("NetCDF/"+ name + f"/{dirname}/{solar_cost}solar_cost.nc")


def import_cdf_data(filepath):
    '''This is critical to getting the data that we previously stored'''

    n = pypsa.Network()
    n.import_from_netcdf(filepath)

    solar_penetration = n.generators_t.p['solar'].sum()/sum(n.generators_t.p.sum())
    wind_penetration = n.generators_t.p['onshorewind'].sum()/sum(n.generators_t.p.sum())
    gas_penetration = n.generators_t.p['OCGT'].sum()/sum(n.generators_t.p.sum())


   
    solar_cost = n.generators.loc[['solar'],['capital_cost']].values[0]
    wind_cost = n.generators.loc[['onshorewind'],['capital_cost']].values[0]
    batt_cost = n.stores.loc[['battery'], ['capital_cost']].values[0]

    name = n.name


    return name, solar_cost, solar_penetration, wind_penetration, gas_penetration, wind_cost, batt_cost

#function that gets the same name file from each 

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def netcdf_to_csv(dataset):
    #I am modifying this to get all into one csv. It might be best practice
    header_list = ["country", "solar_cost", "solar_penetration", 'wind_penetration', 'gas_penetration', 'wind_cost', 'batt_cost']
    with open (f"results/csvs/{dataset}.csv", "w", newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(header_list)
    
    mynames = ["Denmark", "Spain", "CA", "CO"]

    for name in mynames:
        mypath = "NetCDF/" + name + f"/{dataset}"
        for filename in natural_sort(os.listdir(mypath)):
            f = os.path.join(mypath, filename)
            solution_list = import_cdf_data(f)
            with open(f"results/csvs/{dataset}.csv", mode = "a") as afile:
                writer = csv.writer(afile)
                writer.writerow(solution_list)


def latitude_plot():
    df = pd.read_csv("results/csvs/Apr7_latitude_run3.csv")
    countrydict = {"Denmark": "DK", "Spain": "ES", "CO": "CO", "CA": "CA"}
    df['Country Code'] = df["country"].map(countrydict)
    df = df.drop(["country"], axis = 1)

    states = pd.read_csv("data/states_lat.csv")
    countries = pd.read_csv("data/countries_lat.csv")

    
    newdf = pd.merge(df, countries[['country', 'latitude']], left_on = "Country Code", right_on = "country")
    #newdf['latitude'] = newdf.apply(lambda row: np.nan if row["Country Code"] =='CA' or row['Country Code'] == "CO" else row["latitude"], axis = 1)
    
    #newdf.loc[newdf['latitude'].isna(), 'latitude'] =  pd.merge(newdf, states[['state','latitude']], left_on = "Country Code", right_on = "state")
    statelist = ["CA", "CO"]
    for state in statelist:

        newdf.loc[newdf['Country Code'] == state, 'latitude'] = states.loc[states["state"] == state, 'latitude'].values
    
    fig, ax = plt.subplots()
    ax.scatter(newdf['latitude'], newdf['solar_penetration'])
    ax.set_xlabel("latitude (degrees)")
    ax.set_ylabel("solar penetration (%)")
    ax.set_title("solar penetration by latitude for four regions")

    for idx, row in newdf.iterrows():
        ax.annotate(row['Country Code'], (row['latitude']* 1.01, row['solar_penetration']))

    plt.savefig("Images/Latitude/local_four")
    
    plt.show()



if __name__ == "__main__":

    Denmark = pypsa.Network()
    Spain = pypsa.Network()
    CA = pypsa.Network()
    CO = pypsa.Network()    
        
    mynetworks = [Denmark, Spain, CA, CO]

    set_hours(mynetworks)

    for network in mynetworks:
        add_bus(network)
        

    df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0)# in MWh
    df_elec.index = pd.to_datetime(df_elec.index) #change index to datatime

    df_cal_elec = pd.read_csv('data_extra/CaliforniaTimeSeries.csv', index_col=0)
    df_cal_elec.index = pd.to_datetime(df_cal_elec.index)

    df_co_elec = pd.read_csv('data_extra/ColoradoTimeSeries.csv', index_col=0)
    df_co_elec.index = pd.to_datetime(df_co_elec.index)


    Denmark.add("Load",
                "load", 
                bus="electricity bus", 
                p_set=df_elec['DNK'])

    Spain.add("Load",
                "load", 
                bus="electricity bus", 
                p_set=df_elec['ESP'])

    CA.add("Load",
                "load", 
                bus="electricity bus", 
                p_set=df_cal_elec['demand_mwh'])

    CO.add("Load",
                "load", 
                bus="electricity bus", 
                p_set=df_co_elec['demand_mwh'])


    add_carriers(mynetworks)
    # 1A add onshore wind data for DNK and ESP

    df_onshorewind = pd.read_csv('data_extra/onshore_wind_1979-2017.csv', sep=';', index_col=0)
    df_onshorewind.index = pd.to_datetime(df_onshorewind.index)

    df_cal_onshorewind = pd.read_csv('data_extra/CaliforniaTimeSeries.csv', index_col=0)
    df_cal_onshorewind.index = pd.to_datetime(df_cal_onshorewind.index)

    df_co_onshorewind = pd.read_csv('data_extra/ColoradoTimeSeries.csv', index_col=0)
    df_co_onshorewind.index = pd.to_datetime(df_co_onshorewind.index)


    # 1B Capacity factors (wind)


    CFw_Denmark = df_onshorewind['DNK'][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in Denmark.snapshots]]
    CFw_Spain = df_onshorewind['ESP'][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in Spain.snapshots]]
    CFw_CA = df_cal_onshorewind['onwind'][[hour.strftime("%Y-%m-%dT%H:%M:%S") for hour in CA.snapshots]]
    CFw_CO = df_co_onshorewind['onwind'][[hour.strftime("%Y-%m-%dT%H:%M:%S") for hour in CO.snapshots]]

    CF_wind = [CFw_Denmark, CFw_Spain, CFw_CA, CFw_CO]





    # 2A Add solar data for DNK and ESP
    df_solar = pd.read_csv('data_extra/pv_optimal.csv', sep=';', index_col=0)
    df_solar.index = pd.to_datetime(df_solar.index)

    df_cal_solar = pd.read_csv('data_extra/CaliforniaTimeSeries.csv', index_col=0)
    df_cal_solar.index = pd.to_datetime(df_cal_solar.index)

    df_co_solar = pd.read_csv('data_extra/ColoradoTimeSeries.csv', index_col=0)
    df_co_solar.index = pd.to_datetime(df_co_solar.index)


    # 2BCapacity factors (solar) 
    CFs_Denmark = df_solar['DNK'][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in Denmark.snapshots]]
    CFs_Spain = df_solar['ESP'][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in Spain.snapshots]]
    CFs_CA = df_cal_solar['solar'][[hour.strftime("%Y-%m-%dT%H:%M:%S") for hour in CA.snapshots]]
    CFs_CO = df_co_solar['solar'][[hour.strftime("%Y-%m-%dT%H:%M:%S") for hour in CO.snapshots]]

    CF_solar = [CFs_Denmark, CFs_Spain, CFs_CA, CFs_CO]


    costs = pd.read_csv("data/costs.csv")

    capital_cost_onshorewind = annuity(30,0.04)*1040000*(1+0.0245) #EUR/MW
    capital_cost_solar = annuity(25,0.04)*600000*(1+0.02)
    capital_cost_OCGT = annuity(30,0.04)*400000*(1+0.0375) # in €/MW
    fuel_cost = 21.6 # in €/MWh_th
    efficiency = 0.39
    marginal_cost_OCGT = fuel_cost/efficiency # in €/MWh_el
    add_generators(mynetworks, CF_wind, capital_cost_onshorewind, 
               CF_solar, capital_cost_solar, 
               capital_cost_OCGT, marginal_cost_OCGT)


    capital_cost_electrolysis = annuity(25, 0.07) * 350000 * (1 + 0.02) 

    capital_cost_H2 = annuity(100, 0.07) * 3000 #EUR/MWh

    capital_cost_fuelcell = annuity(20, 0.04) * 339000 * (1 + 0.03)

    capital_cost_inverter = annuity(10, 0.07) * 270000 * (1 + 0.002)

    capital_cost_battery = annuity(20, 0.07) * 411000 * (1 + 0.04)
    
    for network in mynetworks:
        add_storage(network, capital_cost_electrolysis, capital_cost_H2, 
                    capital_cost_fuelcell, capital_cost_battery, capital_cost_inverter)

    networkdict = {Denmark: "Denmark", Spain:"Spain", CA:"CA", CO:"CO"}

    #make_dir("Apr7_latitude_run3")

    for network in networkdict.keys():
        network.name = networkdict[network]
        find_solar_data(network, networkdict[network], 600000, "Apr7_latitude_run3")

    netcdf_to_csv("Apr7_latitude_run3")


    
    

    
    #function that
        #puts relevant information into csv
            #solar and wind production
            #solar cost
            #
    #Function that
        #extracts the desired nc from each 