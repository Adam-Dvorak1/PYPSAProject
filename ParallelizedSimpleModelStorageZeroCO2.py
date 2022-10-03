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
import glob
from multiprocessing import Pool


'''Author: Adam Dvorak. Inspired by code from the RES project by Marta. 
Thanks to Marta Victoria, Ebbe Gøtske, and Tim Pedersen for help'''
#network = Denmark, nspain, ncal, ncolorado
__spec__ = "ModuleSpec(name='builtins', loader=<class'_frozen_importlib.BuiltinImporter'>)"  ##This has been necessary for some of my multiprocessing statements




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
                 p_max_pu = CF_wind[networks.index(n)]) #This is the timeseries.
        
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


def reset_stats(n):
    n.generators.loc[['solar'], ['capital_cost']] = annuity(35,0.07)*529000*(1+0.016)
    n.generators.loc[['onshorewind'], ['capital_cost']] = annuity(27,0.07)*1118000*(1+0.012)
    n.generators.loc[['OCGT'], ['capital_cost']] = annuity(25,0.07)*453000*(1+0.018)
    n.stores.loc[['battery'], ['capital_cost']] =  annuity(20, 0.07) * 232000

def find_solar_data(*args):
    print(args)
    #Takes annualized coefficient and multiplies by investment cost
    n = args[0]
    name = args[1]
    dirname = args[2]
    solar_cost = args[3]
  
    annualized_solar_cost =  0.07846970300338728* solar_cost
    n.generators.loc[['solar'],['capital_cost']] = annualized_solar_cost
    
    #this substitutes the current solar cost in our generator for a new cost

    
    n.lopf(n.snapshots, 
             pyomo=False,
             solver_name='gurobi')


    n.export_to_netcdf("NetCDF/"+ name + f"/{dirname}/{solar_cost}solar_cost.nc")
    
    
def make_dir(foldername):
    dir = pathlib.Path().resolve() #This is the current path
    mypath = str(dir) + "/NetCDF" #Going into the NetCDF folder

    for item in os.listdir(mypath):
        if os.path.isdir(os.path.join(mypath, item)): #searching for directories in my path
            newpath = os.path.join(mypath, item) #Make the path to be our previous path + directory 
            os.mkdir(newpath + f"/{foldername}") #Make new folder in the directory


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


def find_C02lim_data(n, name, co2lim):
    #Takes annualized coefficient and multiplies by investment cost
    n.global_constraints.loc[['co2_limit'],['constant']] = co2lim
    
    
    #this substitutes the current solar cost in our generator for a new cost

    
    n.lopf(n.snapshots, 
             pyomo=False,
             solver_name='gurobi')

    n.export_to_netcdf("NetCDF/"+ name + f"/constraintLIN/co2constraint{co2lim}.nc")


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



def iterate_netcdf_co2(country):
    '''Takes country as a string'''
    solution_list = []
    mypath = "NetCDF/" + country + "/constraintLIN"
    for filename in natural_sort(os.listdir(mypath)):
        #if "solar" in filename:
        #if "wind" in filename:
        if "constraint" in filename:
            f = os.path.join(mypath, filename)
            solution_list += [import_cdf_data(f)]
    
    return solution_list






def make_new_run(runname, runtype):
    '''The purpose of this function is to only have one runname. Then, it runs the program and performs the following functions
    
    The runtype can be either "solar", "wind", or "batt"

    This code is not completed and has not been implemented as of yet
    '''
    if runtype == "solar":
        list(map(find_solar_data, repeat(Denmark), repeat("Denmark"), np.logspace(4, 6.31, 100), repeat(filename)))
        list(map(find_solar_data, repeat(Spain), repeat("Spain"), np.logspace(4, 6.31, 100), repeat("solarcost_elec_27_Sept")))
        list(map(find_solar_data, repeat(CA), repeat("CA"), np.logspace(4, 6.31, 100), repeat("solarcost_elec_27_Sept")))
        list(map(find_solar_data, repeat(CO), repeat("CO"), np.logspace(4, 6.31, 100), repeat("solarcost_elec_27_Sept")))


def flex_plus_curtailALL():
    '''This is one of the functions we use to make an abstract figure'''
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    inner_dnk = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], wspace=0.1, hspace=0, height_ratios = [1, 2])
    inner_esp = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], wspace=0.1, hspace=0, height_ratios = [1, 2])
    inner_col = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2], wspace=0.1, hspace=0, height_ratios = [1, 2])
    inner_cal = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[3], wspace=0.1, hspace=0, height_ratios = [1, 2])

    axden0 = plt.Subplot(fig, inner_dnk[0])
    axden1 = plt.Subplot(fig, inner_dnk[1])
    fig.add_subplot(axden0)
    fig.add_subplot(axden1)

    axesp0 = plt.Subplot(fig, inner_esp[0])
    axesp1 = plt.Subplot(fig, inner_esp[1])
    fig.add_subplot(axesp0)
    fig.add_subplot(axesp1)

    axcol0 = plt.Subplot(fig, inner_col[0])
    axcol1 = plt.Subplot(fig, inner_col[1])
    fig.add_subplot(axcol0)
    fig.add_subplot(axcol1)

    axcal0 = plt.Subplot(fig, inner_cal[0])
    axcal1 = plt.Subplot(fig, inner_cal[1])
    fig.add_subplot(axcal0)
    fig.add_subplot(axcal1)       

    ####DENMARK###
    DNK_sp = [x[1] for x in co2dnk]
    DNK_sc = list(map(abs,[x[3] for x in co2dnk]))
    DNK_wp = [x[2] for x in co2dnk]
    DNK_gas = [x[5] for x in co2dnk]

    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axden0.scatter(DNK_gas, DNK_sc, color = "C1") #Scatter or plot?
    axden0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1, decimals = 0))
    axden0.set_ylabel("Curtailment")
    axden0.set_facecolor("#eeeeee")
    axden0.set_ylim(0, 0.32)


    axden1.stackplot(DNK_gas, DNK_sp, DNK_wp, DNK_gas, colors = ["#f1c232","#2986cc", "#cbbcf4"], labels = ["Solar", "Wind", "Gas"])
    axden1.set_ylim(0, 1)

    axden1.set_ylabel("Penetration")
    #axden1.set_xlabel("Percent flexible source")

    axden1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    axden0.spines["top"].set_visible(False)
    axden1.spines["right"].set_visible(False)
    axden0.set_title("Denmark")

    xticks = axden1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)


    ####SPAIN####

    ESP_sp = [x[1] for x in co2esp]
    ESP_sc = list(map(abs,[x[3] for x in co2esp]))
    ESP_wp = [x[2] for x in co2esp]
    ESP_gas = [x[5] for x in co2esp]

    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axesp0.scatter(ESP_gas, ESP_sc, color = "C1") #Scatter or plot?
    axesp0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1, decimals = 0))
    #axesp0.set_ylabel("Curtailment")
    axesp0.set_facecolor("#eeeeee")
    axesp0.set_title("Spain")
    axesp0.set_ylim(0, 0.32)


    axesp1.stackplot(ESP_gas, ESP_sp, ESP_wp, ESP_gas, colors = ["#f1c232","#2986cc", "#cbbcf4"])
    axesp1.set_ylim(0, 1)

    #axesp1.set_ylabel("Penetration")
    #axesp1.set_xlabel("Percent flexible source")

    axesp1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axesp0.spines["top"].set_visible(False)
    axesp1.spines["right"].set_visible(False)

    xticks = axesp1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)
    axesp0.yaxis.set_ticklabels([])
    axesp1.yaxis.set_ticklabels([])
    ####Colorado#####

    COL_sp = [x[1] for x in co2col]
    COL_sc = list(map(abs,[x[3] for x in co2col]))
    COL_wp = [x[2] for x in co2col]
    COL_gas = [x[5] for x in co2col]

    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axcol0.scatter(COL_gas, COL_sc, color = "C1") #Scatter or plot?
    axcol0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1, decimals = 0))
    axcol0.set_ylabel("Curtailment")
    axcol0.set_facecolor("#eeeeee")
    axcol0.set_title("Colorado")
    axcol0.set_ylim(0, 0.32)


    axcol1.stackplot(COL_gas, COL_sp, COL_wp, COL_gas, colors = ["#f1c232","#2986cc", "#cbbcf4"])
    axcol1.set_ylim(0, 1)

    axcol1.set_ylabel("Penetration")
    axcol1.set_xlabel("Fraction flexible source")

    axcol1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axcol0.spines["top"].set_visible(False)
    axcol1.spines["right"].set_visible(False)

    xticks = axcol1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)


    ###California####

    CAL_sp = [x[1] for x in co2cal]
    CAL_sc = list(map(abs,[x[3] for x in co2cal]))
    CAL_wp = [x[2] for x in co2cal]
    CAL_gas = [x[5] for x in co2cal]

    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axcal0.scatter(CAL_gas, CAL_sc, color = "C1") #Scatter or plot?
    axcal0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1, decimals = 0))
    #axcal0.set_ylabel("Curtailment")
    axcal0.set_facecolor("#eeeeee")
    axcal0.set_title("California")
    axcal0.set_ylim(0, 0.32)


    axcal1.stackplot(CAL_gas, CAL_sp, CAL_wp, CAL_gas, colors = ["#f1c232","#2986cc", "#cbbcf4"])
    axcal1.set_ylim(0, 1)

    #axcal1.set_ylabel("Penetration")
    axcal1.set_xlabel("Fraction flexible source")

    axcal1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axcal0.spines["top"].set_visible(False)
    axcal1.spines["right"].set_visible(False)
    
    xticks = axcal1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)

    axcal0.yaxis.set_ticklabels([])
    axcal1.yaxis.set_ticklabels([])


    for ax in plt.gcf().get_axes():
        ax.minorticks_on()
      
        ax.label_outer()

        ax.margins(x = 0)


    lines1, labels1 = axden1.get_legend_handles_labels()

    fig.legend(lines1, labels1, bbox_to_anchor=(0.7, 0.055), ncol=3)
    plt.savefig("Images/Figure3_flexible_Var3.png")
    plt.show()


def pen_plus_curtailALL():
    '''This makes a 2x2 grid of two axes each showing resource penetration and solar curtailment vs.
    a scaling log of solar. It is very long. It uses gridspec to order the axes, and other than that
    it is about the same as the other pen_plus_curtail() functions. 17/1'''
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    inner_dnk = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], wspace=0.1, hspace=0, height_ratios = [1, 2])
    inner_esp = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], wspace=0.1, hspace=0, height_ratios = [1, 2])
    inner_col = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2], wspace=0.1, hspace=0, height_ratios = [1, 2])
    inner_cal = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[3], wspace=0.1, hspace=0, height_ratios = [1, 2])

    axden0 = plt.Subplot(fig, inner_dnk[0])
    axden1 = plt.Subplot(fig, inner_dnk[1])
    fig.add_subplot(axden0)
    fig.add_subplot(axden1)

    axesp0 = plt.Subplot(fig, inner_esp[0])
    axesp1 = plt.Subplot(fig, inner_esp[1])
    fig.add_subplot(axesp0)
    fig.add_subplot(axesp1)

    axcol0 = plt.Subplot(fig, inner_col[0])
    axcol1 = plt.Subplot(fig, inner_col[1])
    fig.add_subplot(axcol0)
    fig.add_subplot(axcol1)

    axcal0 = plt.Subplot(fig, inner_cal[0])
    axcal1 = plt.Subplot(fig, inner_cal[1])
    fig.add_subplot(axcal0)
    fig.add_subplot(axcal1)       

    ####DENMARK###
    s_cost = [x[0] for x in solardnk]
    s_cost = [item for sublist in s_cost for item in sublist]
    s_cost = [x / 10**6 /0.07846970300338728 for x in s_cost] 
    DNK_sp = [x[1] for x in solardnk]
    DNK_wp = [x[2] for x in solardnk]
    DNK_sc = list(map(abs,[x[3] for x in solardnk]))



    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axden0.scatter(s_cost, DNK_sc, s = 15, color = "C1") #Scatter or plot?
    axden0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axden0.set_ylabel("Curtailment")
    axden0.set_facecolor("#eeeeee")



    axden1.stackplot(s_cost, DNK_sp, DNK_wp, colors = ["#f1c232","#2986cc"], labels = ["Solar", "Wind"])
    axden1.set_ylim(0, 1)
    axden0.set_ylim(0, 1)

    axden1.set_ylabel("Penetration")
    #axden1.set_xlabel("Percent flexible source")

    axden1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axden0.set_title("Denmark")

    xticks = axden1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)



    axden1.axvline(0.529, color='black',ls='--')
    axden1.axvline(1.3, color='black',ls='--')
    #We want to make a range for today's prices. the upper range is 
    axden1.text(0.85,0.05,  "Today's range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    axden1.axvline(0.019, color='black',ls='--')
    axden1.text(0.025,0.05, "2050--Optimistic",  fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    axden1.axvline(0.095, color='black',ls='--')
    axden1.text(0.13,0.05,  "2050--Less Optimistic", fontsize = 12, horizontalalignment = "center", rotation = "vertical")

    axden0.spines["top"].set_visible(False)
    axden0.spines["right"].set_visible(False)
    plt.rcParams['hatch.linewidth'] = 1
    axden1.fill_between([0.529, 1.3], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
    


    ####SPAIN####

    ESP_sp = [x[1] for x in solaresp]
    ESP_wp = [x[2] for x in solaresp]
    ESP_sc = list(map(abs,[x[3] for x in solaresp]))



    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axesp0.scatter(s_cost, ESP_sc, s = 15, color = "C1") #Scatter or plot?
    axesp0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    #axesp0.set_ylabel("Curtailment")
    axesp0.set_facecolor("#eeeeee")
    axesp0.set_title("Spain")
 


    axesp1.stackplot(s_cost, ESP_sp, ESP_wp, colors = ["#f1c232","#2986cc"])
    axesp1.set_ylim(0, 1)
    axesp0.set_ylim(0, 1)


    #axesp1.set_ylabel("Penetration")
    #axesp1.set_xlabel("Percent flexible source")

    axesp1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))



    xticks = axesp1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)


    axesp1.axvline(0.529, color='black',ls='--')
    
    axesp1.axvline(0.019, color='black',ls='--')
    axesp1.text(0.025,0.05, "2050--Optimistic", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    axesp1.axvline(0.095, color='black',ls='--')
    axesp1.text(0.13,0.05, "2050--Less Optimistic", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    axesp0.spines["top"].set_visible(False)
    axesp0.spines["right"].set_visible(False)
    axesp0.yaxis.set_ticklabels([])
    axesp1.yaxis.set_ticklabels([])
    axesp1.fill_between([0.529, 1.3], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
    axesp1.axvline(1.3, color='black',ls='--')
    axesp1.text(0.85,0.05,  "Today's range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")

    ####Colorado#####

    COL_sp = [x[1] for x in solarcol]
    COL_wp = [x[2] for x in solarcol]
    COL_sc = list(map(abs,[x[3] for x in solarcol]))


    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axcol0.scatter(s_cost, COL_sc, s = 15, color = "C1") #Scatter or plot?
    axcol0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axcol0.set_ylabel("Curtailment")
    axcol0.set_facecolor("#eeeeee")
    axcol0.set_title("Colorado")



    axcol1.stackplot(s_cost, COL_sp, COL_wp, colors = ["#f1c232","#2986cc"])
    axcol1.set_ylim(0, 1)
    axcol0.set_ylim(0, 1)

    axcol1.set_ylabel("Penetration")
    axcol1.set_xlabel("Cost of Solar")

    axcol1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))


    xticks = axcol1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)

    axcol1.axvline(0.529, color='black',ls='--')
    
    axcol1.axvline(0.019, color='black',ls='--')
    axcol1.text(0.025,0.05, "2050--Optimistic", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    axcol1.axvline(0.095, color='black',ls='--')
    axcol1.text(0.13,0.05, "2050--Less Optimistic", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    axcol0.spines["top"].set_visible(False)
    axcol0.spines["right"].set_visible(False)
    axcol1.fill_between([0.529, 1.3], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
    axcol1.axvline(1.3, color='black',ls='--')
    axcol1.text(0.85,0.05,  "Today's range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    ###California####

    CAL_sp = [x[1] for x in solarcal]
    CAL_wp = [x[2] for x in solarcal]
    CAL_sc = list(map(abs,[x[3] for x in solarcal]))



    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axcal0.scatter(s_cost, CAL_sc, s = 15, color = "C1") #Scatter or plot?
    axcal0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    #axcal0.set_ylabel("Curtailment")
    axcal0.set_facecolor("#eeeeee")
    axcal0.set_title("California")



    axcal1.stackplot(s_cost, CAL_sp, CAL_wp, colors = ["#f1c232","#2986cc"])
    axcal1.set_ylim(0, 1)
    axcal0.set_ylim(0, 1)


    #axcal1.set_ylabel("Penetration")
    axcal1.set_xlabel("Cost of Solar")

    axcal1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    xticks = axcal1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)

    axcal1.axvline(0.529, color='black',ls='--')
  
    axcal1.axvline(0.019, color='black',ls='--')
    axcal1.text(0.025,0.05, "2050--Optimistic", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    axcal1.axvline(0.095, color='black',ls='--')
    axcal1.text(0.13,0.05, "2050--Less Optimistic", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    axcal0.spines["top"].set_visible(False)
    axcal0.spines["right"].set_visible(False)
    axcal0.yaxis.set_ticklabels([])
    axcal1.yaxis.set_ticklabels([])
    axcal1.fill_between([0.529, 1.3], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
    axcal1.axvline(1.3, color='black',ls='--')
    axcal1.text(0.85,0.05,  "Today's range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
 
    for ax in plt.gcf().get_axes():
        ax.minorticks_on()
      
        ax.label_outer()

        ax.margins(x = 0)
        ax.set_xscale('log')        
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))
    


    axesp1.xaxis.set_ticklabels([])
    axden1.xaxis.set_ticklabels([])

    lines1, labels1 = axden1.get_legend_handles_labels()

    fig.legend(lines1, labels1, bbox_to_anchor=(0.65, 0.055), ncol=3)
    plt.savefig("Images/Figure2_solar_compare1.png")
    plt.show()
    


def pen_plus_wind_curtailALL():
    '''This makes a 2x2 grid of two axes each showing resource penetration and solar curtailment vs.
    a scaling log of solar. It is very long. It uses gridspec to order the axes, and other than that
    it is about the same as the other pen_plus_curtail() functions. 17/1
    
    This uses the github (Danish energy agency) value of onshore wind for the lower bound,
    and the NREL annual technology baseline for the upper bound.'''
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['hatch.linewidth'] = 1
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    inner_dnk = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], wspace=0.1, hspace=0, height_ratios = [1, 2])
    inner_esp = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], wspace=0.1, hspace=0, height_ratios = [1, 2])
    inner_col = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2], wspace=0.1, hspace=0, height_ratios = [1, 2])
    inner_cal = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[3], wspace=0.1, hspace=0, height_ratios = [1, 2])

    axden0 = plt.Subplot(fig, inner_dnk[0])
    axden1 = plt.Subplot(fig, inner_dnk[1])
    fig.add_subplot(axden0)
    fig.add_subplot(axden1)

    axesp0 = plt.Subplot(fig, inner_esp[0])
    axesp1 = plt.Subplot(fig, inner_esp[1])
    fig.add_subplot(axesp0)
    fig.add_subplot(axesp1)

    axcol0 = plt.Subplot(fig, inner_col[0])
    axcol1 = plt.Subplot(fig, inner_col[1])
    fig.add_subplot(axcol0)
    fig.add_subplot(axcol1)

    axcal0 = plt.Subplot(fig, inner_cal[0])
    axcal1 = plt.Subplot(fig, inner_cal[1])
    fig.add_subplot(axcal0)
    fig.add_subplot(axcal1)       

    ####DENMARK###
    w_cost = [x[6] for x in winddnk]
    w_cost = [item for sublist in w_cost for item in sublist]
    w_cost = [x / 10**6 /0.08442684282600257 for x in w_cost] 
    DNK_sp = [x[1] for x in winddnk]
    DNK_wp = [x[2] for x in winddnk]
    DNK_wc = list(map(abs,[x[4] for x in winddnk]))



    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axden0.scatter(w_cost, DNK_wc, s = 15, color = "C1") #Scatter or plot?
    axden0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axden0.set_ylabel("Curtailment")
    axden0.set_facecolor("#eeeeee")



    axden1.stackplot(w_cost, DNK_sp, DNK_wp, colors = ["#f1c232","#2986cc"], labels = ["Solar", "Wind"])
    axden1.set_ylim(0, 1)
    axden0.set_ylim(0, 1)

    axden1.set_ylabel("Penetration")
    #axden1.set_xlabel("Percent flexible source")

    axden1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axden0.set_title("Denmark")

    xticks = axden1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)


    #The 2.7 is the offshore wind price


    #We want to make a range for today's prices. the upper range is 
    axden0.spines["top"].set_visible(False)
    axden0.spines["right"].set_visible(False)
   


    ####SPAIN####

    ESP_sp = [x[1] for x in windesp]
    ESP_wp = [x[2] for x in windesp]
    ESP_wc = list(map(abs,[x[4] for x in windesp]))



    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axesp0.scatter(w_cost, ESP_wc, s = 15, color = "C1") #Scatter or plot?
    axesp0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    axesp0.set_facecolor("#eeeeee")
    axesp0.set_title("Spain")
 


    axesp1.stackplot(w_cost, ESP_sp, ESP_wp, colors = ["#f1c232","#2986cc"])
    axesp1.set_ylim(0, 1)
    axesp0.set_ylim(0, 1)

    axesp1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))


    xticks = axesp1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)

    axesp0.spines["top"].set_visible(False)
    axesp0.spines["right"].set_visible(False)
    axesp0.yaxis.set_ticklabels([])
    axesp1.yaxis.set_ticklabels([])

    ####Colorado#####

    COL_sp = [x[1] for x in windcol]
    COL_wp = [x[2] for x in windcol]
    COL_wc = list(map(abs,[x[4] for x in windcol]))

    axcol0.scatter(w_cost, COL_wc, s = 15, color = "C1") #Scatter or plot?
    axcol0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axcol0.set_ylabel("Curtailment")
    axcol0.set_facecolor("#eeeeee")
    axcol0.set_title("Colorado")



    axcol1.stackplot(w_cost, COL_sp, COL_wp, colors = ["#f1c232","#2986cc"])

    axcol1.set_ylim(0, 1)
    axcol0.set_ylim(0, 1)

    axcol1.set_ylabel("Penetration")
    axcol1.set_xlabel("Cost of Wind")

    axcol1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))


    xticks = axcol1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)

    axcol0.spines["top"].set_visible(False)
    axcol0.spines["right"].set_visible(False)

    
    
    ###California####

    CAL_sp = [x[1] for x in windcal]
    CAL_wp = [x[2] for x in windcal]
    CAL_wc = list(map(abs,[x[4] for x in windcal]))




    axcal0.scatter(w_cost, CAL_wc, s = 15, color = "C1") #Scatter or plot?
    axcal0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    axcal0.set_facecolor("#eeeeee")
    axcal0.set_title("California")



    axcal1.stackplot(w_cost, CAL_sp, CAL_wp, colors = ["#f1c232","#2986cc"])

    axcal1.set_ylim(0, 1)
    axcal0.set_ylim(0, 1)


    axcal1.set_xlabel("Cost of Wind")

    axcal1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    xticks = axcal1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)


  
    axcal0.spines["top"].set_visible(False)
    axcal0.spines["right"].set_visible(False)
    axcal0.yaxis.set_ticklabels([])
    axcal1.yaxis.set_ticklabels([])

 
    #This applies things for all axes
    for ax in plt.gcf().get_axes():
        ax.minorticks_on()
      
        ax.label_outer()

        ax.margins(x = 0)
        ax.set_xscale('log')        
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))


    #This applies things for only the axes of penetration.
    for ax in plt.gcf().get_axes()[1::2]:
        ax.fill_between([1.12, 1.22], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
        ax.axvline(1.12, color='black',ls='--')
        ax.axvline(1.22, color='black',ls='--')
        ax.text(1.4 ,0.05,  "Today's range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")

        ax.fill_between([0.57, 0.77], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "purple")
        ax.axvline(0.57, color='black',ls='--')
        ax.axvline(0.77, color='black',ls='--')
        ax.text(0.65, 0.05, "Future range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    
    # for ax in plt.gcf().get_axes()[::2]:
    #     ax.set_ylim(0,1)


    axesp1.xaxis.set_ticklabels([])
    axden1.xaxis.set_ticklabels([])

    lines1, labels1 = axden1.get_legend_handles_labels()

    fig.legend(lines1, labels1, bbox_to_anchor=(0.65, 0.055), ncol=3)

    #print(fig.axes[1::2])

    plt.savefig("Images/Figure_wind_compare_heat_var2.png")
    plt.show()


def pen_plus_batt_curtailALL():
    '''This makes a 2x2 grid of two axes each showing resource penetration and solar curtailment vs.
    a scaling log of solar. It is very long. It uses gridspec to order the axes, and other than that
    it is about the same as the other pen_plus_curtail() functions. 17/1
    
    This uses the github (Danish energy agency) value of onshore wind for the lower bound,
    and the NREL annual technology baseline for the upper bound.'''
    plt.rcdefaults()

    plt.rcParams.update({'font.size': 14})
    plt.rcParams['hatch.linewidth'] = 1
    fig = plt.figure(figsize=(10, 9))
    outer = gridspec.GridSpec(3, 2, wspace=0.2, hspace=0.3, height_ratios = [1, 1, 0.01])
    inner_dnk = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], wspace=0.1, hspace=0, height_ratios = [1, 2])
    inner_esp = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], wspace=0.1, hspace=0, height_ratios = [1, 2])
    inner_col = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2], wspace=0.1, hspace=0, height_ratios = [1, 2]) #(gridspec(3,1)) to add some extra space
    inner_cal = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[3], wspace=0.1, hspace=0, height_ratios = [1, 2])

    axden0 = plt.Subplot(fig, inner_dnk[0])
    axden1 = plt.Subplot(fig, inner_dnk[1])
    fig.add_subplot(axden0)
    fig.add_subplot(axden1)

    axesp0 = plt.Subplot(fig, inner_esp[0])
    axesp1 = plt.Subplot(fig, inner_esp[1])
    fig.add_subplot(axesp0)
    fig.add_subplot(axesp1)


    axcol0 = plt.Subplot(fig, inner_col[0])
    axcol1 = plt.Subplot(fig, inner_col[1])
    fig.add_subplot(axcol0)
    fig.add_subplot(axcol1)




    axcal0 = plt.Subplot(fig, inner_cal[0])
    axcal1 = plt.Subplot(fig, inner_cal[1])
    fig.add_subplot(axcal0)
    fig.add_subplot(axcal1)     

    ####DENMARK###
    b_cost = [x[7] for x in battdnk]
    b_cost = [item for sublist in b_cost for item in sublist]
    b_cost = [x / 10**6 /0.09439292574325567  for x in b_cost] 

    DNK_sp = [x[1] for x in battdnk]
    DNK_wp = [x[2] for x in battdnk]
    DNK_sc = list(map(abs,[x[3] for x in battdnk]))
    DNK_wc = list(map(abs,[x[4] for x in battdnk]))
    DNK_bp = list(map(abs,[x[8] for x in battdnk]))



    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axden0.scatter(b_cost, DNK_sc, s = 15, color = "C1", label = "solar curtailment")
    axden0.scatter(b_cost, DNK_wc, s = 15, color = "C0", label = "wind curtailment")
       
    axden0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axden0.set_ylabel("Curtailment")
    axden0.set_facecolor("#eeeeee")



    axden1.stackplot(b_cost, DNK_sp, DNK_wp, colors = ["#f1c232","#2986cc"], labels = ["Solar", "Wind"])
    axden1.set_ylim(0, 1)
    axden0.set_ylim(0, 1)

    axden1.set_ylabel("Penetration")


    axden1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axden0.set_title("Denmark")

    xticks = axden1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)


    #We want to make a range for today's prices. the upper range is 
    axden0.spines["top"].set_visible(False)
    axden0.spines["right"].set_visible(False)
   


    ####SPAIN####

    ESP_sp = [x[1] for x in battesp]
    ESP_wp = [x[2] for x in battesp]
    ESP_wc = list(map(abs,[x[4] for x in battesp]))
    ESP_sc = list(map(abs,[x[3] for x in battesp]))
    ESP_bp = list(map(abs,[x[8] for x in battesp]))

    axesp0.scatter(b_cost, ESP_wc, s = 15, color = "C0") 
    axesp0.scatter(b_cost, ESP_sc, s = 15, color = "C1") 
    axesp0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    axesp0.set_facecolor("#eeeeee")
    axesp0.set_title("Spain")




    axesp1.stackplot(b_cost, ESP_sp, ESP_wp, colors = ["#f1c232","#2986cc"])
    axesp1.set_ylim(0, 1)
    axesp0.set_ylim(0, 1)

    axesp1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))


    xticks = axesp1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)

    axesp0.spines["top"].set_visible(False)
    axesp0.spines["right"].set_visible(False)
    axesp0.yaxis.set_ticklabels([])
    axesp1.yaxis.set_ticklabels([])

    ####Colorado#####

    COL_sp = [x[1] for x in battcol]
    COL_wp = [x[2] for x in battcol]
    COL_wc = list(map(abs,[x[4] for x in battcol]))
    COL_sc = list(map(abs,[x[3] for x in battcol]))    
    COL_bp = list(map(abs,[x[8] for x in battcol]))

    axcol0.scatter(b_cost, COL_wc, s = 15, color = "C0") 
    axcol0.scatter(b_cost, COL_sc, s = 15, color = "C1")     
    axcol0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axcol0.set_ylabel("Curtailment")
    axcol0.set_facecolor("#eeeeee")
    axcol0.set_title("Colorado")



    axcol1.stackplot(b_cost, COL_sp, COL_wp, colors = ["#f1c232","#2986cc"])

    axcol1.set_ylim(0, 1)
    axcol0.set_ylim(0, 1)

    axcol1.set_ylabel("Penetration")
    axcol1.set_xlabel("Cost of Battery")

    axcol1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))


    xticks = axcol1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)

    axcol0.spines["top"].set_visible(False)
    axcol0.spines["right"].set_visible(False)

    
    
    ###California####

    CAL_sp = [x[1] for x in battcal]
    CAL_wp = [x[2] for x in battcal]
    CAL_wc = list(map(abs,[x[4] for x in battcal]))
    CAL_sc = list(map(abs,[x[3] for x in battcal]))
    CAL_bp = list(map(abs,[x[8] for x in battcal]))    




    axcal0.scatter(b_cost, CAL_wc, s = 15, color = "C0")
    axcal0.scatter(b_cost, CAL_sc, s = 15, color = "C1")
    axcal0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    axcal0.set_facecolor("#eeeeee")
    axcal0.set_title("California")



    axcal1.stackplot(b_cost, CAL_sp, CAL_wp, colors = ["#f1c232","#2986cc"])

    axcal1.set_ylim(0, 1)
    axcal0.set_ylim(0, 1)


    axcal1.set_xlabel("Cost of Battery")

    axcal1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    xticks = axcal1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)


  
    axcal0.spines["top"].set_visible(False)
    axcal0.spines["right"].set_visible(False)
    axcal0.yaxis.set_ticklabels([])
    axcal1.yaxis.set_ticklabels([])

 
    #This applies things for all axes

    #This applies things for only the axes of penetration.
    for ax in plt.gcf().get_axes()[1::2]:
        ax.fill_between([0.232, 0.311], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
        ax.axvline(0.232, color='black',ls='--')
        ax.axvline(0.311, color='black',ls='--')
        ax.text(0.27 ,0.05,  "Today's range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")

        ax.fill_between([0.075, 0.22], y1 = 1, alpha = 0.2, edgecolor = "k", hatch = "XX", facecolor = "purple")
        ax.axvline(0.075, color='black',ls='--')
        ax.axvline(0.22, color='black',ls='--')
        ax.text(0.135, 0.05, "Future range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    
    # for ax in plt.gcf().get_axes()[::2]:
    #     ax.set_ylim(0,1)


    axesp1.xaxis.set_ticklabels([])
    axden1.xaxis.set_ticklabels([])






    for ax in plt.gcf().get_axes():
        ax.minorticks_on()
      
        ax.label_outer()

        ax.margins(x = 0)
        ax.set_xscale('log')        
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))

    
#    plt.rcParams["font.weight"] = "bold"
    axden0b = axden0.twinx()
    axden0b.scatter(b_cost, DNK_bp, s = 15, color = "C2", label = "Battery") 
    axden0b.tick_params(axis = "y", labelcolor = "C2")

    axesp0b = axesp0.twinx()
    axesp0b.scatter(b_cost, ESP_bp, s = 15, color = "C2", label = "battery penetration")
    axesp0b.tick_params(axis = "y", labelcolor = "C2") 
    axesp0b.set_ylabel("Battery\nfraction")
 
    axcol0b = axcol0.twinx()
    axcol0b.scatter(b_cost, COL_bp, s = 15, color = "C2", label = "battery penetration") 
    axcol0b.tick_params(axis = "y", labelcolor = "C2") 

    axcal0b = axcal0.twinx()
    axcal0b.scatter(b_cost, CAL_bp, s = 15, color = "C2", label = "battery penetration") 
    axcal0b.tick_params(axis = "y", labelcolor = "C2")
    axcal0b.set_ylabel("Battery\nfraction")


    lines1, labels1 = axden1.get_legend_handles_labels()
    #lines2, labels2 = axden0.get_legend_handles_labels()
    lines3, labels3 = axden0b.get_legend_handles_labels()

    fig.legend(lines1 +lines3, labels1+labels3, bbox_to_anchor=(0.75, 0.1), ncol = 3)

    #print(fig.axes[1::2])

    plt.savefig("Images/Figure_batt_compare2.png")
    plt.show()






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






##--##

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

    capital_cost_onshorewind = annuity(27,0.07)*1118000*(1+0.012) #EUR/MW



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



    #utility scale solar
    capital_cost_solar = annuity(35,0.07)*529000*(1+0.016)




    #For gas, we assume the same cost in all countries. we just need to add a generator to each network,
    # network, nspain, etc
    capital_cost_OCGT = annuity(25,0.07)*453000*(1+0.018) # in €/MW
    fuel_cost = 21.6 # in €/MWh_th
    efficiency = 0.39
    marginal_cost_OCGT = fuel_cost/efficiency # in €/MWh_el

    add_generators(mynetworks, CF_wind, capital_cost_onshorewind, 
                CF_solar, capital_cost_solar, 
                capital_cost_OCGT, marginal_cost_OCGT)


    capital_cost_electrolysis = annuity(25, 0.07) * 650000 * (1 + 0.02) 

    capital_cost_H2 = annuity(100, 0.07) * 3000 #EUR/MWh

    capital_cost_fuelcell = annuity(10, 0.07) * 1300000 * (1 + 0.05)

    capital_cost_inverter = annuity(10, 0.07) * 270000 * (1 + 0.002)

    capital_cost_battery = annuity(20, 0.07) * 232000


    for network in mynetworks:
        add_storage(network, capital_cost_electrolysis, capital_cost_H2, 
                    capital_cost_fuelcell, capital_cost_battery, capital_cost_inverter)


    ##--------------------<<<Model Runs: Just electricity>>>------------------------

    #These four below return 100 points of cost vs solar penetration. 
    # for network in mynetworks:
    #     reset_stats(network)

    #make_dir("This_is_a_test")
    #find_solar_data(Denmark, "Denmark", 100000, "This_is_a_test")

    with Pool(processes=4) as pool:
        pool.map(find_solar_data, [repeat(Denmark), repeat("Denmark"), np.logspace(4, 6.31, 4), repeat("This_is_a_test")]) 



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



    ##------------------------<<<Mod electricity loads with increased cooling/global warming demand>>>---------------
    '''I wonder if what I was doing before was comparing the electricity demand with the cooling/global warming,
    or the elec with heating with the cooling/global warming'''


    #df_elec, df_co_elec, df_cal_elec = increase_temp (2, 2)


    # Denmark.remove("Load", "load")
    # Denmark.add("Load",
    #             "load", 
    #             bus="electricity bus", 
    #             p_set=df_elec['DNKcombine'])
    # Spain.remove("Load", "load")
    # Spain.add("Load",
    #             "load", 
    #             bus="electricity bus", 
    #             p_set=df_elec['ESPcombine'])
    # CA.remove("Load", "load")
    # CA.add("Load",
    #             "load", 
    #             bus="electricity bus", 
    #             p_set=df_cal_elec["adjust_elec_demand"])
    # CO.remove("Load", "load")
    # CO.add("Load",
    #             "load", 
    #             bus="electricity bus", 
    #             p_set=df_co_elec["adjust_elec_demand"])



    ###------------------<<<New Model Runs>>>---------------------------------
    # 27/9: Finding sensitivity to solar cost with and without heating demand. 
    # 3/10: Want to make new runs for the addition of heating demand for the wind and battery cost

    #make_dir("solarcost_w_heat_27_Sept")
    for network in mynetworks:
        reset_stats(network)

    # list(map(find_solar_data, repeat(Denmark), repeat("Denmark"), np.logspace(4, 6.31, 100), repeat("solarcost_w_heat_27_Sept")))
    # list(map(find_solar_data, repeat(Spain), repeat("Spain"), np.logspace(4, 6.31, 100), repeat("solarcost_w_heat_27_Sept")))

    # list(map(find_solar_data, repeat(CA), repeat("CA"), np.logspace(4, 6.31, 100), repeat("solarcost_w_heat_27_Sept")))
    # list(map(find_solar_data, repeat(CO), repeat("CO"), np.logspace(4, 6.31, 100), repeat("solarcost_w_heat_27_Sept")))

    # make_dir("windcostLOGFeb24_2deg_2sl")


    # for network in mynetworks:
    #     reset_stats(network)




    # list(map(find_wind_data, repeat(Denmark), repeat("Denmark"), np.logspace(5, 6.5, 100), repeat("windcostLOGFeb24_2deg_2sl")))
    # list(map(find_wind_data, repeat(Spain), repeat("Spain"), np.logspace(5, 6.5, 100), repeat("windcostLOGFeb24_2deg_2sl")))

    # list(map(find_wind_data, repeat(CA), repeat("CA"), np.logspace(5, 6.5, 100), repeat("windcostLOGFeb24_2deg_2sl")))
    # list(map(find_wind_data, repeat(CO), repeat("CO"), np.logspace(5, 6.5, 100), repeat("windcostLOGFeb24_2deg_2sl")))




    # for network in mynetworks:
    #     reset_stats(network)

    # make_dir("battcostLOGFeb24_2deg_2sl")
    # list(map(find_batt_data, repeat(Denmark), repeat("Denmark"), np.logspace(4.5, 6, 100), repeat("battcostLOGFeb24_2deg_2sl")))
    # list(map(find_batt_data, repeat(Spain), repeat("Spain"), np.logspace(4.5, 6, 100), repeat("battcostLOGFeb24_2deg_2sl")))

    # list(map(find_batt_data, repeat(CA), repeat("CA"), np.logspace(4.5, 6, 100), repeat("battcostLOGFeb24_2deg_2sl")))
    # list(map(find_batt_data, repeat(CO), repeat("CO"), np.logspace(4.5, 6, 100), repeat("battcostLOGFeb24_2deg_2sl")))



    ##------------------<<<Make into csv file>>>------------------------------
    '''Looks at a list of the chosen directories and makes them into a csv file'''
    #netcdf_csv_all(datasets) #Takes a 