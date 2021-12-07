import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import repeat


#network = Denmark, nspain, ncal, ncolorado
Denmark = pypsa.Network()
Spain = pypsa.Network()
CA = pypsa.Network()
CO = pypsa.Network()    
    
mynetworks = [Denmark, Spain, CA, CO]

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

    



add_generators(mynetworks, CF_wind, capital_cost_onshorewind, 
               CF_solar, capital_cost_solar, 
               capital_cost_OCGT, marginal_cost_OCGT)


capital_cost_electrolysis = annuity(25, 0.07) * 650000 * (1 + 0.02) 

capital_cost_H2 = annuity(100, 0.07) * 3000 #EUR/MWh

capital_cost_fuelcell = annuity(10, 0.07) * 1300000 * (1 + 0.05)

capital_cost_inverter = annuity(10, 0.07) * 270000 * (1 + 0.002)

capital_cost_battery = annuity(20, 0.07) * 232000

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
    
    
    
for network in mynetworks:
    add_storage(network, capital_cost_electrolysis, capital_cost_H2, 
                capital_cost_fuelcell, capital_cost_battery, capital_cost_inverter)




def reset_stats(n):
    n.generators.loc[['solar'], ['capital_cost']] = annuity(35,0.07)*529000*(1+0.016)
    n.generators.loc[['onshorewind'], ['capital_cost']] = annuity(27,0.07)*1118000*(1+0.012)
    n.generators.loc[['OCGT'], ['capital_cost']] = annuity(25,0.07)*453000*(1+0.018)

def find_solar_data(n, solar_cost):
    #Takes annualized coefficient and multiplies by investment cost
  
    annualized_solar_cost =  0.07846970300338728* solar_cost
    n.generators.loc[['solar'],['capital_cost']] = annualized_solar_cost
    
    #this substitutes the current solar cost in our generator for a new cost

    
    n.lopf(n.snapshots, 
             pyomo=False,
             solver_name='gurobi')
    
    #commenting out the sum of generators--battery is so small, we need raw values
    solar_penetration = n.generators_t.p['solar'].sum()/sum(n.generators_t.p.sum())
    wind_penetration = n.generators_t.p['onshorewind'].sum()/sum(n.generators_t.p.sum())
    gas_penetration = n.generators_t.p['OCGT'].sum()/sum(n.generators_t.p.sum())
    
    
    systemcost = n.objective/n.loads_t.p.sum()
    
    
    #This now expresses solar in terms of a percent of its capacity
    s_curtailment = (n.generators_t.p-n.generators.p_nom_opt * n.generators_t.p_max_pu)['solar'].sum()
    
    w_curtailment = (n.generators_t.p-n.generators.p_nom_opt * n.generators_t.p_max_pu)['onshorewind'].sum()
    ###If you want to get a plot of curtailment alone, then delete the following lines 
    #of code until the return statement###
    max_gen = (n.generators.p_nom_opt * n.generators_t.p_max_pu)['solar'].sum()
    max_gen_w = (n.generators.p_nom_opt * n.generators_t.p_max_pu)['onshorewind'].sum()
    

    #We want to get the percent of energy curtailed. However, if max_gen is 0, then
    #we get a number that is undefined. We must use loc 
    if max_gen == 0:
        s_curtailment = 0
    else:
        s_curtailment = s_curtailment/max_gen

    if max_gen_w == 0:
        w_curtailment = 0
    else:
        w_curtailment = w_curtailment/max_gen
    
    ###You can delete the code above if you wish###
    
    
    ##We also want to  find out the amount of power used by battery
    battery_pen = n.links_t.p1["battery discharger"].sum()/sum(n.generators_t.p.sum())
    
    hydrogen_pen = n.links_t.p1["H2 Fuel Cell"].sum()/sum(n.generators_t.p.sum())
    
    
    return ((solar_penetration, wind_penetration, gas_penetration, battery_pen, hydrogen_pen), systemcost, 
            (s_curtailment, w_curtailment))





#These four below return 100 points of cost vs solar penetration. 
for network in mynetworks:
    reset_stats(network)




DNK_solar_data = list(map(find_solar_data, repeat(Denmark), np.linspace(50000, 600000, 10)))
ESP_solar_data = list(map(find_solar_data, repeat(Spain), np.linspace(50000, 600000, 10)))
CAL_solar_data = list(map(find_solar_data, repeat(CA), np.linspace(50000, 600000, 10)))
CO_solar_data = list(map(find_solar_data, repeat(CO), np.linspace(50000, 600000, 10)))


def penetration_chart():
    #This is our x axis, solar_cost (s_cost)
    s_cost = np.linspace(50000, 600000, 10)

    #This is our y axis. sp = solar penetration, wp = wind penetration, gp = gas penetration
    DNK_sp = [x[0][0] for x in DNK_solar_data]
    ESP_sp = [x[0][0] for x in ESP_solar_data]
    CAL_sp = [x[0][0]  for x in CAL_solar_data]
    CO_sp = [x[0][0]  for x in CO_solar_data]

    DNK_wp = [x[0][1] for x in DNK_solar_data]
    ESP_wp = [x[0][1] for x in ESP_solar_data]
    CAL_wp = [x[0][1] for x in CAL_solar_data]
    CO_wp = [x[0][1] for x in CO_solar_data]

    DNK_gp = [x[0][2] for x in DNK_solar_data]
    ESP_gp = [x[0][2] for x in ESP_solar_data]
    CAL_gp = [x[0][2] for x in CAL_solar_data]
    CO_gp = [x[0][2] for x in CO_solar_data]

    DNK_bp = [x[0][3] for x in DNK_solar_data]
    ESP_bp = [x[0][3] for x in ESP_solar_data]
    CAL_bp = [x[0][3] for x in CAL_solar_data]
    CO_bp = [x[0][3] for x in CO_solar_data]

    DNK_hp = [x[0][4] for x in DNK_solar_data]  
    ESP_hp = [x[0][4] for x in ESP_solar_data]
    CAL_hp = [x[0][4] for x in CAL_solar_data]
    CO_hp = [x[0][4] for x in CO_solar_data]


    fig, axs = plt.subplots(2,2)
    axs[0, 0].stackplot(s_cost, DNK_sp, DNK_wp, DNK_gp, DNK_bp, DNK_hp, labels = ["solar", "wind", "gas"]
                    , colors = ["#ffd966", "#2986cc", "#d5a6bd"])
    axs[0, 0].set_title("Denmark penetration")
    axs[0, 1].stackplot(s_cost, ESP_sp, ESP_wp, ESP_gp, ESP_bp, ESP_hp, labels = ["solar", "wind", "gas"]
                    , colors = ["#ffd966", "#2986cc", "#d5a6bd"])
    axs[0, 1].set_title("Spain penetration")
    axs[1, 0].stackplot(s_cost, CO_sp, CO_wp, CO_gp, CO_bp, CO_hp, labels = ["solar", "wind", "gas"]
                    , colors = ["#ffd966", "#2986cc", "#d5a6bd"])
    axs[1, 0].set_title("Colorado penetration")
    axs[1, 1].stackplot(s_cost, CAL_sp, CAL_wp, CAL_gp, CAL_bp, CAL_hp, labels = ["solar", "wind", "gas"]
                    , colors = ["#ffd966", "#2986cc", "#d5a6bd"])
    axs[1, 1].set_title("California penetration")



    for ax in axs.flat:
        ax.minorticks_on()
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title='Resource type', loc='upper right')
        ax.grid(which = 'major', linestyle='-', linewidth='0.5', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax.set(xlabel='solar cost (EUR/MW)', ylabel='penetration')
    #Hide x labels and tick labels for top plots and y ticks for right plots.
        ax.label_outer()
        ax.axvline(529000, color='black',ls='--')
        ax.text(529000,1.1, "Current cost = 529000 EUR", horizontalalignment = "center")


    plt.suptitle("Penetration per technology by solar overnight investment cost", fontsize = 20)
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    fig.set_size_inches(18.5, 10.5)
    #plt.savefig("Images/PenPerTechbySolarCost", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()


def system_cost():
    s_cost = np.linspace(50000, 600000, 10)

    DNK_cst = [x[1] for x in DNK_solar_data]
    ESP_cst = [x[1] for x in ESP_solar_data]
    CAL_cst = [x[1] for x in CAL_solar_data]
    CO_cst = [x[1] for x in CO_solar_data]




    fig, axs = plt.subplots(2,2)
    axs[0, 0].plot(s_cost, DNK_cst, 'ro')
    axs[0, 0].set_title("Denmark system cost")
    axs[0, 1].plot(s_cost, ESP_cst, 'bo')
    axs[0, 1].set_title("Spain system cost")
    axs[1, 0].plot(s_cost, CO_cst, 'co')
    axs[1, 0].set_title("Colorado system cost")
    axs[1, 1].plot(s_cost, CAL_cst, 'go')
    axs[1, 1].set_title("California system cost")



    for ax in axs.flat:
        ax.minorticks_on()
        ax.set_ylim(50, 110)
        ax.grid(which = 'major', linestyle='-', linewidth='0.5', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax.set(xlabel='solar cost (EUR/MW)', ylabel='system cost (EUR/MWh)')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
        ax.label_outer()
        ax.axvline(529000, color='black',ls='--')
        ax.text(529000,66, "Current cost = 529000 EUR", horizontalalignment = "center")

    plt.suptitle("Total system cost by solar overnight investment cost", fontsize = 20)
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    fig.set_size_inches(18.5, 10.5)
    #plt.savefig("Images/TotalSystemCostbySolarCost", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()

def showcurtailment():
    s_cost = np.linspace(50000, 600000, 10)

    DNK_sc = list(map(abs, [x[2][0] for x in DNK_solar_data]))
    ESP_sc = list(map(abs, [x[2][0] for x in ESP_solar_data]))
    CAL_sc = list(map(abs, [x[2][0] for x in CAL_solar_data]))
    CO_sc = list(map(abs, [x[2][0] for x in CO_solar_data]))
    

    DNK_wc = list(map(abs, [x[2][1] for x in DNK_solar_data]))
    ESP_wc = list(map(abs, [x[2][1] for x in ESP_solar_data]))
    CAL_wc = list(map(abs, [x[2][1] for x in CAL_solar_data]))
    CO_wc = list(map(abs, [x[2][1] for x in CO_solar_data]))

    # DNK_sc_wo = list(map(abs, [x[2] for x in DNK_solar_wo]))
    # ESP_sc_wo = list(map(abs, [x[2] for x in ESP_solar_wo]))
    # CAL_sc_wo = list(map(abs, [x[2] for x in CAL_solar_wo]))
    # CO_sc_wo = list(map(abs, [x[2] for x in CO_solar_wo]))


    fig, axs = plt.subplots(2,2)
    axs[0, 0].plot(s_cost, DNK_sc, 'o', color = "orange", label = "Solar")
    axs[0, 0].plot(s_cost, DNK_wc, 'bo', label = "Wind")
    #axs[0, 0].plot(w_cost, DNK_sc, 'ko', label = "Without storage")
    axs[0, 0].set_title("Denmark tech curtailed percent")
    axs[0, 1].plot(s_cost, ESP_sc, 'o', color = "orange", label = "Solar")
    axs[0, 1].plot(s_cost, ESP_wc, 'bo', label = "Wind")
    #axs[0, 1].plot(s_cost, ESP_sc_wo, 'ko', label = "Without storage")
    axs[0, 1].set_title("Spain tech curtailed percent")
    axs[1, 0].plot(s_cost, CO_sc, 'o', color = "orange", label = "Solar")
    axs[1, 0].plot(s_cost, CO_wc, 'bo', label = "Wind")
    #axs[1, 0].plot(s_cost, CO_sc_wo, 'ko', label = "Without storage")
    axs[1, 0].set_title("Colorado tech curtailed percent")
    axs[1, 1].plot(s_cost, CAL_sc, 'o', color = "orange", label = "Solar")
    axs[1, 1].plot(s_cost, CAL_wc, 'bo', label = "Wind")
    #axs[1, 1].plot(s_cost, CAL_sc_wo, 'ko', label = "Without storage")
    axs[1, 1].set_title("California tech curtailed percent")



    for ax in axs.flat:
        ax.minorticks_on()
        ax.legend(title= "type of resource", loc='upper right')
        ax.grid(which = 'major', linestyle='-', linewidth='0.5', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax.set(xlabel='solar cost (EUR/MW)', ylabel='solar curtailment fraction')
        ax.set_ylim(-0.02, 0.7)
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        ax.label_outer()
        ax.axvline(529000, color='black',ls='--')
        ax.text(529000,0.75, "Current cost = 529000 EUR", horizontalalignment = "center")

    plt.suptitle("Fraction of curtailment by solar overnight investment cost", fontsize = 20)
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    fig.set_size_inches(18.5, 10.5)
    plt.savefig("Images/FracSolarCurtailbySolarCostStore_w_and_wo", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()
#penetration_chart()