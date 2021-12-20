import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import repeat



'''As it stands, this code does not work unforunately'''

def annuity(n,r):
    """Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20,0.05)*20 = 1.6"""

    if r > 0:
        return r/(1. - 1./(1.+r)**n)
    else:
        return 1/n

def prep_networks():
    '''This function preps the four systems (DNK, ESP, CA, CO) by making networks for each system, loading hours and demand,
    adding carriers. It also preps the capacity factors of wind and solar'''
    Denmark = pypsa.Network()
    Spain = pypsa.Network()
    CA = pypsa.Network()
    CO = pypsa.Network() 

    mynetworks = [Denmark, Spain, CA, CO]

    hours_in_2015 = pd.date_range('2015-01-01T00:00Z','2015-12-31T23:00Z', freq='H') #for network, nspain
    hours_in_2011 = pd.date_range('2011-01-01T00:00:00','2011-12-31T23:00:00', freq='H') #for ncal, nco

    #Set the hours for each network
    for n in mynetworks:
        if mynetworks.index(n)//2 == 0:
            n.set_snapshots(hours_in_2015)
        else:
            n.set_snapshots(hours_in_2011)

    #Add electricity bus
    for n in mynetworks:
        n.add("Bus", "electricity bus")  


    df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0)# in MWh
    df_elec.index = pd.to_datetime(df_elec.index) #change index to datatime

    df_cal_elec = pd.read_csv('data_extra/CaliforniaTimeSeries.csv', index_col=0)
    df_cal_elec.index = pd.to_datetime(df_cal_elec.index)

    df_co_elec = pd.read_csv('data_extra/ColoradoTimeSeries.csv', index_col=0)
    df_co_elec.index = pd.to_datetime(df_co_elec.index)

    #Add loads
    mynetworks[0].add("Load",
                "load", 
                bus="electricity bus", 
                p_set=df_elec['DNK'])

    mynetworks[1].add("Load",
                "load", 
                bus="electricity bus", 
                p_set=df_elec['ESP'])

    mynetworks[2].add("Load",
                "load", 
                bus="electricity bus", 
                p_set=df_cal_elec['demand_mwh'])

    mynetworks[3].add("Load",
                "load", 
                bus="electricity bus", 
                p_set=df_co_elec['demand_mwh'])


    #Add carriers
    for n in mynetworks:
        n.add("Carrier", "gas", co2_emissions = 0.19)
        n.add("Carrier", "onshorewind")
        n.add("Carrier", "solar")


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
    #utility scale solar
    

    return mynetworks, CF_wind, CF_solar

def add_generators():

    '''This retrieves the tuple of networks and capacity factors for wind and solar from prep_networks(),
    adds the generators according to the capacity factors, and returns the networks. '''
    
    networks, CF_wind, CF_solar = prep_networks()

    cap_cost_solar = annuity(35,0.07)*529000*(1+0.016)
    cap_cost_wind = annuity(27,0.07)*1118000*(1+0.012) #onshore wind
    cap_cost_gas = annuity(25,0.07)*453000*(1+0.018) # in €/MW
    fuel_cost = 21.6 # in €/MWh_th
    efficiency = 0.39
    marginal_cost_gas = fuel_cost/efficiency
    
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

    return networks

def add_storage():
    '''This retrieves the networks from add_generators, adds storage, and returns the networks'''
    networks = add_generators()

    cost_electro = annuity(25, 0.07) * 650000 * (1 + 0.02) #capital cost electrolysis

    cost_H2 = annuity(100, 0.07) * 3000 #EUR/MWh, capital cost H2

    cost_fuelcell= annuity(10, 0.07) * 1300000 * (1 + 0.05) #capital cost fuelcell

    cost_inverter = annuity(10, 0.07) * 270000 * (1 + 0.002)

    cost_batt = annuity(20, 0.07) * 232000

    for n in networks:
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

    return networks



    #add CO2 constraint


#Unpacking the networks might be required
Denmark, Spain, CA, CO = add_storage()

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
    
    
    ##systemcost = n.objective/n.loads_t.p.sum()
    
    
    #This now expresses solar in terms of a percent of its capacity
    s_curtailment = (n.generators_t.p-n.generators.p_nom_opt * n.generators_t.p_max_pu)['solar'].sum()
    
    ##w_curtailment = (n.generators_t.p-n.generators.p_nom_opt * n.generators_t.p_max_pu)['onshorewind'].sum()

    ###If you want to get a plot of curtailment alone, then delete the following lines 
    #of code until the return statement###
    max_gen = (n.generators.p_nom_opt * n.generators_t.p_max_pu)['solar'].sum()
    ##max_gen_w = (n.generators.p_nom_opt * n.generators_t.p_max_pu)['onshorewind'].sum()
    

    #We want to get the percent of energy curtailed. However, if max_gen is 0, then
    #we get a number that is undefined. We must use loc 
    if max_gen == 0:
        s_curtailment = 0
    else:
        s_curtailment = s_curtailment/max_gen

    # if max_gen_w == 0:
    #     w_curtailment = 0
    # else:
    #     w_curtailment = w_curtailment/max_gen
    
    ###You can delete the code above if you wish###
    
    
    ##We also want to  find out the amount of power used by battery
    #battery_pen = n.links_t.p1["battery discharger"].sum()/sum(n.generators_t.p.sum())
    
    #hydrogen_pen = n.links_t.p1["H2 Fuel Cell"].sum()/sum(n.generators_t.p.sum())
    
    
    return ((solar_penetration, wind_penetration, gas_penetration), 
            s_curtailment)



DNK_solar_data = list(map(find_solar_data, repeat(Denmark), np.logspace(10000, 1000000, 2)))
# ESP_solar_data = list(map(find_solar_data, repeat(Spain), np.linspace(50000, 600000, 10)))
# CAL_solar_data = list(map(find_solar_data, repeat(CA), np.linspace(50000, 600000, 10)))
# CO_solar_data = list(map(find_solar_data, repeat(CO), np.linspace(50000, 600000, 10)))


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