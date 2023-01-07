import pypsa
import pandas as pd
from helpers import annual_cost

def set_hours(n):
    '''So far, we have been using 2011 data for CO and CA. Now 2019 data may be available.'''
    hours_in_2015 = pd.date_range('2015-01-01T00:00Z','2015-12-31T23:00Z', freq='H') #for network, nspain
    hours_in_2011 = pd.date_range('2011-01-01T00:00:00','2011-12-31T23:00:00', freq='H') #for ncal, nco

    if n.name in ['Denmark', 'Spain']:
        n.set_snapshots(hours_in_2015)
    else:
        n.set_snapshots(hours_in_2011)
    
    return n

def add_buses(n):

    n.add("Bus", "electricity bus")  
    n.add("Bus", "battery", carrier = "battery")

    return n

def add_load(n):
    '''This simply adds the electricity load to the different networks. It differentiates
    them based on the name'''

    #The df_elec is made with data from t

    #I believe that the CaliforniaTimeSeries was taken from CAISO

    
    if n.name == "Denmark":
        df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0)# in MWh
        df_elec.index = pd.to_datetime(df_elec.index) #change index to datatime

        n.add("Load",
            "load", 
            bus="electricity bus", 
            p_set=df_elec['DNK'])

    elif n.name == "Spain":
        df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0)# in MWh
        df_elec.index = pd.to_datetime(df_elec.index) #change index to datatime

        n.add("Load",
            "load", 
            bus="electricity bus", 
            p_set=df_elec['ESP'])

    elif n.name == "California":
        df_cal_elec = pd.read_csv('data_extra/CaliforniaTimeSeries.csv', index_col=0)
        df_cal_elec.index = pd.to_datetime(df_cal_elec.index)

        n.add("Load",
            "load", 
            bus="electricity bus", 
            p_set=df_cal_elec['demand_mwh'])

    elif n.name == "Colorado":
        df_co_elec = pd.read_csv('data_extra/ColoradoTimeSeries.csv', index_col=0)
        df_co_elec.index = pd.to_datetime(df_co_elec.index)

        n.add("Load",
            "load", 
            bus="electricity bus", 
            p_set=df_co_elec['demand_mwh'])

    else:
 
        print(KeyError("Network name not recognized as valid: (choose 'Denmark', 'Spain', 'California', or 'Colorado')"))

    
    return n

def add_generators(n):
    '''We are only interested in systems with no '''
    n.add("Generator",
                "onshorewind",
                bus = "electricity bus",
                p_nom_extendable = True,
                carrier = "onshorewind",
                capital_cost = annual_cost("onwind"),
                marginal_cost = 0) #This is the timeseries.
    
    n.add("Generator",
                "solar",
                bus = "electricity bus",
                p_nom_extendable = True,
                carrier = "solar",
                capital_cost = annual_cost("solar-utility"),
                marginal_cost = 0)

    if n.name == "Denmark":
        df_onshorewind = pd.read_csv('data_extra/onshore_wind_1979-2017.csv', sep=';', index_col=0)
        df_onshorewind.index = pd.to_datetime(df_onshorewind.index)
        CFw_Denmark = df_onshorewind['DNK'][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in n.snapshots]]
        
        df_solar = pd.read_csv('data_extra/pv_optimal.csv', sep=';', index_col=0)
        df_solar.index = pd.to_datetime(df_solar.index)
        CFs_Denmark = df_solar['DNK'][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in n.snapshots]]

        n.generators_t.p_max_pu['onshorewind'] = CFw_Denmark
        n.generators_t.p_max_pu['solar'] = CFs_Denmark

    elif n.name == "Spain":
        df_onshorewind = pd.read_csv('data_extra/onshore_wind_1979-2017.csv', sep=';', index_col=0)
        df_onshorewind.index = pd.to_datetime(df_onshorewind.index)
        CFw_Spain = df_onshorewind['ESP'][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in n.snapshots]]
        
        df_solar = pd.read_csv('data_extra/pv_optimal.csv', sep=';', index_col=0)
        df_solar.index = pd.to_datetime(df_solar.index)
        CFs_Spain = df_solar['ESP'][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in n.snapshots]]

        n.generators_t.p_max_pu['onshorewind'] = CFw_Spain
        n.generators_t.p_max_pu['solar'] = CFs_Spain

    elif n.name == "California":
        df_cal_onshorewind = pd.read_csv('data_extra/CaliforniaTimeSeries.csv', index_col=0)
        df_cal_onshorewind.index = pd.to_datetime(df_cal_onshorewind.index)
        CFw_CA = df_cal_onshorewind['onwind'][[hour.strftime("%Y-%m-%dT%H:%M:%S") for hour in n.snapshots]]

        df_cal_solar = pd.read_csv('data_extra/CaliforniaTimeSeries.csv', index_col=0)
        df_cal_solar.index = pd.to_datetime(df_cal_solar.index)
        CFs_CA = df_cal_solar['solar'][[hour.strftime("%Y-%m-%dT%H:%M:%S") for hour in n.snapshots]]

        n.generators_t.p_max_pu['onshorewind'] = CFw_CA
        n.generators_t.p_max_pu['solar'] = CFs_CA

    elif n.name == "Colorado":
        df_co_onshorewind = pd.read_csv('data_extra/ColoradoTimeSeries.csv', index_col=0)
        df_co_onshorewind.index = pd.to_datetime(df_co_onshorewind.index)
        CFw_CO = df_co_onshorewind['onwind'][[hour.strftime("%Y-%m-%dT%H:%M:%S") for hour in n.snapshots]]
    
        df_co_solar = pd.read_csv('data_extra/ColoradoTimeSeries.csv', index_col=0)
        df_co_solar.index = pd.to_datetime(df_co_solar.index)
        CFs_CO = df_co_solar['solar'][[hour.strftime("%Y-%m-%dT%H:%M:%S") for hour in n.snapshots]]

        n.generators_t.p_max_pu['onshorewind'] = CFw_CO
        n.generators_t.p_max_pu['solar'] = CFs_CO


    return n


def add_storage(n):
        
    #to really get at the basics, we will only use battery storage, and not hydrogen 19/12

    n.add("Link",
         "battery charger",
         bus0 = "electricity bus",
         bus1 = "battery",
         carrier = "battery charger",
         efficiency = 0.95**0.5,
         p_nom_extendable = True,
         capital_cost = annual_cost("battery inverter"))
    
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
         capital_cost = annual_cost("battery storage"))

    return n





def add_global_constraint(n):
    '''This function is not actually needed because we will only include wind and solar generators'''
    co2_limit=1 #tonCO2 #global as in system global
    n.add("GlobalConstraint",
        "co2_limit",
        type="primary_energy",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=co2_limit)
    return n

 