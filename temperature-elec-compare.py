#%%
from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from numpy.polynomial.polynomial import polyfit
# import seaborn as sns
# import statsmodels.api as sm
import matplotlib.dates as mdates
from decimal import Decimal
import matplotlib.gridspec as gridspec
import pypsa
from matplotlib.ticker import AutoLocator
#from matplotlib import gridspec

#%%
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

#print (df_cal_elec)
df_cal_elec["HDD"] = heatCA["HDD"]
df_cal_elec["heating_demand"] = df_cal_elec.apply(lambda row: 1715 * row["HDD"] + 6356, axis = 1)# 1715 is California's G factor, MWh/HDD. 6356 is the constant, that we get from water heating
df_cal_elec["adjust_elec_demand"] =  df_cal_elec["demand_mwh"] + 1/3 * df_cal_elec["heating_demand"]

##print (df_cal_elec)
df_co_elec["HDD"] = heatCO["HDD"]
df_co_elec["heating_demand"]= df_co_elec.apply(lambda row: 1782 * row["HDD"] + 6472, axis = 1)
df_co_elec["adjust_elec_demand"] =  df_co_elec["demand_mwh"] + 1/3 * df_co_elec["heating_demand"]



def get_temp_data(timeframe):
    
    
    ESData= pd.read_csv('data/TemperatureData/ninja_weather_country_ES_merra-2_population_weighted.csv', skiprows = 2, index_col=0)
    ESData.index = pd.to_datetime(ESData.index)
    DKData = pd.read_csv('data/TemperatureData/ninja_weather_country_DK_merra-2_population_weighted.csv', skiprows = 2, index_col=0)
    DKData.index = pd.to_datetime(DKData.index)
    COData = pd.read_csv('data/TemperatureData/ninja_weather_country_US.CO_merra-2_population_weighted.csv', skiprows = 2, index_col=0)
    COData.index = pd.to_datetime(COData.index)
    CAData = pd.read_csv('data/TemperatureData/ninja_weather_country_US.CA_merra-2_population_weighted.csv', skiprows = 2, index_col=0)
    CAData.index = pd.to_datetime(CAData.index)


    
    hours_in_2015 = pd.date_range('2015-01-01T00:00Z','2015-12-31T23:00Z', freq='H')

    ESTemp = ESData['temperature']['2015-01-01 00:00:00':'2015-12-31 23:00:00']
    DKTemp = DKData['temperature']['2015-01-01 00:00:00':'2015-12-31 23:00:00']
    COTemp = COData[ 'temperature']['2011-01-01 00:00:00':'2011-12-31 23:00:00']
    CATemp = CAData['temperature']['2011-01-01 00:00:00':'2011-12-31 23:00:00']

    ESTemp.index = hours_in_2015
    DKTemp.index = hours_in_2015


    
    if timeframe == "weekly":
        ESDayTemp = ESTemp.rolling(168).mean() #Daily average. Creates first row of NaN and then makes daily average such that the date 1/1 is labeled 1/2
        ESDayTemp = ESDayTemp.iloc[::168].shift(-1)[:-1] #This shifts our values up by one and then drops the last row
        DKDayTemp = DKTemp.rolling(168).mean() #Daily average. Creates first row of NaN and then makes daily average such that the date 1/1 is labeled 1/2
        DKDayTemp = DKDayTemp.iloc[::168].shift(-1)[:-1] #This shifts our values up by one and then drops the last row
        CODayTemp = COTemp.rolling(168).mean() #Daily average. Creates first row of NaN and then makes daily average such that the date 1/1 is labeled 1/2
        CODayTemp = CODayTemp.iloc[::168].shift(-1)[:-1] #This shifts our values up by one and then drops the last row
        CADayTemp = CATemp.rolling(168).mean() #Daily average. Creates first row of NaN and then makes daily average such that the date 1/1 is labeled 1/2
        CADayTemp = CADayTemp.iloc[::168].shift(-1)[:-1] #This shifts our values up by one and then drops the last row
    elif timeframe == "daily":
        ESDayTemp = ESTemp.rolling(24).mean() #Daily average. Creates first row of NaN and then makes daily average such that the date 1/1 is labeled 1/2
        ESDayTemp = ESDayTemp.iloc[::24].shift(-1)[:-1] #This shifts our values up by one and then drops the last row
        DKDayTemp = DKTemp.rolling(24).mean() #Daily average. Creates first row of NaN and then makes daily average such that the date 1/1 is labeled 1/2
        DKDayTemp = DKDayTemp.iloc[::24].shift(-1)[:-1] #This shifts our values up by one and then drops the last row
        CODayTemp = COTemp.rolling(24).mean() #Daily average. Creates first row of NaN and then makes daily average such that the date 1/1 is labeled 1/2
        CODayTemp = CODayTemp.iloc[::24].shift(-1)[:-1] #This shifts our values up by one and then drops the last row
        CADayTemp = CATemp.rolling(24).mean() #Daily average. Creates first row of NaN and then makes daily average such that the date 1/1 is labeled 1/2
        CADayTemp = CADayTemp.iloc[::24].shift(-1)[:-1] #This shifts our values up by one and then drops the last row
    else:
        ESDayTemp = ESTemp
        DKDayTemp = DKTemp
        CODayTemp = COTemp
        CADayTemp = CATemp
        
    
    return ESDayTemp, DKDayTemp, CODayTemp, CADayTemp

def plot_temp_data(temp_df):
    temp_df.plot()
    plt.show()

def plot_all_temp():
    for each in get_temp_data():
        plot_temp_data(each)




'''This section of equations considers what happens if you take advantage of the relationship between electricity demand and temperature
to see what happens if you are to increase the temperature by x degrees. In addition, one can '''

#%%
def plot_ED_and_CF_data_all():
    '''This function plots the electricity demand for one country on one axis and the capacity factors on other axes (sharing a y axis).
    Modify it to change which country (make sure CA and CO use "Time in 2011")
    
    we can plot either just elec, or elec+heat, using temp_to_elec
    
    This serves as figure 1 '''
    plt.rcdefaults()
    plt.rcParams.update({'font.size': 12})

    EUdf = pd.read_csv("data/EUcfs.csv", index_col=0, parse_dates = True)
    EUdf = EUdf.rolling(2).mean()
    EUdf = EUdf.iloc[::2].shift(-1)[:-1]

    USdf = pd.read_csv("data/CA_CO_modelenergy.csv", index_col = 0, parse_dates=True)
    USdf = USdf.rolling(2).mean()
    USdf = USdf.iloc[::2].shift(-1)[:-1]

    elecdnk = EUdf["DNKdem"]
    elecdnk = elecdnk/elecdnk.mean()
    solardnk = EUdf["DNKsol"]
    solardnk = solardnk/solardnk.mean()
    winddnk = EUdf['DNKwind']
    winddnk = winddnk/winddnk.mean()


    
    elecesp = EUdf["ESPdem"]
    elecesp = elecesp/elecesp.mean()
    solaresp = EUdf["ESPsol"]
    solaresp = solaresp/solaresp.mean()
    windesp = EUdf['ESPwind']
    windesp = windesp/windesp.mean()


    elecCo = USdf["COdem"].copy()
    elecCo = elecCo/elecCo.mean()
    solarCo = USdf["COsol"].copy()
    solarCo = solarCo/solarCo.mean()
    windCo = USdf['COwind'].copy()
    windCo = windCo/windCo.mean()



    elecCA = USdf["CAdem"]
    elecCA = elecCA/elecCA.mean()
    solarCA = USdf["CAsol"]
    solarCA = solarCA/solarCA.mean()
    windCA = USdf['CAwind']
    windCA = windCA/windCA.mean()

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(6.4, 8)


    '''Denmark'''
    
    axdnk1 = axs[0, 0]
    # axdnk2=axdnk1.twinx()
    # axdnk3 = axdnk1.twinx()

    axdnk1.plot(elecdnk, 'k-', label = "Electricity demand")
    axdnk1.plot(solardnk, 'C1-',  label = "Solar CF")
    axdnk1.plot(winddnk, 'C0-',  label = "Wind CF")

    axdnk1.set_title("Denmark")

    axdnk1.yaxis.label.set_color('k')

    axdnk1.text(0.83, 0.93, "0.29", color = 'C0', transform = axdnk1.transAxes)
    axdnk1.text(0.63, 0.93, "0.11", color = 'C1', transform = axdnk1.transAxes)

    dnkticks = np.array([0, 1, 2])
    axdnk1.set_yticks(dnkticks)



    '''Spain'''
    
    axdnk1 = axs[0, 1]
 

    axdnk1.plot(elecesp, 'k-', label = "Electricity demand")
    axdnk1.plot(solaresp, 'C1-',  label = "Solar CF")
    axdnk1.plot(windesp, 'C0-',  label = "Wind CF")

    axdnk1.set_title("Spain")

    
    axdnk1.text(0.83, 0.93, "0.23", color = 'C0', transform = axdnk1.transAxes)
    axdnk1.text(0.63, 0.93, "0.17", color = 'C1', transform = axdnk1.transAxes)


    axdnk1.yaxis.label.set_color('k')


    espticks = np.array([0, 1, 2])

    axdnk1.set_yticks(espticks)





    '''Colorado'''
    
    axdnk1 = axs[1, 0]


    axdnk1.plot(elecCo, 'k-', label = "Electricity demand")
    axdnk1.plot(solarCo, 'C1-',  label = "Solar CF")
    axdnk1.plot(windCo, 'C0-',  label = "Wind CF")

    axdnk1.set_title("Colorado")

    axdnk1.text(0.83, 0.93, "0.31", color = 'C0', transform = axdnk1.transAxes)
    axdnk1.text(0.63, 0.93, "0.20", color = 'C1', transform = axdnk1.transAxes)


    axdnk1.yaxis.label.set_color('k')


    dnkticks = np.array([0.5, 1, 1.5])
    axdnk1.set_yticks(dnkticks)



    '''California'''
    
    axdnk1 = axs[1, 1]


    axdnk1.plot(elecCA, 'k-', label = "Electricity demand")
    axdnk1.plot(solarCA, 'C1-',  label = "Solar CF")
    axdnk1.plot(windCA, 'C0-',  label = "Wind CF")

    axdnk1.set_title("California")

    axdnk1.text(0.83, 0.93, "0.18", color = 'C0', transform = axdnk1.transAxes)
    axdnk1.text(0.63, 0.93, "0.20", color = 'C1', transform = axdnk1.transAxes)


    axdnk1.yaxis.label.set_color('k')

    dnkticks = np.array([0.5, 1, 1.5])
    axdnk1.set_yticks(dnkticks)


    # axdnk1.xaxis.set_major_locator(AutoLocator()) 
    for ax in axs.flat[:2]:



        ax.set_xticklabels([])
        # ax.yaxis.labelpad = 30
    for ax in axs.flat[2:]:
        fmt = mdates.DateFormatter("%b")
        ax.xaxis.set_major_formatter(fmt)


    fig.supylabel("Capacity factors normalized to average")

    lines1, labels1 = axdnk1.get_legend_handles_labels()

    fig.legend(lines1, labels1, bbox_to_anchor=(0.85, 0.06), fontsize = 12, ncol=3)

    plt.subplots_adjust(hspace=0.1)

    plt.savefig("images/Paper/Figure1_EDandCF12Jan_2weeksavg.png", dpi = 500)
    plt.show()



plot_ED_and_CF_data_all()

# %%

# %%
#Question from reviewer #3--why is the capacity factor data 
n = pypsa.Network()
path = "NetCDF/CA/solarcost_elec_27_Sept/14565.766155244572solar_cost.nc"
n.import_from_netcdf(path)
powergen = n.generators_t.p
powergen = powergen['solar']


CAmonthpower = powergen.rolling(672).mean()#4 week rolling average
CAmonthpower = CAmonthpower.iloc[::672].shift(-1)[:-1]

opt_cap = n.generators.p_nom_opt['solar']
capfacsolar = n.generators_t.p_max_pu['solar']
capfacwind = n.generators_t.p_max_pu['onshorewind']

maxgen = opt_cap * capfacs
maxgen = maxgen.sum()

totgen = n.generators_t.p['solar'].sum()

totcapfacsolar = capfacsolar.sum()/8760
totcapfacwind = capfacwind.sum()/8760


print(totcapfacwind)
print(totcapfacsolar)

fig, ax = plt.subplots()

ax.plot(CAmonthpower)

plt.savefig("")
plt.show()
# %%
