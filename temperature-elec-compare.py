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



#plot_ED_and_CF_data_all()



# %%
########################################################################################
#################### SECTION ABOUT FIGURES 2 and 3 in the PAPER ########################


'''This section of code covers how Figures 2 and 3 are made.

'''
def get_electricity_data(timeframe):


    ES_elec = df_elec['ESP']
    DK_elec = df_elec['DNK']
    CO_elec = df_co_elec['demand_mwh']
    CA_elec = df_cal_elec['demand_mwh']
    
    
    if timeframe == "weekly":
        ESDayElec = ES_elec.rolling(168).mean()
        ESDayElec = ESDayElec.iloc[::168].shift(-1)[:-1]
        DKDayElec = DK_elec.rolling(168).mean()
        DKDayElec = DKDayElec.iloc[::168].shift(-1)[:-1]
        CODayElec = CO_elec.rolling(168).mean()
        CODayElec = CODayElec.iloc[::168].shift(-1)[:-1]
        CADayElec = CA_elec.rolling(168).mean()
        CADayElec = CADayElec.iloc[::168].shift(-1)[:-1]
    elif timeframe == "daily":
        ESDayElec = ES_elec.rolling(24).mean()
        ESDayElec = ESDayElec.iloc[::24].shift(-1)[:-1]
        DKDayElec = DK_elec.rolling(24).mean()
        DKDayElec = DKDayElec.iloc[::24].shift(-1)[:-1]
        CODayElec = CO_elec.rolling(24).mean()
        CODayElec = CODayElec.iloc[::24].shift(-1)[:-1]
        CADayElec = CA_elec.rolling(24).mean()
        CADayElec = CADayElec.iloc[::24].shift(-1)[:-1]

    else:
        ESDayElec = ES_elec
        DKDayElec = DK_elec
        CODayElec = CO_elec
        CADayElec = CA_elec
    
    ESDayElec.columns = ["SpainElec"]
    DKDayElec.columns = ["DenmarkElec"]
    CODayElec.columns = ["ColoradoElec"]
    CADayElec.columns = ["CaliforniaElec"]

    #print (DKDayElec.head())

    CODayElec.index = pd.to_datetime(CODayElec.index)
    CADayElec.index = pd.to_datetime(CADayElec.index)

    return ESDayElec, DKDayElec, CODayElec, CADayElec

#get_electricity_data("weekly")

def get_heat_demand_data(timeframe):
    df_heat = pd.read_csv('data/heat_demand.csv', sep=';', index_col=0)# in MWh
    df_heat.index = pd.to_datetime(df_heat.index) #change index to datatime

    ES_heat = df_heat['ESP']
    DK_heat = df_heat['DNK']

    CA_heat = df_cal_elec["heating_demand"]
    CO_heat = df_co_elec["heating_demand"]



    if timeframe == "weekly":
        ESDayheat = ES_heat.rolling(168).mean()
        ESDayheat = ESDayheat.iloc[::168].shift(-1)[:-1]
        DKDayheat = DK_heat.rolling(168).mean()
        DKDayheat = DKDayheat.iloc[::168].shift(-1)[:-1]
        CADay_heat = CA_heat.rolling(168).mean()
        CADay_heat = CA_heat.iloc[::168].shift(-1)[:-1]
        CODay_heat = CO_heat.rolling(168).mean()
        CODay_heat = CO_heat.iloc[::168].shift(-1)[:-1]

    elif timeframe == "daily":
        ESDayheat = ES_heat.rolling(24).mean()
        ESDayheat = ESDayheat.iloc[::24].shift(-1)[:-1]
        DKDayheat = DK_heat.rolling(24).mean()
        DKDayheat = DKDayheat.iloc[::24].shift(-1)[:-1]
        CADay_heat = CA_heat.rolling(24).mean()
        CADay_heat = CA_heat.iloc[::24].shift(-1)[:-1]
        CODay_heat = CO_heat.rolling(24).mean()
        CODay_heat = CO_heat.iloc[::24].shift(-1)[:-1]

    else:
        ESDayheat = ES_heat
        DKDayheat = DK_heat
        CADay_heat = CA_heat
        CODay_heat = CO_heat


    return ESDayheat, DKDayheat, CODay_heat, CADay_heat


def heat_to_elec():
    '''In this function, we want to add 3x of the heating demand to elec.
    We are assuming a world of 100% electricity.'''
    
    #Here, we unpack the elec and heat data from get_electricity_data and 
    # get_heat_demand_data
    elec_data = get_electricity_data("weekly")
    heat_data = get_heat_demand_data("weekly")

    ESP_elec = elec_data[0]
    DNK_elec = elec_data[1]
    CO_elec = elec_data[2]
    CA_elec = elec_data[3]

    ESP_heat = heat_data[0]
    DNK_heat = heat_data[1]
    CO_heat = heat_data[2]
    CA_heat = heat_data[3]

    new_elec = pd.DataFrame()
    states_elec = pd.DataFrame()


    new_elec["ESP_demand"] = ESP_elec + ESP_heat/3
    new_elec["DNK_demand"] = DNK_elec + DNK_heat/3
    states_elec["CA_demand"] = CA_elec + CA_heat/3
    states_elec["CO_demand"] = CO_elec + CO_heat/3


    return new_elec["ESP_demand"], new_elec["DNK_demand"], states_elec["CO_demand"], states_elec["CA_demand"]

    #Here, we return the new dataframe
#heat_to_elec()

#%%
def gw_elec_Spain_t(degree_change, slope_factor, yseries, ax):
    '''This considers a universal degree change across all days. 
    
    We are modifying the functions such that there is a more accurate reflection of what we
    are actually testing in the model--the presence or absence of heating demand. Luckily
    we already have heating demand
    
    Note: we are using "degree_change", which is outdated, given we do no longer use it as global warming, as
    an indicator of whether or not we want to include the 
    
    As of 23.1.23, we are no longer considering the global warming'''
    df = pd.DataFrame()
    if yseries == "elec":
        y = get_electricity_data("weekly")[0]
    else:
        y = heat_to_elec()[0]
        
    x = get_temp_data("weekly")[0]
    y = y/1000/47.3
    df["x"] = x
    df["y"] = y
    #print(df)

    total_elec_demand = round(df["y"].sum())

    #fig, ax = plt.subplots()

    ax.scatter(df["x"], df["y"], s = 15, color = "C0", label = "original")
    

    y = heat_to_elec()[0]
    y = y/1000/47.3
    df["y"] = y
    if degree_change == 2:
        ax.scatter(df["x"], df["y"], s = 15, marker = "^", color = "C3", label = "with heating demand")
    ax.set_title("Spain")
    #ax.set_ylabel("Electricity demand (MWh)")

    #plt.savefig(f"images/GWESP_incr{degree_change}_slope{slope_factor}")
    #plt.close(fig)
    return ax

def gw_elec_Colorado_t(degree_change, slope_factor, yseries, ax):
    '''As it stands, we do not '''
    df = pd.DataFrame()
    if yseries == "elec":
        y = get_electricity_data("weekly")[2]
    else:
        y = heat_to_elec()[2]
        
    x = get_temp_data("weekly")[2]
    y = y/1000/5.78
    df["x"] = x
    df["y"] = y
    #print(df)

    total_elec_demand = round(df["y"].sum())

    #fig, ax = plt.subplots()

    ax.scatter(df["x"], df["y"], s = 15, color = "C0", label = "original")
    y = heat_to_elec()[2]
    y = y/1000/5.78
    df["y"] = y
    if degree_change == 2:
        ax.scatter(df["x"], df["y"], s = 15, marker = "^", color = "C3", label = "with heating demand")

    ax.set_title("Colorado")
    ax.set_ylabel("Electricity demand (GWh)")

    return ax




def gw_elec_California_t(degree_change, slope_factor, yseries, ax):
    '''Here, I am trying to model what would happen to electricity demand in California if
    the temperature increases uniformly by x degrees due to global warming
    
    For california, we assume that the electricity demand would be constant with change in 
    temperature until it reaches a threshold temperature (15.79 degrees). Then, there is
    a linear increase
    
    Right now, I want to make a graph which shows no heating demand added
    '''
    df = pd.DataFrame()
    
    x = get_temp_data("weekly")[3]
    if yseries == "elec":
        y = get_electricity_data("weekly")[3]
    else:
        y = heat_to_elec()[3]
   

    y = y/1000/39.5
    df["x"] = x
    df["y"] = y


    total_elec_demand = round(df["y"].sum())


    ax.scatter(df["x"], df["y"], s = 15, color = "C0", label = "original")

    y = heat_to_elec()[3]

    y = y/1000/39.5

    df["y2"] = y

    if degree_change == 2:
        ax.scatter(df["x"], df["y2"], s = 15, marker = "^", color = "C3", label = "with heating demand")
    #ax.legend()
    ax.set_title("California")
    #ax.set_ylabel("Electricity demand (MWh)")
    #fig.subplots_adjust(bottom=0.2)
    #print(df['y'].sum())
    #plt.savefig(f"images/GWCali_incr{degree_change}_slope{slope_factor}")
    #plt.close(fig)
    return ax



def gw_elec_Denmark_t(degree_change, yseries, ax):
    '''Here, I am trying to model what would happen to electricity demand in Denmark if
    the temperature increases uniformly by x degrees due to global warming
    
    For Denmark we assume that the electricity demand would be constant with change in 
    temperature until it reaches a threshold temperature (15.79 degrees). Then, there is
    a linear increase'''
    df = pd.DataFrame()
    


    x = get_temp_data("weekly")[1]
    if yseries == "elec":
        y = get_electricity_data("weekly")[1]
    else:
        y = heat_to_elec()[1]

    y = y/1000/5.86
    df["x"] = x
    df["y"] = y

    ax.scatter(df["x"], df["y"], s = 15, color = "C0", label = "Historical")

    y = heat_to_elec()[1]
    y = y/1000/5.86
    df["y"] = y


    if degree_change == 2:
        ax.scatter(df["x"], df["y"], s = 15, marker = "^", color = "C3", label = "with heating demand")

    ax.set_title("Denmark")
    ax.set_ylabel("Electricity demand (GWh)")
    


    return ax

def gw_elec_all_mod(yseries):
    #yseries can be 'elec' or "w_heat"
    plt.rcParams.update({'font.size': 14})
    fig,ax = plt.subplots(2,2,figsize = (7.5,6),sharex=True,sharey='row')
    ax = ax.flatten()




    gw_elec_Denmark_t(2, yseries, ax[0])
    gw_elec_Spain_t(2,1, yseries, ax[1])
    gw_elec_Colorado_t(2,1, yseries, ax[2])
    gw_elec_California_t(2,1, yseries, ax[3])
 

    # ax[1].set_ylabel("")
    # ax[3].set_ylabel("")
    # ax[2].set_title("")
    # ax[3].set_title("")


    fig.supylabel (r"$\bf{Demand\:\:per\:\:capita\:\:(kWh)}$",fontweight="bold",fontsize=14, y = 0.55)
    fig.supxlabel (r"$\bf{Temperature\:\:(˚C)}$",fontweight="bold",fontsize=14, y = 0.11)
    for ax in plt.gcf().get_axes():
        ax.set_xlim(-10, 30)
        ax.set_ylim(0.5, 4)
        ax.set_ylabel("")

    



    #fig2.set_size_inches(9, 7) #For some reason this is best for the pdf
    fig.patch.set_facecolor("white")
    #fig2.suptitle("Increase of 4 degrees")

    lines1, labels1 = ax.get_legend_handles_labels()

    fig.legend(lines1, labels1, bbox_to_anchor=(0.82, 0.1), ncol=2)

    #fig.text(0.93, 0.13, "˚C")
    #fig2.text(0.93, 0.53, "˚C", fontsize = 12)
    fig.suptitle(r"$\bf{Electricity\:Demand\:Sensitivity\:To\:Temperature$", fontsize = 14)
    fig.tight_layout(rect = [0.02, 0.05, 1, 1])

    #plt.tight_layout()
    #fig2.savefig("Images/Paper/elecdem_temp_all.pdf")

    fig.savefig("Images/Paper/elecdem_plusheat_revision.png", dpi = 500)
    
    
    plt.show()

gw_elec_all_mod("elec")
# %%


