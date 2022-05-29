from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from numpy.polynomial.polynomial import polyfit
# import seaborn as sns
import statsmodels.api as sm
import matplotlib.dates as mdates
from decimal import Decimal
#from matplotlib import gridspec


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


def get_solar_data(timeframe):
    '''This function fetches solar data capacity factors for all regions. It returns a tuple of dataframes'''
    df_solar = pd.read_csv('data_extra/pv_optimal.csv', sep=';', index_col=0)
    df_solar.index = pd.to_datetime(df_solar.index)

    df_cal_solar = pd.read_csv('data_extra/CaliforniaTimeSeries.csv', index_col=0)
    df_cal_solar.index = pd.to_datetime(df_cal_solar.index)

    df_co_solar = pd.read_csv('data_extra/ColoradoTimeSeries.csv', index_col=0)
    df_co_solar.index = pd.to_datetime(df_co_solar.index)

    ESP_solar = df_solar["ESP"]['2015-01-01 00:00:00':'2015-12-31 23:00:00']
    DNK_solar = df_solar["DNK"]['2015-01-01 00:00:00':'2015-12-31 23:00:00']
    CO_solar = df_co_solar['solar']['2011-01-01 00:00:00':'2011-12-31 23:00:00']
    CA_solar = df_cal_solar['solar']['2011-01-01 00:00:00':'2011-12-31 23:00:00']
    
    if timeframe == "weekly":
        ESP_solar = ESP_solar.rolling(168).mean()
        ESP_solar = ESP_solar.iloc[::168].shift(-1)[:-1]
        DNK_solar = DNK_solar.rolling(168).mean()
        DNK_solar = DNK_solar.iloc[::168].shift(-1)[:-1]
        CO_solar = CO_solar.rolling(168).mean()
        CO_solar = CO_solar.iloc[::168].shift(-1)[:-1]
        CA_solar = CA_solar.rolling(168).mean()
        CA_solar  = CA_solar .iloc[::168].shift(-1)[:-1]
    elif timeframe == "daily":
        ESP_solar = ESP_solar.rolling(24).mean()
        ESP_solar = ESP_solar.iloc[::24].shift(-1)[:-1]
        DNK_solar = DNK_solar.rolling(24).mean()
        DNK_solar = DNK_solar.iloc[::24].shift(-1)[:-1]
        CO_solar = CO_solar.rolling(24).mean()
        CO_solar = CO_solar.iloc[::24].shift(-1)[:-1]
        CA_solar = CA_solar.rolling(24).mean()
        CA_solar  = CA_solar .iloc[::24].shift(-1)[:-1]

    return ESP_solar, DNK_solar, CO_solar, CA_solar

def get_wind_data(timeframe):
    '''This function fetches wind data capacity factors for all regions. It returns a tuple of dataframes'''
    df_onshorewind = pd.read_csv('data_extra/onshore_wind_1979-2017.csv', sep=';', index_col=0)
    df_onshorewind.index = pd.to_datetime(df_onshorewind.index)

    df_cal_onshorewind = pd.read_csv('data_extra/CaliforniaTimeSeries.csv', index_col=0)
    df_cal_onshorewind.index = pd.to_datetime(df_cal_onshorewind.index)

    df_co_onshorewind = pd.read_csv('data_extra/ColoradoTimeSeries.csv', index_col=0)
    df_co_onshorewind.index = pd.to_datetime(df_co_onshorewind.index)

    ESP_wind = df_onshorewind['ESP']['2015-01-01 00:00:00':'2015-12-31 23:00:00']
    DNK_wind = df_onshorewind['DNK']['2015-01-01 00:00:00':'2015-12-31 23:00:00']
    CO_wind = df_cal_onshorewind['onwind']['2011-01-01 00:00:00':'2011-12-31 23:00:00']
    CA_wind = df_co_onshorewind['onwind']['2011-01-01 00:00:00':'2011-12-31 23:00:00']
    
    if timeframe == "weekly":
        ESP_wind = ESP_wind.rolling(168).mean()
        ESP_wind = ESP_wind.iloc[::168].shift(-1)[:-1]
        DNK_wind  = DNK_wind.rolling(168).mean()
        DNK_wind  = DNK_wind.iloc[::168].shift(-1)[:-1]
        CO_wind = CO_wind.rolling(168).mean()
        CO_wind = CO_wind.iloc[::168].shift(-1)[:-1]
        CA_wind = CA_wind.rolling(168).mean()
        CA_wind  = CA_wind.iloc[::168].shift(-1)[:-1]
    elif timeframe == "daily":
        ESP_wind = ESP_wind.rolling(24).mean()
        ESP_wind = ESP_wind.iloc[::24].shift(-1)[:-1]
        DNK_wind  = DNK_wind.rolling(24).mean()
        DNK_wind  = DNK_wind.iloc[::24].shift(-1)[:-1]
        CO_wind = CO_wind.rolling(24).mean()
        CO_wind = CO_wind.iloc[::24].shift(-1)[:-1]
        CA_wind = CA_wind.rolling(24).mean()
        CA_wind  = CA_wind.iloc[::24].shift(-1)[:-1]

    return ESP_wind, DNK_wind, CO_wind, CA_wind

#Select between "daily" or "weekly" timeframe averages
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
#get_heat_demand_data("weekly")





#In this function, I will plot electricity data and temperature data on two separate subplots

def plot_elec_and_temp(elec_df, temp_df):
    fig, axs = plt.subplots (2)
    axs[0].plot(elec_df)
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axs[0].xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    axs[1].plot(temp_df)
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axs[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.subplots_adjust(bottom=0.25)
    plt.xticks(rotation = 45)

    plt.show()

#In this function, I will plot electricity data and temperature data on the same subplot
def same_plot_elec_temp(elec_df, temp_df, year, country, timeframe):
        
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(elec_df, 'g-', label = "Electricity demand")
    ax2.plot(temp_df, 'b-', label = "Temperature")
    ax1.set_xlabel("Time in " + year)
    ax1.set_ylabel('Electricity demand (MWh)')
    ax2.set_ylabel("Temperature (C)")
    ax1.set_title(country +" "+timeframe + " Electricity Demand and Temperature")


    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax2.legend(lines1 + lines2, labels1 + labels2)
    fig.set_size_inches(12, 10)
    #plt.savefig("images/"+country+timeframe+"EDandT.png")
    plt.show()


#In this function, I will plot the first week of January
def Jan_plot_elec_temp(elec_df, temp_df, year, country):
    
    elec_selec = elec_df[0:168]
    temp_selec = temp_df[0:168]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(elec_selec, 'g-', label = "Electricity demand")
    ax2.plot(temp_selec, 'b-', label = "Temperature")
    ax1.set_xlabel("Time in " + year)
    ax1.set_ylabel('Electricity demand (MWh)')
    ax2.set_ylabel("Temperature (C)")
    ax1.set_title("A week of January in " + country + ", Electricity Demand and Temperature")


    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax2.legend(lines1 + lines2, labels1 + labels2)
    fig.set_size_inches(12, 10)
    plt.savefig("images/Jan"+country+"EDandT.png")
    plt.show()

def July_plot_elec_temp(elec_df, temp_df, year, country):
    
    elec_selec = elec_df[4344:4512]
    temp_selec = temp_df[4344:4512]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(elec_selec, 'g-', label = "Electricity demand")
    ax2.plot(temp_selec, 'b-', label = "Temperature")
    ax1.set_xlabel("Time in " + year)
    ax1.set_ylabel('Electricity demand (MWh)')
    ax2.set_ylabel("Temperature (C)")
    ax1.set_title("A week of July in " + country + ", Electricity Demand and Temperature")


    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax2.legend(lines1 + lines2, labels1 + labels2)
    fig.set_size_inches(12, 10)
    plt.savefig("images/July"+country+"EDandT.png")
    plt.show()





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


    #Here, we make a new dataframe and add elec and 1/3 heat to it

    new_elec = pd.DataFrame()
    states_elec = pd.DataFrame()

    new_elec["ESP_demand"] = ESP_elec + ESP_heat/3
    new_elec["DNK_demand"] = DNK_elec + DNK_heat/3
    states_elec["CA_demand"] = CA_elec + CA_heat/3
    states_elec["CO_demand"] = CO_elec + CO_heat/3
    #print(new_elec.head())

    return new_elec["ESP_demand"], new_elec["DNK_demand"], states_elec["CO_demand"], states_elec["CA_demand"]

    #Here, we return the new dataframe
#heat_to_elec()

def elec_vs_temp_Spain():

    '''This makes a plot of the electricity demand (or heating demand or elec+heating demand) vs temperature
    I need to do some more work playing with annotate vs '''
    # x = temp
    # y = elec
   
    #y = get_electricity_data("weekly")[0]
    #y = get_heat_demand_data("weekly")[0]
    y = heat_to_elec()[0]
    x = get_temp_data("weekly")[0]

    

    fig, ax = plt.subplots()
    fig.subplots_adjust(left = 0.2)
    ax.scatter(x, y)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Electricity demand (MWh)")
    ax.set_title ("Weekly electricity (incl heating) demand vs temperature in Spain")

    mod_x  = x[x>22]
    mod_y = y[x>22]

    corr = np.corrcoef(mod_x, mod_y).round(decimals = 3)[0, 1]

    theta = np.polyfit(mod_x, mod_y, 1).round(decimals = 3)
    ax.axhline(y=27000, xmin = 0.5, xmax = 0.75, color = "C2")
    #settings 
    ax.annotate(f'{theta[0]}x + {theta[1]}', xy = (mod_x[4], mod_y[4]), xytext = (mod_x[4]-5 , mod_y[4] +1000))
    ax.annotate(f'correlation = {corr} ', xy = (mod_x[4], mod_y[4]), xytext = (mod_x[4]-5, mod_y[4]+ 3000))

    plt.plot(np.unique(mod_x), np.poly1d(np.polyfit(mod_x, mod_y, 1))(np.unique(mod_x)))


    new_x = x[x<16]
    new_y = y[x<16]
    corr2 = np.corrcoef(new_x, new_y).round(decimals = 3)[0, 1]

    theta2 = np.polyfit(new_x, new_y, 1).round(decimals = 3)
    ax.annotate(f'{theta2[0]}x + {theta2[1]}', xy = (new_x[4], new_y[4]), xytext = (new_x[4], new_y[4]))
    ax.annotate(f'correlation = {corr2} ', xy = (new_x[4], new_y[4]), xytext = (new_x[4], new_y[4]+ 2000))



    # print(deg_2_theta)
    plt.plot(np.unique(new_x), np.poly1d(np.polyfit(new_x, new_y, 1))(np.unique(new_x)))


    #plt.savefig("images/SpainED_plus_HDvsTw_eq")
    plt.show()
#elec_vs_temp_Spain()
def elec_vs_temp_Denmark():

    '''This makes a plot of the electricity demand (or heating demand or elec+heating demand) vs temperature
    for Denmark
    
    NB: you need to change which one you want'''
    # x = temp
    # y = elec
   
    y = heat_to_elec()["DNK_demand"]
    # y = get_electricity_data("weekly")[1]
    #y = get_heat_demand_data("weekly")[1]
    x = get_temp_data("weekly")[1]

    

    fig, ax = plt.subplots()
    
    fig.subplots_adjust(left = 0.2)
    ax.scatter(x, y)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Electricity plus heating demand (MWh)")
    ax.set_title ("Weekly electricity demand vs temperature in Denmark")

    mod_x  = x[x<16]
    mod_y = y[x<16]

    ax.axhline(y = 3900, xmin = 0.82, xmax = 0.95, color = "C1")
    
    corr = np.corrcoef(mod_x, mod_y).round(decimals = 3)[0, 1]

    theta = np.polyfit(mod_x, mod_y, 1).round(decimals = 3)

    #settings 
    ax.annotate(f'{theta[0]}x + {theta[1]}', xy = (mod_x[8], mod_y[8]), xytext = (mod_x[8] , mod_y[8]* 1.01 ))
    ax.annotate(f'correlation = {corr} ', xy = (mod_x[8], mod_y[8]), xytext = (mod_x[8] + 10, mod_y[8] * 1.05))


    plt.plot(np.unique(mod_x), np.poly1d(np.polyfit(mod_x, mod_y, 1))(np.unique(mod_x)))

    plt.savefig("images/DenmarkHDvsTw_eq")
    plt.show()
#elec_vs_temp_Denmark()

def elec_vs_temp_Colorado():
    x = get_temp_data("weekly")[2]
    y = heat_to_elec()[2]

    fig, ax = plt.subplots()
    
    fig.subplots_adjust(left = 0.2)
    ax.scatter(x, y)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Electricity (MWh)")
    ax.set_title ("Modified demand vs temp in Colorado")

    mod_x  = x[x>17]
    mod_y = y[x>17]

    
    
    corr = np.corrcoef(mod_x, mod_y).round(decimals = 3)[0, 1]

    theta = np.polyfit(mod_x, mod_y, 1).round(decimals = 3)
    #ax.axhline(y = 6600, xmin = 0.5, xmax = 0.7, color = "C2")
    #settings 
    ax.annotate(f'{theta[0]}x + {theta[1]}', xy = (mod_x[6], mod_y[6]), xytext = (mod_x[6]-6, mod_y[6]))
    ax.annotate(f'correlation = {corr} ', xy = (mod_x[6], mod_y[6]), xytext = (mod_x[6]-9, mod_y[6]-700))


    plt.plot(np.unique(mod_x), np.poly1d(np.polyfit(mod_x, mod_y, 1))(np.unique(mod_x)))

    new_x = x[x<15]
    new_y = y[x<15]
    corr2 = np.corrcoef(new_x, new_y).round(decimals = 3)[0, 1]

    theta2 = np.polyfit(new_x, new_y, 1).round(decimals = 3)
    ax.annotate(f'{theta2[0]}x + {theta2[1]}', xy = (new_x[4], new_y[4]), xytext = (new_x[4], new_y[4]))
    ax.annotate(f'correlation = {corr2} ', xy = (new_x[4], new_y[4]), xytext = (new_x[4], new_y[4]-700))

    plt.plot(np.unique(new_x), np.poly1d(np.polyfit(new_x, new_y, 1))(np.unique(new_x)))
    plt.savefig("images/COTotalDemandvsTempvar1")
    plt.show()
#elec_vs_temp_Colorado()

def elec_vs_temp_California():
    x = get_temp_data("weekly")[3]
    y = heat_to_elec()[3]

    fig, ax = plt.subplots()
    
    fig.subplots_adjust(left = 0.2)
    ax.scatter(x, y)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Electricity (MWh)")
    ax.set_title ("Modified demand vs temp in California")

    mod_x  = x[x>17]
    mod_y = y[x>17]

    
    
    corr = np.corrcoef(mod_x, mod_y).round(decimals = 3)[0, 1]

    theta = np.polyfit(mod_x, mod_y, 1).round(decimals = 3)

    #settings 
    #ax.axhline(y = 32500, xmin =.25, xmax = 0.45)
    ax.annotate(f'{theta[0]}x + {theta[1]}', xy = (mod_x[2], mod_y[2]), xytext = (mod_x[2]-5 , mod_y[2] ))
    ax.annotate(f'correlation = {corr} ', xy = (mod_x[2], mod_y[2]), xytext = (mod_x[2] -5, mod_y[2]-500))

    plt.plot(np.unique(mod_x), np.poly1d(np.polyfit(mod_x, mod_y, 1))(np.unique(mod_x)))
    mod_x = x[x<15]
    mod_y = y[x< 15]

    corr = np.corrcoef(mod_x, mod_y).round(decimals = 3)[0, 1]

    theta = np.polyfit(mod_x, mod_y, 1).round(decimals = 3)

    #settings 
    #ax.axhline(y = 32500, xmin =.25, xmax = 0.45)
    ax.annotate(f'{theta[0]}x + {theta[1]}', xy = (mod_x[2], mod_y[2]), xytext = (mod_x[2] , mod_y[2] ))
    ax.annotate(f'correlation = {corr} ', xy = (mod_x[2], mod_y[2]), xytext = (mod_x[2], mod_y[2]-500))

    plt.tight_layout()
    plt.plot(np.unique(mod_x), np.poly1d(np.polyfit(mod_x, mod_y, 1))(np.unique(mod_x)))

    plt.savefig("images/CaliTotalDemandvsTempvar1")
    plt.show()
#elec_vs_temp_California()

# elec_vs_temp_Denmark()
# elec_vs_temp_Spain()
# elec_vs_temp_Colorado()
# elec_vs_temp_California()

def plot_ED_and_CF_data():
    '''This function plots the electricity demand for one country on one axis and the capacity factors on other axes (sharing a y axis).
    Modify it to change which country (make sure CA and CO use "Time in 2011")
    
    we can plot either just elec, or elec+heat, using temp_to_elec
    
    This serves as figure 1 '''
    elec = get_electricity_data("weekly")[0]
    print(elec)

    #elec = temp_to_elec()["DNK_demand"]
    solar = get_solar_data("weekly")[0]
    wind = get_wind_data("weekly")[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2= fig.add_subplot(111, frame_on=False)
    ax3 = fig.add_subplot(111, sharey = ax2, frame_on=False)
    ax1.plot(elec, 'C0-', label = "Electricity demand")
    ax2.plot(solar, 'C1-', label = "Solar CF")
    ax3.plot(wind, 'C2-', label = "Wind CF")


    fmt = mdates.DateFormatter("%b")
    ax1.xaxis.set_major_formatter(fmt)
    ax1.set_xlabel("Time in 2015")
    ax1.set_ylabel('Electricity demand (MWh)')
    ax2.set_xticks([]) #We have two graphs sharing one axis, and without this we would be seeing double
    ax2.set_ylabel('Capacity factors')
    ax2.yaxis.set_label_position("right")


    ax2.yaxis.tick_right()
    ax3.set_xticks([])
    ax3.yaxis.set_visible(False)

    ax1.set_title("Spain Electricity Demand and Capacity factors")


    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()

    ax2.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3)
    fig.set_size_inches(6.4, 6)
    plt.savefig("images/EDandCFSpain")
    plt.show()

'''This section of equations considers what happens if you take advantage of the relationship between electricity demand and temperature
to see what happens if you are to increase the temperature by x degrees. In addition, one can '''

def plot_ED_and_CF_data_all():
    '''This function plots the electricity demand for one country on one axis and the capacity factors on other axes (sharing a y axis).
    Modify it to change which country (make sure CA and CO use "Time in 2011")
    
    we can plot either just elec, or elec+heat, using temp_to_elec
    
    This serves as figure 1 '''
    plt.rcdefaults()
    plt.rcParams.update({'font.size': 12})
    elecdnk = get_electricity_data("weekly")[1] #note: this returns a DATAFRAME so to to keep it as a dataframe do not use list comprehension
    elecdnk = elecdnk/elecdnk.mean()
    solardnk = get_solar_data("weekly")[1]
    solardnk = solardnk/solardnk.mean()
    winddnk = get_wind_data("weekly")[1]
    winddnk = winddnk/winddnk.mean()
    #elec = temp_to_elec()["DNK_demand"]
    
    elecesp = get_electricity_data("weekly")[0]
    elecesp = elecesp/elecesp.mean()
    solaresp = get_solar_data("weekly")[0]
    solaresp = solaresp/solaresp.mean()
    windesp = get_wind_data("weekly")[0]
    windesp = windesp/windesp.mean()

    elecCo = get_electricity_data("weekly")[2]
    elecCo = elecCo/elecCo.mean()
    #elecCo = [x/max(elecCo) for x in elecCo]
    solarCo = get_solar_data("weekly")[2]
    solarCo = solarCo/solarCo.mean()
    windCo = get_wind_data("weekly")[2]
    windCo = windCo/windCo.mean()

    elecCA = get_electricity_data("weekly")[3]
    elecCA = elecCA/elecCA.mean()

    #elecCA = [x/max(elecCA) for x in elecCA]
    solarCA = get_solar_data("weekly")[3]
    solarCA = solarCA/solarCA.mean()
    windCA = get_wind_data("weekly")[3]
    windCA = windCA/windCA.mean()

    fig = plt.figure()
    fig.set_figheight(5)

    #gs = gridspec.GridSpec(2, 1)
    axdnk1 = fig.add_subplot(411)

    axdnk2=fig.add_subplot(411, frame_on=False)

    axdnk3 = fig.add_subplot(411, sharey = axdnk2, frame_on=False)
    axdnk1.plot(elecdnk, 'k-', label = "Electricity demand")
    axdnk2.plot(solardnk, 'C1-',  label = "Solar CF")
    axdnk2.plot(winddnk, 'C0-',  label = "Wind CF")
    axdnk1.set_ylabel("Denmark")
    axdnk1.tick_params(direction = "in")

    axesp1 = fig.add_subplot(412)

    axesp2= fig.add_subplot(412, frame_on=False)

    axesp3 = fig.add_subplot(412, sharey = axesp2, frame_on=False)
    axesp1.plot(elecesp, 'k-', label = "Electricity demand")
    axesp2.plot(solaresp, 'C1-', label = "Solar CF")
    axesp3.plot(windesp, 'C0-', label = "Wind CF")
    axesp1.set_ylabel("Spain")
    axesp1.tick_params(direction = "in")


    axco1 = fig.add_subplot(413)

    axco2= fig.add_subplot(413, frame_on=False)

    axco3 = fig.add_subplot(413, sharey = axco2, frame_on=False)
    axco1.plot(elecCo, 'k-', label = "Electricity demand")
    axco2.plot(solarCo, 'C1-', label = "Solar CF")
    axco3.plot(windCo, 'C0-', label = "Wind CF")
    axco1.set_ylabel("Colorado")
    axco1.tick_params(direction = "in")

    axca1 = fig.add_subplot(414)

    axca2= fig.add_subplot(414, frame_on=False)

    axca3 = fig.add_subplot(414, sharey = axca2, frame_on=False)
    axca1.plot(elecCA, 'k-', label = "Electricity demand")
    axca2.plot(solarCA, 'C1-', label = "Solar CF")
    axca3.plot(windCA, 'C0-', label = "Wind CF")
    axca1.set_ylabel("California")
    axca1.tick_params(direction = "in")

    for ax in plt.gcf().get_axes():
        ax.set_ylim(0.01, 2.4)

    fmt = mdates.DateFormatter("%b")
 

    plt.setp(axdnk1.get_xticklabels(), visible=False)
    axdnk2.set_xticks([]) #We have two graphs sharing one axis, and without this we would be seeing double


    plt.setp(axesp1.get_xticklabels(), visible=False)
    axesp2.set_xticks([]) #We have two graphs sharing one axis, and without this we would be seeing double

 
    plt.setp(axco1.get_xticklabels(), visible=False)
    axco2.set_xticks([]) #We have two graphs sharing one axis, and without this we would be seeing double


    axca1.xaxis.set_major_formatter(fmt)
    axca2.set_xticks([]) #We have two graphs sharing one axis, and without this we would be seeing double

    axdnk2.yaxis.set_visible(False)
    axdnk3.set_xticks([])
    axdnk3.yaxis.set_visible(False)

    axesp2.yaxis.set_visible(False)
    axesp3.set_xticks([])
    axesp3.yaxis.set_visible(False)

    axco2.yaxis.set_visible(False)
    axco3.set_xticks([])
    axco3.yaxis.set_visible(False)

    axca2.yaxis.set_visible(False)
    axca3.set_xticks([])
    axca3.yaxis.set_visible(False)

    #axdnk1.set_title("Denmark Electricity Demand and Capacity factors")


    lines1, labels1 = axdnk1.get_legend_handles_labels()
    lines2, labels2 = axdnk2.get_legend_handles_labels()
    lines3, labels3 = axdnk3.get_legend_handles_labels()


    axca1.set_xticks(axca1.get_xticks()[:-1])


    fig.legend(lines1+lines2+lines3, labels1+labels2+labels3, bbox_to_anchor=(0.85, 0.06), fontsize = 10, ncol=3)
    fig.suptitle(r"$\bf{Seasonal\:Variation\:of\:Wind,\:Solar,\:and\:Electricity\:Demand}$", fontsize = 14)
    #fig.set_size_inches(6.4, 6)
    plt.subplots_adjust(hspace=0)
    plt.savefig("images/EDandCFALL_postervar")
    plt.show()
plot_ED_and_CF_data_all()

#plot_ED_and_CF_data()


degrees = [2, 4, 6]
slopes = [1, 2, 3]


degrees2 = [0]
degrees3 = [8]






###Used to be testing, now current functions####
def gw_elec_Spain_t(degree_change, slope_factor):
    '''This considers a universal degree change across all days. '''
    df = pd.DataFrame()
    y = heat_to_elec()[0]
 
    y = get_electricity_data("weekly")[0]
    x = get_temp_data("weekly")[0]
    y = y/1000
    df["x"] = x
    df["y"] = y
    #print(df)

    total_elec_demand = round(df["y"].sum())

    fig, ax = plt.subplots()

    ax.scatter(df["x"], df["y"], s = 15, color = "C0", label = "original")
    
    df["y"] = df.apply(lambda row: row["y"] + .8652 * degree_change if row["x"] > 22.267 #Add according to positive slope if starts in cooling region
    else row["y"] + .8652 * (row["x"] - 22.267 + degree_change) if row["x"]+ degree_change  >22.267 and row["x"] > 16
    else row["y"] if row["x"] > 16 
    else row["y"] - 1.356544 * (16- row["x"]) + 865.2 * (row["x"] + degree_change - 22.267) if row["x"] + degree_change > 16 and row["x"] + degree_change  > 22.267 
    else row["y"] - 1.356544 * (16- row["x"]) if row["x"] + degree_change > 16
    else row["y"] - 1.356544 * degree_change, axis = 1)

    #This applies 
    df["y"] = df.apply(lambda row: row["y"] + (row ["y"] - 30.000) * (slope_factor-1) if row["x"] > 22.267 and row["y"] > 30.000
    else row["y"], axis = 1)

    df["x"] = df.apply(lambda row: row["x"] + degree_change, axis = 1)
    #print(df)


    ax.axvline(16, color='black',ls='--', alpha = 0.5)
    ax.text(16, ax.get_ybound()[1]-1.500, "T_th", horizontalalignment = "center", color = "C3")
    
    ax.axvline(22.267, color='black',ls='--', alpha = 0.5)
    ax.text(22.267, ax.get_ybound()[1]-1.500, "T_th", horizontalalignment = "center", color = "C2")

    
    ax.scatter(df["x"], df["y"], s = 15, marker = "^", color = "C1", label = "with modification")
    ax.set_title("Spain")
    #ax.set_ylabel("Electricity demand (MWh)")

    #plt.savefig(f"images/GWESP_incr{degree_change}_slope{slope_factor}")
    plt.close(fig)
    return ax

def gw_elec_Colorado_t(degree_change, slope_factor):
    '''As it stands, we do not '''
    df = pd.DataFrame()
    y = get_electricity_data("weekly")[2]
    # y = heat_to_elec()[2]
    x = get_temp_data("weekly")[2]
    y = y/1000
    df["x"] = x
    df["y"] = y
    #print(df)

    total_elec_demand = round(df["y"].sum())

    fig, ax = plt.subplots()

    ax.scatter(df["x"], df["y"], s = 15, color = "C0", label = "original")

    df["y"] = df.apply(lambda row: row["y"] + .232 * degree_change if row["x"] > 15.56 #Add according to positive slope if starts in cooling region
    else row["y"] + .232 * (row["x"] - 15.56 + degree_change) if row["x"]+ degree_change  >15.56 and row["x"] > 7.32
    else row["y"] if row["x"] > 7.32
    else row["y"] - 0.097281* (7.32- row["x"]) + 232 * (row["x"] + degree_change - 15.56) if row["x"] + degree_change > 7.32 and row["x"] + degree_change  > 15.56
    else row["y"] - 0.097281 * (7.32- row["x"]) if row["x"] + degree_change > 7.32
    else row["y"] - 0.097281 * degree_change, axis = 1)

    df["y"] = df.apply(lambda row: row["y"] + (row ["y"] - 6.600) * (slope_factor-1) if row["x"] > 15.56 and row["y"] > 6.600
    else row["y"], axis = 1)


    df["x"] = df.apply(lambda row: row["x"] + degree_change, axis = 1)
    #print(df)
    ax.scatter(df["x"], df["y"], s = 15, marker = "^", color = "C1", label = "with modification")

    ax.axvline(7.32, color='black',ls='--', alpha = 0.5)
    ax.text(7.32, ax.get_ybound()[1]-0.500, "T_th", horizontalalignment = "center", color = "C3")


    ax.axvline(15.56, color='black',ls='--', alpha = 0.5)  
    ax.text(15.56, ax.get_ybound()[1]-0.500, "T_th", horizontalalignment = "center", color = "C2")

    #ax.legend()
    ax.set_title("Colorado")
    ax.set_ylabel("Electricity demand (GWh)")
    #plt.savefig(f"images/GWCO_incr{degree_change}_slope{slope_factor}")
    # plt.show()
    plt.close(fig)
    return ax


#gw_elec_Colorado_t(2, 2)
# for degree in degrees + degrees3:
#     gw_elec_Spain(degree, slopes[0])

# for degree in degrees + degrees2:
#     for slope in slopes[1:]:
#         gw_elec_Spain(degree, slope)

def gw_elec_California_t(degree_change, slope_factor):
    '''Here, I am trying to model what would happen to electricity demand in California if
    the temperature increases uniformly by x degrees due to global warming
    
    For california, we assume that the electricity demand would be constant with change in 
    temperature until it reaches a threshold temperature (15.79 degrees). Then, there is
    a linear increase'''
    df = pd.DataFrame()
    
    x = get_temp_data("weekly")[3]
    y = get_electricity_data("weekly")[3]
    y = y/1000
    df["x"] = x
    df["y"] = y

    total_elec_demand = round(df["y"].sum())

    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"], s = 15, color = "C0", label = "original")


    df["y"] = df.apply(lambda row: row ["y"] + 1.093304 * degree_change if row["x"] > 15.79 
    else row["y"] + 1.093304 * (row["x"] - 15.79 + degree_change) if row["x"]+ degree_change - 15.79 > 0 
    else row["y"], axis = 1)

    df["x"] = df.apply(lambda row: row["x"] + degree_change, axis = 1)
    
    #Use this line if you also want to include slope
    df["y"] = df.apply(lambda row: row["y"] + (row ["y"] - 32.500) * (slope_factor-1) if row["x"] > 15.79 and row["y"] > 32.500
    else row["y"], axis = 1)

    ax.scatter(df["x"], df["y"], s = 15, marker = "^", color = "C1", label = "with modification")

    ax.axvline(15.79, color='black',ls='--', alpha = 0.5)
    ax.text(15.79, ax.get_ybound()[1]-1.500, "T_th", horizontalalignment = "center", color = "C2")


    #ax.legend()
    ax.set_title("California")
    #ax.set_ylabel("Electricity demand (MWh)")
    fig.subplots_adjust(bottom=0.2)
    #print(df['y'].sum())
    #plt.savefig(f"images/GWCali_incr{degree_change}_slope{slope_factor}")
    plt.close(fig)
    return ax
    #if x+degree_change-15.79 is greater than 0, then add this value times 1093.394 to y

def gw_elec_Denmark_t(degree_change):
    '''Here, I am trying to model what would happen to electricity demand in Denmark if
    the temperature increases uniformly by x degrees due to global warming
    
    For Denmark we assume that the electricity demand would be constant with change in 
    temperature until it reaches a threshold temperature (15.79 degrees). Then, there is
    a linear increase'''
    df = pd.DataFrame()
    
    x = get_temp_data("weekly")[1]
    #y = get_electricity_data("weekly")[1]
    y = heat_to_elec()[1]
    y = y/1000
    df["x"] = x
    df["y"] = y

    #total_elec_demand = round(df["y"].sum())
    #print(df['y'].sum())
    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"], s = 15, color = "C0", label = "original")

    df["y"] = df.apply(lambda row: row ["y"] if row["x"] > 15.8
    else row["y"] - .273665 * (15.8-row["x"]) if row["x"]+ degree_change - 15.8 > 0 
    else row["y"] - .273665 * degree_change, axis = 1)
    #Like California, there are only three cases. Unlike california, the three cases are a bit different
    #flat to flat, heat to flat, heat to heat

    df["x"] = df.apply(lambda row: row["x"] + degree_change, axis = 1)
    
  
  
    ax.axvline(15.8, color='black',ls='--', alpha = 0.5)
    ax.text(15.8, ax.get_ybound()[1]-.500, "T_th", horizontalalignment = "center", color = "C3")

    ax.scatter(df["x"], df["y"], s = 15, marker = "^", color = "C1", label = "synthetic global warming (+2˚C) and additional cooling demand")

    ax.set_title("Denmark")
    ax.set_ylabel("Electricity demand (GWh)")
    


    plt.close(fig)
    #plt.savefig(f"images/GWDen_incr{degree_change}")
    #print(df['y'].sum())

    return ax




def gw_elec_all():
    
    plt.rcdefaults()
    fig2 = plt.figure()

    ax1  = gw_elec_Denmark_t(2)
    ax2 = gw_elec_Spain_t(2,2)
    ax3 = gw_elec_Colorado_t(2,2)
    ax4 = gw_elec_California_t(2,2)
 
    ax1.figure = fig2
    ax2.figure = fig2
    ax3.figure = fig2
    ax4.figure = fig2

    fig2.axes.append(ax1)
    fig2.add_axes(ax1)
    fig2.axes.append(ax2)
    fig2.add_axes(ax2)
    fig2.axes.append(ax3)
    fig2.add_axes(ax3)
    fig2.axes.append(ax4)
    fig2.add_axes(ax4)

    dummy = fig2.add_subplot(221)
    ax1.set_position(dummy.get_position())
    dummy.remove()
    axpos1 = ax1.get_position()

    ax1.set_position([axpos1.x0, axpos1.y0+0.04, axpos1.width, axpos1.height])


    dummy = fig2.add_subplot(222)
    ax2.set_position(dummy.get_position())
    dummy.remove()
    axpos2 = ax2.get_position()
    ax2.set_position([axpos2.x0+0.05, axpos2.y0+0.04, axpos2.width, axpos2.height])

    dummy = fig2.add_subplot(223)
    ax3.set_position(dummy.get_position())
    dummy.remove()
    axpos3 = ax3.get_position()
    ax3.set_position([axpos3.x0, axpos3.y0+0.01, axpos3.width, axpos3.height])

    dummy = fig2.add_subplot(224)
    ax4.set_position(dummy.get_position())
    dummy.remove()
    axpos4 = ax4.get_position()
    ax4.set_position([axpos4.x0+0.05, axpos4.y0+0.01, axpos4.width, axpos4.height])

    #fig2.set_size_inches(6.4, 5)
    fig2.patch.set_facecolor("white")
    #fig2.suptitle("Increase of 4 degrees")

    lines1, labels1 = ax1.get_legend_handles_labels()

    fig2.legend(lines1, labels1, bbox_to_anchor=(1, 0.075), ncol=2)

    fig2.text(0.93, 0.08, "˚C", fontsize = 12)
    #fig2.text(0.93, 0.53, "˚C", fontsize = 12)
    fig2.suptitle(r"$\bf{Electricity\:Demand\:Sensitivity\:To\:Temperature$", fontsize = 18)

    fig2.savefig("Images/elct_dmd_gw_2C_2slope")
    
    
    plt.show()

gw_elec_all()

#Before, California and Colorado did not have the added heating demand. Since I calculated the 
#heating demand out of heating degree days, I was able to add this to California
def gw_elec_Colorado_t_mod(degree_change, slope_factor):
    '''As it stands, we do not '''
    df = pd.DataFrame()
    y = heat_to_elec()[2]
    x = get_temp_data("weekly")[2]

    df["x"] = x
    df["y"] = y
    #print(df)

    total_elec_demand = round(df["y"].sum())

    fig, ax = plt.subplots()

    ax.scatter(df["x"], df["y"], s = 15, color = "C0", label = "original")
    # df["y"] = df.apply(lambda row: row["y"] + .232 * degree_change if row["x"] > 15.56 #Add according to positive slope if starts in cooling region
    #     else row["y"] + .232 * (row["x"] - 15.56 + degree_change) if row["x"]+ degree_change  >15.56 and row["x"] > 7.32
    #     else row["y"] if row["x"] > 7.32
    #     else row["y"] - 0.097281* (7.32- row["x"]) + 232 * (row["x"] + degree_change - 15.56) if row["x"] + degree_change > 7.32 and row["x"] + degree_change  > 15.56
    #     else row["y"] - 0.097281 * (7.32- row["x"]) if row["x"] + degree_change > 7.32
    #     else row["y"] - 0.097281 * degree_change, axis = 1)

    df["y"] = df.apply(lambda row: row["y"] + 249.56 * degree_change if row["x"] > 16.966 #Add according to positive slope if starts in cooling region
        else row["y"] + 249.56 * (row["x"] - 16.966 + degree_change) if row["x"]+ degree_change  >16.966 and row["x"] > 13.801
        else row["y"] if row["x"] > 13.801
        else row["y"] - 646.373 * (13.801- row["x"]) + 249.56 * (row["x"] + degree_change - 16.966) if row["x"] + degree_change > 13.801 and row["x"] + degree_change  > 16.966
        else row["y"] - 646.373 * (13.801- row["x"]) if row["x"] + degree_change > 13.801
        else row["y"] - 646.373 * degree_change, axis = 1)

    df["y"] = df.apply(lambda row: row["y"] + (row ["y"] - 9000) * (slope_factor-1) if row["x"] > 16.966 and row["y"] > 9000
    else row["y"], axis = 1)


    df["x"] = df.apply(lambda row: row["x"] + degree_change, axis = 1)

    #print(df)
    ax.scatter(df["x"], df["y"], s = 15, marker = "^", color = "C1", label = "with modification")

    # ax.axvline(7.32, color='black',ls='--', alpha = 0.5)
    # ax.text(7.32, ax.get_ybound()[1]-0.500, "T_th", horizontalalignment = "center", color = "C3")


    # ax.axvline(15.56, color='black',ls='--', alpha = 0.5)  
    # ax.text(15.56, ax.get_ybound()[1]-0.500, "T_th", horizontalalignment = "center", color = "C2")

    #ax.legend()
    ax.set_title("Colorado")
    ax.set_ylabel("Electricity demand (GWh)")
    plt.savefig(f"images/GWCO_incr{degree_change}_slope{slope_factor}_mod")
    # plt.close(fig)
    plt.show()
    return ax

#gw_elec_Colorado_t_mod(4, 1)

def gw_elec_California_t_mod(degree_change, slope_factor):
    '''Here, I am trying to model what would happen to electricity demand in California if
    the temperature increases uniformly by x degrees due to global warming
    
    For california, we assume that the electricity demand would be constant with change in 
    temperature until it reaches a threshold temperature (15.79 degrees). Then, there is
    a linear increase'''
    df = pd.DataFrame()
    
    x = get_temp_data("weekly")[3]
    y = heat_to_elec()[3]

    df["x"] = x
    df["y"] = y

    total_elec_demand = round(df["y"].sum())

    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"], s = 15, color = "C0", label = "original")


    df["y"] = df.apply(lambda row: row["y"] + 1093.304 * degree_change if row["x"] > 16.14 #Add according to positive slope if starts in cooling region
        else row["y"] + 1093.304 * (row["x"] - 16.14 + degree_change) if row["x"]+ degree_change  >16.14 and row["x"] > 14.22
        else row["y"] if row["x"] > 14.22
        else row["y"] - 640.248 * (14.22- row["x"]) + 1093.304 * (row["x"] + degree_change - 16.14) if row["x"] + degree_change > 14.22 and row["x"] + degree_change  > 16.14
        else row["y"] - 640.248 * (14.22- row["x"]) if row["x"] + degree_change > 14.22
        else row["y"] - 640.248 * degree_change, axis = 1)

    df["x"] = df.apply(lambda row: row["x"] + degree_change, axis = 1)
    
    #Use this line if you also want to include slope
    df["y"] = df.apply(lambda row: row["y"] + (row ["y"] - 35000) * (slope_factor-1) if row["x"] > 16.14 and row["y"] > 35000
    else row["y"], axis = 1)

    ax.scatter(df["x"], df["y"], s = 15, marker = "^", color = "C1", label = "with modification")

    # ax.axvline(15.79, color='black',ls='--', alpha = 0.5)
    # ax.text(15.79, ax.get_ybound()[1]-1.500, "T_th", horizontalalignment = "center", color = "C2")


    #ax.legend()
    ax.set_title("California")
    #ax.set_ylabel("Electricity demand (MWh)")
    fig.subplots_adjust(bottom=0.2)
    #print(df['y'].sum())
    plt.savefig(f"images/GWCali_incr{degree_change}_slope{slope_factor}_mod")
    #plt.close(fig)
    plt.show()
    return ax
    #if x+degree_change-15.79 is greater than 0, then add this value times 1093.394 to y

#gw_elec_California_t_mod(4, 2)
#gw_elec_all()
#In this section of code, I want to make a new data table of the averages of the csv files
# plt.scatter([1,1,4,5,2,1],[2,3,6,7,2,0])
# b, m = polyfit([1,1,4,5,2,1],[2,3,6,7,2,0],1)
# x = np.arange(10)
# plt.plot(x, b+m*x, '-')
# plt.show()
# print(b, m)


# sns.regplot(x = [1,1,4,5,2,1],y = [2,3,6,7,2,0])
# plt.show()
# x = [1,1,4,5,2,1]
# y = [2,3,6,7,2,0]

# x = np.array(x)
# y = np.array(y)

# results = sm.OLS(y, x.fit()
# print (results.summary())

# plt.scatter(x, y)

# x_plot = np.linspace(0, 7, 100)
# plt.plot(x_plot, x_plot*results.params[0] + results.params[1])
# plt.show()

####Outdated functions
def gw_elec_Spain(degree_change, slope_factor):
    '''This considers a universal degree change across all days. '''
    df = pd.DataFrame()
    y = heat_to_elec()["ESP_demand"]
    #y = get_electricity_data("weekly")[0]
    x = get_temp_data("weekly")[0]
    df["x"] = x
    df["y"] = y
    #print(df)

    total_elec_demand = round(df["y"].sum())

    fig, ax = plt.subplots()

    ax.scatter(df["x"], df["y"], color = "C0", label = "original")
    
    df["y"] = df.apply(lambda row: row["y"] + 865.2 * degree_change if row["x"] > 22.267 #Add according to positive slope if starts in cooling region
    else row["y"] + 865.2 * (row["x"] - 22.267 + degree_change) if row["x"]+ degree_change  >22.267 and row["x"] > 16
    else row["y"] if row["x"] > 16 
    else row["y"] - 1356.544 * (16- row["x"]) + 865.2 * (row["x"] + degree_change - 22.267) if row["x"] + degree_change > 16 and row["x"] + degree_change  > 22.267 
    else row["y"] - 1356.544 * (16- row["x"]) if row["x"] + degree_change > 16
    else row["y"] - 1356.544 * degree_change, axis = 1)

    df["y"] = df.apply(lambda row: row["y"] + (row ["y"] - 30000) * (slope_factor-1) if row["x"] > 22.267 and row["y"] > 30000
    else row["y"], axis = 1)

    new_elec_demand = round(df["y"].sum())

    change_demand = round((new_elec_demand-total_elec_demand)/total_elec_demand, 2) * 100

    total_elec_demand = "{:.2e}".format(total_elec_demand)
    new_elec_demand = "{:.2e}".format(new_elec_demand)
    textstr = '\n'.join(('Demand unmodified =' + total_elec_demand ,
    'Demand modified =' + new_elec_demand ,
    f'Percent change = {change_demand}%'))
    
    ax.text (0.3, 0.8, textstr, transform = ax.transAxes, fontsize = 10, 
    bbox = dict(boxstyle = "square", facecolor = "white", alpha = 0.5), verticalalignment = "top")


    df["x"] = df.apply(lambda row: row["x"] + degree_change, axis = 1)
    #print(df)


    ax.axvline(16, color='black',ls='--', alpha = 0.5)
    ax.text(16, ax.get_ybound()[1]-1500, "T_th", horizontalalignment = "center", color = "C3")

    ax.axvline(22.267, color='black',ls='--', alpha = 0.5)
    ax.text(22.267, ax.get_ybound()[1]-1500, "T_th", horizontalalignment = "center", color = "C2")

    
    ax.scatter(df["x"], df["y"], marker = "^", color = "C1", label = "with modification")
    ax.legend()
    ax.set_title(f"Spain increase of {degree_change} degrees and slope factor of {slope_factor}")
    ax.set_xlabel("Temperature (˚C)")
    ax.set_ylabel("Electricity demand (MWh)")
    #plt.savefig(f"images/GWESP_incr{degree_change}_slope{slope_factor}")

    return ax

def gw_elec_Colorado(degree_change, slope_factor):
    '''As it stands, we do not '''
    df = pd.DataFrame()
    y = get_electricity_data("weekly")[2]
    x = get_temp_data("weekly")[2]
    df["x"] = x
    df["y"] = y
    #print(df)

    total_elec_demand = round(df["y"].sum())

    fig, ax = plt.subplots()

    ax.scatter(df["x"], df["y"], color = "C0", label = "original")

    df["y"] = df.apply(lambda row: row["y"] + 232 * degree_change if row["x"] > 15.56 #Add according to positive slope if starts in cooling region
    else row["y"] + 232 * (row["x"] - 15.56 + degree_change) if row["x"]+ degree_change  >15.56 and row["x"] > 7.32
    else row["y"] if row["x"] > 7.32
    else row["y"] - 97.281* (7.32- row["x"]) + 232 * (row["x"] + degree_change - 15.56) if row["x"] + degree_change > 7.32 and row["x"] + degree_change  > 15.56
    else row["y"] - 97.281 * (7.32- row["x"]) if row["x"] + degree_change > 7.32
    else row["y"] - 97.281 * degree_change, axis = 1)

    df["y"] = df.apply(lambda row: row["y"] + (row ["y"] - 6600) * (slope_factor-1) if row["x"] > 15.56 and row["y"] > 6600
    else row["y"], axis = 1)

    new_elec_demand = round(df["y"].sum())

    change_demand = round((new_elec_demand-total_elec_demand)/total_elec_demand, 2) * 100

    total_elec_demand = "{:.2e}".format(total_elec_demand)
    new_elec_demand = "{:.2e}".format(new_elec_demand)
    textstr = '\n'.join(('Demand unmodified =' + total_elec_demand ,
    'Demand modified =' + new_elec_demand ,
    f'Percent change = {change_demand}%'))

    ax.text (0.05, 0.8, textstr, transform = ax.transAxes, fontsize = 10, bbox = dict(boxstyle = "square", facecolor = "white", alpha = 0.5), verticalalignment = "top")

    df["x"] = df.apply(lambda row: row["x"] + degree_change, axis = 1)
    #print(df)
    ax.scatter(df["x"], df["y"], marker = "^", color = "C1", label = "with modification")

    ax.axvline(7.32, color='black',ls='--', alpha = 0.5)
    ax.text(7.32, ax.get_ybound()[1]-500, "T_th", horizontalalignment = "center", color = "C3")


    ax.axvline(15.56, color='black',ls='--', alpha = 0.5)  
    ax.text(15.56, ax.get_ybound()[1]-500, "T_th", horizontalalignment = "center", color = "C2")

    ax.legend()
    ax.set_title(f"Colorado increase of {degree_change} degrees and slope factor of {slope_factor}")
    ax.set_xlabel("Temperature (˚C)")
    ax.set_ylabel("Electricity demand (MWh)")
    #plt.savefig(f"images/GWCO_incr{degree_change}_slope{slope_factor}")
    return ax


#gw_elec_Colorado(2, 2)
# for degree in degrees + degrees3:
#     gw_elec_Spain(degree, slopes[0])

# for degree in degrees + degrees2:
#     for slope in slopes[1:]:
#         gw_elec_Spain(degree, slope)

def gw_elec_California(degree_change, slope_factor):
    '''Here, I am trying to model what would happen to electricity demand in California if
    the temperature increases uniformly by x degrees due to global warming
    
    For california, we assume that the electricity demand would be constant with change in 
    temperature until it reaches a threshold temperature (15.79 degrees). Then, there is
    a linear increase'''
    df = pd.DataFrame()
    
    x = get_temp_data("weekly")[3]
    y = get_electricity_data("weekly")[3]
    df["x"] = x
    df["y"] = y

    total_elec_demand = round(df["y"].sum())

    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"], color = "C0", label = "original")


    df["y"] = df.apply(lambda row: row ["y"] + 1093.304 * degree_change if row["x"] > 15.79 
    else row["y"] + 1093.304 * (row["x"] - 15.79 + degree_change) if row["x"]+ degree_change - 15.79 > 0 
    else row["y"], axis = 1)

    df["x"] = df.apply(lambda row: row["x"] + degree_change, axis = 1)
    
    #Use this line if you also want to include slope
    df["y"] = df.apply(lambda row: row["y"] + (row ["y"] - 32500) * (slope_factor-1) if row["x"] > 15.79 and row["y"] > 32500
    else row["y"], axis = 1)

    new_elec_demand = round(df["y"].sum())

    change_demand = round((new_elec_demand-total_elec_demand)/total_elec_demand, 3) * 100
    #print(change_demand)
    total_elec_demand = "{:.2e}".format(total_elec_demand)
    new_elec_demand = "{:.2e}".format(new_elec_demand)
    textstr = '\n'.join(('Demand unmodified =' + total_elec_demand ,
    'Demand modified =' + new_elec_demand ,
    f'Percent change = {change_demand}%'))

    ax.text (0.05, 0.6, textstr, transform = ax.transAxes, fontsize = 10, bbox = dict(boxstyle = "square", facecolor = "white", alpha = 0.5), verticalalignment = "top")

    ax.text(0.9, 0.1, "˚C")
    ax.scatter(df["x"], df["y"], marker = "^", color = "C1", label = "with modification")

    ax.axvline(15.79, color='black',ls='--', alpha = 0.5)
    ax.text(15.79, ax.get_ybound()[1]-1500, "T_th", horizontalalignment = "center", color = "C2")


    ax.legend()
    ax.set_title(f"California increase of {degree_change} degrees and slope factor of {slope_factor}")
    ax.set_xlabel("Temperature (˚C)")
    ax.set_ylabel("Electricity demand (MWh)")
    fig.subplots_adjust(bottom=0.2)
    #print(df['y'].sum())
    #plt.savefig(f"images/GWCali_incr{degree_change}_slope{slope_factor}")
    return ax
    #if x+degree_change-15.79 is greater than 0, then add this value times 1093.394 to y

def gw_elec_Denmark(degree_change):
    '''Here, I am trying to model what would happen to electricity demand in Denmark if
    the temperature increases uniformly by x degrees due to global warming
    
    For Denmark we assume that the electricity demand would be constant with change in 
    temperature until it reaches a threshold temperature (15.79 degrees). Then, there is
    a linear increase'''
    df = pd.DataFrame()
    
    x = get_temp_data("weekly")[1]
    y = heat_to_elec()["DNK_demand"]
    df["x"] = x
    df["y"] = y

    total_elec_demand = round(df["y"].sum())
    #print(df['y'].sum())
    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"], color = "C0", label = "original")

    df["y"] = df.apply(lambda row: row ["y"] if row["x"] > 15.8
    else row["y"] - 273.665 * (15.8-row["x"]) if row["x"]+ degree_change - 15.8 > 0 
    else row["y"] - 273.665 * degree_change, axis = 1)
    #Like California, there are only three cases. Unlike california, the three cases are a bit different
    #flat to flat, heat to flat, heat to heat

    df["x"] = df.apply(lambda row: row["x"] + degree_change, axis = 1)
    
  
    new_elec_demand = round(df["y"].sum())
    change_demand = round((new_elec_demand-total_elec_demand)/total_elec_demand, 2) * 100
    print(change_demand)
    total_elec_demand = "{:.2e}".format(total_elec_demand)
    new_elec_demand = "{:.2e}".format(new_elec_demand)
    textstr = '\n'.join(('Demand unmodified =' + total_elec_demand ,
    'Demand modified =' + new_elec_demand ,
    f'Percent change = {change_demand}%'))

    ax.text (0.45, 0.6, textstr, transform = ax.transAxes, fontsize = 10, bbox = dict(boxstyle = "square", facecolor = "white", alpha = 0.5), verticalalignment = "top")

    ax.axvline(15.8, color='black',ls='--', alpha = 0.5)
    ax.text(15.8, ax.get_ybound()[1]-500, "T_th", horizontalalignment = "center", color = "C3")

    ax.scatter(df["x"], df["y"], color = "C1", label = "with temp increase")
    ax.legend()
    ax.set_title(f"Electricity demand vs. temperature Denmark with increase of {degree_change} degrees")
    ax.set_xlabel("Temperature (˚C)")
    ax.set_ylabel("Electricity demand (MWh)")

    #plt.savefig(f"images/GWDen_incr{degree_change}")
    #print(df['y'].sum())

    return ax




def mod_dfs(elec, temp):
    df = pd.DataFrame()
    df['a'] = elec
    print(temp.head())
    df['b'] = temp
    print(df.head())




