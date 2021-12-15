from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
import seaborn as sns
import statsmodels.api as sm
import matplotlib.dates as mdates


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
    df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0)# in MWh
    df_elec.index = pd.to_datetime(df_elec.index) #change index to datatime

    df_cal_elec = pd.read_csv('data_extra/CaliforniaTimeSeries.csv', index_col=0)
    df_cal_elec.index = pd.to_datetime(df_cal_elec.index)

    df_co_elec = pd.read_csv('data_extra/ColoradoTimeSeries.csv', index_col=0)
    df_co_elec.index = pd.to_datetime(df_co_elec.index)

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


    
    return ESDayElec, DKDayElec, CODayElec, CADayElec

def get_heat_demand_data(timeframe):
    df_heat = pd.read_csv('data/heat_demand.csv', sep=';', index_col=0)# in MWh
    df_heat.index = pd.to_datetime(df_heat.index) #change index to datatime

    ES_heat = df_heat['ESP']
    DK_heat = df_heat['DNK']

    if timeframe == "weekly":
        ESDayheat = ES_heat.rolling(168).mean()
        ESDayheat = ESDayheat.iloc[::168].shift(-1)[:-1]
        DKDayheat = DK_heat.rolling(168).mean()
        DKDayheat = DKDayheat.iloc[::168].shift(-1)[:-1]

    elif timeframe == "daily":
        ESDayheat = ES_heat.rolling(24).mean()
        ESDayheat = ESDayheat.iloc[::24].shift(-1)[:-1]
        DKDayheat = DK_heat.rolling(24).mean()
        DKDayheat = DKDayheat.iloc[::24].shift(-1)[:-1]

    else:
        ESDayheat = ES_heat
        DKDayheat = DK_heat

    return ESDayheat, DKDayheat






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

    ESP_heat = heat_data[0]
    DNK_heat = heat_data[1]

    #Here, we make a new dataframe and add elec and 1/3 heat to it

    new_elec = pd.DataFrame()
    new_elec["ESP_demand"] = ESP_elec + ESP_heat/3
    new_elec["DNK_demand"] = DNK_elec + DNK_heat/3

    return new_elec

    #Here, we return the new dataframe

def elec_vs_temp_Spain():

    '''This makes a plot of the electricity demand (or heating demand or elec+heating demand) vs temperature
    I need to do some more work playing with annotate vs '''
    # x = temp
    # y = elec
   
    y = heat_to_elec()["ESP_demand"]
    #y = get_electricity_data("weekly")[0]
    #y = get_heat_demand_data("weekly")[0]
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
    ax.axhline(y=30000, xmin = 0.5, xmax = 0.75, color = "C2")
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

    plt.plot(np.unique(new_x), np.poly1d(np.polyfit(new_x, new_y, 1))(np.unique(new_x)))


    #plt.savefig("images/SpainED_plus_HDvsTw_eq")
    plt.show()

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

def elec_vs_temp_Colorado():
    x = get_temp_data("weekly")[2]
    y = get_electricity_data("weekly")[2]

    fig, ax = plt.subplots()
    
    fig.subplots_adjust(left = 0.2)
    ax.scatter(x, y)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Electricity (MWh)")
    ax.set_title ("Weekly electricity demand vs temperature in Colorado")

    mod_x  = x[x>15.5]
    mod_y = y[x>15.5]

    
    
    corr = np.corrcoef(mod_x, mod_y).round(decimals = 3)[0, 1]

    theta = np.polyfit(mod_x, mod_y, 1).round(decimals = 3)
    ax.axhline(y = 6600, xmin = 0.5, xmax = 0.7, color = "C2")
    #settings 
    ax.annotate(f'{theta[0]}x + {theta[1]}', xy = (mod_x[2], mod_y[2]), xytext = (mod_x[2]-10 , mod_y[2]* 1.15 ))
    ax.annotate(f'correlation = {corr} ', xy = (mod_x[2], mod_y[2]), xytext = (mod_x[2] - 10, mod_y[2]*1.1 ))


    plt.plot(np.unique(mod_x), np.poly1d(np.polyfit(mod_x, mod_y, 1))(np.unique(mod_x)))

    new_x = x[x<8]
    new_y = y[x<8]
    corr2 = np.corrcoef(new_x, new_y).round(decimals = 3)[0, 1]

    theta2 = np.polyfit(new_x, new_y, 1).round(decimals = 3)
    ax.annotate(f'{theta2[0]}x + {theta2[1]}', xy = (new_x[4], new_y[4]), xytext = (new_x[4], new_y[4]+100))
    ax.annotate(f'correlation = {corr2} ', xy = (new_x[4], new_y[4]), xytext = (new_x[4], new_y[4]+ 400))

    plt.plot(np.unique(new_x), np.poly1d(np.polyfit(new_x, new_y, 1))(np.unique(new_x)))
    #plt.savefig("images/ColEDvsTw_eqf")
    plt.show()

def elec_vs_temp_California():
    x = get_temp_data("weekly")[3]
    y = get_electricity_data("weekly")[3]

    fig, ax = plt.subplots()
    
    fig.subplots_adjust(left = 0.2)
    ax.scatter(x, y)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Electricity (MWh)")
    ax.set_title ("Weekly electricity demand vs temperature in California")

    mod_x  = x[x>17]
    mod_y = y[x>17]

    
    
    corr = np.corrcoef(mod_x, mod_y).round(decimals = 3)[0, 1]

    theta = np.polyfit(mod_x, mod_y, 1).round(decimals = 3)

    #settings 
    ax.axhline(y = 32500, xmin =.25, xmax = 0.45)
    ax.annotate(f'{theta[0]}x + {theta[1]}', xy = (mod_x[2], mod_y[2]), xytext = (mod_x[2] , mod_y[2] ))
    ax.annotate(f'correlation = {corr} ', xy = (mod_x[2], mod_y[2]), xytext = (mod_x[2] - 10, mod_y[2]))


    plt.plot(np.unique(mod_x), np.poly1d(np.polyfit(mod_x, mod_y, 1))(np.unique(mod_x)))

    #plt.savefig("images/CaliEDvsTw_eqf")
    plt.show()


# elec_vs_temp_Denmark()
# elec_vs_temp_Spain()
# elec_vs_temp_Colorado()
# elec_vs_temp_California()

def plot_ED_and_CF_data():
    '''This function plots the electricity demand for one country on one axis and the capacity factors on other axes (sharing a y axis).
    Modify it to change which country (make sure CA and CO use "Time in 2011")
    
    we can plot either just elec, or elec+heat, using temp_to_elec'''
    elec = get_electricity_data("weekly")[3]

    #elec = temp_to_elec()["DNK_demand"]
    solar = get_solar_data("weekly")[3]
    wind = get_wind_data("weekly")[3]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2= fig.add_subplot(111, frame_on=False)
    ax3 = fig.add_subplot(111, sharey = ax2, frame_on=False)
    ax1.plot(elec, 'C0-', label = "Electricity demand")
    ax2.plot(solar, 'C1-', label = "Solar CF")
    ax3.plot(wind, 'C2-', label = "Wind CF")

    ax1.set_xlabel("Time in 2011")
    ax1.set_ylabel('Electricity demand (MWh)')
    ax2.set_xticks([])
    ax2.set_ylabel('Capacity factors')
    ax2.yaxis.set_label_position("right")


    ax2.yaxis.tick_right()
    ax3.set_xticks([])
    ax3.yaxis.set_visible(False)

    ax1.set_title("California Electricity Demand and Capacity factors")


    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()

    ax2.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3)
    fig.set_size_inches(12, 10)
    #plt.savefig("images/EDandCFCaliforniaf")
    plt.show()
'''This section of equations considers what happens if you add degrees to all '''
def gw_elec_Spain(degree_change):
    '''This considers a universal degree change across all days. '''
    df = pd.DataFrame()
    y = heat_to_elec()["ESP_demand"]
    #y = get_electricity_data("weekly")[0]
    x = get_temp_data("weekly")[0]
    df["x"] = x
    df["y"] = y
    print(df)

    fig, ax = plt.subplots()

    ax.scatter(df["x"], df["y"], color = "C0", label = "original")

    df["y"] = df.apply(lambda row: row["y"] + 865.2 * degree_change if row["x"] > 22.267 #Add according to positive slope if starts in cooling region
    else row["y"] + 865.2 * (row["x"] - 22.267 + degree_change) if row["x"]+ degree_change  >22.267 and row["x"] > 16
    else row["y"] if row["x"] > 16 
    else row["y"] - 1356.544 * (16- row["x"]) + 865.2 * (row["x"] + degree_change - 22.267) if row["x"] + degree_change > 16 and row["x"] + degree_change  > 22.267 
    else row["y"] - 1356.544 * (16- row["x"]) if row["x"] + degree_change > 16
    else row["y"] - 1356.544 * degree_change, axis = 1)


    df["x"] = df.apply(lambda row: row["x"] + degree_change, axis = 1)
    print(df)
    ax.scatter(df["x"], df["y"], color = "C1", label = "with temp increase")
    ax.legend()
    ax.set_title(f"Electricity demand vs. temperature Spain with increase of {degree_change} degrees")
    ax.set_xlabel("Temperature (˚C)")
    ax.set_ylabel("Electricity demand (MWh)")

    plt.show()

def gw_elec_Colorado(degree_change):
    df = pd.DataFrame()
    y = get_electricity_data("weekly")[2]
    x = get_temp_data("weekly")[2]
    df["x"] = x
    df["y"] = y
    print(df)

    fig, ax = plt.subplots()

    ax.scatter(df["x"], df["y"], color = "C0", label = "original")

    df["y"] = df.apply(lambda row: row["y"] + 232 * degree_change if row["x"] > 15.56 #Add according to positive slope if starts in cooling region
    else row["y"] + 232 * (row["x"] - 15.56 + degree_change) if row["x"]+ degree_change  >15.56 and row["x"] > 7.32
    else row["y"] if row["x"] > 7.32
    else row["y"] - 97.281* (7.32- row["x"]) + 232 * (row["x"] + degree_change - 15.56) if row["x"] + degree_change > 7.32 and row["x"] + degree_change  > 15.56
    else row["y"] - 97.281 * (7.32- row["x"]) if row["x"] + degree_change > 7.32
    else row["y"] - 97.281 * degree_change, axis = 1)

    #not quite right but close
    df["x"] = df.apply(lambda row: row["x"] + degree_change, axis = 1)
    print(df)
    ax.scatter(df["x"], df["y"], color = "C1", label = "with temp increase")
    ax.legend()
    ax.set_title(f"Electricity demand vs. temperature Colorado with increase of {degree_change} degrees")
    ax.set_xlabel("Temperature (˚C)")
    ax.set_ylabel("Electricity demand (MWh)")

    plt.show()

def gw_elec_California(degree_change):
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

    print(df['y'].sum())
    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"], color = "C0", label = "original")

    df["y"] = df.apply(lambda row: row ["y"] + 1093.304 * degree_change if row["x"] > 15.79 
    else row["y"] + 1093.304 * (row["x"] - 15.79 + degree_change) if row["x"]+ degree_change - 15.79 > 0 
    else row["y"], axis = 1)

    df["x"] = df.apply(lambda row: row["x"] + degree_change, axis = 1)
    
  
    ax.scatter(df["x"], df["y"], marker = "^", color = "C1", label = "with temp increase")
    ax.legend()
    ax.set_title(f"Electricity demand vs. temperature California with increase of {degree_change} degrees")
    ax.set_xlabel("Temperature (˚C)")
    ax.set_ylabel("Electricity demand (MWh)")
    print(df['y'].sum())
    plt.savefig("images/GWCali_incr")
    plt.show()
    #if x+degree_change-15.79 is greater than 0, then add this value times 1093.394 to y

def gw_elec_Denmark(degree_change):
    '''Here, I am trying to model what would happen to electricity demand in California if
    the temperature increases uniformly by x degrees due to global warming
    
    For california, we assume that the electricity demand would be constant with change in 
    temperature until it reaches a threshold temperature (15.79 degrees). Then, there is
    a linear increase'''
    df = pd.DataFrame()
    
    x = get_temp_data("weekly")[1]
    y = heat_to_elec()["DNK_demand"]
    df["x"] = x
    df["y"] = y

    #print(df['y'].sum())
    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"], color = "C0", label = "original")

    df["y"] = df.apply(lambda row: row ["y"] if row["x"] > 15.8
    else row["y"] - 273.665 * (15.8-row["x"]) if row["x"]+ degree_change - 15.8 > 0 
    else row["y"] - 273.665 * degree_change, axis = 1)
    #Like California, there are only three cases. Unlike california, the three cases are a bit different
    #flat to flat, heat to flat, heat to heat

    df["x"] = df.apply(lambda row: row["x"] + degree_change, axis = 1)
    
  
    ax.scatter(df["x"], df["y"], color = "C1", label = "with temp increase")
    ax.legend()
    ax.set_title(f"Electricity demand vs. temperature Denmark with increase of {degree_change} degrees")
    ax.set_xlabel("Temperature (˚C)")
    ax.set_ylabel("Electricity demand (MWh)")
    #print(df['y'].sum())
    plt.show()


def gw2_elec_California(slope_factor):
    df = pd.DataFrame()
    
    x = get_temp_data("weekly")[3]
    y = get_electricity_data("weekly")[3]
    df["x"] = x
    df["y"] = y

    #print(df['y'].sum())
    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"], color = "C0", label = "original")

    df["y"] = df.apply(lambda row: row["y"] + (row ["y"] - 32500) * slope_factor if row["x"] > 15.79 and row["y"] > 32500
     
    else row["y"], axis = 1)


    
  
    ax.scatter(df["x"], df["y"], marker = "^", color = "C1", label = "with temp increase")
    ax.legend()
    ax.set_title(f"Electricity demand vs. temperature California with slope factor of {slope_factor}")
    ax.set_xlabel("Temperature (˚C)")
    ax.set_ylabel("Electricity demand (MWh)")
    

    #plt.savefig("images/GWCali_slopef")
    plt.show()


gw_elec_California(2)
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

def mod_dfs(elec, temp):
    df = pd.DataFrame()
    df['a'] = elec
    print(temp.head())
    df['b'] = temp
    print(df.head())




