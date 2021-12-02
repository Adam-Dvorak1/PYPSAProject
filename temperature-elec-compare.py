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

    

def plot_electricity_data(elec_df):
    elec_df.plot()
    plt.show()



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


#Jan_plot_elec_temp(get_electricity_data("h")[3], get_temp_data("h")[3], "2011", "California")

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

#July_plot_elec_temp(get_electricity_data("h")[0], get_temp_data("h")[0], "2015", "Spain")

def mod_dfs(elec, temp):
    df = pd.DataFrame()
    df['a'] = elec
    print(temp.head())
    df['b'] = temp
    print(df.head())


#mod_dfs(get_electricity_data("weekly")[1], get_temp_data("weekly")[1])

def elec_vs_temp(elec, temp, country):
    # x = temp
    # y = elec
    df = pd.DataFrame()
    df['a'] = temp
    df['b'] = elec

    x = df['a']
    y = df['b']

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Electricity demand (MWh)")
    ax.set_title ("Weekly electricity demand vs temperature in " + country )

    mod_x  = x[x>15]
    mod_y = y[x>15]
    theta = np.polyfit(mod_x, mod_y, 1)

    print(f'The parameters of the curve: {theta}')


    plt.plot(np.unique(mod_x), np.poly1d(np.polyfit(mod_x, mod_y, 1))(np.unique(mod_x)))

    plt.savefig("images/" + country + "EDvsT")
    plt.show()




elec_vs_temp(get_electricity_data("weekly")[1], get_temp_data("weekly")[1], "Denmark")



#plot_elec_and_temp(get_electricity_data()[2], get_temp_data()[2])
#print(get_temp_data()[3])

#same_plot_elec_temp(get_electricity_data("daily")[3], get_temp_data("daily")[3], "2011", "California", "Daily")



#The next step is to make a piece 


#We now have temperature
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

