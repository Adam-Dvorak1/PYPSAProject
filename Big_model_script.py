#%%
import resource
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy
import time
from itertools import repeat
from matplotlib import rc
import os
import re
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import adjustText
from adjustText import adjust_text
import pathlib
import matplotlib
import csv
import glob

from pylab import *

import xarray as xr
#%%
'''The issue is that before we could just get one item from each country. Now we need entire time series'''



eu28 = [
    "FR",
    "DE",
    "GB",
    "IT",
    "ES",
    "PL",
    "SE",
    "NL",
    "BE",
    "FI",
    "DK",
    "PT",
    "RO",
    "AT",
    "BG",
    "EE",
    "GR",
    "LV",
    "CZ",
    "HU",
    "IE",
    "SK",
    "LT",
    "HR",
    "LU",
    "SI",
    "CY",
    "MT",
]

eu7 = ["AL", "BA", "CH", "ME", "MK", "NO", "RS"] #these are countries that are in PyPSA-Eur-Sec but not in EU

eu28.pop(-1) #pops CY
eu28.pop(-1) #pops MT
eu28 = eu28 + eu7
eu28 = tuple(eu28)
#run_name = "adam_solar_costs_low_res"
# filepath = "results/"+ run_name + "/postnetworks/elec_s_37_lv1.0__Co2L0-3H-T-H-B-I-A-solar+c0.1002546-solar+p3-dist1_2030.nc"
#loop over all files in folder
def all_generators(run_name):
    #I am not sure if this actually works with the below
    generator = []
    for filepath in glob.glob("results/" + run_name + "/postnetworks/*"):
        my_run = retrieve_generators(filepath)
        generator.append(my_run)
    generator= pd.concat(generator)
    generator.index = pd.to_numeric(generator.index)

    generator = generator.sort_index(ascending = True)
    generator.to_csv(f'results/' + run_name + '/csvs/generators_T.csv')
    return generator
    
     
    # with open (f"results/csvs/{run_name}_generators.csv", "w", newline = '') as file:
    #     writer = csv.writer(file)
    #     for filepath in glob.glob("results/" + run_name + "/postnetworks/*"):
    #         my_run = retrieve_generators(filepath)
    #         writer.writerow(my_run)
        
def all_generators_real(run_name):
    #I tried to modify it so it will work with my new code. I think that the old function is outdated
    for filepath in glob.glob("results/" + run_name + "/postnetworks/*"):
        retrieve_generators(filepath)
        generator.append(my_run)
    generator= pd.concat(generator)
    generator.index = pd.to_numeric(generator.index)

    generator = generator.sort_index(ascending = True)
    generator.to_csv(f'results/' + run_name + '/csvs/generators_T.csv')
    return generator



def load_temp_data():

    '''This function is to find the time series of '''
    idx = pd.IndexSlice

    countries = eu28

    ds = xr.open_dataset('data/resources/temp_air_rural_elec_s_37.nc') #we find that rural and urban have the same
    # ds2 = xr.open_dataset('data/resources/temp_air_urban_elec_s_37.nc')


    df = ds.to_dataframe()#this is a multi index dataframe
    df= df.unstack(level = 1)#This turns all of the different regions into 

    df.columns = [col[1] for col in df.columns]#When we stacked it, each column name had two names--temperature and country. This gets rid of the country
    
    df = df[df.columns[df.columns.str.startswith(countries)]]

    for country in countries:
        temps = df.copy()
        temps = temps[[col for col in temps.columns if col.startswith(country)]]
        df[country]  = temps.mean(axis = 1) #There are some with multiple nodes, with multiple time series. To solve this we just take the average

    df = df[df.columns[~df.columns.str.contains('[0-9]+')]]

    df = df.append(df.iloc[0])
    df = df.rolling(3).mean()[::3]

    df = df.shift(-1)
    df = df[:-1]

    degreedays = df.copy()
    degreedays = degreedays - 18

    coolDD = degreedays.copy()
    coolDD = coolDD.mask(coolDD < 0, 0)
    heatDD = degreedays.copy()
    heatDD = heatDD.mask(heatDD> 0, 0)


    coolsum = coolDD.sum()/2920
    coolsum = coolsum.to_frame()

    heatsum = heatDD.sum()/2920
    heatsum  = heatsum.to_frame()



    heatcool = pd.concat([coolsum, heatsum], axis = 1)

    heatcool.columns = ["avgcoolDD", "avgheatDD"]

    #heatcool.to_csv("data/avgHDDCDD.csv")
    

    return coolDD, heatDD

    


    
    # df2 = ds2.to_dataframe()
    # df2 = df2.unstack(level = 1)



    




#filepath = "results/adam_latitude_compare3h/postnetworks/elec_s_37_lv1.0__1_Co2L0-3H-T-H-B-I-A-solar+p3-dist0_2030.nc"
#filepath = "results/adam_latitude_compareno_sectors/postnetworks/elec_s_37_lv1.0__1_Co2L0-168H-solar+p3-dist0_2030.nc" #This new filepath now deals with no sectors
# o = re.split('\_', "results/adam_solar_costs_low_res/postnetworks/elec_s_37_lv1.0__0.0319672952694881_Co2L0-168H-T-H-B-I-A-solar+p3-dist1_2030.nc")
# "results/adam_solar_costs_low_res/postnetworks/elec_s_37_lv1.0__0.0319672952694881_Co2L0-168H-T-H-B-I-A-solar+p3-dist1_2030.nc"
# all_generators("adam_solar_costs_test2")  
def make_stackplot(run_name, relative):
    generators = pd.read_csv(f'results/' + run_name + '/csvs/generators.csv')
    DKsolar = generators['DKsolar']
    DKwind = generators['DKonwind'] + generators['DKoffwind']

    ESsolar = generators['ESsolar']
    ESwind = generators['ESonwind'] + generators['ESoffwind']
    ESror = generators['ESror']

    if relative == True:
        total = (DKsolar + DKwind)
        DKsolar = DKsolar/total
        DKwind = DKwind/total

        total2 = (ESsolar + ESwind + ESror)
        ESsolar = ESsolar/total2
        ESwind = ESwind/total2
        ESror = ESror/total2

    x = generators['Solar_cost'] * 0.6
    fig, axs = plt.subplots(2, 1)
    axs[0].stackplot(x, DKsolar, DKwind, colors = ["#f9d002","#235ebc"], labels = ["Solar", "Wind"])
    axs[1].stackplot(x, ESsolar, ESwind, ESror, colors = ["#f9d002","#235ebc",'#3dbfb0'], labels = ["Solar", "Wind"] )

    if relative == "True":
        for ax in plt.gcf().get_axes():
            ax.axvline(0.529, color='black',ls='--')
            ax.axvline(1.3, color='black',ls='--')
            #We want to make a range for today's prices. the upper range is 
            ax.text(0.85,0.5,  "Today's range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
            ax.axvline(0.019, color='black',ls='--')
            ax.text(0.025,0.5, "2050--Optimistic",  fontsize = 12, horizontalalignment = "center", rotation = "vertical")
            ax.axvline(0.095, color='black',ls='--')
            ax.fill_between([0.529, 1.3], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
            ax.text(0.11, 0.05,  "2050--Less Optimistic", fontsize = 11, horizontalalignment = "center", rotation = "vertical")
    else:
        for ax in plt.gcf().get_axes():
            ax.axvline(0.529, color='black',ls='--')
            ax.axvline(1.3, color='black',ls='--')
            #We want to make a range for today's prices. the upper range is 
            ax.text(0.85,2*10**6,  "Today's range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
            ax.axvline(0.019, color='black',ls='--')
            ax.text(0.025,0.05, "2050--Optimistic",  fontsize = 12, horizontalalignment = "center", rotation = "vertical")
            ax.axvline(0.095, color='black',ls='--')
            ax.fill_between([0.529, 1.3], y1 = max(ESsolar), alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
            ax.text(0.11, 2*10**6,  "2050--Less Optimistic", fontsize = 11, horizontalalignment = "center", rotation = "vertical")

    for ax in plt.gcf().get_axes():
        ax.minorticks_on()
      
        ax.label_outer()

        ax.margins(x = 0)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))
        ax.set_xlim(0.05, 3)

    axs[1].set_xlabel("Cost of Solar")
    axs[0].set_title("Denmark")
    axs[1].set_title("Spain")
    lines1, labels1 = axs[0].get_legend_handles_labels()

    fig.legend(lines1, labels1, bbox_to_anchor=(0.92, 0.08), ncol=2)

    fig.suptitle("3h resolution (10 points), relative share")
    
    if relative == True:
        fig.supylabel ("Resource share by fraction")
    else:
        fig.supylabel ("Generation in MWh")
    
    fig.savefig("Images/3h_10_points_relative")


    plt.show()
    return generators
    
#f = make_stackplot("adam_solar_3", True)



def retrieve_timeseries(filepath):
    '''This function takes a completed postnetwork, finds the resource frac'''
    
    europe = pypsa.Network()
    europe.import_from_netcdf(filepath)

    my_gen = ("offwind-ac", "offwind-dc", "solar", "onwind", "ror")

    countries = eu28
    mydata = europe.generators_t.p
    mydata = mydata[mydata.columns[mydata.columns.str.startswith(countries) ]]

    mydata = mydata[mydata.columns[mydata.columns.str.endswith(my_gen)]]


    myloads = europe.loads_t.p
    myloads = myloads[myloads.columns[myloads.columns.str.startswith(countries)]]

    myloads = myloads-myloads.mean()#This is just myloads
    myloads.rename(columns = lambda x: x[:2], inplace = True) #All of the loads are only named by the country
    myloads = myloads.groupby(level = 0, axis = 1).sum() #There are some duplicate loads. This groups them (adds them together)

    for country in countries:
    #The purpose of this section is to get sum all the solar
        resource_fracs = mydata
        resource_fracs = resource_fracs[[col for col in resource_fracs.columns if col.startswith(country)]]
        mydata[country] = resource_fracs[[col for col in resource_fracs.columns if col.endswith('solar')]].sum(axis = 1) #sums all the solar stuff together

    mydata = mydata[mydata.columns[~mydata.columns.str.contains('[0-9]+')]]#gets rid of old columns, not needed in new code

    mydata = mydata-mydata.mean()

    
    covariances = pd.DataFrame()

    for col in myloads.columns:
        covariances[col] = myloads[col] * mydata[col]/(myloads[col].std()*mydata[col].std())#This is a time series of 


    print(covariances)
    print(covariances.sum()/len(covariances))
        


    #The purpose of this section is to multiply the loads and supply together, element-wise

    print(mydata)
    
    
    


    mydata = mydata.to_frame()



    #In this section of the code, I
        #Extract 
    #save csv in new file, related to filepath
    now_path = pathlib.Path(filepath)
    parent_path = now_path.parent
    run_directory = parent_path.parent


    


    #mydata.to_csv(run_directory / "csvs/generators_T.csv")




    return mydata





def retrieve_generators(filepath):
    '''This function takes a completed postnetwork, finds the resource frac
    
    This function is useful if there is one postnetwork per run name'''
    
    europe = pypsa.Network()
    europe.import_from_netcdf(filepath)

    my_gen = ("offwind-ac", "offwind-dc", "solar", "onwind", "ror")

    #Before, we were missing 
    countries = eu28
    mydata = europe.generators_t.p
    mydata = mydata[mydata.columns[mydata.columns.str.startswith(countries) ]]

    mystorage = europe.storage_units_t.p
    #mystorage = mystorage[mystorage.columns[mystorage.columns.str.startswith(countries) ]] #this line is actually useless for PHS because all generators start with one of the european countries
    mystorage = mystorage[mystorage.columns[mystorage.columns.str.endswith("hydro")]]

    #This deals with p_nom_opt
    mydata = mydata[mydata.columns[mydata.columns.str.endswith(my_gen)]]

    
    totalpowers = europe.generators.p_nom_opt#installed generators
    totalpowers = totalpowers[mydata.columns]

    totalpowers = totalpowers.to_frame()
    totalpowers = totalpowers.T #The p_nom_opt is not a timeseries. However, before we were using timeseries. 
                                # So, to make the same code work (combining similar names), we transpose it


    myloads = europe.loads_t.p
    myloads = myloads[myloads.columns[myloads.columns.str.startswith(countries)]]

    myloads = myloads-myloads.mean()
    myloads.rename(columns = lambda x: x[:2], inplace = True) #All of the loads are only named by the country
    myloads = myloads.groupby(level = 0, axis = 1).sum() #There are some duplicate loads. This groups them (adds them together)

    weekloads = myloads.rolling(56).mean()[::56]

    weekloads = weekloads.drop(0)# I did not add this line before. I wonder if this will make a difference
                                    #It did make a bit of a difference--about 1% Nothign to write home about


    #In this section, we attempt to add the covariance of solar with the cooling and heating degree days
    coolDD, heatDD = load_temp_data()

    coolDD = coolDD-coolDD.mean() #in the calculation of covariances, we want to use the difference between the amount of CDD/HDD and the mean
    heatDD = heatDD-heatDD.mean()

    coolDD = coolDD.reset_index()
    coolDD = coolDD.drop('time', axis = 1)

    heatDD = heatDD.reset_index()
    heatDD = heatDD.drop('time', axis = 1)
    heatDD = heatDD * -1

    coolweek = coolDD.rolling(56).mean()[::56]
    heatweek = heatDD.rolling(56).mean()[::56]



    # coolsum = coolsum.T



    for country in countries:
        resource_fracs = mydata
        resource_fracs = resource_fracs[[col for col in resource_fracs.columns if col.startswith(country)]]
        mydata[country + 'solar'] = resource_fracs[[col for col in resource_fracs.columns if col.endswith('solar')]].sum(axis = 1) #sums all the solar stuff together
        mydata[country + 'wind'] = resource_fracs[[col for col in resource_fracs.columns if 'wind' in col]].sum(axis = 1)
        if any('ror' in col for col in resource_fracs.columns):#checks if there is a 'ror' column present
            mydata[country + 'ror'] =  resource_fracs[[col for col in resource_fracs.columns if 'ror' in col]].sum(axis = 1)




        allsources = totalpowers
        allsources =allsources[[col for col in allsources.columns if col.startswith(country)]]
        totalpowers[country + 'solar'] = allsources[[col for col in allsources.columns if col.endswith('solar')]].sum(axis = 1) #sums all the solar stuff together
        totalpowers[country + 'wind'] = allsources[[col for col in allsources.columns if 'wind' in col]].sum(axis = 1)
        if any('ror' in col for col in allsources.columns):#checks if there is a 'ror' column present
            totalpowers[country + 'ror'] =  allsources[[col for col in allsources.columns if 'ror' in col]].sum(axis = 1)



        allstorage = mystorage
        allstorage = allstorage[[col for col in allstorage.columns if col.startswith(country)]]
        if any ('hydro' in col for col in allstorage.columns):
            mydata[country + "hydro"] = allstorage[[col for col in allstorage.columns]].sum(axis = 1) 
        

        




    mydata = mydata[mydata.columns[~mydata.columns.str.contains('[0-9]+')]]#gets rid of old columns, not needed in new code

    #print(mydata)

    totalpowers = totalpowers[totalpowers.columns[~totalpowers.columns.str.contains('[0-9]+')]]
    totalpowers = totalpowers.T


    moddata = mydata-mydata.mean()

    weekdata = moddata.rolling(56).mean()[::56]
    weekdata = weekdata.drop(0)#question: we are dropping the first row because the rolling/mean combo makes the first row NaN.
    #However, what about the loads?


    covariances = pd.DataFrame()

    weekcovariances = pd.DataFrame()
    
    coolvariances = pd.DataFrame()

    coolweekvar = pd.DataFrame()

    heatvariances = pd.DataFrame()

    heatweekvar = pd.DataFrame()
    
    


    for col in myloads.columns:
        covariances[col+'solar'] = myloads[col] * moddata[col + 'solar']/(myloads[col].std()*moddata[col + 'solar'].std())
        covariances[col+ 'wind'] = myloads[col] * moddata[col + 'wind']/(myloads[col].std()*moddata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            covariances[col + 'ror'] =  myloads[col] * moddata[col + 'ror']/(myloads[col].std()*moddata[col + 'ror'].std())


        weekcovariances[col+'solar'] = weekloads[col] * weekdata[col + 'solar']/(weekloads[col].std()*weekdata[col + 'solar'].std())
        weekcovariances[col+ 'wind'] = weekloads[col] * weekdata[col + 'wind']/(weekloads[col].std()*weekdata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            weekcovariances[col + 'ror'] =  weekloads[col] * weekdata[col + 'ror']/(weekloads[col].std()*weekdata[col + 'ror'].std())
    

        coolvariances[col + 'solar'] = coolDD[col] * moddata[col + 'solar']/(coolDD[col].std()*moddata[col + 'solar'].std())
        coolvariances[col + 'wind'] = coolDD[col] * moddata[col + 'wind']/(coolDD[col].std()*moddata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            coolvariances[col + 'ror'] =  coolDD[col] * moddata[col + 'ror']/(coolDD[col].std()*moddata[col + 'ror'].std())
        
        coolweekvar[col + 'solar'] = coolweek[col] * weekdata[col + 'solar']/(coolweek[col].std()*weekdata[col + 'solar'].std())
        coolweekvar[col + 'wind'] = coolweek[col] * weekdata[col + 'wind']/(coolweek[col].std()*weekdata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            coolweekvar[col + 'ror'] =  coolweek[col] * weekdata[col + 'ror']/(coolweek[col].std()*weekdata[col + 'ror'].std())


        

        heatvariances[col + 'solar'] = heatDD[col] * moddata[col + 'solar']/(heatDD[col].std()*moddata[col + 'solar'].std())
        heatvariances[col + 'wind'] = heatDD[col] * moddata[col + 'wind']/(heatDD[col].std()*moddata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            heatvariances[col + 'ror'] =  heatDD[col] * moddata[col + 'ror']/(heatDD[col].std()*moddata[col + 'ror'].std())



        heatweekvar[col + 'solar'] = heatweek[col] * weekdata[col + 'solar']/(heatweek[col].std()*weekdata[col + 'solar'].std())
        heatweekvar[col + 'wind'] = heatweek[col] * weekdata[col + 'wind']/(heatweek[col].std()*weekdata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            heatweekvar[col + 'ror'] =  heatweek[col] * weekdata[col + 'ror']/(heatweek[col].std()*weekdata[col + 'ror'].std())
    
    # print(weekcovariances)
    weekcovariances = weekcovariances.sum()/len(weekcovariances)
    weekcovariances.to_frame()
    # print(weekcovariances)

    covariances = covariances.sum()/len(covariances)
    covariances.to_frame()

    coolvariances = coolvariances.sum()/len(coolvariances)
    coolvariances.to_frame()

    heatvariances = heatvariances.sum()/len(heatvariances)
    heatvariances.to_frame()


    coolweekvar = coolweekvar.sum()/len(coolweekvar)
    coolweekvar.to_frame()

    heatweekvar = heatweekvar.sum()/len(heatweekvar)
    heatweekvar.to_frame()
    #print(covariances.columns)

    #In this section:
        #Bring in the relevant dataframes from temp function
        #Add to mydata: CDDs/day on average, HDDs/day on average
        #Also, covariance between CDD, HDDs
    
    




    
    mydata = mydata.sum()



    


    mydata = mydata.to_frame()

    mydata = pd.concat([mydata, totalpowers, covariances, weekcovariances, coolvariances, coolweekvar, heatvariances, heatweekvar], axis = 1)#We want to do the same thing with the
    #mydata = pd.concat([mydata, totalpowers, weekcovariances], axis = 1)

    #mydata.columns = ['0', 'p_nom_opt', 'load_corr', 'week_corr']

    #In this section of the code, I
        #Extract 
    #save csv in new file, related to filepath
    now_path = pathlib.Path(filepath)
    parent_path = now_path.parent
    run_directory = parent_path.parent


    


    mydata.to_csv(run_directory / "csvs/generators_T.csv")




    return mydata

def mod_retrieve_generators(filepath):
    '''This function takes a completed postnetwork, finds the resource frac
    
    This function is useful if there is one postnetwork per run name
    
    This function is a test to see if the definition of covariance I have is wrong'''
    
    europe = pypsa.Network()
    europe.import_from_netcdf(filepath)

    my_gen = ("offwind-ac", "offwind-dc", "solar", "onwind", "ror")

    #Before, we were missing 
    countries = eu28
    mydata = europe.generators_t.p
    mydata = mydata[mydata.columns[mydata.columns.str.startswith(countries) ]]

    mystorage = europe.storage_units_t.p
    #mystorage = mystorage[mystorage.columns[mystorage.columns.str.startswith(countries) ]] #this line is actually useless for PHS because all generators start with one of the european countries
    mystorage = mystorage[mystorage.columns[mystorage.columns.str.endswith("hydro")]]

    #This deals with p_nom_opt
    mydata = mydata[mydata.columns[mydata.columns.str.endswith(my_gen)]]

    
    totalpowers = europe.generators.p_nom_opt#installed generators
    totalpowers = totalpowers[mydata.columns]

    totalpowers = totalpowers.to_frame()
    totalpowers = totalpowers.T #The p_nom_opt is not a timeseries. However, before we were using timeseries. 
                                # So, to make the same code work (combining similar names), we transpose it


    myloads = europe.loads_t.p
    myloads = myloads[myloads.columns[myloads.columns.str.startswith(countries)]]

    myloads = myloads-myloads.mean()
    myloads.rename(columns = lambda x: x[:2], inplace = True) #All of the loads are only named by the country
    myloads = myloads.groupby(level = 0, axis = 1).sum() #There are some duplicate loads. This groups them (adds them together)

    weekloads = myloads.rolling(56).mean()[::56]

    weekloads = weekloads.drop(0)# I did not add this line before. I wonder if this will make a difference
                                    #It did make a bit of a difference--about 1% Nothign to write home about


    #In this section, we attempt to add the covariance of solar with the cooling and heating degree days
    coolDD, heatDD = load_temp_data()

    coolDD = coolDD-coolDD.mean() #in the calculation of covariances, we want to use the difference between the amount of CDD/HDD and the mean
    heatDD = heatDD-heatDD.mean()

    coolDD = coolDD.reset_index()
    coolDD = coolDD.drop('time', axis = 1)

    heatDD = heatDD.reset_index()
    heatDD = heatDD.drop('time', axis = 1)
    heatDD = heatDD * -1

    coolweek = coolDD.rolling(56).mean()[::56]
    heatweek = heatDD.rolling(56).mean()[::56]



    # coolsum = coolsum.T



    for country in countries:
        resource_fracs = mydata
        resource_fracs = resource_fracs[[col for col in resource_fracs.columns if col.startswith(country)]]
        mydata[country + 'solar'] = resource_fracs[[col for col in resource_fracs.columns if col.endswith('solar')]].sum(axis = 1) #sums all the solar stuff together
        mydata[country + 'wind'] = resource_fracs[[col for col in resource_fracs.columns if 'wind' in col]].sum(axis = 1)
        if any('ror' in col for col in resource_fracs.columns):#checks if there is a 'ror' column present
            mydata[country + 'ror'] =  resource_fracs[[col for col in resource_fracs.columns if 'ror' in col]].sum(axis = 1)




        allsources = totalpowers
        allsources =allsources[[col for col in allsources.columns if col.startswith(country)]]
        totalpowers[country + 'solar'] = allsources[[col for col in allsources.columns if col.endswith('solar')]].sum(axis = 1) #sums all the solar stuff together
        totalpowers[country + 'wind'] = allsources[[col for col in allsources.columns if 'wind' in col]].sum(axis = 1)
        if any('ror' in col for col in allsources.columns):#checks if there is a 'ror' column present
            totalpowers[country + 'ror'] =  allsources[[col for col in allsources.columns if 'ror' in col]].sum(axis = 1)



        allstorage = mystorage
        allstorage = allstorage[[col for col in allstorage.columns if col.startswith(country)]]
        if any ('hydro' in col for col in allstorage.columns):
            mydata[country + "hydro"] = allstorage[[col for col in allstorage.columns]].sum(axis = 1) 
        

        




    mydata = mydata[mydata.columns[~mydata.columns.str.contains('[0-9]+')]]#gets rid of old columns, not needed in new code

    #print(mydata)

    totalpowers = totalpowers[totalpowers.columns[~totalpowers.columns.str.contains('[0-9]+')]]
    totalpowers = totalpowers.T


    moddata = mydata-mydata.mean()#WE SUBTRACT THE MEAN HERE

    weekdata = moddata.rolling(56).mean()[::56]
    weekdata = weekdata.drop(0)#question: we are dropping the first row because the rolling/mean combo makes the first row NaN.
    #However, what about the loads?


    covariances = pd.DataFrame()

    weekcovariances = pd.DataFrame()
    
    coolvariances = pd.DataFrame()

    coolweekvar = pd.DataFrame()

    heatvariances = pd.DataFrame()

    heatweekvar = pd.DataFrame()
    
    


    for col in myloads.columns:
        covariances[col+'solar'] =(myloads[col]-myloads[col].mean()) * (moddata[col + 'solar']-moddata[col + 'solar'].mean())/(myloads[col].std()*moddata[col + 'solar'].std())
        print(myloads[col].mean())
        print(moddata[col+'solar'].mean())
        covariances[col+ 'wind'] = (myloads[col]-myloads[col].mean()) * (moddata[col + 'wind']-moddata[col + 'wind'].mean())/(myloads[col].std()*moddata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            covariances[col + 'ror'] =  myloads[col] * moddata[col + 'ror']/(myloads[col].std()*moddata[col + 'ror'].std())


        weekcovariances[col+'solar'] = weekloads[col] * weekdata[col + 'solar']/(weekloads[col].std()*weekdata[col + 'solar'].std())
        weekcovariances[col+ 'wind'] = weekloads[col] * weekdata[col + 'wind']/(weekloads[col].std()*weekdata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            weekcovariances[col + 'ror'] =  weekloads[col] * weekdata[col + 'ror']/(weekloads[col].std()*weekdata[col + 'ror'].std())
    

        coolvariances[col + 'solar'] = coolDD[col] * moddata[col + 'solar']/(coolDD[col].std()*moddata[col + 'solar'].std())
        coolvariances[col + 'wind'] = coolDD[col] * moddata[col + 'wind']/(coolDD[col].std()*moddata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            coolvariances[col + 'ror'] =  coolDD[col] * moddata[col + 'ror']/(coolDD[col].std()*moddata[col + 'ror'].std())
        
        coolweekvar[col + 'solar'] = coolweek[col] * weekdata[col + 'solar']/(coolweek[col].std()*weekdata[col + 'solar'].std())
        coolweekvar[col + 'wind'] = coolweek[col] * weekdata[col + 'wind']/(coolweek[col].std()*weekdata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            coolweekvar[col + 'ror'] =  coolweek[col] * weekdata[col + 'ror']/(coolweek[col].std()*weekdata[col + 'ror'].std())


        

        heatvariances[col + 'solar'] = heatDD[col] * moddata[col + 'solar']/(heatDD[col].std()*moddata[col + 'solar'].std())
        heatvariances[col + 'wind'] = heatDD[col] * moddata[col + 'wind']/(heatDD[col].std()*moddata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            heatvariances[col + 'ror'] =  heatDD[col] * moddata[col + 'ror']/(heatDD[col].std()*moddata[col + 'ror'].std())



        heatweekvar[col + 'solar'] = heatweek[col] * weekdata[col + 'solar']/(heatweek[col].std()*weekdata[col + 'solar'].std())
        heatweekvar[col + 'wind'] = heatweek[col] * weekdata[col + 'wind']/(heatweek[col].std()*weekdata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            heatweekvar[col + 'ror'] =  heatweek[col] * weekdata[col + 'ror']/(heatweek[col].std()*weekdata[col + 'ror'].std())
    
    # print(weekcovariances)
    weekcovariances = weekcovariances.sum()/len(weekcovariances)
    weekcovariances.to_frame()
    # print(weekcovariances)

    covariances = covariances.sum()/len(covariances)
    covariances.to_frame()

    coolvariances = coolvariances.sum()/len(coolvariances)
    coolvariances.to_frame()

    heatvariances = heatvariances.sum()/len(heatvariances)
    heatvariances.to_frame()


    coolweekvar = coolweekvar.sum()/len(coolweekvar)
    coolweekvar.to_frame()

    heatweekvar = heatweekvar.sum()/len(heatweekvar)
    heatweekvar.to_frame()
    #print(covariances.columns)

    #In this section:
        #Bring in the relevant dataframes from temp function
        #Add to mydata: CDDs/day on average, HDDs/day on average
        #Also, covariance between CDD, HDDs
    
    




    
    mydata = mydata.sum()



    


    mydata = mydata.to_frame()

    mydata = pd.concat([mydata, totalpowers, covariances, weekcovariances, coolvariances, coolweekvar, heatvariances, heatweekvar], axis = 1)#We want to do the same thing with the
    #mydata = pd.concat([mydata, totalpowers, weekcovariances], axis = 1)

    #mydata.columns = ['0', 'p_nom_opt', 'load_corr', 'week_corr']

    #In this section of the code, I
        #Extract 
    #save csv in new file, related to filepath
    now_path = pathlib.Path(filepath)
    parent_path = now_path.parent
    run_directory = parent_path.parent


    


    mydata.to_csv(run_directory / "csvs/generators_test.csv")




    return mydata





def retrieve_generators_choice(filepath):
    '''This function takes a completed postnetwork, finds the resource frac
    
    This function is useful if there are multiple networks per run name'''
    
    europe = pypsa.Network()
    europe.import_from_netcdf(filepath)

    opts = filepath.split('-')


    if 'B' in opts:
        sectors = 'yes'
    else:
        sectors = 'no'


    trans = True

    for opt in opts:
        if opt.startswith('zero'):#If we make another wildcard with zero, that will obviously mess up this equation
            trans = False
    
    if trans == False:
        transmission = "no"
    else:
        transmission = "yes"

    firstopts = opts[0].split('_')

    solarcost = firstopts[-2]#The solar cost is the second to last opt of the first opts, so it is here we assign it
    # solarcost = float(solarcost) #We are using solarcost as an 'opt'

    my_gen = ("offwind-ac", "offwind-dc", "solar", "onwind", "ror")

    countries = eu28
    mydata = europe.generators_t.p
    mydata = mydata[mydata.columns[mydata.columns.str.startswith(countries) ]]

    #This deals with p_nom_opt
    mydata = mydata[mydata.columns[mydata.columns.str.endswith(my_gen)]]

    mystorage = europe.storage_units_t.p
    #mystorage = mystorage[mystorage.columns[mystorage.columns.str.startswith(countries) ]] #this line is actually useless for PHS because all generators start with one of the european countries
    mystorage = mystorage[mystorage.columns[mystorage.columns.str.endswith("hydro")]]


    totalpowers = europe.generators.p_nom_opt
    totalpowers = totalpowers[mydata.columns]

    totalpowers = totalpowers.to_frame()
    totalpowers = totalpowers.T #The p_nom_opt is not a timeseries. However, before we were using timeseries. 
                                # So, to make the same code work (combining similar names), we transpose it


    myloads = europe.loads_t.p
    myloads = myloads[myloads.columns[myloads.columns.str.startswith(countries)]]

    myloads = myloads-myloads.mean()
    myloads.rename(columns = lambda x: x[:2], inplace = True) #All of the loads are only named by the country
    myloads = myloads.groupby(level = 0, axis = 1).sum() #There are some duplicate loads. This groups them (adds them together)

    weekloads = myloads.rolling(56).mean()[::56]

    weekloads = weekloads.drop(0)# I did not add this line before. I wonder if this will make a difference
                                    #It did make a bit of a difference--about 1% Nothign to write home about


    #In this section, we attempt to add the covariance of solar with the cooling and heating degree days
    coolDD, heatDD = load_temp_data()

    coolDD = coolDD-coolDD.mean() #in the calculation of covariances, we want to use the difference between the amount of CDD/HDD and the mean
    heatDD = heatDD-heatDD.mean()

    coolDD = coolDD.reset_index()
    coolDD = coolDD.drop('time', axis = 1)

    heatDD = heatDD.reset_index()
    heatDD = heatDD.drop('time', axis = 1)
    heatDD = heatDD * -1

    coolweek = coolDD.rolling(56).mean()[::56]
    heatweek = heatDD.rolling(56).mean()[::56]



    # coolsum = coolsum.T



    for country in countries:
        resource_fracs = mydata
        resource_fracs = resource_fracs[[col for col in resource_fracs.columns if col.startswith(country)]]
        mydata[country + 'solar'] = resource_fracs[[col for col in resource_fracs.columns if col.endswith('solar')]].sum(axis = 1) #sums all the solar stuff together
        mydata[country + 'wind'] = resource_fracs[[col for col in resource_fracs.columns if 'wind' in col]].sum(axis = 1)
        if any('ror' in col for col in resource_fracs.columns):#checks if there is a 'ror' column present
            mydata[country + 'ror'] =  resource_fracs[[col for col in resource_fracs.columns if 'ror' in col]].sum(axis = 1)




        allsources = totalpowers
        allsources =allsources[[col for col in allsources.columns if col.startswith(country)]]
        totalpowers[country + 'solar'] = allsources[[col for col in allsources.columns if col.endswith('solar')]].sum(axis = 1) #sums all the solar stuff together
        totalpowers[country + 'wind'] = allsources[[col for col in allsources.columns if 'wind' in col]].sum(axis = 1)
        if any('ror' in col for col in allsources.columns):#checks if there is a 'ror' column present
            totalpowers[country + 'ror'] =  allsources[[col for col in allsources.columns if 'ror' in col]].sum(axis = 1)

        
        allstorage = mystorage
        allstorage = allstorage[[col for col in allstorage.columns if col.startswith(country)]]
        if any ('hydro' in col for col in allstorage.columns):
            mydata[country + "hydro"] = allstorage[[col for col in allstorage.columns]].sum(axis = 1) 
        




    mydata = mydata[mydata.columns[~mydata.columns.str.contains('[0-9]+')]]#gets rid of old columns, not needed in new code


    totalpowers = totalpowers[totalpowers.columns[~totalpowers.columns.str.contains('[0-9]+')]]
    totalpowers = totalpowers.T


    moddata = mydata-mydata.mean()

    weekdata = moddata.rolling(56).mean()[::56]
    weekdata = weekdata.drop(0)#question: we are dropping the first row because the rolling/mean combo makes the first row NaN.
    #However, what about the loads?


    covariances = pd.DataFrame()

    weekcovariances = pd.DataFrame()
    
    coolvariances = pd.DataFrame()

    coolweekvar = pd.DataFrame()

    heatvariances = pd.DataFrame()

    heatweekvar = pd.DataFrame()
    
    


    for col in myloads.columns:
        covariances[col+'solar'] = myloads[col] * moddata[col + 'solar']/(myloads[col].std()*moddata[col + 'solar'].std())
        covariances[col+ 'wind'] = myloads[col] * moddata[col + 'wind']/(myloads[col].std()*moddata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            covariances[col + 'ror'] =  myloads[col] * moddata[col + 'ror']/(myloads[col].std()*moddata[col + 'ror'].std())


        weekcovariances[col+'solar'] = weekloads[col] * weekdata[col + 'solar']/(weekloads[col].std()*weekdata[col + 'solar'].std())
        weekcovariances[col+ 'wind'] = weekloads[col] * weekdata[col + 'wind']/(weekloads[col].std()*weekdata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            weekcovariances[col + 'ror'] =  weekloads[col] * weekdata[col + 'ror']/(weekloads[col].std()*weekdata[col + 'ror'].std())
    

        coolvariances[col + 'solar'] = coolDD[col] * moddata[col + 'solar']/(coolDD[col].std()*moddata[col + 'solar'].std())
        coolvariances[col + 'wind'] = coolDD[col] * moddata[col + 'wind']/(coolDD[col].std()*moddata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            coolvariances[col + 'ror'] =  coolDD[col] * moddata[col + 'ror']/(coolDD[col].std()*moddata[col + 'ror'].std())
        
        coolweekvar[col + 'solar'] = coolweek[col] * weekdata[col + 'solar']/(coolweek[col].std()*weekdata[col + 'solar'].std())
        coolweekvar[col + 'wind'] = coolweek[col] * weekdata[col + 'wind']/(coolweek[col].std()*weekdata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            coolweekvar[col + 'ror'] =  coolweek[col] * weekdata[col + 'ror']/(coolweek[col].std()*weekdata[col + 'ror'].std())


        

        heatvariances[col + 'solar'] = heatDD[col] * moddata[col + 'solar']/(heatDD[col].std()*moddata[col + 'solar'].std())
        heatvariances[col + 'wind'] = heatDD[col] * moddata[col + 'wind']/(heatDD[col].std()*moddata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            heatvariances[col + 'ror'] =  heatDD[col] * moddata[col + 'ror']/(heatDD[col].std()*moddata[col + 'ror'].std())



        heatweekvar[col + 'solar'] = heatweek[col] * weekdata[col + 'solar']/(heatweek[col].std()*weekdata[col + 'solar'].std())
        heatweekvar[col + 'wind'] = heatweek[col] * weekdata[col + 'wind']/(heatweek[col].std()*weekdata[col + 'wind'].std())
        if col + 'ror' in moddata.columns:
            heatweekvar[col + 'ror'] =  heatweek[col] * weekdata[col + 'ror']/(heatweek[col].std()*weekdata[col + 'ror'].std())
    
    # print(weekcovariances)
    weekcovariances = weekcovariances.sum()/len(weekcovariances)
    weekcovariances.to_frame()
    # print(weekcovariances)

    covariances = covariances.sum()/len(covariances)
    covariances.to_frame()

    coolvariances = coolvariances.sum()/len(coolvariances)
    coolvariances.to_frame()

    heatvariances = heatvariances.sum()/len(heatvariances)
    heatvariances.to_frame()


    coolweekvar = coolweekvar.sum()/len(coolweekvar)
    coolweekvar.to_frame()

    heatweekvar = heatweekvar.sum()/len(heatweekvar)
    heatweekvar.to_frame()
    #print(covariances.columns)

    #In this section:
        #Bring in the relevant dataframes from temp function
        #Add to mydata: CDDs/day on average, HDDs/day on average
        #Also, covariance between CDD, HDDs
    
    




    
    mydata = mydata.sum()


    


    mydata = mydata.to_frame()

    mydata = pd.concat([mydata, totalpowers, covariances, weekcovariances, coolvariances, coolweekvar, heatvariances, heatweekvar], axis = 1)#We want to do the same thing with the
    #mydata = pd.concat([mydata, totalpowers, weekcovariances], axis = 1)

    #mydata.columns = ['0', 'p_nom_opt', 'load_corr', 'week_corr']

    #In this section of the code, I
        #Extract 
    #save csv in new file, related to filepath
    now_path = pathlib.Path(filepath)
    parent_path = now_path.parent
    run_directory = parent_path.parent


    
    csvpath =  "csvs/" + solarcost + "_generators_"+ sectors + "_sectors_" + transmission + "_transmission" + ".csv"

    mydata.to_csv(run_directory /csvpath)




    return mydata




def add_to_df(run_name):
    #This takes in a csv made from retrieve_generators and then adds to it
    csvpath = "results/" + run_name + "/csvs/generators_T.csv"
    all_gens = pd.read_csv(csvpath)
    all_gens.columns = ['name', "Generation", 'p_nom_opt', 'load_corr', 'week_corr', 'cool_corr', 'week_cool', 'heat_corr', 'week_heat']#There is an extra column here than you might think, because importing from csv turns what was the index into a column. There is probably a way to avoid this but oh well :)
    all_gens['country'] = all_gens.apply(lambda row: row['name'][0:2], axis = 1)
    #all_gens['country'] = all_gens.apply(lambda row: row['name'][0:2], axis = 1)#This makes a new column, lops off the first two letters of each row name, which is the country
    all_gens['carrier'] = all_gens.apply(lambda row: row['name'].replace(row['country'], ''), axis = 1)#This makes an ew column replaces the country in the name with just the resource
    #all_gens = all_gens.rename(columns = {"0": "Generation"})#renames the column of production from "0" to "Generation"
    all_gens["total_gen"] = '' #Makes an empty column
    for country in eu28:
        f = all_gens.loc[all_gens['country'] == country]
        all_gens['total_gen'].loc[all_gens['country'] == country] = f['Generation'].sum()

    all_gens['fraction'] = all_gens.apply(lambda row: 0 if row['total_gen'] == 0 else row['Generation']/row['total_gen'], axis = 1)

    latitude = pd.read_csv("data/countries_lat.csv")
    all_gens = pd.merge(all_gens, latitude[['country', 'latitude', 'name']], how = 'left', on = 'country' )

    HDDCDD = pd.read_csv('data/avgHDDCDD.csv')#Here, this adds the avg HDD and CDD for each country
    HDDCDD.columns = ["country", 'avgCDD', 'avgHDD']

    all_gens = pd.merge(all_gens, HDDCDD, how = 'left', on = 'country')
    all_gens['year_CF'] = all_gens.apply(lambda row: 0 if row['p_nom_opt'] == 0 else row['Generation']/(row['p_nom_opt'] * 2920), axis = 1)

    all_gen_solar = all_gens.loc[all_gens['carrier'] == 'solar']
    all_gen_solar['solarfrac'] = all_gen_solar ['fraction']
    all_gen_solar['solarcf'] = all_gen_solar['year_CF']
    all_gens = pd.merge(all_gens, all_gen_solar[['country','solarfrac', 'solarcf']], how = 'left', on = 'country')
    #create empty column
    #use loc 
    nowpath = pathlib.Path(csvpath)
    parentpath = nowpath.parent

    #print(all_gens)
    all_gens.to_csv(parentpath / 'gen_and_lat.csv')

    latcsv_path = parentpath / 'gen_and_lat.csv'

    return(latcsv_path)


def add_to_df_choice(filepath):
    #This takes in a csv made from retrieve_generators and then adds to it
    opts = filepath.split('_')#hm not so simple because we are looking at the csv itself[[
    print(opts)
    filename = os.path.basename(filepath)
    priceopt = filename.split("_")
    solarcost = priceopt[0]
    if "sectors" in opts:#
        idxsec = opts.index('sectors')
        idxsec -= 1
        sec = opts[idxsec]
        print(sec)
        # if sec == "yes":
        #     sec = 'with'
        # else:
        #     sec = 'without'
        # sec = sec + "_sectors"
    if "transmission.csv" in opts:
        idxtrans = opts.index('transmission.csv')
        idxtrans -= 1
        trans = opts[idxtrans]

        # if trans == "yes":
        #     trans = 'with'
        # else:
        #     trans = 'without'
        # trans = trans + "_transmission"    


    all_gens = pd.read_csv(filepath)
    all_gens.columns = ['name', "Generation", 'p_nom_opt', 'load_corr', 'week_corr', 'cool_corr', 'week_cool', 'heat_corr', 'week_heat']#There is an extra column here than you might think, because importing from csv turns what was the index into a column. There is probably a way to avoid this but oh well :)
    all_gens['country'] = all_gens.apply(lambda row: row['name'][0:2], axis = 1)
    #all_gens['country'] = all_gens.apply(lambda row: row['name'][0:2], axis = 1)#This makes a new column, lops off the first two letters of each row name, which is the country
    all_gens['carrier'] = all_gens.apply(lambda row: row['name'].replace(row['country'], ''), axis = 1)#This makes an ew column replaces the country in the name with just the resource
    #all_gens = all_gens.rename(columns = {"0": "Generation"})#renames the column of production from "0" to "Generation"
    all_gens["total_gen"] = '' #Makes an empty column
    for country in eu28:
        f = all_gens.loc[all_gens['country'] == country]
        all_gens['total_gen'].loc[all_gens['country'] == country] = f['Generation'].sum()

    all_gens['fraction'] = all_gens.apply(lambda row: 0 if row['total_gen'] == 0 else row['Generation']/row['total_gen'], axis = 1)

    latitude = pd.read_csv("data/countries_lat.csv")
    all_gens = pd.merge(all_gens, latitude[['country', 'latitude', 'name']], how = 'left', on = 'country' )

    HDDCDD = pd.read_csv('data/avgHDDCDD.csv')#Here, this adds the avg HDD and CDD for each country
    HDDCDD.columns = ["country", 'avgCDD', 'avgHDD']

    all_gens = pd.merge(all_gens, HDDCDD, how = 'left', on = 'country')
    all_gens['year_CF'] = all_gens.apply(lambda row: 0 if row['p_nom_opt'] == 0 else row['Generation']/(row['p_nom_opt'] * 2920), axis = 1)

    all_gen_solar = all_gens.loc[all_gens['carrier'] == 'solar']
    all_gen_solar['solarfrac'] = all_gen_solar ['fraction']
    all_gen_solar['solarcf'] = all_gen_solar['year_CF']
    all_gens = pd.merge(all_gens, all_gen_solar[['country','solarfrac', 'solarcf']], how = 'left', on = 'country')
    #create empty column
    #use loc 
    nowpath = pathlib.Path(filepath)
    parentpath = nowpath.parent.parent#this is the csvs folder


    #print(all_gens)

    allgenscsv =   "csvs/" + solarcost + "_gen_and_lat_"+ sec + "_sectors_" + trans + "_transmission" + ".csv"
    all_gens.to_csv(parentpath /allgenscsv)

    latcsv_path = parentpath / allgenscsv

    return(latcsv_path)

#%%

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2

#%%
def compare_cost_lat():

    opts = mypath.split("_")
    print(opts)
    for opt in opts:
        contains_digit = len(re.findall('\d+', opt)) > 0
        if contains_digit:
            f = re.findall(r'\d+', opt)
            hrs = f[0]
            hrs = hrs + "h"
    if "sectors" in opts:
        idxsec = opts.index('sectors')
        idxsec -= 1
        sec = opts[idxsec]
        if sec == "yes":
            sec = 'with'
        else:
            sec = 'without'
        sec = sec + " sectors"
    if "transmission" in opts:
        idxtrans = opts.index('transmission')
        idxtrans -= 1
        trans = opts[idxtrans]
        if trans == "yes":
            trans = 'with'
        else:
            trans = 'without'
        trans = trans + " transmission"

    
    

    plt.rcdefaults()
    solar_latdf = pd.read_csv(path)
    solar_latdf = solar_latdf.iloc[:, 1:] #get rid of weird second index
    solar_latdf = solar_latdf[solar_latdf['carrier'] == 'solar']#only interested in solar share
    solar_latdf = solar_latdf.iloc[:-2, :] #get rid of MT and CY, which have 0 according to our model
    solar_latdf['percent'] = solar_latdf["fraction"] * 100
    
    
    #plotting: latitude on x axis, solar fraction on y axis

    fig, ax = plt.subplots()

    x = solar_latdf["latitude"]
    y = solar_latdf["percent"]


    ax.scatter(x, y)
    ax.set_xlabel("Latitude (degrees)")
    ax.set_ylabel("Optimal solar share (%)")
    ax.set_title("Optimal solar share, " + sec + " and " + trans + " " + hrs)



    for idx, row in solar_latdf.iterrows():
        ax.annotate(row['country'], (row['latitude']* 1.007, row['percent']* 0.97))
    

    m, b = np.polyfit(x, y, 1)
    #ax.axline(xy1 = (0, b), slope = m, color = 'r', label=f'$y = {m:.2f}x {b:+.2f}$')



    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label=f'$y = {m:.2f}x {b:+.2f}$')
    
    #ax.annotate("r-squared = {:.3f}".format(r2_score(x, y)), (60, 100))


    fig.legend()
    parentpath = path.parent.parent

    fig.savefig(parentpath / "graphs/solar_by_lat")

    plt.show()
    



def solar_by_latitude(path):
    mypath = str(path)
    
    opts = mypath.split("_")
    print(opts)
    for opt in opts:
        contains_digit = len(re.findall('\d+', opt)) > 0
        if contains_digit:
            f = re.findall(r'\d+', opt)
            hrs = f[0]
            hrs = hrs + "h"
    if "sectors" in opts:
        idxsec = opts.index('sectors')
        idxsec -= 1
        sec = opts[idxsec]
        if sec == "yes":
            sec = 'with'
        else:
            sec = 'without'
        sec = sec + " sectors"
    if "transmission" in opts:
        idxtrans = opts.index('transmission')
        idxtrans -= 1
        trans = opts[idxtrans]
        if trans == "yes":
            trans = 'with'
        else:
            trans = 'without'
        trans = trans + " transmission"

    
    

    plt.rcdefaults()
    solar_latdf = pd.read_csv(path)
    solar_latdf = solar_latdf.iloc[:, 1:] #get rid of weird second index
    solar_latdf = solar_latdf[solar_latdf['carrier'] == 'solar']#only interested in solar share
    solar_latdf = solar_latdf.iloc[:-2, :] #get rid of MT and CY, which have 0 according to our model
    solar_latdf['percent'] = solar_latdf["fraction"] * 100
    
    
    #plotting: latitude on x axis, solar fraction on y axis

    fig, ax = plt.subplots()

    x = solar_latdf["latitude"]
    y = solar_latdf["percent"]


    ax.scatter(x, y)
    ax.set_xlabel("Latitude (degrees)")
    ax.set_ylabel("Optimal solar share (%)")
    ax.set_title("Optimal solar share, " + sec + " and " + trans + " " + hrs)



    for idx, row in solar_latdf.iterrows():
        ax.annotate(row['country'], (row['latitude']* 1.007, row['percent']* 0.97))
    

    m, b = np.polyfit(x, y, 1)
    #ax.axline(xy1 = (0, b), slope = m, color = 'r', label=f'$y = {m:.2f}x {b:+.2f}$')



    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label=f'$y = {m:.2f}x {b:+.2f}$')
    
    #ax.annotate("r-squared = {:.3f}".format(r2_score(x, y)), (60, 100))


    fig.legend()
    parentpath = path.parent.parent

    fig.savefig(parentpath / "graphs/solar_by_lat")

    plt.show()
#%%
def solar_by_latitude_comparecost(path, ax):
    '''The goal of this function is to plot solar share on the y axis and country ordered by
    latitude on the x axis. Then, we will plot the three different costs'''
    mypath = str(path)
    
    opts = mypath.split("_")
    print(opts)
    for opt in opts:
        contains_digit = len(re.findall('\d+', opt)) > 0
        if contains_digit:
            f = re.findall(r'\d+', opt)
            hrs = f[0]
            hrs = hrs + "h"
    if "sectors" in opts:
        idxsec = opts.index('sectors')
        idxsec -= 1
        sec = opts[idxsec]
        if sec == "yes":
            sec = 'with'
            searchsec = 'yes'
        else:
            sec = 'without'
            searchsec = 'no'
        sec = sec + " sectors"
    if "transmission" in opts:
        idxtrans = opts.index('transmission')
        idxtrans -= 1
        trans = opts[idxtrans]
        if trans == "yes":
            trans = 'with'
            searchtrans = 'yes'
        else:
            trans = 'without'
            searchtrans = 'no'
        trans = trans + " transmission"

    
    

    plt.rcdefaults()
    solar_latdf = pd.read_csv(path)
    solar_latdf = solar_latdf.iloc[:, 1:] #get rid of weird second index
    solar_latdf = solar_latdf[solar_latdf['carrier'] == 'solar']#only interested in solar share
    solar_latdf = solar_latdf.iloc[:-2, :] #get rid of MT and CY, which have 0 according to our model
    solar_latdf['percent'] = solar_latdf["fraction"] * 100
    solar_latdf = solar_latdf.sort_values(by = ['latitude'])

    #This is the first cheap path
    costs_path = "results/adam_latitude_compare_anysectors_anytransmission_3h_futcost3/csvs/"
    csvfile_cheap = '0.25_gen_and_lat_'+ searchsec + '_sectors_' + searchtrans + '_transmission.csv'

    total_path = costs_path + csvfile_cheap
    csvcheap_path = pathlib.Path(total_path)

    solar_latdf_cheap = pd.read_csv(csvcheap_path)
    solar_latdf_cheap = solar_latdf_cheap.iloc[:, 1:] #get rid of weird second index
    solar_latdf_cheap = solar_latdf_cheap[solar_latdf_cheap['carrier'] == 'solar']#only interested in solar share
    solar_latdf_cheap = solar_latdf_cheap.iloc[:-2, :] #get rid of MT and CY, which have 0 according to our model
    solar_latdf_cheap['percent'] = solar_latdf_cheap["fraction"] * 100
    solar_latdf_cheap = solar_latdf_cheap.sort_values(by = ['latitude'])
    

    #This is the cheapest path
    csvfile_cheapest = '0.036_gen_and_lat_'+ searchsec + '_sectors_' + searchtrans + '_transmission.csv'
    total_path2 = costs_path + csvfile_cheapest
    csvcheapest_path = pathlib.Path(total_path2)

    
    solar_latdf_cheapest = pd.read_csv(csvcheapest_path)
    solar_latdf_cheapest = solar_latdf_cheapest.iloc[:, 1:] #get rid of weird second index
    solar_latdf_cheapest = solar_latdf_cheapest[solar_latdf_cheapest['carrier'] == 'solar']#only interested in solar share
    solar_latdf_cheapest = solar_latdf_cheapest.iloc[:-2, :] #get rid of MT and CY, which have 0 according to our model
    solar_latdf_cheapest['percent'] = solar_latdf_cheapest["fraction"] * 100
    solar_latdf_cheapest = solar_latdf_cheapest.sort_values(by = ['latitude'])




    
    
    #plotting: latitude on x axis, solar fraction on y axis

    #It is here that we comment out the fig, ax bc we are already passing to a different axis    
    # fig, ax = plt.subplots(figsize = (10, 4.8))

    x = solar_latdf["country"]
    y = solar_latdf["percent"]

    x1 = solar_latdf_cheap["country"]
    y1 = solar_latdf_cheap["percent"]

    x2 = solar_latdf_cheapest["country"]
    y2 = solar_latdf_cheapest["percent"]

    ax.scatter(x, y, label = 'default', color = 'C0')
    ax.scatter(x1, y1, label = 'less optimistic', color = 'C1')
    ax.scatter(x2, y2, label = 'optimistic', color = 'C2')
    ax.grid(True)
    # ax.set_xlabel("Country sorted by latitude")
    ax.set_ylabel("Optimal solar share (%)")
    # ax.set_title("Optimal solar share, " + sec + " and " + trans + " " + hrs)



    # for idx, row in solar_latdf.iterrows():
    #     ax.annotate(row['country'], (row['latitude']* 1.007, row['percent']* 0.97))
    

    # m, b = np.polyfit(x, y, 1)
    #ax.axline(xy1 = (0, b), slope = m, color = 'r', label=f'$y = {m:.2f}x {b:+.2f}$')



    #plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label=f'$y = {m:.2f}x {b:+.2f}$')
    
    #ax.annotate("r-squared = {:.3f}".format(r2_score(x, y)), (60, 100))

    handles, labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    # fig.legend(handles, labels)
    # parentpath = path.parent.parent

    #fig.savefig(parentpath / "graphs/solar_by_lat")

    #No need to show 
    #plt.show()
  
#solar_by_latitude_comparecost("results/adam_latitude_compare_no_sectors_no_transmission_3h3/csvs/gen_and_lat.csv", )
def four_latitude_comparecost():
    '''The goal of this function is to plot four of any given function (in this case solar_by_latitude_comparecost) in a grid
    such that we can see the appropriate differences between sectors/no sectors, '''
    run0 = "adam_latitude_compare_no_sectors_yes_transmission_3h"
    run1 = "adam_latitude_compare_yes_sectors_yes_transmission_3h2"
    run2 = "adam_latitude_compare_no_sectors_no_transmission_3h3"
    run3 = "adam_latitude_compare_yes_sectors_no_transmission_3h"
    path0 = 'results/' + run0 + '/csvs/gen_and_lat.csv'
    path1 = 'results/' + run1 + '/csvs/gen_and_lat.csv'
    path2 = 'results/' + run2 + '/csvs/gen_and_lat.csv'
    path3 = 'results/' + run3 + '/csvs/gen_and_lat.csv'
    path0 = pathlib.Path(path0)
    path1 = pathlib.Path(path1)
    path2 = pathlib.Path(path2)
    path3 = pathlib.Path(path3)


    fs = 20
    plt.rcParams['axes.labelsize'] = fs
    plt.rcParams['xtick.labelsize'] = fs
    plt.rcParams['ytick.labelsize'] = fs

    fig,ax = plt.subplots(2,2,figsize=(13,7),sharex=True,sharey='row')
    ax = ax.flatten()
    solar_by_latitude_comparecost(path0, ax[0])
    solar_by_latitude_comparecost(path1, ax[1])
    solar_by_latitude_comparecost(path2, ax[2])
    solar_by_latitude_comparecost(path3, ax[3])

    fig.supxlabel(r"$\bf{Sectors}$",fontsize=fs, y = 0.11, x = 0.53)
    fig.supylabel (r"$\bf{Transmission}$",fontweight="bold",fontsize=fs, y = 0.6)
    ax[3].set_xlabel(" ",fontsize=fs)
    #ax[2].set_xlabel(r"$\bf{No}$",fontsize=fs)
    #ax[0].set_ylabel(r"$\bf{Yes}$" + '\nSolar Percent', fontsize = fs)
    #ax[2].set_ylabel(r"$\bf{No}$" + '\nSolar Percent', fontsize = fs)
    ax[2].tick_params(axis='x', labelrotation =90, labelsize = fs-6)
    ax[3].tick_params(axis='x', labelrotation = 90, labelsize = fs-6)

    ax[1].set_ylabel("")
    ax[3].set_ylabel("")
    ax[2].set_ylabel("")
    ax[0].set_ylabel("Solar Share", y = -0.1)
    ax[2].set_title("")

    ax[3].set_title("")

    handles1, labels1 = ax[1].get_legend_handles_labels()

    fig.legend(handles1, labels1, prop={'size':fs}, ncol=3, loc = (0.25, 0.03))
    fig.tight_layout(rect = [0.02, 0.1, 1, 1])

    fig.text(0.325, 0.12, "No", fontweight= "bold", fontsize = fs)
    fig.text(0.76, 0.12, "Yes", fontweight= "bold", fontsize = fs)

    fig.text(0.02, 0.81, "Yes", fontweight= "bold", rotation = 90, fontsize = fs)
    fig.text(0.02, 0.33, "No", fontweight= "bold", rotation = 90, fontsize = fs)

    fig.text(0.41, 0.18, "Countries ordered by latitude",fontsize=fs)
    plt.subplots_adjust(wspace=0.02, hspace=0.05)

    #plt.savefig("Images/Paper/fourlatitude_compare.pdf")

    plt.show()
#four_latitude_comparecost()

    # plt.show()
#%%

def solar_by_solar(path, ax):
    mypath = str(path)
    
    opts = mypath.split("_")
    print(opts)
    for opt in opts:
        contains_digit = len(re.findall('\d+', opt)) > 0
        if contains_digit:
            f = re.findall(r'\d+', opt)
            hrs = f[0]
            hrs = hrs + "h"
    if "sectors" in opts:
        idxsec = opts.index('sectors')
        idxsec -= 1
        sec = opts[idxsec]
        if sec == "yes":
            sec = 'with'
        else:
            sec = 'without'
        sec = sec + " sectors"
    if "transmission" in opts:
        idxtrans = opts.index('transmission')
        idxtrans -= 1
        trans = opts[idxtrans]
        if trans == "yes":
            trans = 'with'
        else:
            trans = 'without'
        trans = trans + " transmission"

    
    

    plt.rcdefaults()
    solar_latdf = pd.read_csv(path)
    solar_latdf = solar_latdf.iloc[:, 1:] #get rid of weird second index
    solar_latdf = solar_latdf[solar_latdf['carrier'] == 'solar']#only interested in wind share
    solar_latdf = solar_latdf.iloc[:-2, :] #get rid of MT and CY, which have 0 according to our model
    #solar_latdf['windpercent'] = solar_latdf["fraction"] * 100
    solar_latdf['solarpercent'] = solar_latdf['solarfrac'] * 100
    
    
    #plotting: latitude on x axis, solar fraction on y axis

    # fig, ax = plt.subplots()

    x = solar_latdf["solarcf"]
    y = solar_latdf["solarpercent"]


    ax.scatter(x, y)
    #ax.set_xlabel("Solar capacity factor")
    ax.set_ylabel("Optimal solar share (%)")
    #ax.set_title("Optimal solar share, " + sec + " and " + trans + " " + hrs)


    
    for idx, row in solar_latdf.iterrows():
        ax.annotate(row['country'], (row['solarcf']* 1.007, row['solarpercent']* 0.97), fontsize = 15)
    


    m, b = np.polyfit(x, y, 1)
    #ax.axline(xy1 = (0, b), slope = m, color = 'r', label=f'$y = {m:.2f}x {b:+.2f}$')



    ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label=f'$y = {m:.2f}x {b:+.2f}$')
    
    r_sq = rsquared(x, y)
    #ax.text(0, 1.1, "r² = {:.3f}".format(r_sq), transform = ax.transAxes)
    #ax.annotate("r-squared = {:.3f}".format(r2_score(x, y)), (60, 100))


    #fig.legend()
    handles, labels = ax.get_legend_handles_labels()
    labels[0] = "r² = {:.3f}".format(r_sq)
    ax.legend(handles, [labels[0]])
    ax.legend().set_visible(False)
    
    parentpath = path.parent.parent#path is csv, parent is csv folder, parent is run

    #fig.savefig(parentpath / "graphs/solar_by_solarcf") #subdir graphs

    parentpath = parentpath.parent.parent #parentpath is run folder, parent is results, parent is pypsaproject

    return labels
    #fig.savefig(parentpath / f"Images/Shown images/11:5 meeting/solar_by_solarcf{sec}_{trans}")
    

    #plt.show()



def solar_by_solar_all():
    run0 = "adam_latitude_compare_no_sectors_yes_transmission_3h"
    run1 = "adam_latitude_compare_yes_sectors_yes_transmission_3h2"
    run2 = "adam_latitude_compare_no_sectors_no_transmission_3h3"
    run3 = "adam_latitude_compare_yes_sectors_no_transmission_3h"
    path0 = 'results/' + run0 + '/csvs/gen_and_lat.csv'
    path1 = 'results/' + run1 + '/csvs/gen_and_lat.csv'
    path2 = 'results/' + run2 + '/csvs/gen_and_lat.csv'
    path3 = 'results/' + run3 + '/csvs/gen_and_lat.csv'
    path0 = pathlib.Path(path0)
    path1 = pathlib.Path(path1)
    path2 = pathlib.Path(path2)
    path3 = pathlib.Path(path3)
    
    plt.rcParams.update({'font.size': 18})
    fs = 23
    plt.rcParams['axes.labelsize'] = fs
    plt.rcParams['xtick.labelsize'] = fs
    plt.rcParams['ytick.labelsize'] = fs

    fig,ax = plt.subplots(2,2,figsize=(13,9),sharex=True,sharey='row')
    ax = ax.flatten()
    #solar_by_latitude_comparecost(path0, ax[0]) #We know that solar_by_latitude_comparecost is also compatible

    labels0 = solar_by_solar(path0, ax[0])
    labels1 = solar_by_solar(path1, ax[1])
    
    labels2 = solar_by_solar(path2, ax[2])
    
    labels3 = solar_by_solar(path3, ax[3])
    ax[1].lines[-1].set_color('C1')
    ax[2].lines[-1].set_color('C2')
    ax[3].lines[-1].set_color('C3')

    ax0points = [child for child in ax[0].get_children() if isinstance(child, matplotlib.text.Annotation)]
    adjust_text(ax0points, ax = ax[0])
    ax1points = [child for child in ax[1].get_children() if isinstance(child, matplotlib.text.Annotation)]
    adjust_text(ax1points, ax = ax[1])
    ax2points = [child for child in ax[2].get_children() if isinstance(child, matplotlib.text.Annotation)]
    adjust_text(ax2points, ax = ax[2])
    ax3points = [child for child in ax[3].get_children() if isinstance(child, matplotlib.text.Annotation)]
    adjust_text(ax3points, ax = ax[3])    


    fig.supxlabel(r"$\bf{Sectors}$" ,fontsize=fs, y = 0.11, x = 0.53)
    fig.supylabel ( r"$\bf{Transmission}$",fontsize=fs, y = 0.53)
    ax[3].set_xlabel(" ",fontsize=fs)
    #ax[2].set_xlabel(r"$\bf{No}$",fontsize=fs, y = 0)
    #plt.gca().lines[-].set_color('C2')
    ax[0].set_ylabel("Solar share" , fontsize = fs, y = -0.1)
    #ax[2].set_ylabel(r"$\bf{No}$", fontsize = fs, x = 0.1)
    ax[2].tick_params(axis='x', labelsize = fs-2, rotation = 40)
    ax[3].tick_params(axis='x', labelsize = fs-2, rotation = 40)
    for anax in ax:
        anax.xaxis.set_tick_params(length = 4, width = 2)
        anax.yaxis.set_tick_params(length = 4, width = 2)


    ax[0].set_title("")
    ax[2].set_ylabel("")
    ax[1].set_title("")
    #plt.gca().lines[-3].set_color('C1')
    ax[1].set_ylabel("")
    ax[3].set_ylabel("")
    ax[2].set_title("")
    
    #ax[0].get_lines().set_color('black')



    ax[3].set_title("")

    handles0, labels = ax[0].get_legend_handles_labels()
    handles1, labels = ax[1].get_legend_handles_labels()
    handles2, labels = ax[2].get_legend_handles_labels()
    handles3, labels = ax[3].get_legend_handles_labels()

    handles = handles0 + handles2 + handles1 + handles3
    labels = labels0 + labels2 + labels1 + labels3

    fig.legend(handles, labels, prop={'size':fs-1}, ncol=2, loc = (0.28, 0.01))
    fig.tight_layout(rect = [0.03, 0.1, 1, 0.9])
    plt.subplots_adjust(wspace = 0.02, hspace = 0.05)
    #fig.suptitle("Solar share by solar capacity factor", x = 0.55, fontsize = fs, weight = 'bold')
    fig.text(0.325, 0.12, "No", fontweight= "bold", fontsize = fs)
    fig.text(0.76, 0.12, "Yes", fontweight= "bold", fontsize = fs)

    fig.text(0.02, 0.73, "Yes", fontweight= "bold", rotation = 90, fontsize = fs)
    fig.text(0.02, 0.33, "No", fontweight= "bold", rotation = 90, fontsize = fs)

    fig.text(0.43, 0.16, "Capacity factor of Solar",fontsize=fs)


    plt.savefig("Images/Paper/cap_factor_solar.pdf")
    plt.savefig("Images/Paper/cap_factor_solar.png", dpi = 600)


    plt.show()
solar_by_solar_all()
#%%

def find_solar_share(path):
    df0 = pd.read_csv(path)
    tot = df0['Generation'].sum()
    df0sol = df0.query('carrier == "solar"')
    soltot = df0sol['Generation'].sum()
    print(soltot/tot)


def solar_share_tot():
    run0 = "adam_latitude_compare_no_sectors_yes_transmission_3h"
    run1 = "adam_latitude_compare_yes_sectors_yes_transmission_3h2"
    run2 = "adam_latitude_compare_no_sectors_no_transmission_3h3"
    run3 = "adam_latitude_compare_yes_sectors_no_transmission_3h"
    path0 = 'results/' + run0 + '/csvs/gen_and_lat.csv'
    path1 = 'results/' + run1 + '/csvs/gen_and_lat.csv'
    path2 = 'results/' + run2 + '/csvs/gen_and_lat.csv'
    path3 = 'results/' + run3 + '/csvs/gen_and_lat.csv'
    path0 = pathlib.Path(path0)
    path1 = pathlib.Path(path1)
    path2 = pathlib.Path(path2)
    path3 = pathlib.Path(path3)
    
    find_solar_share(path0)
    find_solar_share(path1)
    find_solar_share(path2)
    find_solar_share(path3)


#solar_share_tot()

#%%
def solar_by_wind(path, ax):
    mypath = str(path)
    
    opts = mypath.split("_")
    print(opts)
    for opt in opts:
        contains_digit = len(re.findall('\d+', opt)) > 0
        if contains_digit:
            f = re.findall(r'\d+', opt)
            hrs = f[0]
            hrs = hrs + "h"
    if "sectors" in opts:
        idxsec = opts.index('sectors')
        idxsec -= 1
        sec = opts[idxsec]
        if sec == "yes":
            sec = 'with'
        else:
            sec = 'without'
        sec = sec + " sectors"
    if "transmission" in opts:
        idxtrans = opts.index('transmission')
        idxtrans -= 1
        trans = opts[idxtrans]
        if trans == "yes":
            trans = 'with'
        else:
            trans = 'without'
        trans = trans + " transmission"

    
    

    plt.rcdefaults()
    solar_latdf = pd.read_csv(path)
    solar_latdf = solar_latdf.iloc[:, 1:] #get rid of weird second index
    solar_latdf = solar_latdf[solar_latdf['carrier'] == 'wind']#only interested in wind share
    solar_latdf = solar_latdf.iloc[:-2, :] #get rid of MT and CY, which have 0 according to our model
    #solar_latdf['windpercent'] = solar_latdf["fraction"] * 100
    solar_latdf['solarpercent'] = solar_latdf['solarfrac'] * 100
    
    
    #plotting: latitude on x axis, solar fraction on y axis

    # fig, ax = plt.subplots(figsize = (6.4, 9.6))
  

    x = solar_latdf["year_CF"]
    y = solar_latdf["solarpercent"]


    ax.scatter(x, y)
    #ax.set_xlabel("Wind capacity factor")
    #ax.set_ylabel("Optimal solar share (%)")
    #ax.set_title("Optimal solar share, " + sec + " and " + trans + " " + hrs)



    for idx, row in solar_latdf.iterrows():
        ax.annotate(row['country'], (row['year_CF']* 1.007, row['solarpercent']* 0.97), fontsize = 15)
    

    m, b = np.polyfit(x, y, 1)
    #ax.axline(xy1 = (0, b), slope = m, color = 'r', label=f'$y = {m:.2f}x {b:+.2f}$')



    ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label=f'$y = {m:.2f}x {b:+.2f}$')
    
    r_sq = rsquared(x, y)
    #ax.text(0, 1.1, "r² = {:.3f}".format(r_sq), transform = ax.transAxes)
    #ax.annotate("r-squared = {:.3f}".format(r2_score(x, y)), (60, 100))


    #fig.legend()
    handles, labels = ax.get_legend_handles_labels()
    labels[0] = "r² = {:.3f}".format(r_sq)
    ax.legend(handles, [labels[0]])
    ax.legend().set_visible(False)

    parentpath = path.parent.parent#path is csv, parent is csv folder, parent is run

    # fig.savefig(parentpath / "graphs/solar_by_wind") #subdir graphs

    parentpath = parentpath.parent.parent #parentpath is run folder, parent is results, parent is pypsaproject

    #fig.savefig(parentpath / f"Images/Shown images/11:5 meeting/solar_by_wind{sec}_{trans}")
    
    return labels

def solar_by_wind_all():
    run0 = "adam_latitude_compare_no_sectors_yes_transmission_3h"
    run1 = "adam_latitude_compare_yes_sectors_yes_transmission_3h2"
    run2 = "adam_latitude_compare_no_sectors_no_transmission_3h3"
    run3 = "adam_latitude_compare_yes_sectors_no_transmission_3h"
    path0 = 'results/' + run0 + '/csvs/gen_and_lat.csv'
    path1 = 'results/' + run1 + '/csvs/gen_and_lat.csv'
    path2 = 'results/' + run2 + '/csvs/gen_and_lat.csv'
    path3 = 'results/' + run3 + '/csvs/gen_and_lat.csv'
    path0 = pathlib.Path(path0)
    path1 = pathlib.Path(path1)
    path2 = pathlib.Path(path2)
    path3 = pathlib.Path(path3)
    

    
    plt.rcParams.update({'font.size': 18})
    fs = 23
    plt.rcParams['axes.labelsize'] = fs
    plt.rcParams['xtick.labelsize'] = fs
    plt.rcParams['ytick.labelsize'] = fs

    fig,ax = plt.subplots(2,2,figsize=(13,9),sharex=True,sharey='row')
    ax = ax.flatten()
    #solar_by_latitude_comparecost(path0, ax[0]) #We know that solar_by_latitude_comparecost is also compatible
    labels0 = solar_by_wind(path0, ax[0])
    labels1 = solar_by_wind(path1, ax[1])
    
    labels2 = solar_by_wind(path2, ax[2])
    
    labels3 = solar_by_wind(path3, ax[3])

    ax0points = [child for child in ax[0].get_children() if isinstance(child, matplotlib.text.Annotation)]
    adjust_text(ax0points, ax = ax[0])
    ax1points = [child for child in ax[1].get_children() if isinstance(child, matplotlib.text.Annotation)]
    adjust_text(ax1points, ax = ax[1])
    ax2points = [child for child in ax[2].get_children() if isinstance(child, matplotlib.text.Annotation)]
    adjust_text(ax2points, ax = ax[2])
    ax3points = [child for child in ax[3].get_children() if isinstance(child, matplotlib.text.Annotation)]
    adjust_text(ax3points, ax = ax[3]) 

    ax[1].lines[-1].set_color('C1')
    ax[2].lines[-1].set_color('C2')
    ax[3].lines[-1].set_color('C3')


    fig.supxlabel(r"$\bf{Sectors}$" ,fontsize=fs, y = 0.11, x = 0.53)
    fig.supylabel ( r"$\bf{Transmission}$",fontsize=fs, y = 0.53)
    ax[3].set_xlabel(" ",fontsize=fs)
    ax[0].set_ylabel("Solar share" , fontsize = fs, y = -0.1)
    # ax[2].set_xlabel(r"$\bf{No}$",fontsize=fs)
    #plt.gca().lines[-].set_color('C2')
    # ax[0].set_ylabel(r"$\bf{Yes}$" , fontsize = fs)
    # ax[2].set_ylabel(r"$\bf{No}$", fontsize = fs)
    ax[2].tick_params(axis='x',  labelsize = fs-1, rotation = 40)
    ax[3].tick_params(axis='x',  labelsize = fs-1, rotation = 40)

    ax[0].set_title("")
    ax[2].set_ylabel("")
    ax[1].set_title("")
    #plt.gca().lines[-3].set_color('C1')
    ax[1].set_ylabel("")
    ax[3].set_ylabel("")
    ax[2].set_title("")
    
    #ax[0].get_lines().set_color('black')



    ax[3].set_title("")

    handles0, labels = ax[0].get_legend_handles_labels()
    handles1, labels = ax[1].get_legend_handles_labels()
    handles2, labels = ax[2].get_legend_handles_labels()
    handles3, labels = ax[3].get_legend_handles_labels()

    handles = handles0 + handles2 + handles1 + handles3
    labels = labels0 + labels2 + labels1 + labels3


    for anax in ax:
        anax.xaxis.set_tick_params(length = 4, width = 2)
        anax.yaxis.set_tick_params(length = 4, width = 2)

    fig.legend(handles, labels, prop={'size':fs-1}, ncol=2, loc = (0.28, 0.01))

    fig.tight_layout(rect = [0.03, 0.1, 1, 0.9])
    plt.subplots_adjust(wspace=0.02, hspace=0.05)
    # fig.suptitle("Solar share by wind capacity factor", x = 0.55, fontsize = fs, weight = 'bold')

    fig.text(0.325, 0.12, "No", fontweight= "bold", fontsize = fs)
    fig.text(0.76, 0.12, "Yes", fontweight= "bold", fontsize = fs)

    fig.text(0.02, 0.73, "Yes", fontweight= "bold", rotation = 90, fontsize = fs)
    fig.text(0.02, 0.33, "No", fontweight= "bold", rotation = 90, fontsize = fs)

    fig.text(0.43, 0.16, "Capacity factor of Wind",fontsize=fs)

    plt.savefig("Images/Paper/cap_wind_paper.png", dpi = 600)

    plt.show()
solar_by_wind_all()

#%%

def solar_by_anycorr(path, timeframe, corrtype, ax):
    '''The goal of this function is to choose between different timeframes and types, such as weekly/3h, or Load, Heating, Cooling
    It builds off of the solar_by_corr function, which only included one type of correlation at a time'''
    mypath = str(path)
    
    opts = mypath.split("_")
    print(opts)
    for opt in opts:#This takes the first number of the last opt with a number
        contains_digit = len(re.findall('\d+', opt)) > 0
        if contains_digit:
            f = re.findall(r'\d+', opt)
            hrs = f[0]
            hrs = hrs + "h"
    if "sectors" in opts:#
        idxsec = opts.index('sectors')
        idxsec -= 1
        sec = opts[idxsec]
        if sec == "yes":
            sec = 'with'
        else:
            sec = 'without'
        sec = sec + " sectors"
    if "transmission" in opts:
        idxtrans = opts.index('transmission')
        idxtrans -= 1
        trans = opts[idxtrans]
        if trans == "yes":
            trans = 'with'
        else:
            trans = 'without'
        trans = trans + " transmission"

    
    

    plt.rcdefaults()
    solar_latdf = pd.read_csv(path)
    solar_latdf = solar_latdf.iloc[:, 1:] #get rid of weird second index
    solar_latdf = solar_latdf[solar_latdf['carrier'] == 'solar']#only interested in wind share
    solar_latdf = solar_latdf.iloc[:-2, :] #get rid of MT and CY, which have 0 according to our model
    #solar_latdf['windpercent'] = solar_latdf["fraction"] * 100
    solar_latdf['solarpercent'] = solar_latdf['solarfrac'] * 100
    
    
    #plotting: latitude on x axis, solar fraction on y axis

    #fig, ax = plt.subplots()

    y = solar_latdf["solarpercent"]
    if timeframe == 'weekly':
        if corrtype == 'Load':
            x = solar_latdf["week_corr"]
            c = 'week_corr'
        if corrtype == 'Heating':
            x = solar_latdf['week_heat']
            c = 'week_heat'
        if corrtype == 'Cooling':
            x = solar_latdf['week_cool']
            c = 'week_cool'
    elif timeframe == '3h':
        if corrtype == 'Load':
            x = solar_latdf["load_corr"]     
            c = 'load_corr'      
        if corrtype == 'Heating':
            x = solar_latdf['heat_corr']
            c = 'heat_corr'
        if corrtype == 'Cooling':
            x = solar_latdf['cool_corr']
            c = 'cool_corr'



    ax.scatter(x, y)
    #ax.set_xlabel(timeframe + " " + corrtype + " correlation with solar production")
    ax.set_ylabel("Optimal solar share (%)")
    ax.set_title("Optimal solar share, " + sec + " and " + trans + " " + hrs)



    for idx, row in solar_latdf.iterrows():
        ax.annotate(row['country'], (row[c]* 1.007, row['solarpercent']* 0.97))
    

    m, b = np.polyfit(x, y, 1)
    #ax.axline(xy1 = (0, b), slope = m, color = 'r', label=f'$y = {m:.2f}x {b:+.2f}$')


    r_sq = rsquared(x, y)
    #print(r_sq)
    ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label=f'$y = {m:.2f}x {b:+.2f}$')
    
    #ax.text(0, 1.1, "r-squared = {:.3f}".format(r_sq), transform = ax.transAxes)


    handles, labels = ax.get_legend_handles_labels()
    labels[0] = "r² = {:.3f}".format(r_sq)
    ax.legend(handles, [labels[0]])
    ax.legend().set_visible(False)
    # fig.legend()
    parentpath = path.parent.parent#path is csv, parent is csv folder, parent is run
    mypath = "graphs/solar_by_"+ timeframe + "_" + corrtype + "_corr"
    #fig.savefig(parentpath / mypath) #subdir graphs

    parentpath = parentpath.parent.parent #parentpath is run folder, parent is results, parent is pypsaproject

    #if I want to save in a second place, then, I will also need to change the code for the path to include which vars it is
    #fig.savefig(parentpath / f"Images/Shown images/11:5 meeting/solar_by_weekcorr{sec}_{trans}")
    #plt.show()
    return labels


def solar_by_anycorr_four(mytime, mycorr):
    '''The purpose of this function is to be able to plot correlation with solar by solar 
    for each of the four scenarios (w/wout transmit, sectors) on the same plot
    
    Choose between "weekly" or "3h" for mytime, and between "Load", "Heating", and "Cooling" for mycorr '''
    run0 = "adam_latitude_compare_no_sectors_yes_transmission_3h"
    run1 = "adam_latitude_compare_yes_sectors_yes_transmission_3h2"
    run2 = "adam_latitude_compare_no_sectors_no_transmission_3h3"
    run3 = "adam_latitude_compare_yes_sectors_no_transmission_3h"
    path0 = 'results/' + run0 + '/csvs/gen_and_lat.csv'
    path1 = 'results/' + run1 + '/csvs/gen_and_lat.csv'
    path2 = 'results/' + run2 + '/csvs/gen_and_lat.csv'
    path3 = 'results/' + run3 + '/csvs/gen_and_lat.csv'
    path0 = pathlib.Path(path0)
    path1 = pathlib.Path(path1)
    path2 = pathlib.Path(path2)
    path3 = pathlib.Path(path3)
    

    fs = 18
    plt.rcParams['axes.labelsize'] = fs
    plt.rcParams['xtick.labelsize'] = fs
    plt.rcParams['ytick.labelsize'] = fs


    fig,ax = plt.subplots(2,2,figsize=(13,9),sharex=True,sharey='row')
    ax = ax.flatten()


    timeframe = mytime


    corrtype = mycorr
    #solar_by_latitude_comparecost(path0, ax[0]) #We know that solar_by_latitude_comparecost is also compatible
    labels0 = solar_by_anycorr(path0, timeframe, corrtype, ax[0])
    labels1 = solar_by_anycorr(path1, timeframe, corrtype, ax[1])
    
    labels2 = solar_by_anycorr(path2, timeframe, corrtype, ax[2])
    
    labels3 = solar_by_anycorr(path3,timeframe, corrtype,  ax[3])
    ax[1].lines[-1].set_color('C1')
    ax[2].lines[-1].set_color('C2')
    ax[3].lines[-1].set_color('C3')


    fig.supxlabel(r"$\bf{Sectors}$" + '\nCorrelation of Solar with ' + corrtype, fontsize=fs, y = 0.11, x = 0.53)
    fig.supylabel ('Solar Percent\n' + r"$\bf{Transmission}$",fontsize=fs, y = 0.6)
    ax[3].set_xlabel(r"$\bf{Yes}$",fontsize=fs)
    ax[2].set_xlabel(r"$\bf{No}$",fontsize=fs)
    #plt.gca().lines[-].set_color('C2')
    ax[0].set_ylabel(r"$\bf{Yes}$" , fontsize = fs)
    ax[2].set_ylabel(r"$\bf{No}$", fontsize = fs)
    ax[2].tick_params(axis='x', labelsize = fs-4)
    ax[3].tick_params(axis='x', labelsize = fs-4)

    ax[0].set_title("")
    ax[1].set_title("")
    #plt.gca().lines[-3].set_color('C1')
    ax[1].set_ylabel("")
    ax[3].set_ylabel("")
    ax[2].set_title("")
    
    #ax[0].get_lines().set_color('black')



    ax[3].set_title("")

    handles0, labels = ax[0].get_legend_handles_labels()
    handles1, labels = ax[1].get_legend_handles_labels()
    handles2, labels = ax[2].get_legend_handles_labels()
    handles3, labels = ax[3].get_legend_handles_labels()

    handles = handles0 + handles2 + handles1 + handles3
    labels = labels0 + labels2 + labels1 + labels3

    fig.legend(handles, labels, prop={'size':fs-4}, ncol=2, loc = (0.4, 0.03))
    fig.tight_layout(rect = [0.03, 0.1, 1, 0.9])
    fig.suptitle("Solar share by solar correlation with " + corrtype, x = 0.55, fontsize = fs, weight = 'bold')

    fig.savefig("Images/Paper/solar_by_loadcorr.png")
    plt.show()

#solar_by_anycorr_four("weekly", "Load")


def check_exist_folder(run_name):
    graphpath = "results/" + run_name + "/graphs"
    csvpath = "results/" + run_name + "/csvs"
    # secondcsv = "results/" + run_name + "/multicsvs"
    if os.path.exists(graphpath) == False:
        os.makedirs(graphpath)
    if os.path.exists(csvpath) == False:
        os.makedirs(csvpath)
    # if os.path.exists(secondcsv) == False:
    #     os.makedirs(secondcsv)










'''
if __name__ == "__main__":
    #Set filepath name
    #This cannot really handle multiple postnetworks yet

    run_name = ["adam_latitude_compare_yes_sectors_no_transmission_3h", "adam_latitude_compare_no_sectors_yes_transmission_3h",
    "adam_latitude_compare_no_sectors_no_transmission_3h3", "adam_latitude_compare_yes_sectors_yes_transmission_3h2"]
    # run_name = "adam_latitude_compare_no_sectors_yes_transmission_3h"
    # run_name = "adam_latitude_compare_no_sectors_no_transmission_3h3"
    # run_name = "adam_latitude_compare_yes_sectors_yes_transmission_3h2"


    # run_name = "adam_latitude_compare_yes_sectors_no_transmission_3h_futcost3"


    
    # check_exist_folder(run_name)



    for run in run_name:
        # for filepath in glob.glob("results/" + run + "/postnetworks/*"):
        #     retrieve_generators(filepath)


        #With multiple postnetworks per run, we may need to add another for loop here as well, as the same as above
        #path = add_to_df(run)
        path = 'results/' + run+ '/csvs/gen_and_lat.csv'
        path = pathlib.Path(path)
        # solar_by_anycorr(path, '3h', 'Heating')
        # solar_by_anycorr(path, '3h', 'Cooling')
        solar_by_anycorr(path, 'weekly', 'Heating')
        solar_by_anycorr(path, 'weekly', 'Cooling')        
        #solar_by_solar(path)

    ###THIS SECTION IS FOR THE NEW FOLDER

    run = "adam_latitude_compare_anysectors_anytransmission_3h_futcost3"
    # for filepath in glob.glob("results/" + run + "/postnetworks/*"):
    #     retrieve_generators_choice(filepath) #for our particular run, this makes a csv in the right folder for everything (8 csvs)
    for filepath in glob.glob("results/" + run + "/csvs/*"):
        # note: for this to work, there can be no pre-existing gen_and_lat csvs. They must be deleted or moved to a different folder
        #print(filepath)
        add_to_df_choice(filepath)

    #solar_by_solar_all()





    

    #all_generators("adam_latitude_compare_no_sectors")

'''
#solar_by_latitude()

#f = f.query('`country`.str.startswith("FR")')

#Function that tests whether a string is in 




#Function that
#loads in the good csv to df
#isolates the solar carrier in the df
#Plots solar share by latitude






















# mydata = mydata.sum()
# mysum = mydata.sum()





###OUTDATED FUNCTIONS#####


def retrieve_generators_old(filepath):
    '''This function takes a completed postnetwork, finds the resource frac
    Old version. This is outdated as of 17/5/22'''
    
    europe = pypsa.Network()
    europe.import_from_netcdf(filepath)

    my_gen = ("offwind-ac", "offwind-dc", "solar", "onwind", "ror")

    countries = eu28
    mydata = europe.generators_t.p
    mydata = mydata[mydata.columns[mydata.columns.str.startswith(countries) ]]
    # mydata = mydata[mydata.columns[mydata.columns.str.contains('|'.join(my_gen))]] #look at column names. look at whether it ends with 
    # mydata = mydata[my_gen]s
    mydata = mydata[mydata.columns[mydata.columns.str.endswith(my_gen)]]
    totalpowers = europe.generators.p_nom_opt
    totalpowers = totalpowers[mydata.columns]

    totalpowers = totalpowers.to_frame()
    totalpowers = totalpowers.T #The p_nom_opt is not a timeseries. However, before we were using timeseries. 
                                # So, to make the same code work (combining similar names), we transpose it


    for country in countries:
        resource_fracs = mydata
        resource_fracs = resource_fracs[[col for col in resource_fracs.columns if col.startswith(country)]]
        mydata[country + 'solar'] = resource_fracs[[col for col in resource_fracs.columns if col.endswith('solar')]].sum(axis = 1) #sums all the solar stuff together
        mydata[country + 'wind'] = resource_fracs[[col for col in resource_fracs.columns if 'wind' in col]].sum(axis = 1)
        if any('ror' in col for col in resource_fracs.columns):#checks if there is a 'ror' column present
            mydata[country + 'ror'] =  resource_fracs[[col for col in resource_fracs.columns if 'ror' in col]].sum(axis = 1)




        allsources = totalpowers
        allsources =allsources[[col for col in allsources.columns if col.startswith(country)]]
        totalpowers[country + 'solar'] = allsources[[col for col in allsources.columns if col.endswith('solar')]].sum(axis = 1) #sums all the solar stuff together
        totalpowers[country + 'wind'] = allsources[[col for col in allsources.columns if 'wind' in col]].sum(axis = 1)
        if any('ror' in col for col in allsources.columns):#checks if there is a 'ror' column present
            totalpowers[country + 'ror'] =  allsources[[col for col in allsources.columns if 'ror' in col]].sum(axis = 1)



    mydata = mydata[mydata.columns[~mydata.columns.str.contains('[0-9]+')]]#gets rid of old columns, not needed in new code


    totalpowers = totalpowers[totalpowers.columns[~totalpowers.columns.str.contains('[0-9]+')]]
    totalpowers = totalpowers.T
    
    

    mydata = mydata.sum()
    print (mydata)
    


    mydata = mydata.to_frame()

    mydata = pd.concat([mydata, totalpowers], axis = 1)


    #In this section of the code, I
        #Extract 
    #save csv in new file, related to filepath
    now_path = pathlib.Path(filepath)
    parent_path = now_path.parent
    run_directory = parent_path.parent


    


    #mydata.to_csv(run_directory / "csvs/generators_T.csv")




    return mydata


def solar_by_latitude_all(path):
    '''The goal of this function is to do the same thing as the solar_by_latitude function but with 
    all three costs. We know that the costs are the full, 1/4, and 0.036 (appx) of cost
    
    Eventually I will need to make a plot with all of the countries on the x axis, ordered by latitude, and the three different costs
    
    I decided not to stick with this function. The different costs make it hard to read what's going on. Instead,
    my goal is to do the line above this one'''
    mypath = str(path)
    
    opts = mypath.split("_")
    print(opts)
    for opt in opts:
        contains_digit = len(re.findall('\d+', opt)) > 0
        if contains_digit:
            f = re.findall(r'\d+', opt)
            hrs = f[0]
            hrs = hrs + "h"
    if "sectors" in opts:
        idxsec = opts.index('sectors')
        idxsec -= 1
        sec = opts[idxsec]
        if sec == "yes":
            sec = 'with'
            searchsec = 'yes'
        else:
            sec = 'without'
            searchsec = 'no'
        sec = sec + " sectors"
    if "transmission" in opts:
        idxtrans = opts.index('transmission')
        idxtrans -= 1
        trans = opts[idxtrans]
        if trans == "yes":
            trans = 'with'
            searchtrans = 'yes'
        else:
            trans = 'without'
            searchtrans = 'no'
        trans = trans + " transmission"

    
    

    plt.rcdefaults()
    solar_latdf = pd.read_csv(path)
    solar_latdf = solar_latdf.iloc[:, 1:] #get rid of weird second index
    solar_latdf = solar_latdf[solar_latdf['carrier'] == 'solar']#only interested in solar share
    solar_latdf = solar_latdf.iloc[:-2, :] #get rid of MT and CY, which have 0 according to our model
    solar_latdf['percent'] = solar_latdf["fraction"] * 100

    #This is the first cheap path
    costs_path = "results/adam_latitude_compare_anysectors_anytransmission_3h_futcost3/csvs/"
    csvfile_cheap = '0.25_gen_and_lat_'+ searchsec + '_sectors_' + searchtrans + '_transmission.csv'

    total_path = costs_path + csvfile_cheap
    csvcheap_path = pathlib.Path(total_path)

    solar_latdf_cheap = pd.read_csv(csvcheap_path)
    solar_latdf_cheap = solar_latdf_cheap.iloc[:, 1:] #get rid of weird second index
    solar_latdf_cheap = solar_latdf_cheap[solar_latdf_cheap['carrier'] == 'solar']#only interested in solar share
    solar_latdf_cheap = solar_latdf_cheap.iloc[:-2, :] #get rid of MT and CY, which have 0 according to our model
    solar_latdf_cheap['percent'] = solar_latdf_cheap["fraction"] * 100

    #This is the cheapest path
    csvfile_cheapest = '0.036_gen_and_lat_'+ searchsec + '_sectors_' + searchtrans + '_transmission.csv'
    total_path2 = costs_path + csvfile_cheapest
    csvcheapest_path = pathlib.Path(total_path2)

    
    solar_latdf_cheapest = pd.read_csv(csvcheapest_path)
    solar_latdf_cheapest = solar_latdf_cheapest.iloc[:, 1:] #get rid of weird second index
    solar_latdf_cheapest = solar_latdf_cheapest[solar_latdf_cheapest['carrier'] == 'solar']#only interested in solar share
    solar_latdf_cheapest = solar_latdf_cheapest.iloc[:-2, :] #get rid of MT and CY, which have 0 according to our model
    solar_latdf_cheapest['percent'] = solar_latdf_cheapest["fraction"] * 100




    
    
    #plotting: latitude on x axis, solar fraction on y axis

    fig, ax = plt.subplots()

    x = solar_latdf["latitude"]
    y = solar_latdf["percent"]

    x1 = solar_latdf_cheap["latitude"]
    y1 = solar_latdf_cheap["percent"]

    x2 = solar_latdf_cheapest["latitude"]
    y2 = solar_latdf_cheapest["percent"]

    ax.scatter(x, y, color = 'b')
    ax.scatter(x1, y1, color = 'g')
    ax.scatter(x2, y2, color = 'r')
    ax.set_xlabel("Latitude (degrees)")
    ax.set_ylabel("Optimal solar share (%)")
    ax.set_title("Optimal solar share, " + sec + " and " + trans + " " + hrs)



    for idx, row in solar_latdf.iterrows():
        ax.annotate(row['country'], (row['latitude']* 1.007, row['percent']* 0.97))
    

    m, b = np.polyfit(x, y, 1)
    #ax.axline(xy1 = (0, b), slope = m, color = 'r', label=f'$y = {m:.2f}x {b:+.2f}$')



    #plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label=f'$y = {m:.2f}x {b:+.2f}$')
    
    #ax.annotate("r-squared = {:.3f}".format(r2_score(x, y)), (60, 100))


    fig.legend()
    parentpath = path.parent.parent

    #fig.savefig(parentpath / "graphs/solar_by_lat")

    plt.show()
  

def solar_by_corr(path):
    '''This function is outdated. Use solar_by_anycorr() and solar_by_anycorr_four()'''
    mypath = str(path)
    
    opts = mypath.split("_")
    print(opts)
    for opt in opts:#This takes the first number of the last opt with a number
        contains_digit = len(re.findall('\d+', opt)) > 0
        if contains_digit:
            f = re.findall(r'\d+', opt)
            hrs = f[0]
            hrs = hrs + "h"
    if "sectors" in opts:#
        idxsec = opts.index('sectors')
        idxsec -= 1
        sec = opts[idxsec]
        if sec == "yes":
            sec = 'with'
        else:
            sec = 'without'
        sec = sec + " sectors"
    if "transmission" in opts:
        idxtrans = opts.index('transmission')
        idxtrans -= 1
        trans = opts[idxtrans]
        if trans == "yes":
            trans = 'with'
        else:
            trans = 'without'
        trans = trans + " transmission"

    
    

    plt.rcdefaults()
    solar_latdf = pd.read_csv(path)
    solar_latdf = solar_latdf.iloc[:, 1:] #get rid of weird second index
    solar_latdf = solar_latdf[solar_latdf['carrier'] == 'solar']#only interested in wind share
    solar_latdf = solar_latdf.iloc[:-2, :] #get rid of MT and CY, which have 0 according to our model
    #solar_latdf['windpercent'] = solar_latdf["fraction"] * 100
    solar_latdf['solarpercent'] = solar_latdf['solarfrac'] * 100
    
    
    #plotting: latitude on x axis, solar fraction on y axis

    fig, ax = plt.subplots()

    x = solar_latdf["week_corr"]
    y = solar_latdf["solarpercent"]


    ax.scatter(x, y)
    ax.set_xlabel("Load correlation with solar production")
    ax.set_ylabel("Optimal solar share (%)")
    ax.set_title("Optimal solar share, " + sec + " and " + trans + " " + hrs)



    for idx, row in solar_latdf.iterrows():
        ax.annotate(row['country'], (row['week_corr']* 1.007, row['solarpercent']* 0.97))
    

    m, b = np.polyfit(x, y, 1)
    #ax.axline(xy1 = (0, b), slope = m, color = 'r', label=f'$y = {m:.2f}x {b:+.2f}$')


    r_sq = rsquared(x, y)
    print(r_sq)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label=f'$y = {m:.2f}x {b:+.2f}$')
    
    ax.text(0, 1.1, "r-squared = {:.3f}".format(r_sq), transform = ax.transAxes)


    fig.legend()
    parentpath = path.parent.parent#path is csv, parent is csv folder, parent is run

    fig.savefig(parentpath / "graphs/solar_by_corr") #subdir graphs

    parentpath = parentpath.parent.parent #parentpath is run folder, parent is results, parent is pypsaproject

    fig.savefig(parentpath / f"Images/Shown images/11:5 meeting/solar_by_weekcorr{sec}_{trans}")
    plt.show()



#mydata = europe.generators
# mydata['index_name'] = mydata.index



# mydata['location'] = mydata.apply(lambda row: row['index'][0:2], axis = 1)
# mydata = mydata[mydata['location'][0][:5]]


# mydata['index'][3][-5:]

# mydata = mydata[mydata['index_name'].str.startswith("PL")]
# f = re.split('\-', "/postnetworks/elec_s_37_lv1.0__Co2L0-3H-T-H-B-I-A-solar+c0.1002546-solar+p3-dist1_2030.nc")

# f[-3]
# 'solar+c0.1002546'

# f[-3][7:]


# f = np.logspace(4, 6.31, 50)
# f = f/600000
# n = []
# for item in f:
#     n.append(item)

# print(n)

# %%
