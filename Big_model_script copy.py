
import resource
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time
from itertools import repeat
from matplotlib import rc
import os
import re
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import seaborn as sns
import pathlib
import matplotlib
import csv
import glob
import yaml
from pylab import *
from sklearn.metrics import r2_score

'''The issue is that before we could just get one item from each country. Now we need entire time series'''

eu28 = (
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
)
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

    myloads = myloads-myloads.mean()
    myloads.rename(columns = lambda x: x[:2], inplace = True) #All of the loads are only named by the country
    myloads = myloads.groupby(level = 0, axis = 1).sum() #There are some duplicate loads. This groups them (adds them together)

    for country in countries:
    #The purpose of this section is to get sum all the solar
        resource_fracs = mydata
        resource_fracs = resource_fracs[[col for col in resource_fracs.columns if col.startswith(country)]]
        mydata[country] = resource_fracs[[col for col in resource_fracs.columns if col.endswith('solar')]].sum(axis = 1) #sums all the solar stuff together

    mydata = mydata[mydata.columns[~mydata.columns.str.contains('[0-9]+')]]#gets rid of old columns, not needed in new code

    mydata = mydata-mydata.mean()#modify this 

    
    covariances = pd.DataFrame()

    for col in myloads.columns:#This will also be looped
        covariances[col] = myloads[col] * mydata[col]/(myloads[col].std()*mydata[col].std())#This is a time series of 


    print(covariances)
    print(covariances.sum()/len(covariances))#This is another step--sum, divide by the length
        


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
    '''This function takes a completed postnetwork, finds the resource frac'''
    
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


def add_to_df(run_name):
    #This takes in a csv made from retrieve_generators and then adds to it
    csvpath = "results/" + run_name + "/csvs/generators_T.csv"
    all_gens = pd.read_csv(csvpath)
    all_gens['country'] = all_gens.apply(lambda row: row['name'][0:2], axis = 1)#This makes a new column, lops off the first two letters of each row name, which is the country
    all_gens['carrier'] = all_gens.apply(lambda row: row['name'].replace(row['country'], ''), axis = 1)#This makes an ew column replaces the country in the name with just the resource
    all_gens = all_gens.rename(columns = {"0": "Generation"})#renames the column of production from "0" to "Generation"
    all_gens["total_gen"] = '' #Makes an empty column
    for country in eu28:
        f = all_gens.loc[all_gens['country'] == country]
        all_gens['total_gen'].loc[all_gens['country'] == country] = f['Generation'].sum()

    all_gens['fraction'] = all_gens.apply(lambda row: 0 if row['total_gen'] == 0 else row['Generation']/row['total_gen'], axis = 1)

    latitude = pd.read_csv("data/countries_lat.csv")
    all_gens = pd.merge(all_gens, latitude[['country', 'latitude']], how = 'left', on = 'country' )

    all_gens['year_CF'] = all_gens.apply(lambda row: 0 if row['p_nom_opt'] == 0 else row['Generation']/(row['p_nom_opt'] * 2920), axis = 1)

    all_gen_solar = all_gens.loc[all_gens['carrier'] == 'solar']
    all_gen_solar['solarfrac'] = all_gen_solar ['fraction']
    all_gen_solar['solarcf'] = all_gen_solar['year_CF']
    all_gens = pd.merge(all_gens, all_gen_solar[['country','solarfrac', 'solarcf']], how = 'left', on = 'country')
    #create empty column
    #use loc 
    nowpath = pathlib.Path(csvpath)
    parentpath = nowpath.parent

    print(all_gens)
    #all_gens.to_csv(parentpath / 'gen_and_lat.csv')

    latcsv_path = parentpath / 'gen_and_lat.csv'

    return(latcsv_path)

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


#Make a function that:
    #Reads in the csv
    #Makes a second dataframe with just solar
    #Makes another column for solar fraction
    #Joins back, adding column
def solar_by_wind(path):
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

    fig, ax = plt.subplots()

    x = solar_latdf["year_CF"]
    y = solar_latdf["solarpercent"]


    ax.scatter(x, y)
    ax.set_xlabel("Wind capacity factor")
    ax.set_ylabel("Optimal solar share (%)")
    ax.set_title("Optimal solar share, " + sec + " and " + trans + " " + hrs)



    for idx, row in solar_latdf.iterrows():
        ax.annotate(row['country'], (row['year_CF']* 1.007, row['solarpercent']* 0.97))
    

    m, b = np.polyfit(x, y, 1)
    #ax.axline(xy1 = (0, b), slope = m, color = 'r', label=f'$y = {m:.2f}x {b:+.2f}$')



    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label=f'$y = {m:.2f}x {b:+.2f}$')
    
    #ax.annotate("r-squared = {:.3f}".format(r2_score(x, y)), (60, 100))


    fig.legend()
    parentpath = path.parent.parent#path is csv, parent is csv folder, parent is run

    # fig.savefig(parentpath / "graphs/solar_by_wind") #subdir graphs

    parentpath = parentpath.parent.parent #parentpath is run folder, parent is results, parent is pypsaproject

    fig.savefig(parentpath / f"Images/Shown images/11:5 meeting/solar_by_wind{sec}_{trans}")
    

    plt.show()

def solar_by_solar(path):
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

    fig, ax = plt.subplots()

    x = solar_latdf["solarcf"]
    y = solar_latdf["solarpercent"]


    ax.scatter(x, y)
    ax.set_xlabel("Solar capacity factor")
    ax.set_ylabel("Optimal solar share (%)")
    ax.set_title("Optimal solar share, " + sec + " and " + trans + " " + hrs)



    for idx, row in solar_latdf.iterrows():
        ax.annotate(row['country'], (row['solarcf']* 1.007, row['solarpercent']* 0.97))
    

    m, b = np.polyfit(x, y, 1)
    #ax.axline(xy1 = (0, b), slope = m, color = 'r', label=f'$y = {m:.2f}x {b:+.2f}$')



    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label=f'$y = {m:.2f}x {b:+.2f}$')
    
    #ax.annotate("r-squared = {:.3f}".format(r2_score(x, y)), (60, 100))


    fig.legend()
    parentpath = path.parent.parent#path is csv, parent is csv folder, parent is run

    fig.savefig(parentpath / "graphs/solar_by_solarcf") #subdir graphs

    parentpath = parentpath.parent.parent #parentpath is run folder, parent is results, parent is pypsaproject

    fig.savefig(parentpath / f"Images/Shown images/11:5 meeting/solar_by_solarcf{sec}_{trans}")
    

    #plt.show()


def check_exist_folder(run_name):
    graphpath = "results/" + run_name + "/graphs"
    csvpath = "results/" + run_name + "/csvs"
    if os.path.exists(graphpath) == False:
        os.makedirs(graphpath)
    if os.path.exists(csvpath) == False:
        os.makedirs(csvpath)




def check_covariance():
    #sums product of elec-elec_mean, solar-solar_mean
    #We want to do this for each country

    x = 3




if __name__ == "__main__":
    #Set filepath name
    #This cannot really handle multiple postnetworks yet

    run_name = ["adam_latitude_compare_yes_sectors_no_transmission_3h", "adam_latitude_compare_no_sectors_yes_transmission_3h",
    "adam_latitude_compare_no_sectors_no_transmission_3h3", "adam_latitude_compare_yes_sectors_yes_transmission_3h2"]
    # run_name = "adam_latitude_compare_no_sectors_yes_transmission_3h"
    # run_name = "adam_latitude_compare_no_sectors_no_transmission_3h3"
    # run_name = "adam_latitude_compare_yes_sectors_yes_transmission_3h2"






    
    #check_exist_folder(run_name)



    for run in run_name:
        # for filepath in glob.glob("results/" + run + "/postnetworks/*"):
        #     retrieve_generators(filepath)





        # path = add_to_df(run)
        path = 'results/' + run+ '/csvs/gen_and_lat.csv'
        path = pathlib.Path(path)
        solar_by_solar(path)

    

    #all_generators("adam_latitude_compare_no_sectors")


#solar_by_latitude()

#f = f.query('`country`.str.startswith("FR")')

#Function that tests whether a string is in 




#Function that
#loads in the good csv to df
#isolates the solar carrier in the df
#Plots solar share by latitude






















# mydata = mydata.sum()
# mysum = mydata.sum()







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
