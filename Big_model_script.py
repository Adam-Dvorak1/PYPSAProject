
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time
from itertools import repeat
import os
import re
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import pathlib
import matplotlib
import csv
import glob
import yaml



run_name = "adam_solar_costs_low_res"
# filepath = "results/"+ run_name + "/postnetworks/elec_s_37_lv1.0__Co2L0-3H-T-H-B-I-A-solar+c0.1002546-solar+p3-dist1_2030.nc"
#loop over all files in folder
def all_generators(run_name):
    #This makes 
    generator = []
    for filepath in glob.glob("results/" + run_name + "/postnetworks/*"):
        my_run = retrieve_generators(filepath)
        generator.append(my_run)
    generator= pd.concat(generator)
    generator.index = pd.to_numeric(generator.index)

    generator = generator.sort_index(ascending = True)
    generator.to_csv(f'results/' + run_name + '/csvs/generators.csv')
    return generator
    
     
    # with open (f"results/csvs/{run_name}_generators.csv", "w", newline = '') as file:
    #     writer = csv.writer(file)
    #     for filepath in glob.glob("results/" + run_name + "/postnetworks/*"):
    #         my_run = retrieve_generators(filepath)
    #         writer.writerow(my_run)
        

def retrieve_generators(filepath):
    europe = pypsa.Network()
    europe.import_from_netcdf(filepath)

    my_gen = ("offwind-ac", "offwind-dc", "solar", "onwind", "ror")

    countries = ("DK", "ES")
    mydata = europe.generators_t.p
    mydata = mydata[mydata.columns[mydata.columns.str.startswith(countries) ]]
    # mydata = mydata[mydata.columns[mydata.columns.str.contains('|'.join(my_gen))]] #look at column names. look at whether it ends with 
    # mydata = mydata[my_gen]
    mydata = mydata[mydata.columns[mydata.columns.str.endswith(my_gen)]]

    for country in countries:
        mydata[country + 'solar'] = mydata[[col for col in mydata.columns if col.endswith('solar') and col.startswith(country)]].sum(axis=1)
        mydata[country + 'onwind'] = mydata[[col for col in mydata.columns if col.endswith('onwind') and col.startswith(country)]].sum(axis=1)
        mydata[country + 'offwind'] = mydata[[col for col in mydata.columns if 'offwind' in col and col.startswith(country)]].sum(axis = 1)

    mydata['ESror'] =  mydata[[col for col in mydata.columns if 'ror' in col and col.startswith("ES")]].sum(axis = 1)

    mydata = mydata[mydata.columns[~mydata.columns.str.contains('[0-9]+')]]


    solar_val = re.split('\_', filepath)
    solar_val = solar_val[-3]


    mydata = mydata.sum()

    mydata = mydata.to_frame(name = solar_val).T
    mydata = mydata.rename_axis("Solar_cost")
    return mydata

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
    
f = make_stackplot("adam_solar_3", True)


with open('./tech_colors.yaml') as file:
    tech_colors = yaml.safe_load(file)['tech_colors']
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
