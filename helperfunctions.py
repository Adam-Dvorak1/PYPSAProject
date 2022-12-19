#%%
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
from datetime import datetime
import matplotlib
import csv
import glob


#%%
def make_a_list(network, country, costrange, folder):
    mylist = []
    for number in costrange:
        atuple = (network, country, number, folder)
        mylist.append(atuple)
    return mylist

#%%
def pen_plus_any_curtailoverlap_t(w_heat, wo_heat, mytype):
    '''This makes a 2x2 grid of two axes each showing resource penetration and solar curtailment vs.
    a scaling log of solar. It is very long. It uses gridspec to order the axes, and other than that
    it is about the same as the other pen_plus_curtail() functions. 17/1
    
    type: can be "solar", "wind", or "batt"
    
    As of 6_October, this is a working and current function for my paper'''
    #solar with heat---solarcost_w_heat_27_Sept.csv 
    #solar without heat--solarcost_elec_27_Sept.csv


    #wind with heat--windcost_heatelec_4_Oct.csv
    #wind without heat--windcost_elec_28_Sept.csv

    #battery with heat--battcost_heatelec_4_Oct.csv
    #battery without heat--battcost_elec_28_Sept.csv

    plt.rcdefaults()
    plt.rcParams.update({'font.size': 14})
    solardnk = pd.read_csv("results/csvs/Denmark/" + w_heat)
    solaresp = pd.read_csv("results/csvs/Spain/"+ w_heat)
    solarcol = pd.read_csv("results/csvs/CO/"+ w_heat)
    solarcal = pd.read_csv("results/csvs/CA/"+ w_heat)

    solardnk2 = pd.read_csv("results/csvs/Denmark/"+ wo_heat)
    solaresp2 = pd.read_csv("results/csvs/Spain/"+ wo_heat)
    solarcol2 = pd.read_csv("results/csvs/CO/"+ wo_heat)
    solarcal2 = pd.read_csv("results/csvs/CA/"+ wo_heat)

    #plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(2, 2, wspace=0.05, hspace=0.13)

    if mytype == "solar":
        inner_dnk = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], wspace=0.05, hspace=0, height_ratios = [1, 2])#reduce wspace?
        inner_esp = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], wspace=0.05, hspace=0, height_ratios = [1, 2])
        inner_col = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2], wspace=0.05, hspace=0, height_ratios = [1, 2])
        inner_cal = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[3], wspace=0.05, hspace=0, height_ratios = [1, 2])

    else:
        inner_dnk = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], wspace=0.1, hspace=0, height_ratios = [0, 2])
        inner_esp = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], wspace=0.1, hspace=0, height_ratios = [0, 2])
        inner_col = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2], wspace=0.1, hspace=0, height_ratios = [0, 2])
        inner_cal = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[3], wspace=0.1, hspace=0, height_ratios = [0, 2])

    axden0 = plt.Subplot(fig, inner_dnk[0])
    axden1 = plt.Subplot(fig, inner_dnk[1])
    fig.add_subplot(axden0)
    fig.add_subplot(axden1)

    axesp0 = plt.Subplot(fig, inner_esp[0])
    axesp1 = plt.Subplot(fig, inner_esp[1])
    fig.add_subplot(axesp0)
    fig.add_subplot(axesp1)

    axcol0 = plt.Subplot(fig, inner_col[0])
    axcol1 = plt.Subplot(fig, inner_col[1])
    fig.add_subplot(axcol0)
    fig.add_subplot(axcol1)

    axcal0 = plt.Subplot(fig, inner_cal[0])
    axcal1 = plt.Subplot(fig, inner_cal[1])
    fig.add_subplot(axcal0)
    fig.add_subplot(axcal1)       

    ####DENMARK###

    if mytype == "solar":
        s_cost = solardnk['solar_cost']
        s_cost = s_cost / 10**6 /0.07846970300338728
    elif mytype == "wind":
        s_cost = solardnk['wind_cost']
        s_cost = s_cost/ 10**6 /0.08442684282600257
    elif mytype == "batt":
        s_cost = solardnk['batt_cost']
        s_cost = s_cost/ 10**6 /0.09439292574325567




    DNK_sp = solardnk['solar_penetration']
    DNK_wp = solardnk['wind_penetration']
    DNK_sc = solardnk['s_curtailment'].abs()

    ESP_sp = solaresp['solar_penetration']
    ESP_wp = solaresp['wind_penetration']
    ESP_sc = solaresp['s_curtailment'].abs()

    COL_sp = solarcol['solar_penetration']
    COL_wp = solarcol['wind_penetration']
    COL_sc = solarcol['s_curtailment'].abs()

    CAL_sp = solarcal['solar_penetration']
    CAL_wp = solarcal['wind_penetration']
    CAL_sc = solarcal['s_curtailment'].abs()

    ###NEW###
    DNK_sp2 = solardnk2['solar_penetration']
    DNK_wp2 = solardnk2['wind_penetration']
    DNK_sc2 = solardnk2['s_curtailment'].abs()

    ESP_sp2 = solaresp2['solar_penetration']
    ESP_wp2 = solaresp2['wind_penetration']
    ESP_sc2 = solaresp2['s_curtailment'].abs()

    CAL_sp2 = solarcal2['solar_penetration']
    CAL_wp2 = solarcal2['wind_penetration']
    CAL_sc2 = solarcal2['s_curtailment'].abs()

    COL_sp2 = solarcol2['solar_penetration']
    COL_wp2 = solarcol2['wind_penetration']
    COL_sc2 = solarcol2['s_curtailment'].abs()

    if mytype == "solar":
        axden0.scatter(s_cost, DNK_sc, s = 15, color = "C1")
        axden0.scatter(s_cost, DNK_sc2, marker = "x", s = 15, color = "C1", alpha = 0.5) #Scatter or plot? 
        axden0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
        axden0.set_ylabel("Curtailment")
        axden0.set_facecolor("#eeeeee")

        axesp0.scatter(s_cost, ESP_sc2, s = 15, color = "C1")
        axesp0.scatter(s_cost, ESP_sc, marker = "x", s = 15, color = "C1", alpha = 0.5)
        # axesp0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
        #axesp0.set_ylabel("Curtailment")
        axesp0.set_facecolor("#eeeeee")


        axcol0.scatter(s_cost, COL_sc2, s = 15, color = "C1")
        axcol0.scatter(s_cost, COL_sc, marker = "x", s = 15, color = "C1", alpha = 0.5)
        axcol0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
        axcol0.set_ylabel("Curtailment")
        axcol0.set_facecolor("#eeeeee")



        axcal0.scatter(s_cost, CAL_sc2, s = 15, color = "C1")
        axcal0.scatter(s_cost, CAL_sc, marker = "x", s = 15, color = "C1", alpha = 0.5)
        # axcal0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
        axcal0.set_facecolor("#eeeeee")


    #The one with the lower alpha also has the one with the "x" marker.


    #Here, the original is the one with the lower alpha, not the one with the change in heat
    # We expect solar to be less favored, so we expect solar to extend more in the original

    #I think the reason why the labels and alphas and stuff are weird is because what is actually
    #coloring the thing is the light blue. 


    
    axden1.stackplot(s_cost, DNK_sp, DNK_wp, colors = ["#f1c232","#2986cc"], labels = ["Solar\nelec + heat", "Wind\nelec + heat"])
    axden1.stackplot(s_cost, DNK_sp2, DNK_wp2, colors = ["#f1c232","#2986cc"], labels = ["Solar\nelec", "Wind\nelec"], alpha = 0.5)
    axden1.set_ylabel("Solar\nShare")


    axden0.set_title("Denmark")

    # axden0.spines["top"].set_visible(False)
    # axden0.spines["right"].set_visible(False)
    plt.rcParams['hatch.linewidth'] = 1

    ####SPAIN####



    axesp1.stackplot(s_cost, ESP_sp, ESP_wp, colors = ["#f1c232","#2986cc"])
    axesp1.stackplot(s_cost, ESP_sp2, ESP_wp2, colors = ["#f1c232","#2986cc"], alpha = 0.5)

    axesp0.set_title("Spain")  




    ####Colorado#####

    axcol1.stackplot(s_cost, COL_sp, COL_wp, colors = ["#f1c232","#2986cc"])
    axcol1.stackplot(s_cost, COL_sp2, COL_wp2, colors = ["#f1c232","#2986cc"], alpha = 0.5)
    axcol1.set_ylabel("Solar\nShare")
    axcol0.set_title("Colorado")


    ###California####


    axcal0.set_facecolor("#eeeeee")
    axcal0.set_title("California")
    axcal1.stackplot(s_cost, CAL_sp, CAL_wp, colors = ["#f1c232","#2986cc"])
    axcal1.stackplot(s_cost, CAL_sp2, CAL_wp2, colors = ["#f1c232","#2986cc"], alpha = 0.5)




    plt.rcParams['hatch.linewidth'] = 1


    if mytype != "solar":
        axden0.yaxis.set_ticklabels([])
        axcol0.yaxis.set_ticklabels([])





    if mytype == "solar":
        axcal1.set_xlabel("Cost Solar PV (€/Wp)")
        axcol1.set_xlabel("Cost Solar PV (€/Wp)")
    elif mytype == "wind":
        axcal1.set_xlabel("Cost Wind (€/Wp)")
        axcol1.set_xlabel("Cost Wind (€/Wp)")
    elif mytype == "batt":
        axcal1.set_xlabel("Cost Battery (€/Wh)")
        axcol1.set_xlabel("Cost Battery (€/Wh)")


    for ax in plt.gcf().get_axes():
        ax.set_ylim(0,1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

        ax.minorticks_on()
      
        ax.label_outer()

        ax.margins(x = 0)
        ax.set_xscale('log')        
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if mytype == "wind":
            ax.set_xlim(0, 1.45)     
        elif mytype == "batt":
            ax.set_xlim(0, 0.40)

    
    for ax in plt.gcf().get_axes()[1::2]:
        '''This section should only act on the axes with the actual solar share'''
        # Note, before I had a number lower than 0.132. i think this is because I assumed
        # "less optimistic" was something different

        if mytype == "solar":
            ax.fill_between([0.019, 0.132], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "purple")
            ax.axvline(0.529, color='black',ls='--')
            ax.axvline(0.019, color='black',ls='--')
            ax.text(0.045,0.05, "Future Range", fontsize = 16, horizontalalignment = "center", rotation = "vertical")
            
            ax.text(0.85,0.05,  "Today's range", fontsize = 16, horizontalalignment = "center", rotation = "vertical")
            ax.axvline(0.132, color='black',ls='--')
            ax.fill_between([0.529, 1.3], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
            ax.axvline(1.3, color='black',ls='--')

        elif mytype == "wind":

            ax.fill_between([1.12, 1.22], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
            ax.axvline(1.12, color='black',ls='--')
            ax.axvline(1.22, color='black',ls='--')
            ax.text(1.35 ,0.05,  "Today's range", fontsize = 16, horizontalalignment = "center", rotation = "vertical")

            ax.fill_between([0.57, 0.77], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "purple")
            ax.axvline(0.57, color='black',ls='--')
            ax.axvline(0.77, color='black',ls='--')
            ax.text(0.65, 0.05, "Future range", fontsize = 16, horizontalalignment = "center", rotation = "vertical")
        
        elif mytype == "batt":
            ax.fill_between([0.232, 0.311], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
            ax.axvline(0.232, color='black',ls='--')
            ax.axvline(0.31, color='black',ls='--')
            ax.text(0.28,  0.05, "Today's range", fontsize = 16, horizontalalignment = "center", rotation = "vertical")

            #ax.text(0.36 ,0.05,  "Today's cost", fontsize = 16, horizontalalignment = "center", rotation = "vertical")

            ax.fill_between([0.056, 0.24], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "purple")
            ax.axvline(0.056, color='black',ls='--')
            ax.axvline(0.24, color='black',ls='--')
            ax.text(0.12, 0.05, "Future Range", fontsize = 16, horizontalalignment = "center", rotation = "vertical")





    axesp0.yaxis.set_ticklabels([])
    axesp1.yaxis.set_ticklabels([])
    axcal0.yaxis.set_ticklabels([])
    axcal1.yaxis.set_ticklabels([])
    axesp1.xaxis.set_ticklabels([])
    axden1.xaxis.set_ticklabels([])


    #Whether or not we want to include the last 100% tick label--no for solar, yes for the others
    if mytype == "solar":
        xticks = axesp1.yaxis.get_major_ticks() 
        xticks[-1].label1.set_visible(False)
        xticks = axcol1.yaxis.get_major_ticks() 
        xticks[-1].label1.set_visible(False)
        xticks = axcal1.yaxis.get_major_ticks() 
        xticks[-1].label1.set_visible(False)
        xticks = axden1.yaxis.get_major_ticks() 
        xticks[-1].label1.set_visible(False)

    lines1, labels1 = axden1.get_legend_handles_labels()

    fig.legend(lines1, labels1, bbox_to_anchor=(0.85, 1.015), ncol=4)

    curDT = datetime.now()
    version = curDT.strftime("_%d_%m_%Y")
    if mytype == "solar":
        save = "Images/Paper/Figure3_solar_sensitivity_w_and_woHeat_ver" + version
        #fig.suptitle(r"$\bf{Sensitivity\;to\;Cost\;of\;Solar\;(€/W)}$", fontsize = 24)
    elif mytype == "wind":
        save = "Images/Paper/Figure4_wind_sensitivity_w_and_woHeat_ver" + version
        #fig.suptitle(r"$\bf{Sensitivity\;to\;Cost\;of\;Wind\;(€/W)}$", fontsize = 24)
    elif mytype == "batt":
        save = "Images/Paper/Figure5_batt_sensitivity_w_and_woHeat_ver" + version
        #fig.suptitle(r"$\bf{Sensitivity\;to\;Cost\;of\;Battery\;(€/Wh)}$", fontsize = 24)




    plt.savefig(save + ".pdf")
    plt.savefig(save + ".png", dpi = 600)
    #fig.text(0.88, 0.08, "€/Wp")

    #fig.supxlabel(r"$\bf{Cost(€/MW)}$", fontsize = 20)
    
    #plt.savefig("Images/solar_compare_wHeat_and_eleconly_sept27data.pdf")
    plt.show()


pen_plus_any_curtailoverlap_t("solarcost_heatelec_26_Oct.csv", "solarcost_elec_26_Oct.csv", "solar")
#%%
def pen_plus_any_poster_curtailoverlap_t(w_heat, wo_heat, mytype):
    '''This makes a 2x2 grid of two axes each showing resource penetration and solar curtailment vs.
    a scaling log of solar. It is very long. It uses gridspec to order the axes, and other than that
    it is about the same as the other pen_plus_curtail() functions. 17/1
    
    type: can be "solar", "wind", or "batt"
    
    As of 6_October, this is a working and current function for my paper'''
    #solar with heat---solarcost_w_heat_27_Sept.csv 
    #solar without heat--solarcost_elec_27_Sept.csv


    #wind with heat--windcost_heatelec_4_Oct.csv
    #wind without heat--windcost_elec_28_Sept.csv

    #battery with heat--battcost_heatelec_4_Oct.csv
    #battery without heat--battcost_elec_28_Sept.csv

    plt.rcdefaults()
    plt.rcParams.update({'font.size': 16})
    solardnk = pd.read_csv("results/csvs/Denmark/" + w_heat)
    solaresp = pd.read_csv("results/csvs/Spain/"+ w_heat)
    solarcol = pd.read_csv("results/csvs/CO/"+ w_heat)
    solarcal = pd.read_csv("results/csvs/CA/"+ w_heat)

    solardnk2 = pd.read_csv("results/csvs/Denmark/"+ wo_heat)
    solaresp2 = pd.read_csv("results/csvs/Spain/"+ wo_heat)
    solarcol2 = pd.read_csv("results/csvs/CO/"+ wo_heat)
    solarcal2 = pd.read_csv("results/csvs/CA/"+ wo_heat)

    #plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(2, 2, wspace=0.05, hspace=0.13)


    inner_dnk = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], wspace=0.1, hspace=0, height_ratios = [0, 2])
    inner_esp = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], wspace=0.1, hspace=0, height_ratios = [0, 2])
    inner_col = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2], wspace=0.1, hspace=0, height_ratios = [0, 2])
    inner_cal = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[3], wspace=0.1, hspace=0, height_ratios = [0, 2])

    axden0 = plt.Subplot(fig, inner_dnk[0])
    axden1 = plt.Subplot(fig, inner_dnk[1])
    fig.add_subplot(axden0)
    fig.add_subplot(axden1)

    axesp0 = plt.Subplot(fig, inner_esp[0])
    axesp1 = plt.Subplot(fig, inner_esp[1])
    fig.add_subplot(axesp0)
    fig.add_subplot(axesp1)

    axcol0 = plt.Subplot(fig, inner_col[0])
    axcol1 = plt.Subplot(fig, inner_col[1])
    fig.add_subplot(axcol0)
    fig.add_subplot(axcol1)

    axcal0 = plt.Subplot(fig, inner_cal[0])
    axcal1 = plt.Subplot(fig, inner_cal[1])
    fig.add_subplot(axcal0)
    fig.add_subplot(axcal1)       

    ####DENMARK###

    if mytype == "solar":
        s_cost = solardnk['solar_cost']
        s_cost = s_cost / 10**6 /0.07846970300338728
    elif mytype == "wind":
        s_cost = solardnk['wind_cost']
        s_cost = s_cost/ 10**6 /0.08442684282600257
    elif mytype == "batt":
        s_cost = solardnk['batt_cost']
        s_cost = s_cost/ 10**6 /0.09439292574325567




    DNK_sp = solardnk['solar_penetration']
    DNK_wp = solardnk['wind_penetration']
    DNK_sc = solardnk['s_curtailment'].abs()

    ESP_sp = solaresp['solar_penetration']
    ESP_wp = solaresp['wind_penetration']
    ESP_sc = solaresp['s_curtailment'].abs()

    COL_sp = solarcol['solar_penetration']
    COL_wp = solarcol['wind_penetration']
    COL_sc = solarcol['s_curtailment'].abs()

    CAL_sp = solarcal['solar_penetration']
    CAL_wp = solarcal['wind_penetration']
    CAL_sc = solarcal['s_curtailment'].abs()

    ###NEW###
    DNK_sp2 = solardnk2['solar_penetration']
    DNK_wp2 = solardnk2['wind_penetration']
    DNK_sc2 = solardnk2['s_curtailment'].abs()

    ESP_sp2 = solaresp2['solar_penetration']
    ESP_wp2 = solaresp2['wind_penetration']
    ESP_sc2 = solaresp2['s_curtailment'].abs()

    CAL_sp2 = solarcal2['solar_penetration']
    CAL_wp2 = solarcal2['wind_penetration']
    CAL_sc2 = solarcal2['s_curtailment'].abs()

    COL_sp2 = solarcol2['solar_penetration']
    COL_wp2 = solarcol2['wind_penetration']
    COL_sc2 = solarcol2['s_curtailment'].abs()


    #The one with the lower alpha also has the one with the "x" marker.


    #Here, the original is the one with the lower alpha, not the one with the change in heat
    # We expect solar to be less favored, so we expect solar to extend more in the original

    #I think the reason why the labels and alphas and stuff are weird is because what is actually
    #coloring the thing is the light blue. 


    
    axden1.stackplot(s_cost, DNK_sp, DNK_wp, colors = ["#f1c232","#2986cc"], labels = ["Solar\nelec + heat", "Wind\nelec + heat"])
    axden1.stackplot(s_cost, DNK_sp2, DNK_wp2, colors = ["#f1c232","#2986cc"], labels = ["Solar\nelec", "Wind\nelec"], alpha = 0.5)
    axden1.set_ylabel("Solar\nShare")


    axden0.set_title("Denmark")

    axden0.spines["top"].set_visible(False)
    axden0.spines["right"].set_visible(False)
    plt.rcParams['hatch.linewidth'] = 1

    ####SPAIN####



    axesp1.stackplot(s_cost, ESP_sp, ESP_wp, colors = ["#f1c232","#2986cc"])
    axesp1.stackplot(s_cost, ESP_sp2, ESP_wp2, colors = ["#f1c232","#2986cc"], alpha = 0.5)

    axesp0.set_title("Spain")  


    #Whether or not we want to include the last 100% tick label--no for solar, yes for the others


    ####Colorado#####

    axcol1.stackplot(s_cost, COL_sp, COL_wp, colors = ["#f1c232","#2986cc"])
    axcol1.stackplot(s_cost, COL_sp2, COL_wp2, colors = ["#f1c232","#2986cc"], alpha = 0.5)
    axcol1.set_ylabel("Solar\nShare")
    axcol0.set_title("Colorado")


    ###California####


    axcal0.set_facecolor("#eeeeee")
    axcal0.set_title("California")
    axcal1.stackplot(s_cost, CAL_sp, CAL_wp, colors = ["#f1c232","#2986cc"])
    axcal1.stackplot(s_cost, CAL_sp2, CAL_wp2, colors = ["#f1c232","#2986cc"], alpha = 0.5)




    plt.rcParams['hatch.linewidth'] = 1



    axden0.yaxis.set_ticklabels([])
    axcol0.yaxis.set_ticklabels([])





    if mytype == "solar":
        axcal1.set_xlabel("Cost Solar PV (€/Wp)")
        axcol1.set_xlabel("Cost Solar PV (€/Wp)")
    elif mytype == "wind":
        axcal1.set_xlabel("Cost Wind (€/Wp)")
        axcol1.set_xlabel("Cost Wind (€/Wp)")
    elif mytype == "batt":
        axcal1.set_xlabel("Cost Battery (€/Wh)")
        axcol1.set_xlabel("Cost Battery (€/Wh)")


    for ax in plt.gcf().get_axes():
        ax.set_ylim(0,1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

        ax.minorticks_on()
      
        ax.label_outer()

        ax.margins(x = 0)
        ax.set_xscale('log')        
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))
        if mytype == "wind":
            ax.set_xlim(0, 1.45)     
        elif mytype == "batt":
            ax.set_xlim(0, 0.40)

    
    for ax in plt.gcf().get_axes()[1::2]:
        '''This section should only act on the axes with the actual solar share'''
        # Note, before I had a number lower than 0.132. i think this is because I assumed
        # "less optimistic" was something different

        if mytype == "solar":
            ax.fill_between([0.019, 0.132], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "purple")
            ax.axvline(0.529, color='black',ls='--')
            ax.axvline(0.019, color='black',ls='--')
            ax.text(0.045,0.05, "Future Range", fontsize = 16, horizontalalignment = "center", rotation = "vertical")
            
            ax.text(0.85,0.05,  "Today's range", fontsize = 16, horizontalalignment = "center", rotation = "vertical")
            ax.axvline(0.132, color='black',ls='--')
            ax.fill_between([0.529, 1.3], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
            ax.axvline(1.3, color='black',ls='--')

        elif mytype == "wind":

            ax.fill_between([1.12, 1.22], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
            ax.axvline(1.12, color='black',ls='--')
            ax.axvline(1.22, color='black',ls='--')
            ax.text(1.35 ,0.05,  "Today's range", fontsize = 16, horizontalalignment = "center", rotation = "vertical")

            ax.fill_between([0.57, 0.77], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "purple")
            ax.axvline(0.57, color='black',ls='--')
            ax.axvline(0.77, color='black',ls='--')
            ax.text(0.65, 0.05, "Future range", fontsize = 16, horizontalalignment = "center", rotation = "vertical")
        
        elif mytype == "batt":
            ax.axvline(0.300, color='black',ls='--')
            ax.text(0.36 ,0.05,  "Today's cost", fontsize = 16, horizontalalignment = "center", rotation = "vertical")

            ax.fill_between([0.056, 0.24], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "purple")
            ax.axvline(0.056, color='black',ls='--')
            ax.axvline(0.24, color='black',ls='--')
            ax.text(0.12, 0.05, "Future Range", fontsize = 16, horizontalalignment = "center", rotation = "vertical")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)




    axesp0.yaxis.set_ticklabels([])
    axesp1.yaxis.set_ticklabels([])
    axcal0.yaxis.set_ticklabels([])
    axcal1.yaxis.set_ticklabels([])
    axesp1.xaxis.set_ticklabels([])
    axden1.xaxis.set_ticklabels([])

    lines1, labels1 = axden1.get_legend_handles_labels()

    fig.legend(lines1, labels1, bbox_to_anchor=(0.9, 1.015), ncol=4)

    curDT = datetime.now()
    version = curDT.strftime("_%d_%m_%Y")
    if mytype == "solar":
        save = "Images/Presentation/solar_sensitivity_w_and_woHeat_ver" + version
        #fig.suptitle(r"$\bf{Sensitivity\;to\;Cost\;of\;Solar\;(€/W)}$", fontsize = 24)
    elif mytype == "wind":
        save = "Images/Presentation/wind_sensitivity_w_and_woHeat_ver" + version
        #fig.suptitle(r"$\bf{Sensitivity\;to\;Cost\;of\;Wind\;(€/W)}$", fontsize = 24)
    elif mytype == "batt":
        save = "Images/Presentation/batt_sensitivity_w_and_woHeat_ver" + version
        #fig.suptitle(r"$\bf{Sensitivity\;to\;Cost\;of\;Battery\;(€/Wh)}$", fontsize = 24)




    plt.savefig(save + ".pdf")
    plt.savefig(save + ".png", dpi = 600)
    #fig.text(0.88, 0.08, "€/Wp")

    #fig.supxlabel(r"$\bf{Cost(€/MW)}$", fontsize = 20)
    
    #plt.savefig("Images/solar_compare_wHeat_and_eleconly_sept27data.pdf")
    plt.show()



# if __name__ == "__main__":
    # Denmark = pypsa.Network()
    # Spain = pypsa.Network()
    # CA = pypsa.Network()
    # CO = pypsa.Network()    
        
    # mynetworks = [Denmark, Spain, CA, CO]

    # set_hours(mynetworks)
    # for network in mynetworks:
    #     add_bus(network)
        

    # df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0)# in MWh
    # df_elec.index = pd.to_datetime(df_elec.index) #change index to datatime

    # df_cal_elec = pd.read_csv('data_extra/CaliforniaTimeSeries.csv', index_col=0)
    # df_cal_elec.index = pd.to_datetime(df_cal_elec.index)

    # df_co_elec = pd.read_csv('data_extra/ColoradoTimeSeries.csv', index_col=0)
    # df_co_elec.index = pd.to_datetime(df_co_elec.index)

    # #We want to simulate electrification of heating. We can then add to Denmark and Spain
    # df_heat = pd.read_csv('data/heat_demand.csv', sep=';', index_col=0)# in MWh
    # df_heat.index = pd.to_datetime(df_heat.index) #change index to datatime
    # heatCA = pd.read_csv("data/TemperatureData/ninja_2011_weather_country_US.CA_merra-2_population_weighted.csv",  header = 2, index_col=0)
    # heatCA.index = pd.to_datetime(heatCA.index)
    # heatCO = pd.read_csv("data/TemperatureData/ninja_2011_weather_country_US.CO_merra-2_population_weighted.csv",  header = 2, index_col=0)
    # heatCO.index = pd.to_datetime(heatCO.index)



    # df_elec["DNKheat"] = df_heat["DNK"]
    # df_elec["ESPheat"] = df_heat["ESP"]

    # df_elec["DNKcombine"] = df_elec.apply(lambda row: row["DNK"] + row["DNKheat"]/3, axis = 1)
    # df_elec["ESPcombine"] = df_elec.apply(lambda row: row["ESP"] + row["ESPheat"]/3, axis = 1)

    # heatCA["HDD"] = heatCA.apply(lambda row: 17 - row["temperature"] if row["temperature"] < 17 else 0, axis = 1)
    # heatCO["HDD"] = heatCO.apply(lambda row: 17 - row["temperature"] if row["temperature"] < 17 else 0, axis = 1)

    # df_cal_elec["HDD"] = heatCA["HDD"]
    # df_cal_elec["heating_demand"] = df_cal_elec.apply(lambda row: 1715 * row["HDD"] + 6356, axis = 1)# 1715 is California's G factor, MWh/HDD. 6356 is the constant, that we get from water heating
    # df_cal_elec["adjust_elec_demand"] =  df_cal_elec["demand_mwh"] + 1/3 * df_cal_elec["heating_demand"]

    # df_co_elec["HDD"] = heatCO["HDD"]
    # df_co_elec["heating_demand"]= df_co_elec.apply(lambda row: 1782 * row["HDD"] + 6472, axis = 1)
    # df_co_elec["adjust_elec_demand"] =  df_co_elec["demand_mwh"] + 1/3 * df_co_elec["heating_demand"]








    # Spain.add("Load",
    #             "load", 
    #             bus="electricity bus", 
    #             p_set=df_elec['ESP'])


    # f = make_a_list("Spain", np.logspace(4, 5, 4), "helloworld")


