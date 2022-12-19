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



######OLD FUNCTIONS######


def flex_plus_curtailDNK(co2):

    DNK_sp = [x[1] for x in co2]
    DNK_sc = list(map(abs,[x[3] for x in co2]))
    DNK_wp = [x[2] for x in co2]
    DNK_gas = [x[5] for x in co2]

    fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axs[0].scatter(DNK_gas, DNK_sc, color = "C1") #Scatter or plot?
    axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axs[0].set_ylabel("Curtailment")
    axs[0].set_facecolor("#eeeeee")


    axs[1].stackplot(DNK_gas, DNK_sp, DNK_wp, DNK_gas, colors = ["#f1c232","#2986cc", "#cbbcf4"])
    axs[1].set_ylim(0, 1)

    axs[1].set_ylabel("Penetration")
    axs[1].set_xlabel("Percent flexible source")

    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axs[0].spines[["top","right"]].set_visible(False)
    axs[1].annotate("Solar", xy = (0.002, 0.6), fontsize = "18")

    xticks = axs[1].yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)


    for ax in plt.gcf().get_axes():
        ax.minorticks_on()
      
        ax.label_outer()

        ax.margins(x = 0)

    

    plt.subplots_adjust(hspace = 0)
    #plt.savefig("Images/Flex_pen_and_curtail_DNK")
    
    plt.show()
    #return axs
#flex_plus_curtailDNK(co2dnk)



def plot_an_image():
    '''What I am making here '''
    fig, axs = plt.subplots(2,1)
    img1 = mpimg.imread("Images/Figure4.png")
    img2 = mpimg.imread("Images/Figure_1.png")
    axs[0].imshow(img1)
    axs[1].imshow(img2)
    plt.subplots_adjust(hspace = 0)

    for ax in axs.flat:
        ax.axis("off")
    plt.show()


def penetration_chart():
    '''This is an old function'''
    #This is our x axis, solar_cost (s_cost)
    s_cost = np.logspace(3, 6, 10)

    #This is our y axis. sp = solar penetration, wp = wind penetration, gp = gas penetration
    DNK_sp = [x[0][0] for x in DNK_solar_data]
    ESP_sp = [x[0][0] for x in ESP_solar_data]
    CAL_sp = [x[0][0]  for x in CAL_solar_data]
    CO_sp = [x[0][0]  for x in CO_solar_data]

    DNK_wp = [x[0][1] for x in DNK_solar_data]
    ESP_wp = [x[0][1] for x in ESP_solar_data]
    CAL_wp = [x[0][1] for x in CAL_solar_data]
    CO_wp = [x[0][1] for x in CO_solar_data]

    DNK_gp = [x[0][2] for x in DNK_solar_data]
    ESP_gp = [x[0][2] for x in ESP_solar_data]
    CAL_gp = [x[0][2] for x in CAL_solar_data]
    CO_gp = [x[0][2] for x in CO_solar_data]

    DNK_bp = [x[0][3] for x in DNK_solar_data]
    ESP_bp = [x[0][3] for x in ESP_solar_data]
    CAL_bp = [x[0][3] for x in CAL_solar_data]
    CO_bp = [x[0][3] for x in CO_solar_data]

    DNK_hp = [x[0][4] for x in DNK_solar_data]  
    ESP_hp = [x[0][4] for x in ESP_solar_data]
    CAL_hp = [x[0][4] for x in CAL_solar_data]
    CO_hp = [x[0][4] for x in CO_solar_data]


    fig, axs = plt.subplots(2,2)
    axs[0, 0].stackplot(s_cost/10**6, DNK_sp, DNK_wp, DNK_gp, labels = ["solar", "wind", "gas"]
                    , colors = ["#ffd966", "#2986cc", "#d5a6bd"])
    axs[0, 0].set_title("Denmark penetration")
    axs[0, 1].stackplot(s_cost/10**6, ESP_sp, ESP_wp, ESP_gp, labels = ["solar", "wind", "gas"]
                    , colors = ["#ffd966", "#2986cc", "#d5a6bd"])
    axs[0, 1].set_title("Spain penetration")
    axs[1, 0].stackplot(s_cost/10**6, CO_sp, CO_wp, CO_gp, labels = ["solar", "wind", "gas"]
                    , colors = ["#ffd966", "#2986cc", "#d5a6bd"])
    axs[1, 0].set_title("Colorado penetration")
    axs[1, 1].stackplot(s_cost/10**6, CAL_sp, CAL_wp, CAL_gp, labels = ["solar", "wind", "gas"]
                    , colors = ["#ffd966", "#2986cc", "#d5a6bd"])
    axs[1, 1].set_title("California penetration")



    for ax in axs.flat:
        ax.minorticks_on()
        ax.set_xscale('log')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title='Resource type', loc='upper right')
        ax.grid(which = 'major', linestyle='-', linewidth='0.5', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax.set(xlabel='solar cost (EUR/MW)', ylabel='penetration')
    #Hide x labels and tick labels for top plots and y ticks for right plots.
        ax.label_outer()
        ax.axvline(0.529, color='black',ls='--')
        ax.text(0.529,1.1, "Current cost = 0.529 EUR/Wh", horizontalalignment = "center")


    plt.suptitle("Penetration per technology by solar overnight investment cost", fontsize = 20)
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    fig.set_size_inches(18.5, 10.5)
    #plt.savefig("Images/PenPerTechbySolarCost", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()

def system_cost():
    '''This is an old function'''
    s_cost = np.linspace(50000, 600000, 10)

    DNK_cst = [x[1] for x in DNK_solar_data]
    ESP_cst = [x[1] for x in ESP_solar_data]
    CAL_cst = [x[1] for x in CAL_solar_data]
    CO_cst = [x[1] for x in CO_solar_data]




    fig, axs = plt.subplots(2,2)
    axs[0, 0].plot(s_cost, DNK_cst, 'ro')
    axs[0, 0].set_title("Denmark system cost")
    axs[0, 1].plot(s_cost, ESP_cst, 'bo')
    axs[0, 1].set_title("Spain system cost")
    axs[1, 0].plot(s_cost, CO_cst, 'co')
    axs[1, 0].set_title("Colorado system cost")
    axs[1, 1].plot(s_cost, CAL_cst, 'go')
    axs[1, 1].set_title("California system cost")



    for ax in axs.flat:
        ax.minorticks_on()
        ax.set_ylim(50, 110)
        ax.grid(which = 'major', linestyle='-', linewidth='0.5', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax.set(xlabel='solar cost (EUR/MW)', ylabel='system cost (EUR/MWh)')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
        ax.label_outer()
        ax.axvline(529000, color='black',ls='--')
        ax.text(529000,66, "Current cost = 529000 EUR", horizontalalignment = "center")

    plt.suptitle("Total system cost by solar overnight investment cost", fontsize = 20)
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    fig.set_size_inches(18.5, 10.5)
    #plt.savefig("Images/TotalSystemCostbySolarCost", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()

def showcurtailment():
    '''This is an old function'''
    s_cost = np.logspace(3, 6, 10)

    DNK_sc = list(map(abs, [x[2][0] for x in DNK_solar_data]))
    ESP_sc = list(map(abs, [x[2][0] for x in ESP_solar_data]))
    CAL_sc = list(map(abs, [x[2][0] for x in CAL_solar_data]))
    CO_sc = list(map(abs, [x[2][0] for x in CO_solar_data]))
    

    DNK_wc = list(map(abs, [x[2][1] for x in DNK_solar_data]))
    ESP_wc = list(map(abs, [x[2][1] for x in ESP_solar_data]))
    CAL_wc = list(map(abs, [x[2][1] for x in CAL_solar_data]))
    CO_wc = list(map(abs, [x[2][1] for x in CO_solar_data]))

    # DNK_sc_wo = list(map(abs, [x[2] for x in DNK_solar_wo]))
    # ESP_sc_wo = list(map(abs, [x[2] for x in ESP_solar_wo]))
    # CAL_sc_wo = list(map(abs, [x[2] for x in CAL_solar_wo]))
    # CO_sc_wo = list(map(abs, [x[2] for x in CO_solar_wo]))


    fig, axs = plt.subplots(2,2)
    axs[0, 0].plot(s_cost/10**6, DNK_sc, 'o', color = "orange", label = "Solar")
    #axs[0, 0].plot(s_cost, DNK_wc, 'bo', label = "Wind")
    #axs[0, 0].plot(w_cost, DNK_sc, 'ko', label = "Without storage")
    axs[0, 0].set_title("Denmark tech curtailed percent")
    axs[0, 1].plot(s_cost/10**6, ESP_sc, 'o', color = "orange", label = "Solar")
    #axs[0, 1].plot(s_cost, ESP_wc, 'bo', label = "Wind")
    #axs[0, 1].plot(s_cost, ESP_sc_wo, 'ko', label = "Without storage")
    axs[0, 1].set_title("Spain tech curtailed percent")
    axs[1, 0].plot(s_cost/10**6, CO_sc, 'o', color = "orange", label = "Solar")
    #axs[1, 0].plot(s_cost, CO_wc, 'bo', label = "Wind")
    #axs[1, 0].plot(s_cost, CO_sc_wo, 'ko', label = "Without storage")
    axs[1, 0].set_title("Colorado tech curtailed percent")
    axs[1, 1].plot(s_cost/10**6, CAL_sc, 'o', color = "orange", label = "Solar")
    #axs[1, 1].plot(s_cost, CAL_wc, 'bo', label = "Wind")
    #axs[1, 1].plot(s_cost, CAL_sc_wo, 'ko', label = "Without storage")
    axs[1, 1].set_title("California tech curtailed percent")



    for ax in axs.flat:
        ax.minorticks_on()
        ax.set_xscale('log')
        ax.legend(title= "type of resource", loc='upper right')
        ax.grid(which = 'major', linestyle='-', linewidth='0.5', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax.set(xlabel='solar cost (EUR/MW)', ylabel='solar curtailment fraction')
        ax.set_ylim(-0.02, 1)
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        ax.label_outer()
        ax.axvline(0.529, color='black',ls='--')
        ax.text(0.529,0.75, "Current cost = 529000 EUR", horizontalalignment = "center")

    plt.suptitle("Fraction of curtailment by solar overnight investment cost", fontsize = 20)
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    fig.set_size_inches(18.5, 10.5)
    plt.savefig("Images/FracSolarCurtailbySolarCostStore_w_and_wo", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()

def pen_plus_curtailDNK(solar):
    s_cost = [x[0] for x in solar]
    s_cost = [item for sublist in s_cost for item in sublist]
    s_cost = [x / 10**6 /0.07846970300338728 for x in s_cost] #we want to convert to Euro/Wp, not Eur/MW
    #I did something really dumb--I had been plotting the annualized solar cost on the x axis, but this was not
    #the value that I was interested in

    #s_cost = s_cost.sort()

    DNK_sp = [x[1]for x in solar]
    DNK_wp = [x[2] for x in solar]

    DNK_sc = list(map(abs, [x[3] for x in solar]))


    fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axs[0].scatter(s_cost, DNK_sc, color = "C1") #Scatter or plot?
    axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axs[0].set_ylabel("Curtailment")
    axs[0].set_facecolor("#eeeeee")

    # labels = axs[0].get_yticklabels()
    # labels[0] = ""
    # axs[0].set_yticklabels(labels)
    
    axs[1].stackplot(s_cost, DNK_sp, DNK_wp, colors = ["#f1c232", "#2986cc"], labels = ["Solar", "Wind"])
    #axs[1].stackplot(s_cost, DNK_sp, colors = ["#f1c232"])
    axs[1].set_ylim(0, 1)
    axs[1].set_xlim(left = 0.001, right = 1)
    axs[0].set_xlim(left = 0.001, right = 1)
    #axs[1].fill_between(s_cost, DNK_sp, color = "#fff2cc")
    #axs[1].fill_between(s_cost, DNK_sp, y2 = 1, color = "#eeeeee")
    axs[1].set_ylabel("Penetration")
    axs[1].set_xlabel("Cost (€/Wp)")


    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    
    axs[1].axvline(0.529, color='black',ls='--')
    axs[1].text(0.7,0.05, "Today", horizontalalignment = "center", rotation = "vertical")
    axs[1].axvline(0.019, color='black',ls='--')
    axs[1].text(0.025,0.05, "2050--Optimistic", horizontalalignment = "center", rotation = "vertical")
    axs[1].axvline(0.095, color='black',ls='--')
    axs[1].text(0.13,0.05, "2050--Less Optimistic", horizontalalignment = "center", rotation = "vertical")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[1].annotate("Solar", xy = (0.002, 0.2), fontsize = "18")
    
    # yticks = axs[0].yaxis.get_major_ticks() 
    # yticks[-1].label1.set_visible(False)
    
    xticks = axs[1].yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)
    axs[1].legend()
    #Use this if you wish to have the 0 on the top graph be invisible

    #axs[0].yaxis.get_major_ticks()[1].label1.set_visible(False)
    for ax in axs.flat:
        ax.minorticks_on()
        ax.set_xscale('log')        
        ax.label_outer()
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))
        ax.margins(x = 0)



    plt.suptitle("Denmark")
    

    plt.subplots_adjust(hspace = 0)
    #plt.savefig("Images/Pen_and_curtail_DNK")
    #plt.show()
    plt.close(fig)
    return axs
#pen_plus_curtailDNK(solardnk)

def pen_plus_curtailESP(solar):
    s_cost = [x[0] for x in solar]
    s_cost = [item for sublist in s_cost for item in sublist]
    s_cost = [x / 10**6 /0.07846970300338728 for x in s_cost] #we want to convert to Euro/Wp, not Eur/MW
    #I did something really dumb--I had been plotting the annualized solar cost on the x axis, but this was not
    #the value that I was interested in

    #s_cost = s_cost.sort()

    ESP_sp = [x[1]for x in solar]
    ESP_wp = [x[2] for x in solar]

    ESP_sc = list(map(abs, [x[3] for x in solar]))


    fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axs[0].scatter(s_cost, ESP_sc, color = "C1") #Scatter or plot?
    axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axs[0].set_ylabel("Curtailment")
    axs[0].set_facecolor("#eeeeee")

    # labels = axs[0].get_yticklabels()
    # labels[0] = ""
    # axs[0].set_yticklabels(labels)
    
    axs[1].stackplot(s_cost, ESP_sp, ESP_wp, colors = ["#f1c232", "#2986cc"], labels = ["Solar", "Wind"])
    #axs[1].scatter(s_cost, ESP_sp, color = "#f1c232")
    axs[1].set_ylim(0, 1)
    axs[1].set_xlim(left = 0.001, right = 1)
    axs[0].set_xlim(left = 0.001, right = 1)
    #axs[1].fill_between(s_cost, DNK_sp, color = "#fff2cc")
    #axs[1].fill_between(s_cost, DNK_sp, y2 = 1, color = "#eeeeee")
    axs[1].set_ylabel("Penetration")
    axs[1].set_xlabel("Cost (€/Wp)")


    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    
    axs[1].axvline(0.529, color='black',ls='--')
    axs[1].text(0.7,0.05, "Today", horizontalalignment = "center", rotation = "vertical")
    axs[1].axvline(0.019, color='black',ls='--')
    axs[1].text(0.025,0.05, "2050--Optimistic", horizontalalignment = "center", rotation = "vertical")
    axs[1].axvline(0.095, color='black',ls='--')
    axs[1].text(0.13,0.05, "2050--Less Optimistic", horizontalalignment = "center", rotation = "vertical")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[1].annotate("Solar", xy = (0.002, 0.2), fontsize = "18")
    
    # yticks = axs[0].yaxis.get_major_ticks() 
    # yticks[-1].label1.set_visible(False)
    
    xticks = axs[1].yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)

    #Use this if you wish to have the 0 on the top graph be invisible

    #axs[0].yaxis.get_major_ticks()[1].label1.set_visible(False)
    for ax in axs.flat:
        ax.minorticks_on()
        ax.set_xscale('log')        
        ax.label_outer()
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))
        ax.margins(x = 0)



    plt.suptitle("Spain")
    

    plt.subplots_adjust(hspace = 0)
    #plt.savefig("Images/Pen_and_curtail_ESP")
    #plt.show()
    plt.close(fig)
    return axs
#pen_plus_curtailESP(solaresp)

def pen_plus_curtailCO(solar):
    s_cost = [x[0] for x in solar]
    s_cost = [item for sublist in s_cost for item in sublist]
    s_cost = [x / 10**6 /0.07846970300338728 for x in s_cost] #we want to convert to Euro/Wp, not Eur/MW
    #I did something really dumb--I had been plotting the annualized solar cost on the x axis, but this was not
    #the value that I was interested in

    #s_cost = s_cost.sort()

    CO_sp = [x[1]for x in solar]
    CO_wp = [x[2] for x in solar]

    CO_sc = list(map(abs, [x[3] for x in solar]))


    fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axs[0].scatter(s_cost, CO_sc, color = "C1") #Scatter or plot?
    axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axs[0].set_ylabel("Curtailment")
    axs[0].set_facecolor("#eeeeee")

    # labels = axs[0].get_yticklabels()
    # labels[0] = ""
    # axs[0].set_yticklabels(labels)
    
    axs[1].stackplot(s_cost, CO_sp, CO_wp, colors = ["#f1c232", "#2986cc"], labels = ["Solar", "Wind"])
    #axs[1].scatter(s_cost, CO_sp, color = "#f1c232")
    axs[1].set_ylim(0, 1)
    axs[1].set_xlim(left = 0.001, right = 1)
    axs[0].set_xlim(left = 0.001, right = 1)
    #axs[1].fill_between(s_cost, DNK_sp, color = "#fff2cc")
    #axs[1].fill_between(s_cost, DNK_sp, y2 = 1, color = "#eeeeee")
    axs[1].set_ylabel("Penetration")
    axs[1].set_xlabel("Cost (€/Wp)")


    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    
    axs[1].axvline(0.529, color='black',ls='--')
    axs[1].text(0.7,0.05, "Today", horizontalalignment = "center", rotation = "vertical")
    axs[1].axvline(0.019, color='black',ls='--')
    axs[1].text(0.025,0.05, "2050--Optimistic", horizontalalignment = "center", rotation = "vertical")
    axs[1].axvline(0.095, color='black',ls='--')
    axs[1].text(0.13,0.05, "2050--Less Optimistic", horizontalalignment = "center", rotation = "vertical")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[1].annotate("Solar", xy = (0.002, 0.2), fontsize = "18")
    
    # yticks = axs[0].yaxis.get_major_ticks() 
    # yticks[-1].label1.set_visible(False)
    
    xticks = axs[1].yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)

    #Use this if you wish to have the 0 on the top graph be invisible

    #axs[0].yaxis.get_major_ticks()[1].label1.set_visible(False)
    for ax in axs.flat:
        ax.minorticks_on()
        ax.set_xscale('log')        
        ax.label_outer()
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))
        ax.margins(x = 0)



    plt.suptitle("Colorado")
    

    plt.subplots_adjust(hspace = 0)
    #plt.savefig("Images/Pen_and_curtail_CO")
    #plt.show()
    plt.close(fig)
    return axs
#pen_plus_curtailCO(solarcol)


def pen_plus_curtailCA(solar):
    s_cost = [x[0] for x in solar]
    s_cost = [item for sublist in s_cost for item in sublist]
    s_cost = [x / 10**6 /0.07846970300338728 for x in s_cost] #we want to convert to Euro/Wp, not Eur/MW
    #I did something really dumb--I had been plotting the annualized solar cost on the x axis, but this was not
    #the value that I was interested in

    #s_cost = s_cost.sort()

    CA_sp = [x[1]for x in solar]
    CA_wp = [x[2] for x in solar]

    CA_sc = list(map(abs, [x[3] for x in solar]))


    fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axs[0].scatter(s_cost, CA_sc, color = "C1") #Scatter or plot?
    axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axs[0].set_ylabel("Curtailment")
    axs[0].set_facecolor("#eeeeee")

    # labels = axs[0].get_yticklabels()
    # labels[0] = ""
    # axs[0].set_yticklabels(labels)
    
    axs[1].stackplot(s_cost, CA_sp, CA_wp, colors = ["#f1c232", "#2986cc"], labels = ["Solar", "Wind"])
    #axs[1].scatter(s_cost, CA_sp, color = "#f1c232")
    axs[1].set_ylim(0, 1)
    axs[1].set_xlim(left = 0.001, right = 1)
    axs[0].set_xlim(left = 0.001, right = 1)

    axs[1].set_ylabel("Penetration")
    axs[1].set_xlabel("Cost (€/Wp)")


    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    
    axs[1].axvline(0.529, color='black',ls='--')
    axs[1].text(0.7,0.05, "Today", horizontalalignment = "center", rotation = "vertical")
    axs[1].axvline(0.019, color='black',ls='--')
    axs[1].text(0.025,0.05, "2050--Optimistic", horizontalalignment = "center", rotation = "vertical")
    axs[1].axvline(0.095, color='black',ls='--')
    axs[1].text(0.13,0.05, "2050--Less Optimistic", horizontalalignment = "center", rotation = "vertical")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[1].annotate("Solar", xy = (0.002, 0.2), fontsize = "18")
    
    # yticks = axs[0].yaxis.get_major_ticks() 
    # yticks[-1].label1.set_visible(False)
    
    xticks = axs[1].yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)

    #Use this if you wish to have the 0 on the top graph be invisible

    #axs[0].yaxis.get_major_ticks()[1].label1.set_visible(False)
    for ax in axs.flat:
        ax.minorticks_on()
        ax.set_xscale('log')        
        ax.label_outer()
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))
        ax.margins(x = 0)



    plt.suptitle("California")
    

    plt.subplots_adjust(hspace = 0)
    #plt.savefig("Images/Pen_and_curtail_CA")
    #plt.show()
    plt.close(fig)
    return axs
#pen_plus_curtailCA(solarcal)


def find_solar_data_old(n, name, solar_cost):



    #This is an outdated function, which returns useful things. Instead, we just care
    #about exporting to the netCDF
    # Takes annualized coefficient and multiplies by investment cost
  
    annualized_solar_cost =  0.07846970300338728* solar_cost
    n.generators.loc[['solar'],['capital_cost']] = annualized_solar_cost
    
    #this substitutes the current solar cost in our generator for a new cost

    
    n.lopf(n.snapshots, 
             pyomo=False,
             solver_name='gurobi')


    n.export_to_netcdf("NetCDF/"+ name + f"/costLOGJan24/{solar_cost}solar_cost.nc")
    
    #commenting out the sum of generators--battery is so small, we need raw values
    solar_penetration = n.generators_t.p['solar'].sum()/sum(n.generators_t.p.sum())
    wind_penetration = n.generators_t.p['onshorewind'].sum()/sum(n.generators_t.p.sum())
    gas_penetration = n.generators_t.p['OCGT'].sum()/sum(n.generators_t.p.sum())
    
    
    systemcost = n.objective/n.loads_t.p.sum()
    
    
    #This now expresses solar in terms of a percent of its capacity
    s_curtailment = (n.generators_t.p-n.generators.p_nom_opt * n.generators_t.p_max_pu)['solar'].sum()
    
    w_curtailment = (n.generators_t.p-n.generators.p_nom_opt * n.generators_t.p_max_pu)['onshorewind'].sum()
    ###If you want to get a plot of curtailment alone, then delete the following lines 
    #of code until the return statement###
    max_gen = (n.generators.p_nom_opt * n.generators_t.p_max_pu)['solar'].sum()
    max_gen_w = (n.generators.p_nom_opt * n.generators_t.p_max_pu)['onshorewind'].sum()
    

    #We want to get the percent of energy curtailed. However, if max_gen is 0, then
    #we get a number that is undefined. We must use loc 
    if max_gen == 0:
        s_curtailment = 0
    else:
        s_curtailment = s_curtailment/max_gen

    if max_gen_w == 0:
        w_curtailment = 0
    else:
        w_curtailment = w_curtailment/max_gen
    
    ###You can delete the code above if you wish###
    
    
    ##We also want to  find out the amount of power used by battery
    battery_pen = n.links_t.p1["battery discharger"].sum()/sum(n.generators_t.p.sum())
    
    hydrogen_pen = n.links_t.p1["H2 Fuel Cell"].sum()/sum(n.generators_t.p.sum())
    
    
    return ((solar_penetration, wind_penetration, gas_penetration, battery_pen, hydrogen_pen), systemcost, 
            (s_curtailment, w_curtailment))




def pen_plus_wind_curtailoverlap():
    '''This makes a 2x2 grid of two axes each showing resource penetration and solar curtailment vs.
    a scaling log of solar. It is very long. It uses gridspec to order the axes, and other than that
    it is about the same as the other pen_plus_curtail() functions. 17/1
    
    This uses the github (Danish energy agency) value of onshore wind for the lower bound,
    and the NREL annual technology baseline for the upper bound.'''
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['hatch.linewidth'] = 1
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    inner_dnk = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], wspace=0.1, hspace=0, height_ratios = [1, 2])
    inner_esp = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], wspace=0.1, hspace=0, height_ratios = [1, 2])
    inner_col = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2], wspace=0.1, hspace=0, height_ratios = [1, 2])
    inner_cal = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[3], wspace=0.1, hspace=0, height_ratios = [1, 2])

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
    w_cost = [x[6] for x in winddnk]
    w_cost = [item for sublist in w_cost for item in sublist]
    w_cost = [x / 10**6 /0.08442684282600257 for x in w_cost] 
    DNK_sp = [x[1] for x in winddnk]
    DNK_wp = [x[2] for x in winddnk]
    DNK_wc = list(map(abs,[x[4] for x in winddnk]))

    #NEW
    DNK_sp2 = [x[1] for x in winddnk2]
    DNK_wp2 = [x[2] for x in winddnk2]
    DNK_wc2 = list(map(abs,[x[4] for x in winddnk2]))

    ESP_sp2 = [x[1] for x in windesp2]
    ESP_wp2 = [x[2] for x in windesp2]
    ESP_wc2 = list(map(abs,[x[4] for x in windesp2]))

    COL_sp2 = [x[1] for x in windcol2]
    COL_wp2 = [x[2] for x in windcol2]
    COL_wc2 = list(map(abs,[x[4] for x in windcol2]))

    CAL_sp2 = [x[1] for x in windcal2]
    CAL_wp2 = [x[2] for x in windcal2]
    CAL_wc2 = list(map(abs,[x[4] for x in windcal2]))    


    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axden0.scatter(w_cost, DNK_wc, marker = "x", s = 15, color = "C1", alpha = 0.5)
    axden0.scatter(w_cost, DNK_wc2, s = 15, color = "C1")
    axden0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axden0.set_ylabel("Curtailment")
    axden0.set_facecolor("#eeeeee")

    #I'm a bit confused. one would think that the alpha should be the other way around. But it's not
    axden1.stackplot(w_cost, DNK_sp2, DNK_wp2, colors = ["#f1c232","#2986cc"], labels = ["Solar_heat", "Wind_heat"]) 
    axden1.stackplot(w_cost, DNK_sp, DNK_wp, colors = ["#f1c232","#2986cc"], labels = ["Solar_og", "Wind_og"], alpha = 0.5)
    
    axden1.set_ylim(0, 1)
    axden0.set_ylim(0, 1)

    axden1.set_ylabel("Penetration")
    #axden1.set_xlabel("Percent flexible source")

    axden1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axden0.set_title("Denmark")

    xticks = axden1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)


    #The 2.7 is the offshore wind price


    #We want to make a range for today's prices. the upper range is 
    axden0.spines["top"].set_visible(False)
    axden0.spines["right"].set_visible(False)
   


    ####SPAIN####

    ESP_sp = [x[1] for x in windesp]
    ESP_wp = [x[2] for x in windesp]
    ESP_wc = list(map(abs,[x[4] for x in windesp]))



    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axesp0.scatter(w_cost, ESP_wc, marker = "x", s = 15, color = "C1", alpha = 0.5) 
    axesp0.scatter(w_cost, ESP_wc2, s = 15, color = "C1") 
    axesp0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    axesp0.set_facecolor("#eeeeee")
    axesp0.set_title("Spain")
 

    #This still plots the right thing, although the order of plotting is actually different than DNK
    axesp1.stackplot(w_cost, ESP_sp, ESP_wp, colors = ["#f1c232","#2986cc"])
    axesp1.stackplot(w_cost, ESP_sp2, ESP_wp2, colors = ["#f1c232","#2986cc"], labels = ["Solar_heat", "Wind_heat"], alpha = 0.5)
    axesp1.set_ylim(0, 1)
    axesp0.set_ylim(0, 1)

    axesp1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))


    xticks = axesp1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)

    axesp0.spines["top"].set_visible(False)
    axesp0.spines["right"].set_visible(False)
    axesp0.yaxis.set_ticklabels([])
    axesp1.yaxis.set_ticklabels([])

    ####Colorado#####

    COL_sp = [x[1] for x in windcol]
    COL_wp = [x[2] for x in windcol]
    COL_wc = list(map(abs,[x[4] for x in windcol]))

    axcol0.scatter(w_cost, COL_wc, marker = "x", s = 15, color = "C1", alpha = 0.5) 
    axcol0.scatter(w_cost, COL_wc2, s = 15, color = "C1") 

    axcol0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axcol0.set_ylabel("Curtailment")
    axcol0.set_facecolor("#eeeeee")
    axcol0.set_title("Colorado")




    axcol1.stackplot(w_cost, COL_sp, COL_wp, colors = ["#f1c232","#2986cc"])
    axcol1.stackplot(w_cost, COL_sp2, COL_wp2, colors = ["#f1c232","#2986cc"], labels = ["Solar_heat", "Wind_heat"], alpha = 0.5)

    axcol1.set_ylim(0, 1)
    axcol0.set_ylim(0, 1)

    axcol1.set_ylabel("Penetration")
    axcol1.set_xlabel("Cost of Wind")

    axcol1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))


    xticks = axcol1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)

    axcol0.spines["top"].set_visible(False)
    axcol0.spines["right"].set_visible(False)

    
    
    ###California####

    CAL_sp = [x[1] for x in windcal]
    CAL_wp = [x[2] for x in windcal]
    CAL_wc = list(map(abs,[x[4] for x in windcal]))



    axcal0.scatter(w_cost, CAL_wc, marker = "x", s = 15, color = "C1", alpha = 0.5) 
    axcal0.scatter(w_cost, CAL_wc2, s = 15, color = "C1") 


    axcal0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    axcal0.set_facecolor("#eeeeee")
    axcal0.set_title("California")


    axcal1.stackplot(w_cost, CAL_sp, CAL_wp, colors = ["#f1c232","#2986cc"])
    axcal1.stackplot(w_cost, CAL_sp2, CAL_wp2, colors = ["#f1c232","#2986cc"], labels = ["Solar_heat", "Wind_heat"], alpha = 0.5)


    axcal1.set_ylim(0, 1)
    axcal0.set_ylim(0, 1)


    axcal1.set_xlabel("Cost of Wind")

    axcal1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    xticks = axcal1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)


  
    axcal0.spines["top"].set_visible(False)
    axcal0.spines["right"].set_visible(False)
    axcal0.yaxis.set_ticklabels([])
    axcal1.yaxis.set_ticklabels([])

 
    #This applies things for all axes
    for ax in plt.gcf().get_axes():
        ax.minorticks_on()
      
        ax.label_outer()

        ax.margins(x = 0)
        ax.set_xscale('log')        
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))


    #This applies things for only the axes of penetration.
    for ax in plt.gcf().get_axes()[1::2]:
        ax.fill_between([1.12, 1.22], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
        ax.axvline(1.12, color='black',ls='--')
        ax.axvline(1.22, color='black',ls='--')
        ax.text(1.4 ,0.05,  "Today's range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")

        ax.fill_between([0.57, 0.77], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "purple")
        ax.axvline(0.57, color='black',ls='--')
        ax.axvline(0.77, color='black',ls='--')
        ax.text(0.65, 0.05, "Future range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    
    # for ax in plt.gcf().get_axes()[::2]:
    #     ax.set_ylim(0,1)


    axesp1.xaxis.set_ticklabels([])
    axden1.xaxis.set_ticklabels([])

    lines1, labels1 = axden1.get_legend_handles_labels()

    fig.legend(lines1, labels1, bbox_to_anchor=(0.85, 0.055), ncol=4)

    #print(fig.axes[1::2])

    plt.savefig("Images/Figure_wind_compare_gw_var1.png")
    plt.show()
    

def pen_plus_batt_curtailoverlap_old():
    '''This makes a 2x2 grid of two axes each showing resource penetration and solar curtailment vs.
    a scaling log of solar. It is very long. It uses gridspec to order the axes, and other than that
    it is about the same as the other pen_plus_curtail() functions. 17/1
    
    This uses the github (Danish energy agency) value of onshore wind for the lower bound,
    and the NREL annual technology baseline for the upper bound.'''
    plt.rcdefaults()

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['hatch.linewidth'] = 1
    fig = plt.figure(figsize=(10, 9))
    outer = gridspec.GridSpec(3, 2, wspace=0.2, hspace=0.3, height_ratios = [1, 1, 0.01])
    inner_dnk = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], wspace=0.1, hspace=0, height_ratios = [0, 2])
    inner_esp = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], wspace=0.1, hspace=0, height_ratios = [0, 2])
    inner_col = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2], wspace=0.1, hspace=0, height_ratios = [0, 2]) #(gridspec(3,1)) to add some extra space
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
    b_cost = battdnk['batt_cost']
    b_cost = b_cost/ 10**6 /0.09439292574325567

    DNK_sp = battdnk['solar_penetration']
    DNK_wp = battdnk['wind_penetration']
    #DNK_sc = list(map(abs,[x[3] for x in battdnk]))
    #DNK_wc = list(map(abs,[x[4] for x in battdnk]))
    #DNK_bp = list(map(abs,[x[8] for x in battdnk]))


    ##NEW###
    DNK_sp2 = battdnk2['solar_penetration']
    DNK_wp2 = battdnk2['wind_penetration']
    #DNK_bp2 = list(map(abs,[x[8] for x in battdnk2]))


    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    # axden0.scatter(b_cost, DNK_sc, s = 15, color = "C1", label = "solar curtailment")
    # axden0.scatter(b_cost, DNK_wc, s = 15, color = "C0", label = "wind curtailment")
       
    axden0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    # axden0.set_ylabel("Curtailment")
    axden0.set_facecolor("#eeeeee")




    axden1.stackplot(b_cost, DNK_sp2, DNK_wp2, colors = ["#f1c232","#2986cc"], labels = ["Solar", "Wind"]) 
    axden1.stackplot(b_cost, DNK_sp, DNK_wp, colors = ["#f1c232","#2986cc"], labels = ["Modified Solar", "Modified Wind"], alpha = 0.5)
     
    axden1.set_ylim(0, 1)
    axden0.set_ylim(0, 1)

    axden1.set_ylabel("Penetration")


    axden1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axden0.set_title("Denmark")

    xticks = axden1.yaxis.get_major_ticks() 
    #xticks[-1].label1.set_visible(False)


    #We want to make a range for today's prices. the upper range is 
    axden0.spines["top"].set_visible(False)
    axden0.spines["right"].set_visible(False)
   


    ####SPAIN####

    ESP_sp = battesp['solar_penetration']
    ESP_wp = battesp['wind_penetration']
    # ESP_wc = list(map(abs,[x[4] for x in battesp]))
    # ESP_sc = list(map(abs,[x[3] for x in battesp]))
    # ESP_bp = list(map(abs,[x[8] for x in battesp]))


    ##NEW###
    ESP_sp2 = battesp2['solar_penetration']
    ESP_wp2 = battesp2['wind_penetration']
    # ESP_bp2 = list(map(abs,[x[8] for x in battesp2]))

    # axesp0.scatter(b_cost, ESP_wc, s = 15, color = "C0") 
    # axesp0.scatter(b_cost, ESP_sc, s = 15, color = "C1") 
    axesp0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    axesp0.set_facecolor("#eeeeee")
    axesp0.set_title("Spain")





    axesp1.stackplot(b_cost, ESP_sp, ESP_wp, colors = ["#f1c232","#2986cc"])
    axesp1.stackplot(b_cost, ESP_sp2, ESP_wp2, colors = ["#f1c232","#2986cc"], labels = ["Solar_heat", "Wind_heat"], alpha = 0.5)
    
    axesp1.set_ylim(0, 1)
    axesp0.set_ylim(0, 1)

    axesp1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))


    xticks = axesp1.yaxis.get_major_ticks() 
    #xticks[-1].label1.set_visible(False)

    axesp0.spines["top"].set_visible(False)
    axesp0.spines["right"].set_visible(False)
    axesp0.yaxis.set_ticklabels([])
    axesp1.yaxis.set_ticklabels([])

    ####Colorado#####

    COL_sp = battcol['solar_penetration']
    COL_wp = battcol['wind_penetration']
    # COL_wc = list(map(abs,[x[4] for x in battcol]))
    # COL_sc = list(map(abs,[x[3] for x in battcol]))    
    # COL_bp = list(map(abs,[x[8] for x in battcol]))


    COL_sp2 = battcol2['solar_penetration']
    COL_wp2 = battcol2['wind_penetration']
    # COL_bp2 = list(map(abs,[x[8] for x in battcol2]))

    # axcol0.scatter(b_cost, COL_wc, s = 15, color = "C0") 
    # axcol0.scatter(b_cost, COL_sc, s = 15, color = "C1")     
    axcol0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    #axcol0.set_ylabel("Curtailment")
    axcol0.set_facecolor("#eeeeee")
    axcol0.set_title("Colorado")


    axcol1.stackplot(b_cost, COL_sp, COL_wp, colors = ["#f1c232","#2986cc"])
    axcol1.stackplot(b_cost, COL_sp2, COL_wp2, colors = ["#f1c232","#2986cc"], labels = ["Solar_heat", "Wind_heat"], alpha = 0.5)

    axcol1.set_ylim(0, 1)
    axcol0.set_ylim(0, 1)

    axcol1.set_ylabel("Penetration")
    #axcol1.set_xlabel(r"$\bf{Cost\;of\;Battery\;(€/MW)}$", x = 1.1, y = 0.1, fontsize = 16)

    axcol1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))


    xticks = axcol1.yaxis.get_major_ticks() 
    #xticks[-1].label1.set_visible(False)

    axcol0.spines["top"].set_visible(False)
    axcol0.spines["right"].set_visible(False)

    
    
    ###California####

    CAL_sp =battcal['solar_penetration']
    CAL_wp = battcal['wind_penetration']
    # CAL_wc = list(map(abs,[x[4] for x in battcal]))
    # CAL_sc = list(map(abs,[x[3] for x in battcal]))
    # CAL_bp = list(map(abs,[x[8] for x in battcal]))    

    CAL_sp2 = battcal2['solar_penetration']
    CAL_wp2 = battcal2['wind_penetration']
    # CAL_bp2 = list(map(abs,[x[8] for x in battcal2]))


    # axcal0.scatter(b_cost, CAL_wc, s = 15, color = "C0")
    # axcal0.scatter(b_cost, CAL_sc, s = 15, color = "C1")
    axcal0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    axcal0.set_facecolor("#eeeeee")
    axcal0.set_title("California")



    axcal1.stackplot(b_cost, CAL_sp, CAL_wp, colors = ["#f1c232","#2986cc"])
    axcal1.stackplot(b_cost, CAL_sp2, CAL_wp2, colors = ["#f1c232","#2986cc"], labels = ["Solar_heat", "Wind_heat"], alpha = 0.5)


    axcal1.set_ylim(0, 1)
    axcal0.set_ylim(0, 1)


    #axcal1.set_xlabel("Cost of Battery")

    axcal1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    xticks = axcal1.yaxis.get_major_ticks() 
    #xticks[-1].label1.set_visible(False)


  
    axcal0.spines["top"].set_visible(False)
    axcal0.spines["right"].set_visible(False)
    axcal0.yaxis.set_ticklabels([])
    axcal1.yaxis.set_ticklabels([])

    # axden0.yaxis.set_ticklabels([])
    # axcol0.yaxis.set_ticklabels([])
 
    #This applies things for all axes

    #This applies things for only the axes of penetration.
    for ax in plt.gcf().get_axes()[1::2]:
        #ax.fill_between([0.232, 0.311], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
        # ax.axvline(0.232, color='black',ls='--')
        # ax.axvline(0.311, color='black',ls='--')
        ax.axvline(0.300, color='black',ls='--')
        ax.text(0.36 ,0.05,  "Today's cost", fontsize = 16, horizontalalignment = "center", rotation = "vertical")

        #ax.fill_between([0.075, 0.22], y1 = 1, alpha = 0.2, edgecolor = "k", hatch = "XX", facecolor = "purple")
        ax.fill_between([0.056, 0.24], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "purple")
        ax.axvline(0.056, color='black',ls='--')
        ax.axvline(0.24, color='black',ls='--')
        ax.text(0.12, 0.05, "Future Range", fontsize = 16, horizontalalignment = "center", rotation = "vertical")
        # ax.text(0.062, 0.05, "More optimistic", fontsize = 14, horizontalalignment = "center", rotation = "vertical")
        # ax.text(0.262, 0.05, "Less optimistic", fontsize = 14, horizontalalignment = "center", rotation = "vertical")

    
    # for ax in plt.gcf().get_axes()[::2]:
    #     ax.set_ylim(0,1)


    axesp1.xaxis.set_ticklabels([])
    axden1.xaxis.set_ticklabels([])






    for ax in plt.gcf().get_axes():
        ax.minorticks_on()
        #ax.grid()
        
      
        ax.label_outer()
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1 / 5))

        ax.margins(x = 0)
        ax.set_xscale('log')        
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))

    axden0.yaxis.set_ticklabels([])
    axcol0.yaxis.set_ticklabels([])
    axden1.xaxis.set_ticklabels([])
    axesp1.xaxis.set_ticklabels([])
#    plt.rcParams["font.weight"] = "bold"
    # axden0b = axden0.twinx()
    # axden0b.scatter(b_cost, DNK_bp, marker = "x", s = 15, color = "C2", label = "Battery share new", alpha = 0.5) 
    # axden0b.scatter(b_cost, DNK_bp2, s = 15, color = "C2", label = "Battery share old")
    # axden0b.tick_params(axis = "y", labelcolor = "C2")

    # axesp0b = axesp0.twinx()
    # axesp0b.scatter(b_cost, ESP_bp, marker = "x", s = 15, color = "C2", label = "battery penetration", alpha = 0.5)
    # axesp0b.scatter(b_cost, ESP_bp2, s = 15, color = "C2", label = "battery penetration")
    # axesp0b.tick_params(axis = "y", labelcolor = "C2") 
    # axesp0b.set_ylabel("Battery\nfraction")
 
    # axcol0b = axcol0.twinx()
    # axcol0b.scatter(b_cost, COL_bp, marker = "x", s = 15, color = "C2", label = "battery penetration", alpha = 0.5)
    # axcol0b.scatter(b_cost, COL_bp2, s = 15, color = "C2", label = "battery penetration")
    # axcol0b.tick_params(axis = "y", labelcolor = "C2") 

    # axcal0b = axcal0.twinx()
    # axcal0b.scatter(b_cost, CAL_bp, marker = "x", s = 15, color = "C2", label = "battery penetration", alpha = 0.5)
    # axcal0b.scatter(b_cost, CAL_bp2, s = 15, color = "C2", label = "battery penetration")
    # axcal0b.tick_params(axis = "y", labelcolor = "C2")
    # axcal0b.set_ylabel("Battery\nfraction")


    lines1, labels1 = axden1.get_legend_handles_labels()
    #lines2, labels2 = axden0.get_legend_handles_labels()
    # lines3, labels3 = axden0b.get_legend_handles_labels()

    #fig.legend(lines1 +lines3, labels1+labels3, bbox_to_anchor=(0.8, 0.1), ncol = 3)
    fig.legend(lines1, labels1, bbox_to_anchor=(0.91, 0.14), ncol = 4)


    #print(fig.axes[1::2])

    #fig.suptitle(r"$\bf{Sensitivity\;to\;Cost\;of\;Battery\;(€/Wh)}$", fontsize = 24)
    plt.savefig("Images/Figure_batt_compare_gw_var1.png")
    plt.show()
#pen_plus_batt_curtailoverlap()

