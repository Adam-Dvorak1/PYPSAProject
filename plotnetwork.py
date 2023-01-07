
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def flex_plus_curtailALL():
    '''This is one of the functions we use to make an abstract figure'''
    plt.rcParams.update({'font.size': 14})
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
    DNK_sp = [x[1] for x in co2dnk]
    DNK_sc = list(map(abs,[x[3] for x in co2dnk]))
    DNK_wp = [x[2] for x in co2dnk]
    DNK_gas = [x[5] for x in co2dnk]

    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axden0.scatter(DNK_gas, DNK_sc, color = "C1") #Scatter or plot?
    axden0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1, decimals = 0))
    axden0.set_ylabel("Curtailment")
    axden0.set_facecolor("#eeeeee")
    axden0.set_ylim(0, 0.32)


    axden1.stackplot(DNK_gas, DNK_sp, DNK_wp, DNK_gas, colors = ["#f1c232","#2986cc", "#cbbcf4"], labels = ["Solar", "Wind", "Gas"])
    axden1.set_ylim(0, 1)

    axden1.set_ylabel("Penetration")
    #axden1.set_xlabel("Percent flexible source")

    axden1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    axden0.spines["top"].set_visible(False)
    axden1.spines["right"].set_visible(False)
    axden0.set_title("Denmark")

    xticks = axden1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)


    ####SPAIN####

    ESP_sp = [x[1] for x in co2esp]
    ESP_sc = list(map(abs,[x[3] for x in co2esp]))
    ESP_wp = [x[2] for x in co2esp]
    ESP_gas = [x[5] for x in co2esp]

    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axesp0.scatter(ESP_gas, ESP_sc, color = "C1") #Scatter or plot?
    axesp0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1, decimals = 0))
    #axesp0.set_ylabel("Curtailment")
    axesp0.set_facecolor("#eeeeee")
    axesp0.set_title("Spain")
    axesp0.set_ylim(0, 0.32)


    axesp1.stackplot(ESP_gas, ESP_sp, ESP_wp, ESP_gas, colors = ["#f1c232","#2986cc", "#cbbcf4"])
    axesp1.set_ylim(0, 1)

    #axesp1.set_ylabel("Penetration")
    #axesp1.set_xlabel("Percent flexible source")

    axesp1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axesp0.spines["top"].set_visible(False)
    axesp1.spines["right"].set_visible(False)

    xticks = axesp1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)
    axesp0.yaxis.set_ticklabels([])
    axesp1.yaxis.set_ticklabels([])
    ####Colorado#####

    COL_sp = [x[1] for x in co2col]
    COL_sc = list(map(abs,[x[3] for x in co2col]))
    COL_wp = [x[2] for x in co2col]
    COL_gas = [x[5] for x in co2col]

    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axcol0.scatter(COL_gas, COL_sc, color = "C1") #Scatter or plot?
    axcol0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1, decimals = 0))
    axcol0.set_ylabel("Curtailment")
    axcol0.set_facecolor("#eeeeee")
    axcol0.set_title("Colorado")
    axcol0.set_ylim(0, 0.32)


    axcol1.stackplot(COL_gas, COL_sp, COL_wp, COL_gas, colors = ["#f1c232","#2986cc", "#cbbcf4"])
    axcol1.set_ylim(0, 1)

    axcol1.set_ylabel("Penetration")
    axcol1.set_xlabel("Fraction flexible source")

    axcol1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axcol0.spines["top"].set_visible(False)
    axcol1.spines["right"].set_visible(False)

    xticks = axcol1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)


    ###California####

    CAL_sp = [x[1] for x in co2cal]
    CAL_sc = list(map(abs,[x[3] for x in co2cal]))
    CAL_wp = [x[2] for x in co2cal]
    CAL_gas = [x[5] for x in co2cal]

    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axcal0.scatter(CAL_gas, CAL_sc, color = "C1") #Scatter or plot?
    axcal0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1, decimals = 0))
    #axcal0.set_ylabel("Curtailment")
    axcal0.set_facecolor("#eeeeee")
    axcal0.set_title("California")
    axcal0.set_ylim(0, 0.32)


    axcal1.stackplot(CAL_gas, CAL_sp, CAL_wp, CAL_gas, colors = ["#f1c232","#2986cc", "#cbbcf4"])
    axcal1.set_ylim(0, 1)

    #axcal1.set_ylabel("Penetration")
    axcal1.set_xlabel("Fraction flexible source")

    axcal1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axcal0.spines["top"].set_visible(False)
    axcal1.spines["right"].set_visible(False)
    
    xticks = axcal1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)

    axcal0.yaxis.set_ticklabels([])
    axcal1.yaxis.set_ticklabels([])


    for ax in plt.gcf().get_axes():
        ax.minorticks_on()
      
        ax.label_outer()

        ax.margins(x = 0)


    lines1, labels1 = axden1.get_legend_handles_labels()

    fig.legend(lines1, labels1, bbox_to_anchor=(0.7, 0.055), ncol=3)
    plt.savefig("Images/Figure3_flexible_Var3.png")
    plt.show()


def pen_plus_curtailALL():
    '''This makes a 2x2 grid of two axes each showing resource penetration and solar curtailment vs.
    a scaling log of solar. It is very long. It uses gridspec to order the axes, and other than that
    it is about the same as the other pen_plus_curtail() functions. 17/1'''
    plt.rcParams.update({'font.size': 14})
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
    s_cost = [x[0] for x in solardnk]
    s_cost = [item for sublist in s_cost for item in sublist]
    s_cost = [x / 10**6 /0.07846970300338728 for x in s_cost] 
    DNK_sp = [x[1] for x in solardnk]
    DNK_wp = [x[2] for x in solardnk]
    DNK_sc = list(map(abs,[x[3] for x in solardnk]))



    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axden0.scatter(s_cost, DNK_sc, s = 15, color = "C1") #Scatter or plot?
    axden0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axden0.set_ylabel("Curtailment")
    axden0.set_facecolor("#eeeeee")



    axden1.stackplot(s_cost, DNK_sp, DNK_wp, colors = ["#f1c232","#2986cc"], labels = ["Solar", "Wind"])
    axden1.set_ylim(0, 1)
    axden0.set_ylim(0, 1)

    axden1.set_ylabel("Penetration")
    #axden1.set_xlabel("Percent flexible source")

    axden1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axden0.set_title("Denmark")

    xticks = axden1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)



    axden1.axvline(0.529, color='black',ls='--')
    axden1.axvline(1.3, color='black',ls='--')
    #We want to make a range for today's prices. the upper range is 
    axden1.text(0.85,0.05,  "Today's range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    axden1.axvline(0.019, color='black',ls='--')
    axden1.text(0.025,0.05, "2050--Optimistic",  fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    axden1.axvline(0.095, color='black',ls='--')
    axden1.text(0.13,0.05,  "2050--Less Optimistic", fontsize = 12, horizontalalignment = "center", rotation = "vertical")

    axden0.spines["top"].set_visible(False)
    axden0.spines["right"].set_visible(False)
    plt.rcParams['hatch.linewidth'] = 1
    axden1.fill_between([0.529, 1.3], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
    


    ####SPAIN####

    ESP_sp = [x[1] for x in solaresp]
    ESP_wp = [x[2] for x in solaresp]
    ESP_sc = list(map(abs,[x[3] for x in solaresp]))



    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axesp0.scatter(s_cost, ESP_sc, s = 15, color = "C1") #Scatter or plot?
    axesp0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    #axesp0.set_ylabel("Curtailment")
    axesp0.set_facecolor("#eeeeee")
    axesp0.set_title("Spain")
 


    axesp1.stackplot(s_cost, ESP_sp, ESP_wp, colors = ["#f1c232","#2986cc"])
    axesp1.set_ylim(0, 1)
    axesp0.set_ylim(0, 1)


    #axesp1.set_ylabel("Penetration")
    #axesp1.set_xlabel("Percent flexible source")

    axesp1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))



    xticks = axesp1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)


    axesp1.axvline(0.529, color='black',ls='--')
    
    axesp1.axvline(0.019, color='black',ls='--')
    axesp1.text(0.025,0.05, "2050--Optimistic", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    axesp1.axvline(0.095, color='black',ls='--')
    axesp1.text(0.13,0.05, "2050--Less Optimistic", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    axesp0.spines["top"].set_visible(False)
    axesp0.spines["right"].set_visible(False)
    axesp0.yaxis.set_ticklabels([])
    axesp1.yaxis.set_ticklabels([])
    axesp1.fill_between([0.529, 1.3], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
    axesp1.axvline(1.3, color='black',ls='--')
    axesp1.text(0.85,0.05,  "Today's range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")

    ####Colorado#####

    COL_sp = [x[1] for x in solarcol]
    COL_wp = [x[2] for x in solarcol]
    COL_sc = list(map(abs,[x[3] for x in solarcol]))


    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axcol0.scatter(s_cost, COL_sc, s = 15, color = "C1") #Scatter or plot?
    axcol0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axcol0.set_ylabel("Curtailment")
    axcol0.set_facecolor("#eeeeee")
    axcol0.set_title("Colorado")



    axcol1.stackplot(s_cost, COL_sp, COL_wp, colors = ["#f1c232","#2986cc"])
    axcol1.set_ylim(0, 1)
    axcol0.set_ylim(0, 1)

    axcol1.set_ylabel("Penetration")
    axcol1.set_xlabel("Cost of Solar")

    axcol1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))


    xticks = axcol1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)

    axcol1.axvline(0.529, color='black',ls='--')
    
    axcol1.axvline(0.019, color='black',ls='--')
    axcol1.text(0.025,0.05, "2050--Optimistic", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    axcol1.axvline(0.095, color='black',ls='--')
    axcol1.text(0.13,0.05, "2050--Less Optimistic", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    axcol0.spines["top"].set_visible(False)
    axcol0.spines["right"].set_visible(False)
    axcol1.fill_between([0.529, 1.3], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
    axcol1.axvline(1.3, color='black',ls='--')
    axcol1.text(0.85,0.05,  "Today's range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    ###California####

    CAL_sp = [x[1] for x in solarcal]
    CAL_wp = [x[2] for x in solarcal]
    CAL_sc = list(map(abs,[x[3] for x in solarcal]))



    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axcal0.scatter(s_cost, CAL_sc, s = 15, color = "C1") #Scatter or plot?
    axcal0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    #axcal0.set_ylabel("Curtailment")
    axcal0.set_facecolor("#eeeeee")
    axcal0.set_title("California")



    axcal1.stackplot(s_cost, CAL_sp, CAL_wp, colors = ["#f1c232","#2986cc"])
    axcal1.set_ylim(0, 1)
    axcal0.set_ylim(0, 1)


    #axcal1.set_ylabel("Penetration")
    axcal1.set_xlabel("Cost of Solar")

    axcal1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    xticks = axcal1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)

    axcal1.axvline(0.529, color='black',ls='--')
  
    axcal1.axvline(0.019, color='black',ls='--')
    axcal1.text(0.025,0.05, "2050--Optimistic", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    axcal1.axvline(0.095, color='black',ls='--')
    axcal1.text(0.13,0.05, "2050--Less Optimistic", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    axcal0.spines["top"].set_visible(False)
    axcal0.spines["right"].set_visible(False)
    axcal0.yaxis.set_ticklabels([])
    axcal1.yaxis.set_ticklabels([])
    axcal1.fill_between([0.529, 1.3], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
    axcal1.axvline(1.3, color='black',ls='--')
    axcal1.text(0.85,0.05,  "Today's range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
 
    for ax in plt.gcf().get_axes():
        ax.minorticks_on()
      
        ax.label_outer()

        ax.margins(x = 0)
        ax.set_xscale('log')        
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))
    


    axesp1.xaxis.set_ticklabels([])
    axden1.xaxis.set_ticklabels([])

    lines1, labels1 = axden1.get_legend_handles_labels()

    fig.legend(lines1, labels1, bbox_to_anchor=(0.65, 0.055), ncol=3)
    plt.savefig("Images/Figure2_solar_compare1.png")
    plt.show()
    


def pen_plus_wind_curtailALL():
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



    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axden0.scatter(w_cost, DNK_wc, s = 15, color = "C1") #Scatter or plot?
    axden0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axden0.set_ylabel("Curtailment")
    axden0.set_facecolor("#eeeeee")



    axden1.stackplot(w_cost, DNK_sp, DNK_wp, colors = ["#f1c232","#2986cc"], labels = ["Solar", "Wind"])
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

    axesp0.scatter(w_cost, ESP_wc, s = 15, color = "C1") #Scatter or plot?
    axesp0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    axesp0.set_facecolor("#eeeeee")
    axesp0.set_title("Spain")
 


    axesp1.stackplot(w_cost, ESP_sp, ESP_wp, colors = ["#f1c232","#2986cc"])
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

    axcol0.scatter(w_cost, COL_wc, s = 15, color = "C1") #Scatter or plot?
    axcol0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axcol0.set_ylabel("Curtailment")
    axcol0.set_facecolor("#eeeeee")
    axcol0.set_title("Colorado")



    axcol1.stackplot(w_cost, COL_sp, COL_wp, colors = ["#f1c232","#2986cc"])

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




    axcal0.scatter(w_cost, CAL_wc, s = 15, color = "C1") #Scatter or plot?
    axcal0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    axcal0.set_facecolor("#eeeeee")
    axcal0.set_title("California")



    axcal1.stackplot(w_cost, CAL_sp, CAL_wp, colors = ["#f1c232","#2986cc"])

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

    fig.legend(lines1, labels1, bbox_to_anchor=(0.65, 0.055), ncol=3)

    #print(fig.axes[1::2])

    plt.savefig("Images/Figure_wind_compare_heat_var2.png")
    plt.show()


def pen_plus_batt_curtailALL():
    '''This makes a 2x2 grid of two axes each showing resource penetration and solar curtailment vs.
    a scaling log of solar. It is very long. It uses gridspec to order the axes, and other than that
    it is about the same as the other pen_plus_curtail() functions. 17/1
    
    This uses the github (Danish energy agency) value of onshore wind for the lower bound,
    and the NREL annual technology baseline for the upper bound.'''
    plt.rcdefaults()

    plt.rcParams.update({'font.size': 14})
    plt.rcParams['hatch.linewidth'] = 1
    fig = plt.figure(figsize=(10, 9))
    outer = gridspec.GridSpec(3, 2, wspace=0.2, hspace=0.3, height_ratios = [1, 1, 0.01])
    inner_dnk = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], wspace=0.1, hspace=0, height_ratios = [1, 2])
    inner_esp = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], wspace=0.1, hspace=0, height_ratios = [1, 2])
    inner_col = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2], wspace=0.1, hspace=0, height_ratios = [1, 2]) #(gridspec(3,1)) to add some extra space
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
    b_cost = [x[7] for x in battdnk]
    b_cost = [item for sublist in b_cost for item in sublist]
    b_cost = [x / 10**6 /0.09439292574325567  for x in b_cost] 

    DNK_sp = [x[1] for x in battdnk]
    DNK_wp = [x[2] for x in battdnk]
    DNK_sc = list(map(abs,[x[3] for x in battdnk]))
    DNK_wc = list(map(abs,[x[4] for x in battdnk]))
    DNK_bp = list(map(abs,[x[8] for x in battdnk]))



    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]})

    axden0.scatter(b_cost, DNK_sc, s = 15, color = "C1", label = "solar curtailment")
    axden0.scatter(b_cost, DNK_wc, s = 15, color = "C0", label = "wind curtailment")
       
    axden0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axden0.set_ylabel("Curtailment")
    axden0.set_facecolor("#eeeeee")



    axden1.stackplot(b_cost, DNK_sp, DNK_wp, colors = ["#f1c232","#2986cc"], labels = ["Solar", "Wind"])
    axden1.set_ylim(0, 1)
    axden0.set_ylim(0, 1)

    axden1.set_ylabel("Penetration")


    axden1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axden0.set_title("Denmark")

    xticks = axden1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)


    #We want to make a range for today's prices. the upper range is 
    axden0.spines["top"].set_visible(False)
    axden0.spines["right"].set_visible(False)
   


    ####SPAIN####

    ESP_sp = [x[1] for x in battesp]
    ESP_wp = [x[2] for x in battesp]
    ESP_wc = list(map(abs,[x[4] for x in battesp]))
    ESP_sc = list(map(abs,[x[3] for x in battesp]))
    ESP_bp = list(map(abs,[x[8] for x in battesp]))

    axesp0.scatter(b_cost, ESP_wc, s = 15, color = "C0") 
    axesp0.scatter(b_cost, ESP_sc, s = 15, color = "C1") 
    axesp0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    axesp0.set_facecolor("#eeeeee")
    axesp0.set_title("Spain")




    axesp1.stackplot(b_cost, ESP_sp, ESP_wp, colors = ["#f1c232","#2986cc"])
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

    COL_sp = [x[1] for x in battcol]
    COL_wp = [x[2] for x in battcol]
    COL_wc = list(map(abs,[x[4] for x in battcol]))
    COL_sc = list(map(abs,[x[3] for x in battcol]))    
    COL_bp = list(map(abs,[x[8] for x in battcol]))

    axcol0.scatter(b_cost, COL_wc, s = 15, color = "C0") 
    axcol0.scatter(b_cost, COL_sc, s = 15, color = "C1")     
    axcol0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    axcol0.set_ylabel("Curtailment")
    axcol0.set_facecolor("#eeeeee")
    axcol0.set_title("Colorado")



    axcol1.stackplot(b_cost, COL_sp, COL_wp, colors = ["#f1c232","#2986cc"])

    axcol1.set_ylim(0, 1)
    axcol0.set_ylim(0, 1)

    axcol1.set_ylabel("Penetration")
    axcol1.set_xlabel("Cost of Battery")

    axcol1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))


    xticks = axcol1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)

    axcol0.spines["top"].set_visible(False)
    axcol0.spines["right"].set_visible(False)

    
    
    ###California####

    CAL_sp = [x[1] for x in battcal]
    CAL_wp = [x[2] for x in battcal]
    CAL_wc = list(map(abs,[x[4] for x in battcal]))
    CAL_sc = list(map(abs,[x[3] for x in battcal]))
    CAL_bp = list(map(abs,[x[8] for x in battcal]))    




    axcal0.scatter(b_cost, CAL_wc, s = 15, color = "C0")
    axcal0.scatter(b_cost, CAL_sc, s = 15, color = "C1")
    axcal0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    axcal0.set_facecolor("#eeeeee")
    axcal0.set_title("California")



    axcal1.stackplot(b_cost, CAL_sp, CAL_wp, colors = ["#f1c232","#2986cc"])

    axcal1.set_ylim(0, 1)
    axcal0.set_ylim(0, 1)


    axcal1.set_xlabel("Cost of Battery")

    axcal1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

    xticks = axcal1.yaxis.get_major_ticks() 
    xticks[-1].label1.set_visible(False)


  
    axcal0.spines["top"].set_visible(False)
    axcal0.spines["right"].set_visible(False)
    axcal0.yaxis.set_ticklabels([])
    axcal1.yaxis.set_ticklabels([])

 
    #This applies things for all axes

    #This applies things for only the axes of penetration.
    for ax in plt.gcf().get_axes()[1::2]:
        ax.fill_between([0.232, 0.311], y1 = 1, alpha = 0.3, edgecolor = "k", hatch = "//", facecolor = "gray")
        ax.axvline(0.232, color='black',ls='--')
        ax.axvline(0.311, color='black',ls='--')
        ax.text(0.27 ,0.05,  "Today's range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")

        ax.fill_between([0.075, 0.22], y1 = 1, alpha = 0.2, edgecolor = "k", hatch = "XX", facecolor = "purple")
        ax.axvline(0.075, color='black',ls='--')
        ax.axvline(0.22, color='black',ls='--')
        ax.text(0.135, 0.05, "Future range", fontsize = 12, horizontalalignment = "center", rotation = "vertical")
    
    # for ax in plt.gcf().get_axes()[::2]:
    #     ax.set_ylim(0,1)


    axesp1.xaxis.set_ticklabels([])
    axden1.xaxis.set_ticklabels([])






    for ax in plt.gcf().get_axes():
        ax.minorticks_on()
      
        ax.label_outer()

        ax.margins(x = 0)
        ax.set_xscale('log')        
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))

    
#    plt.rcParams["font.weight"] = "bold"
    axden0b = axden0.twinx()
    axden0b.scatter(b_cost, DNK_bp, s = 15, color = "C2", label = "Battery") 
    axden0b.tick_params(axis = "y", labelcolor = "C2")

    axesp0b = axesp0.twinx()
    axesp0b.scatter(b_cost, ESP_bp, s = 15, color = "C2", label = "battery penetration")
    axesp0b.tick_params(axis = "y", labelcolor = "C2") 
    axesp0b.set_ylabel("Battery\nfraction")
 
    axcol0b = axcol0.twinx()
    axcol0b.scatter(b_cost, COL_bp, s = 15, color = "C2", label = "battery penetration") 
    axcol0b.tick_params(axis = "y", labelcolor = "C2") 

    axcal0b = axcal0.twinx()
    axcal0b.scatter(b_cost, CAL_bp, s = 15, color = "C2", label = "battery penetration") 
    axcal0b.tick_params(axis = "y", labelcolor = "C2")
    axcal0b.set_ylabel("Battery\nfraction")


    lines1, labels1 = axden1.get_legend_handles_labels()
    #lines2, labels2 = axden0.get_legend_handles_labels()
    lines3, labels3 = axden0b.get_legend_handles_labels()

    fig.legend(lines1 +lines3, labels1+labels3, bbox_to_anchor=(0.75, 0.1), ncol = 3)

    #print(fig.axes[1::2])

    plt.savefig("Images/Figure_batt_compare2.png")
    plt.show()


