# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:02:23 2022

@author: au485969

Modified 24 May au699305
"""
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import cm
import geopandas
import warnings
warnings.filterwarnings("ignore")
import os.path

arrows = False
case = 'inland demand'



# cmap = cm.get_cmap('terrain', 12)  # matplotlib color palette name, n colors
# for i in range(cmap.N):
#     rgb = cmap(i)[:3]  # will return rgba, we take only first 3 so we get rgb

# colpal1 = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in np.arange(12)]

# cmap = cm.get_cmap('gist_rainbow', 12)  # matplotlib color palette name, n colors
# for i in range(cmap.N):
#     rgb = cmap(i)[:3]  # will return rgba, we take only first 3 so we get rgb

# colpal2 = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in np.arange(12)]


Europe_ia3 = {  
            'Albania':'ALB',
            'Austria':'AUT',
            'Bosnia_Herzegovina':'BIH', 
            'Belgium':'BEL',
            'Bulgaria':'BGR', 
            'Switzerland':'CHE', 
            'Czech_Republic':'CZE', 
            'Germany':'DEU', 
            'Denmark':'DNK', 
            'Estonia':'EST', 
            'Spain':'ESP', 
            'Finland':'FIN', 
            'France':'FRA', 
            'United_Kingdom':'GBR', 
            'Greece':'GRC', 
            'Croatia':'HRV', 
            'Hungary':'HUN', 
            'Ireland':'IRL', 
            'Italy':'ITA', 
            'Lithuania':'LTU', 
            'Luxembourg':'LUX', 
            'Latvia':'LVA', 
            'Montenegro':'MNE', 
            'Macedonia':'MKD', 
            'Netherlands':'NLD', 
            'Norway':'NOR', 
            'Poland':'POL', 
            'Portugal':'PRT', 
            'Romania':'ROU', 
            'Serbia':'SRB', 
            'Sweden':'SWE', 
            'Slovenia':'SVN', 
            'Slovakia':'SVK'
            }

cdict = {'Bosnia_Herzegovina':'Bosnia and Herzegovina',
         'Czech_Republic': 'Czechia',
         'United_Kingdom':'United Kingdom',
         'Macedonia':'North Macedonia'}

# import_relative = pd.read_csv('../data/Import_percentages.csv',index_col=0)

# import_relative = import_relative.T
# import_relative['Other'] = import_relative['Not specified'] + import_relative['Other Europe'] + import_relative['Other non-Europe']
# import_relative.drop(columns=['Not specified','Other Europe','Other non-Europe'],inplace=True)
# import_relative = import_relative.T

# cons_by_sector = pd.read_csv('../data/gas_cons_by_sector.csv',index_col=0)
# cons_by_sector = cons_by_sector.T
# cons_by_sector['Other'] = cons_by_sector['Transport'] + cons_by_sector['District heating plants']
# cons_by_sector.drop(columns=['Transport','District heating plants'],inplace=True)
# cons_by_sector = cons_by_sector.T

country_coord = pd.read_csv('Countries_lat_lon.csv',
                            sep=';',header=None) # Country center coordinates
country_coord.columns = ['Code','Lat','Lon','Country']
country_coord.set_index('Country',inplace=True)

country_coord.loc['Italy'] = ['IT',41.4,15]
country_coord.loc['Denmark'] = ['DK',56.3365,10]
country_coord.loc['Portugal'] = ['PT',39.3999,-7]
country_coord.loc['Belgium'] = ['BG',50.5039,5.5]
country_coord.loc['Netherlands'] = ['NL',52.1326,6]
country_coord.loc['Switzerland'] = ['CH',46.8182,8.5]
country_coord.loc['Sweden'] = ['SE',60.5,15.5]
country_coord.loc['United Kingdom'] = ['GB',52.5,-1]
country_coord.loc['Ireland'] = ['IR',53,-7]
country_coord.loc['Bosnia and Herzegovina'] = ['BA',43.9159,19]
country_coord.loc['Serbia'] = ['RS',43.5,21]
country_coord.loc['Greece'] = ['GR',39.0742,22.5]
country_coord.loc['Croatia'] = ['HR',45.1,16.4]

# max_val = gas_df['gas_consumption_2019_TWh'].max()
# max_val = gas_df['gas_imports_2019_TWh'].max()



fig, ax = plt.subplots() # Initialise figure

for c in Europe_ia3.keys():

    
    #Shape files: https://www.gadm.org/download_country_v3.html

    mycountry = geopandas.read_file('shapefiles/gadm36_' + Europe_ia3[c] + '_0.shp')
    mycountry = mycountry.to_crs("EPSG:3395")
    mycountry.boundary.plot(ax = ax, color = "black")


    
    # ax.annotate(CC_plot.Code,xy=m_plot(np.array(CC_plot.Lon.item()-1), np.array(CC_plot.Lat.item())),color='k', zorder=13)
    # x,y=m_plot(np.array(CC_plot.Lon.item()-1), np.array(CC_plot.Lat.item()))
    # m_plot.plot(x,y,"o",markersize=gas_abs/max_val*30,color='magenta',zorder=100)

# xlim = ax.get_xlim()[1] - ax.get_xlim()[0]
# ylim = ax.get_ylim()[1] - ax.get_ylim()[0]

#plt.show()
# # ax.text(0.02*xlim,0.95*ylim,'2019 EU-28+NO+CH gas ' + case,fontsize=fs,fontweight='bold')
# ax.text(0.02*xlim,0.95*ylim,'2019 gas demand',fontsize=fs,fontweight='bold')
# ax.plot(0.05*xlim,0.9*ylim,'o',markersize=900/max_val*30,color='magenta')
# # ax.text(0.1*xlim,0.9*ylim,str(round(max_val)) + ' TWh',fontsize=fs)
# ax.text(0.1*xlim,0.9*ylim,'900 TWh',fontsize=fs)

# ax.plot(0.05*xlim,0.85*ylim,'o',markersize=900/max_val*15,color='magenta')
# # ax.text(0.1*xlim,0.84*ylim,str(round(0.5*max_val)) + ' TWh',fontsize=fs)
# ax.text(0.1*xlim,0.84*ylim,'450 TWh',fontsize=fs)

# max_width = 0.05
# x = 0.8*xlim
# y = 0.8*ylim
# u,v=1,0
# m_plot.quiver(x,0.9*ylim,u,v,width=max_width*(40/38),headlength=0.5,headaxislength=0.5,headwidth=1.5)
# m_plot.quiver(x,0.84*ylim,u,v,width=max_width*(20/38),headlength=0.5,headaxislength=0.5,headwidth=1.5)

# ax.text(x,0.95*ylim,'Gas origin',fontsize=fs)
# ax.text(0.88*xlim,0.9*ylim,'40 %',fontsize=fs)
# ax.text(0.88*xlim,0.835*ylim,'20 %',fontsize=fs)

# cb_ax = fig.add_axes([0.85,0.295,0.02,0.21])
# norm = mpl.colors.Normalize(vmin=-10, vmax=gas_df['perc_of_pe'].max())    
# cb1 = mpl.colorbar.ColorbarBase(cb_ax,orientation='vertical', cmap=cmap,norm=norm,boundaries=np.arange(0,gas_df.max().perc_of_pe)) #,ticks=bounds, boundaries=bounds) #ticks=[0.15,0.25,0.48,0.90])
# cb1.set_label('Gas % of total energy demand',zorder=10,fontsize=fs)
# cb1.ax.tick_params(labelsize=fs)

# #%% Pie chart - import
# pc_imp_ax = fig.add_axes([1.25,0.49,0.35,0.2])
# pc_imp_ax.pie(import_relative['imp'].values,labels=import_relative.index,colors=colpal1[0:len(import_relative['imp'].values)],startangle = 90,textprops={'fontsize': 19})

# pc_imp_ax.set_title('Gas origin (2019)',fontsize=fs,fontweight='bold')

# #%% Pie chart - consumption by sectors
# pc_sec_ax = fig.add_axes([1.25,0.29,0.35,0.2])
# pc_sec_ax.pie(cons_by_sector['consumption [TWh]'].values,labels=cons_by_sector.index,colors=colpal2[0:len(import_relative['imp'].values)],startangle = 90,textprops={'fontsize': 19})

# pc_sec_ax.set_title('Consumption by sector (2015)',fontsize=fs,fontweight='bold')

# pc_sec_ax.text(0.95,-0.5,'(Individual heating',fontsize=19)
# pc_sec_ax.text(1,-0.65,'+ cooking)',fontsize=19)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

#%% Arrows

# if arrows == True:
#     max_width = 0.05
    
#     # From Russia
#     width = max_width #  38 %
#     lon = 28
#     lat = 53
#     x,y=m_plot(lon,lat)
#     u,v=-2,-1.5
#     m_plot.quiver(x,y,u,v,width=width,headlength=0.5,headaxislength=0.5,headwidth=1.5)
#     ax.text(1.01*x,y,'Russia',fontsize=fs)
    
#     # From Algeria 
#     width = max_width*(7/38) # 7 %
#     lon = 4
#     lat = 36
#     x,y=m_plot(lon,lat)
#     u,v=0,1
#     m_plot.quiver(x,y,u,v,width=width,headlength=0.5,headaxislength=0.5,headwidth=2)
#     ax.text(x*1.05,y*1.2,'Algeria',fontsize=fs)
    
#     # Own production
#     width = max_width*(29/38) # 18 % from Norway + 5 % from Netherlands + 6 % from other European
#     lon = 0.3
#     lat = 62
#     x,y=m_plot(lon,lat)
#     u,v=0.6,-1.9
#     m_plot.quiver(x,y,u,v,width=width,headlength=0.5,headaxislength=0.5,headwidth=1.5)
#     ax.text(0.85*x,1.065*y,'Own',fontsize=fs)
#     ax.text(0.75*x,1.02*y,'production',fontsize=fs)
    
#     # Other non-Europe
#     width = max_width*((17 + 7)/38) # 17 % + 7 % not specified
#     lon = -8
#     lat = 47
#     x,y=m_plot(lon,lat)
#     u,v=1,0
#     m_plot.quiver(x,y,u,v,width=width,headlength=0.5,headaxislength=0.5,headwidth=2)
#     ax.text(0.1*x,y*0.95,'Other',fontsize=fs)
#     ax.text(0.1*x,y*0.86,'non-European',fontsize=fs)

#     #fig.savefig('../figures/Gas_' + case + '_map_with_arrowstest.jpeg',format='jpeg',dpi=300,bbox_inches='tight')  

# #else:
    #fig.savefig('../figures/Gas_' + case + '_map_wo_arrowstest.jpeg',format='jpeg',dpi=300,bbox_inches='tight')  

plt.show()
# fig.savefig('../figures/Gas_imports_map_wo_arrows.jpeg',format='jpeg',dpi=300,bbox_inches='tight')  