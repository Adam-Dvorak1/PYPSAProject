# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:02:23 2022

@author: au485969

@author: au699305

"""

'''This plot was made four times: for scenarios between sectors/no sectors, transmission/no transmission'''
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import cm
import warnings
import os
import conda
from Big_model_script import find_solar_share
warnings.filterwarnings("ignore")

'''I was having trouble importing basemap, even though it was installed. This was what got it to work.

I still cannot use resolution of basemap better than low "l" resolution. I could not use intermediate
 'i' resolution'''
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

from mpl_toolkits.basemap import Basemap


arrows = False
case = 'inland demand'

fs = 22

cmap = cm.get_cmap('terrain', 20)  # matplotlib color palette name, n colors
for i in range(cmap.N):
    rgb = cmap(i)[:3]  # will return rgba, we take only first 3 so we get rgb

colpal1 = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in np.arange(12)]
colpal2 = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in np.arange(12)+12]


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
         'Czech_Republic': 'Czech Republic',
         'United_Kingdom':'United Kingdom',
         'Macedonia': 'Macedonia [FYROM]'}


country_coord = pd.read_csv('Countries_lat_lon.csv',
                            sep=';',header=None) # Country center coordinates
country_coord.columns = ['Code','Lat','Lon','Country']
country_coord.set_index('Country',inplace=True)#Setting index to country





#corrections to coordinate, maybe
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

relpath = "results/18Jan2023_ReviewerNewCosts/csvs/"
#file = "0.603_gen_and_lat_no_sectors_no_transmission.csv"
# file = "0.603_gen_and_lat_no_sectors_yes_transmission.csv"
file = "0.603_gen_and_lat_yes_sectors_no_transmission.csv"
# file = "0.603_gen_and_lat_yes_sectors_yes_transmission.csv"


filepath = relpath + file

gen_latdf = pd.read_csv(filepath, index_col = 0)


gen_latdf = gen_latdf[gen_latdf['carrier'] == 'solar']

gen_latdf.set_index('name_y', inplace = True)#For some reason I have two "name"s lol


max_val = 1 #We want the maximum value to be 1


norm = plt.Normalize(0, 1)
cmap = cm.get_cmap('Reds', 12)
cmap.set_under('0.8')
fig, ax = plt.subplots(1,1,figsize=(10,20)) # Initialise figure
m_plot = Basemap(width=11500000/2.9,height=9000000/2.2,projection='laea',
                  resolution='l',lat_0=54.5,lon_0=9.5,ax=ax)

for c in Europe_ia3.keys():
    if c in ['Bosnia_Herzegovina', "Macedonia"]:
        CC_plot = country_coord.loc[cdict[c]]
        gas_rel = gen_latdf.loc[cdict[c]].solarfrac
    elif c  == 'United_Kingdom':
        CC_plot = country_coord.loc[cdict[c]]
        gas_rel = gen_latdf.loc[cdict[c]].solarfrac#Finds the row with % coming from Russia
    # elif c in ['Albania', "Switzerland", "Montenegro", "Norway", 'Serbia']:
    #     CC_plot = country_coord.loc[c]
    #     gas_rel = -1
    elif c == 'Czech_Republic':
        CC_plot = country_coord.loc['Czech Republic']
        gas_rel = gen_latdf.loc[cdict[c]].solarfrac
    else:
        CC_plot = country_coord.loc[c]
        gas_rel = gen_latdf.loc[c].solarfrac
    
    #Shape files: https://www.gadm.org/download_country_v3.html
    m_plot.readshapefile('shapefiles/gadm36_' + Europe_ia3[c] + '_0',c,drawbounds=True,linewidth = 0.1,color='k')
    patches = []
    value = gas_rel
    for info, shape in zip(eval('m_plot.' + c + '_info'), eval('m_plot.' + c)):
        patches.append(Polygon(np.array(shape), True))
    patch1=ax.add_collection(PatchCollection(patches, facecolor= cmap(norm(value))))
    
    x,y=m_plot(np.array(CC_plot.Lon.item()-1), np.array(CC_plot.Lat.item()))


xlim = ax.get_xlim()[1] - ax.get_xlim()[0]
ylim = ax.get_ylim()[1] - ax.get_ylim()[0]


solshare = find_solar_share(filepath)

ax.text(0.5, 0.5, f'{solshare}%', color = 'r', fontsize = 30)

opts = filepath.split("_")

sectors = opts[opts.index('sectors')-1]

transmission = opts[opts.index('transmission.csv')-1]

if transmission == "no" and sectors == "yes":
    cb_ax = fig.add_axes([0.85,0.295,0.02,0.21]) 
    norm = mpl.colors.Normalize(vmin=0, vmax=1)    
    bounds = [0, 0.2, 0.4, 0.6, 0.8, 1]
    cb1 = mpl.colorbar.ColorbarBase(cb_ax,orientation='vertical', cmap=cmap,norm=norm,boundaries=bounds) #,ticks=bounds, boundaries=bounds) #ticks=[0.15,0.25,0.48,0.90])
    cb1.set_label('Fraction of electricity from local solar',zorder=10,fontsize=fs)
    cb1.ax.tick_params(labelsize=fs)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)






#fig.savefig('Images/Latitude/19JanNewCostEurMap_'+ sectors + '_sec_' + transmission + '_trans.png',format='png',bbox_inches='tight')  

plt.show()
# fig.savefig('../figures/Gas_imports_map_wo_arrows.jpeg',format='jpeg',dpi=300,bbox_inches='tight')  