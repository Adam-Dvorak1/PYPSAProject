
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2,1)

x  = [.1, .2, .3, .4, .6]
y = [5, 4, 3, 2, 1]
y2 = [6, 7, 8, 9, 10]
axs[0].plot(x, y)
axs[1].scatter(x, y2)
axs[1].fill_between(x, y2, color = "#ffd966")
axs[1].axvline(0.529, color='black',ls='--')
axs[1].text(0.529 + 0.02,0.75, "Current cost = 529000 EUR", horizontalalignment = "center", rotation = "vertical")
axs[1].axvline(0.019, color='black',ls='--')
axs[1].text(0.019+0.02,0.75, "Optimistic cost = 19000 EUR", horizontalalignment = "center", rotation = "vertical")
axs[1].axvline(0.095, color='black',ls='--')
axs[1].text(0.095+0.02,0.75, "Less optimistic cost = 95000 EUR", horizontalalignment = "center", rotation = "vertical")
axs[0].spines[["top","right"]].set_visible(False)

for ax in axs.flat:
    
    
    ax.label_outer()


plt.subplots_adjust(hspace = 0)
plt.show()