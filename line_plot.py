# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:37:49 2019

PURPOSE: Creating a lineplot with matplotlib alone or with seaborn

@author: Brian DuChez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% Line plot

#create example dataset corresponding to four columns and seven rows.  Column data will be used to plot line
data = {'Control': [0,0,0,0,0,0,0], 'Low': [1, 5, 7, 3, 2, 1, 1], 'Middle': [1, 26, 30, 21, 10, 6, 13], 'High': [1, 70, 50, 48, 23, 11, 28]}

# x-axis values
index = [0.1, 0.5, 1, 2, 4, 8, 12]

markers = ['o', '^', 'p', 'P']

# create figure and axis objects
f, ax = plt.subplots(figsize=(5, 3))

# iterate through key/value pairs in dictionary and use the counter to specify marker for each line
for n, (k, v) in enumerate(data.items()):
     plt.plot(index, data[k], marker=markers[n])
     print(data[k])

ax.set_ylabel('Concentration (ng/mL)')
ax.set_xlabel('Time (hr)')
ax.legend(data.keys())

plt.tight_layout()     
plt.show()


 #%% same plot with seaborn
#TODO: This works but markers some marker options have no color (see 'x' and '+') and legend colors don't match lines.  
 
f, ax = plt.subplots(figsize=(5, 3))
for n, (k, v) in enumerate(data.items()): 
    ax = sns.lineplot(x=index, y=data[k], style=n, markers=markers[n])

ax.set_ylabel('Concentration (ng/mL)')
ax.set_xlabel('Time (hr)')
ax.legend(data.keys())

plt.tight_layout()      
plt.show()