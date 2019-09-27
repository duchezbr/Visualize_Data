# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:24:21 2019

Purpose: Creating a simple bar plot using either matplotlib alone or with seaborn

@author: 604255
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%% same graph without seaborn
f1, ax1= plt.subplots(figsize=(5, 3))

# can reference data of df; note that ax or plt can be used to apply data to figure
#ax1.bar(data['objective'], data['complete'])
plt.bar(df['objective'], df['complete'])

# not different ways you can set ax1 
ax1.set_ylabel('Percent Complete')
ax1.set(xlabel='Objectives')

f1.tight_layout()

#%%
sns.set(style="whitegrid")

data = {'complete': [75, 15, 35, 45, 55], 'objective':['Obj 1', 'Obj 2', 'Obj 3', 'Obj 4', 'Obj 5']}
df = pd.DataFrame(data=data)

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(5, 3))

# can reference data or df; sns.set_color_codes("pastel") is an option
sns.barplot(x="objective", y="complete", data=data, label="Total", color="r")

ax.set(ylabel='Percent Complete')
ax.set(xlabel='')
f.tight_layout()

#%% clear all figures
#plt.close(fig="all")