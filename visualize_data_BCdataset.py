# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:14:44 2019

PURPOSE: Reference for data visualization.  Several 1-3D graph options using BC dataset 
see https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57 for more extensive options

@author: duchezbr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer

#%%
data = load_breast_cancer()
feature = data.data[:, 0:10]
target = data.target[:]
target_labels = data.target_names
df = pd.DataFrame(data=feature)
df['label'] = target
adict = {0: 'malignant', 1: 'benign'}
df['label_name'] = df.label.map(adict)

#%% Correlation Matrix Heatmap

f, ax = plt.subplots(figsize=(10, 6))
corr = df.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('BC Attributes Correlation Heatmap', fontsize=14)

#%% Pair-wise Scatter Plots
#cols = [0 ,1, 2, 3]
cols = np.arange(0,4)
pp = sns.pairplot(df[cols], size=1.8, aspect=1.8,

                  plot_kws=dict(edgecolor="k", linewidth=0.5),

                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 

fig.subplots_adjust(top=0.93, wspace=0.3)

t = fig.suptitle('BC Attributes Pairwise Plots', fontsize=14)


#%% Scaling attribute values to avoid few outiers

cols = list(range(10))

subset_df = df[cols]

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

scaled_df = ss.fit_transform(subset_df)

scaled_df = pd.DataFrame(scaled_df, columns=cols)

final_df = pd.concat([scaled_df, df['label_name']], axis=1)

final_df.head()


# plot parallel coordinates

from pandas.plotting import parallel_coordinates

pc = parallel_coordinates(final_df, 'label_name', color=('#FFE888', '#FF9999'))
                                                    
#%% Joint Plot

jp = sns.jointplot(x=0, y=5, data=df,
                   kind='reg', space=0, size=5, ratio=4)


#%% Box Plots

f, (ax) = plt.subplots(1, 1, figsize=(12, 4))

f.suptitle('Benign/Malignant - Attribute', fontsize=14)



sns.boxplot(x="label_name", y=0, data=df,  ax=ax)

ax.set_xlabel("Benign/Malignant",size = 12,alpha=0.8)

ax.set_ylabel("0",size = 12,alpha=0.8)

#%% Violin Plots

f, (ax) = plt.subplots(1, 1, figsize=(12, 4))

f.suptitle('Benign/Malignant - Attribute', fontsize=14)

sns.violinplot(x="label_name", y=0, data=df,  ax=ax)

ax.set_xlabel("Benign/Malignant",size = 12,alpha=0.8)

ax.set_ylabel("0",size = 12,alpha=0.8)


#%% Scatter Plot with Hue for visualizing data in 3-D

cols = [0, 1, 2, 3, 'label_name']

pp = sns.pairplot(df[cols], hue='label_name', size=1.8, aspect=1.8, 

                  palette={'malignant': "#FF9999", 'benign': "#FFE888"},

                  plot_kws=dict(edgecolor="black", linewidth=0.5))

fig = pp.fig 

fig.subplots_adjust(top=0.93, wspace=0.3)

t = fig.suptitle('BC Attributes Pairwise Plots', fontsize=14)


#%% Visualizing 3-D numeric data with Scatter Plots

# length, breadth and depth
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111, projection='3d')


xs = df[0]

ys = df[9]

zs = df[19]

ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')

ax.set_xlabel(0)

ax.set_ylabel(9)

ax.set_zlabel(19)


#%% Visualizing 3-D numeric data with a bubble chart

# length, breadth and size

plt.scatter(df[0], df[1], s=df[2]*2.5, 

            alpha=0.4, edgecolors='w')

plt.xlabel(0)

plt.ylabel(1)

plt.title('0 - 1 - 2',y=1.05)


#%% Visualizing 3-D mix data using scatter plots

df.columns = df.columns.astype(str)
# leveraging the concepts of hue for categorical dimension

jp = sns.pairplot(df, x_vars=['0'], y_vars=['1'], size=4.5,

                  hue="label_name", palette={"malignant": "#FF9999", "benign": "#FFE888"},

                  plot_kws=dict(edgecolor="k", linewidth=0.5))

                  

# we can also view relationships\correlations as needed                  

lp = sns.lmplot(x='0', y='1', hue='label_name', 

                palette={"malignant": "#FF9999", "benign": "#FFE888"},

                data=df, fit_reg=True, legend=True,

                scatter_kws=dict(edgecolor="k", linewidth=0.5)) 