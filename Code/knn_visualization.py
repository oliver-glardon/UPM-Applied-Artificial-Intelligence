import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pandas.plotting import table
#_______________________________________________________________________________________________________________________
# Set the current directory
os.chdir('../')

# load result data
df = pd.read_pickle('Data/knn_results.pickle')
print(df)

# create table
df_results = pd.DataFrame(columns=['Number of neighbors', 'Average', 'Max', 'Min', 'Average runtime [sec]'])
pos_count = 0
num_neigh_array = [1, 3, 5, 7, 9]
for i in num_neigh_array:
    df_results.loc[pos_count] = [str(i),
                                 df[df['Neighbors']==str(i)]['Prediction Accuracy'].mean(),
                                 df[df['Neighbors']==str(i)]['Prediction Accuracy'].max(),
                                 df[df['Neighbors'] == str(i)]['Prediction Accuracy'].min(),
                                 df[df['Neighbors'] == str(i)]['Runtime [sec]'].mean()
                                 ]
    pos_count = pos_count + 1

print(df_results)
Path = r'C:\Users\Oliver\Documents\Dokumente\Studium\M.Sc. UPM\Intelligencia Artificial Applicada\Trabajo' \
             '\knn_results.csv'
df_results.to_csv(Path, index=False, header=True)

'''
Alternative plot visualization
Reference: https://python-graph-gallery.com/11-grouped-barplot/

# gropus are the different numbers of neighbors
groups = [
    df[df['Neighbors']==str(1)]['Prediction Accuracy'],
    df[df['Neighbors']==str(3)]['Prediction Accuracy'],
    df[df['Neighbors']==str(5)]['Prediction Accuracy'],
    df[df['Neighbors']==str(7)]['Prediction Accuracy'],
    df[df['Neighbors']==str(9)]['Prediction Accuracy']
]
group_labels = ['1 Neighbor', '3 Neighbors', '5 Neighbors', '7 Neighbors', '9 Neighbors']

# Convert data to pandas DataFrame.
df = pd.DataFrame(groups, index=group_labels).T

# Plot.
pd.concat(
    [df.mean().rename('average'), df.min().rename('min'),
     df.max().rename('max')],
    axis=1).plot.bar()

plt.show()
'''