# Group Information: Intelligencia Artificial Applicada, UPM
# Emiliano Capogrossi, M18029
# Oliver Glardon, M19936
# Sorelys Sandoval, M19237
#_______________________________________________________________________________________________________________________
# imports
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import pandas as pd
import time
from sklearn.metrics import confusion_matrix
from utils.plot_cm import plot_confusion_matrix

# Variables
names = ['M18029', 'M19936', 'M19237']
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
techniques = ('KNN', 'BAYES','MLP', 'SOM', 'K-MEANS')
running_time = []
percentage_of_correct_predictions = []

#_______________________________________________________________________________________________________________________
# Set the current directory
os.chdir('../')
#print(os.getcwd()) #use to print the current directory
#_______________________________________________________________________________________________________________________
# Parameters for printing outputs
line_str = "-------------"

#_______________________________________________________________________________________________________________________
# TASK 0: Load the input data
# df_total (dataframe)
# Rows: Every row is an data instance (10000)
# Columns:
#   (1) One column "class" defines the classification (1-10)
#   (2) The remaining 784 columns define the pixels of the image (784 -> 28x28)
print("Loading the data\n" + line_str)
trainnumbers = loadmat('Data/Trainnumbers.mat')
input = trainnumbers['Trainnumbers'][0][0][0]
df_input = pd.DataFrame(input.T)
output = trainnumbers['Trainnumbers'][0][0][1]
df_output = pd.DataFrame(output.T)
df_total = df_input.copy()
df_total['class'] = df_output.values # total dataframe containing all data
print("Input Dataframe"
      "\nRows: \n  Every row is a data instance (10000)\nColumns:\n  (1) One column 'class' defines the "
      "classification (1-10)\n  (2) The remaining 784 columns define the pixels of the image (784 -> 28x28)\n")
print(df_total)
print(line_str)

#_______________________________________________________________________________________________________________________
# CHOOSE DATA USAGE TYPE
split_data = True

# CASE 1: split into training and testing data
if split_data:
    print("Split into training and testing data\n" + line_str)
    # Split into training and testing data
    percentage_of_training_data = 0.8
    df_train_set = df_total.sample(frac=percentage_of_training_data, random_state=0)
    df_test_set = df_total.drop(df_train_set.index)
    # final training and test data
    df_train_input = pd.DataFrame(df_train_set.loc[:, df_train_set.columns != 'class'])
    df_train_output = pd.DataFrame(df_train_set['class'])
    df_test_input = pd.DataFrame(df_test_set.loc[:, df_test_set.columns != 'class'])
    df_test_output = pd.DataFrame(df_test_set['class'])
    train_class = df_train_output['class'].values
# CASE 2: use external testing data
else:
    print("Load external testing data\n" + line_str)
    # Define Testing data
    df_train_set = df_total
    df_train_input = pd.DataFrame(df_train_set.loc[:, df_train_set.columns != 'class'])
    df_train_output = pd.DataFrame(df_train_set['class'])
    # Load external testing data
    # todo load external traing data
    testnumbers = sio.loadmat('Data/Test_numbers.mat')
    input = testnumbers['Test_numbers'][0][0][0]
    df_test_input = pd.DataFrame(input.T)


#_______________________________________________________________________________________________________________________
print("Call the different classfier\n" + line_str)

#_______________________________________________________________________________________________________________________
# Knn
# ....

technique = 'knn'
start_time = time.time()

#Params
param_1 = df_train_input
param_2 = df_train_output

#Function
knn_predictions, no_PCA = predict_knn(df_test_input, param_1, param_2)

runtime_knn = time.time() - start_time

#
if split_data:
    # ....
    # Plot Confusion Matrix

    cm = confusion_matrix(test_class, kme_predictions)
    plot_confusion_matrix(cm, labels)

    # ....
    # Calculate percentage of correct predictions
    alg_results = (test_class == kme_predictions)
    percentage = alg_results.sum() / len(alg_results)
    percentage_of_correct_predictions.append(percentage)
    running_time.append(runtime_knn)

# ....
# Code to convert the results to a matlab file specified in the homework description

print("Export results for algorith: "+ technique +"\n" + line_str)

for name in names:
    sio.savemat('Data/%s_%s.mat' %(name,technique), { 'names': names,'PCA': no_PCA, 'class': knn_predictions})

#_______________________________________________________________________________________________________________________
# Bayesian Classifiers
# ....

technique = 'bay'
start_time = time.time()

#Params
param_1 = df_train_input
param_2 = df_train_output

#Function
bay_predictions, no_PCA = predict_bay(df_test_input, param_1, param_2)

runtime_bay = time.time() - start_time

#
if split_data:
    # ....
    # Plot Confusion Matrix

    cm = confusion_matrix(test_class, bay_predictions)
    plot_confusion_matrix(cm, labels)

    # ....
    # Calculate percentage of correct predictions
    alg_results = (test_class == bay_predictions)
    percentage = alg_results.sum() / len(alg_results)
    percentage_of_correct_predictions.append(percentage)
    running_time.append(runtime_bay)

# ....
# Code to convert the results to a matlab file specified in the homework description

print("Export results for algorith: "+ technique +"\n" + line_str)

for name in names:
    sio.savemat('Data/%s_%s.mat' %(name,technique), { 'names': names,'PCA': no_PCA, 'class': bay_predictions})

#_______________________________________________________________________________________________________________________
# MLP
# ....

technique = 'mlp'
start_time = time.time()

#Params
param_1 = df_train_input
param_2 = df_train_output

#Function
mlp_predictions, no_PCA = predict_mlp(df_test_input, param_1, param_2)

runtime_mlp = time.time() - start_time

#
if split_data:
    # ....
    # Plot Confusion Matrix

    cm = confusion_matrix(test_class, mlp_predictions)
    plot_confusion_matrix(cm, labels)

    # ....
    # Calculate percentage of correct predictions
    alg_results = (test_class == mlp_predictions)
    percentage = alg_results.sum() / len(alg_results)
    percentage_of_correct_predictions.append(percentage)
    running_time.append(runtime_mlp)

# ....
# Code to convert the results to a matlab file specified in the homework description

print("Export results for algorith: "+ technique +"\n" + line_str)

for name in names:
    sio.savemat('Data/%s_%s.mat' %(name,technique), { 'names': names,'PCA': no_PCA, 'class': mlp_predictions})


#_______________________________________________________________________________________________________________________
# SOM
# ....

technique = 'som'
start_time = time.time()

#Params
param_1 = df_train_input
param_2 = df_train_output

#Function
som_predictions, no_PCA = predict_som(df_test_input, param_1, param_2)

runtime_som = time.time() - start_time

#
if split_data:
    # ....
    # Plot Confusion Matrix

    cm = confusion_matrix(test_class, som_predictions)
    plot_confusion_matrix(cm, labels)

    # ....
    # Calculate percentage of correct predictions
    alg_results = (test_class == som_predictions)
    percentage = alg_results.sum() / len(alg_results)
    percentage_of_correct_predictions.append(percentage)
    running_time.append(runtime_knn)
# ....
# Code to convert the results to a matlab file specified in the homework description

print("Export results for algorith: "+ technique +"\n" + line_str)

for name in names:
    sio.savemat('Data/%s_%s.mat' %(name,technique), { 'names': names,'PCA': no_PCA, 'class': som_predictions})


#_______________________________________________________________________________________________________________________
# K-MEANS
# ....

technique = 'kme'
start_time = time.time()

#Params
param_1 = df_train_input
param_2 = df_train_output

#Function
kme_predictions, no_PCA = predict_kme(df_test_input, param_1, param_2)

runtime_knn = time.time() - start_time

#
if split_data:
    # ....
    # Plot Confusion Matrix

    cm = confusion_matrix(test_class, kme_predictions)
    plot_confusion_matrix(cm, labels)

    # ....
    # Calculate percentage of correct predictions
    alg_results = (test_class == kme_predictions)
    percentage = alg_results.sum() / len(alg_results)
    percentage_of_correct_predictions.append(percentage)
    running_time.append(runtime_knn)

# ....
# Code to convert the results to a matlab file specified in the homework description

print("Export results for algorith: "+ technique +"\n" + line_str)

for name in names:
    sio.savemat('Data/%s_%s.mat' %(name,technique), { 'names': names,'PCA': no_PCA, 'class': kme_predictions})


#_______________________________________________________________________________________________________________________
# Graphics
# ....
if split_data:
    # Graphic Bar Percentage
    # ....

    y_pos = np.arange(len(techniques))
    plt.bar(y_pos, percentage_of_correct_predictions, color=(0.2, 0.4, 0.6, 0.6))
    plt.xticks(y_pos, techniques)
    plt.show()

# Graphic Bar Time Elapse
# ....

y_pos = np.arange(len(techniques))
plt.bar(y_pos, running_time, color=(0.2, 0.4, 0.6, 0.6))
plt.xticks(y_pos, techniques)
plt.show()

#_______________________________________________________________________________________________________________________
# Result table
# ....
print("Create result table\n" + line_str)

#todo result table

#_______________________________________________________________________________________________________________________
print("End\n" + line_str)

exit()