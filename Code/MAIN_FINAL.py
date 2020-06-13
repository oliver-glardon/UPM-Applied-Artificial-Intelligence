# Group Information: Inteligencia Artificial Applicada, UPM
# Sorelyss _____
# Emiliano ____
# Oliver Glardon, 19936
#_______________________________________________________________________________________________________________________
# imports
from scipy.io import loadmat
import os
import pandas as pd
import time

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
# CASE 2: use external testing data
else:
    print("Load external testing data\n" + line_str)
    # Define Testing data
    df_train_set = df_total
    df_train_input = pd.DataFrame(df_train_set.loc[:, df_train_set.columns != 'class'])
    df_train_output = pd.DataFrame(df_train_set['class'])
    # Load external testing data
    # todo load external traing data
    testnumbers = loadmat('Data/Trainnumbers.mat')
    input = testnumbers['Trainnumbers'][0][0][0]
    df_test_input = pd.DataFrame(input.T)

#_______________________________________________________________________________________________________________________
print("Call the different classfier\n" + line_str)

# Call knn
start_time = time.time()
param_1 = df_train_input
param_2 = df_train_output
res_knn = predict_knn(df_test_input, param_1, param_2)
runtime_knn = time.time() - start_time

# Call bayes
start_time = time.time()
params = df_test_input
res_bayes = predict_bayes(df_test_input, params)
runtime_bayes = time.time() - start_time

'''
# Call mlp
start_time = time.time()
params = None
res_mlp = predict_mlp(df_test_input, params)
runtime_mlp = time.time() - start_time

# Call som
start_time = time.time()
params = None
res_som = predict_som(df_test_input, params)
runtime_som = time.time() - start_time

# Call kmeans
start_time = time.time()
params = None
res_kmeans = predict_kmeans(df_test_input, params)
runtime_kmeans = time.time() - start_time
'''

#_______________________________________________________________________________________________________________________
print("Create result table\n" + line_str)

#todo result table

#_______________________________________________________________________________________________________________________
print("Export results\n" + line_str)

#todo create matlab file according to the task description

#_______________________________________________________________________________________________________________________
print("End\n" + line_str)