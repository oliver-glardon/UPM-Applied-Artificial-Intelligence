# Group Information: Inteligencia Artificial Applicada, UPM
# .....
# ....
# Oliver Glardon, 19936
# Emiliano Capogrossi, 18029
# Sorelys
# _______________________________________________________________________________________________________________________
# imports
from scipy.io import loadmat
import os
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn import model_selection, metrics, datasets
import pandas as pd
import time

# _______________________________________________________________________________________________________________________
# Set the current directory
os.chdir('../')
# print(os.getcwd()) #use to print the current directory
# _______________________________________________________________________________________________________________________
# Parameters for printing outputs
line_str = "-------------"
# _______________________________________________________________________________________________________________________
# TASK 0: Load the input data
# df_total (dataframe)
# Rows: Every row is an data instance (10000)
# Columns:
#   (1) One column "class" defines the classification (1-10)
#   (2) The remaining 784 columns define the pidata_imgsels of the image (784 -> 28data_imgs28)
print("Loading the data\n" + line_str)
trainnumbers = loadmat('Data/Trainnumbers.mat')
input = trainnumbers['Trainnumbers'][0][0][0]
df_input = pd.DataFrame(input.T)
output = trainnumbers['Trainnumbers'][0][0][1]
df_output = pd.DataFrame(output.T)
df_total = df_input.copy()
df_total['class'] = df_output.values  # total dataframe containing all data
print("Input Dataframe"
      "\nRows: \n  Every row is a data instance (10000)\nColumns:\n  (1) One column 'class' defines the "
      "classification (0-9)\n  (2) The remaining 784 columns define the pidata_imgsels of the image (784 -> 28data_imgs28)\n")
print(df_total)
print(line_str)

# Loop to collect data for different testing and training data
num_repetitions_per_parameter_setting = 1
df_results = pd.DataFrame(columns=['Run', 'K', 'Prediction Accuracy', 'Runtime [sec]'])
pos_count = 0

k_clusters = [20, 30]
for j in range(num_repetitions_per_parameter_setting):
    print("Run: " + str(j + 1) +
          "\nK: " + str(k_clusters[j % 2]) +
          "\n" + line_str)

    start_time = time.time()
    df_total
    # _____________________________________________________________________________________________________________________
    # Standardize/Normalize data
    # data are now named with an additional "_" at the end
    # print("Standardizing the data\n" + line_str)
    if j < 2:
        stsc = Normalizer().fit(df_total)
    else:
        stsc = StandardScaler().fit(df_total)

    df_input_ = pd.DataFrame(stsc.transform(df_total))

    # PCA
    minimum_edata_imgsplained_variance = 0.95
    pca = PCA(minimum_edata_imgsplained_variance)

    # alternative:
    # Choose the number of components by our own
    # number_principal_components = 10
    # pca = PCA(n_components=number_principal_components)

    pca.fit(df_input_)
    pca_components = pca.transform(df_input_)
    number_principal_components = pca.n_components_
    MSE_PCA_train = 1 - pca.explained_variance_ratio_.sum()

    # _______________________________________________________________________________________________________________________
    # TASK 6: Fisher Discriminant
    fisher = LinearDiscriminantAnalysis()
    fisher.fit(df_input_, df_total['class'].values)
    fisher_components_train = fisher.transform(df_input_)

    # Adding Fisher Discrimant Analysis to PCA
    fisher_pca = LinearDiscriminantAnalysis()
    fisher_pca.fit(pca_components, df_total['class'].values)
    fisher_pca_components = fisher_pca.transform(pca_components)

    print("\n" + line_str)
    # _______________________________________________________________________________________________________________________
    # TASK 7: K-MEANS
    # ....
    print("K-Means")

    # _______________________________________________________________________________________________________________________
    # Loop for different data:

    for i in range(3):
        # _______________________________________________________________________________________________________________________
        # Data to be used:

        # Without preprocessing
        if i == 0:
            data_imgs = df_input_
            print("Data being used: Without preprocessing")

        # Using PCA
        elif i == 1:
            data_imgs = pca_components
            print("Data being used: Using PCA")

        # Using Fisher+PCA
        elif i == 2:
            data_imgs = fisher_pca_components
            print("Data being used: Fisher+PCA")

        # Using Encoder
        elif i == 3:
            data_imgs = fisher_pca_components
            print("Data being used: Encoder")


        Nc = range(1, k_clusters[j] )
        kmeans = [KMeans(n_clusters=i) for i in Nc]
        score = [kmeans[i].fit(data_imgs).score(data_imgs) for i in range(len(kmeans))]
        plt.plot(Nc, score)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.title('Elbow Curve')
        plt.show()

