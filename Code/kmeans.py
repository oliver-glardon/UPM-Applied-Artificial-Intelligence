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
import joblib

train = True

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
num_repetitions_per_parameter_setting = 3
df_results = pd.DataFrame(columns=['Run', 'K', 'Prediction Accuracy', 'Runtime [sec]'])
pos_count = 0

k_clusters = [20, 30]
for j in range(num_repetitions_per_parameter_setting):
    print("Run: " + str(j + 1) +
          "\n" + line_str)

    start_time = time.time()

    # _______________________________________________________________________________________________________________________
    # Split into training and testing data

    # print("Split into training and testing data\n" + line_str)
    percentage_of_training_data = 0.8
    df_train_set = df_total.sample(frac=percentage_of_training_data)
    df_test_set = df_total.drop(df_train_set.index)
    # final training and test data
    df_train_input = pd.DataFrame(df_train_set.loc[:, df_train_set.columns != 'class'])
    df_train_output = pd.DataFrame(df_train_set['class'])
    df_test_input = pd.DataFrame(df_test_set.loc[:, df_test_set.columns != 'class'])
    df_test_output = pd.DataFrame(df_test_set['class'])
    # _____________________________________________________________________________________________________________________
    # Standardize/Normalize data
    # data are now named with an additional "_" at the end
    # print("Standardizing the data\n" + line_str)
    if j < 2:
        stsc = Normalizer().fit(df_train_input)
        df_train_input_ = pd.DataFrame(stsc.transform(df_train_input))
        df_test_input_ = pd.DataFrame(stsc.transform(df_test_input))
    else:
        stsc = StandardScaler().fit(df_train_input)
        df_train_input_ = pd.DataFrame(stsc.transform(df_train_input))
        df_test_input_ = pd.DataFrame(stsc.transform(df_test_input))

    # PCA
    minimum_explained_variance = 0.95
    pca = PCA(minimum_explained_variance)

    # alternative:
    # Choose the number of components by our own
    # number_principal_components = 10
    # pca = PCA(n_components=number_principal_components)

    pca.fit(df_train_input_)
    principal_components_train = pca.transform(df_train_input_)
    number_principal_components = pca.n_components_
    MSE_PCA_train = 1 - pca.explained_variance_ratio_.sum()

    pca_output = "Principal Component Analysis (PCA) \n\n" \
                 "Explained Variance: " + str(round(pca.explained_variance_ratio_.sum(), 3)) + "" \
                                                                                               " \nNumber of Principal Components: " + str(
        number_principal_components) + "\n" + "" \
                                              "MSE (training data): " + str(round(MSE_PCA_train, 3)) + "\n" + line_str
    print(pca_output)

    # calculate the PCs for the test data as well
    principal_components_test = pca.transform(df_test_input_)

    # _______________________________________________________________________________________________________________________
    # TASK 6: Fisher Discriminant
    fisher = LinearDiscriminantAnalysis()
    fisher.fit(df_train_input_, df_train_output['class'].values)
    fisher_components_train = fisher.transform(df_train_input_)
    fisher_components_test = fisher.transform(df_test_input_)

    # Adding Fisher Discrimant Analysis to PCA
    fisher_pca = LinearDiscriminantAnalysis()
    fisher_pca.fit(principal_components_train, df_train_output['class'].values)
    fisher_pca_components_train = fisher_pca.transform(principal_components_train)
    fisher_pca_components_test = fisher_pca.transform(principal_components_test)

    print("\n" + line_str)
    # _______________________________________________________________________________________________________________________
    # TASK 7: K-MEANS
    # ....
    print("K-Means")

    # _______________________________________________________________________________________________________________________
    # Loop for different data:

    for i in range(2):
        # _______________________________________________________________________________________________________________________
        # Data to be used:

        # Without preprocessing
        if i == 0:
            train_imgs = df_train_input_
            test_imgs = df_test_input_
            print("Data being used: Without preprocessing")

        # Using PCA
        elif i == 1:
            train_imgs = principal_components_train
            test_imgs = principal_components_test
            print("Data being used: Using PCA")

        # Using Fisher+PCA
        elif i == 2:
            train_imgs = fisher_pca_components_train
            test_imgs = fisher_pca_components_test
            print("Data being used: Fisher+PCA")

        # Using Encoder
        elif i == 3:
            train_imgs = fisher_pca_components_train
            test_imgs = fisher_pca_components_train
            print("Data being used: Encoder")

        train_class = df_train_output['class'].values
        test_class = df_test_output['class'].values



        # _______________________________________________________________________________________________________________________
        # FIT K-MEANS

        filename = 'Data/mlp_models/kmeans_model-run_%d-data_%d.sav' %(j,i)

        if train:
            # FIT K-MEANS

            kmeans = KMeans(n_clusters=k_clusters[j % 2]).fit(train_imgs)

            # save the model to disk
            joblib.dump(kmeans, filename)

            print("\nCreated File:\n"+ filename + '\n' + line_str)

        else:
            # load the model from disk
            kmeans = joblib.load(filename)

            print("\nLoaded File:\n\n"+ filename + '\n' + line_str)

        # _______________________________________________________________________________________________________________________


        # _______________________________________________________________________________________________________________________
        # PREDICT
        centroids = kmeans.cluster_centers_
        kmeans_labels_pred = kmeans.predict(test_imgs)
        df_results = (test_class == kmeans_labels_pred)
        percentage_of_correct_predictions = df_results.sum() / len(df_results)
        print("\nK: " + str(k_clusters[j%2]) +
              "\nPercentage of correct prediction: " + str(percentage_of_correct_predictions) +
              "\n" + line_str)


        # _______________________________________________________________________________________________________________________
        # store values in Dataframe
        end_time = time.time()
        duration = end_time - start_time
        df_results.loc[pos_count] = [str(j), str(k_clusters[j%2]), percentage_of_correct_predictions, duration]
        pos_count = pos_count + 1


# print results and store to a pickle-file as a basis for later visualization
print(df_results)
df_results.to_pickle('Data/kmeans_results.pickle')
