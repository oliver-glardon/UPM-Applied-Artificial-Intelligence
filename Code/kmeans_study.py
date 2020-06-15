# Group Information: Inteligencia Artificial Applicada, UPM
# .....
# Emiliano Capogrossi, M18029
# Oliver Glardon, 19936
# Sorelys Sandoval, M19237
# .....
# _______________________________________________________________________________________________________________________
# imports
from scipy.io import loadmat
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, Binarizer, Normalizer, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from utils.plot_cm import plot_confusion_matrix
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import time
import joblib
from utils.functions import display_n_numbers


train = False
type_run = "NOTRAIN"
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
pca_dims = [25, 50, 100, 200]
k_array = [25, 50, 100, 200]

def infer_cluster_labels(kmeans, actual_labels):
    """
    Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label
    """

    inferred_labels = {}

    for i in range(kmeans.n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

        # print(labels)
        # print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))

    return inferred_labels


def infer_data_labels(X_labels, cluster_labels):
    """
    Determines label for each array, depending on the cluster it has been assigned to.
    returns: predicted labels for each array
    """

    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key

    return predicted_labels

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
#   (2) The remaining 784 columns define the pixels of the image (784 -> 28x28)
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
      "classification (0-9)\n  (2) The remaining 784 columns define the pixels of the image (784 -> 28x28)\n")
print(df_total)
print(line_str)

df_results = pd.DataFrame(
    columns=['n_clusters', 'Preprocessing', 'Data used', 'Prediction Accuracy', 'Runtime [sec]'])
pos_count = 0
# _______________________________________________________________________________________________________________________
# TASK 7: K-MEANS
# ....
print("K-Means")

# _______________________________________________________________________________________________________________________
# Plot n numbers
display_n_numbers(df_input.values, 25, [], 'Data/kme_models/%s_Muestras.png' % (type_run))


# _____________________________________________________________________________________________________________________
# Split into training and testing data

# print("Split into training and testing data\n" + line_str)
percentage_of_training_data = 0.8
df_train_set = df_total.sample(frac=percentage_of_training_data, random_state=0)
df_test_set = df_total.drop(df_train_set.index)
# final training and test data
df_train_input = pd.DataFrame(df_train_set.loc[:, df_train_set.columns != 'class'])
df_train_output = pd.DataFrame(df_train_set['class'])
df_test_input = pd.DataFrame(df_test_set.loc[:, df_test_set.columns != 'class'])
df_test_output = pd.DataFrame(df_test_set['class'])
img_class = df_train_output['class'].values
test_class = df_test_output['class'].values
# _____________________________________________________________________________________________________________________
# Loop
# ...

# Loop to collect data for different testing and training data

loop_array = k_array
if type_run == "PCA":
    loop_array = [1]

total_time = time.time()

for j in range(len(loop_array)):
    n = loop_array[j]
    print("n_clusters: " + str(n) + "\n" + line_str)
    start_time = time.time()

    # _____________________________________________________________________________________________________________________
    # Preprocessing
    # ...

    # Rescale - adding bias to prevent weights from not being updated
    if j == 5:
        fac = 0.99 / 255
        df_input = df_input * fac + 0.01

        img_class[img_class == 0] = 0.01
        img_class[img_class == 1] = 0.99

    else:
        df_train_input /= 255.
        df_test_input /= 255.

    if j == 0:
        preprocessing = 'Without preprocessing'
        df_train_input_ = df_train_input
        df_test_input_ = df_test_input

    if j == 1:
        preprocessing = 'Normalize'
        stsc = Normalizer().fit(df_train_input)
        df_train_input_ = pd.DataFrame(stsc.transform(df_train_input))
        stsc1 = Normalizer().fit(df_test_input)
        df_test_input_ = pd.DataFrame(stsc1.transform(df_test_input))
    elif j == 2:
        preprocessing = 'Standarized'
        stsc = StandardScaler().fit(df_train_input)
        stsc1 = StandardScaler().fit(df_test_input)
        df_train_input_ = pd.DataFrame(stsc.transform(df_train_input))
        df_test_input_ = pd.DataFrame(stsc1.transform(df_test_input))
    elif j == 3:
        preprocessing = 'Normalize+Standarized'
        stsc = Normalizer().fit(df_train_input)
        stsc1 = Normalizer().fit(df_test_input)
        df_train_input_ = pd.DataFrame(stsc.transform(df_train_input))
        df_test_input_ = pd.DataFrame(stsc1.transform(df_test_input))
        stsc = StandardScaler().fit(df_train_input_)
        stsc1 = StandardScaler().fit(df_test_input_)
        df_train_input_ = pd.DataFrame(stsc.transform(df_train_input_))
        df_test_input_ = pd.DataFrame(stsc1.transform(df_test_input_))
    elif j == 4:
        preprocessing = 'simple_github'
        df_train_input_ -= df_train_input.mean(axis=0)
        df_test_input_ -= df_train_input.mean(axis=0)

    if "BIN" in type_run:
        preprocessing = preprocessing + '+Binarized'
        stsc = Binarizer().fit(df_train_input_)
        stsc1 = Binarizer().fit(df_test_input_)
        df_train_input_ = pd.DataFrame(stsc.transform(df_train_input_))
        df_test_input_ = pd.DataFrame(stsc1.transform(df_test_input_))

    # ...
    # Preprocessing Techniques
    # _____________________________________________________________________________________________________________________

    # PCA
    number_principal_components = 100
    pca = PCA(n_components=number_principal_components)

    pca.fit(df_train_input_)
    principal_components_train = pca.transform(df_train_input_)
    principal_components_test = pca.transform(df_test_input_)
    number_principal_components = pca.n_components_
    MSE_PCA_train = 1 - pca.explained_variance_ratio_.sum()

    # Adding Fisher Discrimant Analysis to PCA
    fisher_pca = LinearDiscriminantAnalysis()
    fisher_pca.fit(principal_components_train, img_class)
    fisher_pca_components_train = fisher_pca.transform(principal_components_train)
    fisher_pca_components_test = fisher_pca.transform(principal_components_test)


    # _______________________________________________________________________________________________________________________
    # Loop for different parameter settings:

    for i in range(3):

        train_imgs = []
        kmeans = []
        cluster_labels = []

        if type_run == "PCA":
            train_imgs = principal_components_train
            data_type = 'PCA' + str()
            print("Data Using PCA")

        else:

            # _______________________________________________________________________________________________________________________
            # Data to be used:

            # Without preprocessing
            if i == 0:
                train_imgs = df_train_input_.values
                test_imgs = df_test_input_.values
                data_type = 'No technique'
                print("Data Without Techniques")

            # Using PCA
            elif i == 1:
                train_imgs = principal_components_train
                test_imgs = principal_components_test
                data_type = 'PCA'
                print("Data Using PCA")

            # Using Fisher+PCA
            elif i == 2:
                train_imgs = fisher_pca_components_train
                test_imgs = fisher_pca_components_test
                data_type = 'PCA+FISHER'
                print("Data Using Fisher+PCA")


        # _______________________________________________________________________________________________________________________
        # FIT K-MEANS
        if train:
            filename = 'Data/kme_models/%s_kmeModel-Cluster%d-%s.pickle' % (type_run, n, data_type)
        else:
            filename = 'Data/kme_models/_kmeModel-Cluster%d-%s.pickle' % ( n, data_type)


        if train:
            # FIT K-MEANS

            kmeans = KMeans(n_clusters=n,max_iter=200)
            kmeans.fit(train_imgs)

            # Save pre-trained SOFM network
            with open(filename, 'wb') as f:
                pickle.dump(kmeans, f)

            print("\nCreated File:\n" + filename + '\n' + line_str)

        else:
            # load the model from disk
            with open(filename, 'rb') as f:
                kmeans = pickle.load(f)

            print("\nLoaded File:\n\n" + filename + '\n' + line_str)

        # _______________________________________________________________________________________________________________________

        centroids = kmeans.cluster_centers_
        '''
        # Plot Centroids
        display_n_numbers(kmeans.cluster_centers_,
                          len(kmeans.cluster_centers_),
                          [],
                          'Data/kme_models/%s_kmeModel-Cluster%d-%s-Centroids.png' % (type_run, n, data_type))
        '''
        # determine cluster class
        if train:
            cluster_labels = infer_cluster_labels(kmeans, img_class)
            filename = 'Data/kme_models/%s_kmeModel-ClusterLabels%d-%s.sav' % (type_run, n, data_type)
            joblib.dump(cluster_labels, filename)
        else:
            filename = 'Data/kme_models/_kmeModel-ClusterLabels%d-%s.sav' % (n, data_type)
            cluster_labels = joblib.load(filename)
        '''
        # Plot Centroids with label
        display_n_numbers(kmeans.cluster_centers_,
                          len(kmeans.cluster_centers_),
                          cluster_labels,
                          'Data/kme_models/%s_kmeModel-Cluster%d-%s-CentroidsLabeled.png' % (type_run, n, data_type))
        '''
        # _______________________________________________________________________________________________________________________
        # PREDICT
        predicted_Cluster = kmeans.predict(test_imgs)
        kmeans_labels_pred = infer_data_labels(predicted_Cluster, cluster_labels)

        percentage_of_correct_predictions = metrics.accuracy_score(test_class, kmeans_labels_pred)
        print("\nPercentage of correct prediction: " + str(percentage_of_correct_predictions))

        # Plot Confusion Matrix
        cm = confusion_matrix(test_class, kmeans_labels_pred)
        plot_confusion_matrix(cm,
                              labels,
                              'Confusion matrix',
                              None, True,
                              'Data/kme_models/%s_kmeModel-Cluster%d-%s-CM.png' % (type_run, n, data_type))

        end_time = time.time()
        duration = end_time - start_time
        print("\nElapsed time: " + str(duration) + ' [Sec]\n' + line_str)

        # _______________________________________________________________________________________________________________________
        # store values in Dataframe
        df_results.loc[pos_count] = [str(n), preprocessing, data_type, percentage_of_correct_predictions, duration]
        pos_count = pos_count + 1

# print results and store to a pickle-file as a basis for later visualization
print(df_results)
df_results.to_pickle('Data/%s_kms_results.pickle' % (type_run))

total_duration = time.time() - total_time
print("\nTotal elapsed time: " + str(total_duration) + ' [Sec]\n' + line_str)




