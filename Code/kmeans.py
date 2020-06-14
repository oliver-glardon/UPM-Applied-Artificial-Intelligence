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

import pandas as pd
import time
import joblib
from utils.functions import display_n_numbers

train = False


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
            counts = np.bincount(labels[0][0])
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


def predict_kme(test_imgs, df_input, img_class):

    stsc = Normalizer().fit(df_input)
    df_input_ = pd.DataFrame(stsc.transform(df_input))
    stsc = StandardScaler().fit(df_input_)
    df_input_ = pd.DataFrame(stsc.transform(df_input_))

    # PCA
    minimum_explained_variance = 0.95
    pca = PCA(minimum_explained_variance)

    pca.fit(df_input_)
    principal_components_train = pca.transform(df_input_)
    no_PCA = pca.n_components_

    # Adding Fisher Discrimant Analysis to PCA
    fisher_pca = LinearDiscriminantAnalysis()
    train_imgs = fisher_pca.fit(principal_components_train, img_class)


    if train:
        # FIT K-MEANS

        kmeans = KMeans(n_clusters=200, max_iter=200)
        kmeans.fit(train_imgs)
        cluster_labels = infer_cluster_labels(kmeans, img_class)

    else:
        # load the model from disk
        filename = 'Data/kme_models/_kmeModel-Cluster200-PCA+FISHER.sav'
        kmeans = joblib.load(filename)
        filename = 'Data/kme_models/%s_kmeModel-ClusterLabels%d-%s.sav'
        cluster_labels = joblib.load(filename)


    # _______________________________________________________________________________________________________________________
    # PREDICT
    predicted_Cluster = kmeans.predict(test_imgs)
    predictions = infer_data_labels(predicted_Cluster, cluster_labels)

    return predictions, no_PCA