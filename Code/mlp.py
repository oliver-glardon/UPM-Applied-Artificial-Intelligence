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
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from utils.plot_cm import plot_confusion_matrix
import matplotlib.pyplot as plt
from neupy import algorithms, layers
import pickle
import pandas as pd
import time
import joblib

def predict_mlp(test_input, test_output, df_input, img_class):

    test_input /= 255.
    df_input /= 255.
    if test_output != [] or img_class != []:
        # One-Hot
        target_scaler = OneHotEncoder(sparse=False, categories='auto')
        img_class = target_scaler.fit_transform(img_class.reshape(-1, 1))
        test_class = target_scaler.fit_transform(test_output.reshape(-1, 1))

    stsc = StandardScaler().fit(df_input)
    stsc1 = StandardScaler().fit(test_input)
    df_input_ = pd.DataFrame(stsc.transform(df_input))
    test_input_ = pd.DataFrame(stsc1.transform(test_input))

    stsc = Binarizer().fit(df_input_)
    stsc1 = Binarizer().fit(test_input_)
    df_input_ = pd.DataFrame(stsc.transform(df_input_))
    test_input_ = pd.DataFrame(stsc1.transform(test_input_))

    # PCA
    number_principal_components = 250
    pca = PCA(n_components=number_principal_components)

    pca.fit(df_input_)
    principal_components_train = pca.transform(df_input_)
    principal_components_test = pca.transform(test_input_)
    no_PCA = pca.n_components_

    train_imgs = principal_components_train
    test_imgs = principal_components_test


    # _______________________________________________________________________________________________________________________
    # load the model from disk

    filename = 'Data/mlp_models/_mlp2_model-PCA250-NSB.pickle'
    with open(filename, 'rb') as f:
        mlp2 = pickle.load(f)

    # _______________________________________________________________________________________________________________________
    # PREDICT

    predictions = mlp2.predict(test_imgs).argmax(axis=1)

    return predictions, no_PCA