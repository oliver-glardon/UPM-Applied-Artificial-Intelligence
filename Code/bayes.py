#_______________________________________________________________________________________________________________________
# imports
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Binarizer
import pandas as pd

#_______________________________________________________________________________________________________________________
# Prediction method for bayes

def predict_bay(df_test_input, param_1, param_2):

    df_train_input = param_1
    df_train_output = param_2

    # Binarization
    transformer = Binarizer().fit(df_train_input)
    df_train_input_ = pd.DataFrame(transformer.transform(df_train_input))
    transformer = Binarizer().fit(df_test_input)
    df_test_input_ = pd.DataFrame(transformer.transform(df_test_input))

    # PCA
    # Choose the number of components by our own
    number_principal_components = 100
    pca = PCA(n_components=number_principal_components)
    pca.fit(df_train_input_)
    principal_components_train = pca.transform(df_train_input_)
    # calculate the PCs for the test data as well
    principal_components_test = pca.transform(df_test_input_)

    # making the data non-negative
    lowest_num = 0
    if (principal_components_test.min() < principal_components_train.min()):
        lowest_num = principal_components_test.min()
    else:
        lowest_num = principal_components_train.min()
    principal_components_train = abs(lowest_num)+principal_components_train
    principal_components_test = abs(lowest_num) + principal_components_test

    # Bayes
    bayes = GaussianNB()
    bayes.fit(principal_components_train, df_train_output['class'].values)
    bayes_labels_pred = pd.DataFrame(bayes.predict(principal_components_test))
    return bayes_labels_pred, number_principal_components
