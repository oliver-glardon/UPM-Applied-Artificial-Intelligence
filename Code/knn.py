#_______________________________________________________________________________________________________________________
# imports
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#_______________________________________________________________________________________________________________________
# Prediction method for knn

def predict_knn(df_test_input, param_1, param_2):

    df_train_input = param_1
    df_train_output = param_2

    # PCA
    # Choose the number of components by our own
    number_principal_components = 50
    pca = PCA(n_components=number_principal_components)
    df_train_input_ = df_train_input
    pca.fit(df_train_input_)
    principal_components_train = pca.transform(df_train_input_)
    # calculate the PCs for the test data as well
    principal_components_test = pca.transform(df_test_input)

    # Knn
    num_neighbors = 5
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn.fit(principal_components_train, df_train_output['class'].values)
    knn_labels_pred = pd.DataFrame(knn.predict(principal_components_test))
    return knn_labels_pred, number_principal_components
