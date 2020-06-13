#_______________________________________________________________________________________________________________________
# imports
from scipy.io import loadmat
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import time

#_______________________________________________________________________________________________________________________
# Set the current directory
os.chdir('../')
#print(os.getcwd()) #use to print the current directory
#_______________________________________________________________________________________________________________________
# Parameters for printing outputs
line_str = "-------------"

def predict_knn(df_test_input, param_1, param_2):
    df_
    #_______________________________________________________________________________________________________________________
    #print("Split into training and testing data\n" + line_str)
    # Split into training and testing data
    percentage_of_training_data = 0.8
    df_train_set = df_total.sample(frac=percentage_of_training_data)
    df_test_set = df_total.drop(df_train_set.index)
    # final training and test data
    df_train_input = pd.DataFrame(df_train_set.loc[:, df_train_set.columns != 'class'])
    df_train_output = pd.DataFrame(df_train_set['class'])
    df_test_input = pd.DataFrame(df_test_set.loc[:, df_test_set.columns != 'class'])
    df_test_output = pd.DataFrame(df_test_set['class'])

    df_test_input
    df_train_input

    #_____________________________________________________________________________________________________________________
    # TASK 1: PCA (dimensionality reduction)
    # Viability of PCA for reducing the dimensionality and further reconstruction. Quantify the reconstruction
    # MSE versus the number of dimensionality of the reduced data and display the reconstructed images.

    # Standardize/Normalize data
    # data are now named with an additional "_" at the end
    #print("Standardizing the data\n" + line_str)
    stsc = StandardScaler()
    stsc.fit(df_train_input)
    df_train_input_ = pd.DataFrame(stsc.transform(df_train_input))
    df_test_input_ = pd.DataFrame(stsc.transform(df_test_input))

    df_train_input_ = df_train_input
    df_test_input_ = df_test_input

    # PCA
    minimum_explained_variance = 0.95
    #pca = PCA(minimum_explained_variance)

    # alternative:
    # Choose the number of components by our own
    number_principal_components = 100
    pca = PCA(n_components=number_principal_components)

    pca.fit(df_train_input_)
    principal_components_train = pca.transform(df_train_input_)
    number_principal_components = pca.n_components_
    MSE_PCA_train = 1 - pca.explained_variance_ratio_.sum()

    pca_output = "Principal Component Analysis (PCA) \n\n" \
                 "Explained Variance: " + str(round(pca.explained_variance_ratio_.sum(), 3))+ "" \
                    " \nNumber of Principal Components: " + str(number_principal_components) + "\n" + "" \
                    "MSE (training data): " + str(round(MSE_PCA_train, 3)) + "\n" + line_str
    print(pca_output)

    # calculate the PCs for the test data as well
    principal_components_test = pca.transform(df_test_input_)

    # _______________________________________________________________________________________________________________________
    # Loop for different parameter settings:
    #num_neigh_array = [1, 3, 5, 7, 9]
    num_neigh_array = [1, 3, 5, 7, 9]
    for i in num_neigh_array:
        print("\n" + line_str)

        #_______________________________________________________________________________________________________________________
        # TASK 2: k-NN
        # ....
        print("k-nearest neighbors")
        num_neighbors = i
        knn = KNeighborsClassifier(n_neighbors=num_neighbors)
        knn.fit(principal_components_train, df_train_output['class'].values)
        knn_labels_pred = knn.predict(principal_components_test)
        knn_results = (df_test_output['class'].values == knn_labels_pred)
        knn_percentage_of_correct_predictions = knn_results.sum()/len(knn_results)
        print("Dimensionality reduction: PCA" +
              "\nNumber of nearest neighbors: "+ str(num_neighbors) +
              "\nPercentage of correct prediction: "+ str(knn_percentage_of_correct_predictions) +
              "\n" + line_str)

        '''
        print("knn with PCA and Fisher")
        knn.fit(fisher_pca_components_train, df_train_output['class'].values)
        knn_labels_pred = knn.predict(fisher_pca_components_test)
        knn_results = (df_test_output['class'].values == knn_labels_pred)
        knn_percentage_of_correct_predictions_ = knn_results.sum()/len(knn_results)
        print("Dimensionality reduction: PCA and Fisher (LDA)" +
              "\nNumber of nearest neighbors: "+ str(num_neighbors) +
              "\nPercentage of correct prediction: "+ str(knn_percentage_of_correct_predictions_) +
              "\n" + line_str)
        '''

        # store values in Dataframe
        end_time = time.time()
        duration = end_time - start_time
        df_results.loc[pos_count] = [str(j),str(i),knn_percentage_of_correct_predictions, duration]
        pos_count = pos_count + 1

# print results and store to a pickle-file as a basis for later visualization
df_results.to_pickle('Data/knn_results.pickle')
df = pd.read_pickle('Data/knn_results.pickle')
print(df)