# Group Information: Inteligencia Artificial Applicada, UPM
# .....
# ....
# Oliver Glardon, 19936
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
      "classification (0-9)\n  (2) The remaining 784 columns define the pixels of the image (784 -> 28x28)\n")
print(df_total)
print(line_str)

# Loop to collect data for different testing and training data
num_repetitions_per_parameter_setting = 2
df_results = pd.DataFrame(columns=['Run', 'Neighbors', 'Prediction Accuracy', 'Runtime [sec]'])
pos_count = 0
j = 0
while j < num_repetitions_per_parameter_setting:
    print("Run: " + str(j+1) + "\n" + line_str)
    j = j+1
    start_time = time.time()

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

    #_______________________________________________________________________________________________________________________
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

    #_______________________________________________________________________________________________________________________
    # Text from homework description:
    # Viability of the following techniques for classification using previous dimensionality reduction with PCA. Obtain the
    # confusion matrix. Compare the results with the other techniques

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

# create result table
df_results = pd.DataFrame(columns=['Number of neighbors', 'Average', 'Max', 'Min', 'Average runtime [sec]'])
pos_count = 0
#num_neigh_array = [1, 3, 5, 7, 9]
num_neigh_array = [1, 3, 5, 7, 9]
for i in num_neigh_array:
    df_results.loc[pos_count] = [str(i),
                                 df[df['Neighbors']==str(i)]['Prediction Accuracy'].mean(),
                                 df[df['Neighbors']==str(i)]['Prediction Accuracy'].max(),
                                 df[df['Neighbors'] == str(i)]['Prediction Accuracy'].min(),
                                 df[df['Neighbors'] == str(i)]['Runtime [sec]'].mean()
                                 ]
    pos_count = pos_count + 1

print(df_results)
Path = r'C:\Users\Oliver\Documents\Dokumente\Studium\M.Sc. UPM\Intelligencia Artificial Applicada\Trabajo' \
             '\knn_results.csv'
df_results.to_csv(Path, index=False, header=True)