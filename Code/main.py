# Group Information: Intelligencia Artificial Applicada, UPM
# .....
# ....
# Oliver Glardon, 19936
#_______________________________________________________________________________________________________________________
# imports
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

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

#_______________________________________________________________________________________________________________________
# Show the images
def display_number(data):
    plt.imshow(np.array(data).reshape(28, 28))
    plt.show()
# Select one of the 10000 imagenes in the training data to display and uncomment the two following lines
#image_number = 9999
#display_number(train[image_number])
#_______________________________________________________________________________________________________________________
# TASK 1: PCA (dimensionality reduction)
# Viability of PCA for reducing the dimensionality and further reconstruction. Quantify the reconstruction
# MSE versus the number of dimensionality of the reduced data and display the reconstructed images.

# Standardize/Normalize data
print("Standardizing the data\n" + line_str)
stsc = StandardScaler()
stsc.fit(df_train_input)
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
#_______________________________________________________________________________________________________________________
# TASK 2: k-NN
# ....
print("k-nearest neighbors")
num_neighbors = 3
knn = KNeighborsClassifier(n_neighbors=num_neighbors)
knn.fit(principal_components_train, df_train_output['class'].values)
knn_labels_pred = knn.predict(principal_components_test)
knn_results = (df_test_output['class'].values == knn_labels_pred)
knn_percentage_of_correct_predictions = knn_results.sum()/len(knn_results)
print("Dimensionality reduction: PCA" +
      "\nNumber of nearest neighbors: "+ str(num_neighbors) +
      "\nPercentage of correct prediction: "+ str(knn_percentage_of_correct_predictions) +
      "\n" + line_str)

print("knn with PCA and Fisher")
knn.fit(fisher_pca_components_train, df_train_output['class'].values)
knn_labels_pred = knn.predict(fisher_pca_components_test)
knn_results = (df_test_output['class'].values == knn_labels_pred)
knn_percentage_of_correct_predictions = knn_results.sum()/len(knn_results)
print("Dimensionality reduction: PCA and Fisher (LDA)" +
      "\nNumber of nearest neighbors: "+ str(num_neighbors) +
      "\nPercentage of correct prediction: "+ str(knn_percentage_of_correct_predictions) +
      "\n" + line_str)

#_______________________________________________________________________________________________________________________
# TASK 3: Bayesian Classifiers
# ....
print("Naive Bayes Classification")
bayes = GaussianNB()
bayes.fit(principal_components_train, df_train_output['class'].values)
bayes_labels_pred = bayes.predict(principal_components_test)
bayes_results = (df_test_output['class'].values == bayes_labels_pred)
bayes_percentage_of_correct_predictions = bayes_results.sum()/len(bayes_results)
print("\nMethod: Gaussian Naive Bayes" +
      "\nPercentage of correct prediction: "+ str(bayes_percentage_of_correct_predictions) +
      "\n" + line_str)

#_______________________________________________________________________________________________________________________
# TASK 4: MLP
# ....
print("Multilayer Perceptron (MLP)")
mlp = MLPClassifier()
mlp.fit(principal_components_train, df_train_output['class'].values)
mlp_labels_pred = mlp.predict(principal_components_test)
mlp_results = (df_test_output['class'].values == mlp_labels_pred)
mlp_percentage_of_correct_predictions = mlp_results.sum()/len(mlp_results)
print("\nMethod: STILL TO ADD" +
      "\nPercentage of correct prediction: "+ str(mlp_percentage_of_correct_predictions) +
      "\n" + line_str)

#_______________________________________________________________________________________________________________________
# TASK 5: SOM
# ....

#_______________________________________________________________________________________________________________________
# Optional tasks for obtaining top marks above 7: (WE SHOULD ALSO DO THAT)
# Task 6.- Use Linear Fisher Discriminant for reducing the dimensionality
# Task 7.- Use K-means for clustering the images
# Task 8.- Use Autoencoder for dimensionality reduction (first reduce with PCA to an intermediate reduction level).
# Compare the results with task 1.
# Task 9.- Build a visual map using SOM where the pattern image is at the position of every neuron in the 2D map.
# Task 10.- Please highlight any original improvement or combination of the above mentioned techniques that you have
# tried in order to improve the results
#_______________________________________________________________________________________________________________________
# Code to convert the results to a matlab file specified in the homework description
#.....

#_______________________________________________________________________________________________________________________

exit()