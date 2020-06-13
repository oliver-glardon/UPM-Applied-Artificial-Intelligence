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
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
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
num_repetitions_per_parameter_setting = 10
df_results = pd.DataFrame(columns=['Run', 'Method', 'Prediction Accuracy', 'Runtime [sec]'])
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
    '''
    pca_output = "Principal Component Analysis (PCA) \n\n" \
                 "Explained Variance: " + str(round(pca.explained_variance_ratio_.sum(), 3))+ "" \
                    " \nNumber of Principal Components: " + str(number_principal_components) + "\n" + "" \
                    "MSE (training data): " + str(round(MSE_PCA_train, 3)) + "\n" + line_str
    print(pca_output)
    '''

    # calculate the PCs for the test data as well
    principal_components_test = pca.transform(df_test_input_)

    # making the data non-negative
    lowest_num = 0
    if (principal_components_test.min() < principal_components_train.min()):
        lowest_num = principal_components_test.min()
    else:
        lowest_num = principal_components_train.min()
    principal_components_train = (-lowest_num)+principal_components_train
    principal_components_test = (-lowest_num) + principal_components_test

    #_______________________________________________________________________________________________________________________
    # Text from homework description:
    # Viability of the following techniques for classification using previous dimensionality reduction with PCA. Obtain the
    # confusion matrix. Compare the results with the other techniques

    intermed_time = time.time() - start_time
    start_time = time.time()

    print("Naive Bayes Classification")
    bayes = GaussianNB()
    bayes.fit(principal_components_train, df_train_output['class'].values)
    bayes_labels_pred = bayes.predict(principal_components_test)
    bayes_results = (df_test_output['class'].values == bayes_labels_pred)
    bayes_percentage_of_correct_predictions_gaussian = bayes_results.sum() / len(bayes_results)
    print("\nMethod: Gaussian Naive Bayes" +
          "\nPercentage of correct prediction: " + str(bayes_percentage_of_correct_predictions_gaussian) +
          "\n" + line_str)

    # store values in Dataframe
    df_results.loc[pos_count] = [str(j),'Gaussian',bayes_percentage_of_correct_predictions_gaussian,
                                 intermed_time + time.time() - start_time]
    pos_count = pos_count + 1
    start_time = time.time()

    print("Naive Bayes Classification")
    bayes = MultinomialNB()
    bayes.fit(principal_components_train, df_train_output['class'].values)
    bayes_labels_pred = bayes.predict(principal_components_test)
    bayes_results = (df_test_output['class'].values == bayes_labels_pred)
    bayes_percentage_of_correct_predictions_multi = bayes_results.sum() / len(bayes_results)
    print("\nMethod: Multinomial Naive Bayes" +
          "\nPercentage of correct prediction: " + str(bayes_percentage_of_correct_predictions_multi) +
          "\n" + line_str)

    # store values in Dataframe
    df_results.loc[pos_count] = [str(j),'Multinominal',bayes_percentage_of_correct_predictions_multi,
                                 intermed_time + time.time() - start_time]
    pos_count = pos_count + 1
    start_time = time.time()

    print("Naive Bayes Classification")
    bayes = ComplementNB()
    bayes.fit(principal_components_train, df_train_output['class'].values)
    bayes_labels_pred = bayes.predict(principal_components_test)
    bayes_results = (df_test_output['class'].values == bayes_labels_pred)
    bayes_percentage_of_correct_predictions_complement = bayes_results.sum() / len(bayes_results)
    print("\nMethod: Complement Naive Bayes" +
          "\nPercentage of correct prediction: " + str(bayes_percentage_of_correct_predictions_complement) +
          "\n" + line_str)

    # store values in Dataframe
    df_results.loc[pos_count] = [str(j),'Complement',bayes_percentage_of_correct_predictions_complement,
                                 intermed_time + time.time() - start_time]
    pos_count = pos_count + 1
    start_time = time.time()

    print("Naive Bayes Classification")
    bayes = BernoulliNB()
    bayes.fit(principal_components_train, df_train_output['class'].values)
    bayes_labels_pred = bayes.predict(principal_components_test)
    bayes_results = (df_test_output['class'].values == bayes_labels_pred)
    bayes_percentage_of_correct_predictions_bernoulli = bayes_results.sum() / len(bayes_results)
    print("\nMethod: Bernoulli Naive Bayes" +
          "\nPercentage of correct prediction: " + str(bayes_percentage_of_correct_predictions_bernoulli) +
          "\n" + line_str)

    # store values in Dataframe
    df_results.loc[pos_count] = [str(j),'Bernoulli',bayes_percentage_of_correct_predictions_bernoulli,
                                 intermed_time + time.time() - start_time]
    pos_count = pos_count + 1
    start_time = time.time()

# print results and store to a pickle-file as a basis for later visualization
print(df_results)
df_results.to_pickle('Data/bayes_results.pickle')

# load result data
df = pd.read_pickle('Data/bayes_results.pickle')
print(df)

# create result table
df_results = pd.DataFrame(columns=['Method', 'Average', 'Max', 'Min', 'Average runtime [sec]'])
pos_count = 0
num_neigh_array = ['Gaussian', 'Multinominal', 'Complement', 'Bernoulli']
for i in num_neigh_array:
    df_results.loc[pos_count] = [str(i),
                                 df[df['Method']==str(i)]['Prediction Accuracy'].mean(),
                                 df[df['Method']==str(i)]['Prediction Accuracy'].max(),
                                 df[df['Method'] == str(i)]['Prediction Accuracy'].min(),
                                 df[df['Method'] == str(i)]['Runtime [sec]'].mean()
                                 ]
    pos_count = pos_count + 1

print(df_results)

# Set the current directory
os.chdir('../')
Path = r'C:\ Users\Oliver\Documents\Dokumente\Studium\M.Sc. UPM\Intelligencia Artificial Applicada\Trabajo' \
             '\Bayes_results.csv'
df_results.to_csv(Path, index=False, header=True)