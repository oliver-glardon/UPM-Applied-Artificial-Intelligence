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
from sklearn.preprocessing import StandardScaler, Binarizer, Normalizer, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from utils.plot_cm import plot_confusion_matrix
import matplotlib.pyplot as plt

import pandas as pd
import time
import joblib

train = True
type_run = ""
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

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

# Loop to collect data for different testing and training data
num_repetitions_per_parameter_setting = 4
df_results = pd.DataFrame(columns=['Run', 'Preprocessing', 'Data used', 'NN1 Pred. Acc.', 'NN2 Pred. Acc.', 'Runtime [sec]'])
pos_count = 0

print("Multilayer Perceptron (MLP)\n")

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
train_class = df_train_output['class'].values
test_class = df_test_output['class'].values

# One-Hot
target_scaler = OneHotEncoder(sparse=False, categories='auto')
train_class = target_scaler.fit_transform(train_class.reshape(-1, 1))
test_class = target_scaler.fit_transform(test_class.reshape(-1, 1))

# _____________________________________________________________________________________________________________________
# Loop
# ...
total_time = time.time()
for j in range(num_repetitions_per_parameter_setting):
    print("Run: " + str(j + 1) + "\n" + line_str)
    start_time = time.time()

    # _____________________________________________________________________________________________________________________
    # Preprocessing
    # ...

    # Rescale - adding bias to prevent weights from not being updated
    if j == 5:
        fac = 0.99 / 255
        df_train_input = df_train_input * fac + 0.01
        df_test_input = df_test_input * fac + 0.01

        train_class[train_class == 0] = 0.01
        train_class[train_class == 1] = 0.99
        test_class[test_class == 0] = 0.01
        test_class[test_class == 1] = 0.99

    else:
        df_train_input /= 255.
        df_test_input /= 255.

    #
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
        stsc = StandardScaler().fit(df_train_input)
        stsc1 = StandardScaler().fit(df_test_input)
        df_train_input_ = pd.DataFrame(stsc.transform(df_train_input_))
        df_test_input_ = pd.DataFrame(stsc1.transform(df_test_input_))
    elif j == 4:
        preprocessing = 'simple_github'
        df_train_input_ -= df_train_input.mean(axis=0)
        df_test_input_ -= df_train_input.mean(axis=0)

    if type_run == "BIN":
        preprocessing = preprocessing + '+Binarized'
        stsc = Binarizer().fit(df_train_input)
        stsc1 = Binarizer().fit(df_test_input)
        df_train_input_ = pd.DataFrame(stsc.transform(df_train_input))
        df_test_input_ = pd.DataFrame(stsc1.transform(df_test_input))


    # ...
    # Preprocessing Techniques
    # _____________________________________________________________________________________________________________________

    # PCA
    minimum_explained_variance = 0.95
    pca = PCA(minimum_explained_variance)

    # alternative:
    # Choose the number of components by our own
    # number_principal_components = 10
    # pca = PCA(n_components=number_principal_components)

    pca.fit(df_train_input_)
    principal_components_train = pca.transform(df_train_input_)
    principal_components_test = pca.transform(df_test_input_)
    number_principal_components = pca.n_components_
    MSE_PCA_train = 1 - pca.explained_variance_ratio_.sum()

    # Fisher Discriminant
    fisher = LinearDiscriminantAnalysis()
    fisher.fit(df_train_input_, df_train_output['class'].values)
    fisher_components_train = fisher.transform(df_train_input_)
    fisher_components_test = fisher.transform(df_test_input_)

    # Adding Fisher Discrimant Analysis to PCA
    fisher_pca = LinearDiscriminantAnalysis()
    fisher_pca.fit(principal_components_train, df_train_output['class'].values)
    fisher_pca_components_train = fisher_pca.transform(principal_components_train)
    fisher_pca_components_test = fisher_pca.transform(principal_components_test)

    # _______________________________________________________________________________________________________________________
    # TASK 4: MLP
    # ....
    print("\nPreprocessing Technique: " + preprocessing)

    if train:
        # _______________________________________________________________________________________________________________________
        # Neural Network, 1 Layer - 200 Neurons
        mlp1 = MLPClassifier(hidden_layer_sizes=(200,), max_iter=400,
                             solver='sgd', learning_rate_init=0.01, momentum=0.99, verbose=0,
                             random_state=1)

        # _______________________________________________________________________________________________________________________
        # Neural Network, 2 Layers - 500|300 Neurons
        mlp2 = MLPClassifier(hidden_layer_sizes=(500, 300), max_iter=400,
                            solver='sgd', learning_rate_init=0.01, momentum=0.99, verbose=0,
                            random_state=1)

    # _______________________________________________________________________________________________________________________
    # Loop for different parameter settings:

    # loop_array = [1, 3, 5, 7, 9]
    # for i in loop_array:
    for i in range(3):
        # _______________________________________________________________________________________________________________________
        # Data to be used:

        # Without preprocessing
        if i == 0:
            train_imgs = df_train_input_
            test_imgs = df_test_input_
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

        # Using Encoder
        elif i == 3:
            train_imgs = fisher_pca_components_train
            test_imgs = fisher_pca_components_train
            data_type = 'Encoder'
            print("Data Using Encoder")

        # _______________________________________________________________________________________________________________________
        # TRAIN NN

        filename1 = 'Data/mlp_models/%smlp1_model-run_%d-data_%d.sav' %(type_run,j,i)
        filename2 = 'Data/mlp_models/%smlp2_model-run_%d-data_%d.sav' %(type_run,j,i)

        if train:
            # TRAIN NN
            mlp1.fit(train_imgs, train_class)
            mlp2.fit(train_imgs, train_class)

            # save the model to disk
            joblib.dump(mlp1, filename1)
            joblib.dump(mlp2, filename2)

            print("\nCreated File:\n\n"+ filename1 + '\n' + filename2 + '\n' + line_str)

        else:
            # load the model from disk
            mlp1 = joblib.load(filename1)
            mlp2 = joblib.load(filename2)

            print("\nLoaded File:\n\n"+ filename1 + '\n' + filename2 + '\n' + line_str)

        # _______________________________________________________________________________________________________________________
        # PREDICT

        # 1st NN
        mlp_labels_pred = mlp1.predict(test_imgs)
        mlp1_percentage_of_correct_predictions = metrics.accuracy_score(test_class, mlp_labels_pred)
        print("\n1 Layer - 200 neurons" +
              "\nPercentage of correct prediction: " + str(mlp1_percentage_of_correct_predictions))

        # Plot Confusion Matrix
        cm = confusion_matrix(np.argmax(test_class, axis=1), np.argmax(mlp_labels_pred, axis=1))
        plot_confusion_matrix(cm, labels)
        plt.savefig('Data/mlp_models/CM_%smlp1_model-run_%d-data_%d.png' %(type_run,j,i))
        plt.close()

        # 2nd NN
        mlp2_labels_pred = mlp2.predict(test_imgs)
        mlp2_percentage_of_correct_predictions = metrics.accuracy_score(test_class, mlp2_labels_pred)
        print("\n2 Layers - 500|300 neurons" +
              "\nPercentage of correct prediction: " + str(mlp2_percentage_of_correct_predictions))

        # Plot Confusion Matrix
        cm = confusion_matrix(np.argmax(test_class, axis=1), np.argmax(mlp2_labels_pred, axis=1))
        plot_confusion_matrix(cm, labels)
        plt.savefig('Data/mlp_models/CM_%smlp2_model-run_%d-data_%d.png' % (type_run, j, i))
        plt.close()

        end_time = time.time()
        duration = end_time - start_time
        print("\nElapsed time: " + str(duration) + ' [Sec]\n' + line_str)

        # _______________________________________________________________________________________________________________________
        # store values in Dataframe
        df_results.loc[pos_count] = [str(j+1), preprocessing, data_type, mlp1_percentage_of_correct_predictions, mlp2_percentage_of_correct_predictions, duration]
        pos_count = pos_count + 1


# print results and store to a pickle-file as a basis for later visualization
print(df_results)
df_results.to_pickle('Data/mlp_results.pickle')

total_duration = time.time() - total_time
print("\nTotal elapsed time: "+ str(total_duration) + ' [Sec]\n' + line_str)