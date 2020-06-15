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
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Parameters for printing outputs
line_str = "-------------"


def display_n_numbers(data, n, label=[], savePath=[]):

    data = data.reshape(data.shape[0], 28, 28)
    x1 = min(n // 2, 5)
    if x1 == 0:
        x1 = 1
    y1 = (n // x1)
    x = min(x1, y1)
    y = max(x1, y1)
    fig, ax = plt.subplots(x, y, figsize=(10, 10))
    plt.gray()

    # loop through subplots and add centroid images
    for i, ax in enumerate(ax.flat):
        if label != []:
            # determine inferred label using cluster_labels dictionary
            for key, value in label.items():
                if i in value:
                    ax.set_title('Inferred Class: {}'.format(key))
        ax.matshow(data[i])
        ax.axis('off')

    if savePath!=[]:
        plt.savefig(savePath)


def display_number(data):
    plt.imsave('filename.png', np.array(data).reshape(28, 28), cmap=cm.gray)
    plt.imshow(np.array(data).reshape(28, 28))


# Method to split data into training and testing data
def split_into_train_and_test(data, percentage_of_training_data = 0.8):
    # print("Split into training and testing data\n" + line_str)
    df_train_set = data.sample(frac=percentage_of_training_data, random_state=0)
    df_test_set = data.drop(df_train_set.index)
    df_train_input = pd.DataFrame(df_train_set.loc[:, df_train_set.columns != 'class'])
    df_train_output = pd.DataFrame(df_train_set['class'])
    df_test_input = pd.DataFrame(df_test_set.loc[:, df_test_set.columns != 'class'])
    df_test_output = pd.DataFrame(df_test_set['class'])
    return df_train_input, df_train_output, df_test_input, df_test_output


def standardize(df_train_input, df_test_input):
    stsc = StandardScaler()
    stsc.fit(df_train_input)
    df_train_input_ = pd.DataFrame(stsc.transform(df_train_input))
    df_test_input_ = pd.DataFrame(stsc.transform(df_test_input))
    return df_train_input_, df_test_input_


def load_data(filename):
    # Load the input data
    # (dataframes) df_total, (df_train_input, df_train_output, df_test_input, df_test_output)
    # Rows: Every row is an data instance (10000)
    # Columns:
    #   (1) One column "class" defines the classification (1-10)
    #   (2) The remaining 784 columns define the pixels of the image (784 -> 28x28)
    trainnumbers = loadmat(filename)
    _input = trainnumbers['Trainnumbers'][0][0][0]
    df_input = pd.DataFrame(_input.T)
    output = trainnumbers['Trainnumbers'][0][0][1]
    df_output = pd.DataFrame(output.T)
    df_total = df_input.copy()
    df_total['class'] = df_output.values # total dataframe containing all data
    # Split into training and testing data
    percentage_of_training_data = 0.8
    return df_total, split_into_train_and_test(df_total, percentage_of_training_data)

def load_standarized_data(filename):
    df_total, (df_train_input, df_train_output, df_test_input, df_test_output) = load_data(filename)
    df_train_input, df_test_input = standardize(df_train_input, df_test_input)
    return df_total, (df_train_input, df_train_output, df_test_input, df_test_output)
