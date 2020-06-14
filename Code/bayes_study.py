#_______________________________________________________________________________________________________________________
# imports
from scipy.io import loadmat
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import Binarizer
import warnings
from pandas.core.common import SettingWithCopyWarning
import pandas as pd

#_______________________________________________________________________________________________________________________
# Set the current directory
os.chdir('../')
#print(os.getcwd()) #use to print the current directory
#_______________________________________________________________________________________________________________________
# Parameters for printing outputs
line_str = "-------------"
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
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

# Loop to collect data for different testing and training data and parameter settings
normalized = ['Sin Normalizacion', 'Normalizacion', 'Normalizacion & Binarizacion', 'Binarizacion']
dim_red = ['PCA', 'Fisher']
principal_comps = [25,50,100,200]
NB_meth = ['Gaussian', 'Multinominal']
df_results = pd.DataFrame(columns=['Metodo', 'Normalizacion', 'Metodo reduccion de dimensionalidad', 'Dimensionalidad',
                                   'Precision'])
# Number of runs per parameter setting
num_of_runs = 10
for i in range(num_of_runs):
    pos_count = 0
    # Split into training and testing data
    percentage_of_training_data = 0.8
    df_train_set = df_total.sample(frac=percentage_of_training_data)
    df_test_set = df_total.drop(df_train_set.index)
    # final training and test data
    df_train_input = pd.DataFrame(df_train_set.loc[:, df_train_set.columns != 'class'])
    df_train_output = pd.DataFrame(df_train_set['class'])
    df_test_input = pd.DataFrame(df_test_set.loc[:, df_test_set.columns != 'class'])
    df_test_output = pd.DataFrame(df_test_set['class'])

    # Normalization
    for norm in normalized:

        # no normalization
        df_train_input_ = df_train_input
        df_test_input_ = df_test_input

        if norm==normalized[1]:
            # Normalization
            stsc = StandardScaler()
            stsc.fit(df_train_input)
            df_train_input_ = pd.DataFrame(stsc.transform(df_train_input))
            df_test_input_ = pd.DataFrame(stsc.transform(df_test_input))
        elif norm==normalized[2]:
            # Normalization and Binarization
            stsc = StandardScaler()
            stsc.fit(df_train_input)
            df_train_input_ = pd.DataFrame(stsc.transform(df_train_input))
            df_test_input_ = pd.DataFrame(stsc.transform(df_test_input))
            transformer = Binarizer().fit(df_train_input_)
            df_train_input_ = pd.DataFrame(transformer.transform(df_train_input_))
            transformer = Binarizer().fit(df_test_input_)
            df_test_input_ = pd.DataFrame(transformer.transform(df_test_input_))
        elif norm == normalized[3]:
            # Binarization
            transformer = Binarizer().fit(df_train_input)
            df_train_input_ = pd.DataFrame(transformer.transform(df_train_input))
            transformer = Binarizer().fit(df_test_input)
            df_test_input_ = pd.DataFrame(transformer.transform(df_test_input))

        # Dimensionality reduction method
        for dim in dim_red:

            for num_pca in principal_comps:

                # PCA
                # Choose the number of components by our own
                number_principal_components = num_pca
                pca = PCA(n_components=number_principal_components)
                pca.fit(df_train_input_)
                principal_components_train = pca.transform(df_train_input_)
                number_principal_components = pca.n_components_
                MSE_PCA_train = 1 - pca.explained_variance_ratio_.sum()

                # calculate the PCs for the test data as well
                principal_components_test = pca.transform(df_test_input_)

                # Add Fisher
                if dim=='Fisher':
                    # Adding Fisher Discrimant Analysis to PCA
                    fisher = LinearDiscriminantAnalysis()
                    fisher.fit(df_train_input_, df_train_output['class'].values)
                    principal_components_train = fisher.transform(df_train_input_)
                    principal_components_test = fisher.transform(df_test_input_)

                # making the data non-negative
                lowest_num = 0
                if (principal_components_test.min() < principal_components_train.min()):
                    lowest_num = principal_components_test.min()
                else:
                    lowest_num = principal_components_train.min()
                principal_components_train = abs(lowest_num) + principal_components_train
                principal_components_test = abs(lowest_num) + principal_components_test

                # Test different NB Methods
                for meth in NB_meth:
                    res = 0
                    bayes = None
                    if meth == NB_meth[0]:
                        bayes = GaussianNB()
                    elif meth == NB_meth[1]:
                        bayes = MultinomialNB()
                    elif meth == NB_meth[2]:
                        bayes = ComplementNB()
                    elif meth == NB_meth[3]:
                        bayes = BernoulliNB()

                    bayes.fit(principal_components_train, df_train_output['class'].values)
                    bayes_labels_pred = bayes.predict(principal_components_test)
                    bayes_results = (df_test_output['class'].values == bayes_labels_pred)
                    res = bayes_results.sum() / len(bayes_results)

                    if i >= 1:
                        # calculate mean
                        df_results['Precision'].iloc[pos_count] = (df_results['Precision'].iloc[pos_count]*(i+1)+res)/(i+2)
                    else:
                        # store values in Dataframe
                        df_results.loc[pos_count] = [meth, norm, dim, num_pca, res]
                    pos_count = pos_count + 1

# print results and store to a pickle-file as a basis for later visualization
print(df_results)
df_results.to_pickle('Data/bayes_results.pickle')

# Set the current directory
os.chdir('../')
Path = r'C:\Users\Oliver\Documents\Dokumente\Studium\M.Sc. UPM\Intelligencia Artificial Applicada\Trabajo\Bayes_all_results.csv'
df_results.to_csv(Path, index=False, header=True)

df_results = df_results.nlargest(10, 'Precision')
print(df_results)

# Set the current directory
os.chdir('../')
Path = r'C:\Users\Oliver\Documents\Dokumente\Studium\M.Sc. UPM\Intelligencia Artificial Applicada\Trabajo\Bayes_main_results.csv'
df_results.to_csv(Path, index=False, header=True)
