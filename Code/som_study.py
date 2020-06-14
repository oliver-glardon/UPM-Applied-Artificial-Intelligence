#_______________________________________________________________________________________________________________________
# imports
from sklearn.preprocessing import Binarizer
import warnings
from pandas.core.common import SettingWithCopyWarning
from scipy.io import loadmat
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from som import *
import pandas as pd
import time
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

# Only use 30% of the data to study
df_total = df_total.groupby('class', group_keys=False).apply(pd.DataFrame.sample, frac=.1)

print("Input Dataframe"
      "\nRows: \n  Every row is a data instance (%d)\nColumns:\n  (1) One column 'class' defines the "%df_total.shape[0],
      "classification (0-9)\n  (2) The remaining 784 columns define the pixels of the image (784 -> 28x28)\n")
print(line_str)

# Loop to collect data for different testing and training data and parameter settings

# normalized = ['Sin Normalizacion', 'Normalizacion', 'Normalizacion & Binarizacion', 'Binarizacion']
# dim_red = ['PCA', 'Fisher']
# principal_comps = [56,100,0.95]
# grid_sizes = [10,30]
# list_epochs = [100, 200]
# init_modes = [init.Normal(0, 1), 'init_pca', 'sample_from_data']
# init_modes_str = ['Inorm', 'Ipca', 'Isample']
# learning_radiuses = [2,10]

normalized = ['Sin Normalizacion']
dim_red = ['PCA']
principal_comps = [100]
grid_sizes = [30]
list_epochs = [200]
init_modes = [init.Normal(0, 1), 'init_pca', 'sample_from_data']
init_modes_str = ['Inorm', 'Ipca', 'Isample']
learning_radiuses = [10]

df_results = pd.DataFrame(columns=['Inicializacion','Numero de LR', 'Numero de Epocas','Dim. Rejilla', 'Normalizacion', 'Metodo reduccion de dimensionalidad', 'Dimensionalidad',
                                   'Precision', 'Tiempo Train'])
# Number of runs per parameter setting
num_of_runs = 1

# print results and store to a pickle-file as a basis for later visualization
Path = 'Data/SOM/SOM_all_results'

def main(pos_count):
    t_ini = time.time()
    som = train_som(principal_components_train,df_train_output['class'].values,
                    n_epochs=n_epochs,
                    grid_size=grid_size,
                    lrn_radius=lrn_radius,
                    init_mode=init_mode,
                    pca_model=reducer,
                    extra_str='%s_%s'%(init_mode_str, dim),
                    dir_path='Data/SOM/')
    t_train = time.time() - t_ini
    predictions,_ = predict_som(principal_components_test, som)
    res = accuracy_score(df_test_output['class'].values, predictions)*100

    if False: #i >= 1:
        # calculate mean
        df_results['Precision'].iloc[pos_count] = (df_results['Precision'].iloc[pos_count]*(i+1)+res)/(i+2)
    else:
        # store values in Dataframe
        df_results.loc[pos_count] = [str(init_mode), lrn_radius, n_epochs, grid_size, norm, dim, number_principal_components, res, t_train]
    pos_count = pos_count + 1
    return pos_count

pos_count = 0
for i in range(num_of_runs):
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
    for index_norm, norm in enumerate(normalized):

        # no normalization
        df_train_input_ = df_train_input
        df_test_input_ = df_test_input

        if index_norm==1:
            # Normalization
            stsc = StandardScaler()
            stsc.fit(df_train_input)
            df_train_input_ = pd.DataFrame(stsc.transform(df_train_input))
            df_test_input_ = pd.DataFrame(stsc.transform(df_test_input))
        elif index_norm==2:
            # Normalization and Binarization
            stsc = StandardScaler()
            stsc.fit(df_train_input)
            df_train_input_ = pd.DataFrame(stsc.transform(df_train_input))
            df_test_input_ = pd.DataFrame(stsc.transform(df_test_input))
            transformer = Binarizer().fit(df_train_input_)
            df_train_input_ = pd.DataFrame(transformer.transform(df_train_input_))
            transformer = Binarizer().fit(df_test_input_)
            df_test_input_ = pd.DataFrame(transformer.transform(df_test_input_))
        elif index_norm==3:
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
                reducer = pca
                # Add Fisher
                if dim=='Fisher':
                    # Adding Fisher Discrimant Analysis to PCA
                    fisher = LinearDiscriminantAnalysis()
                    fisher.fit(df_train_input_, df_train_output['class'].values)
                    principal_components_train = fisher.transform(df_train_input_)
                    principal_components_test = fisher.transform(df_test_input_)
                    reducer = fisher
                # Test grid_sizes
                for grid_size in grid_sizes:
                    # Test n epochs
                    for n_epochs in list_epochs:
                        for lrn_radius in learning_radiuses:
                            for (init_mode, init_mode_str) in zip(init_modes, init_modes_str):
                                pos_count = main(pos_count)
                                df_results = df_results.nlargest(10, 'Precision')
                                df_results.to_csv('%s.csv'%Path, index=False, header=True)


print(df_results)
df_results.to_pickle('%s.pickle'%Path)

df_results = df_results.nlargest(10, 'Precision')
df_results.to_csv('%s.csv'%Path, index=False, header=True)
