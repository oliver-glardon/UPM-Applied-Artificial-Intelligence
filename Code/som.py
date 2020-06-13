# Group Information: Intelligencia Artificial Applicada, UPM
# Emiliano Capogrossi, 18029
# Oliver Glardon, 19936
# Sorelys Sandoval, 19237
#_______________________________________________________________________________________________________________________
# imports
from utils.functions import *
import numpy as np
from scipy import stats
from utils.som_lib import SOM
from sklearn.metrics import confusion_matrix
from utils.plot_cm import plot_confusion_matrix
#_______________________________________________________________________________________________________________________
# (dataframes) df_total, df_train_input, df_train_output, df_test_input, df_test_output
df_total, (df_train_input, df_train_output, df_test_input, df_test_output) = load_standarized_data('../Data/Trainnumbers.mat')
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
train_data = df_train_input.to_numpy()
train_targets = df_train_output.to_numpy().ravel()
test_targets = df_test_output.to_numpy().ravel()
test_data = df_test_input.to_numpy()
#_______________________________________________________________________________________________________________________

som = SOM(10, 10)  # initialize the SOM

def init(train=False, train_data=train_data):
    # Train SOM
    if train:
        som.initialize(train_data)
        som.fit(train_data, 200, save_e=True, interval=100, decay='hill')
        print("Fit error: %.4f" % som.error)
    # Load SOM
    else:
        som = SOM(30,30)
        som.load('../Data/modelo_som.p')
        print("SOM Loaded")

def plot_SOM():
    # Plot SOM - MAP
    test_targets = df_test_output.to_numpy().ravel()
    test_data = df_test_input.to_numpy()
    som.plot_point_map(df_test_input.to_numpy(), test_targets, labels)

def predict(test_input=test_data):
    # Evaluate SOM with test data
    som_output = []
    som.winner_neurons(df_test_input)
    som.winner_indices

    for i,digit in enumerate(test_data):
        som_estimations = som.get_neighbors(digit, train_data, train_targets)
        som_output.append(stats.mode(som_estimations).mode[0])
        # display_number(digit)
    return som_output

def plot_confusion_matrix(test_targets, som_output):
    # Plot Confusion Matrix
    cm = confusion_matrix(test_targets, som_output)
    plot_confusion_matrix(cm, labels)

if __name__=='__main__':
    print('SOM main:\n Loading Model and Data')
    init()