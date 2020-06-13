# Group Information: Intelligencia Artificial Applicada, UPM
# Emiliano Capogrossi, 18029
# Oliver Glardon, 19936
# Sorelys Sandoval, 19237
#_______________________________________________________________________________________________________________________
# imports
from scipy.io import loadmat
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Parameters for printing outputs
line_str = "-------------"

def display_number(data):
    plt.imsave('filename.png', np.array(data).reshape(28, 28), cmap=cm.gray)
    plt.imshow(np.array(data).reshape(28, 28))

# Method to split data into training and testing data
