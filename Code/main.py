import scipy as sp
from scipy.io import loadmat
import os

# Set the current directory
os.chdir('../')
#print(os.getcwd()) #use to print the current directory
#_______________________________________________________________________________________________________________________

# load the input data: train 10000x784; numbers 1x10000
trainnumbers = loadmat('Data/Trainnumbers.mat')
train = trainnumbers['Trainnumbers'][0][0][0]
numbers = trainnumbers['Trainnumbers'][0][0][1]
#_______________________________________________________________________________________________________________________

# dimensionality reduction

#




exit()