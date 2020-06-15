# Group Information: Intelligencia Artificial Applicada, UPM
# Emiliano Capogrossi, 18029
# Oliver Glardon, 19936
# Sorelys Sandoval, 19237
#_______________________________________________________________________________________________________________________
# imports
from utils.functions import *
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from neupy import algorithms, utils, init
from sklearn.metrics import mean_squared_error, accuracy_score
import pickle

#_______________________________________________________________________________________________________________________
# Preprocess data
def preprocess_data(data_to_transform, pca_model=None, n_pca=100, data_train=[]):
    if pca_model is None:
        pca = PCA(n_pca)
        if data_train:
            pca.fit(data_train)
        else:
            # If no input data train, fit itself
            pca.fit(data_to_transform)
    else:
        pca = pca_model

    principal_components_res = pca.transform(data_to_transform)
    # number_principal_components = pca.n_components_
    # MSE_PCA_train = 1 - pca.explained_variance_ratio_.sum()
    out_data = principal_components_res

    if pca_model is None:
        return out_data, pca
    else:
        return out_data
#_______________________________________________________________________________________________________________________
# BORRARRRR
# Para entrenar sin normalizar no envies pca
def train_som(train_data, train_targets, 
          grid_size=30, 
          n_epochs=100, 
          lrn_radius=2,
          init_mode=init.Normal(0, 1),
          pca_model=None, n_pca=None, 
          plot_flag=False, 
          extra_str='', dir_path='../Data/som_models/'):
    # Preprocess data if needed
    if type(train_data)==pd.core.frame.DataFrame:
        train_data = train_data.to_numpy()
    if type(train_targets)==pd.core.frame.DataFrame:
        train_targets = train_targets.to_numpy(dtype='int').squeeze()
    if pca_model is None and n_pca is not None:
        train_data, pca_model = preprocess_data(train_data, n_pca)
    
    # Create SOM structure
    GRID_HEIGHT = grid_size
    GRID_WIDTH = grid_size

    som = algorithms.SOFM(
        n_inputs=train_data.shape[1],
        features_grid=(GRID_HEIGHT, GRID_WIDTH),

        learning_radius=lrn_radius,
        weight=init_mode,
        reduce_radius_after=50,

        step=0.5,
        std=1,

        shuffle_data=True,
        verbose=True,
    )

    # Train SOM
    som.train(train_data, epochs=n_epochs)

    # Get model targets for future predictions
    trained_clusters = som.predict(train_data).argmax(axis=1)
    model_targets = np.zeros([GRID_HEIGHT*GRID_WIDTH,1])

    for row_id in range(GRID_HEIGHT):
        for col_id in range(GRID_WIDTH):
            index = row_id * GRID_HEIGHT + col_id
            indices = np.argwhere(trained_clusters == index).ravel()
            clustered_targets = train_targets[indices]

            if len(clustered_targets) > 0:
                # Select the target mode
                target = stats.mode(clustered_targets).mode[0]
            else:
                # If no prediction, assume 0
                target = 0
            model_targets[index] = target

    # Compute training MSE
    som_predictions = model_targets[trained_clusters]
    som.mse = mean_squared_error(train_targets, som_predictions)
    accuracy = accuracy_score(train_targets, som_predictions)
    print('SOM train MSE: ', som.mse)
    print('SOM train Acc: ', accuracy)

    # Save model
    som.model_targets = model_targets.squeeze()
    som.pca_model = pca_model
    save_som_model(som, extra_str, dir_path)

    # Plot SOM map
    if plot_flag:
        plot_SOM(som, train_data, train_targets)

    return som

def plot_som(som, images, train_targets, train_data=None):
    # PLot SOM map
    # Reformat data if needed
    if type(images)==pd.core.frame.DataFrame:
        images = images.to_numpy()
    if type(train_targets)==pd.core.frame.DataFrame:
        train_targets = train_targets.to_numpy(dtype='int')
    if train_targets.dtype != 'int':
        train_targets = train_targets.astype('int')
    if train_data is None and images.shape[1]!=som.n_inputs:
        if som.pca_model is not None:
            train_data = preprocess_data(images, som.pca_model)
        else:
            print("Sorry, you must add the data")
    clusters = som.predict(train_data).argmax(axis=1)
    som_predictions = np.zeros([train_targets.size,1])

    fig = plt.figure(figsize=(12, 12))
    colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000']
    [GRID_HEIGHT, GRID_WIDTH] =  som.features_grid
    grid = gridspec.GridSpec(GRID_HEIGHT, GRID_WIDTH)
    grid.update(wspace=0, hspace=0)
    for row_id in range(GRID_HEIGHT):
        for col_id in range(GRID_WIDTH):
            index = row_id * GRID_HEIGHT + col_id
            indices = np.argwhere(clusters == index).ravel()
            clustered_samples = images[indices]
            clustered_targets = train_targets[indices]

            if len(clustered_samples) > 0:
                # We take the first sample, but it can be any
                # sample from this cluster
                sample = np.mean(clustered_samples,0)
                target = stats.mode(clustered_targets).mode[0]
            else:
                # If we don't have samples in cluster then
                # it means that there is a gap in space
                sample = np.zeros(784)
                target = -1
            som_predictions[indices] = target

            ax = plt.subplot(grid[index])
            plt.setp(ax.spines.values(), color=colors[target], visible=True, linewidth=2)
            plt.imshow(sample.reshape((28, 28)), cmap='Greys')
            ax.set_xticks([])
            ax.set_yticks([])
    
    legend_handles = []
    for label in range(10):
        patch = mpatches.Patch(color=colors[label], label=label)
        legend_handles.append(patch)

    fig.legend(handles=legend_handles, 
               loc="upper center", 
               ncol=10,
              )
    fig.subplots_adjust(top=0.95)    
    fig.savefig('../Data/som_models/somMap2D.png')
    plt.show()
    return fig

def load_som_model(model_filename='../Data/som_models/som_S30_E200_C100_A92.p'):
    with open(model_filename, 'rb') as f:
        som = pickle.load(f)
        return som

def save_som_model(som, extra_str='', dir_path='../Data/som_models/'):
    name = 'som_S%d_E%d_C%d_LR%d_%s.p'%(som.features_grid[0], 
                                        som.n_updates_made, 
                                        som.n_inputs, 
                                        som.learning_radius, 
                                        extra_str)
    with open(dir_path+name, 'wb+') as f:
        pickle.dump(som, f)

def predict_som(eval_data, som=None, model_filename='../Data/som_models/som_S30_E200_C100_A92.p'):
    # Load model
    if som is None: load_som_model(model_filename)

    # Preprocess data if needed
    if eval_data.shape[1] != som.n_inputs:
        if som.pca_model is not None:
            eval_data = preprocess_data(eval_data, pca_model=som.pca_model)
        else:
            print("""Sorry, SOM model input number does not match with evaluation data. 
                    Please associate a som.pca_model of %d number of components"""%som.n_inputs)
            return [], 0
    
    # Evaluate SOM with test data
    eval_clusters = som.predict(eval_data).argmax(axis=1)
    predictions = som.model_targets[eval_clusters]
    return predictions, som.n_inputs

if __name__=='__main__':
    print('SOM main:\n Loading Model and Data')
    init()
    print(predict())