############ IMPORTING LIBRARIES AND DATA ###############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import rcParams, colors
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from sklearn.neighbors import KNeighborsClassifier

############# UTILITY FUNCTIONS ######################
def find_closest(input_point, df_pca, model, N=10):
    '''
    Helper function to find closest points to the input point and predict the 
    class of the input point

    :input_point: 2D coordinate of the input point that have been scaled and undergoing PCA
    :df_pca: dataframe that contains coordinate of all points considered
    :model: pretrained classifier
    :N: number of points to be found
    '''
    distance = np.sum((df_pca[['principal component 1', 'principal component 2']]-input_point)**2, axis=1)
    top_N = np.argsort(distance)[:N]
    predict_prob = model.predict_proba(input_point)
    return top_N, predict_prob

def plot_closest(input_point, df_pca, top_N, predict_prob, ax):
    '''
    Helper function to plot closest points to the input point and show predicted
    class of the input point

    :input_point: 2D coordinate of the input point that have been scaled and undergoing PCA
    :df_pca: dataframe that contains coordinate of all points considered
    :top_N: index of the closest points
    :predict_prob: prediction of the class
    :ax: the axis object to make the plot
    '''
    color_map = ['blue', 'green', 'red']
    df_temp = df_pca.loc[top_N,:]
    sns.scatterplot(x='principal component 1', y='principal component 2',
                hue='species', data=df_pca, s=50, alpha=.3, ax=ax)
    sns.scatterplot(x='principal component 1', y='principal component 2',
                hue='species', data=df_temp, s=80, alpha=1., ax=ax, legend=False)
    plt.scatter(x=input_point[:,0], y=input_point[:,1], color='black', 
                s=100, edgecolor='black', linewidth=2)
    plt.scatter(input_point[:,0], input_point[:,1], s=500, facecolors='none', 
                edgecolors=color_map[np.argmax(predict_prob)], linestyle='--',
                linewidth=2, alpha=.8)
    title = 'setosa: %.1f%%, versicolor: %.1f%%, virginica: %.1f%%' %(predict_prob[0,0]*100, predict_prob[0,1]*100, predict_prob[0,2]*100)
    ax.set_title(title)

def take_input(message,low_limit=0, high_limit=1):
    '''
    Helper function to take input value

    :message: prompt message to take input
    :low_limit: low bound of allowed input
    :high_limit high bound of allowed input
    '''
    while True:
        try:
            input_value = float(input(message))
        except ValueError:
            print("Invalid input format. Please try again")
            continue
    
        if input_value <= low_limit or input_value > high_limit:
            print("Value out of range. Please try again")
            continue
        else:
            break