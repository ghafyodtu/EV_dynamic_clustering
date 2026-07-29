import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random as rnd_


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from sklearn.metrics import adjusted_rand_score, accuracy_score

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from tensorflow.keras.datasets import mnist


def plot1_params(fsize1_=10, dpi1_=200, font1_="Palatino Linotype"):

    plt.rcParams['font.family'] = font1_
    plt.rcParams['figure.dpi'] = dpi1_  # Set the global DPI
    plt.rcParams['font.size'] = fsize1_