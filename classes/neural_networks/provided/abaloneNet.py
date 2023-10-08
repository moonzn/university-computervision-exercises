import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers

# workaround para o caso do "from" da linha anterior dar um erro
# import keras.api._v2.keras as keras
# from keras import layers

# Leitura dos datasets
abalone_train = pd.read_csv('abalone_train.csv',
                            names=["Length", "Diameter", "Height", "Whole weight",
                                   "Shucked weight", "Viscera weight", "Shell weight", "Age"])
abalone_test = pd.read_csv('abalone_test.csv',
                           names=["Length", "Diameter", "Height", "Whole weight",
                                  "Shucked weight", "Viscera weight", "Shell weight", "Age"])

# Preparacao dos dados (separar a Age do resto, visto que esta e' o que se pretende estimar)
abalone_train_features = abalone_train
abalone_train_labels = abalone_train_features.pop("Age")
abalone_test_features = abalone_test
abalone_test_labels = abalone_test_features.pop("Age")

# Construir conjuntos de treino e teste
x_train = np.array(abalone_train_features)
y_train = np.array(abalone_train_labels)
x_test = np.array(abalone_test_features)
y_test = np.array(abalone_test_labels)

# Mostrar as dimensoes
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Desenvolver a partir daqui
