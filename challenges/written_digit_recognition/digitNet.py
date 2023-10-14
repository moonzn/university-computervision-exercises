import tensorflow as tf
from tensorflow import keras
from keras import layers

import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


# Utilizar os datasets builtin do tensorflow - facilita a preparacao dos dados
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizacao dos valores de pixel para o intervalo [0 ... 1] - com imagens
# este passo normalmente conduz a resultados melhores
x_train = x_train / 255.0
x_test = x_test / 255.0

# Preparar a ground truth para o formato adequado, usando 10 classes
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Id's das labels e dimensoes das imagens
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
img_height = 28
img_width = 28

# Mostrar as dimensoes das matrizes para treino e teste
print("Original training set shape:   ", x_train.shape)
print("Original training labels shape:", y_train.shape)

print("Test set shape:                ", x_test.shape)
print("Test labels shape:             ", y_test.shape)

# Visualizar as primeiras 25 imagens do training set original
fig, ax = plt.subplots(5, 5)
for i in range(5):
    for j in range(5):
        ax[i, j].imshow(x_train[i*5+j], cmap=plt.get_cmap('gray'))
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
fig.suptitle('Dataset samples', fontsize=16)
plt.show()

# Desenvolver a partir daqui!
