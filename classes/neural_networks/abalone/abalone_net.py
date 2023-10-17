import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers

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
split = abalone_train.shape[0] * 3 // 4


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

abaloneModel = tf.keras.Sequential([
    layers.Dense(14, activation='relu', input_shape=(None, 1, 7)),
    layers.Dense(1, activation='relu')
])

abaloneModel.summary()

abaloneModel.compile(loss=tf.losses.MeanSquaredError(),
                     optimizer=tf.optimizers.Adam(learning_rate=0.01),
                     metrics=['accuracy'])

history = abaloneModel.fit(x_train, y_train, batch_size=32, epochs=1000, validation_data=(x_test, y_test))

y_pred = abaloneModel(x_test)

plt.figure(num=1)
plt.plot(y_test, y_pred, 'bo')

plt.figure(num=2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper left")
plt.grid(True, ls='--')

plt.figure(num=3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper right")
plt.grid(True, ls='--')

plt.show()
