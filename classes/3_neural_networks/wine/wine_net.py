import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

# Ler um ficheiro com o dataset wineData - este data set contém 13 atributos
# (features) referentes a vinhos italinanos de tres produtores diferentes.
# A ideia e' identificar o produtor do vinho com base nos atributos do vinho.
# Os atributos são os seguintes (https://archive.ics.uci.edu/ml/datasets/wine):

# 1) Alcohol
# 2) Malic acid
# 3) Ash
# 4) Alcalinity of ash
# 5) Magnesium
# 6) Total phenols
# 7) Flavanoids
# 8) Nonflavanoid phenols
# 9) Proanthocyanins
# 10)Color intensity
# 11)Hue
# 12)OD280/OD315 of diluted wines
# 13)Proline

##################################################################
# Ler e preparar o dataset para ser compativel com o tensorflow

# Leitura do ficheiro
rawData = np.array(pd.read_csv("wineData.csv", sep=";"))

# n. de amostras e n. de features
nSamples = rawData.shape[0]
nFeatures = rawData.shape[1]-1

# "baralhar" as amostras
rawData = np.random.permutation(rawData)

# separar as feature das classificacoes previas (labels)
samples = rawData[:, 0:nFeatures]
labels = rawData[:, nFeatures].astype(int) - 1

# colocar as labels no formato adequado para treino (matriz N x C)
labels = tf.keras.utils.to_categorical(labels, 3)

# divisao treino/validacao (75% treino - 25% validacao)
split = samples.shape[0] * 3 // 4

x_train = samples[:split, :]
y_train = labels[:split, :]
x_val = samples[split + 1:, :]
y_val = labels[split + 1:, :]

print(x_train.shape)

##################################################################
# Definicao, compilacao e treino do modelo

# definicao da arquitetura da rede neuronal
wineModel = tf.keras.Sequential([
    layers.Dense(26, activation='sigmoid', input_shape=(None, 1, 13)),
    layers.Dense(3, activation='softmax')
])

# Mostrar um sumario do modelo (organizacao e n. de pesos a otimizar em cada camada)
wineModel.summary()

# compilar o modelo, definindo a loss function e o algoritmo de optimizacao
wineModel.compile(loss=tf.losses.CategoricalCrossentropy(),
                  optimizer=tf.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

# treinar, guardaando os dados do treino na variavel history
history = wineModel.fit(x_train, y_train, batch_size=32, epochs=1000, validation_data=(x_val, y_val))

##################################################################
# Obter e mostrar resultados

# obter os id's das classes verdadeiras
y_true = np.argmax(y_val, axis=1)

# realizar as predicoes e obter os id's das classes preditas
output_pred = wineModel(x_val)
y_pred = np.argmax(output_pred, axis=1)

# gerar uma matriz de confusao
cm = confusion_matrix(y_true, y_pred)

# mostrar figuras - accuracy, loss e matriz de confusao
# pode dar avisos de deprecated no pycharm pro - nao ligar a isso
plt.figure(num=1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper left")
plt.grid(True, ls='--')

plt.figure(num=2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper right")
plt.grid(True, ls='--')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Produtor A", "Produtor B", "Produtor C"])
disp.plot(cmap=plt.cm.Blues)
plt.show()
