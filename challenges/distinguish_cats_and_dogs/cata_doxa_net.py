import logging
import os
import numpy as np

logging.disable(logging.WARNING)
logging.disable(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from keras import layers

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

###################################################################
#
# Neste exemplo o dataset e' carregado a partir do sistema de ficheiros
# e apenas e' dividido em treino e validacao

BATCH_SIZE = 32
IMG_HEIGHT = 160
IMG_WIDTH = 160
TRAIN_DATASET_PATH = "cats_and_dogs_dataset/train"  # ajustar consoante a localizacao
VAL_DATASET_PATH = "cats_and_dogs_dataset/validation"  # ajustar consoante a localizacao
SEED = 1245  # semente do gerador de numeros aleotorios que faz o split treino/validacao
VAL_TEST_SPLIT = 0.5  # fracao de imagens para o conjunto de validacao
NUM_CLASSES = 2

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATASET_PATH,
    labels='inferred',
    label_mode='categorical',
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

val_ds, test_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DATASET_PATH,
    labels='inferred',
    label_mode='categorical',
    validation_split=VAL_TEST_SPLIT,
    subset="both",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

# labels inferidas a partir dos nomes dos diretorios
labels = train_ds.class_names
print(train_ds.class_names)
print(val_ds.class_names)
print(test_ds.class_names)

plt.figure(1, figsize=(10, 10))
for x_batch, y_batch in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(x_batch[i].numpy().astype("uint8"))
        plt.title(labels[np.argmax(y_batch[i, :])])
        plt.axis("off")
plt.show()

# optimazacoes para manter a imagens em memoria
train_ds = train_ds.cache()
val_ds = val_ds.cache()
test_ds = test_ds.cache()

# nota - os layers de data augmentation originam warnings (em versoes do tensorflow superiores a 2.8.3)
# esses warnings sao para ignorar
model = tf.keras.models.Sequential([
    layers.Rescaling(1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.Conv2D(8, 5, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

EPOCHS = 100
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

# -----------------------------------------------------------------------------------------------------
# Make predictions and show results

# opter as predicoes e ground thruth num formato mais facil de tratar para mostrar os resultados
# (um vetor de ids das classes)
y_pred = model.predict(test_ds)
y_pred = tf.argmax(y_pred, axis=1)

y_true = tf.concat([y for x, y in test_ds], axis=0)
y_true = tf.argmax(y_true, axis=1)

# Confusion matrix and metrics
cm = confusion_matrix(y_true, y_pred)
cm_accuracy = accuracy_score(y_true, y_pred)

fp = cm.sum(axis=0) - np.diag(cm)
fn = cm.sum(axis=1) - np.diag(cm)
tp = np.diag(cm)
tn = cm.sum() - (fp + fn + tp)

# True positive rate
tpr = tp / (tp + fn)
# False positive rate
fpr = fp / (fp + tn)
# True negative rate
tnr = tn / (tn + fp)
# False negative rate
fnr = fn / (tp + fn)

# Per class accuracy
acc = (tp + tn) / (tp + fp + fn + tn)

# Print metrics
print(f"Per class True Positive Rate: {tpr}\n")
print(f"Per class False Positive Rate: {fpr}\n")
print(f"Per class True Negative Rate: {tnr}\n")
print(f"Per class False Negative Rate: {fnr}\n")
print(f"Per class accuracy: {acc}\n")
print(f"\033[1mOverall accuracy\033[0m: {cm_accuracy}")

# Show figures - accuracy, loss and confusion matrix
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

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()
