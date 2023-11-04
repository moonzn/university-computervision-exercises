"""
APVC - Challenge 5 (Cat/Dog classifier with knowledge transfer)

This file implements a dog and cat classifier, using the pre-trained network of the ResNet family.

Instructions:
• To run this program you must place a "cats_and_dogs_dataset" directory in the same directory as this script.
• The directory should be organized like this:
    - cats_and_dogs_dataset
        -train
            -cats
                -cat.0.jpg
                -cat.1.jpg
                (...)
            -dogs
                (...)
        -validation
            -cats
                (...)
            -dogs
                (...)

• To make the program run faster, the trained model was saved in a file.
• This model is loaded and used to classify the images.
• The area of code used to define, train and compile the model is commented out.
• You should also place the "models" directory in the same directory as this script.
• The directory should be organized like this:
    - models
        -resnet
            -resnet.keras
        -vgg
            -vgg.keras


Authors:
• Bernardo Grilo, n.º 93251
• Gonçalo Carrasco, n.º 109379
• Raúl Nascimento, n.º 87405
"""

import logging
import os
import numpy as np
import tensorflow as tf
from keras import layers
from keras.applications.resnet_v2 import ResNet152V2
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

logging.disable(logging.WARNING)
logging.disable(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# -----------------------------------------------------------------------------------------------------
# Read and prepare dataset

BATCH_SIZE = 100
IMG_HEIGHT = 224
IMG_WIDTH = 224
TRAIN_DATASET_PATH = "cats_and_dogs_dataset/train"
VAL_DATASET_PATH = "cats_and_dogs_dataset/validation"
SEED = 1245  # Seed for split
VAL_TEST_SPLIT = 0.5  # Fraction of images for validation
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

# Get labels
labels = train_ds.class_names

# Cache images in memory
train_ds = train_ds.cache()
val_ds = val_ds.cache()
test_ds = test_ds.cache()

# -----------------------------------------------------------------------------------------------------
# Define, compile and train model
"""
resnetModel = ResNet152V2(weights="imagenet", include_top=False)
resnetModel.trainable = False

callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=2)

model = tf.keras.models.Sequential([
    layers.Rescaling(2./255, offset=-1, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    resnetModel,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

EPOCHS = 10
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[callback])

model.save("models/resnet/resnet.keras")
"""

model = tf.keras.models.load_model("models/resnet/resnet.keras")

# -----------------------------------------------------------------------------------------------------
# Make predictions and show results

# Make predictions
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

"""
# Show figures - accuracy, loss and confusion matrix
plt.figure(num=1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper left")
plt.grid(True, ls='--')
plt.savefig("models/resnet/accuracy.png")

plt.figure(num=2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper right")
plt.grid(True, ls='--')
plt.savefig("models/resnet/loss.png")

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
disp.figure_.savefig("models/resnet/cm.png")
plt.show()


plt.figure(num=1)
img_acc = plt.imread("models/resnet/accuracy.png")
plt.axis("off")
plt.imshow(img_acc)
plt.figure(num=2)
img_loss = plt.imread("models/resnet/loss.png")
plt.axis("off")
plt.imshow(img_loss)
plt.figure(num=3)
img_cm = plt.imread("models/resnet/cm.png")
plt.axis("off")
plt.imshow(img_cm)
plt.show()
"""