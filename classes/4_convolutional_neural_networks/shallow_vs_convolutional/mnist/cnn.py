import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import confusion_matrix

# -----------------------------------------------------------------------------------------------------
# Read and prepare dataset

# Use the built-in mnist dataset from Keras
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize values
x_train = x_train / 255.0
x_test = x_test / 255.0

val_split = 10000
x_val = x_train[:val_split, :, :]
x_train = x_train[val_split:, :, :]

# Prepare ground truth, for 10 classes
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

y_val = y_train[:val_split, :]
y_train = y_train[val_split:, :]

# Labels IDs and image dimensions
labels = ["0", "1", "2", "3", "4", "5", "6_object_detector", "7", "8", "9"]
img_height = 28
img_width = 28

# Show matrix dimensions for train, validation and test
print("Training set shape:            ", x_train.shape)
print("Training labels shape:         ", y_train.shape)

print("Validation set shape:          ", x_val.shape)
print("Validation labels shape:       ", y_val.shape)

print("Test set shape:                ", x_test.shape)
print("Test labels shape:             ", y_test.shape)

# -----------------------------------------------------------------------------------------------------
# Define, compile and train model

# Define neural network arquitecture
digit_model = tf.keras.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 1)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Neural network summary
digit_model.summary()

# Compile model with loss function and optimization algorithm
digit_model.compile(loss=tf.losses.CategoricalCrossentropy(),
                    optimizer=tf.optimizers.Adam(learning_rate=0.001),
                    metrics=['accuracy'])

# Train model and store training data
history = digit_model.fit(x_train, y_train, batch_size=1000, epochs=100, validation_data=(x_val, y_val))

# -----------------------------------------------------------------------------------------------------
# Make predictions and show results

# True class ids
y_true = np.argmax(y_test, axis=1)

# Make predictions
output_pred = digit_model(x_test)
y_pred = np.argmax(output_pred, axis=1)

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

cm_copy = cm.copy()
np.fill_diagonal(cm_copy, 0)
max_indexes = np.argpartition(-cm_copy, 5, axis=None)[:5]
print("\nThe 5 most misclassified digits are:")
for s in max_indexes:
    digit = str(s)
    print(f"{digit[0]} misclassified as {digit[1]}")

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

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1", "2", "3", "4", "5", "6_object_detector", "7", "8", "9"])
disp.plot(cmap=plt.cm.Blues)
plt.show()
