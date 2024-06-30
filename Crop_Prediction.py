from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical

crop_data_df = pd.read_csv('Crop_recommendation.csv')  # Assuming 'Crop_recommendation.csv' is in the same directory
crop_data_df = crop_data_df.dropna()

X = np.array(crop_data_df.drop(columns=['label'], axis=1))
y = pd.get_dummies(crop_data_df['label'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, shuffle=True)

NB_classes = 22  # number of outputs
NB_neurones = 30  # main number of neurons
NB_features = 7  # number of inputs
activation_func = tf.keras.activations.relu  # activation function used

model = tf.keras.Sequential([
    tf.keras.layers.Dense(NB_neurones, activation=activation_func, input_shape=(NB_features,)),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dense(NB_neurones, activation=activation_func),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(NB_classes, activation=tf.keras.activations.softmax)
])

model.compile(optimizer="adam", loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.summary()
history = model.fit(x=X_train,
          y=y_train,
          epochs=500,
          validation_data=(X_test, y_test),
          verbose=1,
          shuffle=True)  # Train our model
model.save("output.h5")
performance = model.evaluate(X_test, y_test, batch_size=32, verbose=1)[1] * 100
print('Final accuracy : ', round(performance), '%')

tf.keras.utils.plot_model(
    model,
    to_file="model.png",
)

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_classification_report(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0), annot=True, cmap='Blues')
    plt.title('Classification Report')
    plt.show()


plot_loss(history)
plot_accuracy(history)

val_data.reset()
y_true = val_data.classes
y_pred = model_1.predict(val_data)
y_pred = np.argmax(y_pred, axis=1)
class_names = val_data.class_indices.keys()
plot_confusion_matrix(y_true, y_pred, class_names)
plot_classification_report(y_true, y_pred, class_names)
