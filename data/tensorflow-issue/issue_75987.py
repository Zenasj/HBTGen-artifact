from tensorflow import keras
from keras import layers

nn_clf = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[76]),
    layers.Dense(64, activation='relu'),
    layers.Dropout(rate=0.3),
    layers.Dense(15, activation='softmax'),
])