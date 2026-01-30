from tensorflow.keras import layers
from tensorflow.keras import models

{'verbose': 1, 'epochs': 2, 'steps': 1}

{'batch_size': 4, 'epochs': 2, 'steps': None, 'samples': 4, 'verbose': 1, 'do_validation': False, 'metrics': ['loss']}

import numpy as np
# import keras
import tensorflow.keras as keras

def build_xor_data():
    x_train = [np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)]
    y_train = [np.array([[0], [1], [1], [0]], dtype=float)]

    return x_train, y_train


def build_xor_model_keras():
    input_layer = keras.layers.Input(shape=(2,))
    hidden_layer = keras.layers.Dense(2, activation="sigmoid")(input_layer)
    output_layer = keras.layers.Dense(1, activation="sigmoid")(hidden_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="mse", optimizer="adam")
    return model

x_train, y_train = build_xor_data()

model = build_xor_model_keras()

class DebugCallback(keras.callbacks.Callback):

    def set_params(self, params):
        print("SET PARAMS", locals())

print("KERAS VERSION", keras.__version__)
model.fit(x_train, y_train, batch_size=4, epochs=2, callbacks=[DebugCallback()])