import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras.layers import Dense, Add, LayerNormalization
from tensorflow.keras.models import Model

class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(32, activation='relu')
        self.dense3 = Dense(10, activation='relu')
        self.output_layer = Dense(10)
        self.add_layer = Add()
        self.norm_layer = LayerNormalization()

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)

        x = self.add_layer([x, inputs])

        x = self.norm_layer(x)

        return self.output_layer(x)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
model = CustomModel()

model.compile(optimizer='adam', loss='mse')

import numpy as np
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100, 10)

model.fit(X_train, y_train, epochs=5)

X_test = np.random.rand(10, 10)

predictions = model.predict(X_test)

print(predictions)